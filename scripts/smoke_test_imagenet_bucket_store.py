import os
import random
import tempfile
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tdcf.bucketed_store import (
    BucketBatchSampler,
    BucketedTileBandDataset,
    BucketedTileBandStore,
)
from tdcf.prepare_imagenet1k_crop_store import build_store_from_samples


def make_image(height: int, width: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def build_split_store(specs, out_dir: str):
    samples = []
    labels = []
    for idx, (height, width, label) in enumerate(specs):
        samples.append(make_image(height, width, seed=idx + label * 1000))
        labels.append(label)
    build_store_from_samples(
        samples=samples,
        labels=labels,
        out_dir=out_dir,
        resize_shorter=64,
        block_size=8,
        num_bands=16,
        tile_blocks=2,
        batch_size=4,
        device=torch.device("cpu"),
    )


def one_train_step(store: BucketedTileBandStore, device: torch.device):
    dataset = BucketedTileBandDataset(store)
    sampler = BucketBatchSampler(
        store.sample_bucket_ids,
        batch_size=2,
        num_replicas=1,
        rank=0,
        shuffle=True,
        drop_last=True,
        seed=7,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 3),
    ).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    crit = nn.CrossEntropyLoss()

    indices, labels = next(iter(loader))
    idx_np = indices.numpy()
    labels = labels.to(device)
    coeffs_zz, crop_params, _visible, _k, bucket_id = store.read_coeffs(idx_np)
    coeffs_zz = coeffs_zz.to(device).detach().requires_grad_(True)
    x = store.reconstruct_crops(coeffs_zz, idx_np, crop_params, bucket_id)
    logits = model(x)
    loss = crit(logits, labels)
    loss.backward()
    opt.step()
    return float(loss.item()), coeffs_zz.grad is not None


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        store_root = os.path.join(tmpdir, "store")

        train_specs = [
            (72, 96, 0),
            (88, 64, 1),
            (96, 120, 2),
            (128, 80, 0),
            (90, 140, 1),
            (110, 78, 2),
            (76, 100, 0),
            (84, 132, 1),
        ]
        val_specs = [
            (74, 92, 0),
            (86, 68, 1),
            (100, 118, 2),
            (126, 82, 0),
        ]

        build_split_store(train_specs, os.path.join(store_root, "train"))
        build_split_store(val_specs, os.path.join(store_root, "val"))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_store = BucketedTileBandStore(
            os.path.join(store_root, "train"),
            device=device,
            output_size=32,
        )

        assert train_store.N == len(train_specs), "Unexpected number of train samples"
        assert len(train_store.bucket_infos) >= 2, "Expected multiple buckets for variable shapes"

        dataset = BucketedTileBandDataset(train_store)
        sampler = BucketBatchSampler(
            train_store.sample_bucket_ids,
            batch_size=2,
            num_replicas=1,
            rank=0,
            shuffle=False,
            drop_last=False,
            seed=0,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=0)
        indices, labels = next(iter(loader))
        idx_np = indices.numpy()

        train_store.reset_epoch_io()
        train_store.set_full_fidelity()
        x_full, _, _, _, _ = train_store.serve_indices(idx_np, deterministic_val=True)
        assert tuple(x_full.shape[1:]) == (3, 32, 32), f"Unexpected output shape: {x_full.shape}"
        assert 0.95 <= train_store.get_io_ratio() <= 1.05, "Full-fidelity IO ratio should be ~1"

        train_store.reset_epoch_io()
        train_store.set_budget_ratio(
            0.6,
            band_sensitivity=np.linspace(1.0, 0.2, train_store.num_bands, dtype=np.float32),
            k_low=1,
            patch_policy="greedy",
        )
        x_cap, _, _, _, _ = train_store.serve_indices(idx_np, deterministic_val=False)
        assert tuple(x_cap.shape[1:]) == (3, 32, 32), f"Unexpected capped output shape: {x_cap.shape}"
        assert train_store.bytes_read_epoch <= train_store.full_layout_bytes_epoch, (
            "Capped run should not read more than full visible bytes"
        )

        loss, got_grad = one_train_step(train_store, device)
        assert np.isfinite(loss), "Loss is not finite"
        assert got_grad, "Pilot-style coefficient gradient path did not backprop"

        print("Smoke test passed.")


if __name__ == "__main__":
    main()
