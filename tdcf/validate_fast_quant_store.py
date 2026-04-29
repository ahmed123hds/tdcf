"""Validate fast quantized DCT store against the source WebDataset."""

from __future__ import annotations

import argparse
import math
import os

import torch

from tdcf.fast_quant_store import FastQuantizedDCTStore
from tdcf.prepare_imagenet1k_fast_quant_store import build_loader, expand_shards, fixed_view


def parse_args():
    p = argparse.ArgumentParser("Validate fast quantized DCT store")
    p.add_argument("--store_dir", type=str, required=True)
    p.add_argument("--shards", type=str, required=True)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def psnr_from_mse(mse: float):
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def main():
    args = parse_args()
    if not os.path.isdir(args.store_dir):
        raise FileNotFoundError(args.store_dir)

    store = FastQuantizedDCTStore(args.store_dir, device=torch.device(args.device))
    store.set_full_fidelity()

    # Reuse the exact fixed-view transform from store construction.
    loader_args = argparse.Namespace(
        source="webdataset",
        shards=args.shards,
        view_size=store.view_size,
        num_workers=args.num_workers,
        max_samples=0,
        calibration_samples=0,
    )
    urls = expand_shards(args.shards)
    print(
        f"[validate] source shards={len(urls)} first={urls[0]} last={urls[-1]}",
        flush=True,
    )

    total = 0
    label_mismatch = 0
    mse_sum = 0.0
    mae_sum = 0.0
    max_abs = 0.0
    batch_imgs = []
    batch_labels = []
    batch_indices = []

    def flush():
        nonlocal total, label_mismatch, mse_sum, mae_sum, max_abs
        if not batch_imgs:
            return
        idx = torch.tensor(batch_indices, dtype=torch.long)
        recon = store.serve_indices(idx).detach().cpu().float()
        src = torch.stack(batch_imgs).float()
        labels = torch.tensor(batch_labels, dtype=torch.long)
        stored_labels = torch.as_tensor(store.labels[batch_indices], dtype=torch.long)
        label_mismatch += int((labels != stored_labels).sum().item())
        diff = (recon - src).abs()
        mse_sum += float((diff ** 2).mean(dim=(1, 2, 3)).sum().item())
        mae_sum += float(diff.mean(dim=(1, 2, 3)).sum().item())
        max_abs = max(max_abs, float(diff.max().item()))
        total += len(batch_imgs)
        batch_imgs.clear()
        batch_labels.clear()
        batch_indices.clear()

    for i, (img, label) in enumerate(build_loader(loader_args)):
        if i >= args.samples:
            break
        batch_imgs.append(img)
        batch_labels.append(int(label))
        batch_indices.append(i)
        if len(batch_imgs) >= args.batch_size:
            flush()
    flush()
    store.close()

    mean_mse = mse_sum / max(total, 1)
    mean_mae = mae_sum / max(total, 1)
    print("=" * 72)
    print(f"Fast quant store validation | samples={total}")
    print(f"label_mismatch={label_mismatch}/{total}")
    print(f"PSNR={psnr_from_mse(mean_mse):.2f} dB | MSE={mean_mse:.8f}")
    print(f"MAE={mean_mae:.6f} | max_abs={max_abs:.6f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
