import os
import argparse
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torch.utils.data.distributed import DistributedSampler

# --- PyTorch XLA Imports ---
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

# Re-use our components
from tdcf.sensitivity import BlockSensitivityEstimator
from tdcf.scheduler import FidelityScheduler
from tdcf.io_dataloader import BlockBandStore
from tdcf.transforms import block_idct2d

# Tiny ImageNet constants
NUM_CLASSES = 200
IMAGE_SIZE = 64
BLOCK_SIZE = 8
P = (IMAGE_SIZE // BLOCK_SIZE) ** 2  # 8x8 = 64 patches
NUM_BANDS = 16
TINY_MEAN = (0.485, 0.456, 0.406)
TINY_STD = (0.229, 0.224, 0.225)


class InputNormalize(nn.Module):
    """Normalize raw [0,1] RGB images inside the model graph."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def make_tiny_imagenet_model(device: torch.device) -> nn.Module:
    """
    Standard ImageNet-style ResNet-18 adapted to Tiny-ImageNet classes.

    Keeping the normalization inside the module ensures the sensitivity
    estimator, training loop, and eval loop all see the same input pipeline.
    """
    backbone = tv_models.resnet18(weights=None)
    backbone.fc = nn.Linear(backbone.fc.in_features, NUM_CLASSES)
    model = nn.Sequential(InputNormalize(TINY_MEAN, TINY_STD), backbone)
    return model.to(device)


def augment_tiny_batch(x: torch.Tensor) -> torch.Tensor:
    """GPU/XLA-friendly Tiny-ImageNet augmentation on raw [0,1] tensors."""
    B, _, H_img, W_img = x.shape
    x = x.clamp_(0.0, 1.0)

    # Random crop with 8-pixel padding.
    x_pad = torch.nn.functional.pad(x, (8, 8, 8, 8))
    crop_grid = (
        x_pad.unfold(2, H_img, 1)
        .unfold(3, W_img, 1)
        .permute(0, 2, 3, 1, 4, 5)
    )
    top = torch.randint(0, 17, (B,), device=x.device)
    left = torch.randint(0, 17, (B,), device=x.device)
    batch_idx = torch.arange(B, device=x.device)
    x = crop_grid[batch_idx, top, left]

    # Random horizontal flip.
    flip_mask = torch.rand(B, device=x.device) < 0.5
    if flip_mask.any():
        x[flip_mask] = torch.flip(x[flip_mask], dims=[3])

    # Lightweight color jitter in tensor space.
    brightness = 1.0 + torch.empty(B, 1, 1, 1, device=x.device).uniform_(-0.2, 0.2)
    contrast = 1.0 + torch.empty(B, 1, 1, 1, device=x.device).uniform_(-0.2, 0.2)
    saturation = 1.0 + torch.empty(B, 1, 1, 1, device=x.device).uniform_(-0.2, 0.2)

    x = x * brightness
    x_mean = x.mean(dim=(2, 3), keepdim=True)
    x = (x - x_mean) * contrast + x_mean
    x_gray = x.mean(dim=1, keepdim=True)
    x = (x - x_gray) * saturation + x_gray
    return x.clamp_(0.0, 1.0)


class BlockStoreDataset(Dataset):
    def __init__(self, store: BlockBandStore, labels: np.ndarray):
        self.store = store
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return idx, self.labels[idx]


class ShardSampler(Sampler):
    """Non-duplicating strided sampler for distributed evaluation."""

    def __init__(self, dataset_len: int, num_replicas: int, rank: int):
        self.indices = list(range(rank, dataset_len, num_replicas))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class BlockStoreCollator:
    """Pickle-safe collator that reads coefficient batches from disk."""

    def __init__(self, store: BlockBandStore):
        self.store = store

    def __call__(self, batch):
        indices = np.array([b[0] for b in batch], dtype=np.int64)
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        coeffs = self.store._read_batch(indices)
        return coeffs, labels


def build_loader(dataset, store, batch_size, sampler, num_workers, drop_last):
    collator = BlockStoreCollator(store)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collator,
        persistent_workers=num_workers > 0,
    )


def make_epoch_accumulators(device: torch.device):
    return (
        torch.zeros((), device=device),
        torch.zeros((), device=device),
        torch.zeros((), device=device),
    )


def train_tpu_process(index, args):
    """
    This function runs on each of the 8 TPU cores.
    """
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()
    is_master = xm.is_master_ordinal()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if is_master:
        print("=" * 60)
        print("  TDCF — TPU v4-8  |  Tiny ImageNet (64x64)")
        print("=" * 60)

    # ── 1. Setup BlockBandStore ──
    train_store = BlockBandStore(os.path.join(args.data_dir, "train"))
    test_store = BlockBandStore(os.path.join(args.data_dir, "val"))

    train_dataset = BlockStoreDataset(train_store, train_store.labels)
    test_dataset = BlockStoreDataset(test_store, test_store.labels)

    # Pilot uses a fixed subset for lower cost and reproducibility.
    pilot_n = max(int(len(train_dataset) * args.pilot_ratio), args.batch_size)
    pilot_rng = np.random.default_rng(args.seed)
    pilot_idx = pilot_rng.choice(len(train_dataset), size=pilot_n, replace=False)
    pilot_dataset = Subset(train_dataset, pilot_idx.tolist())

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    pilot_sampler = DistributedSampler(
        pilot_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    test_sampler = ShardSampler(len(test_dataset), num_replicas=world_size, rank=rank)

    train_loader = build_loader(
        train_dataset, train_store, args.batch_size, train_sampler,
        args.data_workers, drop_last=True
    )
    pilot_loader = build_loader(
        pilot_dataset, train_store, args.batch_size, pilot_sampler,
        args.data_workers, drop_last=True
    )
    test_loader = build_loader(
        test_dataset, test_store, args.eval_batch_size, test_sampler,
        args.data_workers, drop_last=False
    )

    # ── 2. Model & Optimizer ──
    model = make_tiny_imagenet_model(device)
    scaled_lr = args.lr * world_size
    pilot_lr = args.pilot_lr if args.pilot_lr is not None else args.lr
    scaled_pilot_lr = pilot_lr * world_size

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    pilot_opt = optim.SGD(
        model.parameters(),
        lr=scaled_pilot_lr,
        momentum=0.9,
        weight_decay=args.wd,
        nesterov=True,
    )

    # ── 3. TDCF Components ──
    nph = npw = IMAGE_SIZE // BLOCK_SIZE
    estimator = BlockSensitivityEstimator(BLOCK_SIZE, nph, npw, NUM_BANDS, device=device)
    fsched = FidelityScheduler(NUM_BANDS, P, args.eta_f, args.eta_s)

    # --- PILOT PHASE ---
    if is_master:
        print(f"\n[PILOT PHASE] - {args.pilot_epochs} epochs, {pilot_n} samples")

    for ep in range(args.pilot_epochs):
        pilot_sampler.set_epoch(ep)
        model.train()
        train_store.set_fidelity(K_high=NUM_BANDS, K_low=NUM_BANDS, q=P)
        para_loader = pl.MpDeviceLoader(pilot_loader, device)

        for coeffs, y in para_loader:
            # Sensitivity probe in eval mode to avoid BN/dropout drift.
            model.eval()
            estimator.measure_sensitivity(coeffs, y, model, crit)

            # Standard pilot training step.
            model.train()
            x = augment_tiny_batch(block_idct2d(coeffs, nph, npw))
            pilot_opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite pilot loss.")
            loss.backward()
            xm.optimizer_step(pilot_opt)

        estimator.finalize_epoch()
        if is_master:
            K = estimator.compute_band_cutoff(ep, args.eta_f)
            q = estimator.compute_patch_quota(ep, args.eta_s)
            bs = estimator.band_sensitivity_history[ep]
            print(
                f"  Pilot {ep+1:2d}/{args.pilot_epochs} | "
                f"K_high={K:2d}/{NUM_BANDS} | q={q:2d}/{P} | "
                f"band_low={bs[0]:.4f} band_high={bs[-1]:.4f}"
            )

    fsched.fit_from_pilot(estimator, args.total_epochs)
    if is_master:
        print(fsched.summary())
        print("\n[ADAPTIVE TRAINING PHASE]")

    # Fresh optimizer/scheduler so pilot state does not leak.
    opt = optim.SGD(
        model.parameters(),
        lr=scaled_lr,
        momentum=0.9,
        weight_decay=args.wd,
        nesterov=True,
    )
    sched_lr = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.total_epochs)

    # --- ADAPTIVE TRAINING PHASE ---
    for ep in range(args.total_epochs):
        train_sampler.set_epoch(ep)

        K_high, q_e = fsched.get_fidelity(ep)
        ps_idx = min(ep, len(estimator.patch_sensitivity_history) - 1)
        bs_idx = min(ep, len(estimator.band_sensitivity_history) - 1)
        ps = estimator.patch_sensitivity_history[ps_idx]
        bs = estimator.band_sensitivity_history[bs_idx]

        train_store.set_fidelity(
            K_high, args.k_low, q_e,
            patch_sensitivity=ps,
            band_sensitivity=bs,
            patch_policy=args.patch_policy,
        )

        model.train()
        loss_s, correct, n = make_epoch_accumulators(device)

        para_train = pl.MpDeviceLoader(train_loader, device)
        for coeffs, y in para_train:
            x = augment_tiny_batch(block_idct2d(coeffs, nph, npw))
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            xm.optimizer_step(opt)

            # Keep metrics on-device to avoid TPU sync on every step.
            batch_n = y.new_tensor(y.size(0), dtype=torch.float32)
            loss_s += loss.detach() * batch_n
            correct += (logits.argmax(1) == y).sum().to(dtype=torch.float32)
            n += batch_n

        n_total = xm.mesh_reduce("n_total", n, sum)
        corr_total = xm.mesh_reduce("corr_total", correct, sum)
        loss_total = xm.mesh_reduce("loss_total", loss_s, sum)
        tr_acc = (corr_total / n_total).item()
        tr_loss = (loss_total / n_total).item()

        # EVALUATION (Full fidelity)
        model.eval()
        test_store.set_fidelity(K_high=NUM_BANDS, K_low=NUM_BANDS, q=P)
        te_loss_s, te_correct, te_n = make_epoch_accumulators(device)

        para_test = pl.MpDeviceLoader(test_loader, device)
        with torch.no_grad():
            for coeffs, y in para_test:
                x = block_idct2d(coeffs, nph, npw).clamp_(0.0, 1.0)
                logits = model(x)
                loss = crit(logits, y)
                batch_n = y.new_tensor(y.size(0), dtype=torch.float32)
                te_loss_s += loss * batch_n
                te_correct += (logits.argmax(1) == y).sum().to(dtype=torch.float32)
                te_n += batch_n

        te_n_total = xm.mesh_reduce("te_n_total", te_n, sum)
        te_corr_total = xm.mesh_reduce("te_corr_total", te_correct, sum)
        te_loss_total = xm.mesh_reduce("te_loss_total", te_loss_s, sum)
        te_acc = (te_corr_total / te_n_total).item()
        te_loss_avg = (te_loss_total / te_n_total).item()

        sched_lr.step()

        if is_master:
            io_ratio = train_store.get_io_ratio()
            print(
                f"  E {ep+1:3d}/{args.total_epochs} | "
                f"K_hi={K_high:2d} K_lo={args.k_low} q={q_e:2d} | "
                f"IO={io_ratio:.3f} | "
                f"Tr Loss={tr_loss:.4f} Tr Acc={tr_acc:.4f} | "
                f"Te Loss={te_loss_avg:.4f} Te Acc={te_acc:.4f} | "
                f"LR={opt.param_groups[0]['lr']:.4f}"
            )

    if is_master:
        print("Training complete! Run model save logic here.")
        # xm.save(model.state_dict(), os.path.join(args.save_dir, "model.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/tiny_imagenet_block_store")
    parser.add_argument("--save_dir", type=str, default="./results/tpu_tiny_imagenet")
    parser.add_argument("--total_epochs", type=int, default=100)
    parser.add_argument("--pilot_epochs", type=int, default=10)
    parser.add_argument("--pilot_ratio", type=float, default=0.10)
    parser.add_argument("--batch_size", type=int, default=128)  # Per core
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--pilot_lr", type=float, default=None)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_low", type=int, default=1)
    parser.add_argument("--eta_f", type=float, default=0.9)
    parser.add_argument("--eta_s", type=float, default=0.85)
    parser.add_argument("--patch_policy", choices=["gradient", "random", "static", "greedy"],
                        default="greedy")
    parser.add_argument("--data_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Spawn 8 processes for TPU v4-8
    xmp.spawn(train_tpu_process, args=(args,), nprocs=8, start_method="spawn")
