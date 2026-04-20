import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# --- PyTorch XLA Imports ---
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

# Re-use our components
from tdcf.sensitivity import SensitivityEstimator
from tdcf.scheduler import FidelityScheduler
from tdcf.io_dataloader import BlockBandStore

# Tiny ImageNet constants
NUM_CLASSES = 200
IMAGE_SIZE = 64
BLOCK_SIZE = 8
P = (IMAGE_SIZE // BLOCK_SIZE) ** 2  # 8x8 = 64 patches
NUM_BANDS = 16

# Dummy backbone for Tiny ImageNet (can be swapped for ResNet later)
class TinyImageNetCNN(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # 32x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # 16x16
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # 8x8
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# --- Dataloader Wrapper for BlockBandStore ---
class BlockStoreDataset(Dataset):
    def __init__(self, store: BlockBandStore, labels: np.ndarray):
        self.store = store
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return idx, self.labels[idx]

def collate_block_store(batch, store):
    indices = np.array([b[0] for b in batch], dtype=np.int64)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    # The store handles batch reading directly from disk
    coeffs = store._read_batch(indices)
    return coeffs, labels


def train_tpu_process(index, args):
    """
    This function runs on each of the 8 TPU cores.
    """
    device = xm.xla_device()
    is_master = xm.is_master_ordinal()

    if is_master:
        print(f"============================================================")
        print(f"  TDCF — TPU v4-8  |  Tiny ImageNet (64x64)")
        print(f"============================================================")

    # ── 1. Setup BlockBandStore ──
    # Note: Ensure the dataset has been preprocessed for Tiny ImageNet first!
    train_store = BlockBandStore(os.path.join(args.data_dir, "train"))
    test_store = BlockBandStore(os.path.join(args.data_dir, "val"))

    train_dataset = BlockStoreDataset(train_store, train_store.labels)
    test_dataset = BlockStoreDataset(test_store, test_store.labels)

    # Distributed Samplers for TPU Multiprocessing
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.data_workers,
        collate_fn=lambda b: collate_block_store(b, train_store), drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.data_workers,
        collate_fn=lambda b: collate_block_store(b, test_store), drop_last=False
    )

    # ── 2. Model & Optimizer ──
    model = TinyImageNetCNN(num_classes=NUM_CLASSES).to(device)
    
    # Scale LR by world size (standard practice for DDP/XLA)
    scaled_lr = args.lr * xm.xrt_world_size()
    opt = optim.SGD(model.parameters(), lr=scaled_lr, momentum=0.9, weight_decay=args.wd)
    crit = nn.CrossEntropyLoss()
    
    sched_lr = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.total_epochs)

    # ── 3. TDCF Components ──
    estimator = SensitivityEstimator(P, NUM_BANDS)
    fsched = FidelityScheduler(NUM_BANDS, P, args.eta_f, args.eta_s)

    # --- PILOT PHASE ---
    if is_master:
        print(f"\n[PILOT PHASE] - {args.pilot_epochs} Epochs")
    
    for ep in range(args.pilot_epochs):
        model.train()
        train_store.set_fidelity(K_high=NUM_BANDS, K_low=NUM_BANDS, q=P)

        para_loader = pl.MpDeviceLoader(train_loader, device)

        for coeffs, y in para_loader:
            coeffs = coeffs.detach().requires_grad_(True)
            logits = model(coeffs)
            loss = crit(logits, y)
            loss.backward()
            xm.optimizer_step(opt)
            opt.zero_grad()

            if coeffs.grad is not None:
                estimator.update(coeffs.grad.cpu())

        estimator.finalize_epoch()
        if is_master:
            print(f"  Pilot {ep+1}/{args.pilot_epochs} completed.")

    # Fit the schedule from pilot statistics
    fsched.fit_from_pilot(estimator, args.total_epochs)
    if is_master:
        print(fsched.summary())

    if is_master:
        print(f"\n[ADAPTIVE TRAINING PHASE]")

    # --- ADAPTIVE TRAINING PHASE ---
    for ep in range(args.total_epochs):
        train_sampler.set_epoch(ep)
        
        # Get Schedule
        K_high, q_e = fsched.get_fidelity(ep)
        ps_idx = min(ep, len(estimator.patch_sensitivity_history) - 1)
        bs_idx = min(ep, len(estimator.band_sensitivity_history) - 1)
        
        ps = estimator.patch_sensitivity_history[ps_idx]
        bs = estimator.band_sensitivity_history[bs_idx]

        train_store.set_fidelity(K_high, args.k_low, q_e, patch_sensitivity=ps, band_sensitivity=bs, patch_policy=args.patch_policy)
        
        model.train()
        loss_s, correct, n = 0.0, 0.0, 0.0
        
        para_train = pl.MpDeviceLoader(train_loader, device)
        for coeffs, y in para_train:
            opt.zero_grad()
            logits = model(coeffs)
            loss = crit(logits, y)
            loss.backward()
            xm.optimizer_step(opt)
            
            # Local metrics
            loss_s += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)

        # Sync metrics across all TPU cores
        n_tensor = torch.tensor(n, device=device)
        corr_tensor = torch.tensor(correct, device=device)
        loss_tensor = torch.tensor(loss_s, device=device)
        
        n_total = xm.mesh_reduce("n_total", n_tensor, sum).item()
        corr_total = xm.mesh_reduce("corr_total", corr_tensor, sum).item()
        loss_total = xm.mesh_reduce("loss_total", loss_tensor, sum).item()
        
        tr_acc = corr_total / n_total
        tr_loss = loss_total / n_total

        # EVALUATION (Full fidelity)
        model.eval()
        test_store.set_fidelity(K_high=NUM_BANDS, K_low=NUM_BANDS, q=P)
        te_loss_s, te_correct, te_n = 0.0, 0.0, 0.0
        
        para_test = pl.MpDeviceLoader(test_loader, device)
        with torch.no_grad():
            for coeffs, y in para_test:
                logits = model(coeffs)
                loss = crit(logits, y)
                te_loss_s += loss.item() * y.size(0)
                te_correct += (logits.argmax(1) == y).sum().item()
                te_n += y.size(0)

        te_n_total = xm.mesh_reduce("te_n_total", torch.tensor(te_n, device=device), sum).item()
        te_corr_total = xm.mesh_reduce("te_corr_total", torch.tensor(te_correct, device=device), sum).item()
        te_loss_total = xm.mesh_reduce("te_loss_total", torch.tensor(te_loss_s, device=device), sum).item()

        te_acc = te_corr_total / te_n_total
        te_loss_avg = te_loss_total / te_n_total

        sched_lr.step()

        if is_master:
            io_ratio = train_store.get_io_ratio()
            print(f"  E {ep+1:3d}/{args.total_epochs} | K={K_high:2d} q={q_e:2d} | IO={io_ratio:.3f} | "
                  f"Tr Acc={tr_acc:.4f} | Te Acc={te_acc:.4f} | LR={opt.param_groups[0]['lr']:.4f}")
            train_store.reset_epoch_io()

    if is_master:
        print("Training complete! Run model save logic here.")
        # xm.save(model.state_dict(), "tpu_model.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/tiny_imagenet_block_store")
    parser.add_argument("--save_dir", type=str, default="./results/tpu_tiny_imagenet")
    parser.add_argument("--total_epochs", type=int, default=100)
    parser.add_argument("--pilot_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128) # Per core!
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--k_low", type=int, default=1)
    parser.add_argument("--eta_f", type=float, default=0.9)
    parser.add_argument("--eta_s", type=float, default=0.85)
    parser.add_argument("--patch_policy", type=str, default="greedy")
    parser.add_argument("--data_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    # Spawn 8 processes for TPU v4-8
    xmp.spawn(train_tpu_process, args=(args,), nprocs=8, start_method='spawn')
