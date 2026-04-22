import os
import time
import math
import random
import argparse
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla.distributed.xla_backend
import torchvision.models as tv_models

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from tdcf.sensitivity import BlockSensitivityEstimator
from tdcf.scheduler import FidelityScheduler, BudgetScheduler
from tdcf.io_dataloader import CropAwareBlockBandStore, BlockBandDataset


os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_BF16", "1")


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InputNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class ShardSampler(Sampler):
    def __init__(self, dataset_len: int, num_replicas: int, rank: int):
        self.indices = list(range(rank, dataset_len, num_replicas))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def parse_args():
    p = argparse.ArgumentParser("TDCF ImageNet-1K TPU Store Trainer")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Root containing train/ and val/ crop-aware stores.")
    p.add_argument("--save_dir", type=str, default="./results/tdcf_imagenet1k_store")

    p.add_argument("--backbone", choices=["resnet50", "vit_b16"], default="resnet50")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--n_classes", type=int, default=1000)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--pilot_epochs", type=int, default=10)
    p.add_argument("--skip_pilot", action="store_true")
    p.add_argument("--pilot_ratio", type=float, default=0.10)
    p.add_argument("--base_lr", type=float, default=0.1)
    p.add_argument("--pilot_lr", type=float, default=None)
    p.add_argument("--lr_ref_batch", type=int, default=256)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--label_smooth", type=float, default=0.1)
    p.add_argument("--amp_bf16", action="store_true")

    p.add_argument("--eta_f", type=float, default=0.9)
    p.add_argument("--eta_s", type=float, default=0.85)
    p.add_argument("--patch_policy", choices=["gradient", "random", "static", "greedy"],
                   default="greedy")
    p.add_argument("--budget_mode", action="store_true")
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--max_beta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.5)
    p.add_argument("--k_low", type=int, default=1)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def build_model(args, device: torch.device):
    if args.backbone == "resnet50":
        backbone = tv_models.resnet50(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, args.n_classes)
    else:
        backbone = tv_models.vit_b_16(weights=None, image_size=args.img_size)
        backbone.heads.head = nn.Linear(backbone.heads.head.in_features, args.n_classes)
    return nn.Sequential(InputNormalize(IMAGENET_MEAN, IMAGENET_STD), backbone).to(device)


def make_optimizer(args, model, lr: float):
    if args.backbone == "vit_b16":
        return optim.AdamW(
            model.parameters(), lr=lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
        )
    return optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )


def make_scheduler(args, optimizer, total_epochs: int):
    warmup_epochs = max(0, min(args.warmup_epochs, max(total_epochs - 1, 0)))

    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(args.min_lr / optimizer.param_groups[0]["initial_lr"], cosine)

    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_loader(dataset, batch_size, sampler, drop_last):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=drop_last,
        pin_memory=False,
    )


def accumulate_block_sensitivity(estimator, coeffs_grad: torch.Tensor):
    grad = coeffs_grad.detach()
    patch_sens = grad.abs().sum(dim=(2, 3, 4)).mean(dim=0)
    estimator._patch_accum += patch_sens
    grad_flat = grad.abs().mean(dim=(0, 1)).sum(dim=0).reshape(-1)
    for b, band_idx in enumerate(estimator.bands):
        estimator._band_accum[b] += grad_flat[band_idx.to(grad.device)].sum()
    estimator._step_count += 1


def train_process(index, args):
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()
    is_master = xm.is_master_ordinal()

    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_store = CropAwareBlockBandStore(
        os.path.join(args.data_dir, "train"), device=device, output_size=args.img_size
    )
    val_store = CropAwareBlockBandStore(
        os.path.join(args.data_dir, "val"), device=device, output_size=args.img_size
    )

    train_dataset = BlockBandDataset(train_store)
    val_dataset = BlockBandDataset(val_store)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = ShardSampler(len(val_dataset), num_replicas=world_size, rank=rank)

    train_loader = make_loader(train_dataset, args.batch_size, train_sampler, drop_last=True)
    val_loader = make_loader(val_dataset, args.eval_batch_size, val_sampler, drop_last=False)

    global_bs = args.batch_size * world_size
    scaled_lr = args.base_lr * (global_bs / float(args.lr_ref_batch))
    pilot_lr = args.pilot_lr if args.pilot_lr is not None else args.base_lr
    scaled_pilot_lr = pilot_lr * (global_bs / float(args.lr_ref_batch))

    estimator = BlockSensitivityEstimator(
        train_store.bs_h, train_store.nph, train_store.npw,
        train_store.num_bands, device=device
    )

    if args.budget_mode:
        scheduler = BudgetScheduler(
            train_store.num_bands, train_store.P,
            beta=args.beta, max_beta=args.max_beta,
            gamma=args.gamma, k_low=args.k_low,
        )
    else:
        scheduler = FidelityScheduler(
            train_store.num_bands, train_store.P, args.eta_f, args.eta_s
        )

    model = build_model(args, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    pilot_opt = make_optimizer(args, model, scaled_pilot_lr)

    train_steps = math.ceil(len(train_dataset) / global_bs)
    pilot_steps = max(1, int(train_steps * args.pilot_ratio))

    if is_master:
        print("=" * 72)
        print(
            f"TDCF ImageNet-1K Store | backbone={args.backbone} | img={args.img_size} | "
            f"canvas_patches={train_store.P} | global_bs={global_bs}"
        )
        print("=" * 72)

    if not args.skip_pilot:
        if is_master:
            print(f"\n[PILOT PHASE] - {args.pilot_epochs} epochs, ~{pilot_steps} steps/epoch")

        train_store.set_fidelity(
            train_store.num_bands, train_store.num_bands, train_store.P,
            patch_policy="gradient"
        )
        for ep in range(args.pilot_epochs):
            train_sampler.set_epoch(ep)
            model.train()
            pilot_step = 0

            for indices, labels in train_loader:
                if pilot_step >= pilot_steps:
                    break
                idx_np = indices.numpy()
                labels = labels.to(device)
                coeffs, crop_params, _visible, _k = train_store.read_coeffs(idx_np)
                coeffs = coeffs.to(device, non_blocking=True).detach().requires_grad_(True)
                x = train_store.reconstruct_crops(coeffs, idx_np, crop_params)

                pilot_opt.zero_grad(set_to_none=True)
                with torch.autocast("xla", dtype=torch.bfloat16, enabled=args.amp_bf16):
                    logits = model(x)
                    loss = criterion(logits, labels)
                loss.backward()

                if coeffs.grad is not None:
                    accumulate_block_sensitivity(estimator, coeffs.grad)

                xm.reduce_gradients(pilot_opt)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                xm.optimizer_step(pilot_opt)
                pilot_step += 1

            estimator.finalize_epoch()
            if is_master:
                K = estimator.compute_band_cutoff(ep, args.eta_f)
                q = estimator.compute_patch_quota(ep, args.eta_s)
                bs = estimator.band_sensitivity_history[ep]
                print(
                    f"  Pilot {ep+1:2d}/{args.pilot_epochs} | "
                    f"K_high={K:2d}/{train_store.num_bands} | q={q:3d}/{train_store.P} | "
                    f"band_low={bs[0]:.4f} band_high={bs[-1]:.4f}",
                    flush=True,
                )

        scheduler.fit_from_pilot(estimator, args.epochs)
        if is_master:
            print(scheduler.summary())
            print("\n[ADAPTIVE TRAINING PHASE]")
    else:
        if is_master:
            print("\n[SKIP PILOT] - Proceeding directly to store baseline training.")

    opt = make_optimizer(args, model, scaled_lr)
    sched_lr = make_scheduler(args, opt, args.epochs)

    best_acc = 0.0
    start_epoch = 0
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "latest.pt")
    if args.resume and os.path.isfile(ckpt_path):
        if is_master:
            print(f"\n[RESUME] Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        sched_lr.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        best_acc = ckpt.get("best_acc", 0.0)

    for ep in range(start_epoch, args.epochs):
        if not args.skip_pilot:
            ps_idx = min(ep, len(estimator.patch_sensitivity_history) - 1)
            bs_idx = min(ep, len(estimator.band_sensitivity_history) - 1)
            ps = estimator.patch_sensitivity_history[ps_idx]
            bs = estimator.band_sensitivity_history[bs_idx]
        else:
            ps = bs = None

        if args.skip_pilot:
            train_store.set_fidelity(
                train_store.num_bands, train_store.num_bands, train_store.P,
                patch_policy="gradient"
            )
        elif args.budget_mode:
            budget = scheduler.get_budget(ep)
            train_store.set_budget(
                budget, patch_sensitivity=ps, band_sensitivity=bs,
                k_low=args.k_low, patch_policy=args.patch_policy
            )
        else:
            K_high, q_e = scheduler.get_fidelity(ep)
            train_store.set_fidelity(
                K_high, args.k_low, q_e,
                patch_sensitivity=ps, band_sensitivity=bs,
                patch_policy=args.patch_policy
            )

        val_store.set_fidelity(
            val_store.num_bands, val_store.num_bands, val_store.P,
            patch_policy="gradient"
        )

        train_store.reset_epoch_io()
        train_sampler.set_epoch(ep)
        model.train()

        tr_loss_accum = 0.0
        tr_corr_accum = 0.0
        tr_n_accum = 0.0
        train_start = time.time()
        step_start = time.time()

        for step, (indices, labels) in enumerate(train_loader):
            idx_np = indices.numpy()
            labels = labels.to(device)
            x, _crop, _visible, _k = train_store.serve_indices(idx_np, deterministic_val=False)

            if step == 0 and is_master:
                print("  -> Got first batch from store loader! Building XLA graph...", flush=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast("xla", dtype=torch.bfloat16, enabled=args.amp_bf16):
                logits = model(x)
                loss = criterion(logits, labels)
            loss.backward()
            xm.reduce_gradients(opt)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            xm.optimizer_step(opt)

            batch_n = labels.size(0)
            tr_loss_accum += loss.item() * batch_n
            tr_corr_accum += (logits.argmax(1) == labels).sum().item()
            tr_n_accum += batch_n

            if is_master:
                if step == 0:
                    elapsed = time.time() - step_start
                    imgs_sec = global_bs / max(elapsed, 1e-6)
                    print(
                        f"  [Train] Epoch {ep+1} | Step {step:4d}/{len(train_loader)} | "
                        f"Time: {elapsed:.2f}s | {imgs_sec:.1f} imgs/sec | "
                        f"Loss: {loss.item():.4f} | Acc: {(logits.argmax(1) == labels).float().mean().item():.4f}",
                        flush=True,
                    )
                    step_start = time.time()
                elif step % 100 == 0:
                    elapsed = time.time() - step_start
                    imgs_sec = (100 * global_bs) / max(elapsed, 1e-6)
                    print(
                        f"  [Train] Epoch {ep+1} | Step {step:4d}/{len(train_loader)} | "
                        f"Time for 100 steps: {elapsed:.2f}s | {imgs_sec:.1f} imgs/sec",
                        flush=True,
                    )
                    step_start = time.time()

        tr_n_total = xm.mesh_reduce(f"store_tr_n_{ep}", tr_n_accum, sum)
        tr_corr_total = xm.mesh_reduce(f"store_tr_corr_{ep}", tr_corr_accum, sum)
        tr_loss_total = xm.mesh_reduce(f"store_tr_loss_{ep}", tr_loss_accum, sum)
        tr_acc = tr_corr_total / tr_n_total if tr_n_total > 0 else 0.0
        tr_loss = tr_loss_total / tr_n_total if tr_n_total > 0 else 0.0
        train_time = time.time() - train_start

        model.eval()
        va_loss_accum = 0.0
        va_corr_accum = 0.0
        va_n_accum = 0.0
        val_start = time.time()

        with torch.no_grad():
            for indices, labels in val_loader:
                idx_np = indices.numpy()
                labels = labels.to(device)
                x, _crop, _visible, _k = val_store.serve_indices(idx_np, deterministic_val=True)
                with torch.autocast("xla", dtype=torch.bfloat16, enabled=args.amp_bf16):
                    logits = model(x)
                    loss = criterion(logits, labels)
                batch_n = labels.size(0)
                va_loss_accum += loss.item() * batch_n
                va_corr_accum += (logits.argmax(1) == labels).sum().item()
                va_n_accum += batch_n

        va_n_total = xm.mesh_reduce(f"store_va_n_{ep}", va_n_accum, sum)
        va_corr_total = xm.mesh_reduce(f"store_va_corr_{ep}", va_corr_accum, sum)
        va_loss_total = xm.mesh_reduce(f"store_va_loss_{ep}", va_loss_accum, sum)
        va_acc = va_corr_total / va_n_total if va_n_total > 0 else 0.0
        va_loss = va_loss_total / va_n_total if va_n_total > 0 else 0.0
        val_time = time.time() - val_start

        io_ratio = train_store.get_io_ratio()
        sched_lr.step()

        if is_master:
            if args.skip_pilot:
                print(
                    f"E {ep+1:3d}/{args.epochs} | IO={io_ratio:.3f} (store baseline) | "
                    f"Tr Loss={tr_loss:.4f} Tr Acc={tr_acc:.4f} ({train_time:.1f}s) | "
                    f"Val Loss={va_loss:.4f} Val Acc={va_acc:.4f} ({val_time:.1f}s) | "
                    f"LR={opt.param_groups[0]['lr']:.2e}",
                    flush=True,
                )
            elif args.budget_mode:
                budget = scheduler.get_budget(ep)
                print(
                    f"E {ep+1:3d}/{args.epochs} | Budget={budget:4d}/{train_store.P * train_store.num_bands} | "
                    f"IO={io_ratio:.3f} | Tr Loss={tr_loss:.4f} Tr Acc={tr_acc:.4f} ({train_time:.1f}s) | "
                    f"Val Loss={va_loss:.4f} Val Acc={va_acc:.4f} ({val_time:.1f}s) | "
                    f"LR={opt.param_groups[0]['lr']:.2e}",
                    flush=True,
                )
            else:
                K_high, q_e = scheduler.get_fidelity(ep)
                print(
                    f"E {ep+1:3d}/{args.epochs} | K_hi={K_high} K_lo={args.k_low} q={q_e} | "
                    f"IO={io_ratio:.3f} | Tr Loss={tr_loss:.4f} Tr Acc={tr_acc:.4f} ({train_time:.1f}s) | "
                    f"Val Loss={va_loss:.4f} Val Acc={va_acc:.4f} ({val_time:.1f}s) | "
                    f"LR={opt.param_groups[0]['lr']:.2e}",
                    flush=True,
                )

        ckpt = {
            "epoch": ep + 1,
            "state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched_lr.state_dict(),
            "best_acc": best_acc,
            "val_acc": va_acc,
            "args": vars(args),
        }
        xm.save(ckpt, os.path.join(args.save_dir, "latest.pt"))
        if (ep + 1) % args.save_every == 0:
            xm.save(ckpt, os.path.join(args.save_dir, f"epoch_{ep+1:03d}.pt"))
        if va_acc > best_acc:
            best_acc = va_acc
            ckpt["best_acc"] = best_acc
            xm.save(ckpt, os.path.join(args.save_dir, "best.pt"))

    xm.master_print(f"Training complete. Best val acc: {best_acc:.4f}")


def train_process_entry(index, args):
    try:
        train_process(index, args)
    except Exception:
        print(f"[rank {index}] Unhandled exception in TPU store worker", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    xmp.spawn(train_process_entry, args=(args,), nprocs=8, start_method="spawn")
