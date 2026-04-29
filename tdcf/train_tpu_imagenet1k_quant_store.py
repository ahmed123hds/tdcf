import argparse
import math
import os
import random
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla.distributed.xla_backend
import torchvision.models as tv_models

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from torch.utils.data import DataLoader

from tdcf.quantized_store import BucketBatchSampler, OriginalQuantizedDCTStore
from tdcf.scheduler import BudgetScheduler


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


class FixedPilotStats:
    def __init__(self, num_bands: int, band_size: int):
        self.num_bands = int(num_bands)
        self.band_size = int(band_size)
        self.patch_sensitivity_history_by_bucket = {}
        self.band_sensitivity_history = []
        self.begin_epoch()

    def begin_epoch(self):
        self.patch_accum_by_bucket = {}
        self.patch_count_by_bucket = {}
        self.band_accum = None
        self.step_count = 0

    def accumulate(self, bucket_id: int, grad_zz: torch.Tensor):
        grad = grad_zz.detach().abs()
        patch = grad.sum(dim=(2, 3)).mean(dim=0).cpu()
        band_vals = []
        for band in range(self.num_bands):
            start = band * self.band_size
            end = (band + 1) * self.band_size if band < self.num_bands - 1 else grad.shape[-1]
            band_vals.append(grad[:, :, :, start:end].mean(dim=(0, 1, 2, 3)).cpu())
        band = torch.stack(band_vals)
        if bucket_id not in self.patch_accum_by_bucket:
            self.patch_accum_by_bucket[bucket_id] = torch.zeros_like(patch)
            self.patch_count_by_bucket[bucket_id] = 0
        self.patch_accum_by_bucket[bucket_id] += patch
        self.patch_count_by_bucket[bucket_id] += 1
        self.band_accum = band if self.band_accum is None else self.band_accum + band
        self.step_count += 1

    def finalize_epoch(self):
        if self.step_count == 0:
            return
        band = self.band_accum / self.step_count
        band = band / (band.sum() + 1e-8)
        self.band_sensitivity_history.append(band.numpy().astype(np.float32))
        for bucket_id, patch_accum in self.patch_accum_by_bucket.items():
            patch = patch_accum / max(self.patch_count_by_bucket[bucket_id], 1)
            patch = patch / (patch.sum() + 1e-8)
            self.patch_sensitivity_history_by_bucket.setdefault(bucket_id, []).append(
                patch.numpy().astype(np.float32)
            )


def parse_args():
    p = argparse.ArgumentParser("TDCF ImageNet-1K Quantized Store TPU Trainer")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Root containing train/ and val/ quantized stores.")
    p.add_argument("--save_dir", type=str, default="./results/imagenet1k_quant_store")
    p.add_argument("--backbone", choices=["resnet50", "vit_b16"], default="resnet50")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--n_classes", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--pilot_epochs", type=int, default=10)
    p.add_argument("--pilot_ratio", type=float, default=0.10)
    p.add_argument("--skip_pilot", action="store_true")
    p.add_argument("--base_lr", type=float, default=0.1)
    p.add_argument("--pilot_lr", type=float, default=None)
    p.add_argument("--lr_ref_batch", type=int, default=256)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--label_smooth", type=float, default=0.1)
    p.add_argument("--amp_bf16", action="store_true")
    p.add_argument("--budget_mode", action="store_true")
    p.add_argument("--beta", type=float, default=0.6)
    p.add_argument("--max_beta", type=float, default=0.9)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--k_low", type=int, default=1)
    p.add_argument("--patch_policy", choices=["greedy", "random", "static"], default="greedy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def build_model(args, device):
    if args.backbone == "resnet50":
        backbone = tv_models.resnet50(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, args.n_classes)
    else:
        backbone = tv_models.vit_b_16(weights=None, image_size=args.img_size)
        backbone.heads.head = nn.Linear(backbone.heads.head.in_features, args.n_classes)
    return nn.Sequential(InputNormalize(IMAGENET_MEAN, IMAGENET_STD), backbone).to(device)


def make_optimizer(args, model, lr):
    if args.backbone == "vit_b16":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    return optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )


def make_scheduler(args, optimizer):
    warmup_epochs = max(0, min(args.warmup_epochs, max(args.epochs - 1, 0)))

    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(args.min_lr / optimizer.param_groups[0]["initial_lr"], cosine)

    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_loader(store, batch_size, world_size, rank, shuffle, drop_last, seed):
    sampler = BucketBatchSampler(
        bucket_ids=store.sample_bucket_ids,
        batch_size=batch_size,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )
    loader = DataLoader(
        store.get_dataset(),
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=False,
    )
    return loader, sampler


def train_process(index, args):
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()
    is_master = xm.is_master_ordinal()

    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if is_master:
        print("[INIT] Opening quantized train store...", flush=True)
    train_store = OriginalQuantizedDCTStore(
        os.path.join(args.data_dir, "train"),
        device=device,
        output_size=args.img_size,
    )
    if is_master:
        print("[INIT] Opening quantized val store...", flush=True)
    val_store = OriginalQuantizedDCTStore(
        os.path.join(args.data_dir, "val"),
        device=device,
        output_size=args.img_size,
    )
    train_loader, train_sampler = make_loader(
        train_store, args.batch_size, world_size, rank, True, True, args.seed
    )
    val_loader, val_sampler = make_loader(
        val_store, args.eval_batch_size, world_size, rank, False, False, args.seed
    )

    global_bs = args.batch_size * world_size
    scaled_lr = args.base_lr * (global_bs / float(args.lr_ref_batch))
    pilot_lr = args.pilot_lr if args.pilot_lr is not None else args.base_lr
    scaled_pilot_lr = pilot_lr * (global_bs / float(args.lr_ref_batch))

    if args.budget_mode:
        scheduler = BudgetScheduler(
            train_store.num_bands,
            num_patches=196,
            beta=args.beta,
            max_beta=args.max_beta,
            gamma=args.gamma,
            k_low=args.k_low,
        )
    else:
        scheduler = None

    model = build_model(args, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    pilot_opt = make_optimizer(args, model, scaled_pilot_lr)
    pilot_stats = FixedPilotStats(train_store.num_bands, train_store.band_size)
    pilot_steps = max(1, int(len(train_loader) * args.pilot_ratio))

    if is_master:
        print("=" * 72)
        print(
            f"TDCF ImageNet-1K Quantized Store | backbone={args.backbone} | "
            f"buckets={len(train_store.bucket_infos)} crop={args.img_size} "
            f"global_bs={global_bs} scaled_lr={scaled_lr:.5f}"
        )
        print("=" * 72)

    if not args.skip_pilot:
        if is_master:
            print(f"\n[PILOT PHASE] - {args.pilot_epochs} epochs, ~{pilot_steps} steps/epoch")
        train_store.set_full_fidelity()
        for ep in range(args.pilot_epochs):
            epoch_start = time.time()
            train_sampler.set_epoch(ep)
            pilot_stats.begin_epoch()
            model.train()
            for step, (indices, labels) in enumerate(train_loader):
                if step >= pilot_steps:
                    break
                idx_np = indices.numpy()
                labels = labels.to(device)
                coeffs_zz, crop_params, _visible, _k, bucket_id = train_store.read_coeffs(idx_np)
                coeffs_zz = coeffs_zz.to(device, non_blocking=True).detach().requires_grad_(True)
                x = train_store.reconstruct_crops(coeffs_zz, idx_np, crop_params, bucket_id)

                pilot_opt.zero_grad(set_to_none=True)
                with torch.autocast("xla", dtype=torch.bfloat16, enabled=args.amp_bf16):
                    logits = model(x)
                    loss = criterion(logits, labels)
                loss.backward()
                if coeffs_zz.grad is not None:
                    pilot_stats.accumulate(bucket_id, coeffs_zz.grad)
                xm.reduce_gradients(pilot_opt)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                xm.optimizer_step(pilot_opt)
            pilot_stats.finalize_epoch()
            if is_master:
                bs = pilot_stats.band_sensitivity_history[-1]
                print(
                    f"  Pilot {ep+1:2d}/{args.pilot_epochs} | "
                    f"band_low={bs[0]:.4f} band_high={bs[-1]:.4f} | "
                    f"{time.time() - epoch_start:.1f}s",
                    flush=True,
                )
        if scheduler is not None:
            scheduler.fit_from_pilot(pilot_stats, args.epochs)
        if is_master:
            if scheduler is not None:
                print(scheduler.summary())
            print("\n[ADAPTIVE TRAINING PHASE]")
    else:
        if is_master:
            print("\n[SKIP PILOT] - Proceeding directly to quantized-store baseline.")

    opt = make_optimizer(args, model, scaled_lr)
    sched_lr = make_scheduler(args, opt)

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
        start_epoch = int(ckpt["epoch"])
        best_acc = float(ckpt.get("best_acc", 0.0))

    for ep in range(start_epoch, args.epochs):
        train_sampler.set_epoch(ep)
        if args.skip_pilot:
            train_store.set_full_fidelity()
        elif args.budget_mode:
            ps = {
                bucket_id: hist[min(ep, len(hist) - 1)]
                for bucket_id, hist in pilot_stats.patch_sensitivity_history_by_bucket.items()
            }
            bs = pilot_stats.band_sensitivity_history[min(ep, len(pilot_stats.band_sensitivity_history) - 1)]
            train_store.set_budget_ratio(
                scheduler.get_io_ratio(ep),
                patch_sensitivity_by_bucket=ps,
                band_sensitivity=bs,
                k_low=args.k_low,
                patch_policy=args.patch_policy,
            )
        else:
            raise ValueError("Use --budget_mode for capped quantized-store runs.")
        val_store.set_full_fidelity()

        train_store.reset_epoch_io()
        model.train()
        tr_loss_accum = tr_corr_accum = tr_n_accum = 0.0
        train_start = time.time()
        step_start = time.time()
        for step, (indices, labels) in enumerate(train_loader):
            idx_np = indices.numpy()
            labels = labels.to(device)
            x, _crop, _visible, _k, _bucket_id = train_store.serve_indices(idx_np, deterministic_val=False)

            if step == 0 and is_master:
                print("  -> Got first batch from quantized store! Building XLA graph...", flush=True)

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
                    print(
                        f"  [Train] Epoch {ep+1} | Step {step:4d}/{len(train_loader)} | "
                        f"Time: {elapsed:.2f}s | {global_bs / max(elapsed, 1e-6):.1f} imgs/sec | "
                        f"Loss: {loss.item():.4f}",
                        flush=True,
                    )
                    step_start = time.time()
                elif step % 100 == 0:
                    elapsed = time.time() - step_start
                    print(
                        f"  [Train] Epoch {ep+1} | Step {step:4d}/{len(train_loader)} | "
                        f"Time for 100 steps: {elapsed:.2f}s | "
                        f"{100 * global_bs / max(elapsed, 1e-6):.1f} imgs/sec",
                        flush=True,
                    )
                    step_start = time.time()

        tr_n_total = xm.mesh_reduce(f"qstore_tr_n_{ep}", tr_n_accum, sum)
        tr_corr_total = xm.mesh_reduce(f"qstore_tr_corr_{ep}", tr_corr_accum, sum)
        tr_loss_total = xm.mesh_reduce(f"qstore_tr_loss_{ep}", tr_loss_accum, sum)
        tr_acc = tr_corr_total / tr_n_total if tr_n_total > 0 else 0.0
        tr_loss = tr_loss_total / tr_n_total if tr_n_total > 0 else 0.0
        train_time = time.time() - train_start

        val_store.reset_epoch_io()
        model.eval()
        va_loss_accum = va_corr_accum = va_n_accum = 0.0
        val_start = time.time()
        with torch.no_grad():
            for indices, labels in val_loader:
                idx_np = indices.numpy()
                labels = labels.to(device)
                x, _crop, _visible, _k, _bucket_id = val_store.serve_indices(idx_np, deterministic_val=True)
                with torch.autocast("xla", dtype=torch.bfloat16, enabled=args.amp_bf16):
                    logits = model(x)
                    loss = criterion(logits, labels)
                batch_n = labels.size(0)
                va_loss_accum += loss.item() * batch_n
                va_corr_accum += (logits.argmax(1) == labels).sum().item()
                va_n_accum += batch_n

        va_n_total = xm.mesh_reduce(f"qstore_va_n_{ep}", va_n_accum, sum)
        va_corr_total = xm.mesh_reduce(f"qstore_va_corr_{ep}", va_corr_accum, sum)
        va_loss_total = xm.mesh_reduce(f"qstore_va_loss_{ep}", va_loss_accum, sum)
        va_acc = va_corr_total / va_n_total if va_n_total > 0 else 0.0
        va_loss = va_loss_total / va_n_total if va_n_total > 0 else 0.0
        val_time = time.time() - val_start

        io_ratio = train_store.get_io_ratio()
        read_stats = train_store.get_read_stats()
        sched_lr.step()

        if is_master:
            if args.skip_pilot:
                budget_text = "baseline"
            else:
                budget_text = f"{scheduler.get_io_ratio(ep):.3f}"
            print(
                f"E {ep+1:3d}/{args.epochs} | BudgetRatio={budget_text} | "
                f"PhysIO={io_ratio:.3f} | Tr Loss={tr_loss:.4f} Tr Acc={tr_acc:.4f} ({train_time:.1f}s) | "
                f"Val Loss={va_loss:.4f} Val Acc={va_acc:.4f} ({val_time:.1f}s) | "
                f"ReadGB={read_stats['bytes_read_epoch']/1e9:.3f}/{read_stats['full_bytes_epoch']/1e9:.3f} | "
                f"Reads={read_stats['read_ops_epoch']} avg_read={read_stats['avg_bytes_per_read']:.1f} | "
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

    train_store.close()
    val_store.close()
    xm.master_print(f"Training complete. Best val acc: {best_acc:.4f}")


def train_process_entry(index, args):
    try:
        train_process(index, args)
    except Exception:
        print(f"[rank {index}] Unhandled exception in quantized-store worker", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    xmp.spawn(train_process_entry, args=(args,), start_method="spawn")
