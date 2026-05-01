"""TPU trainer for the CropDCT tile-band ImageNet-1K store.

This trainer reads full-resolution CropDCT records, executes random crop
queries against the tile-band store, reconstructs the selected crop with iDCT,
and trains the same ImageNet backbones used by the online WebDataset trainer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import traceback
from typing import Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla.distributed.xla_backend
import torchvision.models as tv_models

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data import DataLoader, Dataset

from tdcf.cropdct_store import CropDCTStore
from tdcf.fast_quant_store import FastShardBatchSampler


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


class CropDCTDecodedDataset(Dataset):
    """Decode CropDCT samples inside DataLoader workers.

    The TPU process receives fixed-size image tensors, while each worker owns a
    private CropDCTStore handle. This keeps irregular crop/iDCT work out of the
    XLA process and lets CPU decoding overlap with TPU training.
    """

    def __init__(self, root: str, img_size: int, freq_bands: Sequence[int], train: bool):
        self.root = root
        self.img_size = int(img_size)
        self.freq_bands = tuple(int(x) for x in freq_bands)
        self.train = bool(train)
        with open(os.path.join(root, "metadata.json")) as f:
            self.meta = json.load(f)
        self.records_per_shard = int(self.meta.get("records_per_shard", 1))
        self.num_bands = len(self.meta["band_specs"])
        self.global_index = np.load(os.path.join(root, "global_index.npy"), allow_pickle=False)
        self._store = None

    def _get_store(self):
        if self._store is None:
            self._store = CropDCTStore(self.root, device=torch.device("cpu"))
        return self._store

    def __len__(self):
        return len(self.global_index)

    def __getitem__(self, idx):
        store = self._get_store()
        store.reset_stats()
        image = store.image_record(int(idx))
        crop_box = None if self.train else make_center_crop_box(image.height, image.width)
        x, label = store.read_crop(
            int(idx),
            crop_box=crop_box,
            freq_bands=self.freq_bands,
            output_size=self.img_size,
        )
        stats = store.get_read_stats()
        stat_tensor = torch.tensor(
            [
                stats["bytes_read"],
                stats["touched_full_band_bytes"],
                stats["full_image_bytes"],
            ],
            dtype=torch.float64,
        )
        return x.squeeze(0), int(label), stat_tensor


def parse_args():
    p = argparse.ArgumentParser("CropDCT ImageNet-1K TPU Trainer")
    p.add_argument("--data_dir", type=str, default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--save_dir", type=str, default="./results/imagenet1k_cropdct")
    p.add_argument("--backbone", choices=["resnet50", "vit_b16"], default="resnet50")
    p.add_argument("--n_classes", type=int, default=1000)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--base_lr", type=float, default=0.1)
    p.add_argument("--lr_ref_batch", type=int, default=256)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--label_smooth", type=float, default=0.1)
    p.add_argument("--amp_bf16", action="store_true")
    p.add_argument("--loader_workers", type=int, default=int(os.environ.get("LOADER_WORKERS", os.environ.get("DECODE_WORKERS", "4"))))
    p.add_argument("--prefetch_factor", type=int, default=int(os.environ.get("PREFETCH_FACTOR", "2")))
    p.add_argument("--cpu_threads", type=int, default=int(os.environ.get("CPU_THREADS", "1")))
    p.add_argument("--budget_mode", action="store_true")
    p.add_argument("--cap", type=float, default=100.0, help="Maximum physical frequency-band cap in percent.")
    p.add_argument("--beta", type=float, default=None, help="Initial budget ratio. Defaults to cap/100.")
    p.add_argument("--max_beta", type=float, default=None, help="Final budget ratio. Defaults to cap/100.")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if not args.data_dir:
        raise ValueError("--data_dir is required, or set DATA_DIR=/path/to/cropdct_store")
    cap_ratio = max(0.0, min(1.0, args.cap / 100.0))
    if args.beta is None:
        args.beta = cap_ratio
    if args.max_beta is None:
        args.max_beta = cap_ratio
    if args.beta != 1.0 or args.max_beta != 1.0:
        args.budget_mode = True
    return args


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


def budget_ratio(args, epoch):
    if not args.budget_mode:
        return 1.0
    t = min(epoch, max(args.epochs - 1, 1))
    return args.beta + (args.max_beta - args.beta) * (t / max(args.epochs - 1, 1)) ** args.gamma


def freq_bands_for_ratio(num_bands: int, ratio: float) -> Sequence[int]:
    keep = max(1, min(num_bands, int(math.ceil(num_bands * float(ratio)))))
    return tuple(range(keep))


def make_center_crop_box(height: int, width: int):
    crop = min(int(height), int(width))
    return (int(height - crop) // 2, int(width - crop) // 2, crop, crop)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def make_loader(
    root,
    img_size,
    freq_bands,
    train,
    batch_size,
    world_size,
    rank,
    shuffle,
    drop_last,
    seed,
    epoch,
    num_workers,
    prefetch_factor,
):
    dataset = CropDCTDecodedDataset(root, img_size, freq_bands, train=train)
    sampler = FastShardBatchSampler(
        num_samples=len(dataset),
        records_per_shard=dataset.records_per_shard,
        batch_size=batch_size,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )
    sampler.set_epoch(epoch)
    generator = torch.Generator()
    generator.manual_seed(seed + epoch * 1009 + rank)
    kwargs = {
        "batch_sampler": sampler,
        "num_workers": int(num_workers),
        "pin_memory": False,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
        kwargs["persistent_workers"] = False
    return DataLoader(dataset, **kwargs), sampler, dataset


def train_process(index, args):
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()
    is_master = xm.is_master_ordinal()

    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(max(1, int(args.cpu_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if is_master:
        print("[INIT] Reading CropDCT metadata...", flush=True)
    train_root = os.path.join(args.data_dir, "train")
    val_root = os.path.join(args.data_dir, "val")
    train_info = CropDCTDecodedDataset(train_root, args.img_size, (0,), train=True)
    val_info = CropDCTDecodedDataset(val_root, args.img_size, tuple(range(train_info.num_bands)), train=False)
    full_val_bands = tuple(range(val_info.num_bands))
    val_loader, _val_sampler, _ = make_loader(
        val_root,
        args.img_size,
        full_val_bands,
        False,
        args.eval_batch_size,
        world_size,
        rank,
        False,
        False,
        args.seed,
        0,
        args.loader_workers,
        args.prefetch_factor,
    )

    global_bs = args.batch_size * world_size
    scaled_lr = args.base_lr * (global_bs / float(args.lr_ref_batch))
    model = build_model(args, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    opt = make_optimizer(args, model, scaled_lr)
    sched_lr = make_scheduler(args, opt)

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "latest.pt")
    best_acc = 0.0
    start_epoch = 0
    if args.resume and os.path.isfile(ckpt_path):
        if is_master:
            print(f"[RESUME] Loading checkpoint from {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        sched_lr.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"])
        best_acc = float(ckpt.get("best_acc", 0.0))

    if is_master:
        print("=" * 72, flush=True)
        print(
            f"TDCF ImageNet-1K CropDCT | backbone={args.backbone} | "
            f"bands={train_info.num_bands} img={args.img_size} "
            f"loader_workers={args.loader_workers} cpu_threads={args.cpu_threads} "
            f"global_bs={global_bs} scaled_lr={scaled_lr:.5f}",
            flush=True,
        )
        print("=" * 72, flush=True)

    for ep in range(start_epoch, args.epochs):
        ratio = budget_ratio(args, ep)
        freq_bands = freq_bands_for_ratio(train_info.num_bands, ratio)
        train_loader, train_sampler, _ = make_loader(
            train_root,
            args.img_size,
            freq_bands,
            True,
            args.batch_size,
            world_size,
            rank,
            True,
            True,
            args.seed,
            ep,
            args.loader_workers,
            args.prefetch_factor,
        )
        model.train()
        tr_loss_accum = tr_corr_accum = tr_n_accum = 0.0
        read_bytes_accum = touched_bytes_accum = full_bytes_accum = 0.0
        tr_loss_t = torch.zeros(1, device=device)
        tr_corr_t = torch.zeros(1, device=device)
        tr_n_t = torch.zeros(1, device=device)
        train_start = time.time()
        step_start = time.time()

        for step, (x_cpu, labels_cpu, stat_cpu) in enumerate(train_loader):
            stat_sum = stat_cpu.sum(dim=0)
            read_bytes_accum += float(stat_sum[0].item())
            touched_bytes_accum += float(stat_sum[1].item())
            full_bytes_accum += float(stat_sum[2].item())
            x = x_cpu.to(device)
            labels = labels_cpu.to(device)
            if step == 0 and is_master:
                print("  -> Got first batch from CropDCT store! Warming XLA step...", flush=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast("xla", dtype=torch.bfloat16, enabled=args.amp_bf16):
                logits = model(x)
                loss = criterion(logits, labels)
            loss.backward()
            xm.reduce_gradients(opt)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            xm.optimizer_step(opt)

            batch_n = torch.tensor([labels.size(0)], device=device, dtype=torch.float32)
            tr_loss_t.add_(loss.detach().float() * batch_n)
            tr_corr_t.add_((logits.argmax(1) == labels).sum().to(dtype=torch.float32).view(1))
            tr_n_t.add_(batch_n)
            xm.mark_step()

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

            if step % 100 == 0 or step == len(train_loader) - 1:
                tr_loss_accum += tr_loss_t.item()
                tr_corr_accum += tr_corr_t.item()
                tr_n_accum += tr_n_t.item()
                tr_loss_t.zero_()
                tr_corr_t.zero_()
                tr_n_t.zero_()

        tr_n_total = xm.mesh_reduce(f"cropdct_tr_n_{ep}", tr_n_accum, sum)
        tr_corr_total = xm.mesh_reduce(f"cropdct_tr_corr_{ep}", tr_corr_accum, sum)
        tr_loss_total = xm.mesh_reduce(f"cropdct_tr_loss_{ep}", tr_loss_accum, sum)
        read_bytes_total = xm.mesh_reduce(f"cropdct_read_bytes_{ep}", read_bytes_accum, sum)
        touched_bytes_total = xm.mesh_reduce(f"cropdct_touched_bytes_{ep}", touched_bytes_accum, sum)
        full_bytes_total = xm.mesh_reduce(f"cropdct_full_bytes_{ep}", full_bytes_accum, sum)
        tr_acc = tr_corr_total / tr_n_total if tr_n_total > 0 else 0.0
        tr_loss = tr_loss_total / tr_n_total if tr_n_total > 0 else 0.0
        train_time = time.time() - train_start

        model.eval()
        va_loss_accum = va_corr_accum = va_n_accum = 0.0
        va_loss_t = torch.zeros(1, device=device)
        va_corr_t = torch.zeros(1, device=device)
        va_n_t = torch.zeros(1, device=device)
        val_start = time.time()
        with torch.no_grad():
            for val_step, (x_cpu, labels_cpu, _stat_cpu) in enumerate(val_loader):
                x = x_cpu.to(device)
                labels = labels_cpu.to(device)
                with torch.autocast("xla", dtype=torch.bfloat16, enabled=args.amp_bf16):
                    logits = model(x)
                    loss = criterion(logits, labels)
                batch_n = torch.tensor([labels.size(0)], device=device, dtype=torch.float32)
                va_loss_t.add_(loss.detach().float() * batch_n)
                va_corr_t.add_((logits.argmax(1) == labels).sum().to(dtype=torch.float32).view(1))
                va_n_t.add_(batch_n)
                xm.mark_step()
                if val_step % 100 == 0 or val_step == len(val_loader) - 1:
                    va_loss_accum += va_loss_t.item()
                    va_corr_accum += va_corr_t.item()
                    va_n_accum += va_n_t.item()
                    va_loss_t.zero_()
                    va_corr_t.zero_()
                    va_n_t.zero_()

        va_n_total = xm.mesh_reduce(f"cropdct_va_n_{ep}", va_n_accum, sum)
        va_corr_total = xm.mesh_reduce(f"cropdct_va_corr_{ep}", va_corr_accum, sum)
        va_loss_total = xm.mesh_reduce(f"cropdct_va_loss_{ep}", va_loss_accum, sum)
        va_acc = va_corr_total / va_n_total if va_n_total > 0 else 0.0
        va_loss = va_loss_total / va_n_total if va_n_total > 0 else 0.0
        val_time = time.time() - val_start
        phys_io = read_bytes_total / touched_bytes_total if touched_bytes_total > 0 else 0.0
        img_io = read_bytes_total / full_bytes_total if full_bytes_total > 0 else 0.0
        sched_lr.step()

        if is_master:
            print(
                f"E {ep+1:3d}/{args.epochs} | BudgetRatio={ratio:.3f} | "
                f"Bands={len(freq_bands)}/{train_info.num_bands} | "
                f"PhysIO={phys_io:.3f} ImgIO={img_io:.3f} | "
                f"Tr Loss={tr_loss:.4f} Tr Acc={tr_acc:.4f} ({train_time:.1f}s) | "
                f"Val Loss={va_loss:.4f} Val Acc={va_acc:.4f} ({val_time:.1f}s) | "
                f"ReadGB={read_bytes_total/1e9:.3f}/{touched_bytes_total/1e9:.3f} | "
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
        xm.save(ckpt, ckpt_path)
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
        print(f"[rank {index}] Unhandled exception in CropDCT worker", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    parsed = parse_args()
    os.makedirs(parsed.save_dir, exist_ok=True)
    xmp.spawn(train_process_entry, args=(parsed,), start_method="spawn")
