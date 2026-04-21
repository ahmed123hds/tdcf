import os
import math
import argparse
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models
import torchvision.transforms as T
import webdataset as wds

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from tdcf.sensitivity import BlockSensitivityEstimator
from tdcf.scheduler import FidelityScheduler, BudgetScheduler
from tdcf.transforms import block_dct2d, block_idct2d


os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_BF16", "1")


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InputNormalize(nn.Module):
    """Normalize raw [0,1] images inside the model graph."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def parse_args():
    p = argparse.ArgumentParser("TDCF ImageNet-1K TPU Trainer")
    p.add_argument("--train_shards", type=str, required=True)
    p.add_argument("--val_shards", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./results/tdcf_imagenet1k")

    p.add_argument("--backbone", choices=["resnet50", "vit_b16"], default="resnet50")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--block_size", type=int, default=16)
    p.add_argument("--num_bands", type=int, default=16)
    p.add_argument("--n_classes", type=int, default=1000)

    p.add_argument("--batch_size", type=int, default=64, help="Per TPU core")
    p.add_argument("--eval_batch_size", type=int, default=64, help="Per TPU core")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--pilot_epochs", type=int, default=10)
    p.add_argument("--skip_pilot", action="store_true",
                   help="Skip pilot phase entirely and run full fidelity (for baseline runs).")
    p.add_argument("--pilot_ratio", type=float, default=0.10,
                   help="Fraction of an epoch used for each pilot epoch.")
    p.add_argument("--base_lr", type=float, default=0.1,
                   help="Reference LR at --lr_ref_batch samples.")
    p.add_argument("--pilot_lr", type=float, default=None,
                   help="Optional LR used only during pilot.")
    p.add_argument("--lr_ref_batch", type=int, default=256,
                   help="Global batch size used as the LR reference point.")
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
    p.add_argument("--budget_mode", action="store_true",
                   help="Use budget-constrained scheduling.")
    p.add_argument("--beta", type=float, default=0.5,
                   help="Starting budget ratio when --budget_mode is enabled.")
    p.add_argument("--max_beta", type=float, default=1.0,
                   help="Ending budget ratio when --budget_mode is enabled.")
    p.add_argument("--gamma", type=float, default=1.5,
                   help="Budget ramp exponent when --budget_mode is enabled.")
    p.add_argument("--k_low", type=int, default=1)

    p.add_argument("--train_samples", type=int, default=1281167)
    p.add_argument("--val_samples", type=int, default=50000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest.pt checkpoint in --save_dir.")
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


def identity_label(label):
    return int(label)


def build_wds_loader(shards_url: str, batch_size: int, args, is_training: bool):
    if is_training:
        transform = T.Compose([
            T.RandomResizedCrop(
                args.img_size, scale=(0.08, 1.0), interpolation=T.InterpolationMode.BICUBIC
            ),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ColorJitter(0.4, 0.4, 0.4),
            T.ToTensor(),
        ])
    else:
        transform = T.Compose([
            T.Resize(int(args.img_size * 1.14), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(args.img_size),
            T.ToTensor(),
        ])

    dataset = (
        wds.WebDataset(
            shards_url,
            resampled=is_training,
            shardshuffle=is_training,
            nodesplitter=wds.split_by_node,
        )
        .compose(wds.split_by_worker)
        .shuffle(5000 if is_training else 0)
        .decode("pil")
        .to_tuple("jpg;jpeg;png", "cls")
        .map_tuple(transform, identity_label)
        .batched(batch_size, partial=not is_training)
    )
    if is_training:
        dataset = dataset.with_epoch(math.ceil(args.train_samples / (batch_size * xm.xrt_world_size())))

    return wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )


def make_epoch_accumulators(device: torch.device):
    return (
        torch.zeros((), device=device),
        torch.zeros((), device=device),
        torch.zeros((), device=device),
    )


def _topk_mask(scores: torch.Tensor, q: int) -> torch.Tensor:
    B, P = scores.shape
    if q >= P:
        return torch.ones((B, P), dtype=torch.bool, device=scores.device)
    top_idx = torch.topk(scores, k=q, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros((B, P), dtype=torch.bool, device=scores.device)
    mask.scatter_(1, top_idx, True)
    return mask


def compute_k_allocation(
    coeffs: torch.Tensor,
    patch_sensitivity,
    band_sensitivity,
    num_bands: int,
    k_low: int,
    patch_policy: str,
    budget_bands: int = None,
    K_high: int = None,
    q: int = None,
):
    """
    XLA-safe per-sample, per-patch frequency budget computation.

    Returns k_alloc (B, P) int64 on the same device as coeffs.
    Uses only static shapes and no dynamic scatter/index_select.
    """
    B, P, _, bs_h, bs_w = coeffs.shape
    device = coeffs.device

    patch_signal = coeffs.detach().abs().mean(dim=(2, 3, 4))  # (B, P)
    patch_weights = torch.as_tensor(
        patch_sensitivity, device=device, dtype=patch_signal.dtype
    )
    band_weights = torch.as_tensor(
        band_sensitivity, device=device, dtype=patch_signal.dtype
    )
    utility = patch_signal * patch_weights.unsqueeze(0)  # (B, P)

    if budget_bands is not None:
        # Greedy budget: start everyone at k_low, then assign extra bands
        # to the (patch, band) slots with highest marginal utility.
        # Done with a threshold on a static score matrix — fully XLA-safe.
        max_k = num_bands
        extra_total = int(budget_bands) - P * int(k_low)
        if extra_total <= 0:
            return torch.full((B, P), int(k_low), dtype=torch.long, device=device)

        # Score matrix (B, P, extra_bands): marginal utility of each extra band
        extra_bands = max_k - int(k_low)          # number of band increments possible
        bs_extra = band_weights[int(k_low):]       # (extra_bands,)
        # (B, P, extra_bands)
        scores_3d = utility.unsqueeze(-1) * bs_extra.view(1, 1, -1)
        # Flatten to (B, P*extra_bands), find threshold via topk
        flat = scores_3d.reshape(B, P * extra_bands)
        k_budget = min(int(extra_total), P * extra_bands)
        threshold = torch.topk(flat, k=k_budget, dim=1, largest=True, sorted=True).values[:, -1:]  # (B, 1)
        # Each patch gets the number of extra bands whose score > threshold
        band_granted = (scores_3d > threshold.unsqueeze(-1)).sum(dim=-1)  # (B, P)
        k_alloc = (int(k_low) + band_granted).clamp(int(k_low), int(num_bands))
        return k_alloc.long()

    # Binary fidelity: top-q patches get K_high, rest get k_low
    assert K_high is not None and q is not None
    if patch_policy == "random":
        scores = torch.rand(B, P, device=device)
    elif patch_policy == "static":
        scores = patch_weights.unsqueeze(0).expand(B, P)
    else:
        scores = utility
    important = _topk_mask(scores, int(q))
    return torch.where(
        important,
        torch.full((B, P), int(K_high), dtype=torch.long, device=device),
        torch.full((B, P), int(k_low), dtype=torch.long, device=device),
    )


def apply_k_allocation(
    coeffs: torch.Tensor,
    k_alloc: torch.Tensor,
    num_bands: int,
):
    """
    XLA-safe frequency masking using a static positional cutoff.

    Replaces dynamic index_select zig-zag reordering with a broadcasted
    boolean mask over zig-zag band position, which compiles statically.

    Args:
        coeffs:    (B, P, C, bs_h, bs_w) block-DCT coefficients.
        k_alloc:   (B, P) int64 — number of bands to keep per patch.
        num_bands: Total number of bands.
    Returns:
        (B, P, C, bs_h, bs_w) masked coefficients.
    """
    B, P, C, bs_h, bs_w = coeffs.shape
    total_coeffs = bs_h * bs_w
    band_size = total_coeffs // num_bands

    # pos: (1, 1, 1, total_coeffs),  cutoff: (B, P, 1, 1)
    # mask: (B, P, 1, total_coeffs) broadcasts over C dimension
    pos = torch.arange(total_coeffs, device=coeffs.device, dtype=torch.long).view(1, 1, 1, total_coeffs)
    cutoff = (k_alloc * band_size).view(B, P, 1, 1)  # (B, P, 1, 1)
    mask = (pos < cutoff)  # (B, P, 1, total_coeffs) — broadcasts over C

    coeffs_flat = coeffs.reshape(B, P, C, total_coeffs)
    coeffs_masked = coeffs_flat * mask.float()
    return coeffs_masked.reshape(B, P, C, bs_h, bs_w)


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
    core_seed = args.seed + xm.get_ordinal()
    torch.manual_seed(core_seed)
    np.random.seed(core_seed)
    random.seed(core_seed)

    global_bs = args.batch_size * world_size
    scaled_lr = args.base_lr * (global_bs / float(args.lr_ref_batch))
    pilot_lr = args.pilot_lr if args.pilot_lr is not None else args.base_lr
    scaled_pilot_lr = pilot_lr * (global_bs / float(args.lr_ref_batch))

    nph = args.img_size // args.block_size
    npw = args.img_size // args.block_size
    num_patches = nph * npw
    total_budget = num_patches * args.num_bands
    train_steps = math.ceil(args.train_samples / global_bs)
    val_steps = math.ceil(args.val_samples / (args.eval_batch_size * world_size))
    pilot_steps = max(1, int(train_steps * args.pilot_ratio))

    if xm.is_master_ordinal():
        print("=" * 72)
        print(f"TDCF ImageNet-1K | backbone={args.backbone} | img={args.img_size} | block={args.block_size}")
        print(f"cores={world_size} | global_bs={global_bs} | scaled_lr={scaled_lr:.5f} | amp_bf16={args.amp_bf16}")
        print("=" * 72)

    model = build_model(args, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    pilot_opt = make_optimizer(args, model, scaled_pilot_lr)
    estimator = BlockSensitivityEstimator(args.block_size, nph, npw, args.num_bands, device=device)

    if args.budget_mode:
        scheduler = BudgetScheduler(
            args.num_bands, num_patches,
            beta=args.beta, max_beta=args.max_beta,
            gamma=args.gamma, k_low=args.k_low,
        )
    else:
        scheduler = FidelityScheduler(args.num_bands, num_patches, args.eta_f, args.eta_s)

    train_loader = build_wds_loader(args.train_shards, args.batch_size, args, True)
    pilot_loader = build_wds_loader(args.train_shards, args.batch_size, args, True)
    val_loader = build_wds_loader(args.val_shards, args.eval_batch_size, args, False)

    # Pilot
    if not args.skip_pilot:
        if xm.is_master_ordinal():
            print(f"\n[PILOT PHASE] - {args.pilot_epochs} epochs, ~{pilot_steps} steps/epoch")

        for ep in range(args.pilot_epochs):
            pl_pilot = pl.ParallelLoader(pilot_loader, [device])
            para_pilot = pl_pilot.per_device_loader(device)
            model.train()

            for step, batch in enumerate(para_pilot):
                if step >= pilot_steps:
                    break
                images, labels = batch
                labels = labels.long()
                coeffs = block_dct2d(images, block_size=args.block_size).detach().requires_grad_(True)
                x = block_idct2d(coeffs, nph, npw)

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

            estimator.finalize_epoch()
            if xm.is_master_ordinal():
                K = estimator.compute_band_cutoff(ep, args.eta_f)
                q = estimator.compute_patch_quota(ep, args.eta_s)
                bs = estimator.band_sensitivity_history[ep]
                print(
                    f"  Pilot {ep+1:2d}/{args.pilot_epochs} | "
                    f"K_high={K:2d}/{args.num_bands} | q={q:3d}/{num_patches} | "
                    f"band_low={bs[0]:.4f} band_high={bs[-1]:.4f}"
                )

        scheduler.fit_from_pilot(estimator, args.epochs)
        if xm.is_master_ordinal():
            print(scheduler.summary())
            print("\n[ADAPTIVE TRAINING PHASE]")
    else:
        if xm.is_master_ordinal():
            print("\n[SKIP PILOT] - Proceeding directly to baseline training.")

    # Fresh optimizer after pilot.
    opt = make_optimizer(args, model, scaled_lr)
    sched_lr = make_scheduler(args, opt, args.epochs)

    best_acc = 0.0
    start_epoch = 0
    os.makedirs(args.save_dir, exist_ok=True)

    # Resume from checkpoint if requested
    ckpt_path = os.path.join(args.save_dir, "latest.pt")
    if args.resume and os.path.isfile(ckpt_path):
        if xm.is_master_ordinal():
            print(f"\n[RESUME] Loading checkpoint from {ckpt_path}")
        ckpt_data = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt_data["state_dict"])
        opt.load_state_dict(ckpt_data["optimizer"])
        sched_lr.load_state_dict(ckpt_data["scheduler"])
        start_epoch = ckpt_data["epoch"]   # already-completed epochs
        best_acc = ckpt_data.get("best_acc", 0.0)
        if xm.is_master_ordinal():
            print(f"[RESUME] Resuming from epoch {start_epoch + 1} | best_acc={best_acc:.4f}")
    elif args.resume:
        if xm.is_master_ordinal():
            print(f"[RESUME] No checkpoint found at {ckpt_path}, starting from scratch.")

    for ep in range(start_epoch, args.epochs):
        if not args.skip_pilot:
            ps_idx = min(ep, len(estimator.patch_sensitivity_history) - 1)
            bs_idx = min(ep, len(estimator.band_sensitivity_history) - 1)
            ps = estimator.patch_sensitivity_history[ps_idx]
            bs = estimator.band_sensitivity_history[bs_idx]

        if args.skip_pilot:
            budget = total_budget
            io_ratio = 1.0
            K_high = q_e = None
        elif args.budget_mode:
            budget = scheduler.get_budget(ep)
            io_ratio = scheduler.get_io_ratio(ep)
            K_high = q_e = None
        else:
            K_high, q_e = scheduler.get_fidelity(ep)
            io_ratio = (q_e * K_high + (num_patches - q_e) * args.k_low) / float(total_budget)
            budget = None

        # Train
        pl_train = pl.ParallelLoader(train_loader, [device])
        para_train = pl_train.per_device_loader(device)
        model.train()
        tr_loss_s, tr_correct, tr_n = make_epoch_accumulators(device)

        for step, batch in enumerate(para_train):
            if step >= train_steps:
                break
            images, labels = batch
            labels = labels.long()
            coeffs = block_dct2d(images, block_size=args.block_size)

            if args.skip_pilot:
                # Bypass masking entirely for baseline
                x = block_idct2d(coeffs, nph, npw)
            else:
                if args.budget_mode:
                    k_alloc = compute_k_allocation(
                        coeffs, ps, bs, args.num_bands, args.k_low,
                        patch_policy="greedy", budget_bands=budget
                    )
                else:
                    k_alloc = compute_k_allocation(
                        coeffs, ps, bs, args.num_bands, args.k_low,
                        patch_policy=args.patch_policy, K_high=K_high, q=q_e
                    )
                coeffs_masked = apply_k_allocation(coeffs, k_alloc, args.num_bands)
                x = block_idct2d(coeffs_masked, nph, npw)

            opt.zero_grad(set_to_none=True)
            with torch.autocast("xla", dtype=torch.bfloat16, enabled=args.amp_bf16):
                logits = model(x)
                loss = criterion(logits, labels)
            loss.backward()
            xm.reduce_gradients(opt)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            xm.optimizer_step(opt)

            batch_n = labels.new_tensor(labels.size(0), dtype=torch.float32)
            tr_loss_s += loss.detach() * batch_n
            tr_correct += (logits.argmax(1) == labels).sum().to(dtype=torch.float32)
            tr_n += batch_n

        tr_n_total = xm.mesh_reduce("tr_n", tr_n, sum)
        tr_corr_total = xm.mesh_reduce("tr_corr", tr_correct, sum)
        tr_loss_total = xm.mesh_reduce("tr_loss", tr_loss_s, sum)
        tr_acc = (tr_corr_total / tr_n_total).item()
        tr_loss = (tr_loss_total / tr_n_total).item()

        # Validate at full fidelity
        pl_val = pl.ParallelLoader(val_loader, [device])
        para_val = pl_val.per_device_loader(device)
        model.eval()
        va_loss_s, va_correct, va_n = make_epoch_accumulators(device)

        with torch.no_grad():
            for step, batch in enumerate(para_val):
                if step >= val_steps:
                    break
                images, labels = batch
                labels = labels.long()
                with torch.autocast("xla", dtype=torch.bfloat16, enabled=args.amp_bf16):
                    logits = model(images)
                    loss = criterion(logits, labels)
                batch_n = labels.new_tensor(labels.size(0), dtype=torch.float32)
                va_loss_s += loss * batch_n
                va_correct += (logits.argmax(1) == labels).sum().to(dtype=torch.float32)
                va_n += batch_n

        va_n_total = xm.mesh_reduce("va_n", va_n, sum)
        va_corr_total = xm.mesh_reduce("va_corr", va_correct, sum)
        va_loss_total = xm.mesh_reduce("va_loss", va_loss_s, sum)
        va_acc = (va_corr_total / va_n_total).item()
        va_loss = (va_loss_total / va_n_total).item()

        sched_lr.step()

        if xm.is_master_ordinal():
            if args.skip_pilot:
                print(
                    f"E {ep+1:3d}/{args.epochs} | IO=1.000 (baseline) | "
                    f"Tr Loss={tr_loss:.4f} Tr Acc={tr_acc:.4f} | "
                    f"Val Loss={va_loss:.4f} Val Acc={va_acc:.4f} | "
                    f"LR={opt.param_groups[0]['lr']:.2e}",
                    flush=True,
                )
            elif args.budget_mode:
                print(
                    f"E {ep+1:3d}/{args.epochs} | Budget={budget:4d}/{total_budget} | "
                    f"IO={io_ratio:.3f} | Tr Loss={tr_loss:.4f} Tr Acc={tr_acc:.4f} | "
                    f"Val Loss={va_loss:.4f} Val Acc={va_acc:.4f} | "
                    f"LR={opt.param_groups[0]['lr']:.2e}",
                    flush=True,
                )
            else:
                print(
                    f"E {ep+1:3d}/{args.epochs} | K_hi={K_high:2d} K_lo={args.k_low} q={q_e:3d} | "
                    f"IO={io_ratio:.3f} | Tr Loss={tr_loss:.4f} Tr Acc={tr_acc:.4f} | "
                    f"Val Loss={va_loss:.4f} Val Acc={va_acc:.4f} | "
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
        print(f"[rank {index}] Unhandled exception in TPU worker", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    xmp.spawn(train_process_entry, args=(args,), nprocs=8, start_method="spawn")
