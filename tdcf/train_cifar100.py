"""
TDCF training script for CIFAR-100.
Handles RGB 3-channel 32×32 images. Key differences from MNIST:
  - 3-channel DCT (per-channel, sensitivity aggregated across channels)
  - On-the-fly augmentation after decode for fair TDCF vs baseline training
  - ResNet-18 and CIFAR-ViT-S backbones
  - 100-epoch training for meaningful learning curves
  - Full-fidelity baseline for comparison
"""
import os, sys, time, json, argparse, logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms as T

from .models import make_model
from .sensitivity import SensitivityEstimator, BlockSensitivityEstimator
from .scheduler import FidelityScheduler, BudgetScheduler
from .dataloader import TDCFServer
from .io_dataloader import BandShardedStore, PhysicalTDCFStore, BlockBandStore
from .storage import (
    build_band_store_from_tensor,
    build_patch_residual_store_from_tensor,
    build_block_band_store_from_tensor,
)
from .transforms import (
    build_nested_masks, idct2d, precompute_dct_dataset,
    block_dct2d, block_idct2d, precompute_block_dct_dataset,
)

# ── config ────────────────────────────────────────────────────────────
H, W, C_IMG = 32, 32, 3
CIFAR_MEAN   = (0.5071, 0.4867, 0.4408)
CIFAR_STD    = (0.2675, 0.2565, 0.2761)
NUM_CLASSES  = 100
# 32×32 / patch_size=8 → 4×4=16 patches  (same P as MNIST for comparison)
PATCH_SIZE   = 8
NUM_BANDS    = 16
_NORM_CACHE  = {}


def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fmt = "[%(asctime)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(os.path.join(save_dir, "train.log"), "w")])
    return logging.getLogger("TDCF-CIFAR100")


def parse_args():
    p = argparse.ArgumentParser("TDCF-CIFAR100")
    p.add_argument("--backbone",     choices=["cnn","vit"], default="cnn")
    p.add_argument("--total_epochs", type=int,   default=100)
    p.add_argument("--pilot_epochs", type=int,   default=10)
    p.add_argument("--pilot_ratio",  type=float, default=0.10)
    p.add_argument("--eta_f",        type=float, default=0.80,
                   help="Frequency retention threshold")
    p.add_argument("--eta_s",        type=float, default=0.70,
                   help="Spatial retention threshold")
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--pilot_lr",     type=float, default=None,
                   help="Optional learning rate used only during the pilot phase.")
    p.add_argument("--wd",           type=float, default=5e-4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--data_workers", type=int,   default=0)
    p.add_argument("--use_band_store", action="store_true",
                   help="Use the on-disk band-sharded store for exact band I/O accounting.")
    p.add_argument("--band_store_mode", choices=["k_only", "kq"], default="k_only",
                   help="`k_only`: exact physical K-band loading with q logged as a proxy. "
                        "`kq`: experimental full physical K+q loading with residual patch shards.")
    p.add_argument("--build_band_store", action="store_true",
                   help="Build the CIFAR-100 band store before training.")
    p.add_argument("--band_store_dir", type=str, default=None,
                   help="Directory for the CIFAR-100 band store.")
    p.add_argument("--data_dir",     type=str,   default="./data")
    p.add_argument("--save_dir",     type=str,   default="./results/cifar100_cnn")
    p.add_argument("--run_baseline", action="store_true")
    p.add_argument("--baseline_only", action="store_true",
                   help="Run ONLY the baseline, skipping TDCF.")
    p.add_argument("--block_dct", action="store_true",
                   help="Use Block-DCT (JPEG-style) for unified K+q I/O savings.")
    p.add_argument("--k_low", type=int, default=1,
                   help="Bands for background patches in Block-DCT mode (default=1=DC only).")
    p.add_argument("--patch_policy", choices=["gradient", "random", "static", "greedy"], default="gradient",
                   help="Policy for picking the q important patches. gradient=TDCF, random=random, static=fixed, greedy=variable-K.")
    p.add_argument("--budget_mode", action="store_true",
                   help="Use budget-constrained scheduler instead of coverage thresholds.")
    p.add_argument("--beta", type=float, default=0.5,
                   help="Starting budget ratio (0 < β < 1). Only used with --budget_mode.")
    p.add_argument("--max_beta", type=float, default=1.0,
                   help="Ending budget ratio. Capping this below 1.0 enforces permanent I/O savings.")
    p.add_argument("--gamma", type=float, default=1.5,
                   help="Ramp exponent. γ=1 linear, γ>1 hold-then-ramp. Only with --budget_mode.")
    return p.parse_args()


# ── augmentation ──────────────────────────────────────────────────────
def get_plain_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def _get_norm_stats(device: torch.device, dtype: torch.dtype):
    key = (device, dtype)
    if key not in _NORM_CACHE:
        mean = torch.tensor(CIFAR_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
        _NORM_CACHE[key] = (mean, std)
    return _NORM_CACHE[key]


def denormalize_batch(x: torch.Tensor) -> torch.Tensor:
    mean, std = _get_norm_stats(x.device, x.dtype)
    return x * std + mean


def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    mean, std = _get_norm_stats(x.device, x.dtype)
    return (x - mean) / std


def augment_cifar_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Apply CIFAR-style training augmentation to a normalized tensor batch.

    Augmentation is applied after decode so the DCT cache remains stable.
    """
    B, _, H_img, W_img = x.shape
    x = denormalize_batch(x).clamp_(0.0, 1.0)

    # Random crop with 4-pixel padding.
    x_pad = F.pad(x, (4, 4, 4, 4))
    crop_grid = (
        x_pad.unfold(2, H_img, 1)
        .unfold(3, W_img, 1)
        .permute(0, 2, 3, 1, 4, 5)
    )
    top = torch.randint(0, 9, (B,), device=x.device)
    left = torch.randint(0, 9, (B,), device=x.device)
    batch_idx = torch.arange(B, device=x.device)
    x = crop_grid[batch_idx, top, left]

    # Random horizontal flip.
    flip_mask = torch.rand(B, device=x.device) < 0.5
    if flip_mask.any():
        x[flip_mask] = torch.flip(x[flip_mask], dims=[3])

    # Lightweight color jitter.
    brightness = 1.0 + torch.empty(B, 1, 1, 1, device=x.device).uniform_(-0.2, 0.2)
    contrast = 1.0 + torch.empty(B, 1, 1, 1, device=x.device).uniform_(-0.2, 0.2)
    saturation = 1.0 + torch.empty(B, 1, 1, 1, device=x.device).uniform_(-0.2, 0.2)

    x = x * brightness
    x_mean = x.mean(dim=(2, 3), keepdim=True)
    x = (x - x_mean) * contrast + x_mean
    x_gray = x.mean(dim=1, keepdim=True)
    x = (x - x_gray) * saturation + x_gray

    return normalize_batch(x.clamp_(0.0, 1.0))


# ── helpers ───────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval(); loss_s = correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_s += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item(); n += y.size(0)
    return loss_s/n, correct/n


def train_epoch(model, loader, opt, crit, device, server=None, augment_fn=None):
    model.train(); loss_s = correct = n = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        if server is not None:
            x = server.serve(x)
        if augment_fn is not None:
            x = augment_fn(x)
        opt.zero_grad(set_to_none=True)
        logits = model(x); loss = crit(logits, y)
        loss.backward(); opt.step()
        loss_s += loss.item()*y.size(0)
        correct += (logits.argmax(1)==y).sum().item(); n += y.size(0)
    return loss_s/n, correct/n


def train_epoch_from_coeff_loader(model, loader, opt, crit, device, augment_fn=None):
    """
    Train from a loader that yields DCT coefficients rather than spatial images.

    This is used by the physical band-store mode, where the loader is
    responsible for reading only the requested frequency shards from disk.
    """
    model.train(); loss_s = correct = n = 0
    for coeffs, y in loader:
        coeffs, y = coeffs.to(device), y.to(device)
        x = idct2d(coeffs)
        if augment_fn is not None:
            x = augment_fn(x)
        opt.zero_grad(set_to_none=True)
        logits = model(x); loss = crit(logits, y)
        loss.backward(); opt.step()
        loss_s += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item(); n += y.size(0)
    return loss_s / n, correct / n


def train_epoch_from_physical_store(model, loader, store, opt, crit, device, augment_fn=None):
    """
    Train from an index loader backed by the full physical K+q store.
    """
    model.train(); loss_s = correct = n = 0
    for indices, y in loader:
        x = store.serve_indices(indices)
        y = y.to(device)
        if augment_fn is not None:
            x = augment_fn(x)
        opt.zero_grad(set_to_none=True)
        logits = model(x); loss = crit(logits, y)
        loss.backward(); opt.step()
        loss_s += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item(); n += y.size(0)
    return loss_s / n, correct / n


@torch.no_grad()
def evaluate_from_coeff_loader(model, loader, criterion, device):
    """Evaluate from a loader that yields DCT coefficients."""
    model.eval(); loss_s = correct = n = 0
    for coeffs, y in loader:
        coeffs, y = coeffs.to(device), y.to(device)
        x = idct2d(coeffs)
        logits = model(x)
        loss_s += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item(); n += y.size(0)
    return loss_s / n, correct / n


def ensure_band_store_from_tensors(train_coeffs, train_labels,
                                   test_coeffs, test_labels,
                                   args, log, residual_k_values=None):
    """
    Build the exact physical TDCF store from the same coefficient tensors
    used by the pilot phase. This keeps the band store bit-exact and builds
    patch residual shards only for the K values that the learned schedule
    can actually request.
    """
    store_dir = args.band_store_dir or os.path.join(args.data_dir, "cifar100_band_store")
    train_root = os.path.join(store_dir, "train")
    test_root = os.path.join(store_dir, "test")
    train_meta = os.path.join(train_root, "metadata.json")
    test_meta = os.path.join(test_root, "metadata.json")
    residual_meta = os.path.join(train_root, "residual_metadata.json")
    residual_k_values = sorted({int(k) for k in (residual_k_values or [])})

    if args.build_band_store or not (os.path.exists(train_meta) and os.path.exists(test_meta)):
        log.info("Building band store from precomputed tensors at %s ...", store_dir)
        build_band_store_from_tensor(train_coeffs, train_labels,
                                     train_root, num_bands=NUM_BANDS)
        build_band_store_from_tensor(test_coeffs, test_labels,
                                     test_root, num_bands=NUM_BANDS)
    else:
        log.info("Using existing band store at %s", store_dir)

    if residual_k_values is not None:
        rebuild_residual = args.build_band_store or not os.path.exists(residual_meta)
        if not rebuild_residual and residual_k_values:
            with open(residual_meta) as f:
                existing_meta = json.load(f)
            existing_k = sorted(int(k) for k in existing_meta.get("k_values", []))
            rebuild_residual = (
                existing_k != residual_k_values or
                existing_meta.get("patch_size") != PATCH_SIZE
            )

        if rebuild_residual:
            log.info("Building residual patch store for K=%s ...", residual_k_values)
            build_patch_residual_store_from_tensor(
                train_coeffs, train_root, residual_k_values,
                patch_size=PATCH_SIZE, num_bands=NUM_BANDS,
            )
        elif residual_k_values:
            log.info("Using existing residual patch store at %s", train_root)

    return train_root, test_root


# ── pilot phase ───────────────────────────────────────────────────────
def run_pilot(model, pilot_coeffs, pilot_labels, masks_gpu,
              estimator, opt, crit, device, pilot_epochs, bs, eta_f, eta_s, log):
    log.info("="*60)
    log.info("PILOT PHASE  (%d epochs, %d samples)", pilot_epochs, pilot_coeffs.shape[0])
    log.info("="*60)
    N = pilot_coeffs.shape[0]
    base_mask = masks_gpu[0].unsqueeze(0).unsqueeze(0)   # (1,1,H,W)

    for ep in range(pilot_epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        for i in range(0, N, bs):
            idx      = perm[i:i+bs]
            c_batch  = pilot_coeffs[idx]         # (b,3,32,32) coeffs
            y_batch  = pilot_labels[idx]
            x_batch  = idct2d(c_batch)            # spatial, full-fidelity

            # ── sensitivity (coefficient-based, no extra DCT) ──
            # Measure sensitivity in eval mode so BatchNorm/dropout are not
            # updated three times per pilot batch.
            model.eval()
            phi_t = estimator.measure_coefficient_sensitivity_from_coeffs(
                c_batch, y_batch, model, crit)

            # ── patch sensitivity using low-freq base ──
            x_base = idct2d(c_batch * base_mask)
            g_t = estimator.measure_patch_sensitivity(
                x_batch, x_base, y_batch, model, crit)
            if not torch.isfinite(phi_t).all() or not torch.isfinite(g_t).all():
                raise RuntimeError(
                    "Non-finite pilot sensitivity detected. "
                    "Pilot measurement became unstable before schedule fitting."
                )

            # ── normal training step ──
            model.train()
            opt.zero_grad(set_to_none=True)
            logits = model(x_batch); loss = crit(logits, y_batch)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite pilot loss detected. "
                    "Try lowering the pilot learning rate or batch size."
                )
            loss.backward(); opt.step()

        estimator.finalize_epoch()
        K = estimator.compute_band_cutoff(ep, eta_f)
        q = estimator.compute_patch_quota(ep, eta_s)
        bs_arr = estimator.band_sensitivity_history[ep]
        if not np.isfinite(bs_arr).all():
            raise RuntimeError(
                "Pilot band sensitivity contains non-finite values after "
                "epoch aggregation."
            )
        log.info("  pilot %2d/%d | K=%2d/%d | q=%2d/%d | "
                 "low-band=%.4f  high-band=%.4f",
                 ep+1, pilot_epochs, K, NUM_BANDS,
                 q, estimator.P, bs_arr[0], bs_arr[-1])


# ── TDCF adaptive training ────────────────────────────────────────────
def run_tdcf(args, log):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    log.info("Device: %s  |  GPU: %s", device,
             torch.cuda.get_device_name(0) if device.type=="cuda" else "N/A")
    t0 = time.time()

    # ── data (plain normalized tensors for stable DCT; augment after decode) ──
    plain_tfm = get_plain_transform()
    train_plain = datasets.CIFAR100(args.data_dir, True,  download=True, transform=plain_tfm)
    test_ds     = datasets.CIFAR100(args.data_dir, False, download=True, transform=plain_tfm)

    log.info("Pre-computing training DCT coefficients on GPU ...")
    # Precompute on plain (no aug) for stable frequency structure
    train_coeffs, train_labels = precompute_dct_dataset(
        train_plain, device, num_workers=args.data_workers)
    log.info("Pre-computing test DCT coefficients on GPU ...")
    test_coeffs, test_labels = precompute_dct_dataset(
        test_ds, device, num_workers=args.data_workers)

    if args.use_band_store:
        if args.band_store_mode == "kq":
            log.info("Physical store mode enabled: exact K+q I/O will be measured "
                     "from the on-disk band and residual-patch shards.")
        else:
            log.info("Band-store mode enabled: exact physical K-band I/O will be measured. "
                     "q(e) remains a logged schedule proxy in this mode.")
        log.info("  train: %s  test: %s", train_coeffs.shape, test_coeffs.shape)
        test_loader = None
    else:
        test_images = idct2d(test_coeffs)
        vram_mb = (train_coeffs.nelement() + test_coeffs.nelement()) * 4 / 1e6
        log.info("  train: %s  test: %s  (%.0f MB GPU)", train_coeffs.shape,
                 test_coeffs.shape, vram_mb)

        test_loader = DataLoader(
            TensorDataset(test_images, test_labels),
            batch_size=512, shuffle=False)

    # ── model ──
    model = make_model(args.backbone, "cifar100", device)
    nparams = sum(p.numel() for p in model.parameters())
    log.info("Backbone: %s  |  params: %.2fM", args.backbone, nparams/1e6)

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    pilot_lr = args.pilot_lr if args.pilot_lr is not None else args.lr
    pilot_opt = optim.SGD(model.parameters(), lr=pilot_lr, momentum=0.9,
                          weight_decay=args.wd, nesterov=True)
    log.info("Pilot LR: %.3e  |  Main LR: %.3e", pilot_lr, args.lr)

    # ── masks & estimator ──
    masks     = build_nested_masks(H, W, NUM_BANDS)
    masks_gpu = [m.to(device) for m in masks]
    estimator = SensitivityEstimator(H, W, NUM_BANDS, PATCH_SIZE, device=device)

    # ── pilot ──
    pilot_n   = max(int(len(train_plain)*args.pilot_ratio), args.batch_size)
    pilot_idx = np.random.choice(len(train_plain), pilot_n, replace=False)
    run_pilot(model,
              train_coeffs[pilot_idx], train_labels[pilot_idx],
              masks_gpu, estimator, pilot_opt, crit, device,
              args.pilot_epochs, args.batch_size, args.eta_f, args.eta_s, log)

    # ── fit schedule ──
    if args.budget_mode:
        fsched = BudgetScheduler(NUM_BANDS, estimator.P, beta=args.beta, max_beta=args.max_beta, gamma=args.gamma, k_low=args.k_low)
    else:
        fsched = FidelityScheduler(NUM_BANDS, estimator.P, args.eta_f, args.eta_s)
    fsched.fit_from_pilot(estimator, args.total_epochs)
    log.info("\n%s\n", fsched.summary())
    residual_k_values = None
    if args.use_band_store and args.band_store_mode == "kq":
        residual_k_values = sorted({
            fsched.get_fidelity(ep)[0]
            for ep in range(args.total_epochs)
            if fsched.get_fidelity(ep)[1] < estimator.P
        })

    # Fresh optimizer for the full training run so pilot momentum/state does
    # not bleed into the main experiment.
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                    weight_decay=args.wd, nesterov=True)
    sched_lr = optim.lr_scheduler.CosineAnnealingLR(opt, args.total_epochs)

    # ── adaptive training ──
    proxy_server = TDCFServer(H, W, NUM_BANDS, PATCH_SIZE, device)
    if args.use_band_store:
        train_root, test_root = ensure_band_store_from_tensors(
            train_coeffs, train_labels, test_coeffs, test_labels,
            args, log, residual_k_values=residual_k_values)
        if args.band_store_mode == "kq":
            train_store = PhysicalTDCFStore(train_root, device)
            train_loader = train_store.get_index_loader(
                batch_size=args.batch_size, shuffle=True, num_workers=0)
        else:
            train_store = BandShardedStore(train_root, device)
            train_loader = train_store.get_loader(
                batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_store = BandShardedStore(test_root, device)
        test_store.set_fidelity(NUM_BANDS)
        test_loader = test_store.get_loader(
            batch_size=512, shuffle=False, num_workers=0)
        server = None
    else:
        server = TDCFServer(H, W, NUM_BANDS, PATCH_SIZE, device)
        train_tds = TensorDataset(train_coeffs, train_labels)
        train_loader = DataLoader(train_tds, batch_size=args.batch_size, shuffle=True)

    log.info("="*60)
    log.info("ADAPTIVE TRAINING  (%d epochs)", args.total_epochs)
    log.info("="*60)

    train_augment = augment_cifar_batch
    hist_keys = ["epoch", "K", "q", "approx_ratio",
                 "train_loss", "train_acc", "test_loss", "test_acc",
                 "wall_s", "lr"]
    if args.use_band_store:
        hist_keys.extend(["true_band_ratio", "true_band_bytes"])
        if args.band_store_mode == "kq":
            hist_keys.extend(["true_io_ratio", "true_io_bytes"])
    hist = {k: [] for k in hist_keys}

    for ep in range(args.total_epochs):
        if args.budget_mode:
            if not args.block_dct:
                raise NotImplementedError("--budget_mode currently requires --block_dct")
            budget = fsched.get_budget(ep)
            K_e, q_e = NUM_BANDS, estimator.P
        else:
            K_e, q_e = fsched.get_fidelity(ep)
            train_server.set_fidelity(K_e, args.k_low, q_e, args.patch_policy)
        ps_idx   = min(ep, len(estimator.patch_sensitivity_history)-1)
        ps       = (estimator.patch_sensitivity_history[ps_idx]
                    if estimator.patch_sensitivity_history else None)
        proxy_server.set_fidelity(K_e, q_e, ps)
        br = proxy_server.get_bytes_ratio()

        if args.use_band_store:
            train_store.reset_epoch_io()
            if args.band_store_mode == "kq":
                train_store.set_fidelity(K_e, q_e, ps)
                tr_l, tr_a = train_epoch_from_physical_store(
                    model, train_loader, train_store, opt, crit, device,
                    augment_fn=train_augment)
            else:
                train_store.set_fidelity(K_e)
                tr_l, tr_a = train_epoch_from_coeff_loader(
                    model, train_loader, opt, crit, device, augment_fn=train_augment)
            test_store.set_fidelity(NUM_BANDS)
            te_l, te_a = evaluate_from_coeff_loader(
                model, test_loader, crit, device)
            if args.band_store_mode == "kq":
                true_band_ratio = train_store.get_epoch_band_ratio()
                true_band_bytes = train_store.band_bytes_read_epoch
                true_io_ratio = train_store.get_epoch_total_ratio()
                true_io_bytes = train_store.bytes_read_epoch
            else:
                true_band_ratio = train_store.get_io_ratio(K_e)
                true_band_bytes = train_store.bytes_read_epoch
        else:
            server.set_fidelity(K_e, q_e, ps)
            tr_l, tr_a = train_epoch(
                model, train_loader, opt, crit, device,
                server=server, augment_fn=train_augment)
            te_l, te_a = evaluate(model, test_loader, crit, device)
        sched_lr.step()
        lr_now  = opt.param_groups[0]["lr"]
        elapsed = time.time() - t0

        for k,v in [("epoch",ep),("K",K_e),("q",q_e),("approx_ratio",br),
                    ("train_loss",tr_l),("train_acc",tr_a),
                    ("test_loss",te_l),("test_acc",te_a),
                    ("wall_s",elapsed),("lr",lr_now)]:
            hist[k].append(v)
        if args.use_band_store:
            hist["true_band_ratio"].append(true_band_ratio)
            hist["true_band_bytes"].append(true_band_bytes)
            if args.band_store_mode == "kq":
                hist["true_io_ratio"].append(true_io_ratio)
                hist["true_io_bytes"].append(true_io_bytes)

        if args.use_band_store:
            if args.band_store_mode == "kq":
                log.info("  E %3d/%d | K=%2d q=%2d | approx=%.2f | "
                         "band=%.3f | io=%.3f | band_mb=%.1f | io_mb=%.1f | "
                         "tr_loss=%.4f tr_acc=%.4f | te_loss=%.4f te_acc=%.4f | "
                         "lr=%.2e | %.0fs",
                         ep+1, args.total_epochs, K_e, q_e, br,
                         true_band_ratio, true_io_ratio,
                         true_band_bytes / 1e6, true_io_bytes / 1e6,
                         tr_l, tr_a, te_l, te_a, lr_now, elapsed)
            else:
                log.info("  E %3d/%d | K=%2d q=%2d | approx=%.2f | "
                         "band=%.3f | band_mb=%.1f | "
                         "tr_loss=%.4f tr_acc=%.4f | te_loss=%.4f te_acc=%.4f | "
                         "lr=%.2e | %.0fs",
                         ep+1, args.total_epochs, K_e, q_e, br,
                         true_band_ratio, true_band_bytes / 1e6,
                         tr_l, tr_a, te_l, te_a, lr_now, elapsed)
        else:
            log.info("  E %3d/%d | K=%2d q=%2d | approx=%.2f | "
                     "tr_loss=%.4f tr_acc=%.4f | te_loss=%.4f te_acc=%.4f | "
                     "lr=%.2e | %.0fs",
                     ep+1, args.total_epochs, K_e, q_e, br,
                     tr_l, tr_a, te_l, te_a, lr_now, elapsed)

    return model, hist, estimator, fsched


# ── full-fidelity baseline ─────────────────────────────────────────────
def run_baseline(args, log):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model    = make_model(args.backbone, "cifar100", device)
    opt      = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                         weight_decay=args.wd, nesterov=True)
    sched_lr = optim.lr_scheduler.CosineAnnealingLR(opt, args.total_epochs)
    crit     = nn.CrossEntropyLoss(label_smoothing=0.1)

    if args.use_band_store:
        store_dir = args.band_store_dir or os.path.join(args.data_dir, "cifar100_band_store")
        train_root = os.path.join(store_dir, "train")
        test_root = os.path.join(store_dir, "test")
        train_store = BandShardedStore(train_root, device)
        test_store = BandShardedStore(test_root, device)
        train_store.set_fidelity(NUM_BANDS)
        test_store.set_fidelity(NUM_BANDS)
        tr_loader = train_store.get_loader(
            batch_size=args.batch_size, shuffle=True, num_workers=0)
        te_loader = test_store.get_loader(
            batch_size=512, shuffle=False, num_workers=0)
    else:
        plain_tfm = get_plain_transform()
        train_ds = datasets.CIFAR100(args.data_dir, True,  download=True, transform=plain_tfm)
        test_ds  = datasets.CIFAR100(args.data_dir, False, download=True, transform=plain_tfm)
        pin = device.type == "cuda"
        tr_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                               num_workers=args.data_workers, pin_memory=pin)
        te_loader = DataLoader(test_ds,  512, num_workers=args.data_workers,
                               pin_memory=pin)

    log.info("="*60); log.info("BASELINE (full fidelity)"); log.info("="*60)
    hist_keys = ["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "wall_s"]
    if args.use_band_store:
        hist_keys.extend(["true_band_ratio", "true_band_bytes"])
    hist = {k: [] for k in hist_keys}
    t0 = time.time()
    for ep in range(args.total_epochs):
        if args.use_band_store:
            train_store.reset_epoch_io()
            tr_l, tr_a = train_epoch_from_coeff_loader(
                model, tr_loader, opt, crit, device, augment_fn=augment_cifar_batch)
            te_l, te_a = evaluate_from_coeff_loader(model, te_loader, crit, device)
            true_band_ratio = train_store.get_io_ratio(NUM_BANDS)
            true_band_bytes = train_store.bytes_read_epoch
        else:
            tr_l, tr_a = train_epoch(
                model, tr_loader, opt, crit, device, augment_fn=augment_cifar_batch)
            te_l, te_a = evaluate(model, te_loader, crit, device)
        sched_lr.step(); elapsed = time.time()-t0
        for k,v in [("epoch",ep),("train_loss",tr_l),("train_acc",tr_a),
                    ("test_loss",te_l),("test_acc",te_a),("wall_s",elapsed)]:
            hist[k].append(v)
        if args.use_band_store:
            hist["true_band_ratio"].append(true_band_ratio)
            hist["true_band_bytes"].append(true_band_bytes)
            log.info("  E %3d/%d | band=%.3f | band_mb=%.1f | "
                     "tr_acc=%.4f te_acc=%.4f | lr=%.2e | %.0fs",
                     ep+1, args.total_epochs, true_band_ratio,
                     true_band_bytes / 1e6, tr_a, te_a,
                     opt.param_groups[0]["lr"], elapsed)
        else:
            log.info("  E %3d/%d | tr_acc=%.4f te_acc=%.4f | lr=%.2e | %.0fs",
                     ep+1, args.total_epochs, tr_a, te_a,
                     opt.param_groups[0]["lr"], elapsed)
    return model, hist


# ── visualization ─────────────────────────────────────────────────────
def save_plots(ht, hb, est, fs, save_dir, args):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    os.makedirs(save_dir, exist_ok=True)
    c_tdcf, c_base = "#2196F3", "#FF5722"

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f"TDCF on CIFAR-100 — {args.backbone.upper()} backbone",
                 fontsize=18, fontweight="bold", y=0.98)
    gs = GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.32)

    # 1 — test accuracy
    ax = fig.add_subplot(gs[0,0])
    ax.plot(ht["epoch"], ht["test_acc"], "o-", c=c_tdcf, lw=2, ms=3, label="TDCF")
    if hb: ax.plot(hb["epoch"], hb["test_acc"], "s--", c=c_base, lw=2, ms=3, label="Baseline")
    ax.set(xlabel="Epoch", ylabel="Test Accuracy", title="Test Accuracy")
    ax.legend(); ax.grid(True, alpha=.3)

    # 2 — train loss
    ax = fig.add_subplot(gs[0,1])
    ax.plot(ht["epoch"], ht["train_loss"], "o-", c=c_tdcf, lw=2, ms=3, label="TDCF")
    if hb: ax.plot(hb["epoch"], hb["train_loss"], "s--", c=c_base, lw=2, ms=3, label="Baseline")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training Loss")
    ax.legend(); ax.grid(True, alpha=.3)

    # 3 — monotone schedule
    ax = fig.add_subplot(gs[0,2])
    Ks, qs = fs.get_full_schedule()
    ax.plot(Ks, "o-", c="#4CAF50", lw=2, ms=3, label="K(e) bands")
    ax.plot(qs, "s-", c="#9C27B0", lw=2, ms=3, label="q(e) patches")
    ax.set(xlabel="Epoch", ylabel="Count", title="Monotone Fidelity Schedule K(e) & q(e)")
    ax.legend(); ax.grid(True, alpha=.3)

    # 4 — approx ratio
    ax = fig.add_subplot(gs[1,0])
    if "true_band_ratio" in ht:
        ax.fill_between(ht["epoch"], ht["true_band_ratio"],
                        alpha=.35, color="#FF9800")
        ax.plot(ht["epoch"], ht["true_band_ratio"], "o-", c="#FF9800",
                lw=2, ms=3, label="True band I/O")
        ax.plot(ht["epoch"], ht["approx_ratio"], "s--", c="#8D6E63",
                lw=1.5, ms=3, label="Proxy schedule ratio")
        ratio_title = "True Band I/O / Full"
    else:
        ax.fill_between(ht["epoch"], ht["approx_ratio"], alpha=.35, color="#FF9800")
        ax.plot(ht["epoch"], ht["approx_ratio"], "o-", c="#FF9800", lw=2, ms=3)
        ratio_title = "Approx. Data Served / Full"
    ax.axhline(1, ls="--", c="gray", label="Full fidelity"); ax.set_ylim(0, 1.1)
    ax.set(xlabel="Epoch", ylabel="Fraction", title=ratio_title)
    ax.legend(); ax.grid(True, alpha=.3)

    # 5 — band sensitivity heatmap
    ax = fig.add_subplot(gs[1,1])
    if est.band_sensitivity_history:
        sm = np.stack(est.band_sensitivity_history)
        im = ax.imshow(sm.T, aspect="auto", cmap="viridis", origin="lower")
        ax.set(xlabel="Pilot Epoch", ylabel="Band (low→high freq)",
               title="Band Sensitivity $s_e(b)$")
        plt.colorbar(im, ax=ax)

    # 6 — patch sensitivity map
    ax = fig.add_subplot(gs[1,2])
    if est.patch_sensitivity_history:
        pm = est.patch_sensitivity_history[-1].reshape(est.nph, est.npw)
        im = ax.imshow(pm, cmap="hot", interpolation="nearest")
        ax.set_title("Spatial Patch Sensitivity (last pilot)\n"
                     "(shows which 8×8 regions matter most)")
        plt.colorbar(im, ax=ax)

    # 7 — coefficient sensitivity sorted spectra
    ax = fig.add_subplot(gs[2,0])
    if est.coeff_sensitivity_history:
        for i, ch in enumerate(est.coeff_sensitivity_history[::2]):  # every 2nd
            ax.plot(np.sort(ch)[::-1], alpha=.7, lw=1.2, label=f"pilot {i*2}")
        ax.set(xlabel="Coefficient rank", ylabel="|∂L/∂c_k|",
               title="Sensitivity Spectrum (sorted)")
        ax.set_yscale("log"); ax.legend(fontsize=7); ax.grid(True, alpha=.3)

    # 8 — wall-clock
    ax = fig.add_subplot(gs[2,1])
    ax.plot(ht["epoch"], ht["wall_s"], "o-", c=c_tdcf, lw=2, ms=3, label="TDCF")
    if hb: ax.plot(hb["epoch"], hb["wall_s"], "s--", c=c_base, lw=2, ms=3, label="Baseline")
    ax.set(xlabel="Epoch", ylabel="Seconds", title="Wall-clock Time"); ax.legend(); ax.grid(True, alpha=.3)

    # 9 — LR
    ax = fig.add_subplot(gs[2,2])
    ax.plot(ht["epoch"], ht["lr"], "o-", c="#00BCD4", lw=2, ms=3)
    ax.set(xlabel="Epoch", ylabel="LR", title="Learning Rate"); ax.grid(True, alpha=.3)

    fig.savefig(os.path.join(save_dir, "tdcf_cifar100_results.png"), dpi=200, bbox_inches="tight")
    plt.close()


# ── Block-DCT training ────────────────────────────────────────────────
def run_tdcf_block(args, log):
    """TDCF training using Block-DCT for unified K+q I/O savings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    log.info("Device: %s  |  GPU: %s", device,
             torch.cuda.get_device_name(0) if device.type=="cuda" else "N/A")
    t0 = time.time()

    # ── data ──
    plain_tfm = get_plain_transform()
    train_plain = datasets.CIFAR100(args.data_dir, True,  download=True, transform=plain_tfm)
    test_ds     = datasets.CIFAR100(args.data_dir, False, download=True, transform=plain_tfm)

    log.info("Pre-computing block-DCT coefficients (block=%d) ...", PATCH_SIZE)
    train_coeffs, train_labels = precompute_block_dct_dataset(
        train_plain, device, block_size=PATCH_SIZE, num_workers=args.data_workers)
    test_coeffs, test_labels = precompute_block_dct_dataset(
        test_ds, device, block_size=PATCH_SIZE, num_workers=args.data_workers)
    nph, npw = H // PATCH_SIZE, W // PATCH_SIZE
    log.info("  train: %s  test: %s", train_coeffs.shape, test_coeffs.shape)

    # ── model ──
    model = make_model(args.backbone, "cifar100", device)
    nparams = sum(p.numel() for p in model.parameters())
    log.info("Backbone: %s  |  params: %.2fM", args.backbone, nparams/1e6)

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    pilot_lr = args.pilot_lr if args.pilot_lr is not None else args.lr
    pilot_opt = optim.SGD(model.parameters(), lr=pilot_lr, momentum=0.9,
                          weight_decay=args.wd, nesterov=True)

    # ── pilot ──
    estimator = BlockSensitivityEstimator(
        PATCH_SIZE, nph, npw, NUM_BANDS, device=device)
    pilot_n = max(int(len(train_plain) * args.pilot_ratio), args.batch_size)
    pilot_idx = np.random.choice(len(train_plain), pilot_n, replace=False)
    pilot_coeffs = train_coeffs[pilot_idx]
    pilot_labels = train_labels[pilot_idx]

    log.info("=" * 60)
    log.info("PILOT PHASE  (%d epochs, %d samples, Block-DCT)", args.pilot_epochs, pilot_n)
    log.info("=" * 60)

    for ep in range(args.pilot_epochs):
        model.train()
        perm = torch.randperm(pilot_n, device=device)
        for i in range(0, pilot_n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            c_batch = pilot_coeffs[idx]
            y_batch = pilot_labels[idx]

            # Sensitivity measurement should not update BatchNorm/dropout state.
            model.eval()
            estimator.measure_sensitivity(c_batch, y_batch, model, crit)

            model.train()
            x_batch = block_idct2d(c_batch, nph, npw)
            x_batch = augment_cifar_batch(x_batch)
            pilot_opt.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = crit(logits, y_batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite pilot loss.")
            loss.backward()
            pilot_opt.step()

        estimator.finalize_epoch()
        K = estimator.compute_band_cutoff(ep, args.eta_f)
        q = estimator.compute_patch_quota(ep, args.eta_s)
        bs = estimator.band_sensitivity_history[ep]
        log.info("  pilot %2d/%d | K_high=%2d/%d | q=%2d/%d | "
                 "band_low=%.4f band_high=%.4f",
                 ep+1, args.pilot_epochs, K, NUM_BANDS, q, estimator.P,
                 bs[0], bs[-1])

    # ── fit schedule ──
    if args.budget_mode:
        fsched = BudgetScheduler(NUM_BANDS, estimator.P, beta=args.beta, max_beta=args.max_beta, gamma=args.gamma, k_low=args.k_low)
    else:
        fsched = FidelityScheduler(NUM_BANDS, estimator.P, args.eta_f, args.eta_s)
    fsched.fit_from_pilot(estimator, args.total_epochs)
    log.info("\n%s\n", fsched.summary())

    # ── build block band store ──
    store_dir = args.band_store_dir or os.path.join(args.data_dir, "cifar100_block_store")
    train_root = os.path.join(store_dir, "train")
    test_root = os.path.join(store_dir, "test")
    train_meta = os.path.join(train_root, "metadata.json")
    test_meta = os.path.join(test_root, "metadata.json")
    if args.build_band_store or not (os.path.exists(train_meta) and os.path.exists(test_meta)):
        log.info("Building block-DCT band store at %s ...", store_dir)
        build_block_band_store_from_tensor(
            train_coeffs, train_labels, train_root, nph, npw, NUM_BANDS)
        build_block_band_store_from_tensor(
            test_coeffs, test_labels, test_root, nph, npw, NUM_BANDS)
    else:
        log.info("Using existing block-DCT band store at %s", store_dir)

    train_store = BlockBandStore(train_root, device)
    test_store = BlockBandStore(test_root, device)
    train_loader = train_store.get_loader(
        batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_store.set_fidelity(K_high=NUM_BANDS, K_low=NUM_BANDS, q=estimator.P)
    test_loader = test_store.get_loader(
        batch_size=512, shuffle=False, num_workers=0)

    # Fresh optimizer
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                    weight_decay=args.wd, nesterov=True)
    sched_lr = optim.lr_scheduler.CosineAnnealingLR(opt, args.total_epochs)

    # ── adaptive training ──
    log.info("=" * 60)
    log.info("ADAPTIVE TRAINING  (%d epochs, Block-DCT)", args.total_epochs)
    log.info("=" * 60)

    K_low = args.k_low
    hist_keys = ["epoch", "K_high", "K_low", "q", "io_ratio",
                 "train_loss", "train_acc", "test_loss", "test_acc",
                 "wall_s", "lr", "io_bytes"]
    hist = {k: [] for k in hist_keys}

    for ep in range(args.total_epochs):
        ps_idx = min(ep, len(estimator.patch_sensitivity_history) - 1)
        ps = estimator.patch_sensitivity_history[ps_idx]
        bs_idx = min(ep, len(estimator.band_sensitivity_history) - 1)
        bs = estimator.band_sensitivity_history[bs_idx]

        if args.budget_mode:
            budget = fsched.get_budget(ep)
            train_store.set_budget(
                budget, patch_sensitivity=ps,
                band_sensitivity=bs, k_low=args.k_low,
            )
            K_high, q_e = NUM_BANDS, estimator.P  # For logging
        else:
            K_high, q_e = fsched.get_fidelity(ep)
            train_store.set_fidelity(K_high, K_low, q_e, patch_sensitivity=ps,
                                     band_sensitivity=bs, patch_policy=args.patch_policy)
        io_ratio = train_store.get_io_ratio()
        train_store.reset_epoch_io()
        test_store.reset_epoch_io()

        # Train
        model.train()
        loss_s = correct = n = 0
        for indices, y in train_loader:
            x = train_store.serve_indices(indices)
            y = y.to(device)
            x = augment_cifar_batch(x)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_s += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
        tr_l, tr_a = loss_s / n, correct / n

        # Evaluate at full fidelity for a standard accuracy comparison.
        test_store.set_fidelity(K_high=NUM_BANDS, K_low=NUM_BANDS, q=estimator.P)
        model.eval()
        loss_s = correct = n = 0
        with torch.no_grad():
            for indices, y in test_loader:
                x = test_store.serve_indices(indices)
                y = y.to(device)
                logits = model(x)
                loss_s += crit(logits, y).item() * y.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                n += y.size(0)
        te_l, te_a = loss_s / n, correct / n

        sched_lr.step()
        lr_now = opt.param_groups[0]["lr"]
        elapsed = time.time() - t0
        io_bytes = train_store.bytes_read_epoch

        for k, v in zip(hist_keys,
                        [ep+1, int(K_high), int(K_low), int(q_e), float(io_ratio),
                         tr_l, tr_a, te_l, te_a, elapsed, lr_now, io_bytes]):
            hist[k].append(v)

        log.info("  E %3d/%d | K_hi=%2d K_lo=%d q=%2d | io=%.3f | "
                 "io_mb=%.1f | tr_loss=%.4f tr_acc=%.4f | "
                 "te_loss=%.4f te_acc=%.4f | lr=%.2e | %.0fs",
                 ep+1, args.total_epochs, K_high, K_low, q_e,
                 io_ratio, io_bytes / 1e6,
                 tr_l, tr_a, te_l, te_a, lr_now, elapsed)

    return model, hist, estimator, fsched


# ── main ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    log  = setup_logger(args.save_dir)
    log.info("="*60)
    log.info("  TDCF — CIFAR-100  |  %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("="*60)
    log.info("Config: %s", json.dumps(vars(args), indent=2))

    if not args.baseline_only:
        if args.block_dct:
            model_t, ht, est, fs = run_tdcf_block(args, log)
        else:
            model_t, ht, est, fs = run_tdcf(args, log)
    else:
        ht, est, fs = {}, None, None

    hb = None
    if args.run_baseline or args.baseline_only:
        _, hb = run_baseline(args, log)

    # ── save ──
    results = {
        "config": vars(args),
    }
    if not args.baseline_only:
        results.update({
            "tdcf": ht,
            "band_sensitivity":  [s.tolist() for s in est.band_sensitivity_history] if est else [],
            "patch_sensitivity": [s.tolist() for s in est.patch_sensitivity_history] if est else [],
            "schedule_K": fs.get_full_schedule()[0].tolist() if fs else [],
            "schedule_q": fs.get_full_schedule()[1].tolist() if fs else [],
        })
        torch.save(model_t.state_dict(), os.path.join(args.save_dir, "model_tdcf.pt"))
    
    if hb is not None:
        results["baseline"] = hb

    class _NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, cls=_NpEncoder)

    # ── final summary ──
    log.info("\n" + "="*60)
    log.info("FINAL RESULTS")
    log.info("="*60)
    if not args.baseline_only:
        log.info("  TDCF  best test acc:  %.4f", max(ht["test_acc"]))
        log.info("  TDCF  final test acc: %.4f", ht["test_acc"][-1])
    if hb:
        log.info("  Base  best test acc:  %.4f", max(hb["test_acc"]))
        log.info("  Base  final test acc: %.4f", hb["test_acc"][-1])
        if not args.baseline_only:
            log.info("  Accuracy drop:        %+.4f", ht["test_acc"][-1] - hb["test_acc"][-1])
    if not args.baseline_only:
        avg_io = np.mean(ht.get("io_ratio", ht.get("approx_ratio", [1.0])))
    log.info("  Avg I/O ratio:        %.3f  (%.1f%% I/O saved)",
             avg_io, (1 - avg_io) * 100)
    if "io_bytes" in ht:
        log.info("  Avg I/O bytes/epoch:  %.1f MB", np.mean(ht["io_bytes"]) / 1e6)
    log.info("  TDCF wall time:       %.0fs", ht["wall_s"][-1])
    if hb:
        log.info("  Base wall time:       %.0fs", hb["wall_s"][-1])
    log.info("  Schedule:\n%s", fs.summary())
    log.info("  Results → %s/", args.save_dir)


if __name__ == "__main__":
    main()
