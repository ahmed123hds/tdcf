"""
Matched-budget baselines for TDCF ablation (Section 6 of the paper).

Each baseline uses the EXACT SAME data budget as the TDCF run, but differs
in HOW it decides what to drop. This isolates the contribution of
gradient-informed scheduling.

Baselines:
  1. static_lowpass   — Fixed K for all epochs (matched avg I/O ratio)
  2. static_kq        — Fixed (K, q) for all epochs (matched avg ratio)
  3. random_coeff     — Same K schedule, but randomly chosen coefficients
  4. random_patch     — Same (K, q) schedule, but randomly chosen patches

All baselines use the same model architecture, optimizer, LR schedule,
and augmentation as the TDCF run for a fair comparison.
"""

import os, sys, time, json, argparse, logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms as T

from .models import make_model
from .transforms import (
    build_nested_masks, build_frequency_bands,
    idct2d, precompute_dct_dataset,
)
from .train_cifar100 import (
    CIFAR_MEAN, CIFAR_STD, NUM_BANDS, PATCH_SIZE,
    get_plain_transform, augment_cifar_batch, evaluate,
)

H, W = 32, 32
NUM_PATCHES = (H // PATCH_SIZE) * (W // PATCH_SIZE)


def setup_logger(save_dir, name="BASELINES"):
    os.makedirs(save_dir, exist_ok=True)
    fmt = "[%(asctime)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt, force=True,
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(os.path.join(save_dir, "baselines.log"), "w")])
    return logging.getLogger(name)


def parse_args():
    p = argparse.ArgumentParser("TDCF-Baselines")
    p.add_argument("--tdcf_results", type=str, required=True,
                   help="Path to TDCF results.json to match budget from")
    p.add_argument("--backbone",     choices=["cnn", "vit"], default="cnn")
    p.add_argument("--dataset",      choices=["cifar100"], default="cifar100")
    p.add_argument("--total_epochs", type=int, default=100)
    p.add_argument("--batch_size",   type=int, default=256)
    p.add_argument("--lr",           type=float, default=0.1)
    p.add_argument("--wd",           type=float, default=5e-4)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--data_workers", type=int, default=0)
    p.add_argument("--data_dir",     type=str, default="./data")
    p.add_argument("--save_dir",     type=str, default="./results/baselines")
    # Which baselines to run
    p.add_argument("--run_static_lowpass", action="store_true")
    p.add_argument("--run_static_kq",     action="store_true")
    p.add_argument("--run_random_coeff",  action="store_true")
    p.add_argument("--run_random_patch",  action="store_true")
    p.add_argument("--run_all",           action="store_true")
    return p.parse_args()


def k_only_ratio(K: int, num_bands: int = NUM_BANDS) -> float:
    """I/O ratio for a pure low-pass baseline keeping only the first K bands."""
    return K / num_bands


def kq_ratio(K: int, q: int, num_bands: int = NUM_BANDS,
             num_patches: int = NUM_PATCHES) -> float:
    """Schedule-level ratio used by the current TDCF prototype."""
    freq_ratio = K / num_bands
    spatial_ratio = q / num_patches
    return freq_ratio + (1.0 - freq_ratio) * spatial_ratio


def match_static_lowpass(target_ratio: float) -> tuple[int, float]:
    """Choose the low-pass K whose ratio is closest to the target budget."""
    candidates = [
        (K, k_only_ratio(K), abs(k_only_ratio(K) - target_ratio))
        for K in range(1, NUM_BANDS + 1)
    ]
    K_best, ratio_best, _ = min(candidates, key=lambda item: item[2])
    return K_best, ratio_best


def match_static_kq(target_ratio: float) -> tuple[int, int, float]:
    """Choose the fixed (K, q) pair closest to the target schedule ratio."""
    candidates = [
        (K, q, kq_ratio(K, q), abs(kq_ratio(K, q) - target_ratio))
        for K in range(1, NUM_BANDS + 1)
        for q in range(1, NUM_PATCHES + 1)
    ]
    K_best, q_best, ratio_best, _ = min(candidates, key=lambda item: item[3])
    return K_best, q_best, ratio_best


# ---------------------------------------------------------------------------
# Serving functions for each baseline
# ---------------------------------------------------------------------------

class StaticLowPassServer:
    """
    Baseline 1: Fixed K for all epochs, no patch selection.
    Zeroes out all bands >= K using the same nested masks as TDCF.
    """

    def __init__(self, K: int, H: int, W: int, num_bands: int,
                 device: torch.device):
        masks = build_nested_masks(H, W, num_bands)
        self.mask = masks[K - 1].to(device).unsqueeze(0).unsqueeze(0)
        self.K = K
        self.num_bands = num_bands

    @torch.no_grad()
    def serve(self, coeffs: torch.Tensor) -> torch.Tensor:
        return idct2d(coeffs * self.mask)

    def get_io_ratio(self):
        return self.K / self.num_bands


class StaticKQServer:
    """
    Baseline 2: Fixed (K, q) for all epochs.
    Same band truncation + spatial patch dropping as TDCF, but static.
    Patches are selected by energy (no gradient info).
    """

    def __init__(self, K: int, q: int, H: int, W: int, num_bands: int,
                 patch_size: int, device: torch.device):
        masks = build_nested_masks(H, W, num_bands)
        self.mask = masks[K - 1].to(device).unsqueeze(0).unsqueeze(0)
        self.K = K
        self.q = q
        self.num_bands = num_bands
        self.patch_size = patch_size
        self.nph = H // patch_size
        self.npw = W // patch_size
        self.P = self.nph * self.npw
        self.device = device

    @torch.no_grad()
    def serve(self, coeffs: torch.Tensor) -> torch.Tensor:
        B, C_ch = coeffs.shape[:2]
        ps = self.patch_size
        c_base = coeffs * self.mask
        x_base = idct2d(c_base)

        if self.q >= self.P:
            return idct2d(coeffs)

        x_full = idct2d(coeffs)
        residual = x_full - x_base

        # Extract patches
        r_patches = (
            residual[:, :, :self.nph * ps, :self.npw * ps]
            .reshape(B, C_ch, self.nph, ps, self.npw, ps)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(B, self.P, C_ch, ps, ps)
        )

        # Select top-q by energy (no gradient info)
        energy = r_patches.reshape(B, self.P, -1).pow(2).sum(dim=-1)
        avg_energy = energy.mean(dim=0)
        _, top_idx = avg_energy.topk(self.q)
        keep_mask = torch.zeros(self.P, device=self.device, dtype=torch.bool)
        keep_mask[top_idx] = True
        r_patches[:, ~keep_mask] = 0.0

        # Reconstruct
        r_spatial = (
            r_patches
            .reshape(B, self.nph, self.npw, C_ch, ps, ps)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(B, C_ch, self.nph * ps, self.npw * ps)
        )
        out = x_base.clone()
        out[:, :, :self.nph * ps, :self.npw * ps] += r_spatial
        return out

    def get_io_ratio(self):
        freq_ratio = self.K / self.num_bands
        spatial_ratio = self.q / self.P
        return freq_ratio + (1.0 - freq_ratio) * spatial_ratio


class RandomCoeffServer:
    """
    Baseline 3: Same number of coefficients as TDCF's K schedule,
    but chosen RANDOMLY instead of by zig-zag frequency order.
    Proves that frequency-ordered selection matters.
    """

    def __init__(self, K_schedule, H: int, W: int, num_bands: int,
                 device: torch.device, seed: int = 0):
        bands = build_frequency_bands(H, W, num_bands)
        total = H * W
        self.num_bands = num_bands
        self.device = device
        self.K_schedule = [int(K) for K in K_schedule]
        self._band_prefix_sizes = np.cumsum([len(band) for band in bands])

        # Randomly select a single global ordering over coefficients.
        rng = np.random.RandomState(seed)
        self._random_order = torch.from_numpy(rng.permutation(total)).to(device)
        self._mask_cache = {}
        self.current_K = None
        self.mask = None
        self.set_epoch(0)

    def _get_mask_for_K(self, K: int) -> torch.Tensor:
        mask = self._mask_cache.get(K)
        if mask is None:
            n_keep = int(self._band_prefix_sizes[K - 1])
            flat_mask = torch.zeros_like(self._random_order, dtype=torch.float32)
            flat_mask[self._random_order[:n_keep]] = 1.0
            mask = flat_mask.reshape(1, 1, H, W)
            self._mask_cache[K] = mask
        return mask

    def set_epoch(self, epoch: int):
        self.current_K = self.K_schedule[min(epoch, len(self.K_schedule) - 1)]
        self.mask = self._get_mask_for_K(self.current_K)

    @torch.no_grad()
    def serve(self, coeffs: torch.Tensor) -> torch.Tensor:
        return idct2d(coeffs * self.mask)

    def get_io_ratio(self):
        return k_only_ratio(self.current_K, self.num_bands)


class RandomPatchServer:
    """
    Baseline 4: Same (K, q) as TDCF schedule, but patches are chosen
    RANDOMLY instead of by gradient sensitivity.
    Proves that gradient-informed patch selection matters.
    """

    def __init__(self, K_schedule, q_schedule, H: int, W: int, num_bands: int,
                 patch_size: int, device: torch.device, seed: int = 0):
        masks = build_nested_masks(H, W, num_bands)
        self.num_bands = num_bands
        self.patch_size = patch_size
        self.nph = H // patch_size
        self.npw = W // patch_size
        self.P = self.nph * self.npw
        self.device = device
        self.K_schedule = [int(K) for K in K_schedule]
        self.q_schedule = [int(q) for q in q_schedule]
        self._freq_masks = {
            K: masks[K - 1].to(device).unsqueeze(0).unsqueeze(0)
            for K in sorted(set(self.K_schedule))
        }

        # Random patch ordering shared across epochs; prefixes define q(e).
        rng = np.random.RandomState(seed)
        self._random_order = torch.from_numpy(
            rng.permutation(self.P)).to(device)
        self._keep_mask_cache = {}
        self.current_K = None
        self.current_q = None
        self.mask = None
        self.keep_mask = None
        self.set_epoch(0)

    def _get_keep_mask(self, q: int) -> torch.Tensor:
        keep_mask = self._keep_mask_cache.get(q)
        if keep_mask is None:
            keep_mask = torch.zeros(self.P, device=self.device, dtype=torch.bool)
            keep_mask[self._random_order[:q]] = True
            self._keep_mask_cache[q] = keep_mask
        return keep_mask

    def set_epoch(self, epoch: int):
        epoch_idx = min(epoch, len(self.K_schedule) - 1)
        self.current_K = self.K_schedule[epoch_idx]
        self.current_q = self.q_schedule[min(epoch, len(self.q_schedule) - 1)]
        self.mask = self._freq_masks[self.current_K]
        self.keep_mask = self._get_keep_mask(self.current_q)

    @torch.no_grad()
    def serve(self, coeffs: torch.Tensor) -> torch.Tensor:
        B, C_ch = coeffs.shape[:2]
        ps = self.patch_size
        c_base = coeffs * self.mask
        x_base = idct2d(c_base)

        if self.current_q >= self.P:
            return idct2d(coeffs)

        x_full = idct2d(coeffs)
        residual = x_full - x_base

        r_patches = (
            residual[:, :, :self.nph * ps, :self.npw * ps]
            .reshape(B, C_ch, self.nph, ps, self.npw, ps)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(B, self.P, C_ch, ps, ps)
        )

        r_patches[:, ~self.keep_mask] = 0.0

        r_spatial = (
            r_patches
            .reshape(B, self.nph, self.npw, C_ch, ps, ps)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(B, C_ch, self.nph * ps, self.npw * ps)
        )
        out = x_base.clone()
        out[:, :, :self.nph * ps, :self.npw * ps] += r_spatial
        return out

    def get_io_ratio(self):
        return kq_ratio(self.current_K, self.current_q,
                        self.num_bands, self.P)


# ---------------------------------------------------------------------------
# Generic training loop (shared by all baselines)
# ---------------------------------------------------------------------------

def train_epoch_with_server(model, loader, opt, crit, device, server):
    model.train()
    loss_s = correct = n = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        x = server.serve(x)
        x = augment_cifar_batch(x)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        loss_s += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss_s / n, correct / n


def run_single_baseline(name, server, args, train_coeffs, train_labels,
                        test_loader, device, log):
    """Run one baseline from scratch and return history dict."""
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = make_model(args.backbone, args.dataset, device)
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                    weight_decay=args.wd, nesterov=True)
    sched_lr = optim.lr_scheduler.CosineAnnealingLR(opt, args.total_epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    train_tds = TensorDataset(train_coeffs, train_labels)
    train_loader = DataLoader(train_tds, batch_size=args.batch_size, shuffle=True)

    log.info("=" * 60)
    log.info("BASELINE: %s", name)
    log.info("=" * 60)

    hist = {k: [] for k in ["epoch", "train_loss", "train_acc",
                             "test_loss", "test_acc", "wall_s", "io_ratio"]}
    t0 = time.time()

    for ep in range(args.total_epochs):
        if hasattr(server, "set_epoch"):
            server.set_epoch(ep)
        io_ratio = server.get_io_ratio()
        tr_l, tr_a = train_epoch_with_server(
            model, train_loader, opt, crit, device, server)
        te_l, te_a = evaluate(model, test_loader, crit, device)
        sched_lr.step()
        elapsed = time.time() - t0
        lr_now = opt.param_groups[0]["lr"]

        for k, v in [("epoch", ep), ("train_loss", tr_l), ("train_acc", tr_a),
                     ("test_loss", te_l), ("test_acc", te_a),
                     ("wall_s", elapsed), ("io_ratio", io_ratio)]:
            hist[k].append(v)

        log.info("  E %3d/%d | ratio=%.3f | tr_acc=%.4f te_acc=%.4f | "
                 "lr=%.2e | %.0fs",
                 ep + 1, args.total_epochs, io_ratio,
                 tr_a, te_a, lr_now, elapsed)

    log.info("  FINAL: %s  best_te=%.4f  final_te=%.4f  ratio=%.3f",
             name, max(hist["test_acc"]), hist["test_acc"][-1], io_ratio)
    return hist


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    log = setup_logger(args.save_dir)

    log.info("=" * 60)
    log.info("  TDCF Matched-Budget Baselines")
    log.info("  %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 60)

    # Load TDCF results to extract budget
    with open(args.tdcf_results) as f:
        tdcf_res = json.load(f)

    tdcf_hist = tdcf_res["tdcf"]
    ratio_hist = tdcf_hist.get("approx_ratio", tdcf_hist.get("io_ratio", [1.0]))
    tdcf_avg_ratio = float(np.mean(ratio_hist))
    tdcf_K_schedule = [int(K) for K in tdcf_res.get("schedule_K", [])]
    tdcf_q_schedule = [int(q) for q in tdcf_res.get("schedule_q", [])]
    static_lowpass_K, static_lowpass_ratio = match_static_lowpass(tdcf_avg_ratio)
    static_kq_K, static_kq_q, static_kq_ratio = match_static_kq(tdcf_avg_ratio)

    if not tdcf_K_schedule:
        tdcf_K_schedule = [static_kq_K] * args.total_epochs
    if not tdcf_q_schedule:
        tdcf_q_schedule = [static_kq_q] * args.total_epochs
    tdcf_K = int(np.median(tdcf_K_schedule))
    tdcf_q = int(np.median(tdcf_q_schedule))

    log.info("TDCF reference: avg_ratio=%.3f  median(K,q)=(%d,%d)",
             tdcf_avg_ratio, tdcf_K, tdcf_q)
    log.info("Matched static low-pass target: K=%d  ratio=%.3f",
             static_lowpass_K, static_lowpass_ratio)
    log.info("Matched static (K,q) target: K=%d  q=%d  ratio=%.3f",
             static_kq_K, static_kq_q, static_kq_ratio)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    plain_tfm = get_plain_transform()
    train_ds = datasets.CIFAR100(args.data_dir, True, download=True, transform=plain_tfm)
    test_ds = datasets.CIFAR100(args.data_dir, False, download=True, transform=plain_tfm)

    log.info("Pre-computing DCT coefficients ...")
    train_coeffs, train_labels = precompute_dct_dataset(
        train_ds, device, num_workers=args.data_workers)
    test_coeffs, test_labels = precompute_dct_dataset(
        test_ds, device, num_workers=args.data_workers)
    test_images = idct2d(test_coeffs)
    test_loader = DataLoader(
        TensorDataset(test_images, test_labels), batch_size=512, shuffle=False)

    all_results = {"tdcf_reference": {
        "avg_ratio": tdcf_avg_ratio,
        "median_K": tdcf_K,
        "median_q": tdcf_q,
        "schedule_K": tdcf_K_schedule,
        "schedule_q": tdcf_q_schedule,
    }}

    run_all = args.run_all

    # --- Baseline 1: Static Low-Pass ---
    if run_all or args.run_static_lowpass:
        server = StaticLowPassServer(static_lowpass_K, H, W, NUM_BANDS, device)
        hist = run_single_baseline(
            "static_lowpass", server, args,
            train_coeffs, train_labels, test_loader, device, log)
        all_results["static_lowpass"] = hist

    # --- Baseline 2: Static (K, q) ---
    if run_all or args.run_static_kq:
        server = StaticKQServer(static_kq_K, static_kq_q, H, W, NUM_BANDS,
                                PATCH_SIZE, device)
        hist = run_single_baseline(
            "static_kq", server, args,
            train_coeffs, train_labels, test_loader, device, log)
        all_results["static_kq"] = hist

    # --- Baseline 3: Random Coefficient Dropping ---
    if run_all or args.run_random_coeff:
        server = RandomCoeffServer(tdcf_K_schedule, H, W, NUM_BANDS, device,
                                   seed=args.seed)
        hist = run_single_baseline(
            "random_coeff", server, args,
            train_coeffs, train_labels, test_loader, device, log)
        all_results["random_coeff"] = hist

    # --- Baseline 4: Random Patch Dropping ---
    if run_all or args.run_random_patch:
        server = RandomPatchServer(tdcf_K_schedule, tdcf_q_schedule, H, W,
                                   NUM_BANDS,
                                   PATCH_SIZE, device, seed=args.seed)
        hist = run_single_baseline(
            "random_patch", server, args,
            train_coeffs, train_labels, test_loader, device, log)
        all_results["random_patch"] = hist

    # --- Save ---
    all_results["config"] = vars(args)
    with open(os.path.join(args.save_dir, "baselines_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # --- Summary table ---
    log.info("\n" + "=" * 60)
    log.info("SUMMARY TABLE")
    log.info("=" * 60)
    log.info("%-20s  %8s  %8s  %8s", "Method", "BestTest", "FinalTest", "Ratio")
    log.info("-" * 55)

    for name in ["static_lowpass", "static_kq", "random_coeff", "random_patch"]:
        if name in all_results:
            h = all_results[name]
            log.info("%-20s  %8.4f  %8.4f  %8.3f",
                     name, max(h["test_acc"]), h["test_acc"][-1],
                     h["io_ratio"][-1])

    log.info("%-20s  %8s  %8s  %8.3f",
             "TDCF (reference)", "—", "—", tdcf_avg_ratio)
    log.info("\nResults saved to %s/", args.save_dir)


if __name__ == "__main__":
    main()
