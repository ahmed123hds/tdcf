"""
Quantized block-DCT precision search.

This module tests whether DCT coefficients can be stored as compact integers
instead of raw float tensors. It is intentionally independent from the training
loops: first prove the representation is numerically safe, then decide whether
it is worth wiring into a physical store.

Search strategy:
  1. Compute block-DCT coefficients for a sample of images.
  2. Store coefficients in zig-zag order.
  3. Try int8 first, then int16 only if int8 cannot meet quality targets.
  4. For each dtype, binary-search the largest quantization multiplier that
     still passes PSNR/SSIM/error thresholds. Larger multiplier means coarser
     quantization, more zeros, and usually better compression.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
import zlib
from dataclasses import asdict, dataclass
from typing import Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torchvision.datasets as tv_datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset

from tdcf.transforms import block_dct2d, block_idct2d, zigzag_order


@dataclass
class TrialResult:
    dtype: str
    multiplier: float
    passed: bool
    psnr_db: float
    mse: float
    mae: float
    max_abs: float
    global_ssim: float
    coeff_mse: float
    coeff_mae: float
    clip_fraction: float
    zero_fraction: float
    raw_coeff_bytes: int
    scale_bytes: int
    zlib_bytes: int
    zlib_ratio_vs_raw_uint8: float
    raw_ratio_vs_raw_uint8: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Quantized DCT coefficient precision search")
    p.add_argument("--source", choices=["synthetic", "cifar100", "imagefolder", "webdataset"], default="synthetic")
    p.add_argument("--data_root", type=str, default=None,
                   help="ImageFolder root when --source=imagefolder.")
    p.add_argument("--shards", type=str, default=None,
                   help="WebDataset shard expression when --source=webdataset.")
    p.add_argument("--split", type=str, default="train",
                   help="Optional split subfolder under --data_root for ImageFolder.")
    p.add_argument("--num_samples", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--block_size", type=int, default=8)
    p.add_argument("--num_bands", type=int, default=16)
    p.add_argument("--scale_scope", choices=["band", "channel_band"], default="band")
    p.add_argument("--scale_stat", choices=["max", "percentile"], default="max")
    p.add_argument("--scale_percentile", type=float, default=99.99)
    p.add_argument("--target_psnr", type=float, default=45.0)
    p.add_argument("--target_ssim", type=float, default=0.995)
    p.add_argument("--target_mae", type=float, default=None)
    p.add_argument("--target_max_abs", type=float, default=None)
    p.add_argument("--dtype_order", type=str, default="int8,int16",
                   help="Comma-separated dtype search order. Supported: int8,int16.")
    p.add_argument("--max_multiplier", type=float, default=256.0)
    p.add_argument("--binary_steps", type=int, default=18)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="./results/dct_quant_precision")
    p.add_argument("--save_recon_grid", action="store_true")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def image_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
    ])


def synthetic_loader(num_samples: int, batch_size: int, img_size: int) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    # Smooth random images are a better DCT smoke test than pure white noise.
    low = torch.rand(num_samples, 3, max(8, img_size // 8), max(8, img_size // 8))
    images = torch.nn.functional.interpolate(
        low, size=(img_size, img_size), mode="bilinear", align_corners=False
    )
    images = (images + 0.03 * torch.randn_like(images)).clamp(0.0, 1.0)
    labels = torch.zeros(num_samples, dtype=torch.long)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=False)


def imagefolder_loader(args: argparse.Namespace) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    root = args.data_root
    if root is None:
        raise ValueError("--data_root is required for --source=imagefolder")
    split_root = os.path.join(root, args.split)
    if os.path.isdir(split_root):
        root = split_root
    dataset = tv_datasets.ImageFolder(root, transform=image_transform(args.img_size))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )


def cifar100_loader(args: argparse.Namespace) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    root = args.data_root or "./data"
    dataset = tv_datasets.CIFAR100(
        root=root,
        train=args.split != "test",
        download=False,
        transform=image_transform(args.img_size),
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )


def webdataset_loader(args: argparse.Namespace) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    if args.shards is None:
        raise ValueError("--shards is required for --source=webdataset")
    try:
        import webdataset as wds
    except ImportError as exc:
        raise RuntimeError("webdataset is required for --source=webdataset") from exc

    dataset = (
        wds.WebDataset(args.shards, resampled=False, shardshuffle=False)
        .decode("pil")
        .to_tuple("jpg;jpeg;png", "cls")
        .map_tuple(image_transform(args.img_size), lambda y: int(y))
    )
    return wds.WebLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )


def load_sample_images(args: argparse.Namespace) -> torch.Tensor:
    if args.source == "synthetic":
        loader = synthetic_loader(args.num_samples, args.batch_size, args.img_size)
    elif args.source == "cifar100":
        loader = cifar100_loader(args)
    elif args.source == "imagefolder":
        loader = imagefolder_loader(args)
    else:
        loader = webdataset_loader(args)

    chunks: List[torch.Tensor] = []
    seen = 0
    for images, _ in loader:
        take = min(images.size(0), args.num_samples - seen)
        chunks.append(images[:take].float().cpu())
        seen += take
        if seen >= args.num_samples:
            break
    if not chunks:
        raise RuntimeError("No images were loaded for the precision test.")
    return torch.cat(chunks, dim=0)


def make_band_slices(total_coeffs: int, num_bands: int) -> List[slice]:
    band_size = total_coeffs // num_bands
    bands = []
    for b in range(num_bands):
        start = b * band_size
        end = (b + 1) * band_size if b < num_bands - 1 else total_coeffs
        bands.append(slice(start, end))
    return bands


def dct_to_zigzag(images: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, int, int, torch.Tensor]:
    coeffs = block_dct2d(images, block_size=block_size)
    _, _, _, bs_h, bs_w = coeffs.shape
    nph = images.shape[-2] // block_size
    npw = images.shape[-1] // block_size
    zz = zigzag_order(bs_h, bs_w)
    flat = coeffs.reshape(coeffs.shape[0], coeffs.shape[1], coeffs.shape[2], bs_h * bs_w)
    return flat[:, :, :, zz].contiguous(), nph, npw, zz


def zigzag_to_dct(coeffs_zz: torch.Tensor, zz: torch.Tensor, block_size: int) -> torch.Tensor:
    coeffs = torch.empty_like(coeffs_zz)
    coeffs[:, :, :, zz] = coeffs_zz
    return coeffs.reshape(coeffs.shape[0], coeffs.shape[1], coeffs.shape[2], block_size, block_size)


def compute_base_scales(
    coeffs_zz: torch.Tensor,
    num_bands: int,
    dtype: str,
    scale_scope: str,
    scale_stat: str,
    scale_percentile: float,
) -> torch.Tensor:
    qmax = 127 if dtype == "int8" else 32767
    _, _, channels, total_coeffs = coeffs_zz.shape
    bands = make_band_slices(total_coeffs, num_bands)
    scale_shape = (channels, num_bands) if scale_scope == "channel_band" else (1, num_bands)
    scales = torch.empty(scale_shape, dtype=torch.float32)

    for b, sl in enumerate(bands):
        if scale_scope == "channel_band":
            for c in range(channels):
                vals = coeffs_zz[:, :, c, sl].abs().reshape(-1)
                max_abs = robust_abs_stat(vals, scale_stat, scale_percentile)
                scales[c, b] = max(max_abs / qmax, 1e-12)
        else:
            vals = coeffs_zz[:, :, :, sl].abs().reshape(-1)
            max_abs = robust_abs_stat(vals, scale_stat, scale_percentile)
            scales[0, b] = max(max_abs / qmax, 1e-12)
    return scales


def robust_abs_stat(vals: torch.Tensor, scale_stat: str, percentile: float) -> float:
    if vals.numel() == 0:
        return 1e-12
    if scale_stat == "max":
        return float(vals.max().item())
    q = min(max(percentile / 100.0, 0.0), 1.0)
    return float(torch.quantile(vals.float(), q).item())


def quantize_dequantize(
    coeffs_zz: torch.Tensor,
    base_scales: torch.Tensor,
    multiplier: float,
    dtype: str,
    num_bands: int,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    qmin, qmax = (-128, 127) if dtype == "int8" else (-32768, 32767)
    np_dtype = np.int8 if dtype == "int8" else np.int16
    bands = make_band_slices(coeffs_zz.shape[-1], num_bands)

    q_np = np.empty(coeffs_zz.shape, dtype=np_dtype)
    deq = torch.empty_like(coeffs_zz)
    clipped = 0
    total = coeffs_zz.numel()

    for b, sl in enumerate(bands):
        if base_scales.shape[0] == 1:
            scale = base_scales[0, b] * multiplier
            raw = coeffs_zz[:, :, :, sl] / scale
            rounded = torch.round(raw)
            q = rounded.clamp(qmin, qmax)
            clipped += int((rounded.ne(q)).sum().item())
            deq[:, :, :, sl] = q * scale
            q_np[:, :, :, sl] = q.cpu().numpy().astype(np_dtype, copy=False)
        else:
            for c in range(coeffs_zz.shape[2]):
                scale = base_scales[c, b] * multiplier
                raw = coeffs_zz[:, :, c, sl] / scale
                rounded = torch.round(raw)
                q = rounded.clamp(qmin, qmax)
                clipped += int((rounded.ne(q)).sum().item())
                deq[:, :, c, sl] = q * scale
                q_np[:, :, c, sl] = q.cpu().numpy().astype(np_dtype, copy=False)

    zero_fraction = float((q_np == 0).sum()) / float(q_np.size)
    clip_fraction = clipped / float(total)
    return deq, torch.from_numpy(q_np), clip_fraction, zero_fraction


def global_ssim(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().reshape(x.shape[0], x.shape[1], -1)
    y = y.float().reshape(y.shape[0], y.shape[1], -1)
    mu_x = x.mean(dim=-1)
    mu_y = y.mean(dim=-1)
    var_x = x.var(dim=-1, unbiased=False)
    var_y = y.var(dim=-1, unbiased=False)
    cov = ((x - mu_x.unsqueeze(-1)) * (y - mu_y.unsqueeze(-1))).mean(dim=-1)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    )
    return float(ssim.mean().item())


def evaluate_trial(
    images: torch.Tensor,
    coeffs_zz: torch.Tensor,
    base_scales: torch.Tensor,
    multiplier: float,
    dtype: str,
    num_bands: int,
    block_size: int,
    nph: int,
    npw: int,
    zz: torch.Tensor,
    target_psnr: float,
    target_ssim: float,
    target_mae: Optional[float],
    target_max_abs: Optional[float],
) -> TrialResult:
    deq_zz, q_tensor, clip_fraction, zero_fraction = quantize_dequantize(
        coeffs_zz, base_scales, multiplier, dtype, num_bands
    )
    coeffs_deq = zigzag_to_dct(deq_zz, zz, block_size)
    recon = block_idct2d(coeffs_deq, nph, npw).clamp(0.0, 1.0)

    diff = recon - images
    mse = float(diff.square().mean().item())
    mae = float(diff.abs().mean().item())
    max_abs = float(diff.abs().max().item())
    psnr = 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
    ssim = global_ssim(images, recon)
    coeff_diff = deq_zz - coeffs_zz
    coeff_mse = float(coeff_diff.square().mean().item())
    coeff_mae = float(coeff_diff.abs().mean().item())

    q_np = q_tensor.numpy()
    coeff_bytes = q_np.nbytes
    scale_bytes = base_scales.numel() * 4
    zlib_bytes = len(zlib.compress(q_np.tobytes(), level=6)) + scale_bytes
    raw_uint8_bytes = int(images.numel())
    raw_total = coeff_bytes + scale_bytes
    passed = psnr >= target_psnr and ssim >= target_ssim
    if target_mae is not None:
        passed = passed and mae <= target_mae
    if target_max_abs is not None:
        passed = passed and max_abs <= target_max_abs

    return TrialResult(
        dtype=dtype,
        multiplier=float(multiplier),
        passed=bool(passed),
        psnr_db=psnr,
        mse=mse,
        mae=mae,
        max_abs=max_abs,
        global_ssim=ssim,
        coeff_mse=coeff_mse,
        coeff_mae=coeff_mae,
        clip_fraction=clip_fraction,
        zero_fraction=zero_fraction,
        raw_coeff_bytes=int(raw_total),
        scale_bytes=int(scale_bytes),
        zlib_bytes=int(zlib_bytes),
        zlib_ratio_vs_raw_uint8=float(zlib_bytes / raw_uint8_bytes),
        raw_ratio_vs_raw_uint8=float(raw_total / raw_uint8_bytes),
    )


def search_dtype(
    images: torch.Tensor,
    coeffs_zz: torch.Tensor,
    dtype: str,
    args: argparse.Namespace,
    nph: int,
    npw: int,
    zz: torch.Tensor,
) -> Tuple[Optional[TrialResult], List[TrialResult], torch.Tensor]:
    base_scales = compute_base_scales(
        coeffs_zz,
        args.num_bands,
        dtype,
        args.scale_scope,
        args.scale_stat,
        args.scale_percentile,
    )
    trials: List[TrialResult] = []

    def run(mult: float) -> TrialResult:
        result = evaluate_trial(
            images=images,
            coeffs_zz=coeffs_zz,
            base_scales=base_scales,
            multiplier=mult,
            dtype=dtype,
            num_bands=args.num_bands,
            block_size=args.block_size,
            nph=nph,
            npw=npw,
            zz=zz,
            target_psnr=args.target_psnr,
            target_ssim=args.target_ssim,
            target_mae=args.target_mae,
            target_max_abs=args.target_max_abs,
        )
        trials.append(result)
        return result

    # First try the most compact endpoint. If it passes, no narrower search is
    # needed for this dtype.
    high = max(1.0, float(args.max_multiplier))
    high_result = run(high)
    if high_result.passed:
        return high_result, trials, base_scales

    low_result = run(1.0)
    if not low_result.passed:
        return None, trials, base_scales

    lo, hi = 1.0, high
    best = low_result
    for _ in range(args.binary_steps):
        mid = math.sqrt(lo * hi)
        result = run(mid)
        if result.passed:
            best = result
            lo = mid
        else:
            hi = mid
    return best, trials, base_scales


def save_report(
    args: argparse.Namespace,
    images: torch.Tensor,
    all_trials: List[TrialResult],
    best: Optional[TrialResult],
    save_dir: str,
    elapsed: float,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    report = {
        "args": vars(args),
        "num_images": int(images.shape[0]),
        "image_shape": list(images.shape[1:]),
        "raw_uint8_bytes_estimate": int(images.numel()),
        "elapsed_sec": elapsed,
        "best": asdict(best) if best is not None else None,
        "trials": [asdict(t) for t in all_trials],
    }
    with open(os.path.join(save_dir, "quantized_dct_precision_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    csv_path = os.path.join(save_dir, "quantized_dct_precision_trials.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_trials[0]).keys()))
        writer.writeheader()
        for trial in all_trials:
            writer.writerow(asdict(trial))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    start = time.time()

    images = load_sample_images(args)
    if images.shape[-1] % args.block_size != 0 or images.shape[-2] % args.block_size != 0:
        raise ValueError("--img_size must be divisible by --block_size for this precision test.")

    coeffs_zz, nph, npw, zz = dct_to_zigzag(images, args.block_size)
    dtype_order = [d.strip() for d in args.dtype_order.split(",") if d.strip()]
    unsupported = [d for d in dtype_order if d not in {"int8", "int16"}]
    if unsupported:
        raise ValueError(f"Unsupported dtype(s): {unsupported}")

    all_trials: List[TrialResult] = []
    best: Optional[TrialResult] = None
    best_scales: Optional[torch.Tensor] = None
    for dtype in dtype_order:
        dtype_best, dtype_trials, dtype_scales = search_dtype(
            images, coeffs_zz, dtype, args, nph, npw, zz
        )
        all_trials.extend(dtype_trials)
        if dtype_best is not None:
            best = dtype_best
            best_scales = dtype_scales
            break

    if not all_trials:
        raise RuntimeError("No quantization trials were executed.")

    os.makedirs(args.save_dir, exist_ok=True)
    if best_scales is not None:
        np.save(os.path.join(args.save_dir, "best_scales.npy"), best_scales.numpy())
    save_report(args, images, all_trials, best, args.save_dir, time.time() - start)

    print("=" * 72)
    print("Quantized DCT precision search")
    print(f"source={args.source} | samples={images.shape[0]} | img={args.img_size} | block={args.block_size}")
    print(f"targets: PSNR>={args.target_psnr:.2f} dB, SSIM>={args.target_ssim:.5f}")
    if best is None:
        print("No dtype/multiplier met the requested quality targets.")
    else:
        print(
            f"best: dtype={best.dtype} multiplier={best.multiplier:.6g} "
            f"PSNR={best.psnr_db:.2f} SSIM={best.global_ssim:.5f} "
            f"MAE={best.mae:.6f} max={best.max_abs:.6f}"
        )
        print(
            f"storage estimate: raw_coeff={best.raw_ratio_vs_raw_uint8:.3f}x raw_uint8, "
            f"zlib={best.zlib_ratio_vs_raw_uint8:.3f}x raw_uint8, "
            f"zero_fraction={best.zero_fraction:.3f}, clipped={best.clip_fraction:.6f}"
        )
    print(f"report: {os.path.join(args.save_dir, 'quantized_dct_precision_report.json')}")
    print("=" * 72)


if __name__ == "__main__":
    main()
