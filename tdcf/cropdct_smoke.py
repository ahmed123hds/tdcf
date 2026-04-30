"""Smoke tests for the CropDCT physical format."""

from __future__ import annotations

import argparse
import math
import os
import shutil

import torch

from tdcf.cropdct_store import CropDCTStore, CropDCTWriter


def parse_args():
    p = argparse.ArgumentParser("CropDCT smoke test")
    p.add_argument("--out_dir", type=str, default="./results/_smoke_cropdct")
    p.add_argument("--keep", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def psnr(x: torch.Tensor, y: torch.Tensor):
    mse = torch.mean((x - y) ** 2).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def main():
    args = parse_args()
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    def make_image(height: int, width: int, phase: float) -> torch.Tensor:
        yy = torch.linspace(0, 1, height).view(1, height, 1)
        xx = torch.linspace(0, 1, width).view(1, 1, width)
        r = (0.65 * xx + 0.25 * yy + 0.10 * torch.sin((xx + phase) * 8.0)).clamp(0, 1)
        g = (0.35 * xx + 0.55 * yy + 0.10 * torch.cos((yy + phase) * 7.0)).clamp(0, 1)
        b = (0.45 * (1 - xx) + 0.35 * yy + 0.08 * torch.sin((xx + yy) * 10.0)).clamp(0, 1)
        return torch.cat([r.expand(1, height, width), g.expand(1, height, width), b.expand(1, height, width)])

    samples = [
        (make_image(96, 128, 0.1), 3),
        (make_image(111, 143, 0.4), 5),
        (make_image(160, 112, 0.7), 7),
    ]
    writer = CropDCTWriter(
        args.out_dir,
        records_per_shard=2,
        block_size=8,
        tile_blocks=4,
        quality=95,
        compression="zstd",
        compression_level=1,
        device=torch.device(args.device),
    )
    for image, label in samples:
        writer.add(image, label)
    writer.close()

    store = CropDCTStore(args.out_dir, device=torch.device(args.device))
    full_psnrs = []
    for idx, (src, label) in enumerate(samples):
        crop, got_label = store.read_crop(
            idx,
            crop_box=(0, 0, src.shape[-2], src.shape[-1]),
            freq_bands=list(range(store.num_bands)),
            output_size=None,
        )
        full_psnrs.append(psnr(crop.cpu().squeeze(0), src))
        assert got_label == label, (got_label, label)
    mean_psnr = sum(full_psnrs) / len(full_psnrs)
    if mean_psnr < 40.0:
        raise AssertionError(f"CropDCT reconstruction PSNR too low: {mean_psnr:.2f} dB")

    crop_box = (16, 16, 64, 64)
    store.reset_stats()
    _img_full, _ = store.read_crop(0, crop_box=crop_box, freq_bands=[0, 1, 2, 3], output_size=64)
    full_stats = store.get_read_stats()
    store.reset_stats()
    _img_low, _ = store.read_crop(0, crop_box=crop_box, freq_bands=[0, 1, 2], output_size=64)
    low_stats = store.get_read_stats()
    if not low_stats["bytes_read"] < full_stats["bytes_read"]:
        raise AssertionError(
            f"Band skipping did not reduce physical reads: {low_stats} vs {full_stats}"
        )
    store.close()

    print("=" * 72)
    print("CropDCT smoke test passed")
    print(f"mean_full_reconstruction_psnr={mean_psnr:.2f} dB")
    print(f"full_read_bytes={full_stats['bytes_read']} low_band_read_bytes={low_stats['bytes_read']}")
    print(f"full_stats={full_stats}")
    print(f"low_band_stats={low_stats}")
    print("=" * 72)
    if not args.keep:
        shutil.rmtree(args.out_dir)


if __name__ == "__main__":
    main()
