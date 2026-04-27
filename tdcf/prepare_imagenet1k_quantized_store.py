"""
Prepare a compressed int8 block-DCT store for ImageNet-1K fixed views.

The store is intentionally fixed-view:
  original image -> resize shorter side to view_size -> center crop view_size
  -> block-DCT -> zig-zag -> int8 quantization -> zlib-compressed tile/band chunks

Training can still sample block-aligned 224x224 crops from the stored 288x288
view, giving a physical-I/O proof without solving the full variable-resolution
crop database problem.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import zlib
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T

from tdcf.quantized_store import build_tile_specs
from tdcf.transforms import block_dct2d, zigzag_order


def parse_args():
    p = argparse.ArgumentParser("Prepare quantized fixed-view ImageNet DCT store")
    p.add_argument("--shards", type=str, default=None)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--source", choices=["webdataset", "synthetic"], default="webdataset")
    p.add_argument("--view_size", type=int, default=288)
    p.add_argument("--block_size", type=int, default=8)
    p.add_argument("--num_bands", type=int, default=16)
    p.add_argument("--tile_blocks", type=int, default=4)
    p.add_argument("--chunk_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--calibration_samples", type=int, default=4096)
    p.add_argument("--scale_scope", choices=["band", "channel_band"], default="band")
    p.add_argument("--scale_stat", choices=["max", "percentile"], default="max")
    p.add_argument("--scale_percentile", type=float, default=99.99)
    p.add_argument("--quant_multiplier", type=float, default=1.05)
    p.add_argument("--compression_level", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def identity_label(label):
    return int(label)


def build_transform(view_size: int):
    return T.Compose([
        T.Resize(view_size),
        T.CenterCrop(view_size),
        T.ToTensor(),
    ])


def build_webdataset_loader(shards: str, view_size: int, num_workers: int):
    import webdataset as wds

    transform = build_transform(view_size)
    dataset = (
        wds.WebDataset(shards, resampled=False, shardshuffle=False)
        .decode("pil")
        .to_tuple("jpg;jpeg;png", "cls")
        .map_tuple(transform, identity_label)
    )
    return wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def build_synthetic_loader(view_size: int, max_samples: int):
    n = max(max_samples, 32)
    low = torch.rand(n, 3, max(8, view_size // 8), max(8, view_size // 8))
    images = torch.nn.functional.interpolate(
        low, size=(view_size, view_size), mode="bilinear", align_corners=False
    ).clamp(0.0, 1.0)
    labels = torch.arange(n, dtype=torch.long) % 1000
    return zip(images, labels)


def iter_samples(args):
    if args.source == "synthetic":
        return build_synthetic_loader(args.view_size, args.max_samples)
    if args.shards is None:
        raise ValueError("--shards is required for --source=webdataset")
    return build_webdataset_loader(args.shards, args.view_size, args.num_workers)


def robust_abs_stat(vals: torch.Tensor, scale_stat: str, percentile: float) -> float:
    vals = vals.reshape(-1).abs().float()
    if vals.numel() == 0:
        return 1e-12
    if scale_stat == "max":
        return float(vals.max().item())
    return float(torch.quantile(vals, min(max(percentile / 100.0, 0.0), 1.0)).item())


@torch.no_grad()
def calibrate_scales(args, device: torch.device):
    print(f"[quant-store] Calibrating scales on {args.calibration_samples} samples", flush=True)
    qmax = 127.0
    nph = args.view_size // args.block_size
    npw = args.view_size // args.block_size
    total_coeffs = args.block_size * args.block_size
    band_size = total_coeffs // args.num_bands
    zz = zigzag_order(args.block_size, args.block_size)
    stats_shape = (3, args.num_bands) if args.scale_scope == "channel_band" else (1, args.num_bands)
    max_stats = torch.zeros(stats_shape, dtype=torch.float32)

    batch_images: List[torch.Tensor] = []
    seen = 0

    def flush_batch():
        nonlocal batch_images
        if not batch_images:
            return
        images = torch.stack(batch_images).to(device)
        coeffs = block_dct2d(images, block_size=args.block_size)
        flat = coeffs.reshape(coeffs.shape[0], nph * npw, 3, total_coeffs)[:, :, :, zz]
        for band in range(args.num_bands):
            start = band * band_size
            end = (band + 1) * band_size if band < args.num_bands - 1 else total_coeffs
            if args.scale_scope == "channel_band":
                for c in range(3):
                    stat = robust_abs_stat(flat[:, :, c, start:end], args.scale_stat, args.scale_percentile)
                    max_stats[c, band] = max(float(max_stats[c, band]), stat)
            else:
                stat = robust_abs_stat(flat[:, :, :, start:end], args.scale_stat, args.scale_percentile)
                max_stats[0, band] = max(float(max_stats[0, band]), stat)
        batch_images = []

    for image, _label in iter_samples(args):
        batch_images.append(image.float())
        seen += 1
        if len(batch_images) >= args.batch_size:
            flush_batch()
        if seen >= args.calibration_samples:
            break
    flush_batch()

    scales = (max_stats / qmax).clamp_min(1e-12) * float(args.quant_multiplier)
    print(
        f"[quant-store] Scales ready | min={scales.min().item():.6g} "
        f"max={scales.max().item():.6g} multiplier={args.quant_multiplier}",
        flush=True,
    )
    return scales.cpu().numpy().astype(np.float32)


def _open_band_files(out_dir: str, num_bands: int):
    return [
        open(os.path.join(out_dir, f"band_{band:02d}.bin"), "wb")
        for band in range(num_bands)
    ]


def _close_all(handles):
    for handle in handles:
        handle.close()


@torch.no_grad()
def build_store(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    scales = calibrate_scales(args, device)
    np.save(os.path.join(args.out_dir, "scales.npy"), scales)

    nph = args.view_size // args.block_size
    npw = args.view_size // args.block_size
    P = nph * npw
    total_coeffs = args.block_size * args.block_size
    band_size = total_coeffs // args.num_bands
    band_boundaries = [b * band_size for b in range(args.num_bands + 1)]
    band_boundaries[-1] = total_coeffs
    zz = zigzag_order(args.block_size, args.block_size).numpy()
    tile_specs = build_tile_specs(nph, npw, args.tile_blocks)

    band_files = _open_band_files(args.out_dir, args.num_bands)
    offsets: List[np.ndarray] = []
    lengths: List[np.ndarray] = []
    raw_lengths: List[np.ndarray] = []
    chunk_sizes: List[int] = []
    labels: List[int] = []

    chunk_images: List[torch.Tensor] = []
    chunk_labels: List[int] = []
    processed = 0
    chunk_id = 0

    def write_chunk():
        nonlocal chunk_images, chunk_labels, chunk_id, processed
        if not chunk_images:
            return
        images = torch.stack(chunk_images).to(device)
        coeffs = block_dct2d(images, block_size=args.block_size)
        flat = coeffs.reshape(coeffs.shape[0], P, 3, total_coeffs).detach().cpu().numpy()
        coeffs_zz = flat[:, :, :, zz]

        chunk_n = coeffs_zz.shape[0]
        chunk_offsets = np.zeros((len(tile_specs), args.num_bands), dtype=np.int64)
        chunk_lengths = np.zeros((len(tile_specs), args.num_bands), dtype=np.int32)
        chunk_raw_lengths = np.zeros((len(tile_specs), args.num_bands), dtype=np.int32)

        for spec in tile_specs:
            tile_index = int(spec["tile_index"])
            block_ids = np.asarray(spec["block_ids"], dtype=np.int64)
            tile_coeffs = coeffs_zz[:, block_ids]
            for band in range(args.num_bands):
                start = band_boundaries[band]
                end = band_boundaries[band + 1]
                data = tile_coeffs[:, :, :, start:end]
                if scales.shape[0] == 1:
                    q = np.rint(data / scales[0, band])
                else:
                    q = np.rint(data / scales[:, band].reshape(1, 1, 3, 1))
                q = np.clip(q, -128, 127).astype(np.int8, copy=False)
                raw = np.ascontiguousarray(q).tobytes()
                payload = zlib.compress(raw, level=args.compression_level)
                handle = band_files[band]
                chunk_offsets[tile_index, band] = handle.tell()
                handle.write(payload)
                chunk_lengths[tile_index, band] = len(payload)
                chunk_raw_lengths[tile_index, band] = len(raw)

        offsets.append(chunk_offsets)
        lengths.append(chunk_lengths)
        raw_lengths.append(chunk_raw_lengths)
        chunk_sizes.append(chunk_n)
        labels.extend(chunk_labels)

        processed += chunk_n
        chunk_id += 1
        if processed % max(args.chunk_size * 10, 1) == 0:
            compressed = sum(os.path.getsize(os.path.join(args.out_dir, f"band_{b:02d}.bin")) for b in range(args.num_bands))
            print(
                f"[quant-store] processed={processed} chunks={chunk_id} "
                f"compressed={compressed / 1e9:.2f}GB",
                flush=True,
            )
        chunk_images = []
        chunk_labels = []

    for image, label in iter_samples(args):
        chunk_images.append(image.float())
        chunk_labels.append(int(label))
        if len(chunk_images) >= args.chunk_size:
            write_chunk()
        if args.max_samples > 0 and processed + len(chunk_images) >= args.max_samples:
            break
    write_chunk()
    _close_all(band_files)

    if not chunk_sizes:
        raise RuntimeError("No samples were written.")

    offsets_arr = np.stack(offsets, axis=0)
    lengths_arr = np.stack(lengths, axis=0)
    raw_lengths_arr = np.stack(raw_lengths, axis=0)
    np.save(os.path.join(args.out_dir, "offsets.npy"), offsets_arr)
    np.save(os.path.join(args.out_dir, "lengths.npy"), lengths_arr)
    np.save(os.path.join(args.out_dir, "raw_lengths.npy"), raw_lengths_arr)
    np.save(os.path.join(args.out_dir, "chunk_sizes.npy"), np.asarray(chunk_sizes, dtype=np.int32))
    np.save(os.path.join(args.out_dir, "labels.npy"), np.asarray(labels, dtype=np.int64))

    band_file_sizes = [
        os.path.getsize(os.path.join(args.out_dir, f"band_{band:02d}.bin"))
        for band in range(args.num_bands)
    ]
    sidecar_files = [
        "labels.npy", "scales.npy", "offsets.npy", "lengths.npy",
        "raw_lengths.npy", "chunk_sizes.npy",
    ]
    sidecar_bytes = sum(os.path.getsize(os.path.join(args.out_dir, p)) for p in sidecar_files)
    total_store_bytes = sum(band_file_sizes) + sidecar_bytes
    raw_uint8_bytes = len(labels) * args.view_size * args.view_size * 3
    metadata = {
        "format": "quantized_fixed_dct_v1",
        "N": int(len(labels)),
        "C": 3,
        "view_size": int(args.view_size),
        "block_size": int(args.block_size),
        "nph": int(nph),
        "npw": int(npw),
        "P": int(P),
        "total_coeffs_per_patch": int(total_coeffs),
        "num_bands_per_patch": int(args.num_bands),
        "band_size": int(band_size),
        "band_boundaries": [int(x) for x in band_boundaries],
        "tile_blocks": int(args.tile_blocks),
        "num_tiles": int(len(tile_specs)),
        "tile_specs": tile_specs,
        "chunk_size": int(args.chunk_size),
        "num_chunks": int(len(chunk_sizes)),
        "dtype": "int8",
        "scale_scope": args.scale_scope,
        "scale_stat": args.scale_stat,
        "scale_percentile": float(args.scale_percentile),
        "quant_multiplier": float(args.quant_multiplier),
        "compression": "zlib",
        "compression_level": int(args.compression_level),
        "zigzag_order": zz.tolist(),
        "band_file_sizes": [int(x) for x in band_file_sizes],
        "sidecar_bytes": int(sidecar_bytes),
        "total_store_bytes": int(total_store_bytes),
        "raw_uint8_bytes_for_fixed_view": int(raw_uint8_bytes),
        "store_ratio_vs_fixed_view_uint8": float(total_store_bytes / max(raw_uint8_bytes, 1)),
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 72)
    print(f"[quant-store] Done: {args.out_dir}")
    print(f"samples={len(labels)} view={args.view_size} chunks={len(chunk_sizes)} tiles={len(tile_specs)}")
    print(f"store={total_store_bytes / 1e9:.3f}GB ratio_vs_fixed_uint8={metadata['store_ratio_vs_fixed_view_uint8']:.3f}")
    print("=" * 72)


def main():
    args = parse_args()
    build_store(args)


if __name__ == "__main__":
    main()
