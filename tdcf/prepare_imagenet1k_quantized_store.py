"""Prepare an original-size compressed int8 block-DCT ImageNet store."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import datetime
import zlib
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms.functional as TF

from tdcf.quantized_store import bucket_key_from_shape, build_tile_specs
from tdcf.transforms import block_dct2d, zigzag_order


def parse_args():
    p = argparse.ArgumentParser("Prepare original-size quantized ImageNet DCT store")
    p.add_argument("--shards", type=str, default=None)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--source", choices=["webdataset", "synthetic"], default="webdataset")
    p.add_argument("--block_size", type=int, default=8)
    p.add_argument("--num_bands", type=int, default=16)
    p.add_argument("--tile_blocks", type=int, default=4)
    p.add_argument("--chunk_size", type=int, default=64)
    p.add_argument("--bucket_count", type=int, default=16,
                   help="Coarse area/aspect buckets. Use powers of two: 8, 16, 32, 64.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--calibration_samples", type=int, default=4096)
    p.add_argument("--scale_scope", choices=["band", "channel_band"], default="band")
    p.add_argument("--scale_stat", choices=["max", "percentile"], default="max")
    p.add_argument("--scale_percentile", type=float, default=99.99)
    p.add_argument("--quant_multiplier", type=float, default=1.05)
    p.add_argument("--compression_level", type=int, default=6)
    p.add_argument("--resize_shorter", type=int, default=256,
                   help="Resize shorter edge to this size. Essential to prevent OOM on massive ImageNet panoramas.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def identity_label(label):
    return int(label)


def build_loader(args):
    if args.source == "synthetic":
        samples = []
        n = max(args.max_samples, args.calibration_samples, 32)
        for i in range(n):
            h = 180 + (i % 5) * 20
            w = 220 + (i % 7) * 16
            img = torch.rand(3, h, w)
            samples.append((img, i % 1000))
        return samples

    def pre_transforms(img):
        if args.resize_shorter > 0:
            img = TF.resize(img, args.resize_shorter)
        return TF.to_tensor(img)

    if args.shards is None:
        raise ValueError("--shards is required for webdataset source")
    import webdataset as wds
    dataset = (
        wds.WebDataset(args.shards, resampled=False, shardshuffle=False)
        .decode("pil")
        .to_tuple("jpg;jpeg;png", "cls")
        .map_tuple(pre_transforms, identity_label)
    )
    return wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )


def pad_to_block(img: torch.Tensor, block_size: int):
    h, w = img.shape[-2:]
    hp = math.ceil(h / block_size) * block_size
    wp = math.ceil(w / block_size) * block_size
    if hp == h and wp == w:
        return img
    out = torch.zeros((img.shape[0], hp, wp), dtype=img.dtype)
    out[:, :h, :w] = img
    return out


def robust_abs_stat(vals: torch.Tensor, args):
    vals = vals.reshape(-1).abs().float()
    if vals.numel() == 0:
        return 1e-12
    if args.scale_stat == "max":
        return float(vals.max().item())
    return float(torch.quantile(vals, min(max(args.scale_percentile / 100.0, 0.0), 1.0)).item())


@torch.no_grad()
def calibrate(args, device):
    qmax = 127.0
    total_coeffs = args.block_size * args.block_size
    band_size = total_coeffs // args.num_bands
    zz = zigzag_order(args.block_size, args.block_size)
    shape = (3, args.num_bands) if args.scale_scope == "channel_band" else (1, args.num_bands)
    stats = torch.zeros(shape, dtype=torch.float32)
    seen = 0
    batch = []

    def flush():
        nonlocal batch
        if not batch:
            return
        max_h = max(x.shape[-2] for x in batch)
        max_w = max(x.shape[-1] for x in batch)
        tensor = torch.zeros((len(batch), 3, max_h, max_w), device=device)
        for i, img in enumerate(batch):
            tensor[i, :, :img.shape[-2], :img.shape[-1]] = img.to(device)
        coeffs = block_dct2d(tensor, block_size=args.block_size)
        flat = coeffs.reshape(coeffs.shape[0], -1, 3, total_coeffs)[:, :, :, zz]
        for band in range(args.num_bands):
            start = band * band_size
            end = (band + 1) * band_size if band < args.num_bands - 1 else total_coeffs
            if args.scale_scope == "channel_band":
                for c in range(3):
                    stats[c, band] = max(float(stats[c, band]), robust_abs_stat(flat[:, :, c, start:end], args))
            else:
                stats[0, band] = max(float(stats[0, band]), robust_abs_stat(flat[:, :, :, start:end], args))
        batch = []

    print(f"[orig-quant] Calibrating on {args.calibration_samples} samples", flush=True)
    for img, _label in build_loader(args):
        batch.append(pad_to_block(img.float(), args.block_size))
        seen += 1
        if len(batch) >= args.batch_size:
            flush()
        if seen >= args.calibration_samples:
            break
    flush()
    scales = (stats / qmax).clamp_min(1e-12) * float(args.quant_multiplier)
    print(f"[orig-quant] scale min={scales.min().item():.6g} max={scales.max().item():.6g}", flush=True)
    return scales.cpu().numpy().astype(np.float32)


def scan(args):
    shapes, labels = [], []
    start_time = time.time()
    for idx, (img, label) in enumerate(build_loader(args), start=1):
        h, w = img.shape[-2:]
        shapes.append((h, w))
        labels.append(int(label))
        if idx % 10000 == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed
            print(f"[orig-quant] scan {idx} samples ({rate:.1f} img/s)", flush=True)
        if args.max_samples > 0 and idx >= args.max_samples:
            break
    return np.asarray(shapes, dtype=np.int32), np.asarray(labels, dtype=np.int64)


def _bucket_factorization(bucket_count: int):
    if bucket_count <= 0 or bucket_count & (bucket_count - 1):
        raise ValueError("--bucket_count must be a power of two")
    ratio_bins = min(8, max(2, int(round(math.sqrt(bucket_count)))))
    while bucket_count % ratio_bins != 0:
        ratio_bins //= 2
    area_bins = bucket_count // ratio_bins
    return area_bins, ratio_bins


def assign_coarse_buckets(shapes: np.ndarray, block_size: int, bucket_count: int):
    grids_h = np.ceil(shapes[:, 0] / block_size).astype(np.int32)
    grids_w = np.ceil(shapes[:, 1] / block_size).astype(np.int32)
    log_area = np.log2(np.maximum(grids_h * grids_w, 1))
    log_ratio = np.log2(np.maximum(grids_h, 1) / np.maximum(grids_w, 1))
    area_bins, ratio_bins = _bucket_factorization(bucket_count)

    area_edges = np.quantile(log_area, np.linspace(0, 1, area_bins + 1)[1:-1])
    ratio_edges = np.quantile(log_ratio, np.linspace(0, 1, ratio_bins + 1)[1:-1])
    area_id = np.searchsorted(area_edges, log_area, side="right")
    ratio_id = np.searchsorted(ratio_edges, log_ratio, side="right")
    raw_ids = (area_id * ratio_bins + ratio_id).astype(np.int32)

    unique_raw = sorted(np.unique(raw_ids).tolist())
    remap = {raw: i for i, raw in enumerate(unique_raw)}
    bucket_ids = np.asarray([remap[int(x)] for x in raw_ids], dtype=np.int32)

    bucket_shapes = []
    for bucket_id in range(len(unique_raw)):
        ids = np.flatnonzero(bucket_ids == bucket_id)
        bucket_shapes.append((int(grids_h[ids].max()), int(grids_w[ids].max())))
    return bucket_ids, bucket_shapes, {
        "requested_bucket_count": int(bucket_count),
        "actual_bucket_count": int(len(bucket_shapes)),
        "area_bins": int(area_bins),
        "ratio_bins": int(ratio_bins),
        "area_edges": area_edges.tolist(),
        "ratio_edges": ratio_edges.tolist(),
    }


def init_bucket(args, bucket_id, key, global_ids, shapes, labels):
    nph, npw = key
    P = nph * npw
    total_coeffs = args.block_size * args.block_size
    band_size = total_coeffs // args.num_bands
    bounds = [b * band_size for b in range(args.num_bands + 1)]
    bounds[-1] = total_coeffs
    zz = zigzag_order(args.block_size, args.block_size).numpy()
    specs = build_tile_specs(nph, npw, args.tile_blocks)
    bucket_dir = os.path.join(args.out_dir, f"bucket_{bucket_id:03d}")
    os.makedirs(bucket_dir, exist_ok=True)

    np.save(os.path.join(bucket_dir, "sample_ids.npy"), global_ids.astype(np.int64))
    np.save(os.path.join(bucket_dir, "sample_shapes.npy"), shapes[global_ids])
    np.save(os.path.join(bucket_dir, "labels.npy"), labels[global_ids])
    patch_signal = np.lib.format.open_memmap(
        os.path.join(bucket_dir, "patch_signal.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(len(global_ids), P),
    )
    band_files = [open(os.path.join(bucket_dir, f"band_{b:02d}.bin"), "wb") for b in range(args.num_bands)]
    return {
        "bucket_id": int(bucket_id),
        "dir": f"bucket_{bucket_id:03d}",
        "nph": int(nph),
        "npw": int(npw),
        "num_samples": int(len(global_ids)),
        "_runtime": {
            "P": P,
            "bounds": bounds,
            "zz": zz,
            "specs": specs,
            "band_files": band_files,
            "patch_signal": patch_signal,
            "offsets": [],
            "lengths": [],
            "raw_lengths": [],
            "chunk_sizes": [],
            "pending_imgs": [],
            "pending_local": [],
        },
    }


def flush_bucket(args, bucket, scales, device):
    rt = bucket["_runtime"]
    if not rt["pending_imgs"]:
        return
    B = len(rt["pending_imgs"])
    canvas_h = bucket["nph"] * args.block_size
    canvas_w = bucket["npw"] * args.block_size
    total_coeffs = args.block_size * args.block_size
    batch = torch.zeros((B, 3, canvas_h, canvas_w), device=device)
    for i, img in enumerate(rt["pending_imgs"]):
        batch[i, :, :img.shape[-2], :img.shape[-1]] = img.to(device)
    coeffs = block_dct2d(batch, block_size=args.block_size)
    flat = coeffs.reshape(B, rt["P"], 3, total_coeffs).detach().cpu().numpy()
    zz_coeffs = flat[:, :, :, rt["zz"]]
    sig = np.abs(flat).mean(axis=(2, 3)).astype(np.float32)
    local_pos = np.asarray(rt["pending_local"], dtype=np.int64)
    rt["patch_signal"][local_pos] = sig

    chunk_offsets = np.zeros((len(rt["specs"]), args.num_bands), dtype=np.int64)
    chunk_lengths = np.zeros((len(rt["specs"]), args.num_bands), dtype=np.int32)
    chunk_raw = np.zeros((len(rt["specs"]), args.num_bands), dtype=np.int32)
    for spec in rt["specs"]:
        tile_index = int(spec["tile_index"])
        block_ids = np.asarray(spec["block_ids"], dtype=np.int64)
        tile = zz_coeffs[:, block_ids]
        for band in range(args.num_bands):
            start, end = rt["bounds"][band], rt["bounds"][band + 1]
            data = tile[:, :, :, start:end]
            if scales.shape[0] == 1:
                q = np.rint(data / scales[0, band])
            else:
                q = np.rint(data / scales[:, band].reshape(1, 1, 3, 1))
            q = np.clip(q, -128, 127).astype(np.int8, copy=False)
            raw = np.ascontiguousarray(q).tobytes()
            payload = zlib.compress(raw, level=args.compression_level)
            chunk_offsets[tile_index, band] = rt["band_files"][band].tell()
            rt["band_files"][band].write(payload)
            chunk_lengths[tile_index, band] = len(payload)
            chunk_raw[tile_index, band] = len(raw)
    rt["offsets"].append(chunk_offsets)
    rt["lengths"].append(chunk_lengths)
    rt["raw_lengths"].append(chunk_raw)
    rt["chunk_sizes"].append(B)
    rt["pending_imgs"], rt["pending_local"] = [], []


def finalize_bucket(args, bucket):
    rt = bucket.pop("_runtime")
    bucket_dir = os.path.join(args.out_dir, bucket["dir"])
    for f in rt["band_files"]:
        f.close()
    rt["patch_signal"].flush()
    np.save(os.path.join(bucket_dir, "offsets.npy"), np.stack(rt["offsets"]))
    np.save(os.path.join(bucket_dir, "lengths.npy"), np.stack(rt["lengths"]))
    np.save(os.path.join(bucket_dir, "raw_lengths.npy"), np.stack(rt["raw_lengths"]))
    np.save(os.path.join(bucket_dir, "chunk_sizes.npy"), np.asarray(rt["chunk_sizes"], dtype=np.int32))


def build_store(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    scales = calibrate(args, device)
    np.save(os.path.join(args.out_dir, "scales.npy"), scales)
    shapes, labels = scan(args)
    bucket_ids, bucket_shapes, bucket_meta = assign_coarse_buckets(
        shapes, args.block_size, args.bucket_count
    )
    positions = np.empty(len(labels), dtype=np.int32)
    for bid in range(len(bucket_shapes)):
        ids = np.flatnonzero(bucket_ids == bid)
        positions[ids] = np.arange(len(ids), dtype=np.int32)
    np.save(os.path.join(args.out_dir, "labels.npy"), labels)
    np.save(os.path.join(args.out_dir, "sample_bucket_ids.npy"), bucket_ids)
    np.save(os.path.join(args.out_dir, "sample_bucket_positions.npy"), positions)

    buckets = []
    for bid, key in enumerate(bucket_shapes):
        ids = np.flatnonzero(bucket_ids == bid).astype(np.int64)
        print(f"[orig-quant] init bucket {bid+1}/{len(bucket_shapes)} key={key} samples={len(ids)}", flush=True)
        buckets.append(init_bucket(args, bid, key, ids, shapes, labels))

    local_counters = np.zeros(len(bucket_shapes), dtype=np.int64)
    total_samples = len(labels)
    start_time = time.time()
    for global_idx, (img, _label) in enumerate(build_loader(args)):
        if args.max_samples > 0 and global_idx >= args.max_samples:
            break
        bid = int(bucket_ids[global_idx])
        rt = buckets[bid]["_runtime"]
        rt["pending_imgs"].append(pad_to_block(img.float(), args.block_size))
        rt["pending_local"].append(int(local_counters[bid]))
        local_counters[bid] += 1
        if len(rt["pending_imgs"]) >= args.chunk_size:
            flush_bucket(args, buckets[bid], scales, device)
        if (global_idx + 1) % 10000 == 0:
            elapsed = time.time() - start_time
            rate = (global_idx + 1) / elapsed
            rem_sec = (total_samples - (global_idx + 1)) / rate
            eta = str(datetime.timedelta(seconds=int(rem_sec)))
            print(f"[orig-quant] wrote {global_idx + 1}/{total_samples} samples ({rate:.1f} img/s) | ETA: {eta}", flush=True)

    for bucket in buckets:
        flush_bucket(args, bucket, scales, device)
        finalize_bucket(args, bucket)

    total_coeffs = args.block_size * args.block_size
    band_size = total_coeffs // args.num_bands
    meta = {
        "format": "original_quantized_dct_v1",
        "N": int(len(labels)),
        "C": 3,
        "block_size": int(args.block_size),
        "num_bands_per_patch": int(args.num_bands),
        "band_size": int(band_size),
        "total_coeffs_per_patch": int(total_coeffs),
        "tile_blocks": int(args.tile_blocks),
        "chunk_size": int(args.chunk_size),
        "dtype": "int8",
        "scale_scope": args.scale_scope,
        "quant_multiplier": float(args.quant_multiplier),
        "compression": "zlib",
        "compression_level": int(args.compression_level),
        "coarse_bucket_metadata": bucket_meta,
        "zigzag_order": zigzag_order(args.block_size, args.block_size).numpy().tolist(),
        "buckets": buckets,
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[orig-quant] Done: {args.out_dir} samples={len(labels)} buckets={len(buckets)}", flush=True)


if __name__ == "__main__":
    build_store(parse_args())
