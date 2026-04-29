"""Prepare a fast sequential int8 DCT store for ImageNet-1K."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF

from tdcf.transforms import block_dct2d, zigzag_order


def parse_args():
    p = argparse.ArgumentParser("Prepare fast sharded ImageNet DCT store")
    p.add_argument("--source", choices=["webdataset", "synthetic"], default="webdataset")
    p.add_argument("--shards", type=str, default=None)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--view_size", type=int, default=224)
    p.add_argument("--block_size", type=int, default=8)
    p.add_argument("--num_bands", type=int, default=16)
    p.add_argument("--records_per_shard", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--calibration_samples", type=int, default=4096)
    p.add_argument("--scale_scope", choices=["band", "channel_band"], default="band")
    p.add_argument("--scale_stat", choices=["max", "percentile"], default="percentile")
    p.add_argument("--scale_percentile", type=float, default=99.99)
    p.add_argument("--quant_multiplier", type=float, default=1.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def identity_label(label):
    return int(label)


def _expand_shard_range(pattern: str):
    m = re.search(r"\{(\d+)\.\.(\d+)\}", pattern)
    if not m:
        return [pattern]
    start_s, end_s = m.group(1), m.group(2)
    width = max(len(start_s), len(end_s))
    start, end = int(start_s), int(end_s)
    if end < start:
        raise ValueError(f"Invalid shard brace range: {pattern}")
    return [
        pattern[:m.start()] + f"{i:0{width}d}" + pattern[m.end():]
        for i in range(start, end + 1)
    ]


def expand_shards(pattern: str):
    urls = []
    for item in (x.strip() for x in pattern.split(",")):
        if item:
            urls.extend(_expand_shard_range(item))
    return urls


def validate_expanded_shards(raw_pattern: str, urls: list[str]):
    if not urls:
        raise ValueError(f"No shards expanded from pattern: {raw_pattern!r}")
    bad = [u for u in urls if "{" in u or "}" in u]
    if bad:
        raise ValueError(
            "Shard pattern still contains braces after expansion. "
            f"raw={raw_pattern!r} first_bad={bad[0]!r}. "
            "Expected format like imagenet1k-train-{0000..1023}.tar"
        )
    if not os.path.exists(urls[0]):
        raise FileNotFoundError(
            f"First expanded shard does not exist: {urls[0]!r}. "
            f"raw pattern was: {raw_pattern!r}"
        )


def fixed_view(img, view_size: int):
    if not torch.is_tensor(img):
        img = img.convert("RGB")
        w, h = img.size
        scale = view_size / min(h, w)
        new_h = max(view_size, int(round(h * scale)))
        new_w = max(view_size, int(round(w * scale)))
        img = TF.resize(img, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        img = TF.center_crop(img, [view_size, view_size])
        return TF.to_tensor(img)
    x = img.float()
    if x.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got {tuple(x.shape)}")
    x = TF.resize(x, [view_size, view_size], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
    return x.clamp(0.0, 1.0)


def build_loader(args):
    if args.source == "synthetic":
        n = max(args.max_samples, args.calibration_samples, 64)
        return [(torch.rand(3, args.view_size, args.view_size), i % 1000) for i in range(n)]

    if args.shards is None:
        raise ValueError("--shards is required for webdataset")
    import webdataset as wds

    def pre_transform(img):
        return fixed_view(img, args.view_size)

    urls = expand_shards(args.shards)
    validate_expanded_shards(args.shards, urls)
    print(
        f"[fast-quant] shards expanded: count={len(urls)} "
        f"first={urls[0]} last={urls[-1]}",
        flush=True,
    )
    dataset = (
        wds.WebDataset(urls, resampled=False, shardshuffle=False)
        .decode("pil")
        .to_tuple("jpg;jpeg;png", "cls")
        .map_tuple(pre_transform, identity_label)
    )
    return wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )


def robust_abs_stat(vals: torch.Tensor, args):
    vals = vals.reshape(-1).abs().float()
    if vals.numel() == 0:
        return 1e-12
    if args.scale_stat == "max":
        return float(vals.max().item())
    q = min(max(args.scale_percentile / 100.0, 0.0), 1.0)
    return float(torch.quantile(vals, q).item())


@torch.no_grad()
def calibrate(args, device):
    total_coeffs = args.block_size * args.block_size
    band_size = total_coeffs // args.num_bands
    zz = zigzag_order(args.block_size, args.block_size)
    shape = (3, args.num_bands) if args.scale_scope == "channel_band" else (1, args.num_bands)
    stats = torch.zeros(shape, dtype=torch.float32)
    batch = []
    seen = 0

    def flush():
        nonlocal batch
        if not batch:
            return
        x = torch.stack(batch).to(device)
        coeffs = block_dct2d(x, block_size=args.block_size)
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

    print(f"[fast-quant] Calibrating on {args.calibration_samples} samples", flush=True)
    for img, _label in build_loader(args):
        batch.append(img.float())
        seen += 1
        if len(batch) >= args.batch_size:
            flush()
        if seen >= args.calibration_samples:
            break
    flush()
    scales = (stats / 127.0).clamp_min(1e-12) * float(args.quant_multiplier)
    print(f"[fast-quant] scale min={scales.min().item():.6g} max={scales.max().item():.6g}", flush=True)
    return scales.cpu().numpy().astype(np.float32)


def open_shard(args, shard_id: int):
    shard_dir = os.path.join(args.out_dir, f"shard_{shard_id:05d}")
    os.makedirs(shard_dir, exist_ok=True)
    nph = args.view_size // args.block_size
    npw = args.view_size // args.block_size
    P = nph * npw
    band_size = args.block_size * args.block_size // args.num_bands
    arrays = []
    for band in range(args.num_bands):
        path = os.path.join(shard_dir, f"band_{band:02d}.npy")
        arr = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.int8,
            shape=(args.records_per_shard, P, 3, band_size),
        )
        arrays.append(arr)
    return shard_dir, arrays


@torch.no_grad()
def write_batch(args, device, arrays, local_start, images, scales):
    total_coeffs = args.block_size * args.block_size
    band_size = total_coeffs // args.num_bands
    zz = zigzag_order(args.block_size, args.block_size).to(device)
    x = torch.stack(images).to(device)
    coeffs = block_dct2d(x, block_size=args.block_size)
    flat = coeffs.reshape(coeffs.shape[0], -1, 3, total_coeffs)[:, :, :, zz]
    flat = flat.detach().cpu().numpy()
    B = flat.shape[0]
    for band in range(args.num_bands):
        start = band * band_size
        end = (band + 1) * band_size if band < args.num_bands - 1 else total_coeffs
        data = flat[:, :, :, start:end]
        if scales.shape[0] == 1:
            q = np.rint(data / scales[0, band])
        else:
            q = np.rint(data / scales[:, band].reshape(1, 1, 3, 1))
        arrays[band][local_start:local_start + B] = np.clip(q, -128, 127).astype(np.int8, copy=False)


def build_store(args):
    if args.view_size % args.block_size != 0:
        raise ValueError("--view_size must be divisible by --block_size")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    scales = calibrate(args, device)
    np.save(os.path.join(args.out_dir, "scales.npy"), scales)

    labels = []
    shards = []
    shard_id = 0
    shard_dir, arrays = open_shard(args, shard_id)
    local = 0
    pending_images = []
    start_time = time.time()

    def flush_pending():
        nonlocal pending_images, local, shard_id, shard_dir, arrays
        while pending_images:
            room = args.records_per_shard - local
            take = min(room, len(pending_images))
            write_batch(args, device, arrays, local, pending_images[:take], scales)
            local += take
            pending_images = pending_images[take:]
            if local == args.records_per_shard:
                for arr in arrays:
                    arr.flush()
                shards.append({"shard_id": shard_id, "dir": os.path.basename(shard_dir), "count": local})
                shard_id += 1
                shard_dir, arrays = open_shard(args, shard_id)
                local = 0

    for idx, (img, label) in enumerate(build_loader(args), start=1):
        pending_images.append(img.float())
        labels.append(int(label))
        if len(pending_images) >= args.batch_size:
            flush_pending()
        if idx % 10000 == 0:
            elapsed = time.time() - start_time
            print(f"[fast-quant] wrote {idx} samples ({idx / max(elapsed, 1e-6):.1f} img/s)", flush=True)
        if args.max_samples > 0 and idx >= args.max_samples:
            break
    flush_pending()
    if local > 0:
        for arr in arrays:
            arr.flush()
        shards.append({"shard_id": shard_id, "dir": os.path.basename(shard_dir), "count": local})

    labels_arr = np.asarray(labels, dtype=np.int64)
    np.save(os.path.join(args.out_dir, "labels.npy"), labels_arr)
    nph = args.view_size // args.block_size
    npw = args.view_size // args.block_size
    total_coeffs = args.block_size * args.block_size
    meta = {
        "format": "fast_quantized_dct_v1",
        "N": int(len(labels_arr)),
        "C": 3,
        "view_size": int(args.view_size),
        "block_size": int(args.block_size),
        "nph": int(nph),
        "npw": int(npw),
        "num_bands": int(args.num_bands),
        "band_size": int(total_coeffs // args.num_bands),
        "total_coeffs": int(total_coeffs),
        "records_per_shard": int(args.records_per_shard),
        "dtype": "int8",
        "compression": "none",
        "scale_scope": args.scale_scope,
        "zigzag_order": zigzag_order(args.block_size, args.block_size).tolist(),
        "shards": shards,
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    raw_gb = len(labels_arr) * args.view_size * args.view_size * 3 / (1024 ** 3)
    print(f"[fast-quant] Done: {args.out_dir} samples={len(labels_arr)} raw_int8={raw_gb:.2f} GiB", flush=True)


if __name__ == "__main__":
    build_store(parse_args())
