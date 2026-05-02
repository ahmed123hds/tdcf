"""Validate CropDCT reconstruction fidelity against source WebDataset images."""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import webdataset as wds

from tdcf.cropdct_store import CropDCTStore


def expand_shards(pattern: str) -> List[str]:
    urls: List[str] = []
    for item in (x.strip() for x in pattern.split(",")):
        if not item:
            continue
        m = re.search(r"\{(\d+)\.\.(\d+)\}", item)
        if m is None:
            urls.append(item)
            continue
        start_s, end_s = m.group(1), m.group(2)
        width = max(len(start_s), len(end_s))
        start, end = int(start_s), int(end_s)
        if end < start:
            raise ValueError(f"Invalid shard range: {item}")
        for i in range(start, end + 1):
            urls.append(item[:m.start()] + f"{i:0{width}d}" + item[m.end():])
    return urls


def iter_reference_samples(shards: str):
    urls = expand_shards(shards)
    if not urls:
        raise ValueError(f"No shards expanded from {shards!r}")
    if not os.path.exists(urls[0]):
        raise FileNotFoundError(f"First shard does not exist: {urls[0]}")
    dataset = (
        wds.WebDataset(urls, resampled=False, shardshuffle=False)
        .decode("pil")
        .to_tuple("__key__", "jpg;jpeg;png", "cls")
    )
    to_tensor = T.ToTensor()
    for key, img, label in dataset:
        yield str(key), to_tensor(img.convert("RGB")), int(label)


def load_reference_by_key(shards: str, needed_keys: Set[str]) -> Dict[str, Tuple[torch.Tensor, int]]:
    refs: Dict[str, Tuple[torch.Tensor, int]] = {}
    for key, img, label in iter_reference_samples(shards):
        if key in needed_keys:
            refs[key] = (img, label)
            if len(refs) == len(needed_keys):
                break
    missing = needed_keys.difference(refs)
    if missing:
        preview = ", ".join(sorted(list(missing))[:5])
        raise RuntimeError(f"Missing {len(missing)} source keys from WebDataset shards. Examples: {preview}")
    return refs


def collect_needed_keys(store_dirs: Iterable[str], samples: int) -> Set[str]:
    needed: Set[str] = set()
    for store_dir in store_dirs:
        store = CropDCTStore(store_dir, device=torch.device("cpu"))
        for image_id in range(min(samples, len(store.global_index))):
            rec = store.image_record(image_id)
            shard = store.shards[rec.shard_id]
            img = shard.images[rec.row]
            if "source_key" not in img.dtype.names:
                store.close()
                raise RuntimeError(
                    f"{store_dir} does not contain source_key metadata. "
                    "Rebuild the random subset with the latest code before running fidelity validation."
                )
            needed.add(str(img["source_key"]))
        store.close()
    return needed


def validate_store(store_dir: str, refs: Dict[str, Tuple[torch.Tensor, int]], samples: int) -> Dict[str, float]:
    store = CropDCTStore(store_dir, device=torch.device("cpu"))
    psnrs: List[float] = []
    maes: List[float] = []
    max_diffs: List[float] = []
    mismatches = 0
    shape_mismatches = 0

    for image_id in range(samples):
        rec = store.image_record(image_id)
        shard = store.shards[rec.shard_id]
        img_meta = shard.images[rec.row]
        source_key = str(img_meta["source_key"])
        ref, label_ref = refs[source_key]
        crop_dct, label_dct = store.read_crop(
            image_id=image_id,
            crop_box=(0, 0, rec.height, rec.width),
            freq_bands=list(range(store.num_bands)),
            output_size=None,
        )
        pred = crop_dct.squeeze(0).cpu()
        if int(label_dct) != int(label_ref):
            mismatches += 1
        if tuple(pred.shape) != tuple(ref.shape):
            shape_mismatches += 1
            continue
        diff = (pred - ref).abs()
        mse = diff.pow(2).mean().clamp_min(1e-12)
        psnrs.append(float(10.0 * torch.log10(1.0 / mse).item()))
        maes.append(float(diff.mean().item()))
        max_diffs.append(float(diff.max().item()))

    store.close()
    if not psnrs:
        raise RuntimeError(f"No comparable samples for {store_dir}; shape_mismatches={shape_mismatches}")

    def mean(values: List[float]) -> float:
        return float(np.mean(np.asarray(values, dtype=np.float64)))

    def std(values: List[float]) -> float:
        return float(np.std(np.asarray(values, dtype=np.float64)))

    return {
        "samples": float(samples),
        "compared": float(len(psnrs)),
        "label_mismatches": float(mismatches),
        "shape_mismatches": float(shape_mismatches),
        "psnr_mean": mean(psnrs),
        "psnr_std": std(psnrs),
        "mae_mean": mean(maes),
        "mae_std": std(maes),
        "max_diff_mean": mean(max_diffs),
        "max_diff_std": std(max_diffs),
    }


def parse_args():
    p = argparse.ArgumentParser("CropDCT fidelity validator")
    p.add_argument("--store_dirs", type=str, required=True,
                   help="Comma-separated train store dirs to validate.")
    p.add_argument("--shards", type=str, required=True,
                   help="Original WebDataset shard pattern matching store order.")
    p.add_argument("--samples", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    store_dirs = [x.strip() for x in args.store_dirs.split(",") if x.strip()]
    if not store_dirs:
        raise ValueError("--store_dirs is empty")
    needed_keys = collect_needed_keys(store_dirs, args.samples)
    print(f"[cropdct-fidelity] loading {len(needed_keys)} reference images by source_key", flush=True)
    refs = load_reference_by_key(args.shards, needed_keys)
    rows = []
    for store_dir in store_dirs:
        stats = validate_store(store_dir, refs, args.samples)
        rows.append(stats)
        print(
            f"[cropdct-fidelity] store={store_dir} compared={int(stats['compared'])}/{args.samples} "
            f"label_mismatch={int(stats['label_mismatches'])} shape_mismatch={int(stats['shape_mismatches'])} "
            f"PSNR={stats['psnr_mean']:.2f}±{stats['psnr_std']:.2f}dB "
            f"MAE={stats['mae_mean']:.6f}±{stats['mae_std']:.6f} "
            f"MAX={stats['max_diff_mean']:.6f}±{stats['max_diff_std']:.6f}",
            flush=True,
        )

    keys = ("psnr_mean", "mae_mean", "max_diff_mean")
    print("[cropdct-fidelity] aggregate across stores")
    for key in keys:
        values = [row[key] for row in rows]
        print(f"  {key}: {np.mean(values):.6f} ± {np.std(values):.6f}")
    total_labels = sum(row["label_mismatches"] for row in rows)
    total_shapes = sum(row["shape_mismatches"] for row in rows)
    print(f"  total_label_mismatches: {int(total_labels)}")
    print(f"  total_shape_mismatches: {int(total_shapes)}")


if __name__ == "__main__":
    main()
