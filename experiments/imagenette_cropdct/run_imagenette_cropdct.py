"""Local Imagenette experiment for CropDCT storage and fidelity.

This script intentionally lives outside the TPU/ImageNet training path. It is a
small, repeatable harness for studying the storage question on a variable-size
ImageNet-like dataset using PyTorch/torchvision only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from tdcf.cropdct_store import DEFAULT_BANDS, CropDCTStore, CropDCTWriter


@dataclass(frozen=True)
class ImageSample:
    path: str
    label: int
    source_key: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Local Imagenette CropDCT experiment")
    p.add_argument("--data_root", type=str, default="experiments/imagenette_cropdct/work/data")
    p.add_argument("--out_root", type=str, default="experiments/imagenette_cropdct/work/runs")
    p.add_argument("--size", choices=["full", "320px", "160px"], default="full")
    p.add_argument("--download", action="store_true")
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--tile_blocks", type=int, default=32)
    p.add_argument("--records_per_shard", type=int, default=1024)
    p.add_argument("--compression_level", type=int, default=1)
    p.add_argument("--max_train", type=int, default=1000,
                   help="0 means all train images.")
    p.add_argument("--max_val", type=int, default=200,
                   help="0 means all validation images.")
    p.add_argument("--fidelity_samples", type=int, default=200)
    p.add_argument("--entropy_samples", type=int, default=200)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--rebuild", action="store_true",
                   help="Delete existing train/val CropDCT stores before building.")
    return p.parse_args()


def import_imagenette():
    try:
        from torchvision.datasets import Imagenette
    except ImportError as exc:
        raise RuntimeError(
            "torchvision.datasets.Imagenette is not available in this environment. "
            "Please upgrade torchvision or manually place Imagenette images under "
            "--data_root and rerun with a torchvision version that includes Imagenette."
        ) from exc
    return Imagenette


def imagenette_extracted_dir(root: str, size: str) -> str:
    suffix = {"full": "", "320px": "-320", "160px": "-160"}[size]
    return os.path.join(root, f"imagenette2{suffix}")


def load_imagenette_samples(root: str, split: str, size: str, download: bool) -> List[ImageSample]:
    Imagenette = import_imagenette()
    if download and os.path.isdir(imagenette_extracted_dir(root, size)):
        download = False
    ds = Imagenette(root=root, split=split, size=size, download=download)
    raw_samples = getattr(ds, "samples", None)
    if raw_samples is None:
        raw_samples = getattr(ds, "_samples", None)
    if raw_samples is None:
        raise RuntimeError("Unexpected torchvision Imagenette object: missing .samples/_samples")

    samples: List[ImageSample] = []
    for path, label in raw_samples:
        # Keep source_key compact and stable across machines.
        try:
            source_key = os.path.relpath(path, root)
        except ValueError:
            source_key = os.path.basename(path)
        samples.append(ImageSample(path=str(path), label=int(label), source_key=source_key))
    if not samples:
        raise RuntimeError(f"No Imagenette samples found for split={split!r} under {root}")
    return samples


def limited(samples: List[ImageSample], max_samples: int) -> List[ImageSample]:
    if max_samples and max_samples > 0:
        return samples[:max_samples]
    return samples


def file_size_sum(paths: Iterable[str]) -> int:
    return int(sum(os.path.getsize(path) for path in paths))


def dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return int(total)


def read_rgb_tensor(path: str) -> torch.Tensor:
    with Image.open(path) as img:
        return T.ToTensor()(img.convert("RGB"))


def build_store(
    samples: List[ImageSample],
    out_dir: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    if args.rebuild and os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    metadata_path = os.path.join(out_dir, "metadata.json")
    if os.path.exists(metadata_path):
        print(f"[imagenette-cropdct] Reusing existing store: {out_dir}", flush=True)
        return {"rebuilt": 0.0, "seconds": 0.0}

    writer = CropDCTWriter(
        out_dir,
        records_per_shard=args.records_per_shard,
        block_size=8,
        tile_blocks=args.tile_blocks,
        band_specs=DEFAULT_BANDS,
        quality=args.quality,
        compression="zstd",
        compression_level=args.compression_level,
        device=device,
    )
    start = time.time()
    for idx, sample in enumerate(samples, start=1):
        writer.add(
            read_rgb_tensor(sample.path),
            sample.label,
            source_key=sample.source_key,
            source_url=sample.path,
        )
        if idx % 500 == 0 or idx == len(samples):
            elapsed = time.time() - start
            print(
                f"[imagenette-cropdct] wrote {idx}/{len(samples)} to {out_dir} "
                f"({idx / max(elapsed, 1e-6):.1f} img/s)",
                flush=True,
            )
    writer.close()
    return {"rebuilt": 1.0, "seconds": time.time() - start}


def psnr_from_mse(mse: float) -> float:
    return 120.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


def validate_fidelity(
    samples: List[ImageSample],
    store_dir: str,
    max_samples: int,
    device: torch.device,
) -> Dict[str, float]:
    n = min(max_samples, len(samples))
    if n <= 0:
        return {}

    store = CropDCTStore(store_dir, device=device)
    psnrs: List[float] = []
    maes: List[float] = []
    max_diffs: List[float] = []
    label_mismatches = 0
    shape_mismatches = 0

    for image_id in range(n):
        sample = samples[image_id]
        rec = store.image_record(image_id)
        recon, label = store.read_crop(
            image_id=image_id,
            crop_box=(0, 0, rec.height, rec.width),
            freq_bands=list(range(store.num_bands)),
            output_size=None,
        )
        ref = read_rgb_tensor(sample.path)
        pred = recon.squeeze(0).cpu()
        if int(label) != int(sample.label):
            label_mismatches += 1
        if tuple(pred.shape) != tuple(ref.shape):
            shape_mismatches += 1
            continue
        diff = (pred - ref).abs()
        mse = float(diff.pow(2).mean().item())
        psnrs.append(psnr_from_mse(mse))
        maes.append(float(diff.mean().item()))
        max_diffs.append(float(diff.max().item()))

    store.close()
    if not psnrs:
        raise RuntimeError(f"No comparable samples for fidelity validation in {store_dir}")

    def mean(xs: List[float]) -> float:
        return float(np.mean(np.asarray(xs, dtype=np.float64)))

    def std(xs: List[float]) -> float:
        return float(np.std(np.asarray(xs, dtype=np.float64)))

    return {
        "samples_requested": int(n),
        "samples_compared": int(len(psnrs)),
        "label_mismatches": int(label_mismatches),
        "shape_mismatches": int(shape_mismatches),
        "psnr_mean": mean(psnrs),
        "psnr_std": std(psnrs),
        "mae_mean": mean(maes),
        "mae_std": std(maes),
        "max_diff_mean": mean(max_diffs),
        "max_diff_std": std(max_diffs),
    }


def empirical_entropy_from_counts(counts: Dict[int, int]) -> float:
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return float(entropy)


def update_counts(counts: Dict[int, int], arr: np.ndarray) -> None:
    values, freqs = np.unique(arr.reshape(-1), return_counts=True)
    for value, freq in zip(values, freqs):
        counts[int(value)] += int(freq)


def entropy_report(store_dir: str, max_images: int) -> Dict[str, object]:
    store = CropDCTStore(store_dir, device=torch.device("cpu"))
    n = min(max_images, len(store.global_index))
    band_counts: List[Dict[int, int]] = [defaultdict(int) for _ in range(store.num_bands)]
    band_payload_bytes = [0 for _ in range(store.num_bands)]
    band_raw_bytes = [0 for _ in range(store.num_bands)]
    band_symbols = [0 for _ in range(store.num_bands)]
    band_zero_symbols = [0 for _ in range(store.num_bands)]

    for image_id in range(n):
        image = store.image_record(image_id)
        shard = store.shards[image.shard_id]
        for tile_id in range(image.num_tiles):
            for band_id, (b0, b1) in enumerate(store.band_specs):
                chunk_idx = image.chunk_base + tile_id * store.num_bands + band_id
                meta = shard.chunks[chunk_idx]
                raw = shard.read_chunk(int(chunk_idx))
                arr = np.frombuffer(raw, dtype=np.int16)
                update_counts(band_counts[band_id], arr)
                band_payload_bytes[band_id] += int(meta["length"])
                band_raw_bytes[band_id] += int(meta["raw_length"])
                band_symbols[band_id] += int(arr.size)
                band_zero_symbols[band_id] += int(np.count_nonzero(arr == 0))
        if (image_id + 1) % 100 == 0 or image_id + 1 == n:
            print(f"[imagenette-cropdct] entropy scanned {image_id + 1}/{n} images", flush=True)

    bands = []
    total_entropy_bytes = 0.0
    for band_id, counts in enumerate(band_counts):
        entropy_bits = empirical_entropy_from_counts(counts)
        lower_bound_bytes = entropy_bits * band_symbols[band_id] / 8.0
        total_entropy_bytes += lower_bound_bytes
        bands.append(
            {
                "band_id": band_id,
                "coeff_range": list(store.band_specs[band_id]),
                "symbols": int(band_symbols[band_id]),
                "zero_fraction": 0.0 if band_symbols[band_id] == 0 else band_zero_symbols[band_id] / band_symbols[band_id],
                "entropy_bits_per_symbol": entropy_bits,
                "entropy_lower_bound_bytes": lower_bound_bytes,
                "raw_int16_bytes": int(band_raw_bytes[band_id]),
                "compressed_payload_bytes": int(band_payload_bytes[band_id]),
            }
        )

    store.close()
    return {
        "images_scanned": int(n),
        "bands": bands,
        "entropy_lower_bound_bytes_total": float(total_entropy_bytes),
        "compressed_payload_bytes_total": int(sum(band_payload_bytes)),
        "raw_int16_bytes_total": int(sum(band_raw_bytes)),
    }


def summarize_split(
    split: str,
    samples: List[ImageSample],
    store_dir: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    jpeg_bytes = file_size_sum(sample.path for sample in samples)
    store_bytes = dir_size(store_dir)
    fidelity = validate_fidelity(samples, store_dir, args.fidelity_samples, device)
    entropy = entropy_report(store_dir, min(args.entropy_samples, len(samples)))
    return {
        "split": split,
        "num_samples": len(samples),
        "original_jpeg_bytes": jpeg_bytes,
        "cropdct_store_bytes": store_bytes,
        "storage_ratio_cropdct_over_jpeg": store_bytes / max(jpeg_bytes, 1),
        "fidelity": fidelity,
        "entropy": entropy,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_root = Path(args.out_root)
    run_name = f"imagenette_{args.size}_q{args.quality}_tb{args.tile_blocks}"
    run_root = out_root / run_name
    train_store = run_root / "train"
    val_store = run_root / "val"
    run_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Imagenette CropDCT local experiment")
    print(f"data_root={args.data_root}")
    print(f"run_root={run_root}")
    print(f"size={args.size} quality={args.quality} tile_blocks={args.tile_blocks}")
    print(f"max_train={args.max_train} max_val={args.max_val}")
    print("=" * 72)

    train_all = load_imagenette_samples(args.data_root, "train", args.size, args.download)
    val_all = load_imagenette_samples(args.data_root, "val", args.size, args.download)
    train_samples = limited(train_all, args.max_train)
    val_samples = limited(val_all, args.max_val)

    build_train = build_store(train_samples, str(train_store), args, device)
    build_val = build_store(val_samples, str(val_store), args, device)

    train_report = summarize_split("train", train_samples, str(train_store), args, device)
    val_report = summarize_split("val", val_samples, str(val_store), args, device)
    report = {
        "args": vars(args),
        "run_root": str(run_root),
        "build": {
            "train": build_train,
            "val": build_val,
        },
        "train": train_report,
        "val": val_report,
    }

    report_path = run_root / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 72)
    print("Imagenette CropDCT report")
    for split_report in (train_report, val_report):
        fidelity = split_report["fidelity"]
        entropy = split_report["entropy"]
        print(
            f"{split_report['split']}: samples={split_report['num_samples']} "
            f"jpeg={split_report['original_jpeg_bytes'] / 1e9:.4f}GB "
            f"cropdct={split_report['cropdct_store_bytes'] / 1e9:.4f}GB "
            f"ratio={split_report['storage_ratio_cropdct_over_jpeg']:.3f} "
            f"PSNR={fidelity['psnr_mean']:.2f}±{fidelity['psnr_std']:.2f}dB "
            f"MAE={fidelity['mae_mean']:.6f} "
            f"MAX={fidelity['max_diff_mean']:.6f}"
        )
        print(
            f"  entropy_lb={entropy['entropy_lower_bound_bytes_total'] / 1e9:.4f}GB "
            f"payload={entropy['compressed_payload_bytes_total'] / 1e9:.4f}GB "
            f"raw_int16={entropy['raw_int16_bytes_total'] / 1e9:.4f}GB "
            f"scanned={entropy['images_scanned']}"
        )
    print(f"report={report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
