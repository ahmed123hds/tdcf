"""
Build a bucketed, tiled, band-grouped ImageNet-1K DCT store.

The implementation supports both:
  1. streaming WebDataset shards (real ImageNet-1K prep)
  2. in-memory sample lists (local smoke tests)
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF

from tdcf.bucketed_store import bucket_key_from_shape, build_tile_specs
from tdcf.transforms import block_dct2d, zigzag_order


def parse_args():
    p = argparse.ArgumentParser("Prepare bucketed ImageNet-1K block store")
    p.add_argument("--shards", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--resize_shorter", type=int, default=256)
    p.add_argument("--block_size", type=int, default=16)
    p.add_argument("--num_bands", type=int, default=16)
    p.add_argument("--tile_blocks", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_samples", type=int, default=0)
    return p.parse_args()


def identity_label(label):
    return int(label)


def build_loader(shards, num_workers):
    import webdataset as wds

    dataset = (
        wds.WebDataset(shards, resampled=False, shardshuffle=False)
        .decode("pil")
        .to_tuple("jpg;jpeg;png", "cls")
        .map_tuple(lambda img: img, identity_label)
    )
    return wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def resize_shorter_pil(img, resize_shorter):
    return TF.resize(
        img,
        resize_shorter,
        interpolation=TF.InterpolationMode.BILINEAR,
        antialias=True,
    )


def scan_shapes(shards, resize_shorter, block_size, num_workers, max_samples=0):
    loader = build_loader(shards, num_workers)
    shapes = []
    labels = []
    bucket_keys = []
    bucket_counts = defaultdict(int)

    for idx, (img, label) in enumerate(loader, start=1):
        img = resize_shorter_pil(img, resize_shorter)
        width, height = img.size
        shapes.append((height, width))
        labels.append(label)
        key = bucket_key_from_shape(height, width, block_size)
        bucket_keys.append(key)
        bucket_counts[key] += 1
        if idx % 10000 == 0:
            print(
                f"[scan] {idx} samples | bucket_count={len(bucket_counts)} | "
                f"largest_bucket_grid={max(bucket_counts.keys())}",
                flush=True,
            )
        if max_samples > 0 and idx >= max_samples:
            break

    return np.asarray(shapes, dtype=np.int32), np.asarray(labels, dtype=np.int64), bucket_keys


def prepare_bucket_batch(images, canvas_h, canvas_w, device):
    batch = torch.zeros((len(images), 3, canvas_h, canvas_w), dtype=torch.float32, device=device)
    for i, img in enumerate(images):
        t = TF.to_tensor(img)
        h, w = t.shape[-2:]
        batch[i, :, :h, :w] = t.to(device=device)
    return batch


def _initialize_bucket_store(
    sample_shapes: np.ndarray,
    labels: np.ndarray,
    bucket_keys: Sequence[Tuple[int, int]],
    out_dir: str,
    resize_shorter: int,
    block_size: int,
    num_bands: int,
    tile_blocks: int,
):
    os.makedirs(out_dir, exist_ok=True)
    N = len(labels)
    unique_bucket_keys = sorted(set(bucket_keys))
    bucket_key_to_id = {key: idx for idx, key in enumerate(unique_bucket_keys)}
    sample_bucket_ids = np.asarray(
        [bucket_key_to_id[key] for key in bucket_keys],
        dtype=np.int32,
    )
    sample_bucket_positions = np.empty(N, dtype=np.int32)
    bucket_sample_ids = {}

    for bucket_id in range(len(unique_bucket_keys)):
        ids = np.flatnonzero(sample_bucket_ids == bucket_id).astype(np.int64)
        bucket_sample_ids[bucket_id] = ids
        sample_bucket_positions[ids] = np.arange(len(ids), dtype=np.int32)

    np.save(os.path.join(out_dir, "labels.npy"), labels)
    np.save(os.path.join(out_dir, "sample_bucket_ids.npy"), sample_bucket_ids)
    np.save(os.path.join(out_dir, "sample_bucket_positions.npy"), sample_bucket_positions)

    total_coeffs = block_size * block_size
    band_size = total_coeffs // num_bands
    band_boundaries = [b * band_size for b in range(num_bands + 1)]
    band_boundaries[-1] = total_coeffs
    zz = zigzag_order(block_size, block_size).numpy()

    bucket_runtime = {}
    bucket_summaries = []
    total_store_files = 0

    for bucket_id, key in enumerate(unique_bucket_keys):
        nph, npw = key
        bucket_dir = os.path.join(out_dir, f"bucket_{bucket_id:03d}")
        os.makedirs(bucket_dir, exist_ok=True)

        ids = bucket_sample_ids[bucket_id]
        bucket_shapes = sample_shapes[ids]
        tile_specs = build_tile_specs(nph, npw, tile_blocks)
        P = nph * npw
        canvas_h = nph * block_size
        canvas_w = npw * block_size

        np.save(os.path.join(bucket_dir, "sample_ids.npy"), ids)
        np.save(os.path.join(bucket_dir, "sample_shapes.npy"), bucket_shapes)

        patch_signal = np.lib.format.open_memmap(
            os.path.join(bucket_dir, "patch_signal.npy"),
            mode="w+",
            dtype=np.float32,
            shape=(len(ids), P),
        )

        tile_band_mmaps = {}
        tile_payload_bytes = []
        tile_file_paths = []
        for spec in tile_specs:
            tile_index = spec["tile_index"]
            blocks_in_tile = spec["blocks_in_tile"]
            for band_idx in range(num_bands):
                start = band_boundaries[band_idx]
                end = band_boundaries[band_idx + 1]
                coeffs_in_band = end - start
                path = os.path.join(bucket_dir, f"tile_{tile_index:03d}_band_{band_idx:02d}.npy")
                mm = np.lib.format.open_memmap(
                    path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(ids), blocks_in_tile, 3, coeffs_in_band),
                )
                tile_band_mmaps[(tile_index, band_idx)] = mm
                tile_payload_bytes.append(len(ids) * blocks_in_tile * 3 * coeffs_in_band * 4)
                tile_file_paths.append(path)

        bucket_runtime[bucket_id] = {
            "bucket_dir": bucket_dir,
            "nph": nph,
            "npw": npw,
            "canvas_h": canvas_h,
            "canvas_w": canvas_w,
            "P": P,
            "tile_specs": tile_specs,
            "patch_signal": patch_signal,
            "tile_band_mmaps": tile_band_mmaps,
            "tile_file_paths": tile_file_paths,
            "tile_payload_bytes": tile_payload_bytes,
        }
        total_store_files += len(tile_file_paths) + 3

        bucket_summaries.append(
            {
                "bucket_id": int(bucket_id),
                "dir": f"bucket_{bucket_id:03d}",
                "nph": int(nph),
                "npw": int(npw),
                "canvas_h": int(canvas_h),
                "canvas_w": int(canvas_w),
                "P": int(P),
                "num_samples": int(len(ids)),
                "bytes_per_sample_full": int(P * 3 * total_coeffs * 4),
            }
        )

    print(
        f"[prep] Bucket layout ready: {len(unique_bucket_keys)} buckets, "
        f"{total_store_files} tile-band files",
        flush=True,
    )

    return {
        "N": N,
        "sample_bucket_ids": sample_bucket_ids,
        "sample_bucket_positions": sample_bucket_positions,
        "bucket_runtime": bucket_runtime,
        "bucket_summaries": bucket_summaries,
        "band_boundaries": band_boundaries,
        "band_size": band_size,
        "total_coeffs": total_coeffs,
        "zz": zz,
        "resize_shorter": resize_shorter,
        "block_size": block_size,
        "num_bands": num_bands,
        "tile_blocks": tile_blocks,
        "out_dir": out_dir,
    }


def _flush_bucket(bucket_state, images, local_positions, device):
    if not images:
        return 0

    batch = prepare_bucket_batch(
        images,
        bucket_state["canvas_h"],
        bucket_state["canvas_w"],
        device,
    )
    total_coeffs = bucket_state["total_coeffs"]
    coeffs = block_dct2d(batch, block_size=bucket_state["block_size"])
    B = coeffs.shape[0]
    coeffs_flat = coeffs.reshape(B, bucket_state["P"], 3, total_coeffs).detach().cpu().numpy()
    coeffs_zz = coeffs_flat[:, :, :, bucket_state["zz"]]
    patch_signal = np.abs(coeffs_flat).mean(axis=(2, 3)).astype(np.float32)

    local_positions = np.asarray(local_positions, dtype=np.int64)
    bucket_state["patch_signal"][local_positions] = patch_signal

    for spec in bucket_state["tile_specs"]:
        tile_index = spec["tile_index"]
        block_ids = spec["block_ids"]
        tile_coeffs = coeffs_zz[:, block_ids]
        for band_idx in range(bucket_state["num_bands"]):
            start = bucket_state["band_boundaries"][band_idx]
            end = bucket_state["band_boundaries"][band_idx + 1]
            bucket_state["tile_band_mmaps"][(tile_index, band_idx)][local_positions] = (
                tile_coeffs[:, :, :, start:end]
            )
    return B


def _finalize_bucket_store(state):
    total_store_bytes = 0
    out_dir = state["out_dir"]
    for bucket_id, summary in enumerate(state["bucket_summaries"]):
        meta = state["bucket_runtime"][bucket_id]
        meta["patch_signal"].flush()
        del meta["patch_signal"]

        tile_file_byte_sizes = []
        for mm in meta["tile_band_mmaps"].values():
            mm.flush()
            del mm
        for path in meta["tile_file_paths"]:
            tile_file_byte_sizes.append(os.path.getsize(path))

        sample_ids_path = os.path.join(meta["bucket_dir"], "sample_ids.npy")
        sample_shapes_path = os.path.join(meta["bucket_dir"], "sample_shapes.npy")
        patch_signal_path = os.path.join(meta["bucket_dir"], "patch_signal.npy")
        bucket_meta = {
            "format": "bucketed_tile_band_dct_bucket",
            "bucket_id": int(bucket_id),
            "resize_shorter": int(state["resize_shorter"]),
            "block_size": int(state["block_size"]),
            "num_bands_per_patch": int(state["num_bands"]),
            "band_size": int(state["band_size"]),
            "band_boundaries": [int(x) for x in state["band_boundaries"]],
            "tile_blocks": int(state["tile_blocks"]),
            "nph": int(summary["nph"]),
            "npw": int(summary["npw"]),
            "canvas_h": int(summary["canvas_h"]),
            "canvas_w": int(summary["canvas_w"]),
            "P": int(summary["P"]),
            "num_samples": int(summary["num_samples"]),
            "bytes_per_sample_full": int(summary["bytes_per_sample_full"]),
            "tile_specs": meta["tile_specs"],
            "tile_payload_bytes": [int(x) for x in meta["tile_payload_bytes"]],
            "tile_file_byte_sizes": [int(x) for x in tile_file_byte_sizes],
            "sample_ids_byte_size": int(os.path.getsize(sample_ids_path)),
            "sample_shapes_byte_size": int(os.path.getsize(sample_shapes_path)),
            "patch_signal_byte_size": int(os.path.getsize(patch_signal_path)),
        }
        with open(os.path.join(meta["bucket_dir"], "metadata.json"), "w") as f:
            json.dump(bucket_meta, f, indent=2)

        total_store_bytes += (
            sum(tile_file_byte_sizes)
            + bucket_meta["sample_ids_byte_size"]
            + bucket_meta["sample_shapes_byte_size"]
            + bucket_meta["patch_signal_byte_size"]
            + os.path.getsize(os.path.join(meta["bucket_dir"], "metadata.json"))
        )

    root_metadata = {
        "format": "bucketed_tile_band_dct",
        "N": int(state["N"]),
        "C": 3,
        "resize_shorter": int(state["resize_shorter"]),
        "block_size": int(state["block_size"]),
        "num_bands_per_patch": int(state["num_bands"]),
        "band_size": int(state["band_size"]),
        "total_coeffs_per_patch": int(state["total_coeffs"]),
        "tile_blocks": int(state["tile_blocks"]),
        "zigzag_order": state["zz"].tolist(),
        "buckets": state["bucket_summaries"],
        "labels_byte_size": int(os.path.getsize(os.path.join(out_dir, "labels.npy"))),
        "sample_bucket_ids_byte_size": int(os.path.getsize(os.path.join(out_dir, "sample_bucket_ids.npy"))),
        "sample_bucket_positions_byte_size": int(os.path.getsize(os.path.join(out_dir, "sample_bucket_positions.npy"))),
        "total_store_bytes": int(total_store_bytes),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(root_metadata, f, indent=2)

    return total_store_bytes


def build_store_from_samples(
    samples: Sequence,
    labels: Sequence[int],
    out_dir: str,
    resize_shorter: int,
    block_size: int,
    num_bands: int,
    tile_blocks: int,
    batch_size: int,
    device: torch.device,
):
    resized_samples = [resize_shorter_pil(img, resize_shorter) for img in samples]
    sample_shapes = np.asarray(
        [(img.size[1], img.size[0]) for img in resized_samples],
        dtype=np.int32,
    )
    labels_np = np.asarray(labels, dtype=np.int64)
    bucket_keys = [
        bucket_key_from_shape(h, w, block_size)
        for h, w in sample_shapes
    ]
    state = _initialize_bucket_store(
        sample_shapes,
        labels_np,
        bucket_keys,
        out_dir,
        resize_shorter,
        block_size,
        num_bands,
        tile_blocks,
    )

    pending = defaultdict(lambda: {"images": [], "local_positions": []})
    for global_idx, img in enumerate(resized_samples):
        bucket_id = int(state["sample_bucket_ids"][global_idx])
        local_pos = int(state["sample_bucket_positions"][global_idx])
        pending[bucket_id]["images"].append(img)
        pending[bucket_id]["local_positions"].append(local_pos)
        if len(pending[bucket_id]["images"]) >= batch_size:
            bucket_state = {
                **state["bucket_runtime"][bucket_id],
                "band_boundaries": state["band_boundaries"],
                "num_bands": state["num_bands"],
                "total_coeffs": state["total_coeffs"],
                "zz": state["zz"],
                "block_size": state["block_size"],
            }
            _flush_bucket(
                bucket_state,
                pending[bucket_id]["images"],
                pending[bucket_id]["local_positions"],
                device,
            )
            pending[bucket_id]["images"].clear()
            pending[bucket_id]["local_positions"].clear()

    for bucket_id, local in pending.items():
        if local["images"]:
            bucket_state = {
                **state["bucket_runtime"][bucket_id],
                "band_boundaries": state["band_boundaries"],
                "num_bands": state["num_bands"],
                "total_coeffs": state["total_coeffs"],
                "zz": state["zz"],
                "block_size": state["block_size"],
            }
            _flush_bucket(bucket_state, local["images"], local["local_positions"], device)

    return _finalize_bucket_store(state)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[prep] First pass: scanning resized shapes and bucket assignment", flush=True)
    sample_shapes, labels, bucket_keys = scan_shapes(
        args.shards,
        args.resize_shorter,
        args.block_size,
        args.num_workers,
        max_samples=args.max_samples,
    )

    state = _initialize_bucket_store(
        sample_shapes,
        labels,
        bucket_keys,
        args.out_dir,
        args.resize_shorter,
        args.block_size,
        args.num_bands,
        args.tile_blocks,
    )

    print("[prep] Second pass: writing bucketed tile-band store", flush=True)
    loader = build_loader(args.shards, args.num_workers)
    device = torch.device(args.device)
    pending = defaultdict(lambda: {"images": [], "local_positions": []})
    offset = 0

    for img, _label in loader:
        img = resize_shorter_pil(img, args.resize_shorter)
        bucket_id = int(state["sample_bucket_ids"][offset])
        local_pos = int(state["sample_bucket_positions"][offset])
        pending[bucket_id]["images"].append(img)
        pending[bucket_id]["local_positions"].append(local_pos)

        if len(pending[bucket_id]["images"]) >= args.batch_size:
            bucket_state = {
                **state["bucket_runtime"][bucket_id],
                "band_boundaries": state["band_boundaries"],
                "num_bands": state["num_bands"],
                "total_coeffs": state["total_coeffs"],
                "zz": state["zz"],
                "block_size": state["block_size"],
            }
            _flush_bucket(
                bucket_state,
                pending[bucket_id]["images"],
                pending[bucket_id]["local_positions"],
                device,
            )
            pending[bucket_id]["images"].clear()
            pending[bucket_id]["local_positions"].clear()

        offset += 1
        if offset % (args.batch_size * 20) == 0 or offset >= state["N"]:
            print(f"[prep] wrote {offset}/{state['N']}", flush=True)
        if args.max_samples > 0 and offset >= args.max_samples:
            break

    for bucket_id, local in pending.items():
        if local["images"]:
            bucket_state = {
                **state["bucket_runtime"][bucket_id],
                "band_boundaries": state["band_boundaries"],
                "num_bands": state["num_bands"],
                "total_coeffs": state["total_coeffs"],
                "zz": state["zz"],
                "block_size": state["block_size"],
            }
            _flush_bucket(bucket_state, local["images"], local["local_positions"], device)

    total_store_bytes = _finalize_bucket_store(state)
    print(
        f"[prep] Done. Bucketed store saved to {args.out_dir} "
        f"({total_store_bytes / 1e9:.2f} GB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
