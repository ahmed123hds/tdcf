"""
Build a crop-aware ImageNet-1K block-DCT store.

This is the first foundation for a physical-I/O ImageNet pipeline that keeps
stochastic crop semantics over a larger precomputed canvas instead of reducing
training to a fixed 224x224 view.

Storage layout:
    <root>/
        metadata.json
        labels.npy
        sample_shapes.npy
        patch_signal.npy
        patch_000.npy
        patch_001.npy
        ...

Each image is:
  1. resized so the shorter side equals `--resize_shorter`
  2. padded to the maximum patch grid seen in the split
  3. transformed into block-DCT coefficients
  4. stored patch-by-patch in zig-zag order

At training time a crop-aware loader can sample crop windows on the resized
canvas and read only the overlapping patch files.
"""

import argparse
import json
import math
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
import webdataset as wds

from tdcf.transforms import block_dct2d, zigzag_order


def parse_args():
    p = argparse.ArgumentParser("Prepare crop-aware ImageNet-1K block store")
    p.add_argument("--shards", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--resize_shorter", type=int, default=256)
    p.add_argument("--block_size", type=int, default=16)
    p.add_argument("--num_bands", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def identity_label(label):
    return int(label)


def build_loader(shards, num_workers):
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
    return TF.resize(img, resize_shorter, interpolation=TF.InterpolationMode.BILINEAR, antialias=True)


def scan_shapes(shards, resize_shorter, num_workers):
    loader = build_loader(shards, num_workers)
    shapes = []
    labels = []
    max_h = 0
    max_w = 0

    for idx, (img, label) in enumerate(loader, start=1):
        img = resize_shorter_pil(img, resize_shorter)
        w, h = img.size
        shapes.append((h, w))
        labels.append(label)
        max_h = max(max_h, h)
        max_w = max(max_w, w)
        if idx % 10000 == 0:
            print(f"[scan] {idx} samples | max_h={max_h} max_w={max_w}", flush=True)

    return np.asarray(shapes, dtype=np.int32), np.asarray(labels, dtype=np.int64), max_h, max_w


def prepare_batch_tensors(images, canvas_h, canvas_w, device):
    batch = torch.zeros((len(images), 3, canvas_h, canvas_w), dtype=torch.float32, device=device)
    shapes = []
    for i, img in enumerate(images):
        t = TF.to_tensor(img)
        h, w = t.shape[-2:]
        batch[i, :, :h, :w] = t.to(device=device)
        shapes.append((h, w))
    return batch, shapes


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[prep] First pass: scanning resized shapes", flush=True)
    sample_shapes, labels, max_h, max_w = scan_shapes(
        args.shards, args.resize_shorter, args.num_workers
    )
    N = len(labels)
    canvas_h = math.ceil(max_h / args.block_size) * args.block_size
    canvas_w = math.ceil(max_w / args.block_size) * args.block_size
    nph = canvas_h // args.block_size
    npw = canvas_w // args.block_size
    P = nph * npw
    total_coeffs = args.block_size * args.block_size
    band_size = total_coeffs // args.num_bands
    zz = zigzag_order(args.block_size, args.block_size).numpy()

    print(
        f"[prep] N={N} resize_shorter={args.resize_shorter} "
        f"max_resized={max_h}x{max_w} canvas={canvas_h}x{canvas_w} "
        f"patches={P}",
        flush=True,
    )

    patch_paths = []
    patch_mmaps = []
    patch_payload_byte_sizes = []
    for p_idx in range(P):
        path = os.path.join(args.out_dir, f"patch_{p_idx:03d}.npy")
        mm = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=(N, 3, total_coeffs),
        )
        patch_paths.append(path)
        patch_mmaps.append(mm)
        patch_payload_byte_sizes.append(N * 3 * total_coeffs * 4)

    np.save(os.path.join(args.out_dir, "labels.npy"), labels)
    np.save(os.path.join(args.out_dir, "sample_shapes.npy"), sample_shapes)
    patch_signal = np.lib.format.open_memmap(
        os.path.join(args.out_dir, "patch_signal.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(N, P),
    )

    print("[prep] Second pass: writing patch coefficient store", flush=True)
    loader = build_loader(args.shards, args.num_workers)
    device = torch.device(args.device)
    pending_images = []
    offset = 0

    def flush_pending(images, start_idx):
        if not images:
            return 0
        batch, _ = prepare_batch_tensors(images, canvas_h, canvas_w, device)
        coeffs = block_dct2d(batch, block_size=args.block_size)  # (B, P, C, bs, bs)
        B = coeffs.shape[0]
        coeffs_flat = coeffs.reshape(B, P, 3, total_coeffs).detach().cpu().numpy()
        coeffs_zz = coeffs_flat[:, :, :, zz]
        sig = np.abs(coeffs_flat).mean(axis=(2, 3)).astype(np.float32)

        for p_idx in range(P):
            patch_mmaps[p_idx][start_idx:start_idx + B] = coeffs_zz[:, p_idx]
        patch_signal[start_idx:start_idx + B] = sig
        return B

    for img, _label in loader:
        img = resize_shorter_pil(img, args.resize_shorter)
        pending_images.append(img)
        if len(pending_images) >= args.batch_size:
            written = flush_pending(pending_images, offset)
            offset += written
            pending_images.clear()
            if offset % (args.batch_size * 20) == 0 or offset >= N:
                print(f"[prep] wrote {offset}/{N}", flush=True)

    if pending_images:
        offset += flush_pending(pending_images, offset)
        pending_images.clear()

    for mm in patch_mmaps:
        mm.flush()
        del mm
    patch_signal.flush()
    del patch_signal

    patch_file_byte_sizes = [os.path.getsize(p) for p in patch_paths]
    labels_byte_size = os.path.getsize(os.path.join(args.out_dir, "labels.npy"))
    sample_shapes_byte_size = os.path.getsize(os.path.join(args.out_dir, "sample_shapes.npy"))
    patch_signal_byte_size = os.path.getsize(os.path.join(args.out_dir, "patch_signal.npy"))
    total_payload_bytes = sum(patch_payload_byte_sizes)
    total_store_bytes = (
        sum(patch_file_byte_sizes)
        + labels_byte_size
        + sample_shapes_byte_size
        + patch_signal_byte_size
    )

    metadata = {
        "format": "crop_aware_block_dct",
        "N": int(N),
        "C": 3,
        "resize_shorter": int(args.resize_shorter),
        "canvas_h": int(canvas_h),
        "canvas_w": int(canvas_w),
        "block_size_h": int(args.block_size),
        "block_size_w": int(args.block_size),
        "nph": int(nph),
        "npw": int(npw),
        "P": int(P),
        "total_coeffs_per_patch": int(total_coeffs),
        "num_bands_per_patch": int(args.num_bands),
        "band_size": int(band_size),
        "zigzag_order": zz.tolist(),
        "patch_payload_byte_sizes": patch_payload_byte_sizes,
        "patch_file_byte_sizes": patch_file_byte_sizes,
        "labels_byte_size": int(labels_byte_size),
        "sample_shapes_byte_size": int(sample_shapes_byte_size),
        "patch_signal_byte_size": int(patch_signal_byte_size),
        "total_payload_bytes": int(total_payload_bytes),
        "total_store_bytes": int(total_store_bytes),
        "bytes_per_sample_full": int(P * 3 * total_coeffs * 4),
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"[prep] Done. Store saved to {args.out_dir} "
        f"({total_store_bytes / 1e9:.2f} GB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
