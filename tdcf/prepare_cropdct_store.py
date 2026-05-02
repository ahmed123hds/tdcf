"""Build a CropDCT store from WebDataset shards or synthetic images."""

from __future__ import annotations

import argparse
import io
import os
import random
import re
import time

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from tdcf.cropdct_store import DEFAULT_BANDS, CropDCTWriter


def parse_args():
    p = argparse.ArgumentParser("Prepare CropDCT store")
    p.add_argument("--source", choices=["webdataset", "synthetic"], default="webdataset")
    p.add_argument("--shards", type=str, default=None)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--records_per_shard", type=int, default=1024)
    p.add_argument("--block_size", type=int, default=8)
    p.add_argument("--tile_blocks", type=int, default=32)
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--compression", choices=["zstd"], default="zstd")
    p.add_argument("--compression_level", type=int, default=1)
    p.add_argument("--max_longer", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--random_subset", action="store_true",
                   help="Shuffle shard order and stream order before taking --max_samples.")
    p.add_argument("--shuffle_buffer", type=int, default=10000,
                   help="WebDataset sample shuffle buffer used with --random_subset.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from completed CropDCT shards in --out_dir.")
    p.add_argument("--skip_log_every", type=int, default=10000,
                   help="Progress interval while skipping already completed source samples during resume.")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def expand_shards(pattern: str):
    urls = []
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


def identity_label(label):
    return int(label)


def build_loader(args, skip_samples: int = 0):
    if args.source == "synthetic":
        n = args.max_samples if args.max_samples > 0 else 64
        g = torch.Generator().manual_seed(args.seed)
        samples = []
        for i in range(n):
            if i < skip_samples:
                continue
            h = 180 + (i % 9) * 23
            w = 190 + (i % 7) * 31
            samples.append((torch.rand(3, h, w, generator=g), i % 1000, f"synthetic-{i:06d}", "synthetic"))
        return samples

    if args.shards is None:
        raise ValueError("--shards is required for WebDataset source")
    if args.num_workers != 0:
        raise ValueError("CropDCT build currently requires --num_workers 0 for deterministic order")
    import webdataset as wds

    def transform(img):
        img = img.convert("RGB")
        if args.max_longer > 0:
            w, h = img.size
            longer = max(h, w)
            if longer > args.max_longer:
                scale = args.max_longer / longer
                img = img.resize((int(round(w * scale)), int(round(h * scale))))
        return TF.to_tensor(img)

    def image_from_raw(value):
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, bytes):
            return Image.open(io.BytesIO(value))
        if hasattr(value, "read"):
            return Image.open(value)
        raise TypeError(f"Unsupported image payload type: {type(value)!r}")

    urls = expand_shards(args.shards)
    if not urls:
        raise ValueError(f"No shards expanded from {args.shards!r}")
    if not os.path.exists(urls[0]):
        raise FileNotFoundError(f"First shard does not exist: {urls[0]}")
    if args.random_subset:
        rng = random.Random(args.seed)
        rng.shuffle(urls)
    print(
        f"[cropdct] shards expanded: count={len(urls)} first={urls[0]} last={urls[-1]} "
        f"random_subset={args.random_subset} seed={args.seed}",
        flush=True,
    )
    dataset = wds.WebDataset(urls, resampled=False, shardshuffle=False)
    if args.random_subset:
        dataset = dataset.shuffle(args.shuffle_buffer, initial=min(args.shuffle_buffer, 1000))
    dataset = dataset.to_tuple("__key__", "__url__", "jpg;jpeg;png", "cls")

    def iterator():
        for source_idx, (key, url, img_raw, label_raw) in enumerate(dataset, start=1):
            if source_idx <= skip_samples:
                if args.skip_log_every > 0 and source_idx % args.skip_log_every == 0:
                    print(
                        f"[cropdct-resume] skipped {source_idx}/{skip_samples} raw source samples",
                        flush=True,
                    )
                continue
            if skip_samples and source_idx == skip_samples + 1:
                print(f"[cropdct-resume] reached resume point at source sample {source_idx}", flush=True)
            img = transform(image_from_raw(img_raw))
            yield img, identity_label(label_raw), str(key), str(url)

    return iterator()


def build_store(args):
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    writer = CropDCTWriter(
        args.out_dir,
        records_per_shard=args.records_per_shard,
        block_size=args.block_size,
        tile_blocks=args.tile_blocks,
        band_specs=DEFAULT_BANDS,
        quality=args.quality,
        compression=args.compression,
        compression_level=args.compression_level,
        device=torch.device(args.device),
        resume=args.resume,
    )
    start = time.time()
    completed = len(writer.global_rows)
    count = completed
    if completed:
        print(f"[cropdct-resume] skipping first {completed} source samples before PIL decode", flush=True)
    for img, label, source_key, source_url in build_loader(args, skip_samples=completed):
        writer.add(img, int(label), source_key=str(source_key), source_url=str(source_url))
        count += 1
        if count % 1000 == 0:
            elapsed = time.time() - start
            new_count = count - completed
            print(
                f"[cropdct] wrote {count} samples total | resumed_new={new_count} "
                f"({new_count / max(elapsed, 1e-6):.1f} img/s)",
                flush=True,
            )
        if args.max_samples > 0 and count >= args.max_samples:
            break
    writer.close()
    elapsed = time.time() - start
    print(f"[cropdct] Done: {args.out_dir} samples={count} time={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    build_store(parse_args())
