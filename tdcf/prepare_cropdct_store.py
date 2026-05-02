"""Build a CropDCT store from WebDataset shards or synthetic images."""

from __future__ import annotations

import argparse
import os
import random
import re
import time

import torch
import torchvision.transforms.functional as TF

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


def build_loader(args):
    if args.source == "synthetic":
        n = args.max_samples if args.max_samples > 0 else 64
        g = torch.Generator().manual_seed(args.seed)
        samples = []
        for i in range(n):
            h = 180 + (i % 9) * 23
            w = 190 + (i % 7) * 31
            samples.append((torch.rand(3, h, w, generator=g), i % 1000, f"synthetic-{i:06d}"))
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
    dataset = (
        dataset.decode("pil")
        .to_tuple("__key__", "jpg;jpeg;png", "cls")
        .map_tuple(str, transform, identity_label)
        .map(lambda sample: (sample[1], sample[2], sample[0]))
    )
    return wds.WebLoader(dataset, batch_size=None, num_workers=0)


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
    )
    start = time.time()
    count = 0
    for count, (img, label, source_key) in enumerate(build_loader(args), start=1):
        writer.add(img, int(label), source_key=str(source_key))
        if count % 1000 == 0:
            elapsed = time.time() - start
            print(f"[cropdct] wrote {count} samples ({count / max(elapsed, 1e-6):.1f} img/s)", flush=True)
        if args.max_samples > 0 and count >= args.max_samples:
            break
    writer.close()
    elapsed = time.time() - start
    print(f"[cropdct] Done: {args.out_dir} samples={count} time={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    build_store(parse_args())
