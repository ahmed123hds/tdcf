#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
import webdataset as wds

from tdcf.train_tpu_imagenet1k import build_model


def parse_args():
    p = argparse.ArgumentParser("Evaluate ImageNet-1K checkpoint deterministically")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--val_shards", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--legacy_transform_normalize",
        action="store_true",
        help="Reproduce the old buggy path where transforms normalized before the model normalized again.",
    )
    p.add_argument("--save_json", type=Path, default=None)
    return p.parse_args()


def identity_label(label):
    return int(label)


def build_eval_loader(
    shards_url: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    legacy_transform_normalize: bool,
):
    transform_ops = [
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
    ]
    if legacy_transform_normalize:
        transform_ops.append(
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        )
    transform = T.Compose(transform_ops)

    dataset = (
        wds.WebDataset(shards_url, resampled=False, shardshuffle=False)
        .decode("pil")
        .to_tuple("jpg;jpeg;png", "cls")
        .map_tuple(transform, identity_label)
        .batched(batch_size, partial=True)
    )
    return wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = argparse.Namespace(**ckpt["args"])

    device = torch.device(args.device)
    model = build_model(train_args, device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_args.label_smooth))
    loader = build_eval_loader(
        args.val_shards,
        img_size=int(train_args.img_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        legacy_transform_normalize=args.legacy_transform_normalize,
    )

    loss_sum = 0.0
    correct = 0
    n = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_n = labels.size(0)
            loss_sum += loss.item() * batch_n
            correct += (logits.argmax(1) == labels).sum().item()
            n += batch_n

    results = {
        "checkpoint": str(args.checkpoint),
        "legacy_transform_normalize": args.legacy_transform_normalize,
        "n_eval": n,
        "val_loss": loss_sum / n,
        "val_acc": correct / n,
    }
    print(json.dumps(results, indent=2))

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
