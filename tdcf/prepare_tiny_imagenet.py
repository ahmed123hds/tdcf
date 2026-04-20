"""
Download Tiny ImageNet and build the BlockBandStore for TDCF training.

Usage (on your TPU server):
    PYTHONPATH=. python3 tdcf/prepare_tiny_imagenet.py --data_dir ./data

This will create:
    ./data/tiny_imagenet_block_store/train/   (patch_00..63.npy, labels.npy, ...)
    ./data/tiny_imagenet_block_store/val/     (same layout)
"""

import os
import sys
import argparse
import shutil
import zipfile
import urllib.request
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from tdcf.transforms import block_dct2d
from tdcf.storage import build_block_band_store_from_tensor

TINY_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
IMAGE_SIZE = 64
BLOCK_SIZE = 8
NUM_BANDS = 16


def download_tiny_imagenet(data_dir: str) -> str:
    """Download and extract Tiny ImageNet if not already present."""
    raw_dir = os.path.join(data_dir, "tiny-imagenet-200")
    if os.path.isdir(raw_dir):
        print(f"[prep] Tiny ImageNet already exists at {raw_dir}")
        return raw_dir

    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
    os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(zip_path):
        print(f"[prep] Downloading Tiny ImageNet (~237 MB) ...")
        urllib.request.urlretrieve(TINY_URL, zip_path)
        print(f"[prep] Download complete.")

    print(f"[prep] Extracting ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)
    print(f"[prep] Extracted to {raw_dir}")
    return raw_dir


def fix_val_folder(raw_dir: str):
    """
    Tiny ImageNet val/ has all images in one folder with a text annotation.
    Reorganize into class subfolders so ImageFolder can load it.
    """
    val_dir = os.path.join(raw_dir, "val")
    ann_path = os.path.join(val_dir, "val_annotations.txt")

    # If already reorganized, skip
    images_dir = os.path.join(val_dir, "images")
    if not os.path.isdir(images_dir):
        print(f"[prep] Val folder already reorganized.")
        return

    print(f"[prep] Reorganizing val/ into class subfolders ...")
    with open(ann_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            fname, class_id = parts[0], parts[1]
            class_dir = os.path.join(val_dir, class_id)
            os.makedirs(class_dir, exist_ok=True)
            src = os.path.join(images_dir, fname)
            dst = os.path.join(class_dir, fname)
            if os.path.exists(src):
                shutil.move(src, dst)

    # Remove the now-empty images/ directory
    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    print(f"[prep] Done reorganizing val/.")


@torch.no_grad()
def build_store(dataset, store_path: str, device: torch.device):
    """Compute block-DCT and build the BlockBandStore."""
    loader = DataLoader(dataset, batch_size=512, shuffle=False,
                        num_workers=4, pin_memory=(device.type == 'cuda'))

    all_coeffs, all_labels = [], []
    total = len(dataset)
    done = 0

    for images, labels in loader:
        images = images.to(device)
        c = block_dct2d(images, BLOCK_SIZE)  # (B, P, C, 8, 8)
        all_coeffs.append(c.cpu())
        all_labels.append(labels)
        done += images.size(0)
        print(f"\r[prep]   DCT: {done}/{total}", end="", flush=True)

    print()
    coeffs = torch.cat(all_coeffs, dim=0)
    labels = torch.cat(all_labels, dim=0)

    nph = IMAGE_SIZE // BLOCK_SIZE  # 8
    npw = IMAGE_SIZE // BLOCK_SIZE  # 8

    print(f"[prep] Coeffs shape: {coeffs.shape}  Labels: {labels.shape}")
    build_block_band_store_from_tensor(
        coeffs, labels, store_path,
        nph=nph, npw=npw,
        num_bands_per_patch=NUM_BANDS,
    )
    print(f"[prep] Store saved to {store_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Root data directory.")
    args = parser.parse_args()

    # Step 1: Download
    raw_dir = download_tiny_imagenet(args.data_dir)

    # Step 2: Fix val folder structure
    fix_val_folder(raw_dir)

    # Step 3: Load with torchvision
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(raw_dir, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(
        os.path.join(raw_dir, "val"), transform=transform)

    print(f"[prep] Train samples: {len(train_dataset)}")
    print(f"[prep] Val samples:   {len(val_dataset)}")

    # Step 4: Compute block-DCT and build stores
    device = torch.device("cpu")  # CPU is fine for preprocessing
    print(f"[prep] Using device: {device}")

    train_store = os.path.join(args.data_dir, "tiny_imagenet_block_store", "train")
    val_store = os.path.join(args.data_dir, "tiny_imagenet_block_store", "val")

    print(f"\n[prep] Building TRAIN store ...")
    build_store(train_dataset, train_store, device)

    print(f"\n[prep] Building VAL store ...")
    build_store(val_dataset, val_store, device)

    print(f"\n[prep] All done! Stores ready at:")
    print(f"  Train: {train_store}")
    print(f"  Val:   {val_store}")


if __name__ == "__main__":
    main()
