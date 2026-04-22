#!/bin/bash
# =============================================================================
# Build a crop-aware ImageNet-1K block-DCT store.
# Usage: bash tdcf/scripts/run_prepare_imagenet1k_crop_store.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PACKAGE_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

TRAIN_SHARDS='/mnt/dataset_disk/imagenet_hf/imagenet1k-train-{0000..1023}.tar'
VAL_SHARDS='/mnt/dataset_disk/imagenet_hf/imagenet1k-validation-{00..63}.tar'
OUT_ROOT='./data/imagenet1k_crop_block_store'
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUT_ROOT}"

python3 -m tdcf.prepare_imagenet1k_crop_store \
    --shards "${TRAIN_SHARDS}" \
    --out_dir "${OUT_ROOT}/train" \
    --resize_shorter 256 \
    --block_size 16 \
    --num_bands 16 \
    --batch_size 64 \
    --num_workers 8 \
    --device "${DEVICE}"

python3 -m tdcf.prepare_imagenet1k_crop_store \
    --shards "${VAL_SHARDS}" \
    --out_dir "${OUT_ROOT}/val" \
    --resize_shorter 256 \
    --block_size 16 \
    --num_bands 16 \
    --batch_size 64 \
    --num_workers 8 \
    --device "${DEVICE}"
