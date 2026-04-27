#!/bin/bash
# =============================================================================
# Quantized DCT precision search.
#
# Smoke test:
#   bash tdcf/scripts/run_quantized_dct_precision.sh synthetic
#
# CIFAR-100 local sample:
#   bash tdcf/scripts/run_quantized_dct_precision.sh cifar100
#
# ImageNet WebDataset sample:
#   SHARDS='/mnt/dataset_disk/imagenet_hf/imagenet1k-train-{0000..1023}.tar' \
#   bash tdcf/scripts/run_quantized_dct_precision.sh webdataset
#
# ImageFolder sample:
#   DATA_ROOT='./data/tiny-imagenet-200' SPLIT='train' \
#   bash tdcf/scripts/run_quantized_dct_precision.sh imagefolder
# =============================================================================

set -euo pipefail

SOURCE="${1:-synthetic}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PACKAGE_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SAVE_DIR="${SAVE_DIR:-./results/dct_quant_precision_${SOURCE}}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-32}"
IMG_SIZE="${IMG_SIZE:-224}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
NUM_BANDS="${NUM_BANDS:-16}"
TARGET_PSNR="${TARGET_PSNR:-45.0}"
TARGET_SSIM="${TARGET_SSIM:-0.995}"
MAX_MULTIPLIER="${MAX_MULTIPLIER:-256}"
BINARY_STEPS="${BINARY_STEPS:-18}"
SCALE_SCOPE="${SCALE_SCOPE:-band}"
SCALE_STAT="${SCALE_STAT:-max}"
SCALE_PERCENTILE="${SCALE_PERCENTILE:-99.99}"
NUM_WORKERS="${NUM_WORKERS:-0}"

COMMON_ARGS=(
    --source "$SOURCE"
    --num_samples "$NUM_SAMPLES"
    --batch_size "$BATCH_SIZE"
    --img_size "$IMG_SIZE"
    --block_size "$BLOCK_SIZE"
    --num_bands "$NUM_BANDS"
    --target_psnr "$TARGET_PSNR"
    --target_ssim "$TARGET_SSIM"
    --max_multiplier "$MAX_MULTIPLIER"
    --binary_steps "$BINARY_STEPS"
    --scale_scope "$SCALE_SCOPE"
    --scale_stat "$SCALE_STAT"
    --scale_percentile "$SCALE_PERCENTILE"
    --num_workers "$NUM_WORKERS"
    --save_dir "$SAVE_DIR"
)

case "$SOURCE" in
    synthetic)
        ;;
    cifar100)
        COMMON_ARGS+=(--data_root "${DATA_ROOT:-./data}" --split "${SPLIT:-train}")
        ;;
    webdataset)
        if [ -z "${SHARDS:-}" ]; then
            SHARDS='/mnt/dataset_disk/imagenet_hf/imagenet1k-train-{0000..1023}.tar'
        fi
        COMMON_ARGS+=(--shards "$SHARDS")
        ;;
    imagefolder)
        if [ -z "${DATA_ROOT:-}" ]; then
            echo "DATA_ROOT is required for imagefolder source"
            exit 2
        fi
        COMMON_ARGS+=(--data_root "$DATA_ROOT" --split "${SPLIT:-train}")
        ;;
    *)
        echo "Unknown source: $SOURCE"
        exit 2
        ;;
esac

python3 -m tdcf.quantized_dct_precision "${COMMON_ARGS[@]}"
