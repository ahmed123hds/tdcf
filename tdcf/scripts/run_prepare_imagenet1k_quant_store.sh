#!/bin/bash
# =============================================================================
# Build compressed int8 fixed-view ImageNet-1K DCT stores under /mnt.
#
# Default layout:
#   /mnt/dataset_disk/imagenet1k_quant_dct_original/train
#   /mnt/dataset_disk/imagenet1k_quant_dct_original/val
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PACKAGE_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [ -z "${TRAIN_SHARDS:-}" ]; then
    TRAIN_SHARDS='/mnt/dataset_disk/imagenet_hf/imagenet1k-train-{0000..1023}.tar'
fi
if [ -z "${VAL_SHARDS:-}" ]; then
    VAL_SHARDS='/mnt/dataset_disk/imagenet_hf/imagenet1k-validation-{00..63}.tar'
fi

OUT_ROOT="${OUT_ROOT:-/mnt/dataset_disk/imagenet1k_quant_dct_original}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
NUM_BANDS="${NUM_BANDS:-16}"
TILE_BLOCKS="${TILE_BLOCKS:-4}"
CHUNK_SIZE="${CHUNK_SIZE:-128}"
BUCKET_COUNT="${BUCKET_COUNT:-16}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-4096}"
QUANT_MULTIPLIER="${QUANT_MULTIPLIER:-1.05}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-6}"
DEVICE="${DEVICE:-cpu}"

mkdir -p "$OUT_ROOT"

echo "[prep-quant] Building train store at $OUT_ROOT/train"
python3 -m tdcf.prepare_imagenet1k_quantized_store \
    --source webdataset \
    --shards "$TRAIN_SHARDS" \
    --out_dir "$OUT_ROOT/train" \
    --block_size "$BLOCK_SIZE" \
    --num_bands "$NUM_BANDS" \
    --tile_blocks "$TILE_BLOCKS" \
    --chunk_size "$CHUNK_SIZE" \
    --bucket_count "$BUCKET_COUNT" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --calibration_samples "$CALIBRATION_SAMPLES" \
    --quant_multiplier "$QUANT_MULTIPLIER" \
    --compression_level "$COMPRESSION_LEVEL" \
    --device "$DEVICE" \
    ${FORECAST_ONLY:+--forecast_only} \
    ${SCAN_CACHE:+--scan_cache "$SCAN_CACHE"}

echo "[prep-quant] Building val store at $OUT_ROOT/val"
python3 -m tdcf.prepare_imagenet1k_quantized_store \
    --source webdataset \
    --shards "$VAL_SHARDS" \
    --out_dir "$OUT_ROOT/val" \
    --block_size "$BLOCK_SIZE" \
    --num_bands "$NUM_BANDS" \
    --tile_blocks "$TILE_BLOCKS" \
    --chunk_size "$CHUNK_SIZE" \
    --bucket_count "$BUCKET_COUNT" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --calibration_samples "$CALIBRATION_SAMPLES" \
    --quant_multiplier "$QUANT_MULTIPLIER" \
    --compression_level "$COMPRESSION_LEVEL" \
    --device "$DEVICE"

echo "[prep-quant] Done. Store root: $OUT_ROOT"
