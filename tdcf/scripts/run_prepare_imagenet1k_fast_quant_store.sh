#!/bin/bash
# Build the fast sequential int8 DCT store.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PACKAGE_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${TRAIN_SHARDS:-}" ]]; then
  TRAIN_SHARDS='/mnt/dataset_disk/imagenet_hf/imagenet1k-train-{0000..1023}.tar'
fi
if [[ -z "${VAL_SHARDS:-}" ]]; then
  VAL_SHARDS='/mnt/dataset_disk/imagenet_hf/imagenet1k-validation-{00..63}.tar'
fi
OUT_ROOT="${OUT_ROOT:-/mnt/dataset_disk/imagenet1k_fast_quant_dct_224}"
VIEW_SIZE="${VIEW_SIZE:-224}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
NUM_BANDS="${NUM_BANDS:-16}"
RECORDS_PER_SHARD="${RECORDS_PER_SHARD:-8192}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-4096}"
QUANT_MULTIPLIER="${QUANT_MULTIPLIER:-1.05}"
DEVICE="${DEVICE:-cpu}"

mkdir -p "$OUT_ROOT"

echo "[fast-prep] Building train store at $OUT_ROOT/train"
echo "[fast-prep] train shards: $TRAIN_SHARDS"
python3 -m tdcf.prepare_imagenet1k_fast_quant_store \
  --source webdataset \
  --shards "$TRAIN_SHARDS" \
  --out_dir "$OUT_ROOT/train" \
  --view_size "$VIEW_SIZE" \
  --block_size "$BLOCK_SIZE" \
  --num_bands "$NUM_BANDS" \
  --records_per_shard "$RECORDS_PER_SHARD" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --calibration_samples "$CALIBRATION_SAMPLES" \
  --quant_multiplier "$QUANT_MULTIPLIER" \
  --device "$DEVICE" \
  ${MAX_SAMPLES:+--max_samples "$MAX_SAMPLES"}

echo "[fast-prep] Building val store at $OUT_ROOT/val"
echo "[fast-prep] val shards: $VAL_SHARDS"
python3 -m tdcf.prepare_imagenet1k_fast_quant_store \
  --source webdataset \
  --shards "$VAL_SHARDS" \
  --out_dir "$OUT_ROOT/val" \
  --view_size "$VIEW_SIZE" \
  --block_size "$BLOCK_SIZE" \
  --num_bands "$NUM_BANDS" \
  --records_per_shard "$RECORDS_PER_SHARD" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --calibration_samples "$CALIBRATION_SAMPLES" \
  --quant_multiplier "$QUANT_MULTIPLIER" \
  --device "$DEVICE" \
  ${MAX_VAL_SAMPLES:+--max_samples "$MAX_VAL_SAMPLES"}

echo "[fast-prep] Done. Store root: $OUT_ROOT"
