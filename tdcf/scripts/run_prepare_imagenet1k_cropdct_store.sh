#!/bin/bash
# Build the CropDCT tile × frequency-band store for ImageNet-1K.

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

OUT_ROOT="${OUT_ROOT:-/mnt/dataset_disk/imagenet1k_cropdct}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
TILE_BLOCKS="${TILE_BLOCKS:-32}"
QUALITY="${QUALITY:-95}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-1}"
RECORDS_PER_SHARD="${RECORDS_PER_SHARD:-1024}"
DEVICE="${DEVICE:-cpu}"
MAX_LONGER="${MAX_LONGER:-0}"
RANDOM_SUBSET="${RANDOM_SUBSET:-0}"
SHUFFLE_BUFFER="${SHUFFLE_BUFFER:-10000}"
SEED="${SEED:-42}"

mkdir -p "$OUT_ROOT"

echo "[cropdct-prep] Building train store at $OUT_ROOT/train"
python3 -m tdcf.prepare_cropdct_store \
  --source webdataset \
  --shards "$TRAIN_SHARDS" \
  --out_dir "$OUT_ROOT/train" \
  --records_per_shard "$RECORDS_PER_SHARD" \
  --block_size "$BLOCK_SIZE" \
  --tile_blocks "$TILE_BLOCKS" \
  --quality "$QUALITY" \
  --compression zstd \
  --compression_level "$COMPRESSION_LEVEL" \
  --max_longer "$MAX_LONGER" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --shuffle_buffer "$SHUFFLE_BUFFER" \
  $([[ "$RANDOM_SUBSET" == "1" ]] && echo "--random_subset") \
  ${MAX_SAMPLES:+--max_samples "$MAX_SAMPLES"}

echo "[cropdct-prep] Building val store at $OUT_ROOT/val"
python3 -m tdcf.prepare_cropdct_store \
  --source webdataset \
  --shards "$VAL_SHARDS" \
  --out_dir "$OUT_ROOT/val" \
  --records_per_shard "$RECORDS_PER_SHARD" \
  --block_size "$BLOCK_SIZE" \
  --tile_blocks "$TILE_BLOCKS" \
  --quality "$QUALITY" \
  --compression zstd \
  --compression_level "$COMPRESSION_LEVEL" \
  --max_longer "$MAX_LONGER" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --shuffle_buffer "$SHUFFLE_BUFFER" \
  $([[ "$RANDOM_SUBSET" == "1" ]] && echo "--random_subset") \
  ${MAX_VAL_SAMPLES:+--max_samples "$MAX_VAL_SAMPLES"}

echo "[cropdct-prep] Done. Store root: $OUT_ROOT"
