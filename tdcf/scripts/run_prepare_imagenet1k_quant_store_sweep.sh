#!/bin/bash
# Build small quantized-store subsets for B in {8,16,32,64}.
# Use this before committing to a full ImageNet store.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PACKAGE_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [ -z "${TRAIN_SHARDS:-}" ]; then
    TRAIN_SHARDS='/mnt/dataset_disk/imagenet_hf/imagenet1k-train-{0000..1023}.tar'
fi

OUT_ROOT="${OUT_ROOT:-/mnt/dataset_disk/imagenet1k_quant_bucket_sweep}"
MAX_SAMPLES="${MAX_SAMPLES:-20000}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-2048}"
DEVICE="${DEVICE:-cuda}"
BUCKETS="${BUCKETS:-8 16 32 64}"

mkdir -p "$OUT_ROOT"

for B in $BUCKETS; do
    OUT_DIR="$OUT_ROOT/B${B}"
    echo "============================================================"
    echo "[bucket-sweep] Building B=$B subset at $OUT_DIR"
    echo "============================================================"
    SECONDS=0
    python3 -m tdcf.prepare_imagenet1k_quantized_store \
        --source webdataset \
        --shards "$TRAIN_SHARDS" \
        --out_dir "$OUT_DIR" \
        --block_size "${BLOCK_SIZE:-8}" \
        --num_bands "${NUM_BANDS:-16}" \
        --tile_blocks "${TILE_BLOCKS:-4}" \
        --chunk_size "${CHUNK_SIZE:-128}" \
        --bucket_count "$B" \
        --batch_size "${BATCH_SIZE:-64}" \
        --num_workers "${NUM_WORKERS:-8}" \
        --calibration_samples "$CALIBRATION_SAMPLES" \
        --quant_multiplier "${QUANT_MULTIPLIER:-1.05}" \
        --compression_level "${COMPRESSION_LEVEL:-6}" \
        --max_samples "$MAX_SAMPLES" \
        --device "$DEVICE"
    echo "[bucket-sweep] B=$B build_time=${SECONDS}s"
    du -sh "$OUT_DIR"
done
