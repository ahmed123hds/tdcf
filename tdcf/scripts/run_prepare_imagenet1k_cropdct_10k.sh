#!/bin/bash
# Build a small corrected CropDCT ImageNet-1K subset for fidelity testing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUT_ROOT="${OUT_ROOT:-/mnt/dataset_disk/imagenet1k_cropdct_10k_int16}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-1000}"
QUALITY="${QUALITY:-95}"
DEVICE="${DEVICE:-cpu}"
RECORDS_PER_SHARD="${RECORDS_PER_SHARD:-1024}"
TILE_BLOCKS="${TILE_BLOCKS:-32}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-1}"

export OUT_ROOT
export MAX_SAMPLES
export MAX_VAL_SAMPLES
export QUALITY
export DEVICE
export RECORDS_PER_SHARD
export TILE_BLOCKS
export COMPRESSION_LEVEL

echo "========================================================================"
echo "CropDCT 10k int16-AC subset build"
echo "out=${OUT_ROOT}"
echo "train_samples=${MAX_SAMPLES} val_samples=${MAX_VAL_SAMPLES}"
echo "quality=${QUALITY} tile_blocks=${TILE_BLOCKS} device=${DEVICE}"
echo "========================================================================"

bash "${SCRIPT_DIR}/run_prepare_imagenet1k_cropdct_store.sh"

python3 - <<'PY'
import json
import os

root = os.environ["OUT_ROOT"]
for split in ("train", "val"):
    split_root = os.path.join(root, split)
    with open(os.path.join(split_root, "metadata.json")) as f:
        meta = json.load(f)
    payload = sum(
        os.path.getsize(os.path.join(split_root, shard["dir"], "payload.bin"))
        for shard in meta["shards"]
    )
    print(
        f"[cropdct-10k] {split}: samples={meta['num_images']} "
        f"payload_gb={payload / 1e9:.4f} "
        f"dtype={meta.get('coefficient_dtype')} ac={meta.get('ac_storage_dtype')}"
    )
PY
