#!/bin/bash
# Build three random 10k corrected CropDCT subsets for storage/fidelity checks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUT_ROOT_BASE="${OUT_ROOT_BASE:-/mnt/dataset_disk/imagenet1k_cropdct_random10k_int16}"
SEEDS="${SEEDS:-42 123 999}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-1000}"
QUALITY="${QUALITY:-95}"
DEVICE="${DEVICE:-cpu}"
RECORDS_PER_SHARD="${RECORDS_PER_SHARD:-1024}"
TILE_BLOCKS="${TILE_BLOCKS:-32}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-1}"
SHUFFLE_BUFFER="${SHUFFLE_BUFFER:-10000}"

echo "========================================================================"
echo "CropDCT random 10k x3 int16-AC subset builds"
echo "base=${OUT_ROOT_BASE}"
echo "seeds=${SEEDS}"
echo "train_samples=${MAX_SAMPLES} val_samples=${MAX_VAL_SAMPLES}"
echo "quality=${QUALITY} tile_blocks=${TILE_BLOCKS} device=${DEVICE}"
echo "========================================================================"

for seed in ${SEEDS}; do
  export OUT_ROOT="${OUT_ROOT_BASE}_seed${seed}"
  export SEED="${seed}"
  export RANDOM_SUBSET=1
  export SHUFFLE_BUFFER
  export MAX_SAMPLES
  export MAX_VAL_SAMPLES
  export QUALITY
  export DEVICE
  export RECORDS_PER_SHARD
  export TILE_BLOCKS
  export COMPRESSION_LEVEL

  echo "========================================================================"
  echo "[cropdct-random10k] Building seed=${seed} at ${OUT_ROOT}"
  echo "========================================================================"
  bash "${SCRIPT_DIR}/run_prepare_imagenet1k_cropdct_store.sh"
done

python3 - <<'PY'
import json
import os
import statistics

base = os.environ["OUT_ROOT_BASE"]
seeds = os.environ["SEEDS"].split()
train_payloads = []
for seed in seeds:
    root = f"{base}_seed{seed}"
    split_root = os.path.join(root, "train")
    with open(os.path.join(split_root, "metadata.json")) as f:
        meta = json.load(f)
    payload = sum(
        os.path.getsize(os.path.join(split_root, shard["dir"], "payload.bin"))
        for shard in meta["shards"]
    )
    gb = payload / 1e9
    train_payloads.append(gb)
    print(
        f"[cropdct-random10k] seed={seed} train_samples={meta['num_images']} "
        f"payload_gb={gb:.4f} dtype={meta.get('coefficient_dtype')} ac={meta.get('ac_storage_dtype')}"
    )

mean = statistics.mean(train_payloads)
std = statistics.pstdev(train_payloads) if len(train_payloads) > 1 else 0.0
full_train_mean = mean * (1281167 / int(os.environ["MAX_SAMPLES"]))
full_train_std = std * (1281167 / int(os.environ["MAX_SAMPLES"]))
print(f"[cropdct-random10k] payload_gb_mean={mean:.4f} std={std:.4f}")
print(f"[cropdct-random10k] estimated_full_train_gb={full_train_mean:.1f} ± {full_train_std:.1f}")
PY
