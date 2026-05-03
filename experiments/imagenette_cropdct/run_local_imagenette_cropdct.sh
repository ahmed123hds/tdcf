#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/work/data}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/work/runs}"
SIZE="${SIZE:-full}"
QUALITY="${QUALITY:-95}"
TILE_BLOCKS="${TILE_BLOCKS:-32}"
RECORDS_PER_SHARD="${RECORDS_PER_SHARD:-1024}"
MAX_TRAIN="${MAX_TRAIN:-1000}"
MAX_VAL="${MAX_VAL:-200}"
FIDELITY_SAMPLES="${FIDELITY_SAMPLES:-200}"
ENTROPY_SAMPLES="${ENTROPY_SAMPLES:-200}"
DEVICE="${DEVICE:-cpu}"

mkdir -p "${DATA_ROOT}" "${OUT_ROOT}"

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 experiments/imagenette_cropdct/run_imagenette_cropdct.py \
  --data_root "${DATA_ROOT}" \
  --out_root "${OUT_ROOT}" \
  --size "${SIZE}" \
  --quality "${QUALITY}" \
  --tile_blocks "${TILE_BLOCKS}" \
  --records_per_shard "${RECORDS_PER_SHARD}" \
  --max_train "${MAX_TRAIN}" \
  --max_val "${MAX_VAL}" \
  --fidelity_samples "${FIDELITY_SAMPLES}" \
  --entropy_samples "${ENTROPY_SAMPLES}" \
  --device "${DEVICE}" \
  --download

