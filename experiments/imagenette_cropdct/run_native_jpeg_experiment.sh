#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/work/data}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/work/native_jpeg}"
SIZE="${SIZE:-full}"
SPLIT="${SPLIT:-train}"
MAX_IMAGES="${MAX_IMAGES:-1000}"
FIDELITY_SAMPLES="${FIDELITY_SAMPLES:-50}"
LOG_EVERY="${LOG_EVERY:-100}"

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 experiments/imagenette_cropdct/native_jpeg_experiment.py \
  --data_root "${DATA_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --size "${SIZE}" \
  --split "${SPLIT}" \
  --max_images "${MAX_IMAGES}" \
  --fidelity_samples "${FIDELITY_SAMPLES}" \
  --log_every "${LOG_EVERY}"

