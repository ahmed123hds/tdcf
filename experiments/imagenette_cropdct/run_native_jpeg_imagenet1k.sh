#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TRAIN_SHARDS="${TRAIN_SHARDS:-/mnt/edit_disk/imagenet_hf/imagenet1k-train-{0000..1023}.tar}"
VAL_SHARDS="${VAL_SHARDS:-/mnt/edit_disk/imagenet_hf/imagenet1k-validation-{00..63}.tar}"
SPLIT="${SPLIT:-train}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/work/native_jpeg_imagenet1k}"
MAX_IMAGES="${MAX_IMAGES:-1000}"
FIDELITY_SAMPLES="${FIDELITY_SAMPLES:-50}"
LOG_EVERY="${LOG_EVERY:-100}"

if [[ "${SPLIT}" == "train" ]]; then
  SHARDS="${TRAIN_SHARDS}"
elif [[ "${SPLIT}" == "val" ]]; then
  SHARDS="${VAL_SHARDS}"
else
  echo "SPLIT must be train or val, got: ${SPLIT}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 experiments/imagenette_cropdct/native_jpeg_experiment.py \
  --source webdataset \
  --shards "${SHARDS}" \
  --out_dir "${OUT_DIR}" \
  --split "${SPLIT}" \
  --max_images "${MAX_IMAGES}" \
  --fidelity_samples "${FIDELITY_SAMPLES}" \
  --log_every "${LOG_EVERY}"

