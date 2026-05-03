#!/usr/bin/env bash
set -euo pipefail

CAP="${1:-}"
if [[ -z "${CAP}" ]]; then
  echo "Usage: $0 <100|90|80|70|60>"
  exit 1
fi

case "${CAP}" in
  100)
    BETA="${BETA:-1.0}"
    MAX_BETA="${MAX_BETA:-1.0}"
    GAMMA="${GAMMA:-1.0}"
    ;;
  90)
    BETA="${BETA:-0.4}"
    MAX_BETA="${MAX_BETA:-0.9}"
    GAMMA="${GAMMA:-1.5}"
    ;;
  80)
    BETA="${BETA:-0.4}"
    MAX_BETA="${MAX_BETA:-0.8}"
    GAMMA="${GAMMA:-1.5}"
    ;;
  70)
    BETA="${BETA:-0.4}"
    MAX_BETA="${MAX_BETA:-0.7}"
    GAMMA="${GAMMA:-1.5}"
    ;;
  60)
    BETA="${BETA:-0.4}"
    MAX_BETA="${MAX_BETA:-0.6}"
    GAMMA="${GAMMA:-1.5}"
    ;;
  *)
    echo "Unknown cap '${CAP}'. Expected one of: 100, 90, 80, 70, 60"
    exit 1
    ;;
esac

export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
export PYTHONPATH="${PYTHONPATH:-.}"

ROOT="${ROOT:-./results/ablation_studies/tiny_imagenet}"
DATA_DIR="${DATA_DIR:-./data/tiny_imagenet_block_store}"
SAVE_DIR="${SAVE_DIR:-${ROOT}/tdcf_cap${CAP}}"
EPOCHS="${EPOCHS:-100}"
PILOT_EPOCHS="${PILOT_EPOCHS:-10}"
PILOT_RATIO="${PILOT_RATIO:-0.10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
LR="${LR:-0.1}"
WD="${WD:-5e-4}"
SEED="${SEED:-42}"
K_LOW="${K_LOW:-1}"
PATCH_POLICY="${PATCH_POLICY:-greedy}"
DATA_WORKERS="${DATA_WORKERS:-0}"

mkdir -p "${SAVE_DIR}"

echo "========================================================================"
echo "Tiny-ImageNet TPU ablation | cap=${CAP} | beta=${BETA} max_beta=${MAX_BETA} gamma=${GAMMA}"
echo "data_dir=${DATA_DIR}"
echo "save_dir=${SAVE_DIR}"
echo "epochs=${EPOCHS} pilot_epochs=${PILOT_EPOCHS} batch_size=${BATCH_SIZE}"
echo "========================================================================"

python3 -m tdcf.train_tpu_tiny_imagenet \
  --data_dir "${DATA_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --total_epochs "${EPOCHS}" \
  --pilot_epochs "${PILOT_EPOCHS}" \
  --pilot_ratio "${PILOT_RATIO}" \
  --batch_size "${BATCH_SIZE}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --lr "${LR}" \
  --wd "${WD}" \
  --seed "${SEED}" \
  --k_low "${K_LOW}" \
  --patch_policy "${PATCH_POLICY}" \
  --budget_mode \
  --beta "${BETA}" \
  --max_beta "${MAX_BETA}" \
  --gamma "${GAMMA}" \
  --data_workers "${DATA_WORKERS}" \
  2>&1 | tee "${SAVE_DIR}/train.log"
