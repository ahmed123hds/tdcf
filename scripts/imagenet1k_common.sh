#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_SHARDS="${TRAIN_SHARDS:-/home/filliones/streaming_dataset/imagenet_hf/imagenet1k-train-{0000..1023}.tar}"
VAL_SHARDS="${VAL_SHARDS:-/home/filliones/streaming_dataset/imagenet_hf/imagenet1k-validation-{00..63}.tar}"
RESULTS_ROOT="${RESULTS_ROOT:-./results}"
BACKBONE="${BACKBONE:-resnet50}"
IMG_SIZE="${IMG_SIZE:-224}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
NUM_BANDS="${NUM_BANDS:-16}"
N_CLASSES="${N_CLASSES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-100}"
PILOT_EPOCHS="${PILOT_EPOCHS:-10}"
PILOT_RATIO="${PILOT_RATIO:-0.10}"
BASE_LR="${BASE_LR:-0.1}"
LR_REF_BATCH="${LR_REF_BATCH:-256}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
LABEL_SMOOTH="${LABEL_SMOOTH:-0.1}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
K_LOW="${K_LOW:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"

common_imagenet_args=(
  -m tdcf.train_tpu_imagenet1k
  --train_shards "${TRAIN_SHARDS}"
  --val_shards "${VAL_SHARDS}"
  --backbone "${BACKBONE}"
  --img_size "${IMG_SIZE}"
  --block_size "${BLOCK_SIZE}"
  --num_bands "${NUM_BANDS}"
  --n_classes "${N_CLASSES}"
  --batch_size "${BATCH_SIZE}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --base_lr "${BASE_LR}"
  --lr_ref_batch "${LR_REF_BATCH}"
  --weight_decay "${WEIGHT_DECAY}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --label_smooth "${LABEL_SMOOTH}"
  --grad_clip "${GRAD_CLIP}"
  --amp_bf16
  --num_workers "${NUM_WORKERS}"
)

run_imagenet_baseline() {
  local save_dir="$1"
  PJRT_DEVICE=TPU PYTHONPATH=. "${PYTHON_BIN}" "${common_imagenet_args[@]}" \
    --skip_pilot \
    --save_dir "${save_dir}"
}

run_imagenet_budgeted() {
  local save_dir="$1"
  local beta="$2"
  local max_beta="$3"
  local gamma="$4"

  PJRT_DEVICE=TPU PYTHONPATH=. "${PYTHON_BIN}" "${common_imagenet_args[@]}" \
    --pilot_epochs "${PILOT_EPOCHS}" \
    --pilot_ratio "${PILOT_RATIO}" \
    --budget_mode \
    --beta "${beta}" \
    --max_beta "${max_beta}" \
    --gamma "${gamma}" \
    --k_low "${K_LOW}" \
    --save_dir "${save_dir}"
}
