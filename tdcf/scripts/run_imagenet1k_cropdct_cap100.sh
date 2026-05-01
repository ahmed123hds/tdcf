#!/usr/bin/env bash
set -u

CAP_NAME=cap100
DATA_DIR="${DATA_DIR:-/mnt/dataset_disk/imagenet1k_cropdct}"
SAVE_DIR="${SAVE_DIR:-./results/imagenet1k_cropdct_${CAP_NAME}}"
LOG_DIR="${LOG_DIR:-./logs}"
BACKBONE="${BACKBONE:-resnet50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-100}"
BASE_LR="${BASE_LR:-0.1}"
LR_REF_BATCH="${LR_REF_BATCH:-1024}"
LOADER_WORKERS="${LOADER_WORKERS:-${DECODE_WORKERS:-4}}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
CPU_THREADS="${CPU_THREADS:-1}"
NUM_ATTEMPTS="${NUM_ATTEMPTS:-20}"

mkdir -p "${LOG_DIR}" "${SAVE_DIR}"

attempt=1
while [ "${attempt}" -le "${NUM_ATTEMPTS}" ]; do
  echo "============================================================"
  echo "[$(date)] CropDCT ${CAP_NAME} attempt ${attempt} / ${NUM_ATTEMPTS}"
  echo "============================================================"
  if [ -f "${SAVE_DIR}/latest.pt" ]; then
    echo "[$(date)] Found checkpoint. Resuming from ${SAVE_DIR}/latest.pt ..."
    RESUME_FLAG="--resume"
  else
    echo "[$(date)] No checkpoint found. Starting fresh run ..."
    RESUME_FLAG=""
  fi

  PJRT_DEVICE=TPU PYTHONPATH=. python3 -m tdcf.train_tpu_imagenet1k_cropdct \
    --data_dir "${DATA_DIR}" \
    --save_dir "${SAVE_DIR}" \
    --backbone "${BACKBONE}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --base_lr "${BASE_LR}" \
    --lr_ref_batch "${LR_REF_BATCH}" \
    --loader_workers "${LOADER_WORKERS}" \
    --prefetch_factor "${PREFETCH_FACTOR}" \
    --cpu_threads "${CPU_THREADS}" \
    --weight_decay 1e-4 \
    --warmup_epochs 5 \
    --label_smooth 0.1 \
    --grad_clip 1.0 \
    --amp_bf16 \
    --cap 100 \
    ${RESUME_FLAG} \
    2>&1 | tee "${LOG_DIR}/cropdct_${CAP_NAME}_attempt_${attempt}.log"

  code=${PIPESTATUS[0]}
  if [ "${code}" -eq 0 ]; then
    echo "[$(date)] CropDCT ${CAP_NAME} completed."
    exit 0
  fi

  echo "[$(date)] Training exited with code ${code}. Waiting 30s before restart..."
  attempt=$((attempt + 1))
  sleep 30
done

echo "[$(date)] CropDCT ${CAP_NAME} failed after ${NUM_ATTEMPTS} attempts."
exit 1
