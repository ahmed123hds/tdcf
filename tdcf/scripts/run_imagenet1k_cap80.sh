#!/bin/bash
# =============================================================================
# Auto-restart wrapper for ImageNet-1K TDCF cap80 training.
# Handles Google Cloud TPU preemptions by automatically resuming from
# the latest checkpoint saved in --save_dir.
# Usage:  bash tdcf/scripts/run_imagenet1k_cap80.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PACKAGE_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export PJRT_DEVICE=TPU
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

TRAIN_SHARDS="${TRAIN_SHARDS:-/mnt/dataset_disk/imagenet_hf/imagenet1k-train-{0000..1023}.tar}"
VAL_SHARDS="${VAL_SHARDS:-/mnt/dataset_disk/imagenet_hf/imagenet1k-validation-{00..63}.tar}"
SAVE_DIR="${SAVE_DIR:-./results/imagenet1k_cap80_mild}"
BETA="${BETA:-0.55}"
MAX_BETA="${MAX_BETA:-0.8}"
GAMMA="${GAMMA:-1.0}"
MAX_RETRIES=20
RETRY_WAIT=30

mkdir -p "$SAVE_DIR"
mkdir -p logs

attempt=0

while [ $attempt -lt $MAX_RETRIES ]; do
    attempt=$((attempt + 1))
    echo "============================================================"
    echo "[$(date)] Attempt $attempt / $MAX_RETRIES"
    echo "============================================================"

    if [ -f "$SAVE_DIR/latest.pt" ]; then
        RESUME_FLAG="--resume"
        echo "[$(date)] Found existing checkpoint. Resuming from $SAVE_DIR ..."
    else
        RESUME_FLAG=""
        echo "[$(date)] No checkpoint found. Starting fresh run ..."
    fi

    python3 -m tdcf.train_tpu_imagenet1k \
        --train_shards "$TRAIN_SHARDS" \
        --val_shards   "$VAL_SHARDS" \
        --backbone resnet50 \
        --img_size 224 --block_size 16 --num_bands 16 \
        --n_classes 1000 \
        --batch_size 128 --eval_batch_size 128 \
        --epochs 100 --pilot_epochs 10 --pilot_ratio 0.10 \
        --base_lr 0.1 --lr_ref_batch 256 \
        --weight_decay 1e-4 --warmup_epochs 5 \
        --label_smooth 0.1 --grad_clip 1.0 \
        --amp_bf16 --num_workers 32 \
        --budget_mode --beta "$BETA" --max_beta "$MAX_BETA" --gamma "$GAMMA" \
        --k_low 1 \
        --save_dir "$SAVE_DIR" \
        $RESUME_FLAG \
        2>&1 | tee -a "logs/cap80_run_attempt_${attempt}.log" \
    && break

    EXIT_CODE=$?
    echo "[$(date)] Training exited with code $EXIT_CODE."

    if grep -q "Training complete" "logs/cap80_run_attempt_${attempt}.log" 2>/dev/null; then
        echo "[$(date)] Training completed successfully!"
        break
    fi

    echo "[$(date)] Preemption detected. Waiting ${RETRY_WAIT}s before restart..."
    sleep $RETRY_WAIT
done

echo "[$(date)] Done. Check $SAVE_DIR for final checkpoint."
