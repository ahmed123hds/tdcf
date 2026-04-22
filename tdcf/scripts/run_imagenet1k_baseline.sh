#!/bin/bash
# =============================================================================
# Auto-restart wrapper for ImageNet-1K baseline training.
# Handles Google Cloud TPU preemptions by automatically resuming from
# the latest checkpoint saved in --save_dir.
# Usage:  bash tdcf/scripts/run_imagenet1k_baseline.sh
# =============================================================================

set -euo pipefail

export PJRT_DEVICE=TPU

TRAIN_SHARDS='/mnt/dataset_disk/imagenet_hf/imagenet1k-train-{0000..1023}.tar'
VAL_SHARDS='/mnt/dataset_disk/imagenet_hf/imagenet1k-validation-{00..63}.tar'
SAVE_DIR='./results/imagenet1k_baseline_full'
MAX_RETRIES=20   # Maximum number of preemption restarts
RETRY_WAIT=30    # Seconds to wait before restarting after a preemption

mkdir -p "$SAVE_DIR"
mkdir -p logs

attempt=0
FIRST_RUN=true

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

    # Run training — exits 0 on success, non-zero on preemption/crash
    python3 -m tdcf.train_tpu_imagenet1k \
        --train_shards "$TRAIN_SHARDS" \
        --val_shards   "$VAL_SHARDS" \
        --backbone resnet50 \
        --img_size 224 --block_size 16 --num_bands 16 \
        --n_classes 1000 \
        --batch_size 128 --eval_batch_size 128 \
        --epochs 100 --base_lr 0.1 --lr_ref_batch 256 \
        --weight_decay 1e-4 --warmup_epochs 5 \
        --label_smooth 0.1 --grad_clip 1.0 \
        --amp_bf16 --num_workers 32 \
        --skip_pilot \
        --save_dir "$SAVE_DIR" \
        $RESUME_FLAG \
        2>&1 | tee -a "logs/baseline_run_attempt_${attempt}.log" \
    && break   # Exit the loop cleanly on success (exit code 0)

    EXIT_CODE=$?
    echo "[$(date)] Training exited with code $EXIT_CODE."

    # Check if the job completed all 100 epochs successfully
    if grep -q "Training complete" "logs/baseline_run_attempt_${attempt}.log" 2>/dev/null; then
        echo "[$(date)] Training completed successfully!"
        break
    fi

    echo "[$(date)] Preemption detected. Waiting ${RETRY_WAIT}s before restart..."
    sleep $RETRY_WAIT
done

echo "[$(date)] Done. Check $SAVE_DIR for final checkpoint."
