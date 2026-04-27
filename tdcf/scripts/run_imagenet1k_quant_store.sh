#!/bin/bash
# =============================================================================
# Auto-restart wrapper for ImageNet-1K compressed quantized DCT-store training.
#
# Usage:
#   bash tdcf/scripts/run_imagenet1k_quant_store.sh cap90
#   bash tdcf/scripts/run_imagenet1k_quant_store.sh cap100
#
# Modes: cap100, cap90, cap80, cap70, cap60
# =============================================================================

set -euo pipefail

MODE="${1:-cap90}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PACKAGE_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export PJRT_DEVICE=TPU
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DATA_DIR="${DATA_DIR:-/mnt/dataset_disk/imagenet1k_quant_dct_original}"
MAX_RETRIES="${MAX_RETRIES:-20}"
RETRY_WAIT="${RETRY_WAIT:-30}"

case "$MODE" in
    cap100)
        SAVE_DIR="${SAVE_DIR:-./results/imagenet1k_quant_store_cap100}"
        BETA="${BETA:-1.0}"
        MAX_BETA="${MAX_BETA:-1.0}"
        GAMMA="${GAMMA:-1.0}"
        ;;
    cap90)
        SAVE_DIR="${SAVE_DIR:-./results/imagenet1k_quant_store_cap90}"
        BETA="${BETA:-0.6}"
        MAX_BETA="${MAX_BETA:-0.9}"
        GAMMA="${GAMMA:-1.0}"
        ;;
    cap80)
        SAVE_DIR="${SAVE_DIR:-./results/imagenet1k_quant_store_cap80}"
        BETA="${BETA:-0.6}"
        MAX_BETA="${MAX_BETA:-0.8}"
        GAMMA="${GAMMA:-1.0}"
        ;;
    cap70)
        SAVE_DIR="${SAVE_DIR:-./results/imagenet1k_quant_store_cap70}"
        BETA="${BETA:-0.55}"
        MAX_BETA="${MAX_BETA:-0.7}"
        GAMMA="${GAMMA:-1.0}"
        ;;
    cap60)
        SAVE_DIR="${SAVE_DIR:-./results/imagenet1k_quant_store_cap60}"
        BETA="${BETA:-0.5}"
        MAX_BETA="${MAX_BETA:-0.6}"
        GAMMA="${GAMMA:-1.0}"
        ;;
    *)
        echo "Unknown mode: $MODE"
        exit 2
        ;;
esac

mkdir -p "$SAVE_DIR"
mkdir -p logs

attempt=0
while [ "$attempt" -lt "$MAX_RETRIES" ]; do
    attempt=$((attempt + 1))
    echo "============================================================"
    echo "[$(date)] quant-store $MODE attempt $attempt / $MAX_RETRIES"
    echo "============================================================"

    if [ -f "$SAVE_DIR/latest.pt" ]; then
        RESUME_FLAG="--resume"
        echo "[$(date)] Found checkpoint. Resuming from $SAVE_DIR ..."
    else
        RESUME_FLAG=""
        echo "[$(date)] No checkpoint found. Starting fresh run ..."
    fi

    python3 -m tdcf.train_tpu_imagenet1k_quant_store \
        --data_dir "$DATA_DIR" \
        --backbone resnet50 \
        --img_size 224 \
        --n_classes 1000 \
        --batch_size "${BATCH_SIZE:-128}" \
        --eval_batch_size "${EVAL_BATCH_SIZE:-128}" \
        --epochs "${EPOCHS:-100}" \
        --pilot_epochs "${PILOT_EPOCHS:-10}" \
        --pilot_ratio "${PILOT_RATIO:-0.10}" \
        --base_lr "${BASE_LR:-0.1}" \
        --lr_ref_batch "${LR_REF_BATCH:-256}" \
        --weight_decay "${WEIGHT_DECAY:-1e-4}" \
        --warmup_epochs "${WARMUP_EPOCHS:-5}" \
        --label_smooth "${LABEL_SMOOTH:-0.1}" \
        --grad_clip "${GRAD_CLIP:-1.0}" \
        --amp_bf16 \
        --budget_mode \
        --beta "$BETA" \
        --max_beta "$MAX_BETA" \
        --gamma "$GAMMA" \
        --k_low "${K_LOW:-1}" \
        --patch_policy "${PATCH_POLICY:-greedy}" \
        --save_dir "$SAVE_DIR" \
        $RESUME_FLAG \
        2>&1 | tee -a "logs/quant_store_${MODE}_attempt_${attempt}.log" \
    && break

    EXIT_CODE=$?
    echo "[$(date)] Training exited with code $EXIT_CODE."
    if grep -q "Training complete" "logs/quant_store_${MODE}_attempt_${attempt}.log" 2>/dev/null; then
        echo "[$(date)] Training completed successfully!"
        break
    fi
    echo "[$(date)] Waiting ${RETRY_WAIT}s before restart..."
    sleep "$RETRY_WAIT"
done

echo "[$(date)] Done. Check $SAVE_DIR"
