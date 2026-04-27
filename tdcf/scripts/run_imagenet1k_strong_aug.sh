#!/bin/bash
# =============================================================================
# Auto-restart wrapper for ImageNet-1K strong-augmentation experiments.
#
# Usage:
#   bash tdcf/scripts/run_imagenet1k_strong_aug.sh baseline resnet50
#   bash tdcf/scripts/run_imagenet1k_strong_aug.sh cap90    resnet50
#   bash tdcf/scripts/run_imagenet1k_strong_aug.sh cap90    vit_b16
#
# Modes: baseline, cap100, cap90, cap80, cap70, cap60
# Backbones: resnet50, vit_b16
# =============================================================================

set -euo pipefail

MODE="${1:-cap90}"
BACKBONE="${2:-resnet50}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PACKAGE_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export PJRT_DEVICE=TPU
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [ -z "${TRAIN_SHARDS:-}" ]; then
    TRAIN_SHARDS='/mnt/dataset_disk/imagenet_hf/imagenet1k-train-{0000..1023}.tar'
fi
if [ -z "${VAL_SHARDS:-}" ]; then
    VAL_SHARDS='/mnt/dataset_disk/imagenet_hf/imagenet1k-validation-{00..63}.tar'
fi

MAX_RETRIES="${MAX_RETRIES:-20}"
RETRY_WAIT="${RETRY_WAIT:-30}"
EPOCHS="${EPOCHS:-100}"
PILOT_EPOCHS="${PILOT_EPOCHS:-10}"
PILOT_RATIO="${PILOT_RATIO:-0.10}"

# Conservative 100-epoch strong recipe. The full TorchVision recipe uses much
# longer training; this version is meant to improve reviewer perception without
# turning this into a 600-epoch benchmark run.
AUTO_AUGMENT="${AUTO_AUGMENT:-randaugment}"
RANDOM_ERASE="${RANDOM_ERASE:-0.1}"
MIXUP_ALPHA="${MIXUP_ALPHA:-0.1}"
CUTMIX_ALPHA="${CUTMIX_ALPHA:-0.0}"
CUTMIX_PROB="${CUTMIX_PROB:-0.0}"
LABEL_SMOOTH="${LABEL_SMOOTH:-0.1}"
VAL_RESIZE_SIZE="${VAL_RESIZE_SIZE:-232}"
EMA_DECAY="${EMA_DECAY:-0.9999}"
EMA_STEPS="${EMA_STEPS:-32}"

case "$BACKBONE" in
    resnet50)
        BATCH_SIZE="${BATCH_SIZE:-128}"
        EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
        BASE_LR="${BASE_LR:-0.125}"
        LR_REF_BATCH="${LR_REF_BATCH:-256}"
        WEIGHT_DECAY="${WEIGHT_DECAY:-2e-5}"
        WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
        ;;
    vit_b16)
        BATCH_SIZE="${BATCH_SIZE:-32}"
        EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
        BASE_LR="${BASE_LR:-3e-4}"
        LR_REF_BATCH="${LR_REF_BATCH:-256}"
        WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
        WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
        ;;
    *)
        echo "Unknown backbone: $BACKBONE"
        exit 2
        ;;
esac

RUN_FLAGS=()
case "$MODE" in
    baseline)
        SAVE_DIR="${SAVE_DIR:-./results/imagenet1k_${BACKBONE}_strong_aug_baseline}"
        RUN_FLAGS+=(--skip_pilot)
        ;;
    cap100)
        SAVE_DIR="${SAVE_DIR:-./results/imagenet1k_${BACKBONE}_strong_aug_cap100}"
        RUN_FLAGS+=(--budget_mode --beta "${BETA:-1.0}" --max_beta "${MAX_BETA:-1.0}" --gamma "${GAMMA:-1.0}")
        ;;
    cap90)
        SAVE_DIR="${SAVE_DIR:-./results/imagenet1k_${BACKBONE}_strong_aug_cap90}"
        RUN_FLAGS+=(--budget_mode --beta "${BETA:-0.6}" --max_beta "${MAX_BETA:-0.9}" --gamma "${GAMMA:-1.0}")
        ;;
    cap80)
        SAVE_DIR="${SAVE_DIR:-./results/imagenet1k_${BACKBONE}_strong_aug_cap80}"
        RUN_FLAGS+=(--budget_mode --beta "${BETA:-0.6}" --max_beta "${MAX_BETA:-0.8}" --gamma "${GAMMA:-1.0}")
        ;;
    cap70)
        SAVE_DIR="${SAVE_DIR:-./results/imagenet1k_${BACKBONE}_strong_aug_cap70}"
        RUN_FLAGS+=(--budget_mode --beta "${BETA:-0.55}" --max_beta "${MAX_BETA:-0.7}" --gamma "${GAMMA:-1.0}")
        ;;
    cap60)
        SAVE_DIR="${SAVE_DIR:-./results/imagenet1k_${BACKBONE}_strong_aug_cap60}"
        RUN_FLAGS+=(--budget_mode --beta "${BETA:-0.5}" --max_beta "${MAX_BETA:-0.6}" --gamma "${GAMMA:-1.0}")
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
    echo "[$(date)] ${BACKBONE} ${MODE} strong-aug attempt $attempt / $MAX_RETRIES"
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
        --backbone "$BACKBONE" \
        --img_size 224 --block_size 16 --num_bands 16 \
        --n_classes 1000 \
        --batch_size "$BATCH_SIZE" --eval_batch_size "$EVAL_BATCH_SIZE" \
        --epochs "$EPOCHS" --pilot_epochs "$PILOT_EPOCHS" --pilot_ratio "$PILOT_RATIO" \
        --base_lr "$BASE_LR" --lr_ref_batch "$LR_REF_BATCH" \
        --weight_decay "$WEIGHT_DECAY" --warmup_epochs "$WARMUP_EPOCHS" \
        --label_smooth "$LABEL_SMOOTH" --grad_clip 1.0 \
        --auto_augment "$AUTO_AUGMENT" \
        --random_erase "$RANDOM_ERASE" \
        --mixup_alpha "$MIXUP_ALPHA" \
        --cutmix_alpha "$CUTMIX_ALPHA" \
        --cutmix_prob "$CUTMIX_PROB" \
        --model_ema --model_ema_decay "$EMA_DECAY" --model_ema_steps "$EMA_STEPS" \
        --val_resize_size "$VAL_RESIZE_SIZE" \
        --amp_bf16 --num_workers 32 \
        --k_low 1 \
        --save_dir "$SAVE_DIR" \
        "${RUN_FLAGS[@]}" \
        $RESUME_FLAG \
        2>&1 | tee -a "logs/${BACKBONE}_strong_aug_${MODE}_attempt_${attempt}.log" \
    && break

    EXIT_CODE=$?
    echo "[$(date)] Training exited with code $EXIT_CODE."

    if grep -q "Training complete" "logs/${BACKBONE}_strong_aug_${MODE}_attempt_${attempt}.log" 2>/dev/null; then
        echo "[$(date)] Training completed successfully!"
        break
    fi

    echo "[$(date)] Preemption detected. Waiting ${RETRY_WAIT}s before restart..."
    sleep "$RETRY_WAIT"
done

echo "[$(date)] Done. Check $SAVE_DIR for final checkpoint."
