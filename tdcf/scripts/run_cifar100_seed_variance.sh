#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-./results/cifar100_ablations}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${ROOT}"

run_seed_pair() {
  local seed="$1"

  "${PYTHON_BIN}" -m tdcf.train_cifar100 \
    --backbone cnn \
    --total_epochs 100 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 5e-4 \
    --seed "${seed}" \
    --data_workers 4 \
    --baseline_only \
    --save_dir "${ROOT}/cnn_baseline_seed${seed}"

  "${PYTHON_BIN}" -m tdcf.train_cifar100 \
    --backbone cnn \
    --total_epochs 100 \
    --pilot_epochs 10 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 5e-4 \
    --seed "${seed}" \
    --data_workers 4 \
    --block_dct \
    --k_low 1 \
    --patch_policy greedy \
    --budget_mode \
    --beta 0.4 \
    --max_beta 0.9 \
    --gamma 1.5 \
    --save_dir "${ROOT}/cnn_tdcf_cap90_seed${seed}"
}

for seed in 123 999; do
  run_seed_pair "${seed}"
done
