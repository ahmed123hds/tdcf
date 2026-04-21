#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-./results/cifar100_ablations}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${ROOT}"

common_args=(
  -m tdcf.train_cifar100
  --backbone cnn
  --total_epochs 100
  --pilot_epochs 10
  --batch_size 256
  --lr 0.1
  --wd 5e-4
  --seed 42
  --data_workers 4
)

run_baseline() {
  "${PYTHON_BIN}" \
    -m tdcf.train_cifar100 \
    --backbone cnn \
    --total_epochs 100 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 5e-4 \
    --seed 42 \
    --data_workers 4 \
    --baseline_only \
    --save_dir "${ROOT}/cnn_baseline"
}

run_cap90() {
  "${PYTHON_BIN}" "${common_args[@]}" \
    --block_dct \
    --k_low 1 \
    --patch_policy greedy \
    --budget_mode \
    --beta 0.4 \
    --max_beta 0.9 \
    --gamma 1.5 \
    --save_dir "${ROOT}/cnn_tdcf_cap90"
}

run_fixed90() {
  "${PYTHON_BIN}" "${common_args[@]}" \
    --block_dct \
    --k_low 1 \
    --patch_policy greedy \
    --budget_mode \
    --beta 0.6005 \
    --max_beta 0.6005 \
    --gamma 1.0 \
    --save_dir "${ROOT}/cnn_fixed_match90"
}

run_random90() {
  "${PYTHON_BIN}" "${common_args[@]}" \
    --block_dct \
    --k_low 1 \
    --patch_policy random \
    --budget_mode \
    --beta 0.4 \
    --max_beta 0.9 \
    --gamma 1.5 \
    --save_dir "${ROOT}/cnn_random_match90"
}

run_cap100() {
  "${PYTHON_BIN}" "${common_args[@]}" \
    --block_dct \
    --k_low 1 \
    --patch_policy greedy \
    --budget_mode \
    --beta 1.0 \
    --max_beta 1.0 \
    --gamma 1.0 \
    --save_dir "${ROOT}/cnn_tdcf_cap100"
}

run_cap80() {
  "${PYTHON_BIN}" "${common_args[@]}" \
    --block_dct \
    --k_low 1 \
    --patch_policy greedy \
    --budget_mode \
    --beta 0.4 \
    --max_beta 0.8 \
    --gamma 1.5 \
    --save_dir "${ROOT}/cnn_tdcf_cap80"
}

run_cap70() {
  "${PYTHON_BIN}" "${common_args[@]}" \
    --block_dct \
    --k_low 1 \
    --patch_policy greedy \
    --budget_mode \
    --beta 0.4 \
    --max_beta 0.7 \
    --gamma 1.5 \
    --save_dir "${ROOT}/cnn_tdcf_cap70"
}

run_dashboard() {
  "${PYTHON_BIN}" generate_combined_dashboard.py \
    "${ROOT}/cnn_tdcf_cap90/results.json" \
    "${ROOT}/cnn_baseline/results.json" \
    "${ROOT}/cnn_tdcf_cap90/cifar100_dashboard_combined.png"
}

run_summary() {
  "${PYTHON_BIN}" scripts/summarize_cifar100_ablations.py \
    --results_root "${ROOT}" \
    --write_markdown
}

usage() {
  cat <<'EOF'
Usage: scripts/run_cifar100_ablations.sh <target>

Targets:
  baseline    Run the full-fidelity CNN baseline
  cap90       Run the headline dynamic 90% cap TDCF experiment
  fixed90     Run the matched fixed-budget control
  random90    Run the matched random-policy control
  cap100      Run the 100% budget pipeline control
  cap80       Run the 80% budget Pareto point
  cap70       Run the 70% budget Pareto point
  dashboard   Build the combined baseline-vs-cap90 dashboard
  summary     Build a Markdown summary table from finished runs
  all         Run baseline, cap90, fixed90, random90, cap100, cap80, cap70, dashboard, summary
EOF
}

target="${1:-}"
case "${target}" in
  baseline) run_baseline ;;
  cap90) run_cap90 ;;
  fixed90) run_fixed90 ;;
  random90) run_random90 ;;
  cap100) run_cap100 ;;
  cap80) run_cap80 ;;
  cap70) run_cap70 ;;
  dashboard) run_dashboard ;;
  summary) run_summary ;;
  all)
    run_baseline
    run_cap90
    run_fixed90
    run_random90
    run_cap100
    run_cap80
    run_cap70
    run_dashboard
    run_summary
    ;;
  *)
    usage
    exit 1
    ;;
esac
