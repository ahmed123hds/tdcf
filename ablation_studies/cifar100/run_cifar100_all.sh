#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for cap in 100 90 80 70 60; do
  bash "${SCRIPT_DIR}/run_cifar100_ablation.sh" "${cap}"
done
