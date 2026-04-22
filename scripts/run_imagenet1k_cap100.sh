#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/imagenet1k_common.sh"

mkdir -p "${RESULTS_ROOT}"
run_imagenet_budgeted "${RESULTS_ROOT}/imagenet1k_cap100" "1.0" "1.0" "1.0"
