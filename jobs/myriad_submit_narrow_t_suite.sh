#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PHYSICS_WEIGHT_FILE="${PHYSICS_WEIGHT_FILE:-$HOME/physics_runs/full_20260312_172122/models/physics_residual_mlp.pt}"
SUITE_TAG="${SUITE_TAG:-narrow_t_suite_$(date +%Y%m%d_%H%M%S)}"

echo "Submitting narrow-T compare benchmark..."
REPO_DIR="${REPO_DIR}" \
PHYSICS_WEIGHT_FILE="${PHYSICS_WEIGHT_FILE}" \
BENCHMARK_TAG="${BENCHMARK_TAG:-${SUITE_TAG}_bench}" \
NUM_TASKS="${BENCH_NUM_TASKS:-36}" \
NUM_SEEDS="${BENCH_NUM_SEEDS:-10}" \
MAX_STEPS="${BENCH_MAX_STEPS:-5000}" \
RUN_COMPARE="${RUN_COMPARE:-1}" \
RUN_PREDICTORS="${RUN_PREDICTORS:-1}" \
bash "${REPO_DIR}/jobs/myriad_submit_parallel_benchmark.sh"

echo
echo "Submitting RH ablation..."
REPO_DIR="${REPO_DIR}" \
PHYSICS_WEIGHT_FILE="${PHYSICS_WEIGHT_FILE}" \
ABLATION_TAG="${ABLATION_TAG:-${SUITE_TAG}_ablate}" \
NUM_TASKS="${ABLATE_NUM_TASKS:-36}" \
NUM_SEEDS="${ABLATE_NUM_SEEDS:-5}" \
MAX_STEPS="${ABLATE_MAX_STEPS:-5000}" \
MAX_COMBOS="${ABLATE_MAX_COMBOS:-64}" \
bash "${REPO_DIR}/jobs/myriad_submit_rh_ablation.sh"
