#!/bin/bash -l
set -euo pipefail

: "${REPO_DIR:?REPO_DIR is required}"
: "${CONDA_SH:?CONDA_SH is required}"
: "${CONDA_ENV:?CONDA_ENV is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${COMPARE_RUN_ID:?COMPARE_RUN_ID is required}"
: "${PREDICTOR_RUN_ID:?PREDICTOR_RUN_ID is required}"
: "${NUM_TASKS:?NUM_TASKS is required}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}"

if [[ "${RUN_COMPARE:-1}" == "1" ]]; then
  python experiments/merge_compare_planners_shards.py \
    --run-id "${COMPARE_RUN_ID}" \
    --output-root "${OUTPUT_ROOT}" \
    --expected-shards "${NUM_TASKS}" \
    --fail-missing
fi

if [[ "${RUN_PREDICTORS:-1}" == "1" ]]; then
  python experiments/merge_compare_predictors_shards.py \
    --run-id "${PREDICTOR_RUN_ID}" \
    --output-root "${OUTPUT_ROOT}" \
    --expected-shards "${NUM_TASKS}" \
    --fail-missing
fi
