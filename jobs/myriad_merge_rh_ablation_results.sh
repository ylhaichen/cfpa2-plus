#!/bin/bash -l
set -euo pipefail

: "${REPO_DIR:?REPO_DIR is required}"
: "${CONDA_SH:?CONDA_SH is required}"
: "${CONDA_ENV:?CONDA_ENV is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${ABLATION_RUN_ID:?ABLATION_RUN_ID is required}"
: "${NUM_TASKS:?NUM_TASKS is required}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}"

python experiments/merge_ablate_rh_shards.py \
  --run-id "${ABLATION_RUN_ID}" \
  --output-root "${OUTPUT_ROOT}" \
  --expected-shards "${NUM_TASKS}" \
  --fail-missing
