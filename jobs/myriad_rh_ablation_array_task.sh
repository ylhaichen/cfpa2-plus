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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${JOB_ID:-na}_${SGE_TASK_ID:-0}"
mkdir -p "${MPLCONFIGDIR}"

TASK_INDEX=0
if [[ -n "${SGE_TASK_ID:-}" ]]; then
  TASK_INDEX=$((SGE_TASK_ID - 1))
fi

IFS=';' read -r -a ABLATION_PLANNERS <<< "${ABLATION_PLANNERS_CSV:-rh_cfpa2;physics_rh_cfpa2}"
IFS=';' read -r -a ABLATION_ENV_CONFIGS <<< "${ABLATION_ENV_CONFIGS_CSV:-configs/env_narrow_t_branches.yaml;configs/env_narrow_t_dense_branches.yaml;configs/env_narrow_t_asymmetric_branches.yaml;configs/env_narrow_t_loop_branches.yaml}"
IFS=';' read -r -a SCORE_MODES <<< "${SCORE_MODES_CSV:-hybrid;immediate_only}"
IFS=';' read -r -a HORIZONS <<< "${HORIZONS_CSV:-3;4;5}"
IFS=';' read -r -a GAMMAS <<< "${GAMMAS_CSV:-0.88;0.92}"
IFS=';' read -r -a IMMEDIATE_WEIGHTS <<< "${IMMEDIATE_WEIGHTS_CSV:-0.85;1.00;1.15}"
IFS=';' read -r -a FUTURE_WEIGHTS <<< "${FUTURE_WEIGHTS_CSV:-0.15;0.25;0.35}"
IFS=';' read -r -a FRONTIER_CONSUMPTION_WEIGHTS <<< "${FRONTIER_CONSUMPTION_WEIGHTS_CSV:-0.10;0.18;0.28}"
IFS=';' read -r -a CONGESTION_SCALES <<< "${CONGESTION_SCALES_CSV:-0.60;1.00}"
IFS=';' read -r -a BRANCH_SCALES <<< "${BRANCH_SCALES_CSV:-0.70;1.00}"
IFS=';' read -r -a TOPK_LIMITS <<< "${TOPK_LIMITS_CSV:-6}"

CMD=(
  python experiments/ablate_rh_hyperparams.py
  --base-config "${BASE_CONFIG:-configs/base.yaml}"
  --planners "${ABLATION_PLANNERS[@]}"
  --env-configs "${ABLATION_ENV_CONFIGS[@]}"
  --seed-start "${SEED_START:-0}"
  --num-seeds "${NUM_SEEDS:-5}"
  --max-steps "${MAX_STEPS:-5000}"
  --run-id "${ABLATION_RUN_ID}"
  --output-root "${OUTPUT_ROOT}"
  --task-index "${TASK_INDEX}"
  --num-tasks "${NUM_TASKS}"
  --score-modes "${SCORE_MODES[@]}"
  --horizons "${HORIZONS[@]}"
  --gammas "${GAMMAS[@]}"
  --immediate-weights "${IMMEDIATE_WEIGHTS[@]}"
  --future-weights "${FUTURE_WEIGHTS[@]}"
  --frontier-consumption-weights "${FRONTIER_CONSUMPTION_WEIGHTS[@]}"
  --congestion-scales "${CONGESTION_SCALES[@]}"
  --branch-scales "${BRANCH_SCALES[@]}"
  --topk-candidate-limits "${TOPK_LIMITS[@]}"
  --max-combos "${MAX_COMBOS:-64}"
  --disable-animation
)

if [[ -n "${PHYSICS_WEIGHT_FILE:-}" ]]; then
  CMD+=(--physics-weight-file "${PHYSICS_WEIGHT_FILE}")
fi

"${CMD[@]}"
