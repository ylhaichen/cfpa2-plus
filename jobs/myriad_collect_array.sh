#!/bin/bash -l
set -euo pipefail

: "${REPO_DIR:?REPO_DIR is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${NUM_TASKS:?NUM_TASKS is required}"

BASE_CONFIG="${BASE_CONFIG:-configs/base.yaml}"
PLANNER_CONFIG="${PLANNER_CONFIG:-configs/planner_rh_cfpa2.yaml}"
PLANNER_NAME="${PLANNER_NAME:-rh_cfpa2}"
PREDICTOR_TYPE="${PREDICTOR_TYPE:-path_follow}"
SEED_START="${SEED_START:-0}"
NUM_SEEDS="${NUM_SEEDS:-4000}"
EPISODES_PER_SEED="${EPISODES_PER_SEED:-1}"
MAX_STEPS="${MAX_STEPS:-450}"
SHARD_SIZE="${SHARD_SIZE:-200000}"
HARD_OVERSAMPLE="${HARD_OVERSAMPLE:-0.70}"
HARD_MAP_TYPES_CSV="${HARD_MAP_TYPES_CSV:-sharp_turn_corridor,narrow_t_branches,bottleneck_rooms,interaction_cross,branching_deadend}"
ENV_CFGS_CSV="${ENV_CFGS_CSV:-configs/env_narrow_t_branches.yaml,configs/env_narrow_t_dense_branches.yaml,configs/env_narrow_t_asymmetric_branches.yaml,configs/env_narrow_t_loop_branches.yaml}"
VENV_PATH="${VENV_PATH:-}"

if [[ -n "${VENV_PATH}" && -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

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

IFS=',' read -r -a ENV_CFGS <<< "${ENV_CFGS_CSV}"
IFS=',' read -r -a HARD_MAP_TYPES <<< "${HARD_MAP_TYPES_CSV}"

echo "[collect] host=$(hostname) task_index=${TASK_INDEX}/${NUM_TASKS} seed_start=${SEED_START} num_seeds=${NUM_SEEDS}"
echo "[collect] output_dir=${RUN_ROOT}/dataset"

python training/collect_physics_residual_dataset.py \
  --base-config "${BASE_CONFIG}" \
  --planner-config "${PLANNER_CONFIG}" \
  --env-configs "${ENV_CFGS[@]}" \
  --planner-name "${PLANNER_NAME}" \
  --predictor-type "${PREDICTOR_TYPE}" \
  --seed-start "${SEED_START}" \
  --num-seeds "${NUM_SEEDS}" \
  --episodes-per-seed "${EPISODES_PER_SEED}" \
  --max-steps "${MAX_STEPS}" \
  --task-index "${TASK_INDEX}" \
  --num-tasks "${NUM_TASKS}" \
  --shard-size "${SHARD_SIZE}" \
  --hard-scenario-oversample-prob "${HARD_OVERSAMPLE}" \
  --hard-scenario-map-types "${HARD_MAP_TYPES[@]}" \
  --output-dir "${RUN_ROOT}/dataset"
