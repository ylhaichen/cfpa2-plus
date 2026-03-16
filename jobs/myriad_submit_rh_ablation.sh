#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_DIR="${REPO_DIR:-${DEFAULT_REPO_DIR}}"
CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-cfpa2rh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
PHYSICS_WEIGHT_FILE="${PHYSICS_WEIGHT_FILE:-$HOME/physics_runs/full_20260312_172122/models/physics_residual_mlp.pt}"

ABLATION_TAG="${ABLATION_TAG:-rh_ablation_$(date +%Y%m%d_%H%M%S)}"
ABLATION_RUN_ID="${ABLATION_RUN_ID:-${ABLATION_TAG}}"

NUM_TASKS="${NUM_TASKS:-36}"
SEED_START="${SEED_START:-0}"
NUM_SEEDS="${NUM_SEEDS:-5}"
MAX_STEPS="${MAX_STEPS:-5000}"
MAX_COMBOS="${MAX_COMBOS:-64}"

ABLATION_PLANNERS_CSV="${ABLATION_PLANNERS_CSV:-rh_cfpa2;physics_rh_cfpa2}"
ABLATION_ENV_CONFIGS_CSV="${ABLATION_ENV_CONFIGS_CSV:-configs/env_narrow_t_branches.yaml;configs/env_narrow_t_dense_branches.yaml;configs/env_narrow_t_asymmetric_branches.yaml;configs/env_narrow_t_loop_branches.yaml}"
SCORE_MODES_CSV="${SCORE_MODES_CSV:-hybrid;immediate_only}"
HORIZONS_CSV="${HORIZONS_CSV:-3;4;5}"
GAMMAS_CSV="${GAMMAS_CSV:-0.88;0.92}"
IMMEDIATE_WEIGHTS_CSV="${IMMEDIATE_WEIGHTS_CSV:-0.85;1.00;1.15}"
FUTURE_WEIGHTS_CSV="${FUTURE_WEIGHTS_CSV:-0.15;0.25;0.35}"
FRONTIER_CONSUMPTION_WEIGHTS_CSV="${FRONTIER_CONSUMPTION_WEIGHTS_CSV:-0.10;0.18;0.28}"
CONGESTION_SCALES_CSV="${CONGESTION_SCALES_CSV:-0.60;1.00}"
BRANCH_SCALES_CSV="${BRANCH_SCALES_CSV:-0.70;1.00}"
TOPK_LIMITS_CSV="${TOPK_LIMITS_CSV:-6}"

TASK_CORES="${TASK_CORES:-1}"
TASK_MEM="${TASK_MEM:-6G}"
TASK_TMPFS="${TASK_TMPFS:-20G}"
TASK_WALLTIME="${TASK_WALLTIME:-48:00:00}"

MERGE_CORES="${MERGE_CORES:-1}"
MERGE_MEM="${MERGE_MEM:-4G}"
MERGE_TMPFS="${MERGE_TMPFS:-8G}"
MERGE_WALLTIME="${MERGE_WALLTIME:-06:00:00}"

LOG_ROOT="${LOG_ROOT:-$HOME/physics_runs/ablation_logs/${ABLATION_TAG}}"
mkdir -p "${LOG_ROOT}"

TASK_SCRIPT="${REPO_DIR}/jobs/myriad_rh_ablation_array_task.sh"
MERGE_SCRIPT="${REPO_DIR}/jobs/myriad_merge_rh_ablation_results.sh"
chmod +x "${TASK_SCRIPT}" "${MERGE_SCRIPT}"

build_vlist() {
  local out=""
  local kv=""
  for kv in "$@"; do
    if [[ -z "${out}" ]]; then
      out="${kv}"
    else
      out="${out},${kv}"
    fi
  done
  printf "%s" "${out}"
}

VLIST="$(build_vlist \
  "REPO_DIR=${REPO_DIR}" \
  "CONDA_SH=${CONDA_SH}" \
  "CONDA_ENV=${CONDA_ENV}" \
  "OUTPUT_ROOT=${OUTPUT_ROOT}" \
  "PHYSICS_WEIGHT_FILE=${PHYSICS_WEIGHT_FILE}" \
  "ABLATION_RUN_ID=${ABLATION_RUN_ID}" \
  "NUM_TASKS=${NUM_TASKS}" \
  "SEED_START=${SEED_START}" \
  "NUM_SEEDS=${NUM_SEEDS}" \
  "MAX_STEPS=${MAX_STEPS}" \
  "MAX_COMBOS=${MAX_COMBOS}" \
  "ABLATION_PLANNERS_CSV=${ABLATION_PLANNERS_CSV}" \
  "ABLATION_ENV_CONFIGS_CSV=${ABLATION_ENV_CONFIGS_CSV}" \
  "SCORE_MODES_CSV=${SCORE_MODES_CSV}" \
  "HORIZONS_CSV=${HORIZONS_CSV}" \
  "GAMMAS_CSV=${GAMMAS_CSV}" \
  "IMMEDIATE_WEIGHTS_CSV=${IMMEDIATE_WEIGHTS_CSV}" \
  "FUTURE_WEIGHTS_CSV=${FUTURE_WEIGHTS_CSV}" \
  "FRONTIER_CONSUMPTION_WEIGHTS_CSV=${FRONTIER_CONSUMPTION_WEIGHTS_CSV}" \
  "CONGESTION_SCALES_CSV=${CONGESTION_SCALES_CSV}" \
  "BRANCH_SCALES_CSV=${BRANCH_SCALES_CSV}" \
  "TOPK_LIMITS_CSV=${TOPK_LIMITS_CSV}")"

ARRAY_RAW="$(qsub -terse \
  -N "pr_rhabt" \
  -cwd \
  -t "1-${NUM_TASKS}" \
  -pe smp "${TASK_CORES}" \
  -l "h_rt=${TASK_WALLTIME},mem=${TASK_MEM},tmpfs=${TASK_TMPFS}" \
  -o "${LOG_ROOT}" \
  -e "${LOG_ROOT}" \
  -v "${VLIST}" \
  "${TASK_SCRIPT}")"
ARRAY_ID="${ARRAY_RAW%%.*}"

MERGE_RAW="$(qsub -terse \
  -N "pr_rhabm" \
  -cwd \
  -hold_jid "${ARRAY_ID}" \
  -pe smp "${MERGE_CORES}" \
  -l "h_rt=${MERGE_WALLTIME},mem=${MERGE_MEM},tmpfs=${MERGE_TMPFS}" \
  -o "${LOG_ROOT}" \
  -e "${LOG_ROOT}" \
  -v "${VLIST}" \
  "${MERGE_SCRIPT}")"

cat <<EOF
array_job_id=${ARRAY_RAW}
merge_job_id=${MERGE_RAW}
log_root=${LOG_ROOT}
ablation_run_id=${ABLATION_RUN_ID}
monitor=qstat -u \$USER
ablation_outputs=${REPO_DIR}/${OUTPUT_ROOT}/benchmarks/${ABLATION_RUN_ID}
EOF
