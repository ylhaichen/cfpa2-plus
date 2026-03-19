#!/bin/bash -l
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N phase15gen
#$ -l h_rt=01:00:00
#$ -l mem=2G
#$ -l tmpfs=2G

set -euo pipefail

: "${REPO_DIR:?REPO_DIR is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

MANIFEST_PROFILE="${MANIFEST_PROFILE:-full}"
MANIFEST_TAG="${MANIFEST_TAG:-phase15_$(date +%Y%m%d_%H%M%S)}"
PLANNER_CHOICE="${PLANNER_CHOICE:-cfpa2_plus_phase1_calib_base}"
PLANNER_CONFIG="${PLANNER_CONFIG:-configs/planner_cfpa2_plus_phase1_calib_base.yaml}"
MAX_STEPS="${MAX_STEPS:-80}"
MANIFEST_PATH="${MANIFEST_PATH:-}"
CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-}"
VENV_PATH="${VENV_PATH:-}"

if [[ -n "${CONDA_SH}" && -n "${CONDA_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
elif [[ -n "${VENV_PATH}" && -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_ROOT}/myriad_logs"

export PYTHONPATH="${REPO_DIR}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${JOB_ID:-na}_manifest"
mkdir -p "${MPLCONFIGDIR}"

echo "hostname=$(hostname)"
echo "date=$(date)"
echo "python=$(python --version 2>&1)"
echo "repo_dir=${REPO_DIR}"
echo "output_root=${OUTPUT_ROOT}"
echo "manifest_profile=${MANIFEST_PROFILE}"
echo "manifest_tag=${MANIFEST_TAG}"

CMD=(
  python experiments/generate_phase15_manifest.py
  --output-root "${OUTPUT_ROOT}"
  --profile "${MANIFEST_PROFILE}"
  --manifest-tag "${MANIFEST_TAG}"
  --planner-choice "${PLANNER_CHOICE}"
  --planner-config "${PLANNER_CONFIG}"
  --max-steps "${MAX_STEPS}"
)

if [[ -n "${MANIFEST_PATH}" ]]; then
  CMD+=(--manifest-path "${MANIFEST_PATH}")
fi

"${CMD[@]}"
