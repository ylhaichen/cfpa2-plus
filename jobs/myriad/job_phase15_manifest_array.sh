#!/bin/bash -l
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N phase15arr
#$ -l h_rt=12:00:00
#$ -l mem=4G
#$ -l tmpfs=8G
#$ -t 1-1

set -euo pipefail

: "${REPO_DIR:?REPO_DIR is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${MANIFEST_PATH:?MANIFEST_PATH is required}"

BASE_CONFIG="${BASE_CONFIG:-configs/base.yaml}"
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${JOB_ID:-na}_${SGE_TASK_ID:-0}"
mkdir -p "${MPLCONFIGDIR}"

ROW_INDEX=0
if [[ -n "${SGE_TASK_ID:-}" ]]; then
  ROW_INDEX=$((SGE_TASK_ID - 1))
fi

echo "hostname=$(hostname)"
echo "date=$(date)"
echo "python=$(python --version 2>&1)"
echo "repo_dir=${REPO_DIR}"
echo "output_root=${OUTPUT_ROOT}"
echo "job_id=${JOB_ID:-na}"
echo "sge_task_id=${SGE_TASK_ID:-na}"
echo "row_index=${ROW_INDEX}"
echo "manifest_path=${MANIFEST_PATH}"

python experiments/run_manifest_row.py \
  --manifest "${MANIFEST_PATH}" \
  --row-index "${ROW_INDEX}" \
  --base-config "${BASE_CONFIG}" \
  --output-root "${OUTPUT_ROOT}" \
  --skip-existing
