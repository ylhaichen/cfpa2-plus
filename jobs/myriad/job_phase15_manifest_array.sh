#!/bin/bash -l
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N phase15arr
#$ -l h_rt=12:00:00
#$ -l mem=4G
#$ -l tmpfs=8G
#$ -t 1-1
#$ -o outputs/myriad_logs/
#$ -e outputs/myriad_logs/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
BASE_CONFIG="${BASE_CONFIG:-configs/base.yaml}"
MANIFEST_PATH="${MANIFEST_PATH:?MANIFEST_PATH is required, e.g. outputs/manifests/phase15_full_xxx.csv}"

cd "${REPO_DIR}"

mkdir -p "${OUTPUT_ROOT}/myriad_logs"

# Option A: module-based Python on Myriad.
if [[ "${PHASE15_USE_MODULE_PYTHON:-1}" == "1" ]]; then
  module purge
  module load python/3.11.3
fi

# Option B: user-managed conda or venv.
# TODO: set PHASE15_USE_MODULE_PYTHON=0 and PHASE15_USE_CUSTOM_PYTHON=1 if you prefer this path.
if [[ "${PHASE15_USE_CUSTOM_PYTHON:-0}" == "1" ]]; then
  # TODO: edit the next lines for your own environment.
  source "${PHASE15_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  conda activate "${PHASE15_CONDA_ENV:-cfpa2rh}"
  # Alternative:
  # source "${PHASE15_VENV_PATH:-$HOME/venvs/cfpa2rh}/bin/activate"
fi

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
