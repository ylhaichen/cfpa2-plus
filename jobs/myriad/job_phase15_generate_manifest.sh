#!/bin/bash -l
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N phase15gen
#$ -l h_rt=01:00:00
#$ -l mem=2G
#$ -l tmpfs=2G
#$ -o outputs/myriad_logs/
#$ -e outputs/myriad_logs/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
MANIFEST_PROFILE="${MANIFEST_PROFILE:-full}"
MANIFEST_TAG="${MANIFEST_TAG:-phase15_$(date +%Y%m%d_%H%M%S)}"
PLANNER_CHOICE="${PLANNER_CHOICE:-cfpa2_plus_phase1_calib_base}"
PLANNER_CONFIG="${PLANNER_CONFIG:-configs/planner_cfpa2_plus_phase1_calib_base.yaml}"
MAX_STEPS="${MAX_STEPS:-80}"

cd "${REPO_DIR}"

mkdir -p "${OUTPUT_ROOT}/myriad_logs"

# Option A: module-based Python on Myriad.
if [[ "${PHASE15_USE_MODULE_PYTHON:-1}" == "1" ]]; then
  module purge
  module load python/3.11.3
fi

# Option B: user-managed conda or venv.
# TODO: edit these lines if you do not want to use the module path above.
if [[ "${PHASE15_USE_CUSTOM_PYTHON:-0}" == "1" ]]; then
  source "${PHASE15_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  conda activate "${PHASE15_CONDA_ENV:-cfpa2rh}"
fi

export PYTHONPATH="${REPO_DIR}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${JOB_ID:-na}_manifest"
mkdir -p "${MPLCONFIGDIR}"

echo "hostname=$(hostname)"
echo "date=$(date)"
echo "python=$(python --version 2>&1)"
echo "manifest_profile=${MANIFEST_PROFILE}"
echo "manifest_tag=${MANIFEST_TAG}"

python experiments/generate_phase15_manifest.py \
  --output-root "${OUTPUT_ROOT}" \
  --profile "${MANIFEST_PROFILE}" \
  --manifest-tag "${MANIFEST_TAG}" \
  --planner-choice "${PLANNER_CHOICE}" \
  --planner-config "${PLANNER_CONFIG}" \
  --max-steps "${MAX_STEPS}"
