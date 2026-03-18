#!/bin/bash -l
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N phase15sum
#$ -l h_rt=02:00:00
#$ -l mem=4G
#$ -l tmpfs=4G
#$ -o outputs/myriad_logs/
#$ -e outputs/myriad_logs/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
MANIFEST_PATH="${MANIFEST_PATH:?MANIFEST_PATH is required}"

cd "${REPO_DIR}"

mkdir -p "${OUTPUT_ROOT}/myriad_logs"

# Option A: module-based Python on Myriad.
if [[ "${PHASE15_USE_MODULE_PYTHON:-1}" == "1" ]]; then
  module purge
  module load python/3.11.3
fi

# Option B: user-managed conda or venv.
# TODO: edit these lines if you use conda or a virtualenv instead.
if [[ "${PHASE15_USE_CUSTOM_PYTHON:-0}" == "1" ]]; then
  source "${PHASE15_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  conda activate "${PHASE15_CONDA_ENV:-cfpa2rh}"
fi

export PYTHONPATH="${REPO_DIR}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${JOB_ID:-na}_summary"
mkdir -p "${MPLCONFIGDIR}"

echo "hostname=$(hostname)"
echo "date=$(date)"
echo "python=$(python --version 2>&1)"
echo "manifest_path=${MANIFEST_PATH}"

python experiments/summarize_phase15.py \
  --manifest "${MANIFEST_PATH}" \
  --output-root "${OUTPUT_ROOT}"

PHASE15_OUTPUT_SUBDIR="$(python -c "import pandas as pd,sys; print(pd.read_csv(sys.argv[1])['output_subdir'].iloc[0])" "${MANIFEST_PATH}")"
PER_RUN_CSV="${OUTPUT_ROOT}/benchmarks/${PHASE15_OUTPUT_SUBDIR}/phase15_summary/results_csv/phase15_per_run_results.csv"

python experiments/plot_metrics.py \
  --input "${PER_RUN_CSV}" \
  --group-by map_family normalization_mode run_group

echo "phase15_output_subdir=${PHASE15_OUTPUT_SUBDIR}"
echo "per_run_csv=${PER_RUN_CSV}"
