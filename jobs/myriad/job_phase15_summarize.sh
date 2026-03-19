#!/bin/bash -l
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N phase15sum
#$ -l h_rt=02:00:00
#$ -l mem=4G
#$ -l tmpfs=4G

set -euo pipefail

: "${REPO_DIR:?REPO_DIR is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${MANIFEST_PATH:?MANIFEST_PATH is required}"

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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${JOB_ID:-na}_summary"
mkdir -p "${MPLCONFIGDIR}"

echo "hostname=$(hostname)"
echo "date=$(date)"
echo "python=$(python --version 2>&1)"
echo "repo_dir=${REPO_DIR}"
echo "output_root=${OUTPUT_ROOT}"
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
