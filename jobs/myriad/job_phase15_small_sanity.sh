#!/bin/bash -l
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N phase15san
#$ -l h_rt=01:30:00
#$ -l mem=4G
#$ -l tmpfs=4G

set -euo pipefail

: "${REPO_DIR:?REPO_DIR is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

SANITY_TAG="${SANITY_TAG:-phase15_sanity_${JOB_ID:-local}}"
SANITY_ROWS="${SANITY_ROWS:-2}"
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${JOB_ID:-na}_sanity"
mkdir -p "${MPLCONFIGDIR}"

echo "hostname=$(hostname)"
echo "date=$(date)"
echo "python=$(python --version 2>&1)"
echo "repo_dir=${REPO_DIR}"
echo "output_root=${OUTPUT_ROOT}"

python experiments/generate_phase15_manifest.py \
  --output-root "${OUTPUT_ROOT}" \
  --profile small_sanity \
  --manifest-tag "${SANITY_TAG}"

MANIFEST_PATH="$(python -c "import pathlib; import sys; paths=sorted(pathlib.Path(sys.argv[1]).glob(f'phase15_small_sanity_{sys.argv[2]}.csv')); print(paths[-1] if paths else '')" "${OUTPUT_ROOT}/manifests" "${SANITY_TAG}")"
if [[ -z "${MANIFEST_PATH}" ]]; then
  echo "Failed to locate generated manifest for tag ${SANITY_TAG}" >&2
  exit 1
fi

for ((i=0; i<${SANITY_ROWS}; i++)); do
  python experiments/run_manifest_row.py \
    --manifest "${MANIFEST_PATH}" \
    --row-index "${i}" \
    --output-root "${OUTPUT_ROOT}" \
    --skip-existing
done

python experiments/summarize_phase15.py \
  --manifest "${MANIFEST_PATH}" \
  --output-root "${OUTPUT_ROOT}"

PHASE15_OUTPUT_SUBDIR="$(python -c "import pandas as pd,sys; print(pd.read_csv(sys.argv[1])['output_subdir'].iloc[0])" "${MANIFEST_PATH}")"
PER_RUN_CSV="${OUTPUT_ROOT}/benchmarks/${PHASE15_OUTPUT_SUBDIR}/phase15_summary/results_csv/phase15_per_run_results.csv"

python experiments/plot_metrics.py \
  --input "${PER_RUN_CSV}" \
  --group-by map_family normalization_mode run_group

echo "manifest_path=${MANIFEST_PATH}"
echo "per_run_csv=${PER_RUN_CSV}"
