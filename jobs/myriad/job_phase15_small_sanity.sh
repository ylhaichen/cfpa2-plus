#!/bin/bash -l
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N phase15san
#$ -l h_rt=01:30:00
#$ -l mem=4G
#$ -l tmpfs=4G
#$ -o outputs/myriad_logs/
#$ -e outputs/myriad_logs/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
SANITY_TAG="${SANITY_TAG:-phase15_sanity_${JOB_ID:-local}}"
SANITY_ROWS="${SANITY_ROWS:-2}"

cd "${REPO_DIR}"

mkdir -p "${OUTPUT_ROOT}/myriad_logs"

# Option A: module-based Python on Myriad.
if [[ "${PHASE15_USE_MODULE_PYTHON:-1}" == "1" ]]; then
  module purge
  module load python/3.11.3
fi

# Option B: user-managed conda or venv.
# TODO: edit these lines if you prefer conda or a virtualenv.
if [[ "${PHASE15_USE_CUSTOM_PYTHON:-0}" == "1" ]]; then
  source "${PHASE15_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  conda activate "${PHASE15_CONDA_ENV:-cfpa2rh}"
fi

export PYTHONPATH="${REPO_DIR}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${JOB_ID:-na}_sanity"
mkdir -p "${MPLCONFIGDIR}"

echo "hostname=$(hostname)"
echo "date=$(date)"
echo "python=$(python --version 2>&1)"

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
