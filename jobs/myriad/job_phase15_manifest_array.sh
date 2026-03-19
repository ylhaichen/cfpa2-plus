#!/bin/bash -l
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N phase15arr
#$ -l h_rt=12:00:00
#$ -l mem=4G
#$ -l tmpfs=8G
#$ -t 1-36
#$ -tc 36

set -euo pipefail

: "${REPO_DIR:?REPO_DIR is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${MANIFEST_PATH:?MANIFEST_PATH is required}"

BASE_CONFIG="${BASE_CONFIG:-configs/base.yaml}"
NUM_TASKS="${NUM_TASKS:-36}"
CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-}"
VENV_PATH="${VENV_PATH:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

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

TASK_INDEX=0
if [[ -n "${SGE_TASK_ID:-}" ]]; then
  TASK_INDEX=$((SGE_TASK_ID - 1))
fi

if (( NUM_TASKS <= 0 )); then
  echo "NUM_TASKS must be positive, got ${NUM_TASKS}" >&2
  exit 2
fi

ROW_COUNT="$(python - <<'PY' "${MANIFEST_PATH}"
import pandas as pd
import sys
print(len(pd.read_csv(sys.argv[1])))
PY
)"

ASSIGNED_COUNT="$(python - <<'PY' "${ROW_COUNT}" "${TASK_INDEX}" "${NUM_TASKS}"
import sys
row_count = int(sys.argv[1])
task_index = int(sys.argv[2])
num_tasks = int(sys.argv[3])
count = 0 if task_index >= row_count else ((row_count - 1 - task_index) // num_tasks + 1)
print(count)
PY
)"

echo "hostname=$(hostname)"
echo "date=$(date)"
echo "python=$(python --version 2>&1)"
echo "repo_dir=${REPO_DIR}"
echo "output_root=${OUTPUT_ROOT}"
echo "job_id=${JOB_ID:-na}"
echo "sge_task_id=${SGE_TASK_ID:-na}"
echo "task_index=${TASK_INDEX}"
echo "num_tasks=${NUM_TASKS}"
echo "row_count=${ROW_COUNT}"
echo "assigned_count=${ASSIGNED_COUNT}"
echo "manifest_path=${MANIFEST_PATH}"

if (( TASK_INDEX >= ROW_COUNT )); then
  echo "No manifest rows assigned to task_index=${TASK_INDEX}; exiting cleanly."
  exit 0
fi

RUN_ARGS=(
  python experiments/run_manifest_row.py
  --manifest "${MANIFEST_PATH}"
  --base-config "${BASE_CONFIG}"
  --output-root "${OUTPUT_ROOT}"
)

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  RUN_ARGS+=(--skip-existing)
fi

for ((ROW_INDEX=TASK_INDEX; ROW_INDEX<ROW_COUNT; ROW_INDEX+=NUM_TASKS)); do
  echo "=== phase15 shard task_index=${TASK_INDEX} row_index=${ROW_INDEX} ==="
  "${RUN_ARGS[@]}" --row-index "${ROW_INDEX}"
done
