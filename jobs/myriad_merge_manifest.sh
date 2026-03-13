#!/bin/bash -l
set -euo pipefail

: "${REPO_DIR:?REPO_DIR is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"

VENV_PATH="${VENV_PATH:-}"

if [[ -n "${VENV_PATH}" && -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}"

mkdir -p "${RUN_ROOT}/dataset"

echo "[merge] host=$(hostname)"
echo "[merge] input=${RUN_ROOT}/dataset/task*/manifest.jsonl"
echo "[merge] output=${RUN_ROOT}/dataset/manifest_merged.jsonl"

python training/merge_dataset_manifests.py \
  --inputs "${RUN_ROOT}/dataset/task*/manifest.jsonl" \
  --output "${RUN_ROOT}/dataset/manifest_merged.jsonl"
