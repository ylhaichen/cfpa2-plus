#!/bin/bash -l
set -euo pipefail

: "${REPO_DIR:?REPO_DIR is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"

VENV_PATH="${VENV_PATH:-}"
TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8192}"
TRAIN_LR="${TRAIN_LR:-1e-3}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-1e-5}"
TRAIN_HIDDEN_DIMS="${TRAIN_HIDDEN_DIMS:-256,256}"
TRAIN_VAL_RATIO="${TRAIN_VAL_RATIO:-0.1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
TRAIN_MAX_SHARDS="${TRAIN_MAX_SHARDS:-}"

if [[ -n "${VENV_PATH}" && -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${JOB_ID:-na}"
mkdir -p "${MPLCONFIGDIR}"

mkdir -p "${RUN_ROOT}/models"

MANIFEST_PATH="${RUN_ROOT}/dataset/manifest_merged.jsonl"
CKPT_PATH="${RUN_ROOT}/models/physics_residual_mlp.pt"

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "manifest not found: ${MANIFEST_PATH}" >&2
  exit 2
fi

echo "[train] host=$(hostname) device=${TRAIN_DEVICE}"
echo "[train] manifest=${MANIFEST_PATH}"
echo "[train] checkpoint=${CKPT_PATH}"

if [[ -n "${TRAIN_MAX_SHARDS}" ]]; then
  EXTRA_MAX_SHARDS=(--max-shards "${TRAIN_MAX_SHARDS}")
else
  EXTRA_MAX_SHARDS=()
fi

python training/train_physics_residual_torch.py \
  --manifest "${MANIFEST_PATH}" \
  --output "${CKPT_PATH}" \
  --epochs "${TRAIN_EPOCHS}" \
  --batch-size "${TRAIN_BATCH_SIZE}" \
  --lr "${TRAIN_LR}" \
  --weight-decay "${TRAIN_WEIGHT_DECAY}" \
  --hidden-dims "${TRAIN_HIDDEN_DIMS}" \
  --val-ratio "${TRAIN_VAL_RATIO}" \
  --seed "${TRAIN_SEED}" \
  --device "${TRAIN_DEVICE}" \
  "${EXTRA_MAX_SHARDS[@]}"

python training/evaluate_physics_residual_torch.py \
  --manifest "${MANIFEST_PATH}" \
  --checkpoint "${CKPT_PATH}" \
  --device "${TRAIN_DEVICE}" \
  --output "${RUN_ROOT}/models/physics_residual_mlp.eval.json" \
  "${EXTRA_MAX_SHARDS[@]}"

echo "[train] done"
