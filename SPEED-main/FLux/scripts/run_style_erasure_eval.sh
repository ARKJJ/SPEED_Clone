#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

SD_CKPT="${SD_CKPT:-black-forest-labs/FLUX.1-dev}"
TARGET_CONCEPT="${TARGET_CONCEPT:-Van Gogh, Picasso, Monet}"
ANCHOR_CONCEPTS="${ANCHOR_CONCEPTS:-art}"
RETAIN_PATH="${RETAIN_PATH:-FLux/data/style_100.csv}"
HEADS="${HEADS:-concept}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-FLux/logs/checkpoints}"
SAVE_ROOT_BASE="${SAVE_ROOT_BASE:-FLux/logs/FLUX}"
ERASE_TYPE="${ERASE_TYPE:-style}"
CONTENTS="${CONTENTS:-Van Gogh, Picasso, Monet, Paul Gauguin, Caravaggio}"
NUM_SAMPLES="${NUM_SAMPLES:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TRACE_NUM_STEPS="${TRACE_NUM_STEPS:-10}"
PARAMS="${PARAMS:-KV}"
R_START="${R_START:-3}"
R_END="${R_END:-7}"

mkdir -p "${CHECKPOINT_DIR}" "${SAVE_ROOT_BASE}"

TARGET_DIR_NAME="${TARGET_CONCEPT//, /_}"
TARGET_SLUG="${TARGET_CONCEPT,,}"
TARGET_SLUG="${TARGET_SLUG//, /_}"
TARGET_SLUG="${TARGET_SLUG// /_}"
TARGET_SLUG="${TARGET_SLUG//,/_}"
ANCHOR_SLUG="${ANCHOR_CONCEPTS,,}"
ANCHOR_SLUG="${ANCHOR_SLUG//, /_}"
ANCHOR_SLUG="${ANCHOR_SLUG// /_}"
ANCHOR_SLUG="${ANCHOR_SLUG//,/_}"

for r in $(seq "${R_START}" "${R_END}"); do
  RUN_NAME="erase_${TARGET_SLUG}_to_${ANCHOR_SLUG}_${PARAMS}_r${r}_t${TRACE_NUM_STEPS}"
  CKPT_PATH="${CHECKPOINT_DIR}/${RUN_NAME}.safetensors"
  SAVE_ROOT="${SAVE_ROOT_BASE}/style${PARAMS}${r}"

  echo "========== [r=${r}] Erasing ${TARGET_CONCEPT} style with params=${PARAMS} =========="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python FLux/CE_Flux.py \
    --sd_ckpt "${SD_CKPT}" \
    --device "cuda:0" \
    --target_concepts "${TARGET_CONCEPT}" \
    --anchor_concepts "${ANCHOR_CONCEPTS}" \
    --retain_path "${RETAIN_PATH}" \
    --heads "${HEADS}" \
    --save_path "${CHECKPOINT_DIR}" \
    --file_name "${RUN_NAME}" \
    --params "${PARAMS}" \
    --trace_num_steps "${TRACE_NUM_STEPS}" \
    --residual_scale "${r}"

  echo "========== [r=${r}] Sampling original/edit images =========="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python FLux/sample.py \
    --sd_ckpt "${SD_CKPT}" \
    --device "cuda:0" \
    --mode "original,edit" \
    --edit_ckpt "${CKPT_PATH}" \
    --save_root "${SAVE_ROOT}" \
    --erase_type "${ERASE_TYPE}" \
    --target_concept "${TARGET_CONCEPT}" \
    --contents "${CONTENTS}" \
    --num_samples "${NUM_SAMPLES}" \
    --batch_size "${BATCH_SIZE}"

  echo "========== [r=${r}] Calculating CS/FID =========="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python FLux/score_cal.py \
    --contents "${CONTENTS}" \
    --root_path "${SAVE_ROOT}/${TARGET_DIR_NAME}" \
    --sub_root "edit" \
    --pretrained_path "${SAVE_ROOT}/${TARGET_DIR_NAME}"

  echo "Metrics saved under: ${SAVE_ROOT}/${TARGET_DIR_NAME}/record_metrics.txt"
done
