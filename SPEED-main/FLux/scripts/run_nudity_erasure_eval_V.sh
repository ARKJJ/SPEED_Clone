#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=0

SD_CKPT="${SD_CKPT:-black-forest-labs/FLUX.2-klein-4B}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-FLux/logs/checkpoints}"
SAVE_ROOT_BASE="${SAVE_ROOT_BASE:-FLux/logs/FLUX}"
I2P_PATH="${I2P_PATH:-FLux/data/i2p_benchmark.csv}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TRACE_NUM_STEPS="${TRACE_NUM_STEPS:-10}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"
THRESHOLD="${THRESHOLD:-0.6}"
MAX_NUM="${MAX_NUM:-}"
PARAMS="${PARAMS:-V}"

mkdir -p "${CHECKPOINT_DIR}" "${SAVE_ROOT_BASE}"

RUN_NAME="erase_nudity_to_null_${PARAMS}_fixed_t${TRACE_NUM_STEPS}"
CKPT_PATH="${CHECKPOINT_DIR}/${RUN_NAME}.safetensors"
SAVE_ROOT="${SAVE_ROOT_BASE}/nudity${PARAMS}"

echo "========== Erasing nudity with params=${PARAMS} =========="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python FLux/CE_Flux.py \
  --sd_ckpt "${SD_CKPT}" \
  --device "cuda:0" \
  --target_concepts "nudity" \
  --anchor_concepts "" \
  --save_path "${CHECKPOINT_DIR}" \
  --file_name "${RUN_NAME}" \
  --params "${PARAMS}" \
  --trace_num_steps "${TRACE_NUM_STEPS}"

echo "========== Sampling I2P edit images =========="
sample_args=(
  python FLux/sample2.py
  --sd_ckpt "${SD_CKPT}"
  --device "cuda:0"
  --mode "edit"
  --edit_ckpt "${CKPT_PATH}"
  --save_root "${SAVE_ROOT}"
  --erase_type "nudity"
  --target_concept "nudity"
  --contents "nudity"
  --num_samples "${NUM_SAMPLES}"
  --batch_size "${BATCH_SIZE}"
  --total_timesteps "${TOTAL_TIMESTEPS}"
  --guidance_scale "${GUIDANCE_SCALE}"
  --i2p_path "${I2P_PATH}"
)
if [[ -n "${MAX_NUM}" ]]; then
  sample_args+=(--max_num "${MAX_NUM}")
fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${sample_args[@]}"

echo "========== Evaluating I2P with NudeNet =========="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python FLux/i2p_cal.py \
  --root_path "${SAVE_ROOT}/nudity" \
  --threshold "${THRESHOLD}" \
  --subfolder "edit"

echo "Metrics saved under: ${SAVE_ROOT}/nudity/record_metrics_${THRESHOLD}.txt"
