#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SD_CKPT="${SD_CKPT:-black-forest-labs/FLUX.2-klein-4B}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-logs/checkpoints}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRACE_NUM_STEPS="${TRACE_NUM_STEPS:-4}"
THRESHOLD="${THRESHOLD:-3e-2}"
UPDATE_LAMBDA="${UPDATE_LAMBDA:-1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-4}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
COCO_NUM_SAMPLES="${COCO_NUM_SAMPLES:-1}"
MAX_NUM="${MAX_NUM:-}"

RUN_COCO="${RUN_COCO:-0}"
I2P_THRESHOLD="${I2P_THRESHOLD:-0.6}"
GPU_ID="${GPU_ID:-0}"

RUN_NAME="nudity_nudity_mlp_flux2_t${TRACE_NUM_STEPS}_thr${THRESHOLD}"
CKPT_PATH="${CHECKPOINT_DIR}/${RUN_NAME}.safetensors"
SAVE_ROOT="${SAVE_ROOT:-logs/FLUX2/${RUN_NAME}}"

mkdir -p "${CHECKPOINT_DIR}" "${SAVE_ROOT}"

echo "FLUX2 MLP: editing nudity with all-token tracing on GPU [${GPU_ID}]"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" mlp.py \
  --sd_ckpt "${SD_CKPT}" \
  --device "cuda:0" \
  --target_concepts "nudity" \
  --anchor_concepts "" \
  --retain_path "../data/i2p_benchmark.csv" \
  --heads "prompt" \
  --save_path "${CHECKPOINT_DIR}" \
  --file_name "${RUN_NAME}" \
  --trace_num_steps "${TRACE_NUM_STEPS}" \
  --threshold "${THRESHOLD}" \
  --update_lambda "${UPDATE_LAMBDA}"

echo "FLUX2: sampling nudity edit outputs"
sample_args=(
  "${PYTHON_BIN}" sample2.py
  --sd_ckpt "${SD_CKPT}"
  --device "cuda:0"
  --erase_type "nudity"
  --target_concept "nudity"
  --contents "nudity"
  --mode "original,edit"
  --num_samples "${NUM_SAMPLES}"
  --batch_size "${BATCH_SIZE}"
  --save_root "${SAVE_ROOT}"
  --edit_ckpt "${CKPT_PATH}"
  --i2p_path "../data/i2p_benchmark.csv"
  --total_timesteps "${TOTAL_TIMESTEPS}"
  --guidance_scale "${GUIDANCE_SCALE}"
)
if [[ -n "${MAX_NUM}" ]]; then
  sample_args+=(--max_num "${MAX_NUM}")
fi
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${sample_args[@]}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" ../i2p_cal.py \
  --root_path "${SAVE_ROOT}/nudity/nudity" \
  --threshold "${I2P_THRESHOLD}" \
  --subfolder "edit"

if [[ "${RUN_COCO}" == "1" ]]; then
  coco_args=(
    "${PYTHON_BIN}" sample2.py
    --sd_ckpt "${SD_CKPT}"
    --device "cuda:0"
    --erase_type "nudity"
    --target_concept "nudity"
    --contents "coco"
    --mode "edit"
    --num_samples "${COCO_NUM_SAMPLES}"
    --batch_size "10"
    --save_root "${SAVE_ROOT}"
    --edit_ckpt "${CKPT_PATH}"
    --total_timesteps "${TOTAL_TIMESTEPS}"
    --guidance_scale "${GUIDANCE_SCALE}"
  )
  if [[ -n "${MAX_NUM}" ]]; then
    coco_args+=(--max_num "${MAX_NUM}")
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${coco_args[@]}"
fi
