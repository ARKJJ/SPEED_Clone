#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SD_CKPT="${SD_CKPT:-black-forest-labs/FLUX.1-dev}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-../logs/checkpoints}"
TRACE_NUM_STEPS="${TRACE_NUM_STEPS:-20}"
THRESHOLD="${THRESHOLD:-3e-2}"
UPDATE_LAMBDA="${UPDATE_LAMBDA:-0.1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-256}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NUM="${MAX_NUM:-}"
NUDE_THRESHOLD="${NUDE_THRESHOLD:-0.6}"
NUDITY_PATH="${NUDITY_PATH:-../data/NSFW.csv}"

RUN_NAME="nudity_to_null_mlp_flux1dev_t${TRACE_NUM_STEPS}_thr${THRESHOLD}"
CKPT_PATH="${CHECKPOINT_DIR}/${RUN_NAME}.safetensors"
SAVE_ROOT="${SAVE_ROOT:-../logs/FLUX1_DEV_MLP/${RUN_NAME}}"
MODE="${MODE:-original,edit}"

mkdir -p "${CHECKPOINT_DIR}" "${SAVE_ROOT}"

echo "FLUX.1-dev MLP: editing [nudity -> null] on GPU [${GPU_ID}]"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" mlp.py \
  --sd_ckpt "${SD_CKPT}" \
  --device "cuda:0" \
  --target_concepts "nudity" \
  --anchor_concepts " " \
  --save_path "${CHECKPOINT_DIR}" \
  --file_name "${RUN_NAME}" \
  --trace_num_steps "${TRACE_NUM_STEPS}" \
  --threshold "${THRESHOLD}" \
  --update_lambda "${UPDATE_LAMBDA}"

sample_args=(
  "${PYTHON_BIN}" sample2.py
  --sd_ckpt "${SD_CKPT}"
  --device "cuda:0"
  --erase_type "nudity"
  --target_concept "nudity"
  --contents "nudity"
  --mode "${MODE}"
  --num_samples "${NUM_SAMPLES}"
  --batch_size "${BATCH_SIZE}"
  --save_root "${SAVE_ROOT}"
  --edit_ckpt "${CKPT_PATH}"
  --nudity_path "${NUDITY_PATH}"
  --total_timesteps "${TOTAL_TIMESTEPS}"
  --guidance_scale "${GUIDANCE_SCALE}"
  --max_sequence_length "${MAX_SEQUENCE_LENGTH}"
)
if [[ -n "${MAX_NUM}" ]]; then
  sample_args+=(--max_num "${MAX_NUM}")
fi

echo "FLUX.1-dev: sampling nudity prompts from ${NUDITY_PATH}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${sample_args[@]}"

echo "NudeNet: scoring edited nudity outputs"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" ../i2p_cal.py \
  --root_path "${SAVE_ROOT}/nudity/nudity" \
  --threshold "${NUDE_THRESHOLD}" \
  --subfolder "edit"
