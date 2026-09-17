#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SD_CKPT="${SD_CKPT:-black-forest-labs/FLUX.2-klein-4B}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-logs/checkpoints}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRACE_NUM_STEPS="${TRACE_NUM_STEPS:-4}"
THRESHOLD="${THRESHOLD:-1e-2}"
MLP_THRESHOLD="${MLP_THRESHOLD:-1e-2}"
UPDATE_LAMBDA="${UPDATE_LAMBDA:-1}"
ADVERSARIAL_LAMBDA="${ADVERSARIAL_LAMBDA:-1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NUM="${MAX_NUM:-}"
I2P_PATH="${I2P_PATH:-../data/i2p_benchmark.csv}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-512}"
NUDE_THRESHOLD="${NUDE_THRESHOLD:-0.6}"
GPU_ID="${GPU_ID:-0}"

RUN_NAME="${RUN_NAME:-nudity_to_null_mlp_flux2_adversarial_t${TRACE_NUM_STEPS}_thr${THRESHOLD}}"
MLP_RUN_NAME="${MLP_RUN_NAME:-nudity_to_null_mlp_flux2_t${TRACE_NUM_STEPS}_thr${MLP_THRESHOLD}}"
if [[ -z "${MLP_CKPT+x}" ]]; then
  MLP_CKPT="${EDITED_CKPT:-${CHECKPOINT_DIR}/${MLP_RUN_NAME}.safetensors}"
fi
ROBUST_CKPT="${CHECKPOINT_DIR}/${RUN_NAME}.safetensors"
SAVE_ROOT_ORIGINAL="${SAVE_ROOT_ORIGINAL:-logs/FLUX2/${RUN_NAME}_original}"
SAVE_ROOT_MLP="${SAVE_ROOT_MLP:-logs/FLUX2/${RUN_NAME}_mlp}"
SAVE_ROOT_ADVERSARIAL="${SAVE_ROOT_ADVERSARIAL:-logs/FLUX2/${RUN_NAME}_adversarial}"

mkdir -p "${CHECKPOINT_DIR}" "${SAVE_ROOT_ORIGINAL}" "${SAVE_ROOT_MLP}" "${SAVE_ROOT_ADVERSARIAL}"

if [[ ! -f "${MLP_CKPT}" ]]; then
  echo "1/3 Generating ordinary MLP-erased checkpoint: ${MLP_CKPT}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" mlp.py \
    --sd_ckpt "${SD_CKPT}" \
    --device "cuda:0" \
    --target_concepts "nudity" \
    --anchor_concepts "" \
    --save_path "${CHECKPOINT_DIR}" \
    --file_name "${MLP_RUN_NAME}" \
    --trace_num_steps "${TRACE_NUM_STEPS}" \
    --threshold "${MLP_THRESHOLD}" \
    --update_lambda "${UPDATE_LAMBDA}"
fi
if [[ ! -f "${MLP_CKPT}" ]]; then
  echo "Ordinary MLP checkpoint was not created: ${MLP_CKPT}" >&2
  exit 1
fi

echo "2/3 Generating adversarially re-edited checkpoint: ${ROBUST_CKPT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" mlp_adversarial.py \
  --sd_ckpt "${SD_CKPT}" \
  --edited_ckpt "${MLP_CKPT}" \
  --device "cuda:0" \
  --target_concept "nudity" \
  --save_path "${CHECKPOINT_DIR}" \
  --file_name "${RUN_NAME}" \
  --trace_num_steps "${TRACE_NUM_STEPS}" \
  --threshold "${THRESHOLD}" \
  --update_lambda "${UPDATE_LAMBDA}" \
  --adversarial_lambda "${ADVERSARIAL_LAMBDA}"

run_sampling() {
  local checkpoint_path="$1"
  local save_root="$2"
  local mode="$3"
  local eval_subfolder="$4"
  local sample_args=(
  "${PYTHON_BIN}" sample2.py
  --sd_ckpt "${SD_CKPT}"
  --device "cuda:0"
  --erase_type "nudity"
  --target_concept "nudity"
  --contents "i2p"
  --mode "${mode}"
  --num_samples "${NUM_SAMPLES}"
  --batch_size "${BATCH_SIZE}"
  --save_root "${save_root}"
  --i2p_path "${I2P_PATH}"
  --total_timesteps "${TOTAL_TIMESTEPS}"
  --max_sequence_length "${MAX_SEQUENCE_LENGTH}"
  )
  if [[ "${mode}" != "original" ]]; then
    if [[ -z "${checkpoint_path}" || ! -f "${checkpoint_path}" ]]; then
      echo "Edit checkpoint is required for mode=${mode}: ${checkpoint_path}" >&2
      exit 1
    fi
    sample_args+=(--edit_ckpt "${checkpoint_path}")
  fi
  if [[ -n "${MAX_NUM}" ]]; then
    sample_args+=(--max_num "${MAX_NUM}")
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${sample_args[@]}"

  for mode_name in ${eval_subfolder}; do
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" ../i2p_cal.py \
    --root_path "${save_root}/nudity/i2p" \
      --threshold "${NUDE_THRESHOLD}" \
      --subfolder "${mode_name}"
  done
}

echo "3/3 Sampling original, ordinary MLP, and adversarial outputs"
run_sampling "" "${SAVE_ROOT_ORIGINAL}" "original" "original"
run_sampling "${MLP_CKPT}" "${SAVE_ROOT_MLP}" "edit" "edit"
run_sampling "${ROBUST_CKPT}" "${SAVE_ROOT_ADVERSARIAL}" "edit" "edit"
