#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

SD_CKPT="${SD_CKPT:-black-forest-labs/FLUX.1-dev}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-FLux/logs/checkpoints}"
SAVE_ROOT_BASE="${SAVE_ROOT_BASE:-FLux/logs/FLUX}"
PARAMS="${PARAMS:-V}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-4}"
TRACE_NUM_STEPS="${TRACE_NUM_STEPS:-20}"
THRESHOLD="${THRESHOLD:-1e-4}"
UPDATE_LAMBDA="${UPDATE_LAMBDA:-1e-2}"
CONTENTS="${CONTENTS:-erase, retain}"
BATCH_SIZE="${BATCH_SIZE:-10}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"
MAX_NUM="${MAX_NUM:-5}"
GCD_SCRIPT="${GCD_SCRIPT:-}"

GPU_IDX=("0" "1")
ERASE_TYPES=("10_celebrity" "50_celebrity" "100_celebrity")

mkdir -p "${CHECKPOINT_DIR}" "${SAVE_ROOT_BASE}"

read_targets_from_csv() {
  local dataset_path="$1"
  python - "${dataset_path}" <<'PY'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
if "type" in df.columns:
    df = df[df["type"].astype(str).str.strip() == "erase"]
concepts = list(dict.fromkeys(df["concept"].dropna().astype(str)))
print(", ".join(concepts))
PY
}

run_task() {
  local erase_type="$1"
  local gpu_id="$2"
  local dataset_path="FLux/data/${erase_type}.csv"
  local target_concepts
  target_concepts="$(read_targets_from_csv "${dataset_path}")"

  local run_name="erase_${erase_type}_to_person_${PARAMS}_r${RESIDUAL_SCALE}_t${TRACE_NUM_STEPS}"
  local ckpt_path="${CHECKPOINT_DIR}/${run_name}.safetensors"
  local save_root="${SAVE_ROOT_BASE}/${erase_type}_${PARAMS}_r${RESIDUAL_SCALE}_t${TRACE_NUM_STEPS}"
  local target_root="${save_root}/${erase_type}"

  echo "========== [${erase_type}] Editing on GPU ${gpu_id} =========="
  CUDA_VISIBLE_DEVICES="${gpu_id}" python FLux/CE_Flux.py \
    --sd_ckpt "${SD_CKPT}" \
    --device "cuda:0" \
    --target_concepts "${target_concepts}" \
    --anchor_concepts "person" \
    --retain_path "${dataset_path}" \
    --heads "concept" \
    --save_path "${CHECKPOINT_DIR}" \
    --file_name "${run_name}" \
    --params "${PARAMS}" \
    --trace_num_steps "${TRACE_NUM_STEPS}" \
    --residual_scale "${RESIDUAL_SCALE}" \
    --threshold "${THRESHOLD}" \
    --update_lambda "${UPDATE_LAMBDA}"

  echo "========== [${erase_type}] Sampling original/edit/combine on GPU ${gpu_id} =========="
  local sample_args=(
    python FLux/sample2.py
    --sd_ckpt "${SD_CKPT}"
    --device "cuda:0"
    --mode "original,edit"
    --edit_ckpt "${ckpt_path}"
    --save_root "${save_root}"
    --erase_type "${erase_type}"
    --target_concept "${erase_type}"
    --contents "${CONTENTS}"
    --dataset_path "${dataset_path}"
    --batch_size "${BATCH_SIZE}"
    --total_timesteps "${TOTAL_TIMESTEPS}"
    --guidance_scale "${GUIDANCE_SCALE}"
  )
  if [[ -n "${MAX_NUM}" ]]; then
    sample_args+=(--max_num "${MAX_NUM}")
  fi
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${sample_args[@]}"

  echo "========== [${erase_type}] SPEED-style GCD paths =========="
  local erase_edit_dir="${target_root}/erase/edit"
  local retain_edit_dir="${target_root}/retain/edit"
  echo "Erase edit images:  ${erase_edit_dir}"
  echo "Retain edit images: ${retain_edit_dir}"
  echo "Combine images:"
  echo "  ${target_root}/erase/combine"
  echo "  ${target_root}/retain/combine"

  if [[ -n "${GCD_SCRIPT}" ]]; then
    mkdir -p "${target_root}/metrics"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python "${GCD_SCRIPT}" \
      --image_folder "${erase_edit_dir}" \
      > "${target_root}/metrics/gcd_erase.txt"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python "${GCD_SCRIPT}" \
      --image_folder "${retain_edit_dir}" \
      > "${target_root}/metrics/gcd_retain.txt"
    echo "GCD metrics saved under: ${target_root}/metrics"
  fi
}

gpu_idx=0
for erase_type in "${ERASE_TYPES[@]}"; do
  run_task "${erase_type}" "${GPU_IDX[$gpu_idx]}" &
  gpu_idx=$((gpu_idx + 1))

  if (( gpu_idx >= ${#GPU_IDX[@]} )); then
    wait
    gpu_idx=0
  fi
done

wait

if [[ -z "${GCD_SCRIPT}" ]]; then
  echo "GCD_SCRIPT is not set. Run GCD manually on the printed erase/edit and retain/edit folders."
fi
