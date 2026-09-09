#!/usr/bin/env bash
set -euo pipefail

declare -A targets_map
declare -A anchors_map
declare -A contents_map

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SD_CKPT="${SD_CKPT:-black-forest-labs/FLUX.1-dev}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-logs/checkpoints}"
SAVE_ROOT_BASE="${SAVE_ROOT_BASE:-logs/FLUX1_DEV_ATTN}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRACE_NUM_STEPS="${TRACE_NUM_STEPS:-20}"
THRESHOLD="${THRESHOLD:-3e-2}"
UPDATE_LAMBDA="${UPDATE_LAMBDA:-0.1}"
PARAMS="${PARAMS:-QKV}"
MODE="${MODE:-original,edit}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
COCO_NUM_SAMPLES="${COCO_NUM_SAMPLES:-1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"
MAX_NUM="${MAX_NUM:-}"
RUN_SCORE="${RUN_SCORE:-1}"
SCORE_ONLY="${SCORE_ONLY:-0}"
IFS=',' read -ra GPU_IDX <<< "${GPU_IDS:-0,1,2,3}"

erase_types=("instance")
targets_map["instance"]="Snoopy;Snoopy, Mickey;Snoopy, Mickey, Spongebob"
anchors_map["instance"]="animal;animal;animal"
contents_map["instance"]="Snoopy, Mickey, Spongebob, Pikachu, Hello Kitty"
targets_map["style"]="Van Gogh;Picasso;Caravaggio"
anchors_map["style"]="painting;painting;painting"
contents_map["style"]="Van Gogh, Picasso, Monet, Paul Gauguin, Caravaggio"

mkdir -p "${CHECKPOINT_DIR}" "${SAVE_ROOT_BASE}"

trim() {
  echo "$1" | xargs
}

limited_target_name() {
  local target="$1"
  local num
  local limited_target

  num=$(echo "$target" | tr -cd ',' | wc -c)
  num=$((num + 1))
  limited_target=$(echo "$target" | awk -F', ' '{for (i=1; i<=NF && i<=5; i++) printf (i<NF && i<5 ? $i "_": $i)}')
  if [[ "$num" -gt 5 ]]; then
    limited_target="${limited_target}_${num}"
  fi
  echo "${limited_target}"
}

run_task() {
  local erase_type="$1"
  local target
  local anchor
  local gpu_id="$4"
  local save_root="$5"
  local limited_target
  local run_name
  local ckpt_path
  local target_root
  local contents
  local content_items
  local content

  target="$(trim "$2")"
  anchor="$(trim "$3")"
  limited_target="$(limited_target_name "$target")"
  run_name="${erase_type}_${limited_target}_to_${anchor}_attn_${PARAMS}_t${TRACE_NUM_STEPS}_thr${THRESHOLD}_lam${UPDATE_LAMBDA}"
  ckpt_path="${CHECKPOINT_DIR}/${run_name}.safetensors"
  target_root="${save_root}/${erase_type}"
  contents="${contents_map[$erase_type]}"

  if [[ "${SCORE_ONLY}" == "1" ]]; then
    RUN_SCORE=1 score_target "${erase_type}" "${target_root}/${limited_target}" "${contents}" "${gpu_id}"
    return
  fi

echo "FLUX: editing [${erase_type}] [${limited_target} -> ${anchor:-null}] on GPU [${gpu_id}] with [params=${PARAMS}, trace_steps=${TRACE_NUM_STEPS}, threshold=${THRESHOLD}]"

  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" attn.py \
    --sd_ckpt "${SD_CKPT}" \
    --device "cuda:0" \
    --target_concepts "${target}" \
    --anchor_concepts "${anchor}" \
    --retain_path "../data/${erase_type}.csv" \
    --heads "concept" \
    --save_path "${CHECKPOINT_DIR}" \
    --file_name "${run_name}" \
    --trace_num_steps "${TRACE_NUM_STEPS}" \
    --threshold "${THRESHOLD}" \
    --params "${PARAMS}" \
    --update_lambda "${UPDATE_LAMBDA}"

  IFS=',' read -ra content_items <<< "${contents}"
  for content in "${content_items[@]}"; do
    content="$(trim "${content}")"
    if [[ -z "${content}" ]]; then
      continue
    fi

    if [[ "${content}" == "coco" ]]; then
      local sample2_args=(
        "${PYTHON_BIN}" sample2.py
        --sd_ckpt "${SD_CKPT}"
        --device "cuda:0"
        --erase_type "${erase_type}"
        --target_concept "${limited_target}"
        --contents "coco"
        --mode "${MODE}"
        --num_samples "${COCO_NUM_SAMPLES}"
        --batch_size "${BATCH_SIZE}"
        --save_root "${target_root}"
        --edit_ckpt "${ckpt_path}"
        --total_timesteps "${TOTAL_TIMESTEPS}"
        --guidance_scale "${GUIDANCE_SCALE}"
      )
      if [[ -n "${MAX_NUM}" ]]; then
        sample2_args+=(--max_num "${MAX_NUM}")
      fi
      CUDA_VISIBLE_DEVICES="${gpu_id}" "${sample2_args[@]}"
    else
      CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" sample.py \
        --sd_ckpt "${SD_CKPT}" \
        --device "cuda:0" \
        --erase_type "${erase_type}" \
        --target_concept "${limited_target}" \
        --contents "${content}" \
        --mode "${MODE}" \
        --num_samples "${NUM_SAMPLES}" \
        --batch_size "${BATCH_SIZE}" \
        --save_root "${target_root}" \
        --edit_ckpt "${ckpt_path}" \
        --total_timesteps "${TOTAL_TIMESTEPS}" \
        --guidance_scale "${GUIDANCE_SCALE}"
    fi
  done

  score_target "${erase_type}" "${target_root}/${limited_target}" "${contents}" "${gpu_id}"
}

score_target() {
  local erase_type="$1"
  local target_path="$2"
  local contents="$3"
  local gpu_id="$4"

  if [[ "${RUN_SCORE}" != "1" ]]; then
    return
  fi

  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" ../score_cal.py \
    --contents "${contents}" \
    --root_path "${target_path}" \
    --sub_root "edit" \
    --pretrained_path "../data/pretrain/${erase_type}"
}

NUM_GPUS=${#GPU_IDX[@]}
gpu_idx=0
save_root="${SAVE_ROOT_BASE}/few_concept"

for erase_type in "${erase_types[@]}"; do
  IFS=';' read -ra targets <<< "${targets_map[$erase_type]}"
  IFS=';' read -ra anchors <<< "${anchors_map[$erase_type]}"
  for i in "${!targets[@]}"; do
    run_task "${erase_type}" "${targets[i]}" "${anchors[i]}" "${GPU_IDX[$gpu_idx]}" "${save_root}" &
    gpu_idx=$((gpu_idx + 1))
    if (( gpu_idx >= NUM_GPUS )); then
      wait
      gpu_idx=0
    fi
  done
  wait
  gpu_idx=0
done

wait
