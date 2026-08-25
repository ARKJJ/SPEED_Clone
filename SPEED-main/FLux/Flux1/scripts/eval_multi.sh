#!/usr/bin/env bash
set -euo pipefail

declare -A targets_map
declare -A anchors_map
declare -A contents_map

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ========== Input Params ==========
SD_CKPT="${SD_CKPT:-black-forest-labs/FLUX.1-dev}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-logs/checkpoints}"
SAVE_ROOT_BASE="${SAVE_ROOT_BASE:-logs/FLUX1_DEV}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRACE_NUM_STEPS="${TRACE_NUM_STEPS:-20}"
THRESHOLD="${THRESHOLD:-5e-2}"
UPDATE_LAMBDA="${UPDATE_LAMBDA:-0.1}"
MODE="${MODE:-edit}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
BATCH_SIZE="${BATCH_SIZE:-10}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"
MAX_NUM="${MAX_NUM:-}"
RUN_SCORE="${RUN_SCORE:-0}"
GCD_SCRIPT="${GCD_SCRIPT:-}"
IFS=',' read -ra GPU_IDX <<< "${GPU_IDS:-0,1,2}"
# ==================================

# Erase Task Config
erase_types=("10_celebrity" "50_celebrity" "100_celebrity")

# ==================================================================
targets_map["10_celebrity"]="\
Adam Driver, Adriana Lima, Amber Heard, Amy Adams, Andrew Garfield, Angelina Jolie, Anjelica Huston, Anna Faris, Anna Kendrick, Anne Hathaway\
"
anchors_map["10_celebrity"]="person"
contents_map["10_celebrity"]="erase, retain"
# contents_map["10_celebrity"]="coco"

targets_map["50_celebrity"]="\
Adam Driver, Adriana Lima, Amber Heard, Amy Adams, Andrew Garfield, Angelina Jolie, Anjelica Huston, Anna Faris, Anna Kendrick, Anne Hathaway, Arnold Schwarzenegger, Barack Obama, Beth Behrs, Bill Clinton, Bob Dylan, Bob Marley, Bradley Cooper, Bruce Willis, Bryan Cranston, Cameron Diaz, Channing Tatum, Charlie Sheen, Charlize Theron, Chris Evans, Chris Hemsworth, Chris Pine, Chuck Norris, Courteney Cox, Demi Lovato, Drake, Drew Barrymore, Dwayne Johnson, Ed Sheeran, Elon Musk, Elvis Presley, Emma Stone, Frida Kahlo, George Clooney, Glenn Close, Gwyneth Paltrow, Harrison Ford, Hillary Clinton, Hugh Jackman, Idris Elba, Jake Gyllenhaal, James Franco, Jared Leto, Jason Momoa, Jennifer Aniston, Jennifer Lawrence\
"
anchors_map["50_celebrity"]="person"
contents_map["50_celebrity"]="erase, retain"
# contents_map["50_celebrity"]="coco"

targets_map["100_celebrity"]="\
Adam Driver, Adriana Lima, Amber Heard, Amy Adams, Andrew Garfield, Angelina Jolie, Anjelica Huston, Anna Faris, Anna Kendrick, Anne Hathaway, Arnold Schwarzenegger, Barack Obama, Beth Behrs, Bill Clinton, Bob Dylan, Bob Marley, Bradley Cooper, Bruce Willis, Bryan Cranston, Cameron Diaz, Channing Tatum, Charlie Sheen, Charlize Theron, Chris Evans, Chris Hemsworth, Chris Pine, Chuck Norris, Courteney Cox, Demi Lovato, Drake, Drew Barrymore, Dwayne Johnson, Ed Sheeran, Elon Musk, Elvis Presley, Emma Stone, Frida Kahlo, George Clooney, Glenn Close, Gwyneth Paltrow, Harrison Ford, Hillary Clinton, Hugh Jackman, Idris Elba, Jake Gyllenhaal, James Franco, Jared Leto, Jason Momoa, Jennifer Aniston, Jennifer Lawrence, Jennifer Lopez, Jeremy Renner, Jessica Biel, Jessica Chastain, John Oliver, John Wayne, Johnny Depp, Julianne Hough, Justin Timberlake, Kate Bosworth, Kate Winslet, Leonardo Dicaprio, Margot Robbie, Mariah Carey, Melania Trump, Meryl Streep, Mick Jagger, Mila Kunis, Milla Jovovich, Morgan Freeman, Nick Jonas, Nicolas Cage, Nicole Kidman, Octavia Spencer, Olivia Wilde, Oprah Winfrey, Paul Mccartney, Paul Walker, Peter Dinklage, Philip Seymour Hoffman, Reese Witherspoon, Richard Gere, Ricky Gervais, Rihanna, Robin Williams, Ronald Reagan, Ryan Gosling, Ryan Reynolds, Shia Labeouf, Shirley Temple, Spike Lee, Stan Lee, Theresa May, Tom Cruise, Tom Hanks, Tom Hardy, Tom Hiddleston, Whoopi Goldberg, Zac Efron, Zayn Malik\
"
anchors_map["100_celebrity"]="person"
contents_map["100_celebrity"]="erase, retain"
# contents_map["100_celebrity"]="coco"
# ==================================================================

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
  local target="$2"
  local anchor="$3"
  local gpu_id="$4"
  local save_root="$5"

  target="$(trim "$target")"
  anchor="$(trim "$anchor")"

  local limited_target
  local anchor_name
  local run_name
  local ckpt_path
  local target_root
  local contents
  local sample_args

  limited_target="$(limited_target_name "$target")"
  anchor_name="${anchor:-null}"
  anchor_name="${anchor_name//, /_}"
  anchor_name="${anchor_name// /_}"
  run_name="${erase_type}_${limited_target}_to_${anchor_name}_mlp_flux1dev_t${TRACE_NUM_STEPS}_thr${THRESHOLD}"
  ckpt_path="${CHECKPOINT_DIR}/${run_name}.safetensors"
  target_root="${save_root}/${erase_type}"
  contents="${contents_map[$erase_type]}"

  echo "FLUX.1-dev MLP: editing [${erase_type}] [${limited_target} -> ${anchor}] on GPU [${gpu_id}] with [trace_steps=${TRACE_NUM_STEPS}, threshold=${THRESHOLD}]"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" mlp.py \
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
    --update_lambda "${UPDATE_LAMBDA}"

  sample_args=(
    "${PYTHON_BIN}" sample2.py
    --sd_ckpt "${SD_CKPT}"
    --device "cuda:0"
    --erase_type "${erase_type}"
    --target_concept "${limited_target}"
    --contents "${contents}"
    --mode "${MODE}"
    --num_samples "${NUM_SAMPLES}"
    --batch_size "${BATCH_SIZE}"
    --save_root "${target_root}"
    --edit_ckpt "${ckpt_path}"
    --dataset_path "../data/${erase_type}.csv"
    --total_timesteps "${TOTAL_TIMESTEPS}"
    --guidance_scale "${GUIDANCE_SCALE}"
  )
  if [[ -n "${MAX_NUM}" ]]; then
    sample_args+=(--max_num "${MAX_NUM}")
  fi
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${sample_args[@]}"

  echo "SPEED-style paths:"
  echo "  ${target_root}/${limited_target}/erase/edit"
  echo "  ${target_root}/${limited_target}/retain/edit"

  if [[ -n "${GCD_SCRIPT}" ]]; then
    mkdir -p "${target_root}/${limited_target}/metrics"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${GCD_SCRIPT}" \
      --image_folder "${target_root}/${limited_target}/erase/edit" \
      > "${target_root}/${limited_target}/metrics/gcd_erase.txt"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${GCD_SCRIPT}" \
      --image_folder "${target_root}/${limited_target}/retain/edit" \
      > "${target_root}/${limited_target}/metrics/gcd_retain.txt"
  fi

  if [[ "${RUN_SCORE}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" ../score_cal.py \
      --contents "${contents}" \
      --root_path "${target_root}" \
      --sub_root "edit" \
      --pretrained_path "../data/pretrain/${erase_type}"
  fi
}

NUM_GPUS=${#GPU_IDX[@]}
gpu_idx=0
save_root="${SAVE_ROOT_BASE}/multi_concept"

for erase_type in "${erase_types[@]}"; do
  run_task "${erase_type}" "${targets_map[$erase_type]}" "${anchors_map[$erase_type]}" "${GPU_IDX[$gpu_idx]}" "${save_root}" &
  gpu_idx=$((gpu_idx + 1))

  if (( gpu_idx >= NUM_GPUS )); then
    wait
    gpu_idx=0
  fi
done

wait

if [[ -z "${GCD_SCRIPT}" ]]; then
  echo "GCD_SCRIPT is not set. Run GCD manually on the printed erase/edit and retain/edit folders if needed."
fi
