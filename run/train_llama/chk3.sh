#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../_run_common.sh"

TRAIN_CONFIG="${LLAMA_MINUTES_ALIGNMENT_SFT_CONFIG:-${ROOT_DIR}/configs/main/minutes_alignment_sft_llama.yaml}"
MAIN_PROCESS_PORT="${LLAMA_MINUTES_ALIGNMENT_SFT_MAIN_PROCESS_PORT:-29503}"

require_path "${ROOT_DIR}/output/training/main/merged/analysis_sft_llama"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

prepare_training_job "minutes_alignment_llama"
print_job_header "minutes_alignment_llama" "minutes_alignment llama sft train + merge"

(
  accelerate launch --config_file "${ROOT_DIR}/configs/accelerate/zero3.yaml" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    -m jobs.train.train_sft \
    --config "${TRAIN_CONFIG}" &&
  python -m jobs.merge_model \
    --config "${TRAIN_CONFIG}"
) > "${LOG_FILE}" 2>&1 &

record_background_pid "$!"
