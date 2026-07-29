#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../_run_common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
ACCELERATE_CONFIG="${LLAMA_ANALYSIS_SFT_ACCELERATE_CONFIG:-${ROOT_DIR}/configs/accelerate/zero3_sft_single.yaml}"
TRAIN_CONFIG="${LLAMA_ANALYSIS_SFT_CONFIG:-${ROOT_DIR}/configs/main/analysis_sft_llama.yaml}"
MAIN_PROCESS_PORT="${LLAMA_ANALYSIS_SFT_MAIN_PROCESS_PORT:-29501}"

prepare_training_job "analysis_sft_llama"
print_job_header "analysis_sft_llama" "analysis_sft llama train + merge"

(
  accelerate launch --config_file "${ACCELERATE_CONFIG}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    -m jobs.train.train_sft \
    --config "${TRAIN_CONFIG}" &&
  python -m jobs.merge_model \
    --config "${TRAIN_CONFIG}"
) > "${LOG_FILE}" 2>&1 &

record_background_pid "$!"
