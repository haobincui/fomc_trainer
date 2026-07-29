#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../_run_common.sh"

DECISION_SFT_CONFIG="${LLAMA_DECISION_SFT_CONFIG:-${ROOT_DIR}/configs/main/decision_sft_llama.yaml}"
DECISION_GRPO_CONFIG="${LLAMA_DECISION_GRPO_CONFIG:-${ROOT_DIR}/configs/main/decision_grpo_llama.yaml}"
DECISION_SFT_MAIN_PROCESS_PORT="${LLAMA_DECISION_SFT_MAIN_PROCESS_PORT:-29504}"
DECISION_GRPO_MAIN_PROCESS_PORT="${LLAMA_DECISION_GRPO_MAIN_PROCESS_PORT:-29505}"

require_path "${ROOT_DIR}/output/training/main/merged/analysis_sft_llama"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

prepare_training_job "decision_llama"
enable_grpo_judge
print_job_header "decision_llama" "decision llama sft train + merge, then decision llama grpo train + merge"

(
  accelerate launch --config_file "${ROOT_DIR}/configs/accelerate/zero3.yaml" \
    --main_process_port "${DECISION_SFT_MAIN_PROCESS_PORT}" \
    -m jobs.train.train_sft \
    --config "${DECISION_SFT_CONFIG}" &&
  python -m jobs.merge_model \
    --config "${DECISION_SFT_CONFIG}" &&
  accelerate launch --config_file "${ROOT_DIR}/configs/accelerate/zero2.yaml" \
    --main_process_port "${DECISION_GRPO_MAIN_PROCESS_PORT}" \
    -m jobs.train.train_grpo \
    --config "${DECISION_GRPO_CONFIG}" &&
  python -m jobs.merge_model \
    --config "${DECISION_GRPO_CONFIG}"
) > "${LOG_FILE}" 2>&1 &

record_background_pid "$!"
