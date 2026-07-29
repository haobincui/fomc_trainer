#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../_run_common.sh"

JUDGE_CONFIG_PATH="${JUDGE_CONFIG_PATH:-${ROOT_DIR}/configs/main/judge_chk2.yaml}"

require_path "${JUDGE_CONFIG_PATH}"
eval "$(load_flat_yaml_vars "${JUDGE_CONFIG_PATH}" "JUDGE_CFG_")"

JUDGE_HOST="${JUDGE_HOST:-${JUDGE_CFG_HOST:-127.0.0.1}}"
JUDGE_PORT="${JUDGE_PORT:-${JUDGE_CFG_PORT:-8000}}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-${JUDGE_CFG_MODEL_NAME:-models/gemma-3-12b-it}}"
ANALYSIS_GRPO_ACCELERATE_CONFIG="${ANALYSIS_GRPO_ACCELERATE_CONFIG:-${ROOT_DIR}/configs/accelerate/zero2_no_cpu_offload.yaml}"

require_path "${ROOT_DIR}/output/training/main/merged/analysis_sft"
require_path "${ANALYSIS_GRPO_ACCELERATE_CONFIG}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OPEN_R1_JUDGE_URL="${OPEN_R1_JUDGE_URL:-http://${JUDGE_HOST}:${JUDGE_PORT}/v1/chat/completions}"
export OPEN_R1_JUDGE_MODEL="${OPEN_R1_JUDGE_MODEL:-${JUDGE_MODEL_NAME}}"

prepare_training_job "analysis_grpo"
require_judge_ready "${OPEN_R1_JUDGE_URL}"
print_job_header "analysis_grpo" "analysis_grpo train + merge"

(
  accelerate launch --config_file "${ANALYSIS_GRPO_ACCELERATE_CONFIG}" \
    -m jobs.train.train_grpo \
    --config "${ROOT_DIR}/configs/main/analysis_grpo.yaml" &&
  python -m jobs.merge_model \
    --config "${ROOT_DIR}/configs/main/analysis_grpo.yaml"
) > "${LOG_FILE}" 2>&1 &

record_background_pid "$!"
