#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_run_common.sh"

JUDGE_CONFIG_PATH="${JUDGE_CONFIG_PATH:-${ROOT_DIR}/configs/main/judge_chk2.yaml}"

require_path "${JUDGE_CONFIG_PATH}"
eval "$(load_flat_yaml_vars "${JUDGE_CONFIG_PATH}" "JUDGE_CFG_")"

JUDGE_HOST="${JUDGE_HOST:-${JUDGE_CFG_HOST:-127.0.0.1}}"
JUDGE_PORT="${JUDGE_PORT:-${JUDGE_CFG_PORT:-8000}}"
JUDGE_GPU_IDS="${JUDGE_GPU_IDS:-${JUDGE_CFG_GPU_IDS:-0}}"
JUDGE_TP_SIZE="${JUDGE_TP_SIZE:-${JUDGE_CFG_TP_SIZE:-1}}"
JUDGE_GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION:-${JUDGE_CFG_GPU_MEMORY_UTILIZATION:-0.9}}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-${JUDGE_CFG_MAX_MODEL_LEN:-8192}}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-${JUDGE_CFG_MODEL_PATH:-models/gemma-3-12b-it}}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-${JUDGE_CFG_MODEL_NAME:-models/gemma-3-12b-it}}"
JUDGE_DISABLE_CUSTOM_ALL_REDUCE="${JUDGE_DISABLE_CUSTOM_ALL_REDUCE:-${JUDGE_CFG_DISABLE_CUSTOM_ALL_REDUCE:-auto}}"
JUDGE_ENFORCE_EAGER="${JUDGE_ENFORCE_EAGER:-${JUDGE_CFG_ENFORCE_EAGER:-0}}"
JUDGE_EXTRA_ARGS="${JUDGE_EXTRA_ARGS:-${JUDGE_CFG_EXTRA_ARGS:-}}"

if [[ "${JUDGE_MODEL_PATH}" != /* ]]; then
  JUDGE_MODEL_PATH="${ROOT_DIR}/${JUDGE_MODEL_PATH}"
fi

require_path "${JUDGE_MODEL_PATH}"

export CUDA_VISIBLE_DEVICES="${JUDGE_GPU_IDS}"
export OPEN_R1_JUDGE_URL="${OPEN_R1_JUDGE_URL:-http://${JUDGE_HOST}:${JUDGE_PORT}/v1/chat/completions}"
export OPEN_R1_JUDGE_MODEL="${OPEN_R1_JUDGE_MODEL:-${JUDGE_MODEL_NAME}}"

prepare_judge_job "analysis_grpo_judge"
print_job_header "analysis_grpo_judge" "judge server for analysis_grpo"

if [[ "${JUDGE_MODEL_PATH}" == *"gemma-3-12b-it"* ]] && [[ "${JUDGE_TP_SIZE}" == "1" ]]; then
  visible_gpu="${CUDA_VISIBLE_DEVICES%%,*}"
  total_mem_mib="$(nvidia-smi --id="${visible_gpu}" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]')"
  if [[ -n "${total_mem_mib}" ]] && [[ "${total_mem_mib}" -lt 40000 ]]; then
    echo "ERROR: ${JUDGE_MODEL_NAME} on a single ${total_mem_mib} MiB GPU is expected to OOM under vLLM." >&2
    echo "Try one of the following:" >&2
    echo "  JUDGE_GPU_IDS=0,1 JUDGE_TP_SIZE=2 ./run/judge_chk2.sh" >&2
    echo "  JUDGE_MODEL_PATH=<smaller-or-quantized-model> JUDGE_MODEL_NAME=<served-name> ./run/judge_chk2.sh" >&2
    exit 1
  fi
fi

vllm_args=(
  --host "${JUDGE_HOST}"
  --port "${JUDGE_PORT}"
  --model "${JUDGE_MODEL_PATH}"
  --served-model-name "${JUDGE_MODEL_NAME}"
  --tensor-parallel-size "${JUDGE_TP_SIZE}"
  --gpu-memory-utilization "${JUDGE_GPU_MEMORY_UTILIZATION}"
  --max-model-len "${JUDGE_MAX_MODEL_LEN}"
)

if [[ "${JUDGE_ENFORCE_EAGER}" == "1" ]]; then
  vllm_args+=(--enforce-eager)
fi

# Multi-GPU judge startup is more reliable on this host when vLLM's
# custom all-reduce path is disabled.
if [[ "${JUDGE_DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]] || {
  [[ "${JUDGE_DISABLE_CUSTOM_ALL_REDUCE}" == "auto" ]] && [[ "${JUDGE_TP_SIZE}" != "1" ]]
}; then
  vllm_args+=(--disable-custom-all-reduce)
fi

# Allow advanced one-off overrides without baking every vLLM flag into this script.
if [[ -n "${JUDGE_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra_args=( ${JUDGE_EXTRA_ARGS} )
  vllm_args+=("${extra_args[@]}")
fi

(
  python -m vllm.entrypoints.openai.api_server "${vllm_args[@]}"
) > "${LOG_FILE}" 2>&1 &

record_background_pid "$!"
