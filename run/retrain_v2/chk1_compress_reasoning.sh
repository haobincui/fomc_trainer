#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="/home/haobin_cui/.conda/envs/fomc_trainer/bin/python"
SOURCE_DATASET="${ROOT_DIR}/dataset/processed/retrain_v2/analysis_base_full_v7_automated_v3_20260804/analysis_sft"
OUTPUT_DIR="${ROOT_DIR}/output/data/retrain_v2/chk1/reasoning_compression_flash_max_v1_20260805"
LOG_PATH="${OUTPUT_DIR}/background.log"
PID_PATH="${OUTPUT_DIR}/background.pid"

load_tmux_credential() {
  if [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then
    return
  fi
  if command -v tmux >/dev/null 2>&1; then
    local binding
    binding="$(tmux show-environment -g DEEPSEEK_API_KEY 2>/dev/null || true)"
    if [[ "${binding}" == DEEPSEEK_API_KEY=* ]]; then
      export DEEPSEEK_API_KEY="${binding#DEEPSEEK_API_KEY=}"
    fi
    unset binding
  fi
}

status() {
  if [[ -f "${PID_PATH}" ]]; then
    local pid
    pid="$(<"${PID_PATH}")"
    if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "status=running pid=${pid}"
    else
      echo "status=not_running recorded_pid=${pid}"
    fi
  else
    echo "status=not_started"
  fi
  if [[ -f "${OUTPUT_DIR}/run_manifest.json" ]]; then
    jq '{status,counts,model,api,reasoning,concurrency,compressed_reasoning_tokens}' \
      "${OUTPUT_DIR}/run_manifest.json"
  fi
  if [[ -f "${LOG_PATH}" ]]; then
    tail -n 20 "${LOG_PATH}"
  fi
}

start() {
  local resume_flag="${1:-}"
  load_tmux_credential
  if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: DEEPSEEK_API_KEY is not set in this shell or tmux global environment." >&2
    exit 2
  fi
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: fomc_trainer Python is unavailable: ${PYTHON_BIN}" >&2
    exit 2
  fi
  mkdir -p "${OUTPUT_DIR}"
  if [[ -f "${PID_PATH}" ]]; then
    local prior_pid
    prior_pid="$(<"${PID_PATH}")"
    if [[ "${prior_pid}" =~ ^[0-9]+$ ]] && kill -0 "${prior_pid}" 2>/dev/null; then
      echo "ERROR: compression is already running as PID ${prior_pid}" >&2
      exit 2
    fi
  fi
  if [[ -f "${OUTPUT_DIR}/run_manifest.json" && "${resume_flag}" != "--resume" ]]; then
    echo "ERROR: an existing run requires the resume command." >&2
    exit 2
  fi
  local -a command=(
    "${PYTHON_BIN}" -m jobs.retrain_v2.compress_chk1_reasoning
    --dataset "${SOURCE_DATASET}"
    --output "${OUTPUT_DIR}"
    --tokenizer "${ROOT_DIR}/models/DeepSeek-R1-Distill-Llama-8B"
    --model deepseek-v4-flash
    --concurrency 1000
    --max-output-tokens 32768
    --min-reasoning-tokens 512
    --max-reasoning-tokens 2400
    --retries 3
    --timeout 180
  )
  if [[ "${resume_flag}" == "--resume" ]]; then
    command+=(--resume)
  fi
  (
    cd "${ROOT_DIR}"
    nohup "${command[@]}" >>"${LOG_PATH}" 2>&1 < /dev/null &
    echo "$!" >"${PID_PATH}"
  )
  chmod 600 "${PID_PATH}" "${LOG_PATH}"
  echo "started pid=$(<"${PID_PATH}") output=${OUTPUT_DIR} concurrency=1000 model=deepseek-v4-flash"
}

case "${1:-status}" in
  start)
    start
    ;;
  resume)
    start --resume
    ;;
  status)
    status
    ;;
  *)
    echo "usage: $0 {start|resume|status}" >&2
    exit 2
    ;;
esac
