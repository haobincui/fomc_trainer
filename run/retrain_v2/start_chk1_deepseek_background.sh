#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENTRYPOINT="${ROOT_DIR}/run/retrain_v2/chk1_data.sh"
LOG_ROOT="${ROOT_DIR}/output/logs/retrain_v2/chk1_deepseek"
PID_ROOT="${ROOT_DIR}/output/pids/retrain_v2/chk1_deepseek"

usage() {
  cat >&2 <<'EOF'
Usage:
  run/retrain_v2/start_chk1_deepseek_background.sh [all|smoke|pilot|full] [--resume]

Default mode is all, which runs smoke -> verify -> pilot -> verify -> full -> verify.
The script prompts for DEEPSEEK_API_KEY when it is not already exported.

Optional environment:
  DEEPSEEK_TEACHER_MODEL     Defaults to deepseek-v4-pro.
  DEEPSEEK_CONCURRENCY       Concurrent API requests, defaults to 8 (max 32).
  DEEPSEEK_BASE_URL          Defaults to https://api.deepseek.com.
  DEEPSEEK_TEACHER_REVISION Provider/model revision label stored in provenance.
EOF
}

run_phase() {
  local phase="$1"
  local resume_flag="$2"
  local -a args=(generate --mode "${phase}")
  if [[ "${resume_flag}" == "1" ]]; then
    args+=(--resume)
  fi
  "${ENTRYPOINT}" "${args[@]}"
  "${ENTRYPOINT}" verify --mode "${phase}"
}

if [[ "${1:-}" == "__worker" ]]; then
  mode="${2:?worker mode is required}"
  resume_flag="${3:-0}"
  cd "${ROOT_DIR}"
  case "${mode}" in
    all)
      run_phase smoke "${resume_flag}"
      run_phase pilot "${resume_flag}"
      run_phase full "${resume_flag}"
      ;;
    smoke|pilot|full)
      run_phase "${mode}" "${resume_flag}"
      ;;
    *)
      echo "ERROR: unsupported worker mode: ${mode}" >&2
      exit 2
      ;;
  esac
  exit 0
fi

mode="all"
resume_flag="0"
for arg in "$@"; do
  case "${arg}" in
    all|smoke|pilot|full)
      mode="${arg}"
      ;;
    --resume)
      resume_flag="1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unsupported argument: ${arg}" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -x "${ENTRYPOINT}" ]]; then
  echo "ERROR: chk1 entrypoint is not executable: ${ENTRYPOINT}" >&2
  exit 2
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  read -r -s -p "DeepSeek API key: " DEEPSEEK_API_KEY
  echo >&2
fi
if [[ -z "${DEEPSEEK_API_KEY}" ]]; then
  echo "ERROR: DEEPSEEK_API_KEY is empty." >&2
  exit 2
fi
export DEEPSEEK_API_KEY
export FOMC_RETRAIN_TRAIN_ENV="${FOMC_RETRAIN_TRAIN_ENV:-fomc_trainer}"
export DEEPSEEK_TEACHER_MODEL="${DEEPSEEK_TEACHER_MODEL:-deepseek-v4-pro}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
export DEEPSEEK_CONCURRENCY="${DEEPSEEK_CONCURRENCY:-8}"

mkdir -p "${LOG_ROOT}" "${PID_ROOT}"
pid_file="${PID_ROOT}/${mode}.pid"
if [[ -f "${pid_file}" ]]; then
  old_pid="$(<"${pid_file}")"
  if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "ERROR: chk1 DeepSeek ${mode} worker is already running as PID ${old_pid}." >&2
    exit 1
  fi
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="${LOG_ROOT}/${timestamp}_${mode}.log"

nohup "${BASH_SOURCE[0]}" __worker "${mode}" "${resume_flag}" \
  >"${log_file}" 2>&1 </dev/null &
worker_pid="$!"
printf '%s\n' "${worker_pid}" >"${pid_file}"

echo "Started chk1 DeepSeek A generation."
echo "mode=${mode}"
echo "model=${DEEPSEEK_TEACHER_MODEL}"
echo "conda_env=${FOMC_RETRAIN_TRAIN_ENV}"
echo "concurrency=${DEEPSEEK_CONCURRENCY}"
echo "pid=${worker_pid}"
echo "pid_file=${pid_file}"
echo "log=${log_file}"
echo "monitor: tail -f '${log_file}'"
echo "status:  ps -p ${worker_pid} -o pid,etime,stat,cmd"
