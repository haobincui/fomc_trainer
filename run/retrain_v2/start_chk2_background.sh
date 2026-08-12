#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="${ROOT_DIR}/run/retrain_v2/start_chk2_background.sh"
JUDGE_URL="http://127.0.0.1:8000/v1/models"
RUN_MANIFEST=""
MAX_STAGE_ATTEMPTS=6

usage() {
  echo "Usage: $0 --run-manifest PATH [--max-stage-attempts 1..6]" >&2
}

cleanup_judge() {
  if [[ -n "${JUDGE_PID:-}" ]] && kill -0 "${JUDGE_PID}" 2>/dev/null; then
    kill -TERM -- "-${JUDGE_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "${JUDGE_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-${JUDGE_PID}" 2>/dev/null || true
    wait "${JUDGE_PID}" 2>/dev/null || true
  fi
  JUDGE_PID=""
}

start_judge() {
  cleanup_judge
  echo "[$(date -u +%FT%TZ)] starting pinned Qwen3.5-9B judge"
  setsid env FOMC_RETRAIN_JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN}" \
    "${ROOT_DIR}/run/retrain_v2/judge.sh" \
    >>"${JUDGE_LOG}" 2>&1 </dev/null &
  JUDGE_PID=$!

  for readiness_attempt in $(seq 1 180); do
    if ! kill -0 "${JUDGE_PID}" 2>/dev/null; then
      echo "ERROR: judge exited during startup; inspect ${JUDGE_LOG}" >&2
      return 2
    fi
    if curl --fail --silent --show-error --max-time 5 "${JUDGE_URL}" \
      | grep -q 'Qwen3.5-9B'; then
      echo "[$(date -u +%FT%TZ)] judge endpoint is ready"
      return 0
    fi
    if ((readiness_attempt == 180)); then
      echo "ERROR: judge did not become ready within 30 minutes" >&2
      return 2
    fi
    sleep 10
  done
}

worker() {
  RUN_MANIFEST="$(realpath "${1}")"
  MAX_STAGE_ATTEMPTS="${2}"
  RUN_ROOT="$(dirname "${RUN_MANIFEST}")"
  LOG_ROOT="${RUN_ROOT}/logs"
  JUDGE_LOG="${LOG_ROOT}/judge.log"
  TRAIN_LOG="${LOG_ROOT}/chk2.log"
  JUDGE_MAX_MODEL_LEN="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["judge"]["max_model_len"])' "${RUN_MANIFEST}")"
  if [[ "${JUDGE_MAX_MODEL_LEN}" != "8192" && "${JUDGE_MAX_MODEL_LEN}" != "12288" ]]; then
    echo "ERROR: unsupported manifest judge max_model_len: ${JUDGE_MAX_MODEL_LEN}" >&2
    return 2
  fi
  mkdir -p "${LOG_ROOT}"

  trap cleanup_judge EXIT INT TERM HUP
  for stage_attempt in $(seq 1 "${MAX_STAGE_ATTEMPTS}"); do
    echo "[$(date -u +%FT%TZ)] chk2 unattended attempt ${stage_attempt}/${MAX_STAGE_ATTEMPTS}"
    if start_judge && "${ROOT_DIR}/run/retrain_v2/stage.sh" \
      --run-manifest "${RUN_MANIFEST}" \
      --stage chk2 \
      --execute >>"${TRAIN_LOG}" 2>&1; then
      echo "[$(date -u +%FT%TZ)] chk2 training, merge, and seal completed"
      return 0
    else
      stage_status=$?
    fi
    echo "[$(date -u +%FT%TZ)] chk2 attempt ${stage_attempt} failed with status ${stage_status}" >&2
    cleanup_judge
    if ((stage_attempt == MAX_STAGE_ATTEMPTS)); then
      echo "ERROR: chk2 exhausted ${MAX_STAGE_ATTEMPTS} unattended attempts" >&2
      return "${stage_status}"
    fi
    echo "[$(date -u +%FT%TZ)] retrying from the latest verified checkpoint in 30 seconds"
    sleep 30
  done
}

if [[ "${1:-}" == "__worker" ]]; then
  [[ $# -eq 3 ]] || { usage; exit 2; }
  worker "$2" "$3"
  exit $?
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-manifest)
      RUN_MANIFEST="${2:-}"
      shift 2
      ;;
    --max-stage-attempts)
      MAX_STAGE_ATTEMPTS="${2:-}"
      shift 2
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

[[ -n "${RUN_MANIFEST}" ]] || { usage; exit 2; }
[[ "${MAX_STAGE_ATTEMPTS}" =~ ^[1-6]$ ]] || {
  echo "ERROR: --max-stage-attempts must be an integer in 1..6." >&2
  exit 2
}
command -v tmux >/dev/null 2>&1 || {
  echo "ERROR: tmux is required for persistent chk2 execution." >&2
  exit 2
}
RUN_MANIFEST="$(realpath "${RUN_MANIFEST}")"
[[ -f "${RUN_MANIFEST}" ]] || {
  echo "ERROR: run manifest is missing: ${RUN_MANIFEST}" >&2
  exit 2
}
RUN_ROOT="$(dirname "${RUN_MANIFEST}")"
RUN_ID="$(basename "${RUN_ROOT}")"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || {
  echo "ERROR: unsafe run ID: ${RUN_ID}" >&2
  exit 2
}
SESSION_NAME="fomc_chk2_${RUN_ID}"
if tmux has-session -t "=${SESSION_NAME}" 2>/dev/null; then
  echo "ERROR: tmux session already exists: ${SESSION_NAME}" >&2
  exit 2
fi

LOG_ROOT="${RUN_ROOT}/logs"
ORCHESTRATOR_LOG="${LOG_ROOT}/orchestrator.log"
mkdir -p "${LOG_ROOT}"
WORKER_COMMAND="$(printf '%q ' "${SCRIPT_PATH}" __worker "${RUN_MANIFEST}" "${MAX_STAGE_ATTEMPTS}")"
WORKER_COMMAND+=">>$(printf '%q' "${ORCHESTRATOR_LOG}") 2>&1"
tmux new-session -d -s "${SESSION_NAME}" "${WORKER_COMMAND}"

echo "Started persistent chk2 session: ${SESSION_NAME}"
echo "Orchestrator log: ${ORCHESTRATOR_LOG}"
echo "Training log: ${LOG_ROOT}/chk2.log"
echo "Judge log: ${LOG_ROOT}/judge.log"
