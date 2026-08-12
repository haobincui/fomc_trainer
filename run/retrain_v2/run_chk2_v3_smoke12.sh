#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_ENV="${FOMC_RETRAIN_TRAIN_ENV:-fomc_trainer}"
RUN_MANIFEST=""
JUDGE_PID=""

usage() {
  echo "Usage: $0 --run-manifest PATH" >&2
}

cleanup_judge() {
  if [[ -n "${JUDGE_PID}" ]] && kill -0 "${JUDGE_PID}" 2>/dev/null; then
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-manifest)
      RUN_MANIFEST="${2:-}"
      shift 2
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

[[ -n "${RUN_MANIFEST}" ]] || { usage; exit 2; }
cd "${ROOT_DIR}"
RUN_MANIFEST="$(realpath "${RUN_MANIFEST}")"
[[ -f "${RUN_MANIFEST}" ]] || {
  echo "ERROR: run manifest is missing: ${RUN_MANIFEST}" >&2
  exit 2
}
TRAIN_PYTHON="$(conda run -n "${TRAIN_ENV}" python -c 'import sys; print(sys.executable)')"
CONFIG_PATH="$("${TRAIN_PYTHON}" -m jobs.retrain_v2.dag \
  --repo-root "${ROOT_DIR}" stage-config-path \
  --run-manifest "${RUN_MANIFEST}" --stage chk2)"
RUN_ROOT="$(dirname "${RUN_MANIFEST}")"
SMOKE_ROOT="${RUN_ROOT}/smoke/chk2_v3_smoke12"
SMOKE_OUTPUT="${SMOKE_ROOT}/policy"
JUDGE_LOG="${SMOKE_ROOT}/judge.log"
TRAIN_LOG="${SMOKE_ROOT}/train.log"

if [[ -e "${SMOKE_ROOT}" || -L "${SMOKE_ROOT}" ]]; then
  echo "ERROR: smoke output already exists: ${SMOKE_ROOT}" >&2
  exit 2
fi
mkdir -p "${SMOKE_ROOT}"
trap cleanup_judge EXIT INT TERM HUP

"${ROOT_DIR}/run/retrain_v2/stage.sh" \
  --run-manifest "${RUN_MANIFEST}" --stage chk2
"${ROOT_DIR}/run/retrain_v2/gpu_gate.sh" 0 1
FOMC_RETRAIN_TRAIN_GPUS=1 \
  "${ROOT_DIR}/run/check_retrain_v2_envs.sh" --train --skip-nccl

setsid "${ROOT_DIR}/run/retrain_v2/judge.sh" >>"${JUDGE_LOG}" 2>&1 </dev/null &
JUDGE_PID=$!
for readiness_attempt in $(seq 1 180); do
  if ! kill -0 "${JUDGE_PID}" 2>/dev/null; then
    echo "ERROR: judge exited during smoke startup; inspect ${JUDGE_LOG}" >&2
    exit 2
  fi
  if curl --fail --silent --show-error --max-time 5 \
    http://127.0.0.1:8000/v1/models | grep -q 'Qwen3.5-9B'; then
    break
  fi
  if ((readiness_attempt == 180)); then
    echo "ERROR: judge did not become ready within 30 minutes" >&2
    exit 2
  fi
  sleep 10
done

"${TRAIN_PYTHON}" -m jobs.retrain_v2.judge_health \
  --config "${CONFIG_PATH}" \
  --run-manifest "${RUN_MANIFEST}" \
  --repo-root "${ROOT_DIR}" >"${SMOKE_ROOT}/judge_health.json"

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
"${TRAIN_PYTHON}" -m jobs.train.train_grpo \
  --config "${CONFIG_PATH}" \
  --max_steps 12 \
  --output_dir "${SMOKE_OUTPUT}" >>"${TRAIN_LOG}" 2>&1

"${TRAIN_PYTHON}" -m jobs.retrain_v2.check_chk2_v3_smoke \
  --output-dir "${SMOKE_OUTPUT}" \
  --expected-steps 12 \
  --enforce | tee "${SMOKE_ROOT}/gate.stdout.json"

echo "chk2 reward-v3 12-step smoke passed: ${SMOKE_ROOT}"
