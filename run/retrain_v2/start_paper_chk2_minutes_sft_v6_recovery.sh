#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="/home/haobin_cui/.conda/envs/fomc_trainer/bin/python"
RUN_ROOT="${ROOT_DIR}/output/training/retrain_v2/paper_chk2_chk1_cp200_minutes_v6_recovery_full3ep_lr1e6_v1_20260901"
OUTPUT_DIR="${RUN_ROOT}/adapters/chk2"
LOG_PATH="${RUN_ROOT}/logs/train.log"
PID_PATH="${RUN_ROOT}/training.pid"
PREFLIGHT_RECEIPT="${RUN_ROOT}/preflight_receipt.json"
RUNTIME_LAUNCH_RECEIPT="${RUN_ROOT}/runtime_launch_receipt.json"
MODULE="jobs.retrain_v2.paper_chk2_sft_launch"

usage() {
  echo "Usage: $0 preflight|start|status" >&2
}

require_python() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: fomc_trainer Python is unavailable: ${PYTHON_BIN}" >&2
    exit 2
  fi
  local observed
  observed="$(${PYTHON_BIN} -c 'import pathlib,sys; print(pathlib.Path(sys.executable).absolute())')"
  if [[ "${observed}" != "${PYTHON_BIN}" ]]; then
    echo "ERROR: resolved training Python drift: ${observed}" >&2
    exit 2
  fi
}

run_module() {
  (
    cd "${ROOT_DIR}"
    PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}" \
      "${PYTHON_BIN}" -m "${MODULE}" --repo-root "${ROOT_DIR}" "$@"
  )
}

status() {
  run_module status
  if [[ -f "${LOG_PATH}" ]]; then
    echo "Recent log output:"
    tail -n 20 "${LOG_PATH}"
  fi
}

start() {
  if [[ -e "${PID_PATH}" || -L "${PID_PATH}" ]]; then
    echo "ERROR: create-only PID receipt already exists: ${PID_PATH}" >&2
    exit 2
  fi
  if [[ -e "${LOG_PATH}" || -L "${LOG_PATH}" ]]; then
    echo "ERROR: create-only training log already exists: ${LOG_PATH}" >&2
    exit 2
  fi
  if [[ -e "${OUTPUT_DIR}" || -L "${OUTPUT_DIR}" ]]; then
    echo "ERROR: fresh-only adapter output already exists: ${OUTPUT_DIR}" >&2
    exit 2
  fi
  if [[ -e "${PREFLIGHT_RECEIPT}" || -L "${PREFLIGHT_RECEIPT}" ]]; then
    echo "ERROR: create-only preflight receipt already exists: ${PREFLIGHT_RECEIPT}" >&2
    exit 2
  fi
  if [[ -e "${RUNTIME_LAUNCH_RECEIPT}" || -L "${RUNTIME_LAUNCH_RECEIPT}" ]]; then
    echo "ERROR: create-only runtime receipt already exists: ${RUNTIME_LAUNCH_RECEIPT}" >&2
    exit 2
  fi

  # Existing GPU workloads are observe-only co-tenants.  The preflight admits
  # them only when each A30 still has the pinned 17 GiB QLoRA memory budget;
  # the worker repeats and records telemetry immediately before exec.
  run_module preflight >/dev/null

  mkdir -p "${RUN_ROOT}/logs"
  (
    cd "${ROOT_DIR}"
    export CUDA_VISIBLE_DEVICES=0,1
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    export TOKENIZERS_PARALLELISM=false
    export PYTHONUNBUFFERED=1
    export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}"
    nohup setsid "${PYTHON_BIN}" -m "${MODULE}" --repo-root "${ROOT_DIR}" \
      launch --execute >"${LOG_PATH}" 2>&1 </dev/null &
    worker_pid=$!
    (umask 077 && printf '%s\n' "${worker_pid}" >"${PID_PATH}")
  )
  chmod 600 "${LOG_PATH}" "${PID_PATH}"

  local pid
  pid="$(<"${PID_PATH}")"
  for _ in 1 2 3 4 5; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "ERROR: paper chk-2 worker exited during startup; inspect ${LOG_PATH}" >&2
      tail -n 40 "${LOG_PATH}" >&2 || true
      exit 2
    fi
    if grep -Eq 'Process rank: [01].*distributed training: True|Start SFT training' "${LOG_PATH}" 2>/dev/null; then
      break
    fi
    sleep 1
  done

  echo "Started paper chk-2 SFT PID=${pid}"
  echo "Environment=fomc_trainer"
  echo "GPUs=0,1 Processes=2 SharedMinFreeMiB=17408 ExpectedSteps=60"
  echo "Log=${LOG_PATH}"
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

require_python
case "$1" in
  preflight)
    run_module preflight
    ;;
  start)
    start
    ;;
  status)
    status
    ;;
  *)
    usage
    exit 2
    ;;
esac
