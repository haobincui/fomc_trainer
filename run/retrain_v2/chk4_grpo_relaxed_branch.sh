#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_ENV="${FOMC_RETRAIN_TRAIN_ENV:-fomc_trainer}"
BRANCH_ID="chk4_grpo_relaxed_v3_from_warm_fix_lr1e5_v2_20260810"
RUN_ROOT="${ROOT_DIR}/output/training/retrain_v2/${BRANCH_ID}"
COMMAND="${1:-}"
EXECUTE=false

if [[ "${2:-}" == "--execute" ]]; then
  EXECUTE=true
elif [[ $# -gt 1 ]]; then
  echo "Usage: $0 preflight|status|authorize|train [--execute]" >&2
  exit 2
fi

case "${COMMAND}" in
  preflight|status|authorize|train) ;;
  *)
    echo "Usage: $0 preflight|status|authorize|train [--execute]" >&2
    exit 2
    ;;
esac

cd "${ROOT_DIR}"
TRAIN_PYTHON="$(conda run -n "${TRAIN_ENV}" python -c 'import sys; print(sys.executable)')"
if [[ -z "${TRAIN_PYTHON}" || "${TRAIN_PYTHON}" == *$'\n'* || ! -x "${TRAIN_PYTHON}" ]]; then
  echo "ERROR: unable to resolve ${TRAIN_ENV} Python." >&2
  exit 2
fi

BASE=(
  "${TRAIN_PYTHON}"
  -m jobs.retrain_v2.chk4_grpo_relaxed_branch
  --repo-root "${ROOT_DIR}"
)

case "${COMMAND}" in
  preflight|status)
    exec "${BASE[@]}" "${COMMAND}"
    ;;
  authorize)
    if [[ "${EXECUTE}" == true ]]; then
      exec "${BASE[@]}" authorize --execute
    fi
    exec "${BASE[@]}" authorize
    ;;
  train)
    if [[ "${EXECUTE}" == false ]]; then
      exec "${BASE[@]}" launch
    fi
    "${ROOT_DIR}/run/retrain_v2/gpu_gate.sh" 1
    LOG_DIR="${RUN_ROOT}/logs"
    LOG_PATH="${LOG_DIR}/grpo.log"
    mkdir -p "${LOG_DIR}"
    exec > >(tee -a "${LOG_PATH}") 2>&1
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launching isolated chk4 GRPO v3 on physical GPU1; log=${LOG_PATH}"
    exec "${BASE[@]}" launch --execute
    ;;
esac
