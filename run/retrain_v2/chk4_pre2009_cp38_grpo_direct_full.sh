#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_ENV="${FOMC_RETRAIN_TRAIN_ENV:-fomc_trainer}"
COMMAND="${1:-}"
EXECUTE="${2:-}"

case "${COMMAND}" in
  preflight|status|authorize|launch) ;;
  *) echo "Usage: $0 preflight|status|authorize|launch [--execute]" >&2; exit 2 ;;
esac
if [[ -n "${EXECUTE}" && "${EXECUTE}" != "--execute" ]]; then
  echo "ERROR: the only supported second argument is --execute." >&2
  exit 2
fi
if [[ $# -gt 2 ]]; then
  echo "ERROR: too many arguments." >&2
  exit 2
fi

cd "${ROOT_DIR}"
TRAIN_PYTHON="$(conda run -n "${TRAIN_ENV}" python -c 'import sys; print(sys.executable)')"
if [[ -z "${TRAIN_PYTHON}" || "${TRAIN_PYTHON}" == *$'\n'* || ! -x "${TRAIN_PYTHON}" ]]; then
  echo "ERROR: unable to resolve ${TRAIN_ENV} Python." >&2
  exit 2
fi
BASE=(
  "${TRAIN_PYTHON}" -m jobs.retrain_v2.chk4_pre2009_cp38_grpo_direct_full
  --repo-root "${ROOT_DIR}"
)

case "${COMMAND}" in
  preflight|status)
    [[ -z "${EXECUTE}" ]] || { echo "ERROR: ${COMMAND} does not accept --execute." >&2; exit 2; }
    exec "${BASE[@]}" "${COMMAND}"
    ;;
  authorize)
    exec "${BASE[@]}" authorize ${EXECUTE:+--execute}
    ;;
  launch)
    if [[ -z "${EXECUTE}" ]]; then
      exec "${BASE[@]}" launch
    fi
    "${ROOT_DIR}/run/retrain_v2/gpu_gate.sh" 1
    LOG_ROOT="${ROOT_DIR}/output/training/retrain_v2/chk4_from_pre2009_cp38_direct_grpo_full_no_smoke_v1_20260812/logs"
    mkdir -p "${LOG_ROOT}"
    exec > >(tee -a "${LOG_ROOT}/grpo.log") 2>&1
    CUDA_VISIBLE_DEVICES=1 exec "${BASE[@]}" launch --execute
    ;;
esac
