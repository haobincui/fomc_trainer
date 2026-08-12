#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_PYTHON="/home/haobin_cui/.conda/envs/fomc_trainer/bin/python"
COMMAND="${1:-}"
EXECUTE="${2:-}"

case "${COMMAND}" in
  preflight|status|authorize|launch) ;;
  *)
    echo "Usage: $0 preflight|status|authorize|launch [--execute]" >&2
    exit 2
    ;;
esac
if [[ -n "${EXECUTE}" && "${EXECUTE}" != "--execute" ]]; then
  echo "ERROR: the only supported second argument is --execute." >&2
  exit 2
fi
if [[ $# -gt 2 ]]; then
  echo "ERROR: too many arguments." >&2
  exit 2
fi
if [[ "${COMMAND}" == "preflight" || "${COMMAND}" == "status" ]]; then
  if [[ -n "${EXECUTE}" ]]; then
    echo "ERROR: ${COMMAND} does not accept --execute." >&2
    exit 2
  fi
fi
if [[ ! -x "${TRAIN_PYTHON}" ]]; then
  echo "ERROR: exact train-env Python is unavailable: ${TRAIN_PYTHON}" >&2
  exit 2
fi

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}"
BASE=(
  "${TRAIN_PYTHON}" -m jobs.retrain_v2.chk4_pre2009_correction_v2_sft
  --repo-root "${ROOT_DIR}"
)

case "${COMMAND}" in
  preflight|status)
    export CUDA_VISIBLE_DEVICES=""
    exec "${BASE[@]}" "${COMMAND}"
    ;;
  authorize)
    export CUDA_VISIBLE_DEVICES=""
    exec "${BASE[@]}" authorize ${EXECUTE:+--execute}
    ;;
  launch)
    if [[ -z "${EXECUTE}" ]]; then
      export CUDA_VISIBLE_DEVICES=""
      exec "${BASE[@]}" launch
    fi
    "${ROOT_DIR}/run/retrain_v2/gpu_gate.sh" 1
    LOG_ROOT="${ROOT_DIR}/output/training/retrain_v2/chk4_from_pre2009_cp38_selected_correction_sft_v2_lr2e6_steps6_20260811/logs"
    mkdir -p "${LOG_ROOT}"
    exec > >(tee -a "${LOG_ROOT}/sft.log") 2>&1
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launching one-shot chk4 correction-v2 SFT on GPU1"
    export CUDA_VISIBLE_DEVICES=1
    exec "${BASE[@]}" launch --execute
    ;;
esac
