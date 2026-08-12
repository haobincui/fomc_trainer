#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_ENV="${FOMC_RETRAIN_TRAIN_ENV:-fomc_trainer}"
COMMAND="${1:-}"
EXECUTE="${2:-}"

case "${COMMAND}" in
  merge-preflight|merge-selected|preflight|status|authorize-smoke|launch-smoke|gate-smoke|authorize-full|launch-full) ;;
  *)
    echo "Usage: $0 merge-preflight|merge-selected|preflight|status|authorize-smoke|launch-smoke|gate-smoke|authorize-full|launch-full [--execute]" >&2
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

cd "${ROOT_DIR}"
TRAIN_PYTHON="$(conda run -n "${TRAIN_ENV}" python -c 'import sys; print(sys.executable)')"
if [[ -z "${TRAIN_PYTHON}" || "${TRAIN_PYTHON}" == *$'\n'* || ! -x "${TRAIN_PYTHON}" ]]; then
  echo "ERROR: unable to resolve ${TRAIN_ENV} Python." >&2
  exit 2
fi

PILOT_ROOT="output/training/retrain_v2/chk4_from_chk1_cp200_sft_grpo_pre2009_balanced_lr1e5_steps39_v1_20260811/selected_sft_checkpoints"
MERGE=(
  "${TRAIN_PYTHON}" -m jobs.retrain_v2.merge_chk4_selected_sft_checkpoint
  --repo-root "${ROOT_DIR}"
  --profile pre2009_balanced_lr1e5_steps39_v1
  --checkpoint-step 38
  --pilot-manifest "${PILOT_ROOT}/pre2009_train_stratified_manifest_v1.json"
  --pilot-manifest-sha256 99853a20f246684b0e03a90087af5dcb3ca23732bf5db06fd5433b09516611a7
  --pilot-summary "${PILOT_ROOT}/pre2009_cp38_train_stratified_probe_v1/summary.json"
  --pilot-summary-sha256 2b33278c9b96b27954e710c0e705b3dc929ade2ecf906d5296e572cc0ba75a0b
)
BASE=(
  "${TRAIN_PYTHON}" -m jobs.retrain_v2.chk4_pre2009_cp38_grpo_continuation
  --repo-root "${ROOT_DIR}"
)

case "${COMMAND}" in
  merge-preflight)
    [[ -z "${EXECUTE}" ]] || { echo "ERROR: merge-preflight does not accept --execute." >&2; exit 2; }
    export CUDA_VISIBLE_DEVICES=""
    exec "${MERGE[@]}"
    ;;
  merge-selected)
    [[ "${EXECUTE}" == "--execute" ]] || { echo "ERROR: merge-selected requires --execute." >&2; exit 2; }
    export CUDA_VISIBLE_DEVICES=""
    exec "${MERGE[@]}" --execute
    ;;
  preflight|status)
    [[ -z "${EXECUTE}" ]] || { echo "ERROR: ${COMMAND} does not accept --execute." >&2; exit 2; }
    exec "${BASE[@]}" "${COMMAND}"
    ;;
  authorize-smoke|gate-smoke|authorize-full)
    exec "${BASE[@]}" "${COMMAND}" ${EXECUTE:+--execute}
    ;;
  launch-smoke|launch-full)
    if [[ -z "${EXECUTE}" ]]; then
      exec "${BASE[@]}" "${COMMAND}"
    fi
    "${ROOT_DIR}/run/retrain_v2/gpu_gate.sh" 1
    RUN_TAG="${COMMAND#launch-}"
    LOG_ROOT="${ROOT_DIR}/output/training/retrain_v2/chk4_from_pre2009_cp38_selected_grpo_${RUN_TAG}_v1_20260811/logs"
    mkdir -p "${LOG_ROOT}"
    exec > >(tee -a "${LOG_ROOT}/grpo.log") 2>&1
    exec "${BASE[@]}" "${COMMAND}" --execute
    ;;
esac
