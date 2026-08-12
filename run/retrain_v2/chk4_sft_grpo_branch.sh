#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_ENV="${FOMC_RETRAIN_TRAIN_ENV:-fomc_trainer}"
BRANCH_PROFILE="${FOMC_CHK4_BRANCH_PROFILE:-core_v3}"
case "${BRANCH_PROFILE}" in
  core_v3)
    BRANCH_ID="chk4_from_chk1_cp200_sft_grpo_core_v3_20260810"
    ;;
  warm_fix_v1)
    BRANCH_ID="chk4_from_chk1_cp200_sft_grpo_warm_fix_v1_20260810"
    ;;
  warm_fix_lr1e5_v2)
    BRANCH_ID="chk4_from_chk1_cp200_sft_grpo_warm_fix_lr1e5_v2_20260810"
    ;;
  warm_fix_lr1e5_steps24_v3)
    BRANCH_ID="chk4_from_chk1_cp200_sft_grpo_warm_fix_lr1e5_steps24_v3_20260811"
    ;;
  hier_balanced_lr1e5_steps24_v4)
    BRANCH_ID="chk4_from_chk1_cp200_sft_grpo_hier_balanced_lr1e5_steps24_v4_20260811"
    ;;
  hier_balanced_lr1e5_steps24_v5)
    BRANCH_ID="chk4_from_chk1_cp200_sft_grpo_hier_balanced_lr1e5_steps24_v5_20260811"
    ;;
  pre2009_balanced_lr1e5_steps39_v1)
    BRANCH_ID="chk4_from_chk1_cp200_sft_grpo_pre2009_balanced_lr1e5_steps39_v1_20260811"
    ;;
  *)
    echo "ERROR: unsupported FOMC_CHK4_BRANCH_PROFILE=${BRANCH_PROFILE}; expected core_v3, warm_fix_v1, warm_fix_lr1e5_v2, warm_fix_lr1e5_steps24_v3, hier_balanced_lr1e5_steps24_v4, hier_balanced_lr1e5_steps24_v5, or pre2009_balanced_lr1e5_steps39_v1." >&2
    exit 2
    ;;
esac
export FOMC_CHK4_BRANCH_PROFILE="${BRANCH_PROFILE}"
BRANCH_RUN_ROOT="${ROOT_DIR}/output/training/retrain_v2/${BRANCH_ID}"
COMMAND="${1:-}"
EXECUTE=false

if [[ "${2:-}" == "--execute" ]]; then
  EXECUTE=true
elif [[ $# -gt 1 ]]; then
  echo "Usage: $0 preflight|status|prepare-parent|train-sft|merge-sft|train-grpo [--execute]" >&2
  exit 2
fi

case "${COMMAND}" in
  preflight|status|prepare-parent|train-sft|merge-sft|train-grpo) ;;
  *)
    echo "Usage: $0 preflight|status|prepare-parent|train-sft|merge-sft|train-grpo [--execute]" >&2
    exit 2
    ;;
esac

cd "${ROOT_DIR}"
TRAIN_PYTHON="$(conda run -n "${TRAIN_ENV}" python -c 'import sys; print(sys.executable)')"
if [[ -z "${TRAIN_PYTHON}" || "${TRAIN_PYTHON}" == *$'\n'* || ! -x "${TRAIN_PYTHON}" ]]; then
  echo "ERROR: unable to resolve ${TRAIN_ENV} Python." >&2
  exit 2
fi

BASE=("${TRAIN_PYTHON}" -m jobs.retrain_v2.chk4_sft_grpo_branch --repo-root "${ROOT_DIR}")

case "${COMMAND}" in
  preflight|status)
    exec "${BASE[@]}" "${COMMAND}"
    ;;
  prepare-parent|merge-sft)
    if [[ "${EXECUTE}" == true ]]; then
      export CUDA_VISIBLE_DEVICES=1
      exec "${BASE[@]}" "${COMMAND}" --execute
    fi
    exec "${BASE[@]}" "${COMMAND}"
    ;;
  train-sft|train-grpo)
    STAGE="${COMMAND#train-}"
    if [[ "${EXECUTE}" == false ]]; then
      exec "${BASE[@]}" launch --stage "${STAGE}"
    fi
    "${ROOT_DIR}/run/retrain_v2/gpu_gate.sh" 1
    LOG_DIR="${BRANCH_RUN_ROOT}/logs"
    LOG_PATH="${LOG_DIR}/${STAGE}.log"
    mkdir -p "${LOG_DIR}"
    exec > >(tee -a "${LOG_PATH}") 2>&1
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launching chk4 ${STAGE}; profile=${BRANCH_PROFILE}; log=${LOG_PATH}"
    exec "${BASE[@]}" launch --stage "${STAGE}" --execute
    ;;
esac
