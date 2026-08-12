#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

RUN_ROOT=${CHECKPOINT_EVAL_RUN_ROOT:-${REPO_ROOT}/output/evaluation/main/checkpoint_generation/retrain_v2_chk0_chk1_chk2_cp150_20260809}
GENERATION_DIR=${RUN_ROOT}/generations
LOG_DIR=${RUN_ROOT}/logs
mkdir -p "${GENERATION_DIR}" "${LOG_DIR}"

wait_for_chk0_session() {
  while tmux has-session -t eval_cp150_chk0_20260809 2>/dev/null; do
    sleep 30
  done
}

run_generation() {
  local artifact=$1
  local gpu=$2
  "${REPO_ROOT}/run/eval_retrain_v2_checkpoint_generation.sh" \
    generate "${artifact}" "${gpu}" >>"${LOG_DIR}/${artifact}.log" 2>&1
}

gpu_has_compute_process() {
  local gpu=$1
  local pids
  pids=$(nvidia-smi --id="${gpu}" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
  [[ -n "${pids}" ]]
}

wait_for_chk0_session
if [[ ! -f "${GENERATION_DIR}/eval-chk0-base.manifest.json" ]]; then
  run_generation eval-chk0-base 1
fi

if gpu_has_compute_process 0; then
  run_generation eval-chk1-compressed-sft 1
  run_generation eval-chk2-reward-v3-cp150 1
else
  run_generation eval-chk1-compressed-sft 0 &
  chk1_pid=$!
  run_generation eval-chk2-reward-v3-cp150 1 &
  chk2_pid=$!
  wait "${chk1_pid}"
  wait "${chk2_pid}"
fi

"${REPO_ROOT}/run/eval_retrain_v2_checkpoint_generation.sh" score \
  >>"${LOG_DIR}/score-strict.log" 2>&1
"${REPO_ROOT}/run/eval_retrain_v2_checkpoint_generation.sh" score-length-tolerant \
  >>"${LOG_DIR}/score-length-tolerant.log" 2>&1
