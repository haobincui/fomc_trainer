#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${ROOT_DIR}/output/training/retrain_v2/chk2_compressed_chk1_v1_reward_v3_long4096_20260805"
TRAIN_LOG="${RUN_DIR}/logs/chk2.log"
MONITOR_LOG="${RUN_DIR}/logs/oom_monitor.log"
REWARD_LOG="${RUN_DIR}/adapters/chk2/reward.jsonl"
POLICY_SESSION="chk2_compressed_policy_v1"
JUDGE_SESSION="chk2_compressed_judge_v1"
CONFIG="configs/retrain_v2/chk2_analysis_grpo_compressed_chk1_v1_20260805.yaml"
PYTHON="/home/haobin_cui/.conda/envs/fomc_trainer/bin/python"
LOCK_PATH="${RUN_DIR}/logs/oom_monitor.lock"

mkdir -p "${RUN_DIR}/logs"
exec 9>"${LOCK_PATH}"
flock -n 9 || {
  echo "$(date -u +%FT%TZ) monitor already running" >&2
  exit 2
}

exec >>"${MONITOR_LOG}" 2>&1
echo "$(date -u +%FT%TZ) monitor started; interval=600s"

latest_checkpoint() {
  find "${RUN_DIR}/adapters/chk2" -maxdepth 1 -type d -name 'checkpoint-*' \
    -printf '%f\n' | sort -V | tail -n 1
}

start_policy() {
  local checkpoint="$1"
  echo "$(date -u +%FT%TZ) restarting policy; latest_checkpoint=${checkpoint}"
  tmux new-session -d -s "${POLICY_SESSION}" \
    "cd ${ROOT_DIR} && export CUDA_VISIBLE_DEVICES=1 PYTORCH_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OPEN_R1_JUDGE_TIMEOUT=600 TOKENIZERS_PARALLELISM=false && ${PYTHON} -m jobs.train.train_grpo --config ${CONFIG} >> ${TRAIN_LOG} 2>&1"
}

policy_alive() {
  tmux has-session -t "${POLICY_SESSION}" 2>/dev/null
}

status_line() {
  local progress rows gpu latest judge
  progress="$(tail -n 240 "${TRAIN_LOG}" 2>/dev/null | tr '\r' '\n' | rg '[0-9]+%\|.*[0-9]+/247' | tail -n 1 || true)"
  rows="$(wc -l < "${REWARD_LOG}" 2>/dev/null || echo 0)"
  gpu="$(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';' || true)"
  latest="$(latest_checkpoint)"
  if tmux has-session -t "${JUDGE_SESSION}" 2>/dev/null; then judge=alive; else judge=dead; fi
  echo "$(date -u +%FT%TZ) status policy=$(policy_alive && echo alive || echo dead) judge=${judge} latest=${latest:-none} rewards=${rows} progress=${progress:-none} gpu=${gpu}"
}

baseline_line="$(wc -l < "${TRAIN_LOG}")"
echo "=== OOM_MONITOR_ATTEMPT $(date -u +%FT%TZ) baseline_line=${baseline_line} ===" >>"${TRAIN_LOG}"
baseline_line="$(wc -l < "${TRAIN_LOG}")"
last_checkpoint="$(latest_checkpoint)"
consecutive_restarts=0

while :; do
  sleep 600
  status_line

  current_checkpoint="$(latest_checkpoint)"
  if [[ -n "${current_checkpoint}" && "${current_checkpoint}" != "${last_checkpoint}" ]]; then
    echo "$(date -u +%FT%TZ) checkpoint advanced ${last_checkpoint:-none} -> ${current_checkpoint}; reset restart counter"
    last_checkpoint="${current_checkpoint}"
    consecutive_restarts=0
  fi

  if policy_alive; then
    continue
  fi

  current_segment="$(tail -n +"${baseline_line}" "${TRAIN_LOG}" 2>/dev/null | tr '\r' '\n')"
  # Treat the trainer's explicit peak-memory guard as an OOM-equivalent exit.
  # It intentionally aborts before CUDA raises torch.OutOfMemoryError when
  # reserved memory crosses the configured 22 GiB safety threshold.
  if ! printf '%s\n' "${current_segment}" | rg -q 'torch.OutOfMemoryError|CUDA out of memory|CUDA error: out of memory|peak reserved memory .* exceeds [0-9]+ GiB'; then
    echo "$(date -u +%FT%TZ) policy exited without a new OOM or memory-guard failure; monitor stopping"
    exit 0
  fi

  consecutive_restarts=$((consecutive_restarts + 1))
  latest="$(latest_checkpoint)"
  if [[ -z "${latest}" ]]; then
    echo "$(date -u +%FT%TZ) OOM detected but no complete checkpoint exists; monitor stopping"
    exit 20
  fi
  echo "$(date -u +%FT%TZ) new OOM/memory-guard failure detected; restart=${consecutive_restarts} from=${latest}"
  echo "=== OOM_MONITOR_RESTART $(date -u +%FT%TZ) from=${latest} ===" >>"${TRAIN_LOG}"
  baseline_line="$(wc -l < "${TRAIN_LOG}")"
  start_policy "${latest}"
done
