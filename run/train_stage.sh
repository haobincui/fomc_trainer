#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${FOMC_TRAINER_CONDA_ENV:-fomc_trainer}"
LOG_DIR="${ROOT_DIR}/logs/train"

usage() {
  cat <<'EOF'
Usage:
  ./run/train_stage.sh <stage>

Stages:
  analysis_sft
  analysis_grpo
  minutes_alignment_sft
  decision_sft
  decision_grpo

Environment overrides:
  FOMC_TRAINER_CONDA_ENV
  CUDA_VISIBLE_DEVICES
  PYTORCH_CUDA_ALLOC_CONF
  OPEN_R1_JUDGE_URL
  OPEN_R1_JUDGE_MODEL
EOF
}

activate_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found in PATH." >&2
    exit 1
  fi

  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck disable=SC1090
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
}

resolve_stage() {
  local stage="$1"
  case "${stage}" in
    analysis_sft)
      TRAIN_MODULE="jobs.train.train_sft"
      TRAIN_CONFIG="${ROOT_DIR}/configs/main/analysis_sft.yaml"
      ACCELERATE_CONFIG="${ROOT_DIR}/configs/accelerate/zero3.yaml"
      ;;
    analysis_grpo)
      TRAIN_MODULE="jobs.train.train_grpo"
      TRAIN_CONFIG="${ROOT_DIR}/configs/main/analysis_grpo.yaml"
      ACCELERATE_CONFIG="${ROOT_DIR}/configs/accelerate/zero2.yaml"
      ;;
    minutes_alignment_sft)
      TRAIN_MODULE="jobs.train.train_sft"
      TRAIN_CONFIG="${ROOT_DIR}/configs/main/minutes_alignment_sft.yaml"
      ACCELERATE_CONFIG="${ROOT_DIR}/configs/accelerate/zero3.yaml"
      ;;
    decision_sft)
      TRAIN_MODULE="jobs.train.train_sft"
      TRAIN_CONFIG="${ROOT_DIR}/configs/main/decision_sft.yaml"
      ACCELERATE_CONFIG="${ROOT_DIR}/configs/accelerate/zero3.yaml"
      ;;
    decision_grpo)
      TRAIN_MODULE="jobs.train.train_grpo"
      TRAIN_CONFIG="${ROOT_DIR}/configs/main/decision_grpo.yaml"
      ACCELERATE_CONFIG="${ROOT_DIR}/configs/accelerate/zero2.yaml"
      ;;
    *)
      echo "ERROR: unsupported stage '${stage}'." >&2
      usage
      exit 1
      ;;
  esac
}

main() {
  local stage="${1:-}"
  if [[ -z "${stage}" ]]; then
    usage
    exit 1
  fi

  resolve_stage "${stage}"
  activate_conda

  mkdir -p "${LOG_DIR}"

  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

  if [[ "${stage}" == "analysis_grpo" || "${stage}" == "decision_grpo" ]]; then
    export OPEN_R1_JUDGE_URL="${OPEN_R1_JUDGE_URL:-http://localhost:11432/api/chat/}"
    export OPEN_R1_JUDGE_MODEL="${OPEN_R1_JUDGE_MODEL:-gemma3:12b}"
  fi

  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  local log_file="${LOG_DIR}/${stage}_${timestamp}.log"
  local pid_file="${LOG_DIR}/${stage}_${timestamp}.pid"

  local command=(
    accelerate launch
    --config_file "${ACCELERATE_CONFIG}"
    -m "${TRAIN_MODULE}"
    --config "${TRAIN_CONFIG}"
  )

  echo "==============================================="
  echo "Launching training stage in background"
  echo "Stage: ${stage}"
  echo "Conda env: ${ENV_NAME}"
  echo "Python: $(which python)"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
  echo "Accelerate config: ${ACCELERATE_CONFIG}"
  echo "Training config: ${TRAIN_CONFIG}"
  echo "Merge command: python -m jobs.merge_model --config ${TRAIN_CONFIG}"
  if [[ "${stage}" == "analysis_grpo" || "${stage}" == "decision_grpo" ]]; then
    echo "Judge URL: ${OPEN_R1_JUDGE_URL}"
    echo "Judge Model: ${OPEN_R1_JUDGE_MODEL}"
  fi
  echo "Log file: ${log_file}"
  echo "==============================================="

  nohup bash -lc '
    train_config="$1"
    shift
    "$@" && python -m jobs.merge_model --config "$train_config"
  ' bash "${TRAIN_CONFIG}" "${command[@]}" > "${log_file}" 2>&1 &
  local pid=$!

  echo "${pid}" > "${pid_file}"

  echo
  echo "Started PID: ${pid}"
  echo "PID file: ${pid_file}"
  echo "Monitor: tail -f ${log_file}"
}

main "$@"
