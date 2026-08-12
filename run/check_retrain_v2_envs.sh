#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_ENV="${FOMC_RETRAIN_TRAIN_ENV:-fomc_trainer}"
JUDGE_ENV="${FOMC_RETRAIN_JUDGE_ENV:-fomc_judge_v2}"
TRAIN_GPUS="${FOMC_RETRAIN_TRAIN_GPUS:-0,1}"
JUDGE_GPUS="${FOMC_RETRAIN_JUDGE_GPUS:-0}"
SMOKE_SCRIPT="${ROOT_DIR}/run/retrain_v2_env_smoke.py"
TRAIN_FREEZE="${ROOT_DIR}/requirements/retrain_v2_train.freeze.txt"
JUDGE_FREEZE="${ROOT_DIR}/requirements/retrain_v2_judge.freeze.txt"

# The two A30s are attached to different NUMA nodes (nvidia-smi topology SYS).
# Direct CUDA P2P hangs during NCCL communicator initialization on this host;
# force the proven shared-memory transport and avoid probing absent IB fabric.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

usage() {
  printf '%s\n' \
    "Usage: $0 [--train | --judge] [--skip-nccl] [--skip-gpu]" \
    "" \
    "With no role flag, both isolated environments are checked." \
    "" \
    "Environment overrides:" \
    "  FOMC_RETRAIN_TRAIN_ENV   Training environment name (default: fomc_trainer)" \
    "  FOMC_RETRAIN_JUDGE_ENV   Judge environment name (default: fomc_judge_v2)" \
    "  FOMC_RETRAIN_TRAIN_GPUS  Policy GPU IDs (default: 0,1)" \
    "  FOMC_RETRAIN_JUDGE_GPUS  Judge GPU ID (default: 0)"
}

check_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available in PATH." >&2
    exit 1
  fi
  if ! command -v timeout >/dev/null 2>&1; then
    echo "ERROR: GNU timeout is required for bounded NCCL checks." >&2
    exit 1
  fi
}

environment_exists() {
  local env_name="$1"
  conda run -n "${env_name}" python -c "import sys; raise SystemExit(0)" \
    >/dev/null 2>&1
}

check_environment() {
  local role="$1"
  local env_name="$2"
  local visible_gpus="$3"
  local skip_gpu="$4"
  local visible_gpu_count="$5"
  local freeze_path

  if [[ "${role}" == "train" ]]; then
    freeze_path="${TRAIN_FREEZE}"
  else
    freeze_path="${JUDGE_FREEZE}"
  fi

  if ! environment_exists "${env_name}"; then
    echo "ERROR: conda environment '${env_name}' does not exist." >&2
    echo "Run ./run/setup_retrain_v2_envs.sh first." >&2
    exit 1
  fi

  echo "Checking ${role} environment: ${env_name}"
  conda run --no-capture-output -n "${env_name}" python -m pip check
  if [[ "${role}" == "train" ]] && ((skip_gpu == 0)) && ((visible_gpu_count == 1)); then
    # chk2 uses one policy A30 while the other A30 hosts the judge. Reuse the
    # exact smoke module but request one visible 24 GiB-class CUDA device.
    CUDA_VISIBLE_DEVICES="${visible_gpus}" \
      conda run --no-capture-output -n "${env_name}" python -c \
      'import runpy, sys; checks = runpy.run_path(sys.argv[1]); checks["check_supplied_freeze_path"]("train", checks["Path"](sys.argv[2])); checks["check_versions"]("train"); checks["check_cuda_and_bitsandbytes"](minimum_gpu_count=1); checks["check_training_imports"](); print("retrain-v2 train environment smoke (single GPU): OK")' \
      "${SMOKE_SCRIPT}" "${freeze_path}"
  else
    local smoke_args=(--role "${role}")
    if ((skip_gpu)); then
      smoke_args+=(--skip-cuda)
    fi
    CUDA_VISIBLE_DEVICES="${visible_gpus}" \
      conda run --no-capture-output -n "${env_name}" \
      python "${SMOKE_SCRIPT}" "${smoke_args[@]}" "${freeze_path}"
  fi
}

visible_gpu_count() {
  local value="$1"
  local -a gpu_ids=()
  local gpu_id
  declare -A seen_gpu_ids=()
  if [[ ! "${value}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "ERROR: GPU list must be a comma-separated sequence of numeric IDs." >&2
    return 2
  fi
  IFS=, read -r -a gpu_ids <<< "${value}"
  if ((${#gpu_ids[@]} == 0)); then
    echo "ERROR: GPU list must contain at least one numeric ID." >&2
    return 2
  fi
  for gpu_id in "${gpu_ids[@]}"; do
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
      echo "ERROR: invalid GPU ID in visible-device list: ${gpu_id:-<empty>}" >&2
      return 2
    fi
    if [[ -n "${seen_gpu_ids[${gpu_id}]:-}" ]]; then
      echo "ERROR: duplicate GPU ID in visible-device list: ${gpu_id}" >&2
      return 2
    fi
    seen_gpu_ids["${gpu_id}"]=1
  done
  echo "${#gpu_ids[@]}"
}

run_nccl_check() {
  echo "Checking two-GPU NCCL in training environment: ${TRAIN_ENV}"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    timeout --signal=INT --kill-after=15s 60s \
    conda run --no-capture-output -n "${TRAIN_ENV}" \
    python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=2 \
    "${SMOKE_SCRIPT}" --role train --nccl-worker "${TRAIN_FREEZE}"
}

main() {
  local check_train=0
  local check_judge=0
  local skip_nccl=0
  local skip_gpu=0
  local role_selected=0

  while (($#)); do
    case "$1" in
      --train)
        check_train=1
        role_selected=1
        ;;
      --judge)
        check_judge=1
        role_selected=1
        ;;
      --skip-nccl)
        skip_nccl=1
        ;;
      --skip-gpu)
        skip_gpu=1
        skip_nccl=1
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "ERROR: unknown argument '$1'." >&2
        usage >&2
        exit 2
        ;;
    esac
    shift
  done

  if ((role_selected == 0)); then
    check_train=1
    check_judge=1
  fi

  check_conda
  if ((check_train)); then
    local train_gpu_count
    train_gpu_count="$(visible_gpu_count "${TRAIN_GPUS}")"
    if ((skip_gpu == 0)) && ((train_gpu_count > 2)); then
      echo "ERROR: retrain-v2 train checks support one policy GPU or two DDP GPUs." >&2
      exit 2
    fi
    if ((skip_nccl == 0)) && ((train_gpu_count != 2)); then
      echo "ERROR: a one-GPU policy check requires --skip-nccl." >&2
      exit 2
    fi
    check_environment train "${TRAIN_ENV}" "${TRAIN_GPUS}" "${skip_gpu}" "${train_gpu_count}"
    if ((skip_nccl == 0)); then
      run_nccl_check
    fi
  fi
  if ((check_judge)); then
    local judge_gpu_count
    judge_gpu_count="$(visible_gpu_count "${JUDGE_GPUS}")"
    if ((judge_gpu_count != 1)); then
      echo "ERROR: judge checks require exactly one visible GPU." >&2
      exit 2
    fi
    check_environment judge "${JUDGE_ENV}" "${JUDGE_GPUS}" "${skip_gpu}" "${judge_gpu_count}"
  fi

  echo "Requested retrain-v2 environment checks passed."
}

main "$@"
