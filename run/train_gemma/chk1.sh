#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../_run_common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ACCELERATE_CONFIG="${FOMC_TRAINER_ACCELERATE_CONFIG:-${ROOT_DIR}/configs/accelerate/zero3_sft_single.yaml}"

prepare_training_job "analysis_sft"
print_job_header "analysis_sft" "analysis_sft train + merge"

(
  accelerate launch --config_file "${ACCELERATE_CONFIG}" \
    -m jobs.train.train_sft \
    --config "${ROOT_DIR}/configs/main/analysis_sft.yaml" &&
  python -m jobs.merge_model \
    --config "${ROOT_DIR}/configs/main/analysis_sft.yaml"
) > "${LOG_FILE}" 2>&1 &

record_background_pid "$!"
