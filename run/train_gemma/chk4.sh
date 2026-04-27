#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../_run_common.sh"

require_path "${ROOT_DIR}/output/training/main/merged/analysis_sft"

prepare_training_job "decision"
enable_grpo_judge
print_job_header "decision" "decision_sft train + merge, then decision_grpo train + merge"

(
  accelerate launch --config_file "${ROOT_DIR}/configs/accelerate/zero3.yaml" \
    -m jobs.train.train_sft \
    --config "${ROOT_DIR}/configs/main/decision_sft.yaml" &&
  python -m jobs.merge_model \
    --config "${ROOT_DIR}/configs/main/decision_sft.yaml" &&
  accelerate launch --config_file "${ROOT_DIR}/configs/accelerate/zero2.yaml" \
    -m jobs.train.train_grpo \
    --config "${ROOT_DIR}/configs/main/decision_grpo.yaml" &&
  python -m jobs.merge_model \
    --config "${ROOT_DIR}/configs/main/decision_grpo.yaml"
) > "${LOG_FILE}" 2>&1 &

record_background_pid "$!"
