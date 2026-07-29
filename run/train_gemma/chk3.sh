#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../_run_common.sh"

require_path "${ROOT_DIR}/output/training/main/merged/analysis_sft"

prepare_training_job "minutes_alignment"
print_job_header "minutes_alignment" "minutes_alignment_sft train + merge"

(
  accelerate launch --config_file "${ROOT_DIR}/configs/accelerate/zero3.yaml" \
    -m jobs.train.train_sft \
    --config "${ROOT_DIR}/configs/main/minutes_alignment_sft.yaml" &&
  python -m jobs.merge_model \
    --config "${ROOT_DIR}/configs/main/minutes_alignment_sft.yaml"
) > "${LOG_FILE}" 2>&1 &

record_background_pid "$!"
