#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_run_common.sh"

print_stage_header "analysis_grpo" "build analysis_grpo prompts and dataset"

mapfile -t analysis_sft_output_dirs < <(analysis_output_dirs "analysis_sft")
for output_dir in "${analysis_sft_output_dirs[@]}"; do
  require_nonempty_dir "${output_dir}" "analysis_sft prerequisite output"
done

run_pipeline_stage "analysis_grpo_prompts"
run_pipeline_stage "analysis_grpo_dataset"

mapfile -t analysis_grpo_output_dirs < <(analysis_output_dirs "analysis_grpo")
for output_dir in "${analysis_grpo_output_dirs[@]}"; do
  require_nonempty_dir "${output_dir}" "analysis_grpo dataset output"
done

log_input "analysis_grpo completed successfully"
