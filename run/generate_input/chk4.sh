#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_run_common.sh"

print_stage_header "decision" "build decision datasets"

decision_resume_state_file="$(mktemp)"
trap 'rm -f "${decision_resume_state_file}"' EXIT
log_input "collecting decision resume state before running chk4"
collect_decision_resume_state "${decision_resume_state_file}"
print_decision_resume_summary "${decision_resume_state_file}"

if ! decision_stage_complete "${decision_resume_state_file}" "decision_prompts"; then
  run_pipeline_stage "decision_prompts"
else
  log_input "skipping decision_prompts; prompt files already match source row counts"
fi

if ! decision_stage_complete "${decision_resume_state_file}" "decision_dataset"; then
  run_pipeline_stage "decision_dataset"
else
  log_input "skipping decision_dataset; train outputs already match expected counts"
fi

mapfile -t analysis_sft_output_dirs < <(analysis_output_dirs "analysis_sft")
for output_dir in "${analysis_sft_output_dirs[@]}"; do
  require_nonempty_dir "${output_dir}" "analysis_sft prerequisite output"
done

mapfile -t analysis_grpo_output_dirs < <(analysis_output_dirs "analysis_grpo")
for output_dir in "${analysis_grpo_output_dirs[@]}"; do
  require_nonempty_dir "${output_dir}" "analysis_grpo prerequisite output"
done

require_nonempty_dir "$(rewrite_output_dir)" "minutes_alignment prerequisite output"
require_nonempty_dir "$(decision_output_dir "decision_sft")" "decision_sft output"
require_nonempty_dir "$(decision_output_dir "decision_grpo")" "decision_grpo output"

log_input "decision datasets completed successfully"
