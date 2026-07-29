#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_run_common.sh"

if [[ "${FOMC_INPUT_CHILD_PROCESS:-0}" != "1" ]]; then
  prepare_input_job "generate_input_chk3"
  print_job_header "generate_input_chk3" "minutes_alignment data build"

  (
    export FOMC_INPUT_CHILD_PROCESS=1
    bash "${BASH_SOURCE[0]}" "$@"
  ) > "${LOG_FILE}" 2>&1 &

  record_background_pid "$!"
  exit 0
fi

print_stage_header "minutes_alignment" "build minutes_alignment prompts and dataset"

mapfile -t analysis_sft_output_dirs < <(analysis_output_dirs "analysis_sft")
for output_dir in "${analysis_sft_output_dirs[@]}"; do
  require_nonempty_dir "${output_dir}" "analysis_sft prerequisite output"
done

minutes_alignment_resume_state_file="$(mktemp)"
trap 'rm -f "${minutes_alignment_resume_state_file}"' EXIT
log_input "collecting minutes_alignment resume state before running chk3"
collect_minutes_alignment_resume_state "${minutes_alignment_resume_state_file}"
print_minutes_alignment_resume_summary "${minutes_alignment_resume_state_file}"

if ! minutes_alignment_stage_complete "${minutes_alignment_resume_state_file}" "minutes_alignment_prompts"; then
  run_pipeline_stage "minutes_alignment_prompts"
else
  log_input "skipping minutes_alignment_prompts; prompt files already match source row counts"
fi

if ! minutes_alignment_stage_complete "${minutes_alignment_resume_state_file}" "minutes_alignment_teacher_responses"; then
  run_pipeline_stage "minutes_alignment_teacher_responses"
else
  log_input "skipping minutes_alignment_teacher_responses; teacher response outputs are already complete"
fi

if ! minutes_alignment_stage_complete "${minutes_alignment_resume_state_file}" "minutes_alignment_dataset"; then
  run_pipeline_stage "minutes_alignment_dataset"
else
  log_input "skipping minutes_alignment_dataset; train outputs already match expected counts"
fi

require_nonempty_dir "$(rewrite_output_dir)" "minutes_alignment dataset output"

log_input "minutes_alignment completed successfully"
