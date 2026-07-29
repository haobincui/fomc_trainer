#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_run_common.sh"

FORCE_TEACHER_REFRESH=0

parse_chk1_args() {
  local arg
  for arg in "$@"; do
    case "${arg}" in
      --force-teacher-refresh=0)
        FORCE_TEACHER_REFRESH=0
        ;;
      --force-teacher-refresh=1)
        FORCE_TEACHER_REFRESH=1
        ;;
      *)
        fail_input "unsupported chk1.sh argument: ${arg}"
        ;;
    esac
  done
}

parse_chk1_args "$@"

if [[ "${FOMC_INPUT_CHILD_PROCESS:-0}" != "1" ]]; then
  prepare_input_job "generate_input_chk1"
  print_job_header "generate_input_chk1" "analysis_sft data build"

  (
    export FOMC_INPUT_CHILD_PROCESS=1
    bash "${BASH_SOURCE[0]}" "$@"
  ) > "${LOG_FILE}" 2>&1 &

  record_background_pid "$!"
  exit 0
fi

print_stage_header "analysis_sft" "build qa master, labeling artifacts, and analysis_sft dataset"

analysis_resume_state_file=""
if [[ "${FORCE_TEACHER_REFRESH}" == "0" ]]; then
  analysis_resume_state_file="$(mktemp)"
  trap 'rm -f "${analysis_resume_state_file}"' EXIT
  log_input "collecting analysis_sft resume state before running chk1"
  collect_analysis_sft_resume_state "${analysis_resume_state_file}"
  print_analysis_sft_resume_summary "${analysis_resume_state_file}"
else
  log_input "force-teacher-refresh=1; disabling chk1 resume/skip checks and running full chain"
fi

if [[ "${FORCE_TEACHER_REFRESH}" == "1" ]] || ! analysis_sft_stage_complete "${analysis_resume_state_file}" "build_qa_master"; then
  run_build_qa_master
else
  log_input "skipping build_qa_master; existing outputs passed resume validation"
fi
require_existing_file "$(qa_master_output_file)" "qa master output"

if [[ "${FORCE_TEACHER_REFRESH}" == "1" ]] || ! analysis_sft_stage_complete "${analysis_resume_state_file}" "normalize_input_sources"; then
  run_pipeline_stage "normalize_input_sources"
else
  log_input "skipping normalize_input_sources; existing audit marker matches current sources"
fi

labeled_dir="$(labeled_output_dir)"
merged_labeled_file="$(merged_labeled_output_file)"
raw_merged_labeled="${ROOT_DIR}/dataset/raw_data/labeled_text/merged_labeled_${DEFAULT_SCOPE}.xlsx"

if [[ "${FORCE_TEACHER_REFRESH}" == "1" ]] || ! analysis_sft_stage_complete "${analysis_resume_state_file}" "merge_labels"; then
  if directory_has_files "${labeled_dir}"; then
    run_pipeline_stage "merge_labels"
  elif [[ -f "${raw_merged_labeled}" ]]; then
    mkdir -p "$(dirname "${merged_labeled_file}")"
    print_shell_command cp "${raw_merged_labeled}" "${merged_labeled_file}"
    cp "${raw_merged_labeled}" "${merged_labeled_file}"
  else
    log_input "labeled minutes are missing at ${labeled_dir}; running label_html stage"
    run_pipeline_stage "label_html"
    require_nonempty_dir "${labeled_dir}" "labeled minutes directory"
    run_pipeline_stage "merge_labels"
  fi
else
  log_input "skipping merge_labels; merged labeled file already has the expected row count"
fi
require_existing_file "${merged_labeled_file}" "merged labeled minutes file"

if [[ "${FORCE_TEACHER_REFRESH}" == "1" ]] || ! analysis_sft_stage_complete "${analysis_resume_state_file}" "analysis_sft_teacher_prompts"; then
  run_pipeline_stage "analysis_sft_teacher_prompts"
else
  log_input "skipping analysis_sft_teacher_prompts; prompt files already match manifest row counts"
fi

if [[ "${FORCE_TEACHER_REFRESH}" == "1" ]] || ! analysis_sft_stage_complete "${analysis_resume_state_file}" "analysis_sft_teacher_responses"; then
  log_input "running teacher model for reasoning + analysis"
  run_pipeline_stage "analysis_sft_teacher_responses" --teacher-source llm "--force-teacher-refresh=${FORCE_TEACHER_REFRESH}"
else
  log_input "skipping analysis_sft_teacher_responses; teacher response outputs are already complete"
fi

if [[ "${FORCE_TEACHER_REFRESH}" == "1" ]] || ! analysis_sft_stage_complete "${analysis_resume_state_file}" "analysis_sft_prompts"; then
  run_pipeline_stage "analysis_sft_prompts"
else
  log_input "skipping analysis_sft_prompts; student prompt files already match manifest row counts"
fi

if [[ "${FORCE_TEACHER_REFRESH}" == "1" ]] || ! analysis_sft_stage_complete "${analysis_resume_state_file}" "analysis_sft_dataset"; then
  run_pipeline_stage "analysis_sft_dataset"
else
  log_input "skipping analysis_sft_dataset; train outputs already match expected counts"
fi

mapfile -t analysis_sft_output_dirs < <(analysis_output_dirs "analysis_sft")
for output_dir in "${analysis_sft_output_dirs[@]}"; do
  require_nonempty_dir "${output_dir}" "analysis_sft dataset output"
done

log_input "analysis_sft completed successfully"
