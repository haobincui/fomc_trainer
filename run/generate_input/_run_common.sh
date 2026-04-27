#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck disable=SC1091
source "${ROOT_DIR}/run/_run_common.sh"

PIPELINE_CONFIG="${FOMC_INPUT_CONFIG:-${ROOT_DIR}/configs/main/prompt_pipeline.yaml}"
DEFAULT_SCOPE="${FOMC_INPUT_SCOPE:-after_2009}"
DEFAULT_PROFILE="${FOMC_INPUT_PROFILE:-compat}"
DEFAULT_SOURCE="${FOMC_INPUT_SOURCE:-archived}"
INPUT_LOG_DIR="${ROOT_DIR}/logs/generate_input"

log_input() {
  printf '[generate_input] %s\n' "$*"
}

fail_input() {
  log_input "ERROR: $*"
  exit 1
}

enter_input_context() {
  activate_conda
  cd "${ROOT_DIR}"
}

print_input_command() {
  printf '[generate_input]'
  printf ' %q' python -m "$1"
  shift
  printf ' %q' "$@"
  printf '\n'
}

print_shell_command() {
  printf '[generate_input]'
  printf ' %q' "$@"
  printf '\n'
}

run_input_module() {
  local module="$1"
  shift

  enter_input_context
  print_input_command "${module}" "$@"
  python -m "${module}" "$@"
}

directory_has_files() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    return 1
  fi

  local first_file
  first_file="$(find "${dir}" -mindepth 1 -maxdepth 1 -type f -print -quit)"
  [[ -n "${first_file}" ]]
}

require_existing_file() {
  local path="$1"
  local label="${2:-required file}"
  if [[ ! -f "${path}" ]]; then
    fail_input "${label} is missing: ${path}"
  fi
}

require_nonempty_dir() {
  local dir="$1"
  local label="${2:-required directory}"
  if ! directory_has_files "${dir}"; then
    fail_input "${label} is missing or empty: ${dir}"
  fi
}

config_read() {
  local dotted_key="$1"

  enter_input_context
  python - "${PIPELINE_CONFIG}" "${dotted_key}" <<'PY'
import sys

import yaml

config_path, dotted_key = sys.argv[1], sys.argv[2]
with open(config_path, encoding="utf-8") as handle:
    payload = yaml.safe_load(handle) or {}

value = payload
for part in dotted_key.split("."):
    if not isinstance(value, dict) or part not in value:
        raise SystemExit(f"ERROR: missing config key: {dotted_key}")
    value = value[part]

if isinstance(value, (dict, list)):
    raise SystemExit(f"ERROR: config key is not scalar: {dotted_key}")

print("" if value is None else str(value))
PY
}

resolve_repo_path() {
  local raw_path="$1"

  python - "${ROOT_DIR}" "${raw_path}" <<'PY'
import sys
from pathlib import Path

root_dir, raw_path = sys.argv[1], sys.argv[2]
path = Path(raw_path)
if not path.is_absolute():
    path = Path(root_dir) / path
print(path.resolve())
PY
}

config_path() {
  local dotted_key="$1"
  resolve_repo_path "$(config_read "${dotted_key}")"
}

print_stage_header() {
  local stage_name="$1"
  local description="$2"
  log_input "==============================================="
  log_input "${stage_name}: ${description}"
  log_input "config=${PIPELINE_CONFIG}"
  log_input "scope=${DEFAULT_SCOPE} profile=${DEFAULT_PROFILE}"
  log_input "==============================================="
}

analysis_output_dirs() {
  local dataset_name="$1"
  local train_root
  train_root="$(config_path "analysis.train_roots.${dataset_name}")"

  case "${DEFAULT_PROFILE}" in
    compat|strict)
      printf '%s\n' "${train_root}"
      ;;
    both)
      printf '%s\n' "${train_root}"
      printf '%s\n' "${train_root}_strict"
      ;;
    *)
      fail_input "unsupported profile for analysis outputs: ${DEFAULT_PROFILE}"
      ;;
  esac
}

prepare_input_job() {
  local job_name="$1"

  activate_conda
  mkdir -p "${INPUT_LOG_DIR}"

  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${INPUT_LOG_DIR}/${job_name}_${TIMESTAMP}.log"
  PID_FILE="${INPUT_LOG_DIR}/${job_name}_${TIMESTAMP}.pid"
}

labeled_output_dir() {
  printf '%s\n' "$(config_path "labeling.output_root")/${DEFAULT_SCOPE}"
}

merged_labeled_output_file() {
  printf '%s\n' "$(config_path "analysis.labeled_paths.${DEFAULT_SCOPE}")"
}

qa_master_output_file() {
  printf '%s\n' "$(config_path "analysis.master_path")"
}

analysis_pipeline_root() {
  printf '%s\n' "$(config_path "analysis.pipeline_root")"
}

rewrite_output_dir() {
  printf '%s\n' "$(config_path "rewrite.train_root")"
}

decision_output_dir() {
  local dataset_name="$1"
  printf '%s\n' "$(config_path "decision.train_roots.${dataset_name}")"
}

run_build_qa_master() {
  local export_root
  export_root="${FOMC_INPUT_EXPORT_ROOT:-$(analysis_pipeline_root)}"

  run_input_module \
    "process_fomc_report.build_qa_master" \
    --config "${PIPELINE_CONFIG}" \
    --source "${DEFAULT_SOURCE}" \
    --scope "${DEFAULT_SCOPE}" \
    --export-root "${export_root}" \
    "$@"
}

run_pipeline_stage() {
  local stage="$1"
  shift

  local command_args=(
    --config "${PIPELINE_CONFIG}"
    --scope "${DEFAULT_SCOPE}"
    --profile "${DEFAULT_PROFILE}"
    --stage "${stage}"
  )

  if [[ -n "${FOMC_INPUT_TEACHER_SOURCE:-}" ]]; then
    command_args+=(--teacher-source "${FOMC_INPUT_TEACHER_SOURCE}")
  fi

  run_input_module \
    "process_fomc_report.generate_prompt_and_response.run_generate_prompt_pipeline" \
    "${command_args[@]}" \
    "$@"
}

collect_analysis_sft_resume_state() {
  local output_file="$1"

  enter_input_context
  python -m process_fomc_report.generate_prompt_and_response.algo.inspect_analysis_sft_resume_state \
    --config "${PIPELINE_CONFIG}" \
    --scope "${DEFAULT_SCOPE}" \
    --profile "${DEFAULT_PROFILE}" > "${output_file}"
}

print_analysis_sft_resume_summary() {
  local state_file="$1"

  python - "${state_file}" <<'PY'
import json
import sys

stage_order = [
    "build_qa_master",
    "normalize_input_sources",
    "merge_labels",
    "analysis_sft_teacher_prompts",
    "analysis_sft_teacher_responses",
    "analysis_sft_prompts",
    "analysis_sft_dataset",
]

state_path = sys.argv[1]
with open(state_path, encoding="utf-8") as handle:
    payload = json.load(handle)

for stage in stage_order:
    item = payload["stages"][stage]
    action = "skip" if item["complete"] else "rebuild"
    expected = json.dumps(item["expected_counts"], ensure_ascii=False, sort_keys=True)
    actual = json.dumps(item["actual_counts"], ensure_ascii=False, sort_keys=True)
    print(
        f"[generate_input] resume_check stage={stage} action={action} "
        f"reason={item['reason']} expected={expected} actual={actual}"
    )
PY
}

analysis_sft_stage_complete() {
  local state_file="$1"
  local stage_name="$2"

  python - "${state_file}" "${stage_name}" <<'PY'
import json
import sys

state_path, stage_name = sys.argv[1], sys.argv[2]
with open(state_path, encoding="utf-8") as handle:
    payload = json.load(handle)

raise SystemExit(0 if payload["stages"][stage_name]["complete"] else 1)
PY
}

collect_decision_resume_state() {
  local output_file="$1"

  enter_input_context
  python -m process_fomc_report.generate_prompt_and_response.algo.inspect_decision_resume_state \
    --config "${PIPELINE_CONFIG}" \
    --scope "${DEFAULT_SCOPE}" > "${output_file}"
}

print_decision_resume_summary() {
  local state_file="$1"

  python - "${state_file}" <<'PY'
import json
import sys

stage_order = [
    "decision_prompts",
    "decision_dataset",
]

state_path = sys.argv[1]
with open(state_path, encoding="utf-8") as handle:
    payload = json.load(handle)

for stage in stage_order:
    item = payload["stages"][stage]
    action = "skip" if item["complete"] else "rebuild"
    expected = json.dumps(item["expected_counts"], ensure_ascii=False, sort_keys=True)
    actual = json.dumps(item["actual_counts"], ensure_ascii=False, sort_keys=True)
    print(
        f"[generate_input] resume_check stage={stage} action={action} "
        f"reason={item['reason']} expected={expected} actual={actual}"
    )
PY
}

decision_stage_complete() {
  local state_file="$1"
  local stage_name="$2"

  python - "${state_file}" "${stage_name}" <<'PY'
import json
import sys

state_path, stage_name = sys.argv[1], sys.argv[2]
with open(state_path, encoding="utf-8") as handle:
    payload = json.load(handle)

raise SystemExit(0 if payload["stages"][stage_name]["complete"] else 1)
PY
}

collect_minutes_alignment_resume_state() {
  local output_file="$1"

  enter_input_context
  python -m process_fomc_report.generate_prompt_and_response.algo.inspect_minutes_alignment_resume_state \
    --config "${PIPELINE_CONFIG}" \
    --scope "${DEFAULT_SCOPE}" > "${output_file}"
}

print_minutes_alignment_resume_summary() {
  local state_file="$1"

  python - "${state_file}" <<'PY'
import json
import sys

stage_order = [
    "minutes_alignment_prompts",
    "minutes_alignment_teacher_responses",
    "minutes_alignment_dataset",
]

state_path = sys.argv[1]
with open(state_path, encoding="utf-8") as handle:
    payload = json.load(handle)

for stage in stage_order:
    item = payload["stages"][stage]
    action = "skip" if item["complete"] else "rebuild"
    expected = json.dumps(item["expected_counts"], ensure_ascii=False, sort_keys=True)
    actual = json.dumps(item["actual_counts"], ensure_ascii=False, sort_keys=True)
    print(
        f"[generate_input] resume_check stage={stage} action={action} "
        f"reason={item['reason']} expected={expected} actual={actual}"
    )
PY
}

minutes_alignment_stage_complete() {
  local state_file="$1"
  local stage_name="$2"

  python - "${state_file}" "${stage_name}" <<'PY'
import json
import sys

state_path, stage_name = sys.argv[1], sys.argv[2]
with open(state_path, encoding="utf-8") as handle:
    payload = json.load(handle)

raise SystemExit(0 if payload["stages"][stage_name]["complete"] else 1)
PY
}
