#!/usr/bin/env bash
# Build/resume the sealed-source -> synthetic Minutes v6 downstream acquisition.
# The provider credential is inherited by the detached child and never written.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HANDOFF_ROOT="$REPO_ROOT/output/data/retrain_v2/chk2/chk1_final_analysis_source_audit_handoff_v1_20260831"
JOB_ROOT="$REPO_ROOT/output/data/retrain_v2/chk2/chk1_final_analysis_to_minutes_flash_official_reference_v6_downstream128_20260831"
LOG_FILE="$JOB_ROOT/background_all.log"
LOCK_FILE="$JOB_ROOT/build.lock"

if [[ "${CONDA_DEFAULT_ENV:-}" != "fomc_trainer" ]]; then
  printf 'Activate the fomc_trainer conda environment before starting this job.\n' >&2
  exit 2
fi

if ! command -v flock >/dev/null 2>&1; then
  printf 'flock is required for single-instance API acquisition.\n' >&2
  exit 2
fi

if [[ ! -f "$HANDOFF_ROOT/handoff_manifest.json" ]]; then
  printf 'The sealed source handoff is not available: %s\n' "$HANDOFF_ROOT" >&2
  exit 2
fi

mkdir -p "$JOB_ROOT"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf 'A v6 downstream process already holds %s; no duplicate was started.\n' "$LOCK_FILE" >&2
  exit 3
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  if [[ ! -t 0 ]]; then
    printf 'DEEPSEEK_API_KEY is required for unattended execution.\n' >&2
    exit 2
  fi
  read -r -s -p "DeepSeek API key (input is hidden): " DEEPSEEK_API_KEY
  printf '\n'
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  printf 'DEEPSEEK_API_KEY is required; no task was started.\n' >&2
  exit 2
fi

cd "$REPO_ROOT"
export DEEPSEEK_API_KEY
nohup setsid env PYTHONUNBUFFERED=1 python -m \
  jobs.generation.generate_paper_chk2_downstream_v6 \
  --phase all --concurrency 128 --resume \
  >> "$LOG_FILE" 2>&1 < /dev/null &
JOB_PID=$!
unset DEEPSEEK_API_KEY
exec 9>&-

printf 'Started paper chk-2 v6 downstream PID=%s\nEnvironment=fomc_trainer\nConcurrency=128\nLog=%s\n' \
  "$JOB_PID" "$LOG_FILE"
