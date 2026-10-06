#!/usr/bin/env bash
# Build/resume the chk1-final-analysis -> synthetic Minutes acquisition.
# The DeepSeek credential is inherited by the detached child and never written.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB_ROOT="$REPO_ROOT/output/data/retrain_v2/chk2/chk1_final_analysis_to_minutes_flash_official_reference_v2_20260831"
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

mkdir -p "$JOB_ROOT"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf 'A paper chk-2 data-build process already holds %s; no duplicate was started.\n' "$LOCK_FILE" >&2
  exit 3
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
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
  jobs.generation.generate_paper_chk2_chk1_analysis_rewrite \
  --phase all --concurrency 8 --preflight-rows 8 --resume \
  >> "$LOG_FILE" 2>&1 < /dev/null &
JOB_PID=$!
unset DEEPSEEK_API_KEY
exec 9>&-

printf 'Started paper chk-2 data build PID=%s\nEnvironment=fomc_trainer\nLog=%s\n' "$JOB_PID" "$LOG_FILE"
