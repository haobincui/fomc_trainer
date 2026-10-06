#!/usr/bin/env bash
# Verify the completed CHK2 DeepSeek Flash generation without storing the API key.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB_ROOT="$REPO_ROOT/output/data/retrain_v2/chk2/target_derived_minutes_deepseek_v4_flash_v1_20260828"
LOG_FILE="$JOB_ROOT/background_verify.log"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  read -r -s -p "DeepSeek API key (input is hidden): " DEEPSEEK_API_KEY
  printf '\n'
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  printf 'DEEPSEEK_API_KEY is required; verification was not started.\n' >&2
  exit 2
fi

cd "$REPO_ROOT"

export DEEPSEEK_API_KEY
nohup setsid env PYTHONUNBUFFERED=1 python -m \
  jobs.generation.generate_chk2_target_derived_minutes \
  --phase verify --concurrency 8 --preflight-rows 8 --resume \
  >> "$LOG_FILE" 2>&1 < /dev/null &
JOB_PID=$!
unset DEEPSEEK_API_KEY

printf 'Started verification PID=%s\nLog=%s\n' "$JOB_PID" "$LOG_FILE"
