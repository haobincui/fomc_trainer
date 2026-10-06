#!/usr/bin/env bash
# Build and run the bounded CHK2 DeepSeek Flash recovery cohorts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$REPO_ROOT/run/recover_chk2_target_derived_minutes.sh"
CONDA_BIN="/usr/local/anaconda3/bin/conda"
CONDA_ENV="fomc_trainer"
PLAN_ROOT="$REPO_ROOT/output/data/retrain_v2/chk2/target_derived_minutes_deepseek_v4_flash_recovery_plan_v1_20260829"
REGENERATE_ROOT="$REPO_ROOT/output/data/retrain_v2/chk2/target_derived_minutes_deepseek_v4_flash_recovery_regenerate_v1_20260829"
REVERIFY_ROOT="$REPO_ROOT/output/data/retrain_v2/chk2/target_derived_minutes_deepseek_v4_flash_recovery_reverify_v1_20260829"
REGENERATE_SOURCE="$REGENERATE_ROOT/inputs/minutes_alignment"
REVERIFY_SOURCE="$REVERIFY_ROOT/inputs/minutes_alignment"
LOG_FILE="$PLAN_ROOT/background_recovery.log"
PID_FILE="$PLAN_ROOT/recovery.pid"
LOCK_FILE="$PLAN_ROOT/recovery.lock"

run_in_env() {
  "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" "$@"
}

worker() {
  mkdir -p "$PLAN_ROOT"
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    printf 'Another CHK2 recovery worker already holds %s\n' "$LOCK_FILE" >&2
    return 3
  fi
  printf '%s\n' "$$" > "$PID_FILE"
  trap 'rm -f "$PID_FILE"' EXIT
  cd "$REPO_ROOT"

  run_in_env python -c 'import os,sys; ok=bool(os.getenv("DEEPSEEK_API_KEY", "").strip()); print("DEEPSEEK_API_KEY=" + ("set" if ok else "missing")); raise SystemExit(0 if ok else 1)'

  printf '[recovery] stage=prepare started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  run_in_env python -m jobs.generation.prepare_chk2_target_derived_recovery

  reverify_status=0
  printf '[recovery] stage=reverify_only rows=53 started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  run_in_env env PYTHONUNBUFFERED=1 python -m \
    jobs.generation.generate_chk2_target_derived_minutes \
    --source-root "$REVERIFY_SOURCE" \
    --output-root "$REVERIFY_ROOT" \
    --phase verify --concurrency 8 --preflight-rows 8 --resume \
    || reverify_status=$?
  printf '[recovery] stage=reverify_only exit=%s finished_at=%s\n' \
    "$reverify_status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ "$reverify_status" -ne 0 ]]; then
    canary_health="$({
      run_in_env python -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    verification = (json.load(handle).get("verification") or {})
usage = verification.get("usage") or {}
rows = int(verification.get("rows") or 0)
successful = int(usage.get("successful_requests") or 0)
transport_failures = len(verification.get("transport_failures") or [])
quality_only = rows > 0 and successful == rows and transport_failures == 0
print("quality_only" if quality_only else "transport_or_provider")
' "$REVERIFY_ROOT/preflight.json"
    } 2>/dev/null)"
    if [[ "$canary_health" == "quality_only" ]]; then
      printf '[recovery] status=reverify_canary_quality_failed; full_reverify_skipped; continuing_to_regeneration\n'
    else
      printf '[recovery] status=stopped_after_provider_or_transport_failure; regeneration_not_started\n' >&2
      return 1
    fi
  fi

  regenerate_status=0
  printf '[recovery] stage=regenerate rows=1531 started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  run_in_env env PYTHONUNBUFFERED=1 python -m \
    jobs.generation.generate_chk2_target_derived_minutes \
    --source-root "$REGENERATE_SOURCE" \
    --output-root "$REGENERATE_ROOT" \
    --acquire-first --concurrency 8 --resume \
    || regenerate_status=$?
  printf '[recovery] stage=regenerate exit=%s finished_at=%s\n' \
    "$regenerate_status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  verify_status=0
  if [[ "$regenerate_status" -eq 0 ]]; then
    printf '[recovery] stage=verify_regenerated started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    run_in_env env PYTHONUNBUFFERED=1 python -m \
      jobs.generation.generate_chk2_target_derived_minutes \
      --source-root "$REGENERATE_SOURCE" \
      --output-root "$REGENERATE_ROOT" \
      --phase verify --concurrency 8 --preflight-rows 8 --resume \
      || verify_status=$?
    printf '[recovery] stage=verify_regenerated exit=%s finished_at=%s\n' \
      "$verify_status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  else
    verify_status=4
    printf '[recovery] stage=verify_regenerated skipped=regeneration_failed\n'
  fi

  if [[ "$reverify_status" -ne 0 || "$regenerate_status" -ne 0 || "$verify_status" -ne 0 ]]; then
    printf '[recovery] status=incomplete reverify=%s regenerate=%s verify=%s\n' \
      "$reverify_status" "$regenerate_status" "$verify_status" >&2
    return 1
  fi
  printf '[recovery] status=machine_screen_complete finished_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

if [[ "${1:-}" == "--worker" ]]; then
  worker
  exit $?
fi

mkdir -p "$PLAN_ROOT"
if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(tr -cd '0-9' < "$PID_FILE")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    printf 'Recovery already running PID=%s\nLog=%s\n' "$existing_pid" "$LOG_FILE"
    exit 3
  fi
fi

if pgrep -af '[p]ython(3)? -m jobs\.generation\.generate_chk2_target_derived_minutes' >/dev/null; then
  printf 'A CHK2 generation/verification process is already running; recovery was not started.\n' >&2
  pgrep -af '[p]ython(3)? -m jobs\.generation\.generate_chk2_target_derived_minutes' >&2 || true
  exit 3
fi

nohup setsid "$SCRIPT_PATH" --worker >> "$LOG_FILE" 2>&1 < /dev/null &
JOB_PID=$!
sleep 1
if ! kill -0 "$JOB_PID" 2>/dev/null; then
  printf 'Recovery worker exited during startup. Log=%s\n' "$LOG_FILE" >&2
  tail -n 40 "$LOG_FILE" >&2 || true
  exit 2
fi
printf 'Started recovery PID=%s\nLog=%s\n' "$JOB_PID" "$LOG_FILE"
