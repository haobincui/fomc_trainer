#!/usr/bin/env bash
# Seal the partial v6 acquisition, resume it in an independent recovery root,
# and preserve the original v6 evidence tree byte-for-byte.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HANDOFF_ROOT="$REPO_ROOT/output/data/retrain_v2/chk2/chk1_final_analysis_source_audit_handoff_v1_20260831"
ORIGINAL_ROOT="$REPO_ROOT/output/data/retrain_v2/chk2/chk1_final_analysis_to_minutes_flash_official_reference_v6_downstream128_20260831"
RECOVERY_ROOT="$REPO_ROOT/output/data/retrain_v2/chk2/chk1_final_analysis_to_minutes_flash_official_reference_v6_downstream128_recovery_v1_20260831"
RECOVERY_LOG="${RECOVERY_ROOT}.log"
RECOVERY_LOCK="${RECOVERY_ROOT}.lock"
ORIGINAL_LOCK="$ORIGINAL_ROOT/build.lock"

if [[ "${CONDA_DEFAULT_ENV:-}" != "fomc_trainer" ]]; then
  printf 'Activate the fomc_trainer conda environment before starting this job.\n' >&2
  exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
  printf 'flock is required for single-instance recovery.\n' >&2
  exit 2
fi
if [[ ! -f "$HANDOFF_ROOT/handoff_manifest.json" ]]; then
  printf 'The sealed source handoff is unavailable: %s\n' "$HANDOFF_ROOT" >&2
  exit 2
fi
if [[ ! -f "$ORIGINAL_ROOT/prompt_contract.json" ]]; then
  printf 'The partial v6 acquisition is unavailable: %s\n' "$ORIGINAL_ROOT" >&2
  exit 2
fi

exec 9>"$RECOVERY_LOCK"
if ! flock -n 9; then
  printf 'A recovery process already holds %s; no duplicate was started.\n' \
    "$RECOVERY_LOCK" >&2
  exit 3
fi

# Holding the legacy producer lock prevents an old v6 launcher from changing
# the evidence tree while the recovery snapshot is sealed and verified.
exec 8>>"$ORIGINAL_LOCK"
if ! flock -n 8; then
  printf 'The original v6 acquisition is active; recovery was not started.\n' >&2
  exit 3
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  if [[ ! -t 0 ]]; then
    printf 'DEEPSEEK_API_KEY is required for unattended recovery.\n' >&2
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
  jobs.generation.recover_paper_chk2_downstream_v6 \
  --phase all \
  --original-root "$ORIGINAL_ROOT" \
  --recovery-root "$RECOVERY_ROOT" \
  --handoff-root "$HANDOFF_ROOT" \
  --concurrency 128 --resume \
  >> "$RECOVERY_LOG" 2>&1 < /dev/null &
JOB_PID=$!
unset DEEPSEEK_API_KEY

printf 'Started paper chk-2 v6 recovery PID=%s\nEnvironment=fomc_trainer\nConcurrency=128\nRecovery root=%s\nLog=%s\n' \
  "$JOB_PID" "$RECOVERY_ROOT" "$RECOVERY_LOG"
