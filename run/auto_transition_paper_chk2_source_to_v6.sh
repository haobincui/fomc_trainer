#!/usr/bin/env bash
# Wait for the v5 source-only stage, seal it, run v6 downstream, then publish.
# Intended to run in one detached tmux session with DEEPSEEK_API_KEY inherited.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="$REPO_ROOT/output/data/retrain_v2/chk2/chk1_final_analysis_to_minutes_flash_official_reference_v2_20260831"
HANDOFF_ROOT="$REPO_ROOT/output/data/retrain_v2/chk2/chk1_final_analysis_source_audit_handoff_v1_20260831"
DOWNSTREAM_ROOT="$REPO_ROOT/output/data/retrain_v2/chk2/chk1_final_analysis_to_minutes_flash_official_reference_v6_downstream128_20260831"
RELEASE_ROOT="$REPO_ROOT/dataset/processed/retrain_v2/chk2_chk1_final_analysis_synthetic_minutes_flash_official_reference_v6_downstream128_20260831"
TRANSITION_LOG="$DOWNSTREAM_ROOT/background_transition.log"
DOWNSTREAM_LOG="$DOWNSTREAM_ROOT/background_all.log"
TRANSITION_LOCK="$DOWNSTREAM_ROOT/transition.lock"
DOWNSTREAM_LOCK="$DOWNSTREAM_ROOT/build.lock"
SOURCE_LOCK="$SOURCE_ROOT/build.lock"
CURRENT_STAGE="initialization"

on_error() {
  local exit_code=$?
  trap - ERR
  unset DEEPSEEK_API_KEY
  printf 'Transition FAILED UTC=%s Stage=%s Exit=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CURRENT_STAGE" "$exit_code" \
    >> "$TRANSITION_LOG"
  exit "$exit_code"
}
trap on_error ERR

if [[ "${CONDA_DEFAULT_ENV:-}" != "fomc_trainer" ]]; then
  printf 'Activate the fomc_trainer conda environment before starting this transition.\n' >&2
  exit 2
fi
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  printf 'DEEPSEEK_API_KEY is required for unattended execution.\n' >&2
  exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
  printf 'flock is required for the source/downstream stage boundary.\n' >&2
  exit 2
fi

mkdir -p "$DOWNSTREAM_ROOT"
exec 9>"$TRANSITION_LOCK"
if ! flock -n 9; then
  printf 'A source-to-v6 transition already holds %s.\n' "$TRANSITION_LOCK" >&2
  exit 3
fi

cd "$REPO_ROOT"
printf 'Transition watcher started UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$TRANSITION_LOG"

CURRENT_STAGE="wait_for_source_receipt"
while [[ ! -f "$SOURCE_ROOT/source_admission_receipt.json" ]]; do
  # The source producer owns this exact root's lock for its full lifetime.
  # Acquiring it before a receipt exists therefore proves that producer exited
  # without completing, without relying on a process-name match.
  exec 8>"$SOURCE_LOCK"
  if flock -n 8; then
    printf 'Source stage exited without a receipt UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$TRANSITION_LOG"
    exit 4
  fi
  exec 8>&-
  sleep 30
done

# The receipt is written just before the v5 process returns.  Acquiring its
# lock guarantees that no producer is still mutating the source acquisition.
exec 8>"$SOURCE_LOCK"
while ! flock -n 8; do
  sleep 2
done

CURRENT_STAGE="seal_source_handoff"
printf 'Source receipt observed; sealing handoff UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$TRANSITION_LOG"
python -m jobs.generation.seal_paper_chk2_source_handoff_v1 \
  --acquisition-root "$SOURCE_ROOT" \
  --handoff-root "$HANDOFF_ROOT" \
  >> "$TRANSITION_LOG" 2>&1
python -m jobs.generation.seal_paper_chk2_source_handoff_v1 \
  --handoff-root "$HANDOFF_ROOT" --verify-only \
  >> "$TRANSITION_LOG" 2>&1
exec 8>&-

CURRENT_STAGE="acquire_downstream_lock"
exec 7>"$DOWNSTREAM_LOCK"
if ! flock -n 7; then
  printf 'A v6 downstream process already holds %s.\n' "$DOWNSTREAM_LOCK" >> "$TRANSITION_LOG"
  exit 5
fi

CURRENT_STAGE="run_v6_downstream"
printf 'Starting v6 downstream UTC=%s Concurrency=128\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$TRANSITION_LOG"
env PYTHONUNBUFFERED=1 python -m jobs.generation.generate_paper_chk2_downstream_v6 \
  --handoff-root "$HANDOFF_ROOT" \
  --output-root "$DOWNSTREAM_ROOT" \
  --phase all --concurrency 128 --resume \
  >> "$DOWNSTREAM_LOG" 2>&1

unset DEEPSEEK_API_KEY
CURRENT_STAGE="publish_release"
printf 'Downstream complete; publishing UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$TRANSITION_LOG"
python -m jobs.generation.publish_paper_chk2_downstream_v6 \
  --source-acquisition-root "$SOURCE_ROOT" \
  --source-handoff-root "$HANDOFF_ROOT" \
  --downstream-root "$DOWNSTREAM_ROOT" \
  --release-root "$RELEASE_ROOT" \
  >> "$TRANSITION_LOG" 2>&1
printf 'Release complete UTC=%s Path=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RELEASE_ROOT" >> "$TRANSITION_LOG"
CURRENT_STAGE="complete"
trap - ERR
