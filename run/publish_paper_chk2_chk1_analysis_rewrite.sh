#!/usr/bin/env bash
# Publish only after every one of the 1,743 acquisition rows is terminal.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != "fomc_trainer" ]]; then
  printf 'Activate the fomc_trainer conda environment before publishing.\n' >&2
  exit 2
fi
cd "$REPO_ROOT"

ACQUISITION_ROOT="$REPO_ROOT/output/data/retrain_v2/chk2/chk1_final_analysis_to_minutes_flash_official_reference_v2_20260831"
RELEASE_ROOT="$REPO_ROOT/dataset/processed/retrain_v2/chk2_chk1_final_analysis_synthetic_minutes_flash_official_reference_v2_20260831"

python -m jobs.generation.publish_paper_chk2_chk1_analysis_rewrite \
  --acquisition-root "$ACQUISITION_ROOT" \
  --release-root "$RELEASE_ROOT" \
  "$@"
