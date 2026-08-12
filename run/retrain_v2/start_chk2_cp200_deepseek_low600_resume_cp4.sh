#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUMMARY_DIR="${ROOT_DIR}/docs/summary/20260810T123000Z/chk1_cp200_deepseek_chk2"
BINDING="${SUMMARY_DIR}/chk2_low600_resume_cp4_launch_binding.json"
RECEIPT="${SUMMARY_DIR}/chk2_low600_resume_cp4_launch_receipt.json"

cd "${ROOT_DIR}"
exec conda run --no-capture-output -n fomc_trainer \
  python -u -m jobs.retrain_v2.start_chk2_deepseek_low600_resume \
  --repo-root "${ROOT_DIR}" \
  --binding "${BINDING}" \
  --receipt "${RECEIPT}" \
  --execute
