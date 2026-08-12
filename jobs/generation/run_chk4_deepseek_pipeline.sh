#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV="${FOMC_TRAIN_ENV:-fomc_trainer}"
CONCURRENCY="${CHK4_DEEPSEEK_CONCURRENCY:-8}"
REPAIR_CONCURRENCY="${CHK4_DEEPSEEK_REPAIR_CONCURRENCY:-4}"
STATE_DIR="${ROOT_DIR}/output/data/retrain_v2/chk4/pipeline_v1"
BRIEF_ROOT="${ROOT_DIR}/output/data/retrain_v2/chk4/meeting_decision_briefs_v1"
TARGET_ROOT="${ROOT_DIR}/output/data/retrain_v2/chk4/deepseek_v4_pro_v2"
TRAINING_ROOT="${ROOT_DIR}/output/data/retrain_v2/chk4/training_data_v2"

mkdir -p "${STATE_DIR}"
exec 9>"${STATE_DIR}/pipeline.lock"
if ! flock -n 9; then
  echo "ERROR: another chk4 DeepSeek pipeline owns ${STATE_DIR}/pipeline.lock" >&2
  exit 2
fi

if [[ ! "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CHK4_DEEPSEEK_CONCURRENCY must be a positive integer." >&2
  exit 2
fi
if [[ ! "${REPAIR_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CHK4_DEEPSEEK_REPAIR_CONCURRENCY must be a positive integer." >&2
  exit 2
fi

cd "${ROOT_DIR}"
if ! conda run -n "${CONDA_ENV}" python -c \
  'import os; assert os.getenv("DEEPSEEK_API_KEY", "").strip()' >/dev/null; then
  echo "ERROR: DEEPSEEK_API_KEY is not set inside conda env ${CONDA_ENV}." >&2
  exit 2
fi
CONDA_PYTHON=(conda run --no-capture-output -n "${CONDA_ENV}" python)

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] chk4 DeepSeek pipeline started"
echo "environment=${CONDA_ENV} concurrency=${CONCURRENCY}"

if [[ -s "${BRIEF_ROOT}/failures.jsonl" ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] preflight: targeted repair of failed meeting briefs"
  "${CONDA_PYTHON[@]}" -m jobs.generation.repair_chk4_meeting_briefs \
    --output-root "${BRIEF_ROOT}" \
    --concurrency "${REPAIR_CONCURRENCY}"
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage 1/4: gold-blind meeting briefs"
"${CONDA_PYTHON[@]}" -m jobs.generation.generate_chk4_meeting_briefs \
  --output-root "${BRIEF_ROOT}" \
  --concurrency "${CONCURRENCY}" \
  --resume

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage 2/4: target dry-run and leakage/token gates"
"${CONDA_PYTHON[@]}" -m jobs.generation.generate_chk4_sft_targets \
  --briefs-root "${BRIEF_ROOT}" \
  --output-root "${TARGET_ROOT}" \
  --dry-run

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage 3/4: hidden-gold reasoning acquisition"
if find "${TARGET_ROOT}/cache/rejected" -type f -print -quit 2>/dev/null | grep -q .; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage 3 repair: retry missing targets with concise reasoning contract"
  "${CONDA_PYTHON[@]}" -m jobs.generation.repair_chk4_sft_targets \
    --briefs-root "${BRIEF_ROOT}" \
    --output-root "${TARGET_ROOT}" \
    --concurrency "${REPAIR_CONCURRENCY}"
fi
if ! "${CONDA_PYTHON[@]}" -m jobs.generation.generate_chk4_sft_targets \
  --briefs-root "${BRIEF_ROOT}" \
  --output-root "${TARGET_ROOT}" \
  --concurrency "${CONCURRENCY}" \
  --resume; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage 3 repair: normal acquisition incomplete"
  "${CONDA_PYTHON[@]}" -m jobs.generation.repair_chk4_sft_targets \
    --briefs-root "${BRIEF_ROOT}" \
    --output-root "${TARGET_ROOT}" \
    --concurrency "${REPAIR_CONCURRENCY}"
fi
"${CONDA_PYTHON[@]}" -m jobs.generation.generate_chk4_sft_targets \
  --briefs-root "${BRIEF_ROOT}" \
  --output-root "${TARGET_ROOT}" \
  --concurrency "${CONCURRENCY}" \
  --resume

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage 4/4: SFT/GRPO repeat materialization"
if [[ -f "${TRAINING_ROOT}/summary.json" ]]; then
  "${CONDA_PYTHON[@]}" -m jobs.generation.materialize_chk4_training_data \
    --teacher-root "${TARGET_ROOT}" \
    --output-root "${TRAINING_ROOT}" \
    --verify-existing
else
  "${CONDA_PYTHON[@]}" -m jobs.generation.materialize_chk4_training_data \
    --teacher-root "${TARGET_ROOT}" \
    --output-root "${TRAINING_ROOT}"
fi

"${CONDA_PYTHON[@]}" - "${STATE_DIR}/complete.json" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": "complete",
    "completed_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
}
fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(name, path)
PY

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] chk4 DeepSeek pipeline complete"
