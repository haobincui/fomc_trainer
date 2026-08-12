#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV="${FOMC_TRAIN_ENV:-fomc_trainer}"
OUTPUT_ROOT="${ROOT_DIR}/output/data/retrain_v2/chk4/decision_supplement_1993_2008_v1"
CORE_TEACHER_ROOT="${ROOT_DIR}/output/data/retrain_v2/chk4/deepseek_v4_pro_v2"
CONCURRENCY="${CHK4_SUPPLEMENT_CONCURRENCY:-8}"
REQUESTS_PER_SECOND="${CHK4_SUPPLEMENT_ALFRED_RPS:-2.0}"
MAX_WORKERS="${CHK4_SUPPLEMENT_ALFRED_WORKERS:-2}"
DRY_RUN=0
RESUME=0

usage() {
  echo "Usage: $0 [--dry-run | --resume] [--output-root PATH] [--core-teacher-root PATH] [--concurrency N]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --core-teacher-root)
      CORE_TEACHER_ROOT="$2"
      shift 2
      ;;
    --concurrency)
      CONCURRENCY="$2"
      shift 2
      ;;
    --requests-per-second)
      REQUESTS_PER_SECOND="$2"
      shift 2
      ;;
    --max-workers)
      MAX_WORKERS="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${DRY_RUN}" -eq 1 && "${RESUME}" -eq 1 ]]; then
  echo "ERROR: --dry-run and --resume are mutually exclusive." >&2
  exit 2
fi
if [[ ! "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: concurrency must be a positive integer." >&2
  exit 2
fi
if [[ ! "${MAX_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: max workers must be a positive integer." >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/pipeline.lock"
if ! flock -n 9; then
  echo "ERROR: another chk4 supplement pipeline owns ${OUTPUT_ROOT}/pipeline.lock" >&2
  exit 2
fi

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
  PYTHON=(python)
else
  PYTHON=(conda run --no-capture-output -n "${CONDA_ENV}" python)
fi
PREPARE=("${PYTHON[@]}" -m jobs.generation.prepare_chk4_supplement --output-root "${OUTPUT_ROOT}")
GENERATE=(
  "${PYTHON[@]}" -m jobs.generation.generate_chk4_supplement
  --output-root "${OUTPUT_ROOT}"
  --concurrency "${CONCURRENCY}"
)
BLIND_V2=(
  "${PYTHON[@]}" -m jobs.generation.run_chk4_supplement_blind_v2
  --output-root "${OUTPUT_ROOT}"
  --concurrency "${CONCURRENCY}"
)
TEACHER_V2=(
  "${PYTHON[@]}" -m jobs.generation.run_chk4_supplement_teacher_v2
  --output-root "${OUTPUT_ROOT}"
  --concurrency "${CONCURRENCY}"
)
if [[ "${RESUME}" -eq 1 ]]; then
  GENERATE+=(--resume)
  BLIND_V2+=(--resume)
  TEACHER_V2+=(--resume)
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] chk4 1993-2008 supplement pipeline started"
echo "environment=${CONDA_ENV} output_root=${OUTPUT_ROOT} concurrency=${CONCURRENCY}"
echo "pythonpath=${ROOT_DIR}/src:${ROOT_DIR}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage 0/7: label and source-contract preflight"
"${PREPARE[@]}" preflight

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] dry-run complete; ALFRED and DeepSeek requests=0"
  exit 0
fi

if ! "${PYTHON[@]}" -c \
  'import os; assert os.getenv("DEEPSEEK_API_KEY", "").strip()' >/dev/null; then
  echo "ERROR: DEEPSEEK_API_KEY is not set inside conda env ${CONDA_ENV}." >&2
  exit 2
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage 1/7: exact-D-1 ALFRED acquisition"
ACQUIRE_ARGS=(
  acquire
  --requests-per-second "${REQUESTS_PER_SECOND}"
  --max-workers "${MAX_WORKERS}"
)
if [[ "${RESUME}" -eq 1 ]]; then
  ACQUIRE_ARGS+=(--resume)
fi
"${PREPARE[@]}" "${ACQUIRE_ARGS[@]}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage 2/7: deterministic evidence compression and admission"
"${PREPARE[@]}" compress

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage 3/7: gold-blind DeepSeek meeting summaries"
"${GENERATE[@]}" summarize

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage 4/7: summary-only blind decision predictions"
"${BLIND_V2[@]}"
"${GENERATE[@]}" audit

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage 5/7: hidden-gold SFT reasoning targets"
"${TEACHER_V2[@]}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage 6/7: core plus supplement materialization"
"${GENERATE[@]}" materialize --core-teacher-root "${CORE_TEACHER_ROOT}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage 7/7: release QA"
"${GENERATE[@]}" qa

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] chk4 supplement pipeline complete"
