#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ENV_NAME="${FOMC_TRAINER_CONDA_ENV:-fomc_trainer}"
SOURCE_ROOT="${LEGACY_7_INDICATOR_SOURCE_ROOT:-${ROOT_DIR}/../synthetic_text/synthetic_text}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-${ROOT_DIR}/models/DeepSeek-R1-Distill-Llama-8B}"
OUTPUT_DIR="${LEGACY_7_INDICATOR_OUTPUT_DIR:-${ROOT_DIR}/output/evaluation/main/legacy_7_indicator_pilot/actual_minutes_rescore}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-1}"
MAX_TOKENS="${LEGACY_7_INDICATOR_MAX_TOKENS:-0}"
LOG_DIR="${ROOT_DIR}/logs/eval"
LAUNCH_ID="$(date +%Y%m%d_%H%M%S_%N)_$$"
LOG_FILE="${LOG_DIR}/legacy_7_indicator_pilot_${LAUNCH_ID}.log"
PID_FILE="${LOG_DIR}/legacy_7_indicator_pilot_${LAUNCH_ID}.pid"
LOCK_FILE="${OUTPUT_DIR}/.evaluation.lock"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available in PATH." >&2
  exit 1
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "ERROR: flock is required to prevent duplicate GPU jobs." >&2
  exit 1
fi
if ! command -v setsid >/dev/null 2>&1; then
  echo "ERROR: setsid is required to detach the background job." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

if [[ ! -d "${SOURCE_ROOT}" ]]; then
  echo "ERROR: legacy source root does not exist: ${SOURCE_ROOT}" >&2
  exit 1
fi
if [[ "${LEGACY_7_INDICATOR_VALIDATE_ONLY:-0}" != "1" && ! -d "${EMBEDDING_MODEL_PATH}" ]]; then
  echo "ERROR: embedding model path does not exist: ${EMBEDDING_MODEL_PATH}" >&2
  exit 1
fi
if ! [[ "${EMBEDDING_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EMBEDDING_BATCH_SIZE must be a positive integer." >&2
  exit 1
fi
if ! [[ "${MAX_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: LEGACY_7_INDICATOR_MAX_TOKENS must be a non-negative integer." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

COMMAND=(
  python -u -m jobs.eval.eval_legacy_7_indicator_pilot
  --source-root "${SOURCE_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE}"
  --max-tokens "${MAX_TOKENS}"
)

if [[ "${LEGACY_7_INDICATOR_VALIDATE_ONLY:-0}" == "1" ]]; then
  COMMAND+=(--validate-only)
else
  COMMAND+=(--embedding-model-path "${EMBEDDING_MODEL_PATH}")
fi
if [[ -n "${EMBEDDING_MODEL_SHA256:-}" ]]; then
  COMMAND+=(--embedding-model-sha256 "${EMBEDDING_MODEL_SHA256}")
fi
if [[ "${LEGACY_7_INDICATOR_ALLOW_SOURCE_HASH_MISMATCH:-0}" == "1" ]]; then
  COMMAND+=(--allow-source-hash-mismatch)
fi
if [[ "${LEGACY_7_INDICATOR_INCLUDE_TEXT:-0}" == "1" ]]; then
  COMMAND+=(--include-text)
fi

print_output_locations() {
  if [[ "${LEGACY_7_INDICATOR_VALIDATE_ONLY:-0}" == "1" ]]; then
    echo "Validation audit: ${OUTPUT_DIR}/input_validation_audit.json"
    echo "Validation status: ${OUTPUT_DIR}/input_validation_status.json"
  else
    echo "Results: ${OUTPUT_DIR}/results.md"
    echo "Run status: ${OUTPUT_DIR}/run_status.json"
  fi
}

echo "==============================================="
echo "Launching legacy seven-indicator re-score"
echo "Formula: delta = cos(full,target) - cos(masked,target)"
echo "Conda environment: ${ENV_NAME}"
echo "Python: $(command -v python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Source root: ${SOURCE_ROOT}"
echo "Embedding model: ${EMBEDDING_MODEL_PATH}"
echo "Embedding batch size: ${EMBEDDING_BATCH_SIZE}"
echo "Max tokens: ${MAX_TOKENS} (0 = no truncation)"
echo "Output directory: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "==============================================="

nohup setsid flock --no-fork --nonblock "${LOCK_FILE}" \
  "${COMMAND[@]}" >"${LOG_FILE}" 2>&1 < /dev/null &
JOB_PID=$!
echo "${JOB_PID}" >"${PID_FILE}"

sleep 1
if ! kill -0 "${JOB_PID}" 2>/dev/null; then
  if wait "${JOB_PID}"; then
    rm -f "${PID_FILE}"
    echo
    echo "Task completed before the startup check."
    echo "Log file: ${LOG_FILE}"
    print_output_locations
    exit 0
  fi
  echo "ERROR: background job exited during startup." >&2
  if [[ -s "${LOG_FILE}" ]]; then
    echo "Last log lines:" >&2
    tail -20 "${LOG_FILE}" >&2
  fi
  rm -f "${PID_FILE}"
  exit 1
fi

echo
echo "Started PID: ${JOB_PID}"
echo "PID file: ${PID_FILE}"
echo "Monitor log: tail -f '${LOG_FILE}'"
echo "Check process: ps -fp ${JOB_PID}"
print_output_locations
