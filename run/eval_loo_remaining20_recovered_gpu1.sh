#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${ROOT_DIR}/output/evaluation/main/canonical_loo/pilot_eval_13/pilot-remaining20-recovered-v1-20260730T100713Z"
SCORE_DIR="${RUN_ROOT}/scoring/actual_minutes_deepseek8b"
REFERENCE_FILE="${SCORE_DIR}/pilot_eval_13_actual_minutes_reference.jsonl"
REFERENCE_SHA256="0a79a75a55f7eea3f6312dabc7f9a3b07e4790e290e330fa3a53ffe4016198e0"
EMBEDDING_MODEL="${ROOT_DIR}/models/DeepSeek-R1-Distill-Llama-8B"
EMBEDDING_SHA256="bfb086cb87e9616e60805fc6f6f95826294b1de70b158cfea2d3e213df38ca11"
LOCK_FILE="${SCORE_DIR}/.evaluation.lock"
PID_FILE="${SCORE_DIR}/evaluation.pid"
LOG_FILE="${SCORE_DIR}/evaluation.log"
ENV_NAME="${FOMC_TRAINER_CONDA_ENV:-fomc_trainer}"

if [[ "${1:-}" != "--worker" ]]; then
  mkdir -p "${SCORE_DIR}"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi is required." >&2
    exit 1
  fi
  if ! command -v flock >/dev/null 2>&1 || ! command -v setsid >/dev/null 2>&1; then
    echo "ERROR: flock and setsid are required." >&2
    exit 1
  fi

  gpu1_uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
    | awk -F ', ' '$1 == 1 {print $2}')"
  if [[ -z "${gpu1_uuid}" ]]; then
    echo "ERROR: physical GPU1 was not found." >&2
    exit 1
  fi
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader \
      | grep -Fxq "${gpu1_uuid}"; then
    echo "ERROR: physical GPU1 already has a compute process." >&2
    exit 1
  fi

  nohup setsid flock --no-fork --nonblock "${LOCK_FILE}" \
    "${BASH_SOURCE[0]}" --worker >"${LOG_FILE}" 2>&1 < /dev/null &
  job_pid=$!
  echo "${job_pid}" >"${PID_FILE}"
  sleep 1
  if ! kill -0 "${job_pid}" 2>/dev/null; then
    echo "ERROR: background evaluation exited during startup." >&2
    tail -n 40 "${LOG_FILE}" >&2 || true
    exit 1
  fi

  echo "Started canonical LOO scoring on physical GPU1."
  echo "PID: ${job_pid}"
  echo "Log: ${LOG_FILE}"
  echo "Output: ${SCORE_DIR}"
  exit 0
fi

cd "${ROOT_DIR}"
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available in PATH." >&2
  exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

if [[ ! -f "${REFERENCE_FILE}" ]]; then
  echo "ERROR: frozen actual-Minutes reference is missing: ${REFERENCE_FILE}" >&2
  exit 1
fi
observed_reference_sha256="$(sha256sum "${REFERENCE_FILE}" | awk '{print $1}')"
if [[ "${observed_reference_sha256}" != "${REFERENCE_SHA256}" ]]; then
  echo "ERROR: actual-Minutes reference SHA-256 mismatch." >&2
  exit 1
fi
if [[ ! -d "${EMBEDDING_MODEL}" ]]; then
  echo "ERROR: embedding model is missing: ${EMBEDDING_MODEL}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "started_at_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "physical_gpu_index=1"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "reference_sha256=${REFERENCE_SHA256}"
echo "embedding_model_sha256=${EMBEDDING_SHA256}"

run_arm() {
  local arm="$1"
  local summary_file="${SCORE_DIR}/${arm}_summary.jsonl"
  local audit_file="${SCORE_DIR}/${arm}_summary.audit.json"

  if [[ -s "${summary_file}" && -s "${audit_file}" ]] \
      && jq -e '.target_mode == "actual-minutes" and .scored_pairs > 0' \
        "${audit_file}" >/dev/null; then
    echo "Skipping already completed arm: ${arm}"
    return
  fi

  echo "arm_started=${arm} $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  python -u -m jobs.eval.eval_leave_one_out \
    --input-folder "${RUN_ROOT}/generations/${arm}" \
    --target-mode actual-minutes \
    --reference-file "${REFERENCE_FILE}" \
    --reference-text-field response \
    --embedding-model-path "${EMBEDDING_MODEL}" \
    --embedding-model-sha256 "${EMBEDDING_SHA256}" \
    --embedding-batch-size 1 \
    --embedding-max-tokens 4096 \
    --embedding-long-text-policy chunk-mean \
    --score-chunk-size 64 \
    --bootstrap-samples 5000 \
    --bootstrap-seed 20260728 \
    --output-file "${summary_file}"
  echo "arm_completed=${arm} $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}

run_arm deletion_primary
run_arm deletion_stochastic

echo "completed_at_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
