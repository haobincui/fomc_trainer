#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JUDGE_ENV="${FOMC_RETRAIN_JUDGE_ENV:-fomc_judge_v2}"
JUDGE_GPU_ID="${FOMC_RETRAIN_JUDGE_GPU_ID:-0}"
JUDGE_HOST="${FOMC_RETRAIN_JUDGE_HOST:-127.0.0.1}"
JUDGE_PORT="${FOMC_RETRAIN_JUDGE_PORT:-8000}"
PINNED_JUDGE_MODEL_PATH="${ROOT_DIR}/models/Qwen3.5-9B"
JUDGE_MODEL_PATH="${FOMC_RETRAIN_JUDGE_MODEL_PATH:-${PINNED_JUDGE_MODEL_PATH}}"
JUDGE_MODEL_NAME="${OPEN_R1_JUDGE_MODEL:-Qwen3.5-9B}"
PINNED_JUDGE_MAX_MODEL_LEN="8192"
JUDGE_MAX_MODEL_LEN="${FOMC_RETRAIN_JUDGE_MAX_MODEL_LEN:-${PINNED_JUDGE_MAX_MODEL_LEN}}"
JUDGE_MAX_NUM_SEQS="${FOMC_RETRAIN_JUDGE_MAX_NUM_SEQS:-4}"
JUDGE_GPU_MEMORY_UTILIZATION="${FOMC_RETRAIN_JUDGE_GPU_MEMORY_UTILIZATION:-0.69}"

if [[ ! "${JUDGE_GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: FOMC_RETRAIN_JUDGE_GPU_ID must contain one numeric GPU ID." >&2
  exit 2
fi
if [[ "${JUDGE_GPU_ID}" != "0" ]]; then
  echo "ERROR: retrain-v2 judge is pinned to physical GPU 0." >&2
  exit 2
fi
if [[ "${JUDGE_HOST}" != "127.0.0.1" || "${JUDGE_PORT}" != "8000" ]]; then
  echo "ERROR: retrain-v2 judge endpoint is pinned to 127.0.0.1:8000." >&2
  exit 2
fi
if [[ "${JUDGE_MAX_MODEL_LEN}" != "8192" && "${JUDGE_MAX_MODEL_LEN}" != "12288" ]]; then
  echo "ERROR: retrain-v2 judge max model length is pinned to 8192 or 12288." >&2
  exit 2
fi
if [[ "${JUDGE_MAX_NUM_SEQS}" != "4" ]]; then
  echo "ERROR: retrain-v2 judge max concurrent sequences is pinned to 4." >&2
  exit 2
fi
if [[ "${JUDGE_GPU_MEMORY_UTILIZATION}" != "0.69" ]]; then
  echo "ERROR: retrain-v2 judge GPU memory utilization is pinned to 0.69 for the current shared-GPU snapshot." >&2
  exit 2
fi
if [[ "$(realpath -m "${JUDGE_MODEL_PATH}")" != "$(realpath -m "${PINNED_JUDGE_MODEL_PATH}")" ]]; then
  echo "ERROR: judge model override disagrees with the pinned DAG artifact: ${PINNED_JUDGE_MODEL_PATH}" >&2
  exit 2
fi
if [[ "${JUDGE_MODEL_NAME}" != "Qwen3.5-9B" ]]; then
  echo "ERROR: served judge alias must be the pinned value Qwen3.5-9B." >&2
  exit 2
fi
if [[ ! -d "${JUDGE_MODEL_PATH}" ]]; then
  echo "ERROR: local Qwen model is missing: ${JUDGE_MODEL_PATH}" >&2
  exit 2
fi
if ! command -v conda >/dev/null 2>&1 || ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: conda and nvidia-smi are required." >&2
  exit 2
fi

"${ROOT_DIR}/run/retrain_v2/gpu_gate.sh" "${JUDGE_GPU_ID}"

export CUDA_VISIBLE_DEVICES="${JUDGE_GPU_ID}"
export OPEN_R1_JUDGE_URL="http://${JUDGE_HOST}:${JUDGE_PORT}/v1/chat/completions"
export OPEN_R1_JUDGE_MODEL="${JUDGE_MODEL_NAME}"

exec conda run --no-capture-output -n "${JUDGE_ENV}" \
  python -m vllm.entrypoints.openai.api_server \
  --host "${JUDGE_HOST}" \
  --port "${JUDGE_PORT}" \
  --model "${JUDGE_MODEL_PATH}" \
  --served-model-name "${JUDGE_MODEL_NAME}" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --language-model-only \
  --seed 42 \
  --gpu-memory-utilization "${JUDGE_GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${JUDGE_MAX_MODEL_LEN}" \
  --max-num-seqs "${JUDGE_MAX_NUM_SEQS}" \
  --enforce-eager \
  --no-enable-log-requests
