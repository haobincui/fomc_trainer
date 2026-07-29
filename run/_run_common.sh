#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${FOMC_TRAINER_CONDA_ENV:-fomc_trainer}"
LOG_DIR="${ROOT_DIR}/logs/train"
JUDGE_LOG_DIR="${ROOT_DIR}/logs/judge"

activate_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found in PATH." >&2
    exit 1
  fi

  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck disable=SC1090
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
}

prepare_training_job() {
  local job_name="$1"

  activate_conda
  mkdir -p "${LOG_DIR}"

  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${LOG_DIR}/${job_name}_${TIMESTAMP}.log"
  PID_FILE="${LOG_DIR}/${job_name}_${TIMESTAMP}.pid"
}

prepare_judge_job() {
  local job_name="$1"

  activate_conda
  mkdir -p "${JUDGE_LOG_DIR}"

  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${JUDGE_LOG_DIR}/${job_name}_${TIMESTAMP}.log"
  PID_FILE="${JUDGE_LOG_DIR}/${job_name}_${TIMESTAMP}.pid"
}

enable_grpo_judge() {
  export OPEN_R1_JUDGE_URL="${OPEN_R1_JUDGE_URL:-http://localhost:11432/api/chat/}"
  export OPEN_R1_JUDGE_MODEL="${OPEN_R1_JUDGE_MODEL:-gemma3:12b}"
}

require_judge_ready() {
  local chat_url="$1"

  python - "${chat_url}" <<'PY'
import json
import sys
import urllib.error
import urllib.request

chat_url = sys.argv[1].rstrip("/")
if chat_url.endswith("/chat/completions"):
    models_url = chat_url.rsplit("/chat/completions", 1)[0] + "/models"
else:
    models_url = chat_url + "/models"

try:
    with urllib.request.urlopen(models_url, timeout=5) as response:
        payload = json.load(response)
except Exception as exc:
    print(f"ERROR: judge is not reachable at {models_url}: {exc}", file=sys.stderr)
    raise SystemExit(1)

if not isinstance(payload, dict) or "data" not in payload:
    print(f"ERROR: unexpected judge response from {models_url}: {payload!r}", file=sys.stderr)
    raise SystemExit(1)

print(f"Judge is reachable: {models_url}")
PY
}

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: required path not found: ${path}" >&2
    exit 1
  fi
}

load_flat_yaml_vars() {
  local yaml_path="$1"
  local prefix="$2"

  activate_conda

  python - "${yaml_path}" "${prefix}" <<'PY'
import shlex
import sys

import yaml

yaml_path, prefix = sys.argv[1], sys.argv[2]
with open(yaml_path, encoding="utf-8") as handle:
    payload = yaml.safe_load(handle) or {}

if not isinstance(payload, dict):
    raise SystemExit(f"ERROR: expected a mapping in {yaml_path}")

for key, value in payload.items():
    if isinstance(value, (dict, list)):
        raise SystemExit(
            f"ERROR: only flat key/value YAML is supported in {yaml_path}; "
            f"key {key!r} has unsupported type {type(value).__name__}"
        )
    if value is None:
        continue

    if isinstance(value, bool):
        rendered = "1" if value else "0"
    else:
        rendered = str(value)

    env_name = f"{prefix}{key.upper()}"
    print(f"{env_name}={shlex.quote(rendered)}")
PY
}

print_job_header() {
  local job_name="$1"
  local description="$2"

  echo "==============================================="
  echo "Launching ${job_name} in background"
  echo "Description: ${description}"
  echo "Conda env: ${ENV_NAME}"
  echo "Python: $(which python)"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
  if [[ -n "${OPEN_R1_JUDGE_URL:-}" ]]; then
    echo "Judge URL: ${OPEN_R1_JUDGE_URL}"
    echo "Judge Model: ${OPEN_R1_JUDGE_MODEL}"
  fi
  echo "Log file: ${LOG_FILE}"
  echo "==============================================="
}

record_background_pid() {
  local pid="$1"
  echo "${pid}" > "${PID_FILE}"
  echo
  echo "Started PID: ${pid}"
  echo "PID file: ${PID_FILE}"
  echo "Monitor: tail -f ${LOG_FILE}"
}
