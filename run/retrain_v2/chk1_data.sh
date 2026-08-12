#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_ENV="${FOMC_RETRAIN_TRAIN_ENV:-fomc_trainer}"

if (($# == 0)); then
  echo "Usage: $0 dry-run | generate --mode smoke|pilot|full [--resume] | verify --mode smoke|pilot|full | publish [args]" >&2
  exit 2
fi

cd "${ROOT_DIR}"
TRAIN_PYTHON="$(conda run -n "${TRAIN_ENV}" python -c 'import sys; print(sys.executable)')"
if [[ -z "${TRAIN_PYTHON}" || "${TRAIN_PYTHON}" == *$'\n'* || ! -x "${TRAIN_PYTHON}" ]]; then
  echo "ERROR: unable to resolve the ${TRAIN_ENV} Python executable." >&2
  exit 2
fi

if [[ "$1" == "generate" ]]; then
  "${ROOT_DIR}/run/retrain_v2/resource_gate.sh"
  conda run --no-capture-output -n "${TRAIN_ENV}" python -c \
    'import inspect, openai, transformers; from openai.resources.chat.completions.completions import Completions; params=inspect.signature(Completions.create).parameters; required={"reasoning_effort","response_format","extra_body"}; missing=required-set(params); assert not missing, f"OpenAI client lacks DeepSeek V4 parameters: {sorted(missing)}"; print(f"chk1 DeepSeek acquisition environment OK: openai={openai.__version__} transformers={transformers.__version__}")'
fi

if [[ "$1" == "publish" ]]; then
  exec "${TRAIN_PYTHON}" -m jobs.retrain_v2.chk1.canonical_workflow \
    --repo-root "${ROOT_DIR}" \
    "$@"
fi

exec "${TRAIN_PYTHON}" -m jobs.retrain_v2.chk1.workflow \
  --repo-root "${ROOT_DIR}" \
  "$@"
