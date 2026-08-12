#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_ENV="${FOMC_RETRAIN_TRAIN_ENV:-fomc_trainer}"
RUN_ID=""
BASE_RELEASE_ID=""
DATASET_RELEASE_ID=""
DAG_PATH="configs/retrain_v2/dag.yaml"
INITIALIZE=false

usage() {
  echo "Usage: $0 --run-id ID (--base-release-id ID | --dataset-release-id ID) [--dag PATH] [--initialize]" >&2
  echo "Without --initialize this is a read-only DAG dry-run." >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --dataset-release-id)
      DATASET_RELEASE_ID="${2:-}"
      shift 2
      ;;
    --base-release-id)
      BASE_RELEASE_ID="${2:-}"
      shift 2
      ;;
    --dag)
      DAG_PATH="${2:-}"
      shift 2
      ;;
    --initialize)
      INITIALIZE=true
      shift
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if [[ -n "${BASE_RELEASE_ID}" && -n "${DATASET_RELEASE_ID}" && "${BASE_RELEASE_ID}" != "${DATASET_RELEASE_ID}" ]]; then
  echo "ERROR: --base-release-id and --dataset-release-id disagree." >&2
  exit 2
fi

BASE_RELEASE_ID="${BASE_RELEASE_ID:-${DATASET_RELEASE_ID}}"
if [[ -z "${RUN_ID}" || -z "${BASE_RELEASE_ID}" ]]; then
  usage
  exit 2
fi
if [[ -z "${DAG_PATH}" ]]; then
  echo "ERROR: --dag cannot be empty." >&2
  exit 2
fi

cd "${ROOT_DIR}"

if [[ "${INITIALIZE}" == false ]]; then
  conda run --no-capture-output -n "${TRAIN_ENV}" python -m jobs.retrain_v2.dag \
    --repo-root "${ROOT_DIR}" \
    --dag "${DAG_PATH}" \
    plan \
    --run-id "${RUN_ID}" \
    --base-release-id "${BASE_RELEASE_ID}"
  conda run --no-capture-output -n "${TRAIN_ENV}" python -m jobs.retrain_v2.dag \
    --repo-root "${ROOT_DIR}" \
    --dag "${DAG_PATH}" \
    validate-base-release \
    --base-release-id "${BASE_RELEASE_ID}"
  exit 0
fi

conda run --no-capture-output -n "${TRAIN_ENV}" python -m jobs.retrain_v2.dag \
  --repo-root "${ROOT_DIR}" \
  --dag "${DAG_PATH}" \
  init-run \
  --run-id "${RUN_ID}" \
  --base-release-id "${BASE_RELEASE_ID}"

echo "Run initialized. Execute each stage explicitly with run/retrain_v2/stage.sh."
