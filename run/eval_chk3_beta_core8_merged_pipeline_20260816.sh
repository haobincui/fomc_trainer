#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${FOMC_PYTHON_BIN:-/home/haobin_cui/.conda/envs/fomc_trainer/bin/python}"
CANDIDATE_ROOT="${REPO_ROOT}/output/data/retrain_v2/chk3/chk3_minutes_post2008_2009_2025_fixed_core8_d1_v1_candidate"
RELEASE_ROOT="${REPO_ROOT}/dataset/processed/retrain_v2/chk3_minutes_post2008_2009_2025_fixed_core8_d1_v1"
RELEASE_MANIFEST="${RELEASE_ROOT}/release_manifest.json"
RUN_ROOT="${REPO_ROOT}/output/evaluation/main/chk3_beta_core8_merged_n2048_k10_20260815_v1"
PIPELINE_LOG="${RUN_ROOT}/pipeline_20260816.log"
PIPELINE_LOCK="/tmp/fomc_trainer_chk3_beta_core8_merged_pipeline_20260816.lock"
ACQUIRE_PATTERN="jobs.eval.build_chk3_post2008_fixed_core8_d1 acquire"

mkdir -p "${RUN_ROOT}"
exec 9>"${PIPELINE_LOCK}"
if ! flock -n 9; then
  echo "another merged beta pipeline already holds ${PIPELINE_LOCK}" >&2
  exit 2
fi
exec >>"${PIPELINE_LOG}" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] pipeline_start"
echo "repo_root=${REPO_ROOT}"
echo "python=${PYTHON_BIN}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "missing Python environment: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

while pgrep -f -- "${ACQUIRE_PATTERN}" >/dev/null; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting_for_active_start_d1_acquisition"
  sleep 30
done

if ! jq -e '
  .status == "complete_pending_core8_publish_gate"
  and .meeting_count == 128
' "${CANDIDATE_ROOT}/reports/source_acquisition.json" >/dev/null 2>&1; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] resuming_start_d1_acquisition"
  "${PYTHON_BIN}" -u -m jobs.eval.build_chk3_post2008_fixed_core8_d1 \
    acquire --resume --requests-per-second 2 --max-workers 2
fi

if ! jq -e '
  .status == "complete_pending_core8_publish_gate"
  and .meeting_count == 128
' "${CANDIDATE_ROOT}/reports/source_acquisition.json" >/dev/null; then
  echo "source acquisition did not reach the publication gate" >&2
  exit 1
fi

if [[ -f "${RELEASE_MANIFEST}" ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] validating_existing_post_release"
else
  if [[ -e "${RELEASE_ROOT}" ]]; then
    echo "post release path exists without a release manifest: ${RELEASE_ROOT}" >&2
    exit 1
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] publishing_post_release"
  "${PYTHON_BIN}" -u -m jobs.eval.build_chk3_post2008_fixed_core8_d1 publish
fi

"${PYTHON_BIN}" -u -m jobs.eval.build_chk3_post2008_fixed_core8_d1 validate
POST_RELEASE_SHA256="$(sha256sum "${RELEASE_MANIFEST}" | awk '{print $1}')"
if [[ ! "${POST_RELEASE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid post release SHA-256" >&2
  exit 1
fi
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] post_release_sha256=${POST_RELEASE_SHA256}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting_gpu0_smoke_then_formal"
"${REPO_ROOT}/run/eval_chk3_beta_core8_merged_n2048_k10.sh" \
  "${POST_RELEASE_SHA256}" all
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] pipeline_complete"
