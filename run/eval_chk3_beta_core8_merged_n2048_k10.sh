#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 POST_RELEASE_SHA256 {smoke|formal|all}" >&2
  exit 2
fi

POST_RELEASE_SHA256="$1"
MODE="$2"
if [[ ! "${POST_RELEASE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "POST_RELEASE_SHA256 must be 64 lowercase hex characters" >&2
  exit 2
fi
if [[ "${MODE}" != "smoke" && "${MODE}" != "formal" && "${MODE}" != "all" ]]; then
  echo "mode must be smoke, formal, or all" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${FOMC_PYTHON_BIN:-/home/haobin_cui/.conda/envs/fomc_trainer/bin/python}"
PRE_RELEASE="${REPO_ROOT}/dataset/processed/retrain_v2/chk3_minutes_external_holdout_1993_2008_all_regular_v1/release_manifest.json"
PRE_RELEASE_SHA256="82b045866d9ed6ccbc0d4f00014bffc97694859c2827215309dc8eabe5406937"
POST_RELEASE="${REPO_ROOT}/dataset/processed/retrain_v2/chk3_minutes_post2008_2009_2025_fixed_core8_d1_v1/release_manifest.json"
RUN_ROOT="${REPO_ROOT}/output/evaluation/main/chk3_beta_core8_merged_n2048_k10_20260815_v1"
PREP_ROOT="${RUN_ROOT}/preparation"
SAMPLE_MANIFEST="${PREP_ROOT}/samples_n2048_k10.json"
SMOKE_ROOT="${RUN_ROOT}/generation_smoke_n1_k1_three_models"
FORMAL_ROOT="${RUN_ROOT}/generation_formal_n2048_k10_three_models"
DOCUMENT_ROOT="${RUN_ROOT}/meeting_documents_n7680"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

cd "${REPO_ROOT}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "missing Python environment: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${POST_RELEASE}" ]]; then
  echo "post release is not ready: ${POST_RELEASE}" >&2
  exit 1
fi

if [[ ! -f "${PREP_ROOT}/preparation.json" ]]; then
  "${PYTHON_BIN}" -u -m jobs.eval.prepare_chk3_beta_core8_merged_k10 \
    --pre-release "${PRE_RELEASE}" \
    --pre-release-sha256 "${PRE_RELEASE_SHA256}" \
    --post-release "${POST_RELEASE}" \
    --post-release-sha256 "${POST_RELEASE_SHA256}" \
    --output-dir "${PREP_ROOT}"
fi

if ! jq -e --arg sha "${POST_RELEASE_SHA256}" \
  '.source_releases.post2008_chk3_release.sha256 == $sha and .coverage.prompts == 2048 and .coverage.total_generation_rows == 61440' \
  "${PREP_ROOT}/preparation.json" >/dev/null; then
  echo "existing preparation does not match the requested post release/profile" >&2
  exit 1
fi
SAMPLE_MANIFEST_SHA256="$(sha256sum "${SAMPLE_MANIFEST}" | awk '{print $1}')"

run_or_validate_suite() {
  local scope="$1"
  local output_root="$2"
  local smoke_flag=()
  if [[ "${scope}" == "infrastructure_smoke" ]]; then
    smoke_flag=(--smoke)
  fi
  if [[ -f "${output_root}/manifest.json" ]]; then
    "${PYTHON_BIN}" -u -m jobs.eval.eval_chk3_beta_core8_merged_stochastic_k10 \
      --pre-release "${PRE_RELEASE}" \
      --pre-release-sha256 "${PRE_RELEASE_SHA256}" \
      --post-release "${POST_RELEASE}" \
      --post-release-sha256 "${POST_RELEASE_SHA256}" \
      validate-suite \
      --manifest "${output_root}/manifest.json" \
      --sample-manifest "${SAMPLE_MANIFEST}" \
      --sample-manifest-sha256 "${SAMPLE_MANIFEST_SHA256}" \
      --scope "${scope}"
    return
  fi
  local resume_flag=()
  if [[ -d "${output_root}" ]]; then
    resume_flag=(--resume)
  fi
  "${PYTHON_BIN}" -u -m jobs.eval.eval_chk3_beta_core8_merged_stochastic_k10 \
    --pre-release "${PRE_RELEASE}" \
    --pre-release-sha256 "${PRE_RELEASE_SHA256}" \
    --post-release "${POST_RELEASE}" \
    --post-release-sha256 "${POST_RELEASE_SHA256}" \
    run-suite \
    --sample-manifest "${SAMPLE_MANIFEST}" \
    --sample-manifest-sha256 "${SAMPLE_MANIFEST_SHA256}" \
    --output-dir "${output_root}" \
    --gpu-wait-timeout-seconds 172800 \
    --gpu-poll-seconds 30 \
    "${smoke_flag[@]}" \
    "${resume_flag[@]}"
}

if [[ "${MODE}" == "smoke" || "${MODE}" == "all" ]]; then
  run_or_validate_suite infrastructure_smoke "${SMOKE_ROOT}"
fi

if [[ "${MODE}" == "formal" || "${MODE}" == "all" ]]; then
  if [[ ! -f "${SMOKE_ROOT}/manifest.json" ]]; then
    echo "formal generation requires the sealed three-model smoke first" >&2
    exit 1
  fi
  run_or_validate_suite infrastructure_smoke "${SMOKE_ROOT}"
  run_or_validate_suite formal_full_test "${FORMAL_ROOT}"
  if [[ -f "${DOCUMENT_ROOT}/manifest.json" ]]; then
    "${PYTHON_BIN}" -u -m jobs.eval.assemble_chk3_beta_core8_meeting_documents \
      validate --manifest "${DOCUMENT_ROOT}/manifest.json"
  else
    "${PYTHON_BIN}" -u -m jobs.eval.assemble_chk3_beta_core8_meeting_documents \
      assemble \
      --pre-release "${PRE_RELEASE}" \
      --pre-release-sha256 "${PRE_RELEASE_SHA256}" \
      --post-release "${POST_RELEASE}" \
      --post-release-sha256 "${POST_RELEASE_SHA256}" \
      --suite-manifest "${FORMAL_ROOT}/manifest.json" \
      --sample-manifest "${SAMPLE_MANIFEST}" \
      --sample-manifest-sha256 "${SAMPLE_MANIFEST_SHA256}" \
      --output-dir "${DOCUMENT_ROOT}"
  fi
fi
