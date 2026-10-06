#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {prepare|smoke|formal|all|validate|status}" >&2
  exit 2
fi

MODE="$1"
case "${MODE}" in
  prepare|smoke|formal|all|validate|status) ;;
  *)
    echo "unsupported mode: ${MODE}" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FOMC_PYTHON_BIN="${FOMC_PYTHON_BIN:-/home/haobin_cui/.conda/envs/fomc_trainer/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/home/haobin_cui/.conda/envs/vllm_env/bin/python}"
PREPARER_MODULE="jobs.eval.prepare_chk3_core8_loo_vllm_k5"
RUNNER_MODULE="jobs.eval.eval_chk3_core8_loo_vllm_k5_dual_dp1"
ORCHESTRATOR_MODULE="jobs.eval.orchestrate_chk3_core8_loo_vllm_k5_dual_dp1"

RUN_ROOT="${REPO_ROOT}/output/evaluation/main/chk3_cp318_core8_loo_vllm_k5_n128_1993_2008_20260817_v1"
PREPARATION_ROOT="${RUN_ROOT}/preparation_v2"
COHORT="${PREPARATION_ROOT}/cohort_n2176_k5.v2.json"
COHORT_SHA256="26c1a06231fd8b10a0ccf2a9de22a59b7d657696da88a18981bad6fb82762e1c"
POLICY="${PREPARATION_ROOT}/execution_policy.v2.json"
POLICY_SHA256="611b25af74ac21409071ab10ce08574d2afe9aaf38b920d9f40e02b3febbbde3"
LEDGER="${PREPARATION_ROOT}/prompt_token_ledger_n2176.v2.jsonl"
LEDGER_SHA256="e5f1ab225c4b784dd00b651f139edb94b3f6c1a38b2e70f53467617595739224"
PREPARATION_MANIFEST="${PREPARATION_ROOT}/preparation.v2.json"
PREPARATION_SHA256="4d811507c938c1d1d6a97b9b02d1f3b07ad42fc716ee8be7a19b7825547dfa4f"
SMOKE_ROOT="${RUN_ROOT}/generation_smoke_two_meetings_k5_v1"
FORMAL_ROOT="${RUN_ROOT}/generation_formal_n2176_k5_v1"
SMOKE_MANIFEST="${SMOKE_ROOT}/chk3/manifest.json"
FORMAL_MANIFEST="${FORMAL_ROOT}/chk3/manifest.json"
PIPELINE_LOG="${RUN_ROOT}/pipeline_dual_dp1_20260817.log"
PIPELINE_LOCK="/tmp/fomc_trainer_chk3_core8_loo_vllm_k5_dual_dp1_pipeline.lock"
MAX_NUM_SEQS=16

GPU_WAIT_TIMEOUT_SECONDS="${GPU_WAIT_TIMEOUT_SECONDS:-172800}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-1800}"

mkdir -p "${RUN_ROOT}"
exec 9>"${PIPELINE_LOCK}"
if ! flock -n 9; then
  echo "another LOO K5 pipeline holds ${PIPELINE_LOCK}" >&2
  exit 2
fi
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
fail() { echo "[$(timestamp)] ERROR: $*" >&2; exit 1; }

require_sha256() {
  local path="$1"
  local expected="$2"
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing or unsafe file: ${path}"
  local observed
  observed="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${observed}" == "${expected}" ]] || \
    fail "SHA-256 drift: ${path} expected=${expected} observed=${observed}"
}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES || true
unset VLLM_DP_MASTER_IP VLLM_DP_MASTER_PORT || true
unset VLLM_DP_RANK VLLM_DP_RANK_LOCAL VLLM_DP_SIZE || true
unset MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE || true
unset GROUP_RANK ROLE_RANK || true
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd "${REPO_ROOT}"
[[ -x "${FOMC_PYTHON_BIN}" ]] || fail "missing FOMC Python"
[[ -x "${VLLM_PYTHON_BIN}" ]] || fail "missing vLLM Python"
command -v jq >/dev/null 2>&1 || fail "jq is required"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum is required"

prepare_or_validate() {
  if [[ ! -e "${PREPARATION_ROOT}" ]]; then
    "${FOMC_PYTHON_BIN}" -u -m "${PREPARER_MODULE}" \
      --output-dir "${PREPARATION_ROOT}"
  fi
  require_sha256 "${COHORT}" "${COHORT_SHA256}"
  require_sha256 "${POLICY}" "${POLICY_SHA256}"
  require_sha256 "${LEDGER}" "${LEDGER_SHA256}"
  require_sha256 "${PREPARATION_MANIFEST}" "${PREPARATION_SHA256}"
  "${FOMC_PYTHON_BIN}" - "${COHORT}" "${COHORT_SHA256}" <<'PY'
import sys
from pathlib import Path
from jobs.eval import eval_chk3_core8_loo_vllm_k5_dual_dp1 as runner

cohort, ledger, sample_manifest, bound_rows = runner.load_cohort(
    Path(sys.argv[1]), sys.argv[2]
)
cases = runner.canonical_cases(
    samples=sample_manifest["samples"], ledger=ledger, smoke=False
)
assert len(cases) == 10880
assert [len(runner.shard_cases(cases, shard)) for shard in (0, 1)] == [5440, 5440]
print("sealed_preparation_valid rows=10880 shard0=5440 shard1=5440")
PY
}

run_model() {
  local root="$1"
  local scope="$2"
  local resume="$3"
  local args=()
  if [[ "${resume}" == "true" ]]; then
    args+=(--allow-resume)
  fi
  if [[ "${scope}" == "formal_merged_panel" ]]; then
    args+=(--formal-authorization "${SMOKE_MANIFEST}")
  fi
  "${FOMC_PYTHON_BIN}" -u -m "${ORCHESTRATOR_MODULE}" \
    --python-bin "${VLLM_PYTHON_BIN}" \
    --model-id chk3 \
    --cohort "${COHORT}" \
    --cohort-sha256 "${COHORT_SHA256}" \
    --model-root "${root}/chk3" \
    --scope "${scope}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-wait-timeout-seconds "${GPU_WAIT_TIMEOUT_SECONDS}" \
    --gpu-poll-seconds "${GPU_POLL_SECONDS}" \
    --ready-timeout-seconds "${READY_TIMEOUT_SECONDS}" \
    "${args[@]}"
}

validate_run() {
  local manifest="$1"
  local scope="$2"
  [[ -f "${manifest}" && ! -L "${manifest}" ]] || \
    fail "sealed ${scope} manifest is missing: ${manifest}"
  "${FOMC_PYTHON_BIN}" -u -m "${RUNNER_MODULE}" validate-run \
    --manifest "${manifest}" \
    --cohort "${COHORT}" \
    --cohort-sha256 "${COHORT_SHA256}" \
    --model-id chk3 \
    --scope "${scope}" \
    --max-num-seqs "${MAX_NUM_SEQS}"
}

run_smoke() {
  run_model "${SMOKE_ROOT}" infrastructure_smoke false
  validate_run "${SMOKE_MANIFEST}" infrastructure_smoke
  jq -e '
    .coverage.cases == 170
    and .coverage.shard_cases == {"shard0":85,"shard1":85}
    and .coverage.input_truncation_cases == 0
    and .coverage.absolute_indexes_exact == true
    and .coverage.overlaps == 0
  ' "${SMOKE_MANIFEST}" >/dev/null || fail "LOO K5 smoke gates failed"
}

run_formal() {
  validate_run "${SMOKE_MANIFEST}" infrastructure_smoke
  run_model "${FORMAL_ROOT}" formal_merged_panel true
  validate_run "${FORMAL_MANIFEST}" formal_merged_panel
  jq -e '
    .coverage.cases == 10880
    and .coverage.shard_cases == {"shard0":5440,"shard1":5440}
    and .coverage.input_truncation_cases == 0
    and .coverage.absolute_indexes_exact == true
    and .coverage.overlaps == 0
  ' "${FORMAL_MANIFEST}" >/dev/null || fail "LOO K5 formal coverage failed"
}

show_status() {
  for root in "${SMOKE_ROOT}/chk3" "${FORMAL_ROOT}/chk3"; do
    echo "root=${root}"
    if [[ -f "${root}/manifest.json" ]]; then
      jq '{status,evaluation_scope,coverage,runtime}' "${root}/manifest.json"
    else
      for shard in 0 1; do
        local state="${root}/shard${shard}/state.progress.v1.json"
        if [[ -f "${state}" ]]; then
          jq '{status,shard_id,completed_cases,expected_cases,resume_count,updated_at_utc}' "${state}"
        else
          echo "shard${shard}: not_started"
        fi
      done
    fi
  done
}

echo "[$(timestamp)] pipeline_start mode=${MODE}"
prepare_or_validate

case "${MODE}" in
  prepare) ;;
  smoke) run_smoke ;;
  formal) run_formal ;;
  all)
    run_smoke
    run_formal
    ;;
  validate)
    validate_run "${SMOKE_MANIFEST}" infrastructure_smoke
    validate_run "${FORMAL_MANIFEST}" formal_merged_panel
    ;;
  status) show_status ;;
esac

echo "[$(timestamp)] pipeline_complete mode=${MODE}"

