#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {prepare|authorize|smoke|formal|all|validate}" >&2
  exit 2
fi

MODE="$1"
case "${MODE}" in
  prepare|authorize|smoke|formal|all|validate) ;;
  *)
    echo "mode must be prepare, authorize, smoke, formal, all, or validate" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FOMC_PYTHON_BIN="${FOMC_PYTHON_BIN:-/home/haobin_cui/.conda/envs/fomc_trainer/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/home/haobin_cui/.conda/envs/vllm_env/bin/python}"
ORCHESTRATOR_MODULE="jobs.eval.orchestrate_chk3_beta_core8_merged_vllm_k5_dual_dp1_stochastic_schedule_v2"
SELECTOR_MODULE="jobs.eval.select_chk3_beta_core8_vllm_k5_dual_dp1_stochastic_schedule_v2"
SUITE_MODULE="jobs.eval.seal_chk3_beta_core8_vllm_k5_dual_dp1_stochastic_schedule_v2_suite"
AMENDMENT_MODULE="jobs.eval.remediate_chk3_beta_core8_vllm_k5_dual_dp1_stochastic_schedule_v2"

PRE_RELEASE="${REPO_ROOT}/dataset/processed/retrain_v2/chk3_minutes_external_holdout_1993_2008_all_regular_v1/release_manifest.json"
PRE_RELEASE_SHA256="82b045866d9ed6ccbc0d4f00014bffc97694859c2827215309dc8eabe5406937"
POST_RELEASE="${REPO_ROOT}/dataset/processed/retrain_v2/chk3_minutes_post2008_2009_2025_fixed_core8_d1_v1/release_manifest.json"
POST_RELEASE_SHA256="d7e6d9ea534039f60343dcab317588b45c1de5aa2814d2e6fff0d8ca0f1feb21"
SOURCE_SAMPLE_MANIFEST="${REPO_ROOT}/output/evaluation/main/chk3_beta_core8_merged_n2048_k10_20260815_v1/preparation/samples_n2048_k10.json"
SOURCE_SAMPLE_MANIFEST_SHA256="e8d602e9ed1da8d5bd3d2e8dae11368667dfd2bc6192003b78a2b7d31887b866"

RUN_ROOT="${REPO_ROOT}/output/evaluation/main/chk3_beta_core8_merged_n2048_vllm_k5_20260816_v1"
PREPARATION_MANIFEST="${RUN_ROOT}/preparation/preparation.json"
COHORT_MANIFEST="${RUN_ROOT}/preparation/cohort_n2048_k5.v1.json"
COHORT_SHA256="a815b5af8e6b33e3d1a2b211e346393f155d1d45acaafae80b822ee08128ab8e"
V3_BENCHMARK_ROOT="${RUN_ROOT}/benchmark_max_num_seqs_chk1_core8_k5_v3_dual_dp1"
V3_MANIFEST_8="${V3_BENCHMARK_ROOT}/max_num_seqs_8/chk1/manifest.json"
V3_MANIFEST_12="${V3_BENCHMARK_ROOT}/max_num_seqs_12/chk1/manifest.json"
V3_MANIFEST_16="${V3_BENCHMARK_ROOT}/max_num_seqs_16/chk1/manifest.json"
V3_REPLAY_16="${V3_BENCHMARK_ROOT}/max_num_seqs_16_replay/chk1/manifest.json"
POLICY_ROOT="${RUN_ROOT}/benchmark_stochastic_schedule_v2_fixed16"
SPEED_SELECTION="${POLICY_ROOT}/selection.speed_only_fixed16.v2.json"
SMOKE_ROOT="${RUN_ROOT}/generation_smoke_core8_k5_three_models_v4_stochastic_schedule_v2"
FORMAL_ROOT="${RUN_ROOT}/generation_formal_n2048_k5_three_models_v4_stochastic_schedule_v2"
AMENDMENT_RECEIPT="${RUN_ROOT}/migration/vllm_dual_dp1_stochastic_schedule_v2_amendment_receipt.json"
PIPELINE_LOG="${RUN_ROOT}/pipeline_vllm_k5_v4_stochastic_schedule_v2_20260816.log"
PIPELINE_LOCK="/tmp/fomc_trainer_chk3_beta_core8_vllm_k5_stochastic_schedule_v2_pipeline.lock"
FIXED_MAX_NUM_SEQS=16
MODEL_ORDER=(chk1 chk3 chk0)

GPU_WAIT_TIMEOUT_SECONDS="${GPU_WAIT_TIMEOUT_SECONDS:-172800}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"

mkdir -p "${RUN_ROOT}"
exec 9>"${PIPELINE_LOCK}"
if ! flock -n 9; then
  echo "another stochastic-schedule-v2 pipeline holds ${PIPELINE_LOCK}" >&2
  exit 2
fi
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
fail() { echo "[$(timestamp)] ERROR: $*" >&2; exit 1; }
sha256_of() { sha256sum "$1" | awk '{print $1}'; }

require_exact_file() {
  local path="$1"
  local expected_sha="$2"
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing/unsafe file: ${path}"
  local observed
  observed="$(sha256_of "${path}")"
  [[ "${observed}" == "${expected_sha}" ]] || \
    fail "SHA-256 mismatch for ${path}: expected=${expected_sha} observed=${observed}"
}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES || true
unset VLLM_DP_MASTER_IP VLLM_DP_MASTER_PORT || true
unset VLLM_DP_RANK VLLM_DP_RANK_LOCAL VLLM_DP_SIZE || true
unset MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK || true
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
command -v setsid >/dev/null 2>&1 || fail "setsid is required"

if [[ -n "${VLLM_MAX_NUM_SEQS:-}" && "${VLLM_MAX_NUM_SEQS}" != "16" ]]; then
  fail "stochastic-schedule-v2 fixes VLLM_MAX_NUM_SEQS at 16"
fi
export VLLM_MAX_NUM_SEQS=16

prepare_or_validate() {
  require_exact_file "${PRE_RELEASE}" "${PRE_RELEASE_SHA256}"
  require_exact_file "${POST_RELEASE}" "${POST_RELEASE_SHA256}"
  require_exact_file "${SOURCE_SAMPLE_MANIFEST}" "${SOURCE_SAMPLE_MANIFEST_SHA256}"
  require_exact_file "${COHORT_MANIFEST}" "${COHORT_SHA256}"
  [[ -f "${PREPARATION_MANIFEST}" && ! -L "${PREPARATION_MANIFEST}" ]] || \
    fail "sealed preparation manifest is missing"
  jq -e '
    .status == "complete"
    and .coverage.meetings == 256
    and .coverage.prompts == 2048
    and .coverage.replicates == 5
    and .coverage.rows_per_model == 10240
    and .coverage.total_generation_rows == 30720
    and .validation.prompt_token_count_mismatches == 0
  ' "${PREPARATION_MANIFEST}" >/dev/null || fail "preparation coverage drift"
  echo "[$(timestamp)] cohort_sha256=${COHORT_SHA256}"
}

create_or_validate_authorization() {
  if [[ ! -f "${AMENDMENT_RECEIPT}" ]]; then
    [[ ! -e "${AMENDMENT_RECEIPT}" ]] || fail "unsafe amendment receipt path"
    "${FOMC_PYTHON_BIN}" -u -m "${AMENDMENT_MODULE}" create \
      --receipt "${AMENDMENT_RECEIPT}"
  fi
  "${FOMC_PYTHON_BIN}" -u -m "${AMENDMENT_MODULE}" validate \
    --receipt "${AMENDMENT_RECEIPT}"
  if [[ ! -f "${SPEED_SELECTION}" ]]; then
    [[ ! -e "${POLICY_ROOT}" ]] || fail "partial/unsafe v4 policy root"
    "${FOMC_PYTHON_BIN}" -u -m "${SELECTOR_MODULE}" select \
      --cohort "${COHORT_MANIFEST}" \
      --cohort-sha256 "${COHORT_SHA256}" \
      --manifest-8 "${V3_MANIFEST_8}" \
      --manifest-12 "${V3_MANIFEST_12}" \
      --manifest-16 "${V3_MANIFEST_16}" \
      --replay-manifest-16 "${V3_REPLAY_16}" \
      --output "${SPEED_SELECTION}"
  fi
  "${FOMC_PYTHON_BIN}" -u -m "${SELECTOR_MODULE}" validate \
    --manifest "${SPEED_SELECTION}" \
    --cohort "${COHORT_MANIFEST}" \
    --cohort-sha256 "${COHORT_SHA256}"
  jq -e '
    .selection.selected_max_num_seqs == 16
    and .selection.rationale == "max_num_seqs_16_is_strict_speed_argmax"
    and .same_config_replay.diagnostic_only_nonblocking == true
    and .same_config_replay.within_five_percent_token_volume == true
    and .generation_gates.speed_only_fixed_max_num_seqs_16 == true
    and .generation_gates.schedule_sensitive_stochastic_token_identity_nonblocking == true
    and .generation_gates.formal_generation_unblocked == true
  ' "${SPEED_SELECTION}" >/dev/null || fail "v4 selection/gate drift"
}

run_or_validate_model() {
  local model_id="$1"
  local suite_root="$2"
  local scope="$3"
  local allow_resume="$4"
  local resume_flag=()
  local auth_flag=()
  [[ "${allow_resume}" == "true" ]] && resume_flag=(--allow-resume)
  if [[ "${scope}" == "formal_merged_panel" ]]; then
    [[ -f "${SMOKE_ROOT}/manifest.json" && ! -L "${SMOKE_ROOT}/manifest.json" ]] || \
      fail "formal generation requires sealed v4 smoke authorization"
    auth_flag=(--formal-authorization "${SMOKE_ROOT}/manifest.json")
  fi
  "${FOMC_PYTHON_BIN}" -u -m "${ORCHESTRATOR_MODULE}" \
    --python-bin "${VLLM_PYTHON_BIN}" \
    --model-id "${model_id}" \
    --cohort "${COHORT_MANIFEST}" \
    --cohort-sha256 "${COHORT_SHA256}" \
    --model-root "${suite_root}/${model_id}" \
    --scope "${scope}" \
    --max-num-seqs "${FIXED_MAX_NUM_SEQS}" \
    --gpu-wait-timeout-seconds "${GPU_WAIT_TIMEOUT_SECONDS}" \
    --gpu-poll-seconds "${GPU_POLL_SECONDS}" \
    "${auth_flag[@]}" \
    "${resume_flag[@]}"
}

seal_or_validate_suite() {
  local suite_root="$1"
  local scope="$2"
  local smoke_flag=()
  if [[ "${scope}" == "formal_merged_panel" ]]; then
    smoke_flag=(--smoke-suite-manifest "${SMOKE_ROOT}/manifest.json")
  fi
  if [[ ! -f "${suite_root}/manifest.json" ]]; then
    "${FOMC_PYTHON_BIN}" -u -m "${SUITE_MODULE}" seal \
      --cohort "${COHORT_MANIFEST}" \
      --cohort-sha256 "${COHORT_SHA256}" \
      --output-dir "${suite_root}" \
      --scope "${scope}" \
      --max-num-seqs "${FIXED_MAX_NUM_SEQS}" \
      --speed-selection "${SPEED_SELECTION}" \
      "${smoke_flag[@]}"
  fi
  "${FOMC_PYTHON_BIN}" -u -m "${SUITE_MODULE}" validate \
    --manifest "${suite_root}/manifest.json" \
    --cohort "${COHORT_MANIFEST}" \
    --cohort-sha256 "${COHORT_SHA256}" \
    --scope "${scope}" \
    --max-num-seqs "${FIXED_MAX_NUM_SEQS}"
}

run_suite() {
  local suite_root="$1"
  local scope="$2"
  local model_id
  for model_id in "${MODEL_ORDER[@]}"; do
    run_or_validate_model "${model_id}" "${suite_root}" "${scope}" true
  done
  seal_or_validate_suite "${suite_root}" "${scope}"
}

validate_smoke() {
  [[ -f "${SMOKE_ROOT}/manifest.json" ]] || fail "sealed v4 smoke is required"
  seal_or_validate_suite "${SMOKE_ROOT}" infrastructure_smoke
  jq -e '
    .coverage.models == 3
    and .coverage.rows_per_model == 40
    and .coverage.rows_per_shard == 20
    and .coverage.total_rows == 120
    and .coverage.input_truncation_rows == 0
    and .coverage.normal_finish_rows == 120
    and .generation_gates.formal_generation_unblocked == true
  ' "${SMOKE_ROOT}/manifest.json" >/dev/null || fail "v4 smoke coverage/gates failed"
}

echo "[$(timestamp)] pipeline_start mode=${MODE}"
echo "gpu_contract=two_independent_dp1_workers physical_gpus=0,1 gpu_memory_utilization=0.95 max_num_seqs=16 stochastic_semantics=schedule_sensitive"
prepare_or_validate

if [[ "${MODE}" == "prepare" ]]; then
  echo "[$(timestamp)] pipeline_complete mode=prepare"
  exit 0
fi

create_or_validate_authorization

if [[ "${MODE}" == "authorize" ]]; then
  echo "[$(timestamp)] pipeline_complete mode=authorize"
  exit 0
fi

case "${MODE}" in
  smoke)
    run_suite "${SMOKE_ROOT}" infrastructure_smoke
    ;;
  formal)
    validate_smoke
    run_suite "${FORMAL_ROOT}" formal_merged_panel
    ;;
  all)
    run_suite "${SMOKE_ROOT}" infrastructure_smoke
    validate_smoke
    run_suite "${FORMAL_ROOT}" formal_merged_panel
    ;;
  validate)
    validate_smoke
    seal_or_validate_suite "${FORMAL_ROOT}" formal_merged_panel
    ;;
esac

echo "[$(timestamp)] pipeline_complete mode=${MODE}"
