#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {prepare|benchmark|smoke|formal|all|assemble|validate}" >&2
  exit 2
fi

MODE="$1"
case "${MODE}" in
  prepare|benchmark|smoke|formal|all|assemble|validate) ;;
  *)
    echo "mode must be prepare, benchmark, smoke, formal, all, assemble, or validate" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FOMC_PYTHON_BIN="${FOMC_PYTHON_BIN:-/home/haobin_cui/.conda/envs/fomc_trainer/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/home/haobin_cui/.conda/envs/vllm_env/bin/python}"
RUNNER_MODULE="jobs.eval.eval_chk3_beta_core8_merged_vllm_k5_dual_dp1"
ORCHESTRATOR_MODULE="jobs.eval.orchestrate_chk3_beta_core8_merged_vllm_k5_dual_dp1"
SELECTOR_MODULE="jobs.eval.select_chk3_beta_core8_vllm_k5_dual_dp1_benchmark"
SUITE_MODULE="jobs.eval.seal_chk3_beta_core8_vllm_k5_dual_dp1_suite"
ASSEMBLER_MODULE="jobs.eval.assemble_chk3_beta_core8_meeting_documents_vllm_k5_dual_dp1"
REMEDIATION_MODULE="jobs.eval.remediate_chk3_beta_core8_vllm_k5_dp2_to_dual_dp1"

PRE_RELEASE="${REPO_ROOT}/dataset/processed/retrain_v2/chk3_minutes_external_holdout_1993_2008_all_regular_v1/release_manifest.json"
PRE_RELEASE_SHA256="82b045866d9ed6ccbc0d4f00014bffc97694859c2827215309dc8eabe5406937"
POST_RELEASE="${REPO_ROOT}/dataset/processed/retrain_v2/chk3_minutes_post2008_2009_2025_fixed_core8_d1_v1/release_manifest.json"
POST_RELEASE_SHA256="d7e6d9ea534039f60343dcab317588b45c1de5aa2814d2e6fff0d8ca0f1feb21"
SOURCE_SAMPLE_MANIFEST="${REPO_ROOT}/output/evaluation/main/chk3_beta_core8_merged_n2048_k10_20260815_v1/preparation/samples_n2048_k10.json"
SOURCE_SAMPLE_MANIFEST_SHA256="e8d602e9ed1da8d5bd3d2e8dae11368667dfd2bc6192003b78a2b7d31887b866"

RUN_ROOT="${REPO_ROOT}/output/evaluation/main/chk3_beta_core8_merged_n2048_vllm_k5_20260816_v1"
PREP_ROOT="${RUN_ROOT}/preparation"
PREPARATION_MANIFEST="${PREP_ROOT}/preparation.json"
COHORT_MANIFEST="${PREP_ROOT}/cohort_n2048_k5.v1.json"
COHORT_SHA256_EXPECTED="a815b5af8e6b33e3d1a2b211e346393f155d1d45acaafae80b822ee08128ab8e"
BENCHMARK_ROOT="${RUN_ROOT}/benchmark_max_num_seqs_chk1_core8_k5_v3_dual_dp1"
BENCHMARK_SELECTION="${BENCHMARK_ROOT}/selection.v1.json"
SMOKE_ROOT="${RUN_ROOT}/generation_smoke_core8_k5_three_models_v3_dual_dp1"
FORMAL_ROOT="${RUN_ROOT}/generation_formal_n2048_k5_three_models_v3_dual_dp1"
DOCUMENT_ROOT="${RUN_ROOT}/meeting_documents_n3840_v3_dual_dp1"
REMEDIATION_RECEIPT="${RUN_ROOT}/migration/vllm_dp2_hang_to_dual_independent_dp1_receipt.json"
PIPELINE_LOG="${RUN_ROOT}/pipeline_vllm_k5_v3_dual_dp1_20260816.log"
PIPELINE_LOCK="/tmp/fomc_trainer_chk3_beta_core8_merged_vllm_k5_dual_dp1_pipeline.lock"

GPU_WAIT_TIMEOUT_SECONDS="${GPU_WAIT_TIMEOUT_SECONDS:-172800}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
REQUESTED_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
VLLM_MAX_NUM_SEQS=""
MODEL_ORDER=(chk1 chk3 chk0)
PHYSICAL_GPU_INDICES=(0 1)

mkdir -p "${RUN_ROOT}"
exec 9>"${PIPELINE_LOCK}"
if ! flock -n 9; then
  echo "another dual-independent-DP1 K=5 pipeline holds ${PIPELINE_LOCK}" >&2
  exit 2
fi
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

fail() {
  echo "[$(timestamp)] ERROR: $*" >&2
  exit 1
}

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

require_exact_file() {
  local path="$1"
  local expected_sha="$2"
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing/unsafe file: ${path}"
  local observed_sha
  observed_sha="$(sha256_of "${path}")"
  [[ "${observed_sha}" == "${expected_sha}" ]] || \
    fail "SHA-256 mismatch for ${path}: expected=${expected_sha} observed=${observed_sha}"
}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES || true
unset VLLM_DP_MASTER_IP VLLM_DP_MASTER_PORT || true
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
[[ -x "${FOMC_PYTHON_BIN}" ]] || fail "missing FOMC Python: ${FOMC_PYTHON_BIN}"
[[ -x "${VLLM_PYTHON_BIN}" ]] || fail "missing vLLM Python: ${VLLM_PYTHON_BIN}"
command -v jq >/dev/null 2>&1 || fail "jq is required"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum is required"
command -v setsid >/dev/null 2>&1 || fail "setsid is required"
require_exact_file "${PRE_RELEASE}" "${PRE_RELEASE_SHA256}"
require_exact_file "${POST_RELEASE}" "${POST_RELEASE_SHA256}"
require_exact_file "${SOURCE_SAMPLE_MANIFEST}" "${SOURCE_SAMPLE_MANIFEST_SHA256}"
if [[ -n "${REQUESTED_MAX_NUM_SEQS}" ]]; then
  case "${REQUESTED_MAX_NUM_SEQS}" in
    8|12|16) ;;
    *) fail "VLLM_MAX_NUM_SEQS, when supplied, must be exactly 8, 12, or 16" ;;
  esac
fi

prepare_or_validate() {
  [[ -f "${PREPARATION_MANIFEST}" && ! -L "${PREPARATION_MANIFEST}" ]] || \
    fail "sealed preparation manifest is missing"
  require_exact_file "${COHORT_MANIFEST}" "${COHORT_SHA256_EXPECTED}"
  jq -e \
    --arg source_sha "${SOURCE_SAMPLE_MANIFEST_SHA256}" \
    --arg pre_sha "${PRE_RELEASE_SHA256}" \
    --arg post_sha "${POST_RELEASE_SHA256}" '
      .status == "complete"
      and .immutable == true
      and .source_sample_manifest.sha256 == $source_sha
      and .harmonized_source_releases.pre2009_external.sha256 == $pre_sha
      and .harmonized_source_releases.post2008_chk3_release.sha256 == $post_sha
      and .generation_design.meetings == 256
      and .generation_design.prompts == 2048
      and .generation_design.replicates == 5
      and .generation_design.rows_per_model == 10240
      and .generation_design.total_rows == 30720
      and .generation_design.replicate_seeds == [20260811,21260811,22260811,23260811,24260811]
      and .token_ledger.full_prompt_token_ids_persisted == true
      and .token_ledger.vllm_runtime_chat_templating == false
    ' "${COHORT_MANIFEST}" >/dev/null || fail "sealed K=5 cohort contract drift"
  jq -e '
      .status == "complete"
      and .coverage.meetings == 256
      and .coverage.prompts == 2048
      and .coverage.replicates == 5
      and .coverage.rows_per_model == 10240
      and .coverage.total_generation_rows == 30720
      and .validation.exact_prompt_token_ids_persisted == true
      and .validation.prompt_token_count_mismatches == 0
    ' "${PREPARATION_MANIFEST}" >/dev/null || fail "preparation coverage drift"
  COHORT_SHA256="$(sha256_of "${COHORT_MANIFEST}")"
  [[ "${COHORT_SHA256}" == "${COHORT_SHA256_EXPECTED}" ]] || fail "cohort SHA drift"
  export COHORT_SHA256
  echo "[$(timestamp)] cohort_sha256=${COHORT_SHA256}"
}

require_topology_remediation_receipt() {
  [[ -f "${REMEDIATION_RECEIPT}" && ! -L "${REMEDIATION_RECEIPT}" ]] || \
    fail "sealed DP2-to-dual-DP1 remediation receipt is required before GPU work"
  "${FOMC_PYTHON_BIN}" -u -m "${REMEDIATION_MODULE}" \
    validate --receipt "${REMEDIATION_RECEIPT}"
}

run_or_validate_model() {
  local model_id="$1"
  local suite_root="$2"
  local scope="$3"
  local max_num_seqs="$4"
  local allow_resume="$5"
  local resume_flag=()
  local authorization_flag=()
  [[ "${allow_resume}" == "true" ]] && resume_flag=(--allow-resume)
  if [[ "${scope}" == "formal_merged_panel" ]]; then
    [[ -f "${SMOKE_ROOT}/manifest.json" && ! -L "${SMOKE_ROOT}/manifest.json" ]] || \
      fail "formal generation requires the official sealed smoke-suite authorization"
    authorization_flag=(--formal-authorization "${SMOKE_ROOT}/manifest.json")
  fi
  "${FOMC_PYTHON_BIN}" -u -m "${ORCHESTRATOR_MODULE}" \
    --python-bin "${VLLM_PYTHON_BIN}" \
    --model-id "${model_id}" \
    --cohort "${COHORT_MANIFEST}" \
    --cohort-sha256 "${COHORT_SHA256}" \
    --model-root "${suite_root}/${model_id}" \
    --scope "${scope}" \
    --max-num-seqs "${max_num_seqs}" \
    --gpu-wait-timeout-seconds "${GPU_WAIT_TIMEOUT_SECONDS}" \
    --gpu-poll-seconds "${GPU_POLL_SECONDS}" \
    "${authorization_flag[@]}" \
    "${resume_flag[@]}"
}

validate_benchmark_selection() {
  [[ -f "${BENCHMARK_SELECTION}" && ! -L "${BENCHMARK_SELECTION}" ]] || \
    fail "sealed dual-independent-DP1 benchmark selection is required"
  "${FOMC_PYTHON_BIN}" -u -m "${SELECTOR_MODULE}" validate \
    --manifest "${BENCHMARK_SELECTION}" \
    --cohort "${COHORT_MANIFEST}" \
    --cohort-sha256 "${COHORT_SHA256}"
  local selected
  selected="$(jq -er '.selection.selected_max_num_seqs' "${BENCHMARK_SELECTION}")"
  case "${selected}" in 8|12|16) ;; *) fail "invalid sealed max_num_seqs" ;; esac
  if [[ -n "${REQUESTED_MAX_NUM_SEQS}" && "${REQUESTED_MAX_NUM_SEQS}" != "${selected}" ]]; then
    fail "VLLM_MAX_NUM_SEQS=${REQUESTED_MAX_NUM_SEQS} conflicts with sealed selection ${selected}"
  fi
  VLLM_MAX_NUM_SEQS="${selected}"
  export VLLM_MAX_NUM_SEQS
  echo "[$(timestamp)] selected_max_num_seqs=${VLLM_MAX_NUM_SEQS}"
}

run_or_validate_benchmark() {
  local candidate
  for candidate in 8 12 16; do
    run_or_validate_model \
      chk1 \
      "${BENCHMARK_ROOT}/max_num_seqs_${candidate}" \
      infrastructure_smoke \
      "${candidate}" \
      false
  done
  if [[ ! -f "${BENCHMARK_SELECTION}" ]]; then
    [[ ! -e "${BENCHMARK_SELECTION}" ]] || fail "unsafe benchmark selection path"
    "${FOMC_PYTHON_BIN}" -u -m "${SELECTOR_MODULE}" select \
      --cohort "${COHORT_MANIFEST}" \
      --cohort-sha256 "${COHORT_SHA256}" \
      --manifest-8 "${BENCHMARK_ROOT}/max_num_seqs_8/chk1/manifest.json" \
      --manifest-12 "${BENCHMARK_ROOT}/max_num_seqs_12/chk1/manifest.json" \
      --manifest-16 "${BENCHMARK_ROOT}/max_num_seqs_16/chk1/manifest.json" \
      --output "${BENCHMARK_SELECTION}"
  fi
  validate_benchmark_selection
}

seal_or_validate_suite() {
  local suite_root="$1"
  local scope="$2"
  local smoke_binding=()
  if [[ "${scope}" == "formal_merged_panel" ]]; then
    smoke_binding=(--smoke-suite-manifest "${SMOKE_ROOT}/manifest.json")
  fi
  if [[ ! -f "${suite_root}/manifest.json" ]]; then
    "${FOMC_PYTHON_BIN}" -u -m "${SUITE_MODULE}" seal \
      --cohort "${COHORT_MANIFEST}" \
      --cohort-sha256 "${COHORT_SHA256}" \
      --output-dir "${suite_root}" \
      --scope "${scope}" \
      --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
      --benchmark-selection "${BENCHMARK_SELECTION}" \
      "${smoke_binding[@]}"
  fi
  "${FOMC_PYTHON_BIN}" -u -m "${SUITE_MODULE}" validate \
    --manifest "${suite_root}/manifest.json" \
    --cohort "${COHORT_MANIFEST}" \
    --cohort-sha256 "${COHORT_SHA256}" \
    --scope "${scope}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
}

run_suite() {
  local suite_root="$1"
  local scope="$2"
  local model_id
  for model_id in "${MODEL_ORDER[@]}"; do
    run_or_validate_model \
      "${model_id}" "${suite_root}" "${scope}" "${VLLM_MAX_NUM_SEQS}" true
  done
  seal_or_validate_suite "${suite_root}" "${scope}"
}

validate_smoke() {
  [[ -f "${SMOKE_ROOT}/manifest.json" ]] || fail "official three-model smoke is required"
  seal_or_validate_suite "${SMOKE_ROOT}" infrastructure_smoke
  jq -e '
    .coverage.models == 3
    and .coverage.rows_per_model == 40
    and .coverage.rows_per_shard == 20
    and .coverage.total_rows == 120
  ' "${SMOKE_ROOT}/manifest.json" >/dev/null || fail "smoke coverage must be 3x40 with 20/20 shards"
}

assemble_or_validate() {
  [[ -f "${FORMAL_ROOT}/manifest.json" ]] || fail "sealed formal suite is required"
  seal_or_validate_suite "${FORMAL_ROOT}" formal_merged_panel
  if [[ -f "${DOCUMENT_ROOT}/manifest.json" ]]; then
    "${FOMC_PYTHON_BIN}" -u -m "${ASSEMBLER_MODULE}" validate \
      --manifest "${DOCUMENT_ROOT}/manifest.json"
  else
    [[ ! -e "${DOCUMENT_ROOT}" ]] || fail "document root is partial/unsafe"
    "${FOMC_PYTHON_BIN}" -u -m "${ASSEMBLER_MODULE}" assemble \
      --cohort-manifest "${COHORT_MANIFEST}" \
      --cohort-manifest-sha256 "${COHORT_SHA256}" \
      --suite-manifest "${FORMAL_ROOT}/manifest.json" \
      --output-dir "${DOCUMENT_ROOT}" \
      --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
    "${FOMC_PYTHON_BIN}" -u -m "${ASSEMBLER_MODULE}" validate \
      --manifest "${DOCUMENT_ROOT}/manifest.json"
  fi
}

echo "[$(timestamp)] pipeline_start mode=${MODE}"
echo "repo_root=${REPO_ROOT}"
echo "gpu_contract=two_independent_single_gpu_vllm_engines physical_gpus=0,1 data_parallel_size_per_engine=1 tensor_parallel_size_per_engine=1 gpu_memory_utilization=0.95 k=5 max_num_seqs=${REQUESTED_MAX_NUM_SEQS:-pending_benchmark}"
prepare_or_validate

if [[ "${MODE}" == "prepare" ]]; then
  echo "[$(timestamp)] preparation_validation_complete"
  exit 0
fi

require_topology_remediation_receipt

if [[ "${MODE}" == "benchmark" ]]; then
  run_or_validate_benchmark
  echo "[$(timestamp)] pipeline_complete mode=${MODE}"
  exit 0
fi

if [[ "${MODE}" == "all" ]]; then
  run_or_validate_benchmark
else
  validate_benchmark_selection
fi

case "${MODE}" in
  smoke)
    run_suite "${SMOKE_ROOT}" infrastructure_smoke
    ;;
  formal)
    validate_smoke
    run_suite "${FORMAL_ROOT}" formal_merged_panel
    ;;
  assemble)
    validate_smoke
    assemble_or_validate
    ;;
  all)
    run_suite "${SMOKE_ROOT}" infrastructure_smoke
    validate_smoke
    run_suite "${FORMAL_ROOT}" formal_merged_panel
    assemble_or_validate
    ;;
  validate)
    validate_smoke
    seal_or_validate_suite "${FORMAL_ROOT}" formal_merged_panel
    assemble_or_validate
    ;;
esac

echo "[$(timestamp)] pipeline_complete mode=${MODE}"
