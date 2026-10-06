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
FAILED_BENCHMARK_ROOT_V1="${RUN_ROOT}/benchmark_max_num_seqs_chk1_core8_k5"
BENCHMARK_ROOT="${RUN_ROOT}/benchmark_max_num_seqs_chk1_core8_k5_v2"
BENCHMARK_SELECTION="${BENCHMARK_ROOT}/selection.v1.json"
SMOKE_ROOT="${RUN_ROOT}/generation_smoke_core8_k5_three_models"
FORMAL_ROOT="${RUN_ROOT}/generation_formal_n2048_k5_three_models"
DOCUMENT_ROOT="${RUN_ROOT}/meeting_documents_n3840"
MIGRATION_RECEIPT="${RUN_ROOT}/migration/nf4_k10_superseded_receipt.json"
REMEDIATION_RECEIPT="${RUN_ROOT}/migration/vllm_v1_async_output_remediation_receipt.json"
FRESH_ROOT_ADDENDUM_RECEIPT="${RUN_ROOT}/migration/vllm_v1_async_output_fresh_root_addendum_receipt.json"
PIPELINE_LOG="${RUN_ROOT}/pipeline_vllm_k5_20260816.log"
PIPELINE_LOCK="/tmp/fomc_trainer_chk3_beta_core8_merged_vllm_k5_pipeline.lock"
GPU_WAIT_TIMEOUT_SECONDS="${GPU_WAIT_TIMEOUT_SECONDS:-172800}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
REQUESTED_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
VLLM_MAX_NUM_SEQS=""
MODEL_ORDER=(chk1 chk3 chk0)

mkdir -p "${RUN_ROOT}"
exec 9>"${PIPELINE_LOCK}"
if ! flock -n 9; then
  echo "another vLLM K=5 pipeline holds ${PIPELINE_LOCK}" >&2
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

require_executables() {
  [[ -x "${FOMC_PYTHON_BIN}" ]] || fail "missing FOMC Python: ${FOMC_PYTHON_BIN}"
  [[ -x "${VLLM_PYTHON_BIN}" ]] || fail "missing vLLM Python: ${VLLM_PYTHON_BIN}"
  command -v jq >/dev/null 2>&1 || fail "jq is required"
  command -v sha256sum >/dev/null 2>&1 || fail "sha256sum is required"
}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_DP_MASTER_IP=127.0.0.1
export VLLM_DP_MASTER_PORT=0

cd "${REPO_ROOT}"
require_executables
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
  if [[ ! -e "${PREPARATION_MANIFEST}" && ! -e "${COHORT_MANIFEST}" ]]; then
    if [[ -e "${PREP_ROOT}" ]]; then
      fail "preparation directory exists without its sealed artifacts: ${PREP_ROOT}"
    fi
    echo "[$(timestamp)] preparing exact K=5 prompt-token cohort with FOMC tokenizer"
    "${FOMC_PYTHON_BIN}" -u -m \
      jobs.eval.prepare_chk3_beta_core8_merged_vllm_k5 \
      --source-sample-manifest "${SOURCE_SAMPLE_MANIFEST}" \
      --source-sample-manifest-sha256 "${SOURCE_SAMPLE_MANIFEST_SHA256}" \
      --pre-release "${PRE_RELEASE}" \
      --pre-release-sha256 "${PRE_RELEASE_SHA256}" \
      --post-release "${POST_RELEASE}" \
      --post-release-sha256 "${POST_RELEASE_SHA256}" \
      --output-dir "${PREP_ROOT}"
  fi
  [[ -f "${PREPARATION_MANIFEST}" && ! -L "${PREPARATION_MANIFEST}" ]] || \
    fail "sealed preparation manifest is missing"
  [[ -f "${COHORT_MANIFEST}" && ! -L "${COHORT_MANIFEST}" ]] || \
    fail "sealed cohort manifest is missing"
  jq -e \
    --arg source_sha "${SOURCE_SAMPLE_MANIFEST_SHA256}" \
    --arg pre_sha "${PRE_RELEASE_SHA256}" \
    --arg post_sha "${POST_RELEASE_SHA256}" '
      .status == "complete"
      and .immutable == true
      and .evaluation_id == "chk3-beta-core8-merged-1993-2025-n2048-vllm-k5-v1"
      and .source_sample_manifest.sha256 == $source_sha
      and .harmonized_source_releases.pre2009_external.sha256 == $pre_sha
      and .harmonized_source_releases.post2008_chk3_release.sha256 == $post_sha
      and .generation_design.backend == "vllm-async-engine-v1-continuous-batching"
      and .generation_design.models == ["chk1", "chk3", "chk0"]
      and .generation_design.meetings == 256
      and .generation_design.prompts == 2048
      and .generation_design.replicates == 5
      and .generation_design.rows_per_model == 10240
      and .generation_design.total_rows == 30720
      and .generation_design.replicate_seeds == [20260811,21260811,22260811,23260811,24260811]
      and .token_ledger.full_prompt_token_ids_persisted == true
      and .token_ledger.vllm_runtime_chat_templating == false
    ' "${COHORT_MANIFEST}" >/dev/null || fail "K=5 cohort contract validation failed"
  jq -e '
      .status == "complete"
      and .coverage.meetings == 256
      and .coverage.prompts == 2048
      and .coverage.replicates == 5
      and .coverage.rows_per_model == 10240
      and .coverage.total_generation_rows == 30720
      and .validation.exact_prompt_token_ids_persisted == true
      and .validation.prompt_token_count_mismatches == 0
    ' "${PREPARATION_MANIFEST}" >/dev/null || \
    fail "K=5 preparation coverage validation failed"
  COHORT_SHA256="$(sha256_of "${COHORT_MANIFEST}")"
  [[ "${COHORT_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid cohort SHA-256"
  export COHORT_SHA256
  echo "[$(timestamp)] cohort_sha256=${COHORT_SHA256}"
}

require_migration_receipt() {
  [[ -f "${MIGRATION_RECEIPT}" && ! -L "${MIGRATION_RECEIPT}" ]] || \
    fail "validated NF4/K10 migration receipt is required before GPU work"
  "${FOMC_PYTHON_BIN}" -u -m jobs.eval.migrate_chk3_beta_k10_to_vllm_k5 \
    validate-receipt --receipt "${MIGRATION_RECEIPT}"
  jq -e '
    .new_suite.data_parallel_size == 2
    and .new_suite.tensor_parallel_size == 1
    and .new_suite.pipeline_parallel_size == 1
    and .new_suite.physical_gpu_indices == [0, 1]
    and .new_suite.full_model_replicas == 2
    and .new_suite.gpu_memory_utilization == 0.95
    and .new_suite.enforce_eager == true
    and .new_suite.cuda_graphs == false
    and .new_suite.max_num_seqs_candidates == [8, 12, 16]
    and .new_suite.max_num_batched_tokens == 4096
    and .new_suite.absolute_chunk_size == 40
    and .new_suite.weight_precision == "bfloat16"
    and .new_suite.dtype == "bfloat16"
    and .new_suite.quantization == null
    and .new_suite.replicates == 5
    and .new_suite.expected_total_rows == 30720
  ' "${MIGRATION_RECEIPT}" >/dev/null || fail "dual-GPU migration receipt drift"
}

require_async_output_remediation_receipt() {
  [[ -f "${REMEDIATION_RECEIPT}" && ! -L "${REMEDIATION_RECEIPT}" ]] || \
    fail "sealed vLLM V1 async-output remediation receipt is required"
}

require_fresh_root_addendum_receipt() {
  [[ -f "${FRESH_ROOT_ADDENDUM_RECEIPT}" && ! -L "${FRESH_ROOT_ADDENDUM_RECEIPT}" ]] || \
    fail "sealed vLLM V1 fresh-root addendum receipt is required"
  "${FOMC_PYTHON_BIN}" -u -m \
    jobs.eval.seal_chk3_beta_core8_vllm_k5_fresh_root_addendum \
    validate
}

require_vllm_bf16_contract() {
  echo "[$(timestamp)] validating vLLM/BF16/DP2/GPU0+GPU1 runtime contract"
  "${VLLM_PYTHON_BIN}" - <<'PY'
import os
import torch
import vllm

assert os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1"
assert os.environ.get("VLLM_USE_V1") == "1"
assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0"
assert os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") == "0"
assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn"
assert os.environ.get("VLLM_DP_MASTER_IP") == "127.0.0.1"
assert os.environ.get("VLLM_DP_MASTER_PORT") == "0"
assert vllm.__version__ == "0.8.5.post1", vllm.__version__
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2
for index in range(2):
    with torch.cuda.device(index):
        assert torch.cuda.is_bf16_supported()
print(
    "vllm_contract=passed "
    f"vllm={vllm.__version__} torch={torch.__version__} "
    f"logical_cuda_devices={torch.cuda.device_count()} "
    "data_parallel_size=2 tensor_parallel_size=1 pipeline_parallel_size=1 "
    "enforce_eager=true dtype=bfloat16 quantization=None gpu_memory_utilization=0.95",
    flush=True,
)
PY
}

run_or_validate_model() {
  local model_id="$1"
  local suite_root="$2"
  local scope="$3"
  local max_num_seqs="${4:-${VLLM_MAX_NUM_SEQS}}"
  local allow_resume="${5:-true}"
  local smoke_flag=()
  local output_dir="${suite_root}/${model_id}"
  case "${max_num_seqs}" in
    8|12|16) ;;
    *) fail "model launch has no sealed max_num_seqs selection" ;;
  esac
  if [[ "${scope}" == "infrastructure_smoke" ]]; then
    smoke_flag=(--smoke)
  fi
  if [[ -f "${output_dir}/manifest.json" ]]; then
    echo "[$(timestamp)] validating sealed ${scope} ${model_id} run"
    "${VLLM_PYTHON_BIN}" -u -m jobs.eval.eval_chk3_beta_core8_merged_vllm_k5 \
      validate-run \
      --manifest "${output_dir}/manifest.json" \
      --cohort "${COHORT_MANIFEST}" \
      --cohort-sha256 "${COHORT_SHA256}" \
      --model-id "${model_id}" \
      --scope "${scope}" \
      --max-num-seqs "${max_num_seqs}"
    return
  fi
  local resume_flag=()
  if [[ -d "${output_dir}" ]]; then
    if [[ "${allow_resume}" != "true" ]]; then
      fail "speed benchmark candidate is partial and cannot be resumed: ${output_dir}"
    fi
    resume_flag=(--resume)
  fi
  echo "[$(timestamp)] running ${scope} ${model_id} with vLLM continuous batching"
  "${VLLM_PYTHON_BIN}" -u -m jobs.eval.eval_chk3_beta_core8_merged_vllm_k5 \
    run-model \
    --model-id "${model_id}" \
    --cohort "${COHORT_MANIFEST}" \
    --cohort-sha256 "${COHORT_SHA256}" \
    --output-dir "${output_dir}" \
    --gpu-wait-timeout-seconds "${GPU_WAIT_TIMEOUT_SECONDS}" \
    --gpu-poll-seconds "${GPU_POLL_SECONDS}" \
    --max-num-seqs "${max_num_seqs}" \
    "${smoke_flag[@]}" \
    "${resume_flag[@]}"
}

validate_benchmark_selection() {
  [[ -f "${BENCHMARK_SELECTION}" && ! -L "${BENCHMARK_SELECTION}" ]] || \
    fail "sealed 8/12/16 max_num_seqs benchmark selection is required"
  "${FOMC_PYTHON_BIN}" -u -m \
    jobs.eval.select_chk3_beta_core8_vllm_k5_benchmark \
    validate \
    --manifest "${BENCHMARK_SELECTION}" \
    --cohort "${COHORT_MANIFEST}" \
    --cohort-sha256 "${COHORT_SHA256}"
  local selected
  selected="$(jq -er '.selection.selected_max_num_seqs' "${BENCHMARK_SELECTION}")"
  case "${selected}" in
    8|12|16) ;;
    *) fail "benchmark selection contains an invalid max_num_seqs" ;;
  esac
  if [[ -n "${REQUESTED_MAX_NUM_SEQS}" && "${REQUESTED_MAX_NUM_SEQS}" != "${selected}" ]]; then
    fail "VLLM_MAX_NUM_SEQS=${REQUESTED_MAX_NUM_SEQS} conflicts with sealed benchmark selection ${selected}"
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
    if [[ -e "${BENCHMARK_SELECTION}" ]]; then
      fail "benchmark selection path exists but is not a regular file"
    fi
    "${FOMC_PYTHON_BIN}" -u -m \
      jobs.eval.select_chk3_beta_core8_vllm_k5_benchmark \
      select \
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
  if [[ ! -f "${suite_root}/manifest.json" ]]; then
    "${VLLM_PYTHON_BIN}" -u -m jobs.eval.eval_chk3_beta_core8_merged_vllm_k5 \
      seal-suite \
      --cohort "${COHORT_MANIFEST}" \
      --cohort-sha256 "${COHORT_SHA256}" \
      --output-dir "${suite_root}" \
      --scope "${scope}" \
      --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
  fi
  "${VLLM_PYTHON_BIN}" -u -m jobs.eval.eval_chk3_beta_core8_merged_vllm_k5 \
    validate-suite \
    --manifest "${suite_root}/manifest.json" \
    --cohort "${COHORT_MANIFEST}" \
    --cohort-sha256 "${COHORT_SHA256}" \
    --scope "${scope}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
}

run_suite() {
  local suite_root="$1"
  local scope="$2"
  for model_id in "${MODEL_ORDER[@]}"; do
    run_or_validate_model "${model_id}" "${suite_root}" "${scope}"
  done
  seal_or_validate_suite "${suite_root}" "${scope}"
}

validate_smoke() {
  [[ -f "${SMOKE_ROOT}/manifest.json" ]] || fail "sealed Core8 smoke is required"
  seal_or_validate_suite "${SMOKE_ROOT}" infrastructure_smoke
  jq -e '
    .evaluation_scope == "infrastructure_smoke"
    and .coverage.models == 3
    and .coverage.rows_per_model == 40
    and .coverage.total_rows == 120
  ' "${SMOKE_ROOT}/manifest.json" >/dev/null || \
    fail "smoke coverage is not 3 x (Core8 x K5)"
}

assemble_or_validate() {
  [[ -f "${FORMAL_ROOT}/manifest.json" ]] || fail "sealed formal suite is required"
  seal_or_validate_suite "${FORMAL_ROOT}" formal_merged_panel
  if [[ -f "${DOCUMENT_ROOT}/manifest.json" ]]; then
    "${FOMC_PYTHON_BIN}" -u -m \
      jobs.eval.assemble_chk3_beta_core8_meeting_documents_vllm_k5 \
      validate --manifest "${DOCUMENT_ROOT}/manifest.json"
  else
    if [[ -e "${DOCUMENT_ROOT}" ]]; then
      fail "meeting-document directory exists without a sealed manifest"
    fi
    "${FOMC_PYTHON_BIN}" -u -m \
      jobs.eval.assemble_chk3_beta_core8_meeting_documents_vllm_k5 \
      assemble \
      --cohort-manifest "${COHORT_MANIFEST}" \
      --cohort-manifest-sha256 "${COHORT_SHA256}" \
      --suite-manifest "${FORMAL_ROOT}/manifest.json" \
      --output-dir "${DOCUMENT_ROOT}" \
      --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
    "${FOMC_PYTHON_BIN}" -u -m \
      jobs.eval.assemble_chk3_beta_core8_meeting_documents_vllm_k5 \
      validate --manifest "${DOCUMENT_ROOT}/manifest.json"
  fi
}

echo "[$(timestamp)] pipeline_start mode=${MODE}"
echo "repo_root=${REPO_ROOT}"
echo "fomc_python=${FOMC_PYTHON_BIN}"
echo "vllm_python=${VLLM_PYTHON_BIN}"
echo "gpu_contract=physical_gpu0_and_gpu1 backend=vLLM_continuous_batching data_parallel=2 tensor_parallel=1 weights=BF16 quantization=None gpu_memory_utilization=0.95 enforce_eager=true k=5 max_num_seqs=${REQUESTED_MAX_NUM_SEQS:-pending_sealed_benchmark_selection}"
prepare_or_validate

if [[ "${MODE}" == "prepare" ]]; then
  echo "[$(timestamp)] preparation_complete"
  exit 0
fi

require_migration_receipt
require_async_output_remediation_receipt
require_fresh_root_addendum_receipt
require_vllm_bf16_contract

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
    assemble_or_validate
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
