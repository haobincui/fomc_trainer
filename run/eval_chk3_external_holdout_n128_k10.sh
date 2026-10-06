#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/haobin_cui/research_files_space_2/fomc_trainer"
PYTHON_BIN="/home/haobin_cui/.conda/envs/fomc_trainer/bin/python"
RUN_ROOT="${REPO_ROOT}/output/evaluation/main/chk3_external_holdout_1993_2008_n128_k10_20260812_v1"
SAMPLE_MANIFEST="${RUN_ROOT}/inputs/samples_n128_k10.json"
SAMPLE_SHA256="665af62bab2b64d5477e04d76632ac4017c570bfb95a01bd04ad70f85d976f0a"
GENERATION_ROOT="${RUN_ROOT}/formal_generation_v1"
SCORE_ROOT="${RUN_ROOT}/six_metric_bootstrap_v1"
SEMANTIC_MANIFEST="${REPO_ROOT}/configs/main/checkpoint_eval_semantic_models.json"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

cd "${REPO_ROOT}"

generation_args=(
  -u -m jobs.eval.eval_chk3_external_holdout_stochastic_k10 run-suite
  --sample-manifest "${SAMPLE_MANIFEST}"
  --sample-manifest-sha256 "${SAMPLE_SHA256}"
  --output-dir "${GENERATION_ROOT}"
  --gpu-wait-timeout-seconds 172800
  --gpu-poll-seconds 30
)

if [[ -f "${GENERATION_ROOT}/manifest.json" ]]; then
  echo '{"status":"generation_already_complete_validating"}'
  "${PYTHON_BIN}" -u -m jobs.eval.eval_chk3_external_holdout_stochastic_k10 \
    validate-suite \
    --manifest "${GENERATION_ROOT}/manifest.json" \
    --sample-manifest "${SAMPLE_MANIFEST}" \
    --sample-manifest-sha256 "${SAMPLE_SHA256}" \
    --scope formal_full_test
elif [[ -e "${GENERATION_ROOT}" ]]; then
  echo '{"status":"resuming_generation"}'
  "${PYTHON_BIN}" "${generation_args[@]}" --resume
else
  echo '{"status":"starting_generation"}'
  "${PYTHON_BIN}" "${generation_args[@]}"
fi

suite_sha256="$(sha256sum "${GENERATION_ROOT}/manifest.json" | awk '{print $1}')"
echo "{\"status\":\"generation_complete\",\"suite_sha256\":\"${suite_sha256}\"}"

if [[ -f "${SCORE_ROOT}/manifest.json" ]]; then
  echo '{"status":"scoring_already_complete"}'
else
  "${PYTHON_BIN}" -u -m jobs.eval.score_chk3_external_holdout_stochastic_k10 \
    --suite-manifest "${GENERATION_ROOT}/manifest.json" \
    --suite-sha256 "${suite_sha256}" \
    --sample-manifest "${SAMPLE_MANIFEST}" \
    --sample-sha256 "${SAMPLE_SHA256}" \
    --semantic-manifest "${SEMANTIC_MANIFEST}" \
    --output-dir "${SCORE_ROOT}" \
    --semantic-batch-size 8
fi

echo '{"status":"generation_and_scoring_complete"}'
