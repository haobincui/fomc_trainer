#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

STAGE=${1:-help}
ARTIFACT_ARGUMENT=${2:-}
GPU_ARGUMENT=${3:-}

# vLLM is present in llama_factory; semantic scoring and validation use the
# user-requested fomc_trainer environment by default.
GENERATION_PYTHON=${CHECKPOINT_EVAL_GENERATION_PYTHON:-/home/haobin_cui/.conda/envs/llama_factory/bin/python}
SCORING_PYTHON=${CHECKPOINT_EVAL_SCORING_PYTHON:-/home/haobin_cui/.conda/envs/fomc_trainer/bin/python}
CONFIG=${CHECKPOINT_EVAL_CONFIG:-${REPO_ROOT}/configs/main/checkpoint_generation_eval_11.json}
DATASET_ROOT=${CHECKPOINT_EVAL_DATASET_ROOT:-${REPO_ROOT}/output/evaluation/main/checkpoint_generation/checkpoint_eval_11_v1/dataset}
PROMPTS=${DATASET_ROOT}/prompts.jsonl
REFERENCES=${DATASET_ROOT}/references.jsonl
TEST_MANIFEST=${DATASET_ROOT}/test_manifest.json
SEMANTIC_MANIFEST=${CHECKPOINT_EVAL_SEMANTIC_MANIFEST:-${REPO_ROOT}/configs/main/checkpoint_eval_semantic_models.json}
EVIDENCE_ROOT=${CHECKPOINT_EVAL_EVIDENCE_ROOT:-${REPO_ROOT}/docs/summary/20260810T124500Z/chk1_cp200_merge_for_chk2}
CHECKPOINT_MANIFEST=${CHECKPOINT_EVAL_CHECKPOINT_MANIFEST:-${EVIDENCE_ROOT}/evaluation_checkpoint_manifest.json}
LINEAGE_MANIFEST=${CHECKPOINT_EVAL_LINEAGE_MANIFEST:-${EVIDENCE_ROOT}/evaluation_lineage_manifest.json}
EXACT_MERGE_EVIDENCE=${CHECKPOINT_EVAL_EXACT_MERGE_EVIDENCE:-${EVIDENCE_ROOT}/chk1_cp200_exact_merge_lineage.json}
RUN_ROOT=${CHECKPOINT_EVAL_RUN_ROOT:-${REPO_ROOT}/output/evaluation/main/checkpoint_generation/retrain_v2_chk0_chk1_cp200_20260810_v1}
GENERATION_DIR=${RUN_ROOT}/generations
SCORE_DIR=${RUN_ROOT}/scores
LENGTH_TOLERANT_SCORE_DIR=${RUN_ROOT}/scores_length_tolerant
BATCH_SIZE=${CHECKPOINT_EVAL_BATCH_SIZE:-2}
SEMANTIC_GPU_INDEX=${CHECKPOINT_EVAL_SEMANTIC_GPU_INDEX:-0}

ARTIFACT_IDS=(
  eval-chk0-base
  eval-chk1-clean-v2-lr1e6-cp200
)

usage() {
  cat <<'EOF'
usage:
  run/eval_retrain_v2_chk0_chk1_generation.sh generate ARTIFACT_ID GPU_INDEX
  run/eval_retrain_v2_chk0_chk1_generation.sh reseal-legacy ARTIFACT_ID
  run/eval_retrain_v2_chk0_chk1_generation.sh score
  run/eval_retrain_v2_chk0_chk1_generation.sh score-length-tolerant
  run/eval_retrain_v2_chk0_chk1_generation.sh summarize OUTPUT_PATH

This versioned runner reuses the frozen 33-row Chapter 2 test. Generation
outputs are immutable and written only to the new chk0/chk1 run directory.
EOF
}

generation_manifest_for_artifact() {
  local artifact=$1
  local primary=${GENERATION_DIR}/${artifact}.manifest.json
  local resealed=${GENERATION_DIR}/${artifact}.manifest.progress-v2.json
  if [[ -f "${primary}" ]] && [[ $(jq -r '.schema_version // ""' "${primary}") == checkpoint-artifact-generation-manifest-v2 ]]; then
    echo "${primary}"
    return 0
  fi
  if [[ -f "${resealed}" ]] && [[ $(jq -r '.schema_version // ""' "${resealed}") == checkpoint-artifact-generation-manifest-v2 ]]; then
    echo "${resealed}"
    return 0
  fi
  echo "Missing progress-bound v2 generation manifest for ${artifact}" >&2
  return 1
}

reseal_legacy() {
  local artifact=$1
  known_artifact "${artifact}" || { echo "Unknown artifact: ${artifact}" >&2; exit 2; }
  validate_frozen_inputs_exist
  require_file "${GENERATION_DIR}/${artifact}.jsonl" "${artifact} generations"
  require_file "${GENERATION_DIR}/${artifact}.manifest.json" "${artifact} legacy manifest"
  "${SCORING_PYTHON}" -m jobs.generation.generate_checkpoint_artifact \
    --checkpoint-manifest "${CHECKPOINT_MANIFEST}" \
    --prompts "${PROMPTS}" \
    --config "${CONFIG}" \
    --artifact-id "${artifact}" \
    --output-dir "${GENERATION_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --reseal-legacy-progress
}

require_dir() {
  local path=$1
  local label=$2
  [[ -d "${path}" ]] || { echo "Missing ${label}: ${path}" >&2; exit 1; }
}

known_artifact() {
  local requested=$1
  local candidate
  for candidate in "${ARTIFACT_IDS[@]}"; do
    [[ "${requested}" == "${candidate}" ]] && return 0
  done
  return 1
}

require_file() {
  local path=$1
  local label=$2
  [[ -f "${path}" ]] || { echo "Missing ${label}: ${path}" >&2; exit 1; }
}

pin_idle_gpu() {
  local index=$1
  [[ "${index}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU index: ${index}" >&2; exit 2; }
  nvidia-smi --id="${index}" --query-gpu=uuid --format=csv,noheader,nounits >/dev/null
  local pids
  pids=$(nvidia-smi --id="${index}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
  if [[ -n "${pids}" && "${CHECKPOINT_EVAL_ALLOW_BUSY_GPU:-0}" != 1 ]]; then
    echo "GPU ${index} is busy (PIDs: ${pids})" >&2
    exit 1
  fi
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="${index}"
}

validate_frozen_inputs_exist() {
  require_file "${CONFIG}" "evaluation config"
  require_file "${PROMPTS}" "frozen prompts"
  require_file "${REFERENCES}" "frozen references"
  require_file "${TEST_MANIFEST}" "frozen test manifest"
  require_file "${SEMANTIC_MANIFEST}" "semantic manifest"
  require_file "${CHECKPOINT_MANIFEST}" "two-artifact checkpoint manifest"
  require_file "${LINEAGE_MANIFEST}" "two-artifact lineage manifest"
  require_file "${EXACT_MERGE_EVIDENCE}" "chk1 exact-merge evidence"
}

generate_one() {
  local artifact=$1
  local gpu=$2
  known_artifact "${artifact}" || { echo "Unknown artifact: ${artifact}" >&2; exit 2; }
  validate_frozen_inputs_exist
  pin_idle_gpu "${gpu}"
  mkdir -p "${GENERATION_DIR}"
  "${GENERATION_PYTHON}" -m jobs.generation.generate_checkpoint_artifact \
    --checkpoint-manifest "${CHECKPOINT_MANIFEST}" \
    --prompts "${PROMPTS}" \
    --config "${CONFIG}" \
    --artifact-id "${artifact}" \
    --output-dir "${GENERATION_DIR}" \
    --batch-size "${BATCH_SIZE}"
}

score_all() {
  local policy=$1
  local destination=$2
  validate_frozen_inputs_exist
  pin_idle_gpu "${SEMANTIC_GPU_INDEX}"
  local args=()
  local artifact
  for artifact in "${ARTIFACT_IDS[@]}"; do
    require_file "${GENERATION_DIR}/${artifact}.jsonl" "${artifact} generations"
    local generation_manifest
    generation_manifest=$(generation_manifest_for_artifact "${artifact}")
    args+=(--generations "${GENERATION_DIR}/${artifact}.jsonl")
    args+=(--generation-manifest "${generation_manifest}")
  done
  "${SCORING_PYTHON}" -m jobs.eval.eval_retrain_v2_chk0_chk1_generation \
    "${args[@]}" \
    --prompts "${PROMPTS}" \
    --references "${REFERENCES}" \
    --config "${CONFIG}" \
    --test-manifest "${TEST_MANIFEST}" \
    --checkpoint-manifest "${CHECKPOINT_MANIFEST}" \
    --lineage-manifest "${LINEAGE_MANIFEST}" \
    --semantic-manifest "${SEMANTIC_MANIFEST}" \
    --exact-merge-evidence "${EXACT_MERGE_EVIDENCE}" \
    --output-dir "${destination}" \
    --repo-root "${REPO_ROOT}" \
    --semantic-device cuda:0 \
    --semantic-batch-size 8 \
    --bootstrap-samples 10000 \
    --bootstrap-seed 20260729 \
    --scoring-policy "${policy}"
}

summarize_results() {
  local output_path=$1
  require_dir "${SCORE_DIR}" "strict score directory"
  require_dir "${LENGTH_TOLERANT_SCORE_DIR}" "length-tolerant score directory"
  require_file "${SCORE_DIR}/audit.json" "strict score audit"
  require_file "${SCORE_DIR}/summary.jsonl" "strict score summary"
  require_file "${SCORE_DIR}/contrasts.jsonl" "strict score contrasts"
  require_file "${LENGTH_TOLERANT_SCORE_DIR}/audit.json" "length-tolerant score audit"
  require_file "${LENGTH_TOLERANT_SCORE_DIR}/summary.jsonl" "length-tolerant score summary"
  require_file "${LENGTH_TOLERANT_SCORE_DIR}/contrasts.jsonl" "length-tolerant score contrasts"
  "${SCORING_PYTHON}" -m jobs.eval.summarize_chk0_chk1_cp200_generation_eval \
    --strict-dir "${SCORE_DIR}" \
    --length-tolerant-dir "${LENGTH_TOLERANT_SCORE_DIR}" \
    --output "${output_path}"
}

case "${STAGE}" in
  generate)
    [[ -n "${ARTIFACT_ARGUMENT}" && -n "${GPU_ARGUMENT}" ]] || { usage >&2; exit 2; }
    generate_one "${ARTIFACT_ARGUMENT}" "${GPU_ARGUMENT}"
    ;;
  reseal-legacy)
    [[ -n "${ARTIFACT_ARGUMENT}" ]] || { usage >&2; exit 2; }
    reseal_legacy "${ARTIFACT_ARGUMENT}"
    ;;
  score)
    score_all strict-final-answer-v2 "${SCORE_DIR}"
    ;;
  score-length-tolerant)
    score_all length-tolerant-open-tags-v1 "${LENGTH_TOLERANT_SCORE_DIR}"
    ;;
  summarize)
    [[ -n "${ARTIFACT_ARGUMENT}" ]] || { usage >&2; exit 2; }
    summarize_results "${ARTIFACT_ARGUMENT}"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
