#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

STAGE=${1:-help}
ARTIFACT_ARGUMENT=${2:-}
GPU_ARGUMENT=${3:-}

MODEL_PYTHON=${CHECKPOINT_EVAL_MODEL_PYTHON:-/home/haobin_cui/.conda/envs/llama_factory/bin/python}
CONFIG=${CHECKPOINT_EVAL_CONFIG:-${REPO_ROOT}/configs/main/checkpoint_generation_eval_11.json}
DATASET_ROOT=${CHECKPOINT_EVAL_DATASET_ROOT:-${REPO_ROOT}/output/evaluation/main/checkpoint_generation/checkpoint_eval_11_v1/dataset}
PROMPTS=${DATASET_ROOT}/prompts.jsonl
REFERENCES=${DATASET_ROOT}/references.jsonl
TEST_MANIFEST=${DATASET_ROOT}/test_manifest.json
SEMANTIC_MANIFEST=${CHECKPOINT_EVAL_SEMANTIC_MANIFEST:-${REPO_ROOT}/configs/main/checkpoint_eval_semantic_models.json}
CHECKPOINT_MANIFEST=${CHECKPOINT_EVAL_CHECKPOINT_MANIFEST:-${REPO_ROOT}/docs/summary/20260809T152718Z/chk2_checkpoint150_merge_eval/checkpoint_manifest.json}
LINEAGE_MANIFEST=${CHECKPOINT_EVAL_LINEAGE_MANIFEST:-${REPO_ROOT}/docs/summary/20260809T152718Z/chk2_checkpoint150_merge_eval/lineage_manifest.json}
RUN_ROOT=${CHECKPOINT_EVAL_RUN_ROOT:-${REPO_ROOT}/output/evaluation/main/checkpoint_generation/retrain_v2_chk0_chk1_chk2_cp150_20260809}
GENERATION_DIR=${RUN_ROOT}/generations
SCORE_DIR=${RUN_ROOT}/scores
LENGTH_TOLERANT_SCORE_DIR=${RUN_ROOT}/scores_length_tolerant
BATCH_SIZE=${CHECKPOINT_EVAL_BATCH_SIZE:-2}
SEMANTIC_GPU_INDEX=${CHECKPOINT_EVAL_SEMANTIC_GPU_INDEX:-1}

ARTIFACT_IDS=(
  eval-chk0-base
  eval-chk1-compressed-sft
  eval-chk2-reward-v3-cp150
)

usage() {
  cat <<'EOF'
usage:
  run/eval_retrain_v2_checkpoint_generation.sh generate ARTIFACT_ID GPU_INDEX
  run/eval_retrain_v2_checkpoint_generation.sh score
  run/eval_retrain_v2_checkpoint_generation.sh score-length-tolerant

The frozen 33-row Chapter 2 common test and its original decoding/metric
settings are reused. Generation outputs are immutable and live in a new run.
EOF
}

known_artifact() {
  local requested=$1
  local candidate
  for candidate in "${ARTIFACT_IDS[@]}"; do
    [[ "${requested}" == "${candidate}" ]] && return 0
  done
  return 1
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

generate_one() {
  local artifact=$1
  local gpu=$2
  known_artifact "${artifact}" || { echo "Unknown artifact: ${artifact}" >&2; exit 2; }
  pin_idle_gpu "${gpu}"
  mkdir -p "${GENERATION_DIR}"
  "${MODEL_PYTHON}" -m jobs.generation.generate_checkpoint_artifact \
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
  pin_idle_gpu "${SEMANTIC_GPU_INDEX}"
  local args=()
  local artifact
  for artifact in "${ARTIFACT_IDS[@]}"; do
    args+=(--generations "${GENERATION_DIR}/${artifact}.jsonl")
    args+=(--generation-manifest "${GENERATION_DIR}/${artifact}.manifest.json")
  done
  "${MODEL_PYTHON}" -m jobs.eval.eval_retrain_v2_checkpoint_generation \
    "${args[@]}" \
    --prompts "${PROMPTS}" \
    --references "${REFERENCES}" \
    --config "${CONFIG}" \
    --test-manifest "${TEST_MANIFEST}" \
    --checkpoint-manifest "${CHECKPOINT_MANIFEST}" \
    --semantic-manifest "${SEMANTIC_MANIFEST}" \
    --lineage-manifest "${LINEAGE_MANIFEST}" \
    --output-dir "${destination}" \
    --repo-root "${REPO_ROOT}" \
    --semantic-device cuda:0 \
    --semantic-batch-size 8 \
    --bootstrap-samples 10000 \
    --bootstrap-seed 20260729 \
    --scoring-policy "${policy}"
}

case "${STAGE}" in
  generate)
    [[ -n "${ARTIFACT_ARGUMENT}" && -n "${GPU_ARGUMENT}" ]] || { usage >&2; exit 2; }
    generate_one "${ARTIFACT_ARGUMENT}" "${GPU_ARGUMENT}"
    ;;
  score)
    score_all strict-final-answer-v2 "${SCORE_DIR}"
    ;;
  score-length-tolerant)
    score_all length-tolerant-open-tags-v1 "${LENGTH_TOLERANT_SCORE_DIR}"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
