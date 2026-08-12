#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

STAGE=${1:-help}
OUTPUT_PATH=${2:-}
SCORING_PYTHON=${CHECKPOINT_EVAL_SCORING_PYTHON:-/home/haobin_cui/.conda/envs/fomc_trainer/bin/python}
CONFIG=${CHECKPOINT_EVAL_CONFIG:-${REPO_ROOT}/configs/main/checkpoint_generation_eval_11.json}
DATASET_ROOT=${CHECKPOINT_EVAL_DATASET_ROOT:-${REPO_ROOT}/output/evaluation/main/checkpoint_generation/checkpoint_eval_11_v1/dataset}
PROMPTS=${DATASET_ROOT}/prompts.jsonl
REFERENCES=${DATASET_ROOT}/references.jsonl
TEST_MANIFEST=${DATASET_ROOT}/test_manifest.json
SEMANTIC_MANIFEST=${CHECKPOINT_EVAL_SEMANTIC_MANIFEST:-${REPO_ROOT}/configs/main/checkpoint_eval_semantic_models.json}

# The sealed chk0/chk1 pair remains in its original immutable run/evidence roots.
PAIR_RUN_ROOT=${CHECKPOINT_EVAL_PAIR_RUN_ROOT:-${REPO_ROOT}/output/evaluation/main/checkpoint_generation/retrain_v2_chk0_chk1_cp200_20260810_v1}
PAIR_GENERATION_DIR=${PAIR_RUN_ROOT}/generations
PAIR_EVIDENCE_ROOT=${CHECKPOINT_EVAL_PAIR_EVIDENCE_ROOT:-${REPO_ROOT}/docs/summary/20260810T124500Z/chk1_cp200_merge_for_chk2}
PAIR_CHECKPOINT_MANIFEST=${CHECKPOINT_EVAL_PAIR_CHECKPOINT_MANIFEST:-${PAIR_EVIDENCE_ROOT}/evaluation_checkpoint_manifest.json}
PAIR_LINEAGE_MANIFEST=${CHECKPOINT_EVAL_PAIR_LINEAGE_MANIFEST:-${PAIR_EVIDENCE_ROOT}/evaluation_lineage_manifest.json}
PAIR_EXACT_MERGE_EVIDENCE=${CHECKPOINT_EVAL_PAIR_EXACT_MERGE_EVIDENCE:-${PAIR_EVIDENCE_ROOT}/chk1_cp200_exact_merge_lineage.json}

# The new standalone leg has its own checkpoint manifest and generation output.
RUN_ROOT=${CHECKPOINT_EVAL_THREE_LEG_RUN_ROOT:-${REPO_ROOT}/output/evaluation/main/checkpoint_generation/retrain_v2_chk0_chk1_chk3_cp250_20260810_v1}
GENERATION_DIR=${RUN_ROOT}/generations
MANIFEST_DIR=${CHECKPOINT_EVAL_THREE_LEG_MANIFEST_DIR:-${REPO_ROOT}/docs/summary/20260810T223534Z/chk3_cp250_generation_comparison}
CHK3_CHECKPOINT_MANIFEST=${CHECKPOINT_EVAL_CHK3_CHECKPOINT_MANIFEST:-${MANIFEST_DIR}/chk3_checkpoint_manifest.json}
THREE_LEG_LINEAGE_MANIFEST=${CHECKPOINT_EVAL_THREE_LEG_LINEAGE_MANIFEST:-${MANIFEST_DIR}/three_leg_lineage_manifest.json}
SCORE_DIR=${RUN_ROOT}/scores
LENGTH_TOLERANT_SCORE_DIR=${RUN_ROOT}/scores_length_tolerant
SEMANTIC_GPU_INDEX=${CHECKPOINT_EVAL_SEMANTIC_GPU_INDEX:-0}

PAIR_ARTIFACT_IDS=(
  eval-chk0-base
  eval-chk1-clean-v2-lr1e6-cp200
)
CHK3_ARTIFACT_ID=eval-chk3-direct-chk1-cp200-sft-cp250

usage() {
  cat <<'EOF'
usage:
  run/eval_retrain_v2_chk0_chk1_chk3_generation.sh score
  run/eval_retrain_v2_chk0_chk1_chk3_generation.sh score-length-tolerant
  run/eval_retrain_v2_chk0_chk1_chk3_generation.sh summarize OUTPUT_PATH

This runner performs scoring only.  It reuses the original sealed chk0/chk1
generation files and their progress-v2 manifests, then adds the separately
sealed chk3 checkpoint-250 generation leg.  GPU 0 is the default semantic
scoring device; override CHECKPOINT_EVAL_SEMANTIC_GPU_INDEX only explicitly.
EOF
}

require_file() {
  local path=$1
  local label=$2
  [[ -f "${path}" ]] || { echo "Missing ${label}: ${path}" >&2; exit 1; }
}

require_dir() {
  local path=$1
  local label=$2
  [[ -d "${path}" ]] || { echo "Missing ${label}: ${path}" >&2; exit 1; }
}

pin_gpu() {
  local index=$1
  [[ "${index}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU index: ${index}" >&2; exit 2; }
  nvidia-smi --id="${index}" --query-gpu=uuid --format=csv,noheader,nounits >/dev/null
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="${index}"
}

progress_v2_manifest() {
  local directory=$1
  local artifact=$2
  local primary=${directory}/${artifact}.manifest.json
  local resealed=${directory}/${artifact}.manifest.progress-v2.json
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

validate_inputs_exist() {
  require_file "${CONFIG}" "evaluation config"
  require_file "${PROMPTS}" "frozen prompts"
  require_file "${REFERENCES}" "frozen references"
  require_file "${TEST_MANIFEST}" "frozen test manifest"
  require_file "${SEMANTIC_MANIFEST}" "semantic manifest"
  require_file "${PAIR_CHECKPOINT_MANIFEST}" "sealed pair checkpoint manifest"
  require_file "${PAIR_LINEAGE_MANIFEST}" "sealed pair lineage manifest"
  require_file "${PAIR_EXACT_MERGE_EVIDENCE}" "chk1 exact-merge evidence"
  require_file "${CHK3_CHECKPOINT_MANIFEST}" "sealed chk3 single-leg checkpoint manifest"
  require_file "${THREE_LEG_LINEAGE_MANIFEST}" "sealed combined three-leg lineage manifest"
}

score_all() {
  local policy=$1
  local destination=$2
  validate_inputs_exist
  pin_gpu "${SEMANTIC_GPU_INDEX}"
  local args=()
  local artifact
  for artifact in "${PAIR_ARTIFACT_IDS[@]}"; do
    local generation=${PAIR_GENERATION_DIR}/${artifact}.jsonl
    require_file "${generation}" "${artifact} sealed generations"
    local manifest
    manifest=$(progress_v2_manifest "${PAIR_GENERATION_DIR}" "${artifact}")
    args+=(--generations "${generation}")
    args+=(--generation-manifest "${manifest}")
  done
  local chk3_generation=${GENERATION_DIR}/${CHK3_ARTIFACT_ID}.jsonl
  require_file "${chk3_generation}" "${CHK3_ARTIFACT_ID} generations"
  local chk3_generation_manifest
  chk3_generation_manifest=$(progress_v2_manifest "${GENERATION_DIR}" "${CHK3_ARTIFACT_ID}")
  args+=(--generations "${chk3_generation}")
  args+=(--generation-manifest "${chk3_generation_manifest}")

  "${SCORING_PYTHON}" -m jobs.eval.eval_retrain_v2_chk0_chk1_chk3_generation \
    "${args[@]}" \
    --prompts "${PROMPTS}" \
    --references "${REFERENCES}" \
    --config "${CONFIG}" \
    --test-manifest "${TEST_MANIFEST}" \
    --pair-checkpoint-manifest "${PAIR_CHECKPOINT_MANIFEST}" \
    --pair-lineage-manifest "${PAIR_LINEAGE_MANIFEST}" \
    --semantic-manifest "${SEMANTIC_MANIFEST}" \
    --pair-exact-merge-evidence "${PAIR_EXACT_MERGE_EVIDENCE}" \
    --chk3-checkpoint-manifest "${CHK3_CHECKPOINT_MANIFEST}" \
    --three-leg-lineage-manifest "${THREE_LEG_LINEAGE_MANIFEST}" \
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
  require_file "${SCORE_DIR}/input_validation.json" "strict input validation"
  require_file "${LENGTH_TOLERANT_SCORE_DIR}/audit.json" "length-tolerant score audit"
  require_file "${LENGTH_TOLERANT_SCORE_DIR}/input_validation.json" "length-tolerant input validation"
  "${SCORING_PYTHON}" -m jobs.eval.summarize_chk0_chk1_chk3_cp250_generation_eval \
    --strict-dir "${SCORE_DIR}" \
    --length-tolerant-dir "${LENGTH_TOLERANT_SCORE_DIR}" \
    --output "${output_path}"
}

case "${STAGE}" in
  score)
    score_all strict-final-answer-v2 "${SCORE_DIR}"
    ;;
  score-length-tolerant)
    score_all length-tolerant-open-tags-v1 "${LENGTH_TOLERANT_SCORE_DIR}"
    ;;
  summarize)
    [[ -n "${OUTPUT_PATH}" ]] || { usage >&2; exit 2; }
    summarize_results "${OUTPUT_PATH}"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
