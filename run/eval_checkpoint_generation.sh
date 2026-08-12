#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

STAGE=${1:-all}
ARTIFACT_ARGUMENT=${2:-}
GPU_ARGUMENT=${3:-}

CONFIG=${CHECKPOINT_EVAL_CONFIG:-"${REPO_ROOT}/configs/main/checkpoint_generation_eval_11.json"}
POPULATION=${CHECKPOINT_EVAL_POPULATION:-"${REPO_ROOT}/configs/main/loo_population_checkpoint_eval_11.json"}
REGISTRY=${CHECKPOINT_EVAL_REGISTRY:-"${REPO_ROOT}/configs/main/loo_indicator_sources.json"}
ROSTER=${CHECKPOINT_EVAL_ROSTER:-"${REPO_ROOT}/configs/main/leave_one_out_roster.json"}
SEMANTIC_MANIFEST=${CHECKPOINT_EVAL_SEMANTIC_MANIFEST:-"${REPO_ROOT}/configs/main/checkpoint_eval_semantic_models.json"}
CHECKPOINT_MANIFEST=${CHECKPOINT_EVAL_CHECKPOINT_MANIFEST:-"${REPO_ROOT}/output/checkpoints/recovered/llama_sft_synthetic_20250526_2_cp1668_recovered_v1_20260729/checkpoint_manifest.json"}
LINEAGE_EVIDENCE=${CHECKPOINT_EVAL_LINEAGE_EVIDENCE:-"${REPO_ROOT}/docs/summary/20260728T091012Z/eval_analysis_sft_exact_merge_lineage_20260730.json"}
RUN_ROOT=${CHECKPOINT_EVAL_RUN_ROOT:-"${REPO_ROOT}/output/evaluation/main/checkpoint_generation/checkpoint_eval_11_v1"}

SNAPSHOT_DIR="${RUN_ROOT}/source_snapshots"
SNAPSHOT_MANIFEST="${SNAPSHOT_DIR}/snapshot_manifest.json"
LEDGER_DIR="${RUN_ROOT}/ledger"
LEDGER_INPUTS="${LEDGER_DIR}/indicator_inputs.jsonl"
LEDGER_MANIFEST="${LEDGER_DIR}/ledger_manifest.json"
DATASET_DIR="${RUN_ROOT}/dataset"
PROMPTS="${DATASET_DIR}/prompts.jsonl"
REFERENCES="${DATASET_DIR}/references.jsonl"
TEST_MANIFEST="${DATASET_DIR}/test_manifest.json"
GENERATION_DIR="${RUN_ROOT}/generations"
SCORE_DIR="${RUN_ROOT}/scores"
LENGTH_TOLERANT_SCORE_DIR="${RUN_ROOT}/scores_length_tolerant"

if [[ -n "${CHECKPOINT_EVAL_DATA_PYTHON:-}" ]]; then
  DATA_PYTHON=${CHECKPOINT_EVAL_DATA_PYTHON}
else
  DATA_PYTHON=python
fi
if [[ -n "${CHECKPOINT_EVAL_MODEL_PYTHON:-}" ]]; then
  MODEL_PYTHON=${CHECKPOINT_EVAL_MODEL_PYTHON}
elif [[ -x /home/haobin_cui/.conda/envs/llama_factory/bin/python ]]; then
  MODEL_PYTHON=/home/haobin_cui/.conda/envs/llama_factory/bin/python
else
  MODEL_PYTHON=python
fi
BATCH_SIZE=${CHECKPOINT_EVAL_BATCH_SIZE:-2}
DEFAULT_GPU_INDEX=${CHECKPOINT_EVAL_GPU_INDEX:-1}
SEMANTIC_GPU_INDEX=${CHECKPOINT_EVAL_SEMANTIC_GPU_INDEX:-1}

ARTIFACT_IDS=(
  eval-base
  eval-analysis-sft
  eval-legacy-grpo-from-chk0
  eval-minutes-sft-from-chk1
)

usage() {
  cat <<'EOF'
usage:
  run/eval_checkpoint_generation.sh prepare
  run/eval_checkpoint_generation.sh generate ARTIFACT_ID [PHYSICAL_GPU_INDEX]
  run/eval_checkpoint_generation.sh generate-all
  run/eval_checkpoint_generation.sh score
  run/eval_checkpoint_generation.sh score-length-tolerant
  run/eval_checkpoint_generation.sh all

This workflow performs data acquisition, inference, and evaluation only.
It never trains a model. Set CHECKPOINT_EVAL_OFFLINE=1 to require cached
official Minutes HTML during prepare. score-length-tolerant reuses the sealed
generations and writes a separate post-hoc robustness result directory.
EOF
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file does not exist: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Required directory does not exist: $1" >&2
    exit 1
  fi
}

is_known_artifact() {
  local requested=$1
  local candidate
  for candidate in "${ARTIFACT_IDS[@]}"; do
    if [[ "${requested}" == "${candidate}" ]]; then
      return 0
    fi
  done
  return 1
}

pin_idle_gpu() {
  local physical_index=$1
  if ! [[ "${physical_index}" =~ ^[0-9]+$ ]]; then
    echo "Physical GPU index must be a non-negative integer." >&2
    exit 2
  fi
  if ! nvidia-smi --id="${physical_index}" --query-gpu=uuid \
    --format=csv,noheader,nounits >/dev/null 2>&1; then
    echo "Physical GPU index does not exist: ${physical_index}" >&2
    exit 1
  fi
  local active_pids
  active_pids=$(
    nvidia-smi --id="${physical_index}" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null \
      | tr -d '[:space:]'
  )
  if [[ -n "${active_pids}" && "${CHECKPOINT_EVAL_ALLOW_BUSY_GPU:-0}" != 1 ]]; then
    echo "Physical GPU ${physical_index} is busy (PIDs: ${active_pids})." >&2
    exit 1
  fi
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="${physical_index}"
}

prepare_data() {
  require_file "${CONFIG}"
  require_file "${POPULATION}"
  require_file "${REGISTRY}"
  require_file "${ROSTER}"

  "${DATA_PYTHON}" -m jobs.main.fetch_loo_source_snapshots \
    --registry "${REGISTRY}" \
    --population "${POPULATION}" \
    --output-dir "${SNAPSHOT_DIR}" \
    --expected-vintage-count 11 \
    --max-workers 2 \
    --requests-per-second 2 \
    --timeout-seconds 60 \
    --max-retries 4 \
    --resume

  "${DATA_PYTHON}" -m jobs.main.build_loo_indicator_ledger \
    --registry "${REGISTRY}" \
    --snapshot-manifest "${SNAPSHOT_MANIFEST}" \
    --population "${POPULATION}" \
    --roster "${ROSTER}" \
    --output-dir "${LEDGER_DIR}"

  "${DATA_PYTHON}" -m jobs.main.validate_loo_indicator_ledger \
    --ledger-manifest "${LEDGER_MANIFEST}" \
    --registry "${REGISTRY}" \
    --snapshot-manifest "${SNAPSHOT_MANIFEST}" \
    --population "${POPULATION}" \
    --roster "${ROSTER}"

  local offline_args=()
  if [[ "${CHECKPOINT_EVAL_OFFLINE:-0}" == 1 ]]; then
    offline_args+=(--offline)
  fi
  "${DATA_PYTHON}" -m jobs.eval.build_checkpoint_eval_dataset \
    --config "${CONFIG}" \
    --indicator-inputs "${LEDGER_INPUTS}" \
    --ledger-manifest "${LEDGER_MANIFEST}" \
    --output-dir "${DATASET_DIR}" \
    "${offline_args[@]}"
}

generate_one() {
  local artifact_id=$1
  local physical_index=$2
  if ! is_known_artifact "${artifact_id}"; then
    echo "Unknown checkpoint-evaluation artifact: ${artifact_id}" >&2
    exit 2
  fi
  require_file "${CHECKPOINT_MANIFEST}"
  require_file "${CONFIG}"
  require_file "${PROMPTS}"
  pin_idle_gpu "${physical_index}"
  mkdir -p "${GENERATION_DIR}"
  "${MODEL_PYTHON}" -m jobs.generation.generate_checkpoint_artifact \
    --checkpoint-manifest "${CHECKPOINT_MANIFEST}" \
    --prompts "${PROMPTS}" \
    --config "${CONFIG}" \
    --artifact-id "${artifact_id}" \
    --output-dir "${GENERATION_DIR}" \
    --batch-size "${BATCH_SIZE}"
}

score_all() {
  local scoring_policy=${1:-strict-final-answer-v2}
  local score_dir=${2:-${SCORE_DIR}}
  require_file "${CONFIG}"
  require_file "${TEST_MANIFEST}"
  require_file "${CHECKPOINT_MANIFEST}"
  require_file "${PROMPTS}"
  require_file "${REFERENCES}"
  require_file "${SEMANTIC_MANIFEST}"
  require_file "${LINEAGE_EVIDENCE}"
  local artifact_id
  local generation_args=()
  local required_args=()
  for artifact_id in "${ARTIFACT_IDS[@]}"; do
    require_file "${GENERATION_DIR}/${artifact_id}.jsonl"
    require_file "${GENERATION_DIR}/${artifact_id}.manifest.json"
    generation_args+=(
      --generations "${GENERATION_DIR}/${artifact_id}.jsonl"
      --generation-manifest "${GENERATION_DIR}/${artifact_id}.manifest.json"
    )
    required_args+=(--required-artifact-id "${artifact_id}")
  done
  local semantic_settings
  semantic_settings=$(
    "${DATA_PYTHON}" - "${SEMANTIC_MANIFEST}" "${REPO_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])
bertscore = manifest["models"]["bertscore"]
mpnet = manifest["models"]["embedding_cosine"]
print(
    root / bertscore["local_path"],
    bertscore["directory_sha256"],
    bertscore["num_layers"],
    root / mpnet["local_path"],
    mpnet["directory_sha256"],
)
PY
  )
  local bertscore_path bertscore_sha bertscore_layers mpnet_path mpnet_sha
  read -r bertscore_path bertscore_sha bertscore_layers mpnet_path mpnet_sha \
    <<<"${semantic_settings}"
  require_dir "${bertscore_path}"
  require_dir "${mpnet_path}"
  pin_idle_gpu "${SEMANTIC_GPU_INDEX}"
  mkdir -p "${score_dir}"

  "${MODEL_PYTHON}" -m jobs.eval.eval_checkpoint_generation \
    "${generation_args[@]}" \
    --references "${REFERENCES}" \
    --prompts "${PROMPTS}" \
    --config "${CONFIG}" \
    --test-manifest "${TEST_MANIFEST}" \
    --checkpoint-manifest "${CHECKPOINT_MANIFEST}" \
    --semantic-manifest "${SEMANTIC_MANIFEST}" \
    --lineage-evidence "${LINEAGE_EVIDENCE}" \
    --output-dir "${score_dir}" \
    --scoring-policy "${scoring_policy}" \
    "${required_args[@]}" \
    --bootstrap-samples 10000 \
    --bootstrap-seed 20260729 \
    --bertscore-model "${bertscore_path}" \
    --bertscore-model-sha256 "${bertscore_sha}" \
    --bertscore-num-layers "${bertscore_layers}" \
    --mpnet-model "${mpnet_path}" \
    --mpnet-model-sha256 "${mpnet_sha}" \
    --semantic-device cuda:0 \
    --semantic-batch-size 8
}

case "${STAGE}" in
  prepare)
    prepare_data
    ;;
  generate)
    if [[ -z "${ARTIFACT_ARGUMENT}" ]]; then
      usage >&2
      exit 2
    fi
    generate_one "${ARTIFACT_ARGUMENT}" "${GPU_ARGUMENT:-${DEFAULT_GPU_INDEX}}"
    ;;
  generate-all)
    for artifact_id in "${ARTIFACT_IDS[@]}"; do
      generate_one "${artifact_id}" "${DEFAULT_GPU_INDEX}"
    done
    ;;
  score)
    score_all strict-final-answer-v2 "${SCORE_DIR}"
    ;;
  score-length-tolerant)
    score_all length-tolerant-open-tags-v1 "${LENGTH_TOLERANT_SCORE_DIR}"
    ;;
  all)
    prepare_data
    for artifact_id in "${ARTIFACT_IDS[@]}"; do
      generate_one "${artifact_id}" "${DEFAULT_GPU_INDEX}"
    done
    score_all strict-final-answer-v2 "${SCORE_DIR}"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
