#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 POPULATION_ID POPULATION_JSON [--worker RUN_ID RUN_ROOT]" >&2
  exit 2
fi

POPULATION_ID=$1
POPULATION_JSON=$2
MODE=${3:-}

case "${POPULATION_ID}" in
  pilot_eval_13)
    PHASE=pilot
    ;;
  formal_test_13)
    PHASE=formal
    ;;
  *)
    echo "Unsupported frozen population: ${POPULATION_ID}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

ANALYSIS_MODEL=${LOO_ANALYSIS_MODEL:-"${REPO_ROOT}/../fomc_trainer_back/fomc_trainer/output/merged/llama_grpo_20250515"}
ANALYSIS_TOKENIZER=${LOO_ANALYSIS_TOKENIZER:-"${REPO_ROOT}/models/DeepSeek-R1-Distill-Llama-8B"}
MINUTES_MODEL=${LOO_MINUTES_MODEL:-"${REPO_ROOT}/../fomc_trainer_back/fomc_trainer/output/merged/llama_sft_synthetic_20250526"}
MINUTES_TOKENIZER=${LOO_MINUTES_TOKENIZER:-"${MINUTES_MODEL}"}
INDICATOR_ROSTER=${LOO_INDICATOR_ROSTER:-"${REPO_ROOT}/configs/main/leave_one_out_roster.json"}
SECTION_ROSTER=${LOO_SECTION_ROSTER:-"${REPO_ROOT}/configs/main/loo_sections.json"}
GENERATION_CONFIG=${LOO_GENERATION_CONFIG:-"${REPO_ROOT}/configs/main/canonical_loo_generation.json"}
INDICATOR_INPUT=${LOO_INDICATOR_INPUT:-"${REPO_ROOT}/dataset/processed/main/evaluation_inputs/canonical_loo/${POPULATION_ID}/indicator_inputs.jsonl"}
LEDGER_MANIFEST=${LOO_LEDGER_MANIFEST:-"${INDICATOR_INPUT%/*}/ledger_manifest.json"}
SNAPSHOT_MANIFEST=${LOO_SNAPSHOT_MANIFEST:-}
SOURCE_REGISTRY=${LOO_SOURCE_REGISTRY:-"${REPO_ROOT}/configs/main/loo_indicator_sources.json"}
if [[ -n "${LOO_PYTHON:-}" ]]; then
  PYTHON_BIN=${LOO_PYTHON}
elif [[ -n "${HOME:-}" && -x "${HOME}/.conda/envs/llama_factory/bin/python" ]]; then
  PYTHON_BIN="${HOME}/.conda/envs/llama_factory/bin/python"
else
  PYTHON_BIN=python
fi
BATCH_SIZE=${LOO_BATCH_SIZE:-20}

require_file() {
  if [[ -z "$1" || ! -f "$1" ]]; then
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

require_file "${POPULATION_JSON}"
require_file "${INDICATOR_INPUT}"
require_file "${INDICATOR_ROSTER}"
require_file "${SECTION_ROSTER}"
require_file "${GENERATION_CONFIG}"
require_file "${LEDGER_MANIFEST}"
require_file "${SNAPSHOT_MANIFEST}"
require_file "${SOURCE_REGISTRY}"
require_dir "${ANALYSIS_MODEL}"
require_dir "${ANALYSIS_TOKENIZER}"
require_dir "${MINUTES_MODEL}"
require_dir "${MINUTES_TOKENIZER}"

if [[ "${MODE}" != "--worker" ]]; then
  RUN_ID=${LOO_RUN_ID:-"${POPULATION_ID}-$(date -u +%Y%m%dT%H%M%SZ)"}
  if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "LOO_RUN_ID must contain only letters, digits, dot, underscore, or hyphen." >&2
    exit 2
  fi
  if [[ -n "${LOO_RUN_ROOT:-}" ]]; then
    RUN_ROOT=${LOO_RUN_ROOT}
  else
    OUTPUT_BASE=${LOO_OUTPUT_BASE:-"${REPO_ROOT}/output/evaluation/main/canonical_loo/${POPULATION_ID}"}
    RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
  fi
  mkdir -p "$(dirname "${RUN_ROOT}")"
  if ! mkdir "${RUN_ROOT}"; then
    echo "Run directory already exists; choose a new LOO_RUN_ID: ${RUN_ROOT}" >&2
    exit 1
  fi

  if [[ "${LOO_FOREGROUND:-0}" == "1" ]]; then
    exec "${SCRIPT_PATH}" "${POPULATION_ID}" "${POPULATION_JSON}" --worker "${RUN_ID}" "${RUN_ROOT}"
  fi

  nohup "${SCRIPT_PATH}" "${POPULATION_ID}" "${POPULATION_JSON}" --worker "${RUN_ID}" "${RUN_ROOT}" \
    >"${RUN_ROOT}/run.log" 2>&1 </dev/null &
  WORKER_PID=$!
  printf '%s\n' "${WORKER_PID}" >"${RUN_ROOT}/run.pid"
  echo "Started canonical LOO generation."
  echo "run_id=${RUN_ID}"
  echo "pid=${WORKER_PID}"
  echo "log=${RUN_ROOT}/run.log"
  echo "output=${RUN_ROOT}"
  exit 0
fi

if [[ $# -ne 5 ]]; then
  echo "Internal worker invocation is incomplete." >&2
  exit 2
fi

RUN_ID=$4
RUN_ROOT=$5
ANALYSIS_DIR="${RUN_ROOT}/analysis"
PROMPT_DIR="${RUN_ROOT}/prompts"
SPEC_FILE="${RUN_ROOT}/generation_spec.json"

if ! command -v flock >/dev/null 2>&1; then
  echo "Required command is unavailable: flock" >&2
  exit 1
fi
exec 9>"${RUN_ROOT}/.worker.lock"
if ! flock -n 9; then
  echo "Another worker already holds the generation lock: ${RUN_ROOT}" >&2
  exit 1
fi

cleanup_pid() {
  local exit_code=$?
  if [[ -f "${RUN_ROOT}/run.pid" ]]; then
    local recorded_pid
    recorded_pid=$(<"${RUN_ROOT}/run.pid")
    if [[ "${recorded_pid}" == "$$" ]]; then
      rm -f "${RUN_ROOT}/run.pid"
    fi
  fi
  return "${exit_code}"
}
trap cleanup_pid EXIT

cd "${REPO_ROOT}"

echo "run_id=${RUN_ID}"
echo "population=${POPULATION_ID}"
echo "generation_only=true"
echo "training_performed=false"
echo "indicator_input=${INDICATOR_INPUT}"
echo "ledger_manifest=${LEDGER_MANIFEST}"
echo "snapshot_manifest=${SNAPSHOT_MANIFEST}"
echo "source_registry=${SOURCE_REGISTRY}"

"${PYTHON_BIN}" -m jobs.generation.canonical_indicator_analysis \
  --input "${INDICATOR_INPUT}" \
  --roster "${INDICATOR_ROSTER}" \
  --population "${POPULATION_JSON}" \
  --ledger-manifest "${LEDGER_MANIFEST}" \
  --snapshot-manifest "${SNAPSHOT_MANIFEST}" \
  --source-registry "${SOURCE_REGISTRY}" \
  --model "${ANALYSIS_MODEL}" \
  --tokenizer "${ANALYSIS_TOKENIZER}" \
  --output-dir "${ANALYSIS_DIR}" \
  --seed 20260728 \
  --batch-size "${BATCH_SIZE}"

"${PYTHON_BIN}" -m jobs.generation.loo_prompt_builder \
  --analysis-blocks "${ANALYSIS_DIR}/indicator_analysis.jsonl" \
  --section-roster "${SECTION_ROSTER}" \
  --indicator-roster "${INDICATOR_ROSTER}" \
  --population "${POPULATION_JSON}" \
  --tokenizer "${MINUTES_TOKENIZER}" \
  --output-dir "${PROMPT_DIR}"

"${PYTHON_BIN}" -m jobs.generation.build_loo_generation_spec \
  --run-id "${RUN_ID}" \
  --phase "${PHASE}" \
  --population-id "${POPULATION_ID}" \
  --output "${SPEC_FILE}" \
  --model "analysis_model=${ANALYSIS_MODEL}" \
  --model "minutes_model=${MINUTES_MODEL}" \
  --tokenizer "analysis_tokenizer=${ANALYSIS_TOKENIZER}" \
  --tokenizer "minutes_tokenizer=${MINUTES_TOKENIZER}" \
  --source "indicator_input=${INDICATOR_INPUT}" \
  --source "ledger_manifest=${LEDGER_MANIFEST}" \
  --source "snapshot_manifest=${SNAPSHOT_MANIFEST}" \
  --source "source_registry=${SOURCE_REGISTRY}" \
  --source "population=${POPULATION_JSON}" \
  --source "indicator_roster=${INDICATOR_ROSTER}" \
  --source "section_roster=${SECTION_ROSTER}" \
  --source "analysis_output=${ANALYSIS_DIR}/indicator_analysis.jsonl" \
  --source "analysis_manifest=${ANALYSIS_DIR}/analysis_manifest.json" \
  --source "prompt_manifest=${PROMPT_DIR}/prompt_manifest.json" \
  --source "exact_delete_prompts=${PROMPT_DIR}/exact_delete" \
  --source "neutral_prompts=${PROMPT_DIR}/neutral" \
  --generation-config "${GENERATION_CONFIG}" \
  --replicate-seed 20260728 \
  --replicate-seed 20260729 \
  --replicate-seed 21260729 \
  --replicate-seed 22260729 \
  --replicate-seed 23260729 \
  --replicate-seed 24260729

SPEC_SHA256=$(sha256sum "${SPEC_FILE}" | awk '{print $1}')

run_minutes_generation() {
  local arm=$1
  local regime=$2
  local strategy=$3
  local prompt_folder=$4
  local intervention_manifest=$5
  local simulation_step=$6
  local base_seed=$7
  local temperature=$8
  local top_p=$9

  "${PYTHON_BIN}" -m jobs.generation.mask_generation \
    --input-folder "${prompt_folder}" \
    --model "${MINUTES_MODEL}" \
    --tokenizer "${MINUTES_TOKENIZER}" \
    --simulation-step "${simulation_step}" \
    --output-dir "${RUN_ROOT}/generations/${arm}_${regime}" \
    --roster-file "${PROMPT_DIR}/run_intervention_roster.json" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${base_seed}" \
    --temperature "${temperature}" \
    --top-p "${top_p}" \
    --max-new-tokens 8192 \
    --max-model-len 16384 \
    --masking-strategy "${strategy}" \
    --seed-policy sample-id-sha256-v1 \
    --require-normal-finish \
    --intervention-manifest "${intervention_manifest}" \
    --prompt-manifest "${PROMPT_DIR}/prompt_manifest.json" \
    --generation-spec "${SPEC_FILE}" \
    --generation-spec-sha256 "${SPEC_SHA256}"
}

run_minutes_generation \
  deletion primary indicator_block_deletion \
  "${PROMPT_DIR}/exact_delete" \
  "${PROMPT_DIR}/intervention_manifest.json" \
  1 20260728 0.0 1.0

run_minutes_generation \
  neutral primary indicator_block_neutral_replacement \
  "${PROMPT_DIR}/neutral" \
  "${PROMPT_DIR}/neutral_intervention_manifest.json" \
  1 20260728 0.0 1.0

run_minutes_generation \
  deletion stochastic indicator_block_deletion \
  "${PROMPT_DIR}/exact_delete" \
  "${PROMPT_DIR}/intervention_manifest.json" \
  5 20260729 0.6 0.9

run_minutes_generation \
  neutral stochastic indicator_block_neutral_replacement \
  "${PROMPT_DIR}/neutral" \
  "${PROMPT_DIR}/neutral_intervention_manifest.json" \
  5 20260729 0.6 0.9

"${PYTHON_BIN}" -m jobs.generation.finalize_loo_generation \
  --run-root "${RUN_ROOT}" \
  --generation-spec "${SPEC_FILE}" \
  --generation-spec-sha256 "${SPEC_SHA256}" \
  --analysis-manifest "${ANALYSIS_DIR}/analysis_manifest.json" \
  --prompt-manifest "${PROMPT_DIR}/prompt_manifest.json" \
  --output "${RUN_ROOT}/generation_release_manifest.json"

echo "status=complete"
echo "generation_spec_sha256=${SPEC_SHA256}"
echo "generation_release_manifest=${RUN_ROOT}/generation_release_manifest.json"
echo "output=${RUN_ROOT}"
