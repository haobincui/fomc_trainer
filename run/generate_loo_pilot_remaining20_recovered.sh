#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
FLOW_ROOT="${REPO_ROOT}/output/evaluation/main/canonical_loo/workflows/canonical-d1-20260729T123849Z"
MINUTES_MODEL="${REPO_ROOT}/output/checkpoints/recovered/llama_sft_synthetic_20250526_2_cp1668_recovered_v1_20260729/model"
PYTHON_BIN="/home/haobin_cui/.conda/envs/llama_factory/bin/python"
RUN_ID=${LOO_RUN_ID:-"pilot-remaining20-recovered-v1-$(date -u +%Y%m%dT%H%M%SZ)"}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to verify physical GPU 1." >&2
  exit 1
fi

GPU1_UUID=$(
  nvidia-smi \
    --id=1 \
    --query-gpu=uuid \
    --format=csv,noheader,nounits
)
GPU1_UUID=${GPU1_UUID//[[:space:]]/}
GPU1_PROCESSES=$(
  nvidia-smi \
    --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits |
    awk -F',' -v uuid="${GPU1_UUID}" '
      {
        observed=$1
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", observed)
        if (observed == uuid) print
      }
    '
)
if [[ -n "${GPU1_PROCESSES}" ]]; then
  echo "Refusing to start because physical GPU 1 has compute processes:" >&2
  printf '%s\n' "${GPU1_PROCESSES}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
env \
  -u LOO_REUSE_PREPARED_RUN_ROOT \
  -u LOO_RUN_ROOT \
  -u LOO_OUTPUT_BASE \
  -u LOO_FOREGROUND \
  -u LOO_RESUME \
  LOO_RUN_ID="${RUN_ID}" \
  LOO_PYTHON="${PYTHON_BIN}" \
  LOO_ANALYSIS_MODEL="${REPO_ROOT}/../fomc_trainer_back/fomc_trainer/output/merged/llama_grpo_20250515" \
  LOO_ANALYSIS_TOKENIZER="${REPO_ROOT}/models/DeepSeek-R1-Distill-Llama-8B" \
  LOO_MINUTES_MODEL="${MINUTES_MODEL}" \
  LOO_MINUTES_TOKENIZER="${MINUTES_MODEL}" \
  LOO_GENERATION_CONFIG="${REPO_ROOT}/configs/main/canonical_loo_generation.json" \
  LOO_INDICATOR_INPUT="${FLOW_ROOT}/ledgers/pilot_eval_13/indicator_inputs.jsonl" \
  LOO_LEDGER_MANIFEST="${FLOW_ROOT}/ledgers/pilot_eval_13/ledger_manifest.json" \
  LOO_SNAPSHOT_MANIFEST="${FLOW_ROOT}/source_snapshots/snapshot_manifest.json" \
  LOO_SOURCE_REGISTRY="${FLOW_ROOT}/inputs/loo_indicator_sources.json" \
  LOO_INDICATOR_ROSTER="${FLOW_ROOT}/inputs/leave_one_out_roster.json" \
  LOO_SECTION_ROSTER="${FLOW_ROOT}/inputs/loo_sections.json" \
  LOO_REUSE_ANALYSIS_MANIFEST="${FLOW_ROOT}/generations/pilot_eval_13/analysis/analysis_manifest.json" \
  LOO_INTERVENTION_INDICATORS="Bank-Capital,Bank-Credit-to-Private-Sector,Business-Investment,Commodity-Prices,Consumer-Confidence-Index,Corporate-Bond-Yields,Exchange-Rate,Federal-Reserve-Balance-Sheet,Government-Purchases,Home-Prices,Housing-Starts,Industrial-Production,International-Equity-Markets,Labour-Market,Market-Volatility-(VIX),Money-Supply,Mortgage-Rates,Overnight-Rate,Personal-Consumption-Expenditures-(PCE),Trade-Balance" \
  LOO_MINUTES_TOKEN_LIMIT_POLICY=exclude \
  LOO_BATCH_SIZE=20 \
  "${SCRIPT_DIR}/_run_canonical_loo_generation.sh" \
    pilot_eval_13 \
    "${FLOW_ROOT}/inputs/loo_population_pilot_eval_13.json"
