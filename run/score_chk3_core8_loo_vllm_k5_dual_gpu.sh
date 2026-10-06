#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
REPO_ROOT="/home/haobin_cui/research_files_space_2/fomc_trainer"
PYTHON_BIN="${LOO_SCORE_PYTHON:-/home/haobin_cui/.conda/envs/fomc_trainer/bin/python}"
MODULE="jobs.eval.score_chk3_core8_loo_vllm_k5"
RUN_ROOT="$REPO_ROOT/output/evaluation/main/chk3_cp318_core8_loo_vllm_k5_n128_1993_2008_20260817_v1"
RUN_MANIFEST="$RUN_ROOT/generation_formal_n2176_k5_v1/chk3/manifest.json"
RUN_MANIFEST_SHA256="23af367ba52ddebbc76e47b3f119f73fcbace052d10f77d117a8741b9fdd40da"
COHORT="$RUN_ROOT/preparation_v2/cohort_n2176_k5.v2.json"
COHORT_SHA256="26c1a06231fd8b10a0ccf2a9de22a59b7d657696da88a18981bad6fb82762e1c"
SEMANTIC_MANIFEST="$REPO_ROOT/configs/main/checkpoint_eval_semantic_models.json"
SEMANTIC_MANIFEST_SHA256="639ca25cf4f5e695b0c6cfeeb47aba75d3e6dfeda1ad87510c23bb4f4a378bf7"
SCORE_ROOT="$RUN_ROOT/score_raw_semantic_dual_gpu_b10000_v2"
PLAN="$SCORE_ROOT/score_plan.json"
BERT_ROOT="$SCORE_ROOT/semantic_bertscore_gpu0"
MPNET_ROOT="$SCORE_ROOT/semantic_mpnet_gpu1"
PIPELINE_LOG="$RUN_ROOT/score_raw_semantic_dual_gpu_b10000_v2.log"

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

prepare() {
    "$PYTHON_BIN" -u -m "$MODULE" prepare \
        --run-manifest "$RUN_MANIFEST" \
        --run-manifest-sha256 "$RUN_MANIFEST_SHA256" \
        --cohort "$COHORT" \
        --cohort-sha256 "$COHORT_SHA256" \
        --semantic-manifest "$SEMANTIC_MANIFEST" \
        --semantic-manifest-sha256 "$SEMANTIC_MANIFEST_SHA256" \
        --output-dir "$SCORE_ROOT"
}

semantic() {
    if [[ ! -f "$PLAN" ]]; then
        echo "ERROR: missing sealed score plan: $PLAN" >&2
        return 1
    fi
    local plan_sha
    plan_sha="$(sha256_file "$PLAN")"
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES=0 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
        "$PYTHON_BIN" -u -m "$MODULE" score-metric \
        --metric bertscore \
        --plan "$PLAN" \
        --plan-sha256 "$plan_sha" \
        --output-dir "$BERT_ROOT" \
        --physical-gpu-index 0 \
        --batch-size 8 \
        >"$SCORE_ROOT/semantic_bertscore_gpu0.console.log" 2>&1 &
    local bert_pid=$!
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
        "$PYTHON_BIN" -u -m "$MODULE" score-metric \
        --metric mpnet \
        --plan "$PLAN" \
        --plan-sha256 "$plan_sha" \
        --output-dir "$MPNET_ROOT" \
        --physical-gpu-index 1 \
        --batch-size 32 \
        >"$SCORE_ROOT/semantic_mpnet_gpu1.console.log" 2>&1 &
    local mpnet_pid=$!
    local failed=0
    wait "$bert_pid" || failed=1
    wait "$mpnet_pid" || failed=1
    if [[ "$failed" -ne 0 ]]; then
        echo "ERROR: at least one semantic metric worker failed" >&2
        return 1
    fi
}

finalize() {
    local plan_sha bert_sha mpnet_sha
    plan_sha="$(sha256_file "$PLAN")"
    bert_sha="$(sha256_file "$BERT_ROOT/manifest.json")"
    mpnet_sha="$(sha256_file "$MPNET_ROOT/manifest.json")"
    "$PYTHON_BIN" -u -m "$MODULE" finalize \
        --plan "$PLAN" \
        --plan-sha256 "$plan_sha" \
        --bertscore-manifest "$BERT_ROOT/manifest.json" \
        --bertscore-manifest-sha256 "$bert_sha" \
        --mpnet-manifest "$MPNET_ROOT/manifest.json" \
        --mpnet-manifest-sha256 "$mpnet_sha" \
        --output-dir "$SCORE_ROOT" \
        --bootstrap-draws 10000 \
        --bootstrap-seed 20260817
}

status() {
    date -u +%FT%TZ
    for path in \
        "$PLAN" \
        "$BERT_ROOT/manifest.json" \
        "$MPNET_ROOT/manifest.json" \
        "$SCORE_ROOT/manifest.json"; do
        if [[ -f "$path" ]]; then
            jq -c '{schema_version,status,metric_worker,coverage,k_adequacy}' "$path"
        else
            echo "not_ready $path"
        fi
    done
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
}

mkdir -p "$(dirname "$PIPELINE_LOG")"
case "$MODE" in
    prepare)
        prepare 2>&1 | tee -a "$PIPELINE_LOG"
        ;;
    semantic)
        semantic 2>&1 | tee -a "$PIPELINE_LOG"
        ;;
    finalize)
        finalize 2>&1 | tee -a "$PIPELINE_LOG"
        ;;
    all)
        prepare 2>&1 | tee -a "$PIPELINE_LOG"
        semantic 2>&1 | tee -a "$PIPELINE_LOG"
        finalize 2>&1 | tee -a "$PIPELINE_LOG"
        ;;
    status)
        status
        ;;
    *)
        echo "usage: $0 {prepare|semantic|finalize|all|status}" >&2
        exit 2
        ;;
esac
