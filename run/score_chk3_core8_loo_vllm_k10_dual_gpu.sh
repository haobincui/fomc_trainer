#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
REPO_ROOT="/home/haobin_cui/research_files_space_2/fomc_trainer"
PYTHON_BIN="${LOO_SCORE_PYTHON:-/home/haobin_cui/.conda/envs/fomc_trainer/bin/python}"
MODULE="jobs.eval.score_chk3_core8_loo_vllm_k10"

OLD_ROOT="$REPO_ROOT/output/evaluation/main/chk3_cp318_core8_loo_vllm_k5_n128_1993_2008_20260817_v1"
OLD_GENERATION_MANIFEST="$OLD_ROOT/generation_formal_n2176_k5_v1/chk3/manifest.json"
OLD_GENERATION_SHA256="23af367ba52ddebbc76e47b3f119f73fcbace052d10f77d117a8741b9fdd40da"
OLD_SCORE_MANIFEST="$OLD_ROOT/score_raw_semantic_dual_gpu_b10000_v2/manifest.json"
OLD_SCORE_SHA256="4df87c36a7215e86fc22c1a33b96b700344f523254c035e5efaf5d00383c5fee"

RUN_ROOT="$REPO_ROOT/output/evaluation/main/chk3_cp318_core8_loo_vllm_k10_n128_1993_2008_20260824_v1"
INCREMENT_MANIFEST="$RUN_ROOT/generation_formal_increment_k6_k10_n2176_v1/chk3/manifest.json"
INCREMENT_COHORT="$RUN_ROOT/preparation_increment_k6_k10_v1/cohort_n2176_increment_k6_k10.v1.json"
SEMANTIC_MANIFEST="$REPO_ROOT/configs/main/checkpoint_eval_semantic_models.json"
SEMANTIC_SHA256="639ca25cf4f5e695b0c6cfeeb47aba75d3e6dfeda1ad87510c23bb4f4a378bf7"
LEGACY_NF4_ROOT="$REPO_ROOT/output/evaluation/main/chk3_cp318_core8_loo_stochastic_k10_n128_1993_2008_20260815_v1"

SCORE_ROOT="$RUN_ROOT/score_incremental_reuse_k5_raw_semantic_b10000_v1"
PLAN="$SCORE_ROOT/score_plan.json"
BERT_ROOT="$SCORE_ROOT/semantic_increment_bertscore_gpu0"
MPNET_ROOT="$SCORE_ROOT/semantic_increment_mpnet_gpu1"
PIPELINE_LOG="$RUN_ROOT/score_incremental_reuse_k5_raw_semantic_b10000_v1.log"

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: missing required file: $1" >&2
        return 1
    fi
}

prepare() {
    require_file "$INCREMENT_MANIFEST"
    require_file "$INCREMENT_COHORT"
    local increment_sha cohort_sha
    increment_sha="$(sha256_file "$INCREMENT_MANIFEST")"
    cohort_sha="$(sha256_file "$INCREMENT_COHORT")"
    "$PYTHON_BIN" -u -m "$MODULE" prepare \
        --old-generation-manifest "$OLD_GENERATION_MANIFEST" \
        --old-generation-manifest-sha256 "$OLD_GENERATION_SHA256" \
        --old-score-manifest "$OLD_SCORE_MANIFEST" \
        --old-score-manifest-sha256 "$OLD_SCORE_SHA256" \
        --increment-manifest "$INCREMENT_MANIFEST" \
        --increment-manifest-sha256 "$increment_sha" \
        --increment-cohort "$INCREMENT_COHORT" \
        --increment-cohort-sha256 "$cohort_sha" \
        --semantic-manifest "$SEMANTIC_MANIFEST" \
        --semantic-manifest-sha256 "$SEMANTIC_SHA256" \
        --legacy-nf4-root "$LEGACY_NF4_ROOT" \
        --output-dir "$SCORE_ROOT"
}

semantic() {
    require_file "$PLAN"
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
        >"$SCORE_ROOT/semantic_increment_bertscore_gpu0.console.log" 2>&1 &
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
        >"$SCORE_ROOT/semantic_increment_mpnet_gpu1.console.log" 2>&1 &
    local mpnet_pid=$!
    local failed=0
    wait "$bert_pid" || failed=1
    wait "$mpnet_pid" || failed=1
    if [[ "$failed" -ne 0 ]]; then
        echo "ERROR: at least one incremental semantic worker failed" >&2
        return 1
    fi
}

finalize() {
    require_file "$PLAN"
    require_file "$BERT_ROOT/manifest.json"
    require_file "$MPNET_ROOT/manifest.json"
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
        --bootstrap-seed 20260824
}

status() {
    date -u +%FT%TZ
    for path in \
        "$INCREMENT_MANIFEST" \
        "$PLAN" \
        "$BERT_ROOT/manifest.json" \
        "$MPNET_ROOT/manifest.json" \
        "$SCORE_ROOT/manifest.json"; do
        if [[ -f "$path" ]]; then
            echo "ready $(sha256_file "$path") $path"
        else
            echo "not_ready $path"
        fi
    done
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    fi
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
