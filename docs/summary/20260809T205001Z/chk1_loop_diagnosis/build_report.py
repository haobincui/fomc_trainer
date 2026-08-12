from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


OUT = Path(__file__).resolve().parent
NOW = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
CURRENT = Path("output/evaluation/diagnostics/chk1_chk2_temp06_20260809T203937Z/chk1_retry")
COMMON = Path("output/evaluation/main/checkpoint_generation/retrain_v2_chk0_chk1_chk2_cp150_20260809/generations")
PRIOR = Path("docs/summary/20260809T181157Z/chk2_optimization_diagnosis")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


current = read_json(CURRENT / "summary.json")
prompt_contract_rows = read_csv(PRIOR / "prompt_contract_comparison.csv")
assert current["sample_id"] == "2025-07-30::participants_views"
assert current["output_token_count"] == 8192
assert current["finish_reason"] == "length"
assert current["has_think_boundary"] is False
assert current["input_was_truncated"] is False

comparison = [
    {"model": "chk0, temp 0", "temperature": 0.0, "tokens": 8192, "finish": "length", "boundary": False, "repetition_pct": 98.0540},
    {"model": "chk1, temp 0", "temperature": 0.0, "tokens": 8192, "finish": "length", "boundary": False, "repetition_pct": 98.7857},
    {"model": "chk2, temp 0", "temperature": 0.0, "tokens": 8192, "finish": "length", "boundary": False, "repetition_pct": 98.1641},
    {"model": "chk1, temp 0.6", "temperature": 0.6, "tokens": 8192, "finish": "length", "boundary": False, "repetition_pct": 99.1084},
]

prefix_repetition = [
    {"prefix": "512", "tokens": 512, "repetition_pct": 85.7759},
    {"prefix": "1,024", "tokens": 1024, "repetition_pct": 92.9412},
    {"prefix": "2,048", "tokens": 2048, "repetition_pct": 96.4800},
    {"prefix": "4,096", "tokens": 4096, "repetition_pct": 98.2162},
    {"prefix": "8,192", "tokens": 8192, "repetition_pct": 99.1084},
]

prompt_scale = []
for row in prompt_contract_rows:
    for statistic, key in (("Median", "prompt_p50"), ("Maximum", "prompt_max")):
        prompt_scale.append(
            {
                "corpus": row["corpus"],
                "statistic": statistic,
                "tokens": float(row[key]),
                "facts_p50": float(row["facts_p50"]),
                "task": row["task"],
            }
        )

binding_errors = [
    {
        "generated_claim": "3-month rate = 4.52%",
        "matching_evidence": "DGS3 = 4.52% on 2024-06-28",
        "error": "3-year Treasury value relabeled as 3-month; historical comparable treated as latest.",
    },
    {
        "generated_claim": "3-year rate = 4.96%",
        "matching_evidence": "DGS30 = 4.96% on 2025-07-28",
        "error": "Current 30-year Treasury value relabeled as 3-year.",
    },
    {
        "generated_claim": "30-year rate = 4.51%",
        "matching_evidence": "DGS30 = 4.51% on 2024-06-28",
        "error": "Year-comparable value repeatedly presented without its historical date; sentence appears 316 times.",
    },
]

causes = [
    {
        "priority": "P0",
        "driver": "Prompt/task is far outside chk1 training distribution",
        "confidence": "High",
        "evidence": "12,475 prompt tokens, 651 facts, 26 topics, 93 series versus SFT median 1,536 tokens, 9 facts, one topic.",
        "implication": "Do not use this row alone to judge chk1 atomic-analysis quality.",
    },
    {
        "priority": "P0",
        "driver": "The Participants' Views request is partly unsatisfiable",
        "confidence": "High",
        "evidence": "The prompt requires participant discussion/outlook while supplying only mechanical D-1 series facts and forbidding invented participant views.",
        "implication": "Allow a grounded insufficiency response or supply participant-view evidence.",
    },
    {
        "priority": "P0",
        "driver": "Cross-series/date binding collapses before termination",
        "confidence": "High",
        "evidence": "The output swaps 3-month/3-year/30-year labels and current/year-comparable dates, then repeats the corrupted bindings.",
        "implication": "Decompose the 651-fact packet before synthesis; temperature alone cannot repair retrieval binding.",
    },
    {
        "priority": "P1",
        "driver": "Inference contract omits chk1/chk2's learned termination cue",
        "confidence": "Medium-high",
        "evidence": "Old evaluator asks for section body only and has no 512-token boundary suffix, while the chat template has already opened native reasoning.",
        "implication": "A/B task scope and boundary suffix independently.",
    },
    {
        "priority": "P2",
        "driver": "SFT double-BOS and verbose-reasoning specialization reduce robustness",
        "confidence": "Medium",
        "evidence": "Training token audit reproduces double BOS; SFT reasoning median is 996 tokens, but targets contain no strict periodic suffixes.",
        "implication": "Fix before a future SFT release, but do not call it the primary cause of this row.",
    },
]

excluded = [
    {"hypothesis": "The SFT target copied a periodic loop", "status": "Rejected", "evidence": "1,743/1,743 targets contain </think>; 0/1,743 have a strict periodic last-128-token suffix."},
    {"hypothesis": "Greedy decoding alone caused it", "status": "Rejected as sufficient cause", "evidence": "temperature 0.6 / top_p 0.95 still reached 8,192 tokens with 99.11% repetition."},
    {"hypothesis": "chk1 alone is broken", "status": "Rejected", "evidence": "On the same row at temperature 0, chk0, chk1, and chk2 all hit 8,192 tokens with no boundary."},
    {"hypothesis": "The token budget was too short", "status": "Rejected", "evidence": "Repetition was already 85.8% at 512 output tokens and 96.5% at 2,048."},
    {"hypothesis": "Context overflow, input truncation, or OOM", "status": "Rejected", "evidence": "12,475 + 8,192 < 24,576; input_was_truncated=false; run completed normally."},
]

evidence = {
    "schema_version": "chk1-loop-diagnosis-v1",
    "generated_at": NOW,
    "current_run": {
        "sample_id": current["sample_id"],
        "prompt_tokens": current["prompt_token_count"],
        "output_tokens": current["output_token_count"],
        "finish_reason": current["finish_reason"],
        "boundary": current["has_think_boundary"],
        "answer": current["has_nonempty_answer"],
        "repetition": current["full_trigram_repetition"],
        "tail_repetition": current["tail_2048_word_trigram_repetition"],
        "dominant_sentence": "The 30-year rate was 4.51 percent.",
        "dominant_sentence_count": 316,
        "exact_periodic_suffix_tokens": 2268,
        "exact_period_tokens": 12,
    },
    "sft_target_audit": {
        "rows": 1743,
        "boundary_rows": 1743,
        "strict_periodic_suffix_rows": 0,
        "token_4gram_repetition_median": 0.08400809716599189,
        "token_4gram_repetition_max": 0.28027210884353737,
    },
}
(OUT / "evidence_snapshot.json").write_text(
    json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)

source_manifest = [
    {"id": "evidence_snapshot", "label": "Reviewed chk1 loop evidence", "path": "docs/summary/20260809T205001Z/chk1_loop_diagnosis/evidence_snapshot.json"},
    {"id": "current_result", "label": "chk1 temperature-0.6 probe", "path": str(CURRENT / "result.json")},
    {"id": "common_generations", "label": "Frozen chk0/chk1/chk2 common generations", "path": str(COMMON)},
    {"id": "stress_prompt", "label": "Frozen 33-row stress prompts", "path": "output/evaluation/main/checkpoint_generation/checkpoint_eval_11_v1/dataset/prompts.jsonl"},
    {"id": "prompt_contract", "label": "Training/evaluation prompt comparison", "path": str(PRIOR / "prompt_contract_comparison.csv")},
    {"id": "sft_targets", "label": "Sealed compressed chk1 SFT targets", "path": "dataset/processed/retrain_v2/chk1_reasoning_compressed_flash_max_v1_20260805/analysis_sft"},
    {"id": "eval_config", "label": "Old Minutes stress evaluator config", "path": "configs/main/checkpoint_generation_eval_11.json"},
    {"id": "sft_config", "label": "chk1 compressed SFT config", "path": "configs/retrain_v2/chk1_analysis_sft_compressed_flash_max_v1_20260805.yaml"},
]

queries = {
    "evidence_snapshot": "SELECT * FROM read_json_auto('docs/summary/20260809T205001Z/chk1_loop_diagnosis/evidence_snapshot.json')",
    "current_result": f"SELECT * EXCLUDE (generated) FROM read_json_auto('{CURRENT / 'result.json'}')",
    "common_generations": f"SELECT * FROM read_json_auto('{COMMON}/*.jsonl', format='newline_delimited', filename=true) WHERE sample_id = '2025-07-30::participants_views'",
    "stress_prompt": "SELECT * FROM read_json_auto('output/evaluation/main/checkpoint_generation/checkpoint_eval_11_v1/dataset/prompts.jsonl', format='newline_delimited') WHERE sample_id = '2025-07-30::participants_views'",
    "prompt_contract": f"SELECT * FROM read_csv_auto('{PRIOR / 'prompt_contract_comparison.csv'}')",
    "sft_targets": "SELECT * FROM read_json_auto('dataset/processed/retrain_v2/chk1_reasoning_compressed_flash_max_v1_20260805/analysis_sft/*.jsonl', format='newline_delimited', filename=true)",
    "eval_config": "SELECT * FROM read_json_auto('configs/main/checkpoint_generation_eval_11.json')",
    "sft_config": "SELECT content FROM read_text('configs/retrain_v2/chk1_analysis_sft_compressed_flash_max_v1_20260805.yaml')",
}

sources = [
    {
        "id": src["id"],
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": queries[src["id"]],
            "description": f"Reads {src['label']} for the bounded diagnosis.",
            "executed_at": NOW,
            "tables_used": [src["path"]],
        },
    }
    for src in source_manifest
]

artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": "为什么 chk1 会循环：旧 Minutes stress 触发了跨序列绑定崩溃",
        "description": "对 chk1 temperature=0.6 单样本输出、冻结三 checkpoint 对照和全部 SFT targets 的技术诊断。",
        "generatedAt": NOW,
        "cards": [
            {"id": "tokens_card", "description": "temperature=0.6 探针实际生成 tokens。", "dataset": "headline", "sourceId": "current_result", "metrics": [{"label": "Output at cap", "field": "output_tokens", "format": "number"}]},
            {"id": "rep_card", "description": "完整输出的 word-trigram repetition。", "dataset": "headline", "sourceId": "current_result", "metrics": [{"label": "Repetition", "field": "repetition", "format": "percent"}]},
            {"id": "boundary_card", "description": "生成的 reasoning/answer boundary 数。", "dataset": "headline", "sourceId": "current_result", "metrics": [{"label": "Boundaries", "field": "boundaries", "format": "number"}]},
            {"id": "sentence_card", "description": "最常见错误句的精确出现次数。", "dataset": "headline", "sourceId": "evidence_snapshot", "metrics": [{"label": "Top sentence repeats", "field": "dominant_sentence_count", "format": "number"}]},
            {"id": "sft_loop_card", "description": "1,743 个 SFT targets 中的严格周期尾部数。", "dataset": "headline", "sourceId": "sft_targets", "metrics": [{"label": "Periodic SFT targets", "field": "periodic_targets", "format": "number"}]},
        ],
        "charts": [
            {
                "id": "checkpoint_comparison",
                "title": "同一 Minutes stress 样本的 checkpoint/temperature 对照",
                "subtitle": "四条输出均达到8,192-token上限且没有 </think>；纵轴为全文word-trigram repetition。",
                "type": "bar",
                "dataset": "comparison",
                "sourceId": "common_generations",
                "encodings": {
                    "x": {"field": "model", "type": "nominal", "label": "Run"},
                    "y": {"field": "repetition_pct", "type": "quantitative", "label": "Repetition (%)"},
                    "tooltip": [
                        {"field": "tokens", "type": "quantitative", "label": "Output tokens"},
                        {"field": "temperature", "type": "quantitative", "label": "Temperature"},
                        {"field": "finish", "type": "nominal", "label": "Finish"},
                    ],
                },
                "yAxisTitle": "Word-trigram repetition (%)",
                "valueFormat": "number",
                "layout": "full",
            },
            {
                "id": "prefix_repetition",
                "title": "chk1 temperature=0.6 输出前缀重复率",
                "subtitle": "循环在预算早期已形成：512 tokens 时为85.8%，2,048 tokens时为96.5%。",
                "type": "bar",
                "dataset": "prefix_repetition",
                "sourceId": "current_result",
                "encodings": {
                    "x": {"field": "prefix", "type": "nominal", "label": "Output prefix tokens"},
                    "y": {"field": "repetition_pct", "type": "quantitative", "label": "Repetition (%)"},
                    "tooltip": [{"field": "repetition_pct", "type": "quantitative", "label": "Repetition (%)"}],
                },
                "yAxisTitle": "Word-trigram repetition (%)",
                "valueFormat": "number",
                "layout": "full",
            },
            {
                "id": "prompt_scale",
                "title": "SFT、GRPO 与旧 Minutes stress 的 rendered prompt 规模",
                "subtitle": "旧 stress 的任务也从单 topic analysis 变成26-topic Minutes synthesis。",
                "type": "bar",
                "dataset": "prompt_scale",
                "sourceId": "prompt_contract",
                "encodings": {
                    "x": {"field": "corpus", "type": "nominal", "label": "Corpus"},
                    "y": {"field": "tokens", "type": "quantitative", "label": "Prompt tokens"},
                    "color": {"field": "statistic", "type": "nominal", "label": "Statistic"},
                    "tooltip": [
                        {"field": "tokens", "type": "quantitative", "label": "Tokens"},
                        {"field": "facts_p50", "type": "quantitative", "label": "Median facts"},
                        {"field": "task", "type": "nominal", "label": "Task"},
                    ],
                },
                "yAxisTitle": "Rendered prompt tokens",
                "valueFormat": "number",
                "layout": "full",
            },
        ],
        "tables": [
            {
                "id": "binding_errors",
                "title": "循环前已经发生的 value/date/series 绑定错误",
                "subtitle": "直接按生成 claim 与651条 evidence facts 中的同值记录核对。",
                "dataset": "binding_errors",
                "sourceId": "stress_prompt",
                "defaultSort": {"field": "generated_claim", "direction": "asc"},
                "columns": [
                    {"field": "generated_claim", "label": "Generated claim", "type": "text"},
                    {"field": "matching_evidence", "label": "Matching evidence", "type": "text"},
                    {"field": "error", "label": "Binding failure", "type": "text"},
                ],
                "layout": "full",
            },
            {
                "id": "causes",
                "title": "原因优先级",
                "subtitle": "P0为当前直接证据最强的触发/机制；P1/P2为可能放大但尚未单独做干预实验的因素。",
                "dataset": "causes",
                "sourceId": "evidence_snapshot",
                "defaultSort": {"field": "priority", "direction": "asc"},
                "columns": [
                    {"field": "priority", "label": "Priority", "type": "text"},
                    {"field": "driver", "label": "Driver", "type": "text"},
                    {"field": "confidence", "label": "Confidence", "type": "text"},
                    {"field": "evidence", "label": "Evidence", "type": "text"},
                    {"field": "implication", "label": "Implication", "type": "text"},
                ],
                "layout": "full",
            },
            {
                "id": "excluded",
                "title": "已排除或降级的解释",
                "subtitle": "这些解释与已完成的 temperature=0.6、跨checkpoint和SFT-target检查不符。",
                "dataset": "excluded",
                "sourceId": "evidence_snapshot",
                "defaultSort": {"field": "hypothesis", "direction": "asc"},
                "columns": [
                    {"field": "hypothesis", "label": "Hypothesis", "type": "text"},
                    {"field": "status", "label": "Status", "type": "text"},
                    {"field": "evidence", "label": "Evidence", "type": "text"},
                ],
                "layout": "full",
            },
        ],
        "sources": source_manifest,
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# 为什么 chk1 会循环：旧 Minutes stress 触发了跨序列绑定崩溃"},
            {"id": "summary", "type": "markdown", "sourceId": "evidence_snapshot", "body": "## Technical summary\n\n**这条结果不能说明 chk1 的 SFT 数据把循环教进了模型。** 当前 `temperature=0.6` 输出使用的是旧的 12,475-token / 651-fact / 26-topic `Participants' Views Minutes` 压力 prompt，而 chk1 训练是中位 1,536 tokens / 9 facts / 单 topic 的 concise analysis。chk1 在前 512 输出 tokens 时重复率已达 **85.8%**，最终 8,192/8,192、无 `</think>`、全文重复率 **99.11%**。输出在循环前已经把 Treasury maturity 和 current/year-comparable dates 交叉错配，表明首要失效是超载检索与任务合同冲突；采样温度从 0 提高到 0.6 并未修复。"},
            {"id": "cards", "type": "metric-strip", "cardIds": ["tokens_card", "rep_card", "boundary_card", "sentence_card", "sft_loop_card"]},
            {"id": "finding_scope", "type": "markdown", "sourceId": "prompt_contract", "body": "## 旧 Minutes stress 不属于 chk1 的训练任务\n\n本行 prompt 是 chk1 训练中位长度的 **8.1 倍**、训练最大值的 **4.7 倍**，facts 是训练中位的 **72.3 倍**。更关键的是任务从单 topic evidence→analysis 变成 26-topic Minutes synthesis；同时要求写参与者讨论/展望，却没有 participant、speaker、view 或 outlook 证据，并禁止臆造。这是部分不可满足的合同，不只是“文本更长”。"},
            {"id": "prompt_chart", "type": "chart", "chartId": "prompt_scale", "layout": "full"},
            {"id": "finding_binding", "type": "markdown", "sourceId": "stress_prompt", "body": "## 循环是绑定崩溃的结果，不是正常分析写得太长\n\n输出把历史 3-year 的 4.52% 写成 3-month，把当前 30-year 的 4.96% 写成 3-year，又把历史 30-year 的 4.51% 当作无日期的当前值反复生成。最常见句 `The 30-year rate was 4.51 percent.` 精确出现 **316 次**。也就是说，模型先在 651 个高度同构 facts 中失去 series/date/maturity 绑定，然后重复错误的局部高概率句。"},
            {"id": "binding_table", "type": "table", "tableId": "binding_errors", "layout": "full"},
            {"id": "finding_sampling", "type": "markdown", "sourceId": "current_result", "body": "## temperature=0.6 证明 greedy 不是充分解释\n\n相同 stress row 上，chk0/chk1/chk2 的 temperature=0 基线都打满 8,192 tokens；chk1 改为 `temperature=0.6, top_p=0.95` 后仍打满，重复率反而为 99.11%。因此 greedy 会加剧确定性锁死，但当前故障的概率 basin 很强，单纯提高 temperature 无法让模型退出。"},
            {"id": "comparison_chart", "type": "chart", "chartId": "checkpoint_comparison", "layout": "full"},
            {"id": "prefix_note", "type": "markdown", "sourceId": "current_result", "body": "## 循环在预算早期形成\n\n重复率随前缀从 512 tokens 的 85.8% 上升到 2,048 的 96.5% 和 8,192 的 99.1%；最后 2,268 个重编码 tokens 还能由 12-token 周期精确复现。因此增加上限只会延长失败，不会给模型更多“完成思考”的空间。"},
            {"id": "prefix_chart", "type": "chart", "chartId": "prefix_repetition", "layout": "full"},
            {"id": "cause_table", "type": "table", "tableId": "causes", "layout": "full"},
            {"id": "scope", "type": "markdown", "sourceId": "evidence_snapshot", "body": "## Scope, data, and definitions\n\n当前结果是一条 `2025-07-30::participants_views` TP=2 probe；比较基线是同一 prompt 的冻结 chk0/chk1/chk2 temperature=0 outputs。`repetition` 定义为 lowercased word-trigram 的 `(总数−唯一数)/总数`；`strict periodic suffix` 要求最后 128 个重编码 tokens 可由周期 1–256 精确复现。SFT-target 检查覆盖 train/eval/test 共 1,743 rows。"},
            {"id": "method", "type": "markdown", "sourceId": "evidence_snapshot", "body": "## Methodology\n\n先核对 generation metadata 排除截断、context 与运行失败；再在固定 sample 上重算全文及固定 token 前缀重复率、最常见句和精确周期尾部；随后将生成数字按 exact value 回连 651 条 evidence facts，检查 series/date/maturity 绑定；最后扫描全部 sealed SFT responses 的 `</think>` 与严格周期尾部，并对照三个冻结 checkpoints。"},
            {"id": "robustness", "type": "markdown", "body": "## Limitations and robustness\n\n当前 temperature=0.6 只有一个 stress sample，足以否定“greedy 是唯一原因”，不能估计 task-aligned failure rate。temperature=0 基线与本次 TP=2 运行的并行/运行时参数不完全相同，因此 repetition 的小幅差异不能解释为 checkpoint 或 temperature 的精确因果效应。merged-tokenizer 的当前 Transformers 版本会给 regex warning，但正文 repetition 是 word-based，结论不依赖该 warning。"},
            {"id": "excluded_table", "type": "table", "tableId": "excluded", "layout": "full"},
            {"id": "recommend", "type": "markdown", "body": "## Recommended next steps\n\n1. **不要再用这条 full Minutes row 单独判断 chk1。** 先在同一条 task-aligned SFT test prompt 上比较 chk1/chk2。\n2. **Minutes 输入要分层处理。** 先按单 topic/小 evidence packet 让 chk2 生成 grounded analyses，再由 chk3 做聚合；不要让 chk1 一次绑定 651 facts。\n3. **为不可满足输入允许显式不足响应。** 对 `participants_views` 没有 participant evidence 的情况，不应同时要求“必须写 section”与“不得臆造”。\n4. **单独 A/B boundary suffix。** 明确要求在前 512 reasoning tokens 内输出 `</think>`，不要与 task/scope/temperature 同时改。\n5. **未来重训前再修 double BOS 和长度合同。** 这些是风险放大器，但当前没有证据说明 SFT target 直接包含周期循环。"},
            {"id": "questions", "type": "markdown", "body": "## Further questions\n\n- task-aligned prompt 下，chk1 与 chk2 在 temperature=0.3 是否能自然输出 `</think>` 和非空 answer？\n- 将 26 topics 分解后，哪个 topic 或 schema 字段首先触发 series/date binding error？\n- boundary suffix、允许 insufficiency response、以及 evidence scope 三项中，哪一项对终止率贡献最大？"},
        ],
    },
    "snapshot": {
        "version": 1,
        "generatedAt": NOW,
        "status": "ready",
        "datasets": {
            "headline": [{"output_tokens": 8192, "repetition": current["full_trigram_repetition"], "boundaries": 0, "dominant_sentence_count": 316, "periodic_targets": 0}],
            "comparison": comparison,
            "prefix_repetition": prefix_repetition,
            "prompt_scale": prompt_scale,
            "binding_errors": binding_errors,
            "causes": causes,
            "excluded": excluded,
        },
    },
    "sources": sources,
    "package_info": {"originUrl": "artifact://chk1-loop-diagnosis-20260809T205001Z", "controls": {"edit": False, "refresh": False}},
}

(OUT / "artifact.json").write_text(
    json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(OUT / "artifact.json")
