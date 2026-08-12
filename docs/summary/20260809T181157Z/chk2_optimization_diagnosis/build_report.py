from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parents[3]
GENERATED_AT = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_csv(name: str) -> list[dict[str, str]]:
    with (OUT_DIR / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


summary = json.loads((OUT_DIR / "diagnostic_summary.json").read_text(encoding="utf-8"))
prompt_rows = load_csv("prompt_contract_comparison.csv")
health_rows = load_csv("chk2_training_health_through_step150.csv")
quality_rows = load_csv("sft_contract_quality.csv")

prompt_scale = []
for row in prompt_rows:
    for statistic, field in (("Median", "prompt_p50"), ("Maximum", "prompt_max")):
        prompt_scale.append(
            {
                "corpus": row["corpus"],
                "statistic": statistic,
                "tokens": float(row[field]),
                "task": row["task"],
            }
        )

termination_health = [
    {
        "scope": "chk2 in-domain steps 131–150",
        "metric": "Reached answer boundary",
        "rate_pct": 100.0 * summary["training_domain_cp150_window"]["answer_boundaries"] / summary["training_domain_cp150_window"]["rows"],
    },
    {
        "scope": "chk2 in-domain steps 131–150",
        "metric": "Hit token limit",
        "rate_pct": 100.0 * summary["training_domain_cp150_window"]["clipped"] / summary["training_domain_cp150_window"]["rows"],
    },
    {
        "scope": "chk2 old Minutes stress",
        "metric": "Reached answer boundary",
        "rate_pct": 0.0,
    },
    {
        "scope": "chk2 old Minutes stress",
        "metric": "Hit token limit",
        "rate_pct": 100.0,
    },
]

rolling_reward = [
    {
        "step": int(row["step"]),
        "reward_rolling20": float(row["reward_rolling20"]),
        "boundary_rolling20_pct": 100.0 * float(row["boundary_rolling20"]),
        "clipped_rolling20_pct": 100.0 * float(row["clipped_rolling20"]),
    }
    for row in health_rows
    if row["reward_rolling20"]
]

sft_contract = [
    {
        "check": row["check"],
        "rows": int(row["rows"]),
        "rate_pct": 100.0 * float(row["rate"]),
    }
    for row in quality_rows
]

root_causes = [
    {
        "priority": "P0",
        "finding": "Primary evaluation contract is out-of-domain",
        "evidence": "12,402-token median / 650 facts / Minutes output versus 1,249-token median / 9 facts / atomic analysis in chk2 training.",
        "decision": "Retain the old test only as a secondary stress benchmark; do not use it to promote or reject chk2.",
    },
    {
        "priority": "P0",
        "finding": "Greedy decoding enters deterministic reasoning loops",
        "evidence": "99/99 outputs hit 8,192 tokens, 0/99 reached </think>, and 93/99 have an exact periodic suffix.",
        "decision": "Run a small fixed-seed decoding A/B before any retraining; do not raise the token cap again.",
    },
    {
        "priority": "P1",
        "finding": "Reasoning-length contracts conflict",
        "evidence": "SFT reasoning spans 517–1,600 tokens (median 996), while the chk2 user suffix asks for at most 512 and reward efficiency changes across 768–1,536.",
        "decision": "Align prompt, reward, and generation cap around a 768–1,024 target with a 1,536 hard reasoning limit.",
    },
    {
        "priority": "P1",
        "finding": "SFT final-answer contract is contaminated",
        "evidence": "201/1,743 rows are JSON-like or contain evidence identifiers; one training answer is only an opening brace.",
        "decision": "Create a new immutable release using deterministic local cleanup and recover the single malformed row from existing local source/cache.",
    },
    {
        "priority": "P2",
        "finding": "SFT train/inference tokenization has a double-BOS mismatch",
        "evidence": "The rendered prompt already contains BOS, then the plain-string tokenizer prepends BOS again; all targets still retain EOS and fit under 4,608 tokens.",
        "decision": "Strip the literal rendered BOS or preserve conversational tokenization, then add token-parity tests before the next SFT run.",
    },
]

rollout_plan = [
    {
        "phase": 0,
        "action": "Freeze checkpoint-150 and all old-test artifacts",
        "pass_criterion": "No checkpoint, dataset, reward, or old evaluator is changed in place.",
        "compute": "None",
    },
    {
        "phase": 1,
        "action": "Build a versioned primary evaluator on the 190-row held-out analysis_grpo/test split",
        "pass_criterion": "Same atomic evidence→analysis contract for chk0/chk1/chk2; prompt ≤2,560; no input truncation; parse first </think> then score answer only.",
        "compute": "Read-only preprocessing",
    },
    {
        "phase": 2,
        "action": "Run a 12-prompt fixed-seed decoding A/B",
        "pass_criterion": "First compare greedy vs temperature 0.6/top-p 0.95 and repetition penalty 1.0 vs 1.05; lock the winner, then separately compare system+user versus all-in-user prompt placement.",
        "compute": "Two 8B models in parallel, one per A30",
    },
    {
        "phase": 3,
        "action": "Evaluate chk0/chk1/chk2 on all 190 held-out rows",
        "pass_criterion": "Report meeting-cluster paired bootstrap, answer boundary/non-empty/plain-text rates, answer-only numeric support and direction, repetition, and truncation; v3 Judge remains secondary.",
        "compute": "One model per A30; fixed seeds, ideally three replicates",
    },
    {
        "phase": 4,
        "action": "Align the reasoning/output contract",
        "pass_criterion": "Prompt target 768–1,024 reasoning tokens, hard limit 1,536; completion cap 2,048 (2,304 for the initial diagnostic only); no </think> stop sequence.",
        "compute": "No training yet",
    },
    {
        "phase": 5,
        "action": "Only if held-out results justify it, produce cleaned chk1 data and retrain",
        "pass_criterion": "Single BOS, terminal EOS, zero malformed answers, pure-text contract verified; compare to frozen checkpoint-150 before promotion.",
        "compute": "SFT/GRPO scheduled after evidence, respecting the two-A30 policy/Judge split",
    },
]

source_manifest = [
    {
        "id": "diagnostic_outputs",
        "label": "Executed chk2 optimization diagnostic summary",
        "path": "docs/summary/20260809T181157Z/chk2_optimization_diagnosis/diagnostic_summary.json",
    },
    {
        "id": "stress_generations",
        "label": "Frozen chk0/chk1/chk2 checkpoint-generation artifacts",
        "path": "output/evaluation/main/checkpoint_generation/retrain_v2_chk0_chk1_chk2_cp150_20260809/generations",
    },
    {
        "id": "training_completions",
        "label": "chk2 reward-v3 completion artifacts",
        "path": "output/training/retrain_v2/chk2_compressed_chk1_v1_reward_v3_long4096_fresh_20260807/adapters/chk2/completions",
    },
    {
        "id": "sft_dataset",
        "label": "Sealed compressed chk1 SFT release",
        "path": "dataset/processed/retrain_v2/chk1_reasoning_compressed_flash_max_v1_20260805/analysis_sft",
    },
    {
        "id": "grpo_dataset",
        "label": "chk2 atomic analysis GRPO release",
        "path": "dataset/processed/retrain_v2/analysis_base_full_v7_automated_v3_20260804/analysis_grpo",
    },
    {
        "id": "eval_config",
        "label": "Frozen old common-test configuration",
        "path": "configs/main/checkpoint_generation_eval_11.json",
    },
    {
        "id": "model_readme",
        "label": "Local DeepSeek-R1-Distill-Llama-8B model guidance",
        "path": "models/DeepSeek-R1-Distill-Llama-8B/README.md",
    },
    {
        "id": "sft_trainer_source",
        "label": "SFT prompt rendering implementation",
        "path": "src/open_r1/trainer/sft_trainer.py",
    },
]

query_by_source = {
    "diagnostic_outputs": ("SELECT * FROM read_json_auto('docs/summary/20260809T181157Z/chk2_optimization_diagnosis/diagnostic_summary.json')", "Reads the fail-closed executed notebook summary used by the report.", ["diagnostic_summary.json"]),
    "stress_generations": ("SELECT * FROM read_json_auto('output/evaluation/main/checkpoint_generation/retrain_v2_chk0_chk1_chk2_cp150_20260809/generations/*.jsonl', format='newline_delimited', filename=true)", "Audits termination boundaries, token limits, and exact periodic suffixes.", ["three generation JSONL files"]),
    "training_completions": ("SELECT * FROM read_parquet('output/training/retrain_v2/chk2_compressed_chk1_v1_reward_v3_long4096_fresh_20260807/adapters/chk2/completions/completions_*.parquet', filename=true)", "Recomputes in-domain termination and rolling reward health.", ["completion parquet files"]),
    "sft_dataset": ("SELECT * FROM read_json_auto('dataset/processed/retrain_v2/chk1_reasoning_compressed_flash_max_v1_20260805/analysis_sft/*.jsonl', format='newline_delimited', filename=true)", "Audits reasoning length, answer format, BOS/EOS behavior, and prompt scale.", ["train.jsonl", "eval.jsonl", "test.jsonl"]),
    "grpo_dataset": ("SELECT * FROM read_json_auto('dataset/processed/retrain_v2/analysis_base_full_v7_automated_v3_20260804/analysis_grpo/*.jsonl', format='newline_delimited', filename=true)", "Measures the intended atomic-analysis prompt and evidence distribution.", ["train.jsonl", "eval.jsonl", "test.jsonl"]),
    "eval_config": ("SELECT * FROM read_json_auto('configs/main/checkpoint_generation_eval_11.json')", "Confirms the old task, temperature, top-p, token limit, and context budget.", ["checkpoint_generation_eval_11.json"]),
    "model_readme": ("SELECT content FROM read_text('models/DeepSeek-R1-Distill-Llama-8B/README.md')", "Reads the local model author's decoding and prompting guidance.", ["README.md"]),
    "sft_trainer_source": ("SELECT content FROM read_text('src/open_r1/trainer/sft_trainer.py')", "Checks training/inference token parity and the double-BOS mechanism.", ["sft_trainer.py"]),
}

sources = []
for source in source_manifest:
    sql, description, tables = query_by_source[source["id"]]
    sources.append(
        {
            **source,
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": sql,
                "description": description,
                "executed_at": GENERATED_AT,
                "tables_used": tables,
            },
        }
    )

artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": "chk2 优化诊断：先修评测合同与解码，再决定是否重训",
        "description": "基于冻结 chk0/chk1/chk2 共测、chk2 训练 completion 和 chk1 SFT 数据的可复现根因诊断与两张 A30 rollout 方案。",
        "generatedAt": GENERATED_AT,
        "cards": [
            {
                "id": "length_card",
                "description": "旧共测中达到 8,192-token 上限的输出。",
                "dataset": "summary",
                "sourceId": "stress_generations",
                "metrics": [{"label": "Length finishes", "field": "length_finished", "format": "number"}],
            },
            {
                "id": "boundary_card",
                "description": "旧共测中真正生成 reasoning/answer 边界的输出。",
                "dataset": "summary",
                "sourceId": "stress_generations",
                "metrics": [{"label": "Answer boundaries", "field": "answer_boundaries", "format": "number"}],
            },
            {
                "id": "loop_card",
                "description": "尾部至少 128 tokens 呈严格固定周期的输出。",
                "dataset": "summary",
                "sourceId": "stress_generations",
                "metrics": [{"label": "Strict loops", "field": "strict_periodic_suffixes", "format": "number"}],
            },
            {
                "id": "domain_boundary_card",
                "description": "chk2 steps 131–150 在训练域内到达 answer 的比例。",
                "dataset": "summary",
                "sourceId": "training_completions",
                "metrics": [{"label": "In-domain boundary", "field": "in_domain_boundary_pct", "format": "percent"}],
            },
            {
                "id": "sft_violation_card",
                "description": "不符合 chk2 纯文本 final-answer 合同的 chk1 SFT 比例。",
                "dataset": "summary",
                "sourceId": "sft_dataset",
                "metrics": [{"label": "SFT contract violations", "field": "sft_violation_pct", "format": "percent"}],
            },
        ],
        "charts": [
            {
                "id": "prompt_scale_chart",
                "title": "旧共测 prompt 约为 chk2 训练域的十倍",
                "subtitle": "相同本地 tokenizer 下的 rendered prompt；虚线式门槛在正文说明为 2,560 tokens。",
                "type": "bar",
                "dataset": "prompt_scale",
                "sourceId": "diagnostic_outputs",
                "encodings": {
                    "x": {"field": "corpus", "type": "nominal", "label": "Corpus"},
                    "y": {"field": "tokens", "type": "quantitative", "label": "Tokens"},
                    "color": {"field": "statistic", "type": "nominal", "label": "Statistic"},
                    "tooltip": [
                        {"field": "tokens", "type": "quantitative", "label": "Tokens"},
                        {"field": "task", "type": "nominal", "label": "Task"},
                    ],
                },
                "yAxisTitle": "Rendered prompt tokens",
                "valueFormat": "number",
                "layout": "full",
            },
            {
                "id": "termination_health_chart",
                "title": "同一 chk2：训练域能结束，旧 Minutes stress 全部失控",
                "subtitle": "checkpoint-150 的训练域最近 80 个 completion 对比旧 stress 的 33 个 completion。",
                "type": "bar",
                "dataset": "termination_health",
                "sourceId": "diagnostic_outputs",
                "encodings": {
                    "x": {"field": "metric", "type": "nominal", "label": "Termination metric"},
                    "y": {"field": "rate_pct", "type": "quantitative", "label": "Rows (%)"},
                    "color": {"field": "scope", "type": "nominal", "label": "Scope"},
                    "tooltip": [{"field": "rate_pct", "type": "quantitative", "label": "Rows (%)"}],
                },
                "yAxisTitle": "Rows (%)",
                "valueFormat": "number",
                "layout": "full",
            },
            {
                "id": "rolling_reward_chart",
                "title": "checkpoint-150 截止当时处于最高 rolling-20 reward 窗口",
                "subtitle": "训练 reward 仅用于 checkpoint 诊断，不作为独立晋级证据。",
                "type": "line",
                "dataset": "rolling_reward",
                "sourceId": "training_completions",
                "encodings": {
                    "x": {"field": "step", "type": "quantitative", "label": "Training step"},
                    "y": {"field": "reward_rolling20", "type": "quantitative", "label": "Rolling-20 reward"},
                    "tooltip": [
                        {"field": "step", "type": "quantitative", "label": "Step"},
                        {"field": "reward_rolling20", "type": "quantitative", "label": "Reward"},
                        {"field": "boundary_rolling20_pct", "type": "quantitative", "label": "Boundary (%)"},
                        {"field": "clipped_rolling20_pct", "type": "quantitative", "label": "Clipped (%)"},
                    ],
                },
                "yAxisTitle": "Mean reward",
                "valueFormat": "number",
                "layout": "full",
            },
            {
                "id": "sft_contract_chart",
                "title": "chk1 final-answer 合同需要在下一版数据中清理",
                "subtitle": "JSON-like 与 evidence identifier 有重叠；union 是应处理的总集合。",
                "type": "bar",
                "dataset": "sft_contract",
                "sourceId": "sft_dataset",
                "encodings": {
                    "x": {"field": "check", "type": "nominal", "label": "Check"},
                    "y": {"field": "rate_pct", "type": "quantitative", "label": "Rows (%)"},
                    "tooltip": [
                        {"field": "rows", "type": "quantitative", "label": "Rows"},
                        {"field": "rate_pct", "type": "quantitative", "label": "Rate (%)"},
                    ],
                },
                "yAxisTitle": "SFT rows (%)",
                "valueFormat": "number",
                "layout": "full",
            },
        ],
        "tables": [
            {
                "id": "root_causes_table",
                "title": "优化优先级与证据",
                "subtitle": "先修影响因果解释与可比性的 P0，再决定是否投入重训。",
                "dataset": "root_causes",
                "sourceId": "diagnostic_outputs",
                "defaultSort": {"field": "priority", "direction": "asc"},
                "columns": [
                    {"field": "priority", "label": "Priority", "type": "text"},
                    {"field": "finding", "label": "Finding", "type": "text"},
                    {"field": "evidence", "label": "Evidence", "type": "text"},
                    {"field": "decision", "label": "Decision", "type": "text"},
                ],
                "layout": "full",
            },
            {
                "id": "rollout_plan_table",
                "title": "最省算力、信息增益最高的 rollout",
                "subtitle": "为两张 24GB A30 设计；任何训练都排在 task-aligned 评测之后。",
                "dataset": "rollout_plan",
                "sourceId": "diagnostic_outputs",
                "defaultSort": {"field": "phase", "direction": "asc"},
                "columns": [
                    {"field": "phase", "label": "Phase", "format": "number"},
                    {"field": "action", "label": "Action", "type": "text"},
                    {"field": "pass_criterion", "label": "Pass criterion", "type": "text"},
                    {"field": "compute", "label": "Compute", "type": "text"},
                ],
                "layout": "full",
            },
        ],
        "sources": source_manifest,
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# chk2 优化诊断：先修评测合同与解码，再决定是否重训"},
            {
                "id": "technical_summary",
                "type": "markdown",
                "sourceId": "diagnostic_outputs",
                "body": "## Technical summary\n\n**可以优化，但当前最高优先级不是继续训练。** 冻结的旧共测同时把输入从 atomic evidence 扩成全部证据、把输出从 concise analysis 改为 Minutes section，并使用 `temperature=0` 的 greedy decoding。结果是 99/99 输出达到 8,192-token 上限、0/99 进入 answer、93/99 尾部形成严格固定周期。与之相反，checkpoint-150 在训练域最近 80 个 completion 中有 78 个正常进入 answer，且 step 150 是截至当时最高 rolling-20 reward 窗口。因此旧结果证明的是 **评测合同与解码组合失效**，不是 checkpoint-150 已退化。",
            },
            {"id": "metrics", "type": "metric-strip", "cardIds": ["length_card", "boundary_card", "loop_card", "domain_boundary_card", "sft_violation_card"]},
            {
                "id": "finding_contract",
                "type": "markdown",
                "sourceId": "diagnostic_outputs",
                "body": "## Key finding 1 — 旧共测不是 chk2 的可比主评测\n\n旧 prompt 中位为 **12,402 tokens / 650 facts / 93 series**，chk2 训练 prompt 中位为 **1,249 tokens / 9 facts / 2–3 series**，长度接近十倍、事实数约 72 倍；输出目标还从单 topic 分析变成约 8 段的 Minutes。33 条输入并未发生 context overflow，但它们在组合规模、展示 schema 和任务语义上都离开训练分布。尤其 `participants_views` reference 需要参与者叙事，而输入没有对应 evidence，使部分合同不可满足。",
            },
            {"id": "prompt_chart", "type": "chart", "chartId": "prompt_scale_chart", "layout": "full"},
            {
                "id": "finding_decoding",
                "type": "markdown",
                "sourceId": "stress_generations",
                "body": "## Key finding 2 — 直接故障是 reasoning 循环，不是 EOS 或 context overflow\n\n三个模型的 EOS 均配置为 `128001`，vLLM 默认不会忽略 EOS；但 raw completion 中从未出现 `</think>`，说明模型根本没有开始 final answer，也就没有机会在 answer 后结束。99 条生成的 `prompt + completion` 最大约 20,667 tokens，仍低于 24,576 context，并且 metadata 明确显示输入未截断。继续把 completion 上限从 8,192 往上调只会延长循环。",
            },
            {"id": "termination_chart", "type": "chart", "chartId": "termination_health_chart", "layout": "full"},
            {
                "id": "finding_checkpoint",
                "type": "markdown",
                "sourceId": "training_completions",
                "body": "## Key finding 3 — 先保留 checkpoint-150\n\n训练域 steps 131–150 的 80 个 completion 中，**78/80** 有 `</think>` 和非空 answer，只有 **2/80** 达到 4,096-token 上限；rolling-20 reward 在 step 150 达到此前最高值。这个信号并不证明 checkpoint-150 已具备最佳泛化质量，但足以说明：在建立可比的 held-out evaluator 之前，没有证据支持回退或重训。",
            },
            {"id": "reward_chart", "type": "chart", "chartId": "rolling_reward_chart", "layout": "full"},
            {"id": "root_table", "type": "table", "tableId": "root_causes_table", "layout": "full"},
            {
                "id": "scope",
                "type": "markdown",
                "sourceId": "diagnostic_outputs",
                "body": "## Scope, data, and definitions\n\n本诊断覆盖冻结的 3×33 common-test generations、chk2 steps 1–150 completion 工件、1,743 条 compressed chk1 SFT 数据、882 条 GRPO 数据、现行评测/训练配置及本地模型说明。`answer boundary` 指 raw completion 首个 `</think>`；`strict loop` 指最后至少 128 个 tokenizer tokens 可由固定 token 周期精确复现；`hit token limit` 以 generation metadata 或训练 completion 的配置上限判定。所有 token 统计均在 `fomc_trainer` 环境用本地 chk0 tokenizer 重算。",
            },
            {
                "id": "methodology",
                "type": "markdown",
                "sourceId": "diagnostic_outputs",
                "body": "## Methodology\n\n先核对生成元数据、EOS/config 和 prompt+completion 总长度，排除输入截断与 context overflow；再检查 `</think>` 边界、严格周期尾部和重复率，定位终止阶段；随后用同一 tokenizer 比较 SFT、GRPO 与旧共测的 rendered prompt、fact、series 和 target 规模；最后用训练 completion 复算 termination health、rolling-20 reward，并审计 SFT final-answer 合同、EOS 与 BOS parity。notebook 已在 `fomc_trainer` kernel 中从头执行。",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": "## Limitations and robustness\n\n严格周期检测是保守下界；未计入的 6 条输出仍表现出明显近周期漂移。训练 reward 与 chk2 的优化目标同源，不能替代独立 held-out 质量判断。当前分析没有实际运行新的采样解码，因此 `temperature=0.6/top-p=0.95/repetition_penalty=1.05` 是由本地模型说明与故障机制支持的实验候选，而不是已验证赢家。SFT 的 201 条 regex 命中需逐条确定性转换验证，不能直接删除。",
            },
            {
                "id": "data_quality",
                "type": "markdown",
                "sourceId": "sft_dataset",
                "body": "## Training-data improvements\n\n压缩后的 reasoning 本身包含正确的 `</think>` 边界，chat template 会预填 `<think>\n`；问题不是缺标签。真正需要修复的是三类合同漂移：reasoning target 为 **517–1,600 tokens（中位 996）**，却要求 chk2 最多 512；**201/1,743（11.5%）** final answers 命中 JSON/evidence-ID 污染；plain-string SFT 路径产生双 BOS。全部样本仍保留终止 EOS，最长完整序列 4,181，小于 4,608，因此 EOS 监督并未被截断。",
            },
            {"id": "sft_chart", "type": "chart", "chartId": "sft_contract_chart", "layout": "full"},
            {
                "id": "recommendations",
                "type": "markdown",
                "sourceId": "model_readme",
                "body": "## Recommended next steps\n\n1. **不要继续提高 token 上限。** 保留 checkpoint-150 与旧共测工件。\n2. **新建版本化的 190-row 主评测。** 复用 `analysis_grpo/test.jsonl`，三模型使用完全相同的 atomic evidence→analysis prompt，prompt ≤2,560，不允许输入截断；先解析首个 `</think>`，只对 answer 算 factual/numeric/direction 指标。\n3. **先做 12-row 解码 A/B。** 比较 greedy 与 `temperature=0.6, top_p=0.95`，同时比较 `repetition_penalty=1.0/1.05`，固定逐样本 seed；不要把 `</think>` 配为 stop。在胜出解码上，再把现有 system+user 与模型说明建议的 all-in-user 指令布局做独立 A/B，避免同时改变两个因素。\n4. **锁定配置后再跑 190 rows。** 主指标用 boundary、non-empty、plain-text、截断、重复、answer-only 数字支持与方向；按 meeting 做 cluster paired bootstrap。v3 Judge 因与训练 reward 同源，只作次指标。\n5. **统一 reasoning 合同。** 建议目标 768–1,024、reasoning 硬上限 1,536；正式 completion cap 2,048（当前全部 SFT completion 最大 1,784），诊断 A/B 可用 2,304 留出探索余量。\n6. **只有 held-out 结果证明需要时才重训。** 下一 SFT release 先清理 201 条合同命中和单条 `{`，修复双 BOS，再决定是否做 reward v4。",
            },
            {"id": "rollout_table", "type": "table", "tableId": "rollout_plan_table", "layout": "full"},
            {
                "id": "hardware",
                "type": "markdown",
                "body": "## Two-A30 execution note\n\n离线评测时每个 8B 模型放一张 A30，两个模型并行生成，比对完再轮换第三个；不需要 tensor parallel。若进入 chk2 训练，当前 QLoRA policy 占 GPU1、Qwen Judge 占 GPU0 的拆分仍是合理边界，不能同时把两卡都给 policy 而又保留本地 Judge。若最终任务是 Minutes，建议让 chk2 先按 atomic topic 产出 analyses，再由 chk3 聚合为 Minutes，而不是让 chk1/chk2 直接吞入约 650 facts。",
            },
            {
                "id": "questions",
                "type": "markdown",
                "body": "## Further questions\n\n- 12-row A/B 中，采样是否能在不牺牲 answer-only factual accuracy 的情况下把 boundary rate 提升到稳定水平？\n- 190-row held-out 对比中，chk2 相对 chk1 的增益是否跨会议稳定，而不是由少数会议驱动？\n- 清理后的 chk1 数据是否值得完整重训，还是只需修 prompt/解码即可恢复 checkpoint-150 的产品表现？",
            },
        ],
    },
    "snapshot": {
        "version": 1,
        "generatedAt": GENERATED_AT,
        "status": "ready",
        "datasets": {
            "summary": [
                {
                    "length_finished": summary["stress_test"]["length_finished"],
                    "answer_boundaries": summary["stress_test"]["answer_boundaries"],
                    "strict_periodic_suffixes": summary["stress_test"]["strict_periodic_suffixes"],
                    "in_domain_boundary_pct": summary["training_domain_cp150_window"]["answer_boundaries"] / summary["training_domain_cp150_window"]["rows"],
                    "sft_violation_pct": summary["sft_contract"]["plain_text_contract_violations"] / summary["sft_contract"]["rows"],
                }
            ],
            "prompt_scale": prompt_scale,
            "termination_health": termination_health,
            "rolling_reward": rolling_reward,
            "sft_contract": sft_contract,
            "root_causes": root_causes,
            "rollout_plan": rollout_plan,
        },
    },
    "sources": sources,
    "package_info": {
        "originUrl": "artifact://chk2-optimization-diagnosis-20260809T181157Z",
        "controls": {"edit": False, "refresh": False},
    },
}

(OUT_DIR / "artifact.json").write_text(
    json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(OUT_DIR / "artifact.json")
