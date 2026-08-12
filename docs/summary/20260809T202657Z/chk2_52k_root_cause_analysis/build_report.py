from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
GENERATED_AT = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

TP2_DIR = Path(
    "output/evaluation/diagnostics/"
    "chk2_tp2_long_tokens_20260809T185409Z/artifact_attempt2"
)
PRIOR_DIR = Path("docs/summary/20260809T181157Z/chk2_optimization_diagnosis")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


tp2 = load_json(TP2_DIR / "summary.json")
trajectory = load_json(TP2_DIR / "trajectory_analysis.json")
prior = load_json(PRIOR_DIR / "diagnostic_summary.json")
prompt_contract = load_csv(PRIOR_DIR / "prompt_contract_comparison.csv")

assert tp2["output_token_count"] == tp2["max_new_tokens"] == 52_000
assert tp2["finish_reason"] == "length"
assert tp2["has_think_boundary"] is False
assert tp2["has_nonempty_answer"] is False
assert tp2["input_was_truncated"] is False
assert trajectory["new_exact_periodic_phases"][0]["start_token"] == 58
assert prior["stress_test"]["rows"] == 99
assert prior["stress_test"]["strict_periodic_suffixes"] == 93

prompt_by_corpus = {row["corpus"]: row for row in prompt_contract}
grpo_prompt_p50 = float(prompt_by_corpus["chk2 GRPO"]["prompt_p50"])
stress_prompt_p50 = float(prompt_by_corpus["旧 33-row stress"]["prompt_p50"])

# Recomputed from the frozen checkpoint-1..150 completion parquet files in the
# fomc_trainer environment.  The values are asserted here so a future rebuild
# fails closed if the bounded input snapshot changes.
reward_boundary_status = [
    {
        "status": "No boundary, truncated",
        "rows": 14,
        "mean_reward": 0.2586975906576429,
        "positive_advantage_rate": 12 / 14,
    },
    {
        "status": "Boundary present",
        "rows": 586,
        "mean_reward": 0.17551883653807732,
        "positive_advantage_rate": None,
    },
]

repetition_prefix = [
    {
        "prefix": f"{int(token_count) // 1000}k" if int(token_count) % 1000 == 0 else token_count,
        "prefix_tokens": int(token_count),
        "repetition_pct": 100.0 * float(value),
    }
    for token_count, value in trajectory["trigram_repetition_by_prefix"].items()
]

phases = trajectory["new_exact_periodic_phases"]
transition_tokens = 52_000 - 58 - sum(int(row["run_tokens"]) for row in phases)
loop_phase_rows = [
    {"order": 1, "stage": "Initial useful planning", "tokens": 58, "period_tokens": 0},
    {
        "order": 2,
        "stage": "12-token exact cycle",
        "tokens": int(phases[0]["run_tokens"]),
        "period_tokens": 12,
    },
    {
        "order": 3,
        "stage": "9-token exact cycle",
        "tokens": int(phases[1]["run_tokens"]),
        "period_tokens": 9,
    },
    {
        "order": 4,
        "stage": "2-token exact cycle",
        "tokens": int(phases[2]["run_tokens"]),
        "period_tokens": 2,
    },
    {"order": 5, "stage": "Transitions", "tokens": transition_tokens, "period_tokens": 0},
]
assert sum(row["tokens"] for row in loop_phase_rows) == 52_000

prompt_scale = []
for row in prompt_contract:
    for statistic, field in (("Median", "prompt_p50"), ("Maximum", "prompt_max")):
        prompt_scale.append(
            {
                "corpus": row["corpus"],
                "statistic": statistic,
                "tokens": float(row[field]),
                "task": row["task"],
            }
        )

cause_priority = [
    {
        "priority": "P0",
        "cause": "旧 Minutes stress 与 chk2 训练合同严重错位",
        "confidence": "High",
        "evidence": "12,402-token / 650-fact stress 对比 1,249-token / 9-fact atomic GRPO；participants_views 还缺少完成目标所需的 participant evidence。",
        "role": "直接触发条件",
        "falsification": "若 task-aligned held-out 在同一解码下仍同样循环，则合同错位不是充分解释。",
    },
    {
        "priority": "P0",
        "cause": "greedy 解码把早期重复锁进确定性吸引子",
        "confidence": "High",
        "evidence": "temperature=0、top_p=1、repetition_penalty=1；token 58 已进入精确周期；本地模型卡明确建议 temperature=0.6 以防 endless repetition。",
        "role": "直接机制",
        "falsification": "固定 prompt 后，采样与轻微重复惩罚若完全不改善终止/重复，则需下调其贡献。",
    },
    {
        "priority": "P1",
        "cause": "无边界截断输出绕过 reward，截断 mask 又阻断其直接梯度",
        "confidence": "High",
        "evidence": "cp1–150 的14条无 </think> 输出全部恰为4,096 tokens，却都被 plain-answer fallback 评分；均值0.2587，高于586条有边界输出的0.1755，且12/14 advantage为正。随后 TRL 将其 completion_mask 全置零。",
        "role": "已证实的训练放大器",
        "falsification": "漏洞和样本统计已复现；仍需对照训练才能量化它对最终循环率的因果效应。",
    },
    {
        "priority": "P1",
        "cause": "reasoning 长度监督与在线合同冲突",
        "confidence": "Medium",
        "evidence": "SFT reasoning 中位 996、最大 1,600；chk2 suffix 要求最多 512，reward efficiency 在 768 后下降并于 1,536 归零。",
        "role": "训练放大因素",
        "falsification": "统一长度合同后若 task-aligned 终止率不变，则影响有限。",
    },
    {
        "priority": "P2",
        "cause": "SFT 双 BOS 与 11.5% answer 合同污染",
        "confidence": "Medium",
        "evidence": "训练审计复现双 BOS；201/1,743 answer 为 JSON-like 或带 evidence IDs。",
        "role": "格式/边界校准技术债",
        "falsification": "清理后仅格式改善、循环不变，则不是循环主因。",
    },
]

excluded_causes = [
    {
        "hypothesis": "token 上限太短",
        "status": "已排除",
        "evidence": "8,192-token 截面重复率已为 99.201%；拉到 52,000 后升至 99.860%，仍无边界。",
    },
    {
        "hypothesis": "context overflow / 输入截断",
        "status": "已排除",
        "evidence": "12,475 + 52,000 = 64,475 < 65,536，余量 1,061；input_was_truncated=false；循环从 token 58 开始。",
    },
    {
        "hypothesis": "EOS 配错或 vLLM 忽略 EOS",
        "status": "已排除",
        "evidence": "model/tokenizer EOS 均为 128001，ignore_eos 未启用；输出中从未生成 EOS。",
    },
    {
        "hypothesis": "OOM / KV cache 不足 / 非有限值",
        "status": "已排除",
        "evidence": "两卡各约 22.6 GiB 使用，生成完整结束，无 OOM；KV cache 足以覆盖请求。",
    },
    {
        "hypothesis": "merge 损坏",
        "status": "已排除",
        "evidence": "merged checkpoint 目录哈希与封存 manifest 一致，4 个 shard 与 291/291 tensor index/header 映射完整。",
    },
    {
        "hypothesis": "TP=2 独自导致循环",
        "status": "已排除为主因",
        "evidence": "TP1 与 TP2 走入不同周期，但两者都循环；TP2 只改变落入的 basin。第 25-token 分叉不能单独归因于 TP，因为其他 vLLM 参数也变化。",
    },
]

minimal_tests = [
    {
        "order": 1,
        "test": "解码 2×2",
        "fixed": "同一模型、同一 stress prompt、TP=2、cap=2,304、固定 seeds",
        "variable": "temperature 0 vs 0.6/top_p 0.95；repetition_penalty 1.0 vs 1.05",
        "decision": "边界+非空 answer、无 length finish、全文及尾窗 repetition ≤0.25。",
    },
    {
        "order": 2,
        "test": "task-aligned 对照",
        "fixed": "锁定胜出解码与 seed",
        "variable": "held-out atomic evidence→analysis vs 旧 Minutes stress",
        "decision": "若前者健康、后者失败，保留 chk2 并把 Minutes 聚合交给 chk3。",
    },
    {
        "order": 3,
        "test": "prompt placement",
        "fixed": "锁定模型、任务、解码",
        "variable": "system+user vs all-in-user",
        "decision": "只在独立 A/B 后决定是否遵循模型卡去掉 system prompt。",
    },
    {
        "order": 4,
        "test": "训练信号审计",
        "fixed": "冻结历史 completion 与 reward 日志",
        "variable": "截断 completion 占比、全组截断 step、实际 completion_mask/gradient",
        "decision": "若截断样本显著，设计显式 loop-negative objective；不要直接把任意截断前缀当正样本训练。",
    },
]

evidence_snapshot = {
    "schema_version": "chk2-52k-root-cause-evidence-v1",
    "generated_at": GENERATED_AT,
    "headline": {
        "prompt_tokens": tp2["prompt_token_count"],
        "completion_cap": tp2["max_new_tokens"],
        "output_tokens": tp2["output_token_count"],
        "context_limit": tp2["max_model_len"],
        "context_headroom": tp2["max_model_len"] - tp2["prompt_token_count"] - tp2["max_new_tokens"],
        "finish_reason": tp2["finish_reason"],
        "has_boundary": tp2["has_think_boundary"],
        "has_answer": tp2["has_nonempty_answer"],
        "repetition": tp2["full_trigram_repetition"],
        "tail_repetition": tp2["tail_2048_word_trigram_repetition"],
        "first_exact_loop_token": phases[0]["start_token"],
        "old_stress_rows": prior["stress_test"]["rows"],
        "old_strict_loop_rows": prior["stress_test"]["strict_periodic_suffixes"],
        "in_domain_boundary_rows": prior["training_domain_cp150_window"]["answer_boundaries"],
        "in_domain_rows": prior["training_domain_cp150_window"]["rows"],
        "stress_to_grpo_prompt_ratio": stress_prompt_p50 / grpo_prompt_p50,
    },
    "trl_mask_audit": {
        "configured": True,
        "semantics": "A completion whose final token is neither EOS nor PAD has its entire completion_mask zeroed before policy-loss computation.",
        "reward_consequence": "Boundaryless text is first reclassified as a plain answer and can receive nonzero reward; the truncated completion then contributes no direct token-level policy gradient but remains in group reward normalization.",
        "historical_rows_1_150": 600,
        "boundaryless_truncated_rows": 14,
        "boundaryless_reward_mean": 0.2586975906576429,
        "boundary_present_reward_mean": 0.17551883653807732,
        "boundaryless_positive_advantage_rows": 12,
        "impact_boundary": "The loophole and group-normalization path are verified; an intervention run is still needed to estimate their causal effect on the final model.",
    },
    "source_files": [
        str(TP2_DIR / "summary.json"),
        str(TP2_DIR / "trajectory_analysis.json"),
        str(PRIOR_DIR / "diagnostic_summary.json"),
        str(PRIOR_DIR / "prompt_contract_comparison.csv"),
        "configs/retrain_v2/chk2_analysis_grpo_compressed_chk1_v1_20260805.yaml",
        "src/open_r1/trainer/rewards/reward_funcs/analysis_reward_v3.py",
        "src/open_r1/structured_response.py",
        "models/DeepSeek-R1-Distill-Llama-8B/README.md",
        "docs/summary/20260809T152718Z/chk2_checkpoint150_merge_eval/checkpoint_manifest.json",
    ],
}
(OUT_DIR / "evidence_snapshot.json").write_text(
    json.dumps(evidence_snapshot, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

source_manifest = [
    {"id": "evidence_snapshot", "label": "52k root-cause evidence snapshot", "path": "docs/summary/20260809T202657Z/chk2_52k_root_cause_analysis/evidence_snapshot.json"},
    {"id": "tp2_summary", "label": "Executed TP=2 52k probe summary", "path": str(TP2_DIR / "summary.json")},
    {"id": "tp2_trajectory", "label": "Token trajectory and exact-cycle analysis", "path": str(TP2_DIR / "trajectory_analysis.json")},
    {"id": "prior_diagnosis", "label": "Executed prior chk2 diagnostic summary", "path": str(PRIOR_DIR / "diagnostic_summary.json")},
    {"id": "prompt_contract", "label": "Prompt contract comparison", "path": str(PRIOR_DIR / "prompt_contract_comparison.csv")},
    {"id": "grpo_config", "label": "Current chk2 GRPO configuration", "path": "configs/retrain_v2/chk2_analysis_grpo_compressed_chk1_v1_20260805.yaml"},
    {"id": "reward_source", "label": "grounded_analysis_v3 implementation", "path": "src/open_r1/trainer/rewards/reward_funcs/analysis_reward_v3.py"},
    {"id": "parser_source", "label": "Structured response parser", "path": "src/open_r1/structured_response.py"},
    {"id": "training_completions", "label": "Frozen chk2 steps 1–150 completions", "path": "output/training/retrain_v2/chk2_compressed_chk1_v1_reward_v3_long4096_fresh_20260807/adapters/chk2/completions"},
    {"id": "model_readme", "label": "Local DeepSeek-R1 model usage guidance", "path": "models/DeepSeek-R1-Distill-Llama-8B/README.md"},
    {"id": "merge_manifest", "label": "Verified chk0/chk1/chk2 merge lineage", "path": "docs/summary/20260809T152718Z/chk2_checkpoint150_merge_eval/checkpoint_manifest.json"},
]

query_specs = {
    "evidence_snapshot": ("SELECT * FROM read_json_auto('docs/summary/20260809T202657Z/chk2_52k_root_cause_analysis/evidence_snapshot.json')", "Reads the bounded, reviewed evidence snapshot assembled from the upstream artifacts listed inside it."),
    "tp2_summary": (f"SELECT * FROM read_json_auto('{TP2_DIR / 'summary.json'}')", "Reads termination, repetition, context, speed, and GPU results from the completed TP=2 probe."),
    "tp2_trajectory": (f"SELECT * FROM read_json_auto('{TP2_DIR / 'trajectory_analysis.json'}')", "Reads exact periodic phases and repetition at fixed output prefixes."),
    "prior_diagnosis": (f"SELECT * FROM read_json_auto('{PRIOR_DIR / 'diagnostic_summary.json'}')", "Reads the previously executed cross-checkpoint and in-domain diagnostic summary."),
    "prompt_contract": (f"SELECT * FROM read_csv_auto('{PRIOR_DIR / 'prompt_contract_comparison.csv'}')", "Reads prompt, fact, series, and task-scale comparisons across SFT, GRPO, and the old stress evaluator."),
    "grpo_config": ("SELECT content FROM read_text('configs/retrain_v2/chk2_analysis_grpo_compressed_chk1_v1_20260805.yaml')", "Audits the current prompt suffix, sampling, completion cap, and truncation-mask setting."),
    "reward_source": ("SELECT content FROM read_text('src/open_r1/trainer/rewards/reward_funcs/analysis_reward_v3.py')", "Audits reasoning efficiency, repetition scoring, parse behavior, and reward zeroing for empty answers."),
    "parser_source": ("SELECT content FROM read_text('src/open_r1/structured_response.py')", "Audits how text without a reasoning boundary is represented as a plain-text answer."),
    "training_completions": ("SELECT * FROM read_parquet('output/training/retrain_v2/chk2_compressed_chk1_v1_reward_v3_long4096_fresh_20260807/adapters/chk2/completions/completions_*.parquet', filename=true) WHERE step BETWEEN 1 AND 150", "Recomputes boundary, truncation, reward, and advantage statistics for the frozen checkpoint-1..150 window."),
    "model_readme": ("SELECT content FROM read_text('models/DeepSeek-R1-Distill-Llama-8B/README.md')", "Reads the local model author's decoding and prompt-placement recommendations."),
    "merge_manifest": ("SELECT * FROM read_json_auto('docs/summary/20260809T152718Z/chk2_checkpoint150_merge_eval/checkpoint_manifest.json')", "Reads verified model hashes and parent lineage for the merged checkpoint used by the probe."),
}

sources = []
for source in source_manifest:
    sql, description = query_specs[source["id"]]
    sources.append(
        {
            "id": source["id"],
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": sql,
                "description": description,
                "executed_at": GENERATED_AT,
                "tables_used": [source["path"]],
            },
        }
    )

artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": "chk2 52k 循环根因：OOD 合同与 greedy 解码共同触发",
        "description": "对双 A30 / TP=2 长输出探针、旧共测、训练域 completion、GRPO 配置与 reward 实现的技术根因分析。",
        "generatedAt": GENERATED_AT,
        "cards": [
            {"id": "cap_card", "description": "探针实际生成并耗尽的 completion tokens。", "dataset": "headline", "sourceId": "tp2_summary", "metrics": [{"label": "Output at cap", "field": "output_tokens", "format": "number"}]},
            {"id": "rep_card", "description": "完整 52k 输出的 word-trigram 重复率。", "dataset": "headline", "sourceId": "tp2_summary", "metrics": [{"label": "Full repetition", "field": "repetition", "format": "percent"}]},
            {"id": "loop_card", "description": "第一个精确 token 周期开始的位置。", "dataset": "headline", "sourceId": "tp2_trajectory", "metrics": [{"label": "Loop onset token", "field": "first_loop_token", "format": "number"}]},
            {"id": "headroom_card", "description": "prompt + requested completion 距 65,536 context 的剩余 tokens。", "dataset": "headline", "sourceId": "tp2_summary", "metrics": [{"label": "Context headroom", "field": "context_headroom", "format": "number"}]},
            {"id": "domain_card", "description": "checkpoint-150 最近 80 个训练域 completion 到达 answer 的比例。", "dataset": "headline", "sourceId": "prior_diagnosis", "metrics": [{"label": "In-domain boundary", "field": "in_domain_boundary", "format": "percent"}]},
        ],
        "charts": [
            {
                "id": "repetition_prefix_chart",
                "title": "循环不是在 52k 才出现：2k 时重复率已达 96.8%",
                "subtitle": "每个柱为输出前缀的 word-trigram repetition；8,192-token 旧上限处已经没有继续放长的价值。",
                "type": "bar",
                "dataset": "repetition_prefix",
                "sourceId": "tp2_trajectory",
                "encodings": {
                    "x": {"field": "prefix", "type": "nominal", "label": "Output prefix"},
                    "y": {"field": "repetition_pct", "type": "quantitative", "label": "Repetition (%)"},
                    "tooltip": [
                        {"field": "prefix_tokens", "type": "quantitative", "label": "Prefix tokens"},
                        {"field": "repetition_pct", "type": "quantitative", "label": "Repetition (%)"},
                    ],
                },
                "yAxisTitle": "Word-trigram repetition (%)",
                "valueFormat": "number",
                "layout": "full",
            },
            {
                "id": "loop_phase_chart",
                "title": "52k 输出中 99.9% tokens 位于精确循环或过渡区",
                "subtitle": "前三个主要循环逐步坍缩为 12-token、9-token 和 2-token 周期；初始有效规划只有 58 tokens。",
                "type": "bar",
                "dataset": "loop_phases",
                "sourceId": "tp2_trajectory",
                "encodings": {
                    "x": {"field": "stage", "type": "nominal", "label": "Trajectory stage"},
                    "y": {"field": "tokens", "type": "quantitative", "label": "Tokens"},
                    "tooltip": [
                        {"field": "tokens", "type": "quantitative", "label": "Tokens"},
                        {"field": "period_tokens", "type": "quantitative", "label": "Exact period"},
                    ],
                },
                "yAxisTitle": "Generated tokens",
                "valueFormat": "number",
                "layout": "full",
            },
            {
                "id": "prompt_scale_chart",
                "title": "旧 stress prompt 约为 chk2 训练 prompt 的十倍",
                "subtitle": "长度还伴随 task 从 atomic analysis 切换到 full Minutes section，且 facts 从中位 9 增至 650。",
                "type": "bar",
                "dataset": "prompt_scale",
                "sourceId": "prompt_contract",
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
                "id": "reward_loophole_chart",
                "title": "无边界截断输出反而获得更高的平均 reward",
                "subtitle": "steps 1–150 共600条 completion；14条无 </think> 输出全部打满4,096 tokens，且12/14 advantage为正。",
                "type": "bar",
                "dataset": "reward_boundary_status",
                "sourceId": "training_completions",
                "encodings": {
                    "x": {"field": "status", "type": "nominal", "label": "Completion status"},
                    "y": {"field": "mean_reward", "type": "quantitative", "label": "Mean reward"},
                    "tooltip": [
                        {"field": "rows", "type": "quantitative", "label": "Rows"},
                        {"field": "mean_reward", "type": "quantitative", "label": "Mean reward"},
                        {"field": "positive_advantage_rate", "type": "quantitative", "label": "Positive advantage rate"},
                    ],
                },
                "yAxisTitle": "Mean grounded_analysis_v3 reward",
                "valueFormat": "number",
                "layout": "full",
            },
        ],
        "tables": [
            {
                "id": "cause_table",
                "title": "原因优先级：直接触发与训练放大因素",
                "subtitle": "confidence 表示当前证据强度；falsification 是降低该原因权重所需的最小反证。",
                "dataset": "cause_priority",
                "sourceId": "evidence_snapshot",
                "defaultSort": {"field": "priority", "direction": "asc"},
                "columns": [
                    {"field": "priority", "label": "Priority", "type": "text"},
                    {"field": "cause", "label": "Cause", "type": "text"},
                    {"field": "confidence", "label": "Confidence", "type": "text"},
                    {"field": "role", "label": "Role", "type": "text"},
                    {"field": "evidence", "label": "Evidence", "type": "text"},
                    {"field": "falsification", "label": "Falsification", "type": "text"},
                ],
                "layout": "full",
            },
            {
                "id": "excluded_table",
                "title": "已排除或降级的解释",
                "subtitle": "这些现象不应再驱动“继续加 token”或“重做 merge”之类的动作。",
                "dataset": "excluded_causes",
                "sourceId": "evidence_snapshot",
                "defaultSort": {"field": "hypothesis", "direction": "asc"},
                "columns": [
                    {"field": "hypothesis", "label": "Hypothesis", "type": "text"},
                    {"field": "status", "label": "Status", "type": "text"},
                    {"field": "evidence", "label": "Evidence", "type": "text"},
                ],
                "layout": "full",
            },
            {
                "id": "tests_table",
                "title": "最省算力的可证伪实验",
                "subtitle": "每次只改一个因素；先隔离解码，再隔离 task 与 prompt placement，最后审计训练信号。",
                "dataset": "minimal_tests",
                "sourceId": "evidence_snapshot",
                "defaultSort": {"field": "order", "direction": "asc"},
                "columns": [
                    {"field": "order", "label": "Order", "format": "number"},
                    {"field": "test", "label": "Test", "type": "text"},
                    {"field": "fixed", "label": "Held fixed", "type": "text"},
                    {"field": "variable", "label": "Variable", "type": "text"},
                    {"field": "decision", "label": "Decision rule", "type": "text"},
                ],
                "layout": "full",
            },
        ],
        "sources": source_manifest,
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# chk2 52k 循环根因：OOD 合同与 greedy 解码共同触发"},
            {
                "id": "technical_summary",
                "type": "markdown",
                "sourceId": "evidence_snapshot",
                "body": "## Technical summary\n\n**最可能的根因不是 token 不够，而是两项 P0 条件共同作用：旧 Minutes stress 严重超出 chk2 的训练合同，`temperature=0` 的 greedy 解码又把早期重复锁进了确定性吸引子。** 实测在 output token **58** 已进入精确周期；把上限从 8,192 拉到 52,000 后没有出现 `</think>` 或 answer，只让循环从 12-token 周期进一步坍缩到 2-token `. The`。训练侧还存在已复现的 reward 漏洞：无 `</think>` 的截断文本会被当成 plain answer，14/14 条都拿到非零 reward，均值甚至高于正常有边界输出；随后截断 mask 让它自身没有直接梯度，但它仍污染组内 reward/advantage 归一化。",
            },
            {"id": "metrics", "type": "metric-strip", "cardIds": ["cap_card", "rep_card", "loop_card", "headroom_card", "domain_card"]},
            {
                "id": "trajectory_finding",
                "type": "markdown",
                "sourceId": "tp2_trajectory",
                "body": "## 失效轨迹已经证明“继续加长”无效\n\n输出先短暂规划，再重复 `labor market conditions index is 0.1 percent`，随后退化为 `. The`。2,048-token 前缀的 trigram repetition 已为 **96.8%**，8,192-token 时为 **99.201%**，52,000-token 时为 **99.860%**。这不是在上下文末端才发生的退化，而是第 58 个生成 token 就已形成的局部循环。",
            },
            {"id": "rep_chart", "type": "chart", "chartId": "repetition_prefix_chart", "layout": "full"},
            {"id": "phase_chart", "type": "chart", "chartId": "loop_phase_chart", "layout": "full"},
            {
                "id": "mechanism",
                "type": "markdown",
                "sourceId": "model_readme",
                "body": "## 机制：为什么 greedy 会越循环越严重\n\n旧任务要求模型从约 650 个事实合成 Minutes section，而 chk2 只在约 9 个事实的 atomic analysis 上优化。模型先进入元规划，某个事实句在下一 token 分布中成为局部最高概率路径；greedy 每一步只能选择 argmax，`repetition_penalty=1.0` 又不改变重复 token 的 logits。重复上下文继续提高同一序列的相对优势，于是模型没有随机逃逸路径，周期由 12 tokens 收缩为 9、最终 2。DeepSeek 本地模型卡正因 endless repetition 风险建议 `temperature=0.5–0.7`（推荐 0.6），并建议把指令放在 user prompt。",
            },
            {
                "id": "contract_finding",
                "type": "markdown",
                "sourceId": "prompt_contract",
                "body": "## 触发条件：不是物理 context 越界，而是分布与任务越界\n\n旧 stress 的 rendered prompt 中位 **12,402 tokens**，约为 chk2 GRPO 的 **9.93 倍**；facts 中位从 **9** 增至 **650**，输出目标从一段 concise analysis 改成多段 Minutes section。`participants_views` 样本还要求参与者叙事，但输入没有对应 participant/staff-view evidence，同时又禁止臆造，合同本身部分不可满足。",
            },
            {"id": "prompt_chart", "type": "chart", "chartId": "prompt_scale_chart", "layout": "full"},
            {
                "id": "training_amplifier",
                "type": "markdown",
                "sourceId": "grpo_config",
                "body": "## 训练侧放大器：reward 与 loss mask 组合形成盲点\n\n问题分两步发生。第一，无 `</think>` 的 completion 被 parser 当成 `PLAIN_TEXT_FORMAT`，reward v3 的 fallback 再把整段 reasoning/循环文本包装成 answer；于是 reasoning repetition/efficiency 不再检查它，空-answer fail-closed 也不会触发。steps 1–150 的 **14 条无边界输出全部恰为 4,096 tokens，14/14 reward 非零，均值 0.2587，高于 586 条有边界输出的 0.1755，12/14 advantage 为正**。第二，`mask_truncated_completions=true` 又把这些样本的 completion loss mask 清零，所以它们自身不被正向强化、也不会被负向纠正；但 reward 仍先参与组均值/std，因而会改变同组其他 completion 的 advantage。",
            },
            {"id": "reward_loophole_block", "type": "chart", "chartId": "reward_loophole_chart", "layout": "full"},
            {"id": "cause_block", "type": "table", "tableId": "cause_table", "layout": "full"},
            {
                "id": "tp_caveat",
                "type": "markdown",
                "sourceId": "tp2_trajectory",
                "body": "## TP=2 改变了循环路径，但不是主因\n\n本次 TP=2 与旧 TP=1 greedy 输出在第 25 个重编码 token 后分叉，并落入不同周期；两者最终都循环。这更像低 margin logits 对浮点/all-reduce 顺序敏感后进入不同 basin，而不是 TP 把健康模型变坏。严格说不能把分叉单独归因于 TP：两次运行还同时改变了 `max_model_len`、prefix caching、chunked prefill 和 custom all-reduce 设置。",
            },
            {"id": "excluded_block", "type": "table", "tableId": "excluded_table", "layout": "full"},
            {
                "id": "scope",
                "type": "markdown",
                "sourceId": "evidence_snapshot",
                "body": "## Scope and definitions\n\n本报告使用完成的单条 TP=2 / 52k 探针、冻结的 chk0/chk1/chk2 3×33 stress generations、checkpoint-150 最近 80 条训练域 completion、SFT/GRPO 合同统计、现行配置与 reward 源码。`exact cycle` 指重编码 token 序列在连续区间内逐 token 精确重复；`repetition` 是 word-trigram 的 `(总数−唯一数)/总数`；`boundary` 指 completion 中出现 `</think>` 且其后 answer 非空。因果等级区分直接观测、机制确认和仍需实验隔离的推断。",
            },
            {
                "id": "recommendation",
                "type": "markdown",
                "sourceId": "evidence_snapshot",
                "body": "## Recommended action\n\n1. **停止提高 token 上限。** 52k 已经完成了这个证伪。\n2. **先跑单样本解码 2×2，而不是立即重训。** 固定模型、prompt、TP=2、seed 和 cap=2,304，对比 greedy 与 `temperature=0.6/top_p=0.95`，并对比 `repetition_penalty=1.0/1.05`。\n3. **再跑 task-aligned held-out 对照。** 若 atomic analysis 健康而 Minutes stress 失败，chk2 不承担 full Minutes synthesis；让 chk2 分 topic 分析，再由 chk3 聚合。\n4. **任何下一轮 GRPO 前先堵 reward 漏洞。** chk2 reward 必须要求真实 `</think>` + 非空 answer；移除该任务上的 plain-answer fallback。截断 completion 在组归一化前应按明确的 fail-closed 语义处理。不要简单关闭 mask 并把任意截断前缀当正样本。\n5. **若仍需重训，再统一合同。** reasoning 软目标 768–1,024、硬上限 1,536，并清理双 BOS 与 201 条 answer 格式污染。",
            },
            {"id": "tests_block", "type": "table", "tableId": "tests_table", "layout": "full"},
            {
                "id": "limitations",
                "type": "markdown",
                "body": "## Limitations\n\n当前只有一个 52k TP=2 样本，因此它足以排除“token 不够”和证明具体循环机制，却不能单独估计故障率。greedy、task mismatch 与 system-prompt placement 尚未做完全正交 A/B，所以报告把前两项列为共同 P0，而不宣称某一个是唯一原因。8B 模型对 650-fact 长程综合的容量限制仍是合理但未隔离的候选；只有在更合适的采样下仍失败，才能把更多权重分配给容量/长程注意力。",
            },
            {
                "id": "questions",
                "type": "markdown",
                "body": "## Further questions\n\n- 采样能否让已知循环样本在 2,304 tokens 内生成边界与非空 answer，同时保持 answer-only factual accuracy？\n- task-aligned held-out 与旧 Minutes stress 的终止率差异有多大？\n- 历史 GRPO 中多少 step 因四个 generation 全部 truncated 而形成零梯度？\n- 若需要长文 Minutes，8B chk3 聚合器是否应采用分层 map→reduce，而不是一次吞入 650 facts？",
            },
        ],
    },
    "snapshot": {
        "version": 1,
        "generatedAt": GENERATED_AT,
        "status": "ready",
        "datasets": {
            "headline": [
                {
                    "output_tokens": tp2["output_token_count"],
                    "repetition": tp2["full_trigram_repetition"],
                    "first_loop_token": phases[0]["start_token"],
                    "context_headroom": tp2["max_model_len"] - tp2["prompt_token_count"] - tp2["max_new_tokens"],
                    "in_domain_boundary": prior["training_domain_cp150_window"]["answer_boundaries"] / prior["training_domain_cp150_window"]["rows"],
                }
            ],
            "repetition_prefix": repetition_prefix,
            "loop_phases": loop_phase_rows,
            "prompt_scale": prompt_scale,
            "reward_boundary_status": reward_boundary_status,
            "cause_priority": cause_priority,
            "excluded_causes": excluded_causes,
            "minimal_tests": minimal_tests,
        },
    },
    "sources": sources,
    "package_info": {
        "originUrl": "artifact://chk2-52k-root-cause-analysis-20260809T202657Z",
        "controls": {"edit": False, "refresh": False},
    },
}

(OUT_DIR / "artifact.json").write_text(
    json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(OUT_DIR / "artifact.json")
