from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RUN_REL = Path(
    "output/training/retrain_v2/"
    "chk1_clean_v2_override_full3ep_lr1e6_maxlen7168_20260810/"
    "adapters/chk1"
)
RUN = ROOT / RUN_REL
GENERATED_AT = "2026-08-10T12:30:00Z"

PROBES = {
    80: Path("docs/summary/20260810T101500Z"),
    150: Path("docs/summary/20260810T110300Z"),
    200: Path("docs/summary/20260810T113500Z"),
    255: Path("docs/summary/20260810T121200Z"),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


trainer_state = read_json(RUN / "trainer_state.json")
eval_results = read_json(RUN / "eval_results.json")
train_results = read_json(RUN / "train_results.json")

train_by_step = {
    int(row["step"]): float(row["loss"])
    for row in trainer_state["log_history"]
    if "loss" in row and "eval_loss" not in row
}
periodic_eval = [
    row for row in trainer_state["log_history"] if "eval_loss" in row
]

loss_rows: list[dict] = []
for row in periodic_eval:
    step = int(row["step"])
    trailing10 = [train_by_step[index] for index in range(step - 9, step + 1)]
    trailing20 = [
        train_by_step[index] for index in range(max(1, step - 19), step + 1)
    ]
    loss_rows.append(
        {
            "step": step,
            "epoch": float(row["epoch"]),
            "train_loss_raw": train_by_step[step],
            "train_loss_rolling10": sum(trailing10) / len(trailing10),
            "train_loss_rolling20": sum(trailing20) / len(trailing20),
            "eval_loss": float(row["eval_loss"]),
            "eval_accuracy": float(row["eval_mean_token_accuracy"]),
            "is_final_eval": False,
        }
    )

final_trailing10 = [train_by_step[index] for index in range(246, 256)]
final_trailing20 = [train_by_step[index] for index in range(236, 256)]
loss_rows.append(
    {
        "step": 255,
        "epoch": 3.0,
        "train_loss_raw": train_by_step[255],
        "train_loss_rolling10": sum(final_trailing10) / 10,
        "train_loss_rolling20": sum(final_trailing20) / 20,
        "eval_loss": float(eval_results["eval_loss"]),
        "eval_accuracy": None,
        "is_final_eval": True,
    }
)

best_row = min(loss_rows, key=lambda row: row["eval_loss"])

probe_rows: list[dict] = []
for checkpoint, summary_dir in PROBES.items():
    probe_dir = summary_dir / f"chk1_lr1e6_cp{checkpoint}_probe"
    summary_path = ROOT / probe_dir / "summary.json"
    comparison_rel = (
        summary_dir / f"chk1_lr1e6_cp{checkpoint}_vs_chk0_comparison.json"
    )
    comparison_path = ROOT / comparison_rel
    summary = read_json(summary_path)
    comparison = read_json(comparison_path)
    probe_rows.append(
        {
            "checkpoint": checkpoint,
            "status": comparison["status"],
            "mean_repetition": summary["mean_full_token_4gram_repetition"],
            "max_repetition": summary["max_full_token_4gram_repetition"],
            "mean_delta_vs_chk0": comparison["aggregate_deltas"][
                "mean_full_repetition"
            ],
            "max_case_delta_vs_chk0": max(
                row["full_repetition_delta"] for row in comparison["case_deltas"]
            ),
            "eos_rate": summary["eos_rate"],
            "contract_valid_rate": summary["contract_valid_rate"],
            "cap_rate": summary["cap_rate"],
            "periodic_tail_rate": summary["periodic_tail_rate"],
            "catastrophic_cases": summary["catastrophic_cases"],
            "results_sha256": summary["results"]["sha256"],
            "comparison_path": str(comparison_rel),
            "comparison_sha256": sha256(comparison_path),
        }
    )

eval_by_step = {row["step"]: row["eval_loss"] for row in loss_rows}
train20_by_step = {row["step"]: row["train_loss_rolling20"] for row in loss_rows}
probe_by_step = {row["checkpoint"]: row for row in probe_rows}

selection_rows: list[dict] = []
for rank, checkpoint in enumerate((200, 255, 150, 80), start=1):
    probe = probe_by_step[checkpoint]
    selection_rows.append(
        {
            "rank": rank,
            "checkpoint": checkpoint,
            "eval_loss": eval_by_step[checkpoint],
            "eval_delta_vs_cp200": eval_by_step[checkpoint] - eval_by_step[200],
            "train_loss_rolling20": train20_by_step[checkpoint],
            "mean_repetition": probe["mean_repetition"],
            "max_repetition": probe["max_repetition"],
            "mean_delta_vs_chk0": probe["mean_delta_vs_chk0"],
            "max_case_delta_vs_chk0": probe["max_case_delta_vs_chk0"],
            "hard_gate": probe["status"],
            "decision": {
                200: "推荐进入 chk2",
                255: "安全但无额外拟合收益",
                150: "保守备选",
                80: "行为稳定但明显欠拟合",
            }[checkpoint],
        }
    )

epoch_rows = []
for epoch_index, (start, end) in enumerate(((1, 85), (86, 170), (171, 255)), 1):
    values = [train_by_step[step] for step in range(start, end + 1)]
    epoch_rows.append(
        {
            "epoch": epoch_index,
            "start_step": start,
            "end_step": end,
            "mean_train_loss": statistics.mean(values),
            "median_train_loss": statistics.median(values),
        }
    )

loss_series = []
for row in loss_rows:
    common = {
        "step": row["step"],
        "epoch": row["epoch"],
        "eval_accuracy": row["eval_accuracy"],
    }
    loss_series.append(
        {
            **common,
            "series": "Train loss（trailing 10）",
            "value": row["train_loss_rolling10"],
        }
    )
    loss_series.append(
        {**common, "series": "Eval loss", "value": row["eval_loss"]}
    )

baseline_summary = read_json(
    ROOT / "docs/summary/20260809T234411Z/chk0_baseline_probe/summary.json"
)
repetition_rows = [
    {
        "checkpoint_label": "chk0",
        "checkpoint": 0,
        "mean_repetition": baseline_summary["mean_full_token_4gram_repetition"],
        "max_repetition": baseline_summary["max_full_token_4gram_repetition"],
        "eval_loss": None,
        "hard_gate": "baseline",
    }
]
for row in sorted(probe_rows, key=lambda item: item["checkpoint"]):
    repetition_rows.append(
        {
            "checkpoint_label": f"cp{row['checkpoint']}",
            "checkpoint": row["checkpoint"],
            "mean_repetition": row["mean_repetition"],
            "max_repetition": row["max_repetition"],
            "eval_loss": eval_by_step[row["checkpoint"]],
            "hard_gate": row["status"],
        }
    )

write_jsonl(OUT / "loss_snapshot.jsonl", loss_rows)
write_jsonl(OUT / "probe_snapshot.jsonl", probe_rows)
write_jsonl(OUT / "selection_snapshot.jsonl", selection_rows)

headline = [
    {
        "selected_checkpoint": 200,
        "selected_eval_loss": eval_by_step[200],
        "selected_mean_repetition": probe_by_step[200]["mean_repetition"],
        "post200_eval_change": eval_by_step[255] - eval_by_step[200],
        "probe_pass_rate": sum(row["status"] == "passed" for row in probe_rows)
        / len(probe_rows),
    }
]

loss_snapshot_rel = (
    "docs/summary/20260810T123000Z/chk1_checkpoint_selection/"
    "loss_snapshot.jsonl"
)
probe_snapshot_rel = (
    "docs/summary/20260810T123000Z/chk1_checkpoint_selection/"
    "probe_snapshot.jsonl"
)
selection_snapshot_rel = (
    "docs/summary/20260810T123000Z/chk1_checkpoint_selection/"
    "selection_snapshot.jsonl"
)

manifest_sources = [
    {
        "id": "loss_snapshot_source",
        "label": "完整训练与评估 loss 快照",
        "path": loss_snapshot_rel,
    },
    {
        "id": "probe_snapshot_source",
        "label": "固定生成退化探针快照",
        "path": probe_snapshot_rel,
    },
    {
        "id": "selection_snapshot_source",
        "label": "Checkpoint 决策对照",
        "path": selection_snapshot_rel,
    },
]

artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": "chk1 checkpoint 选择：推荐 checkpoint-200",
        "description": (
            "结合完整 train/eval loss 曲线与固定 8-case 退化探针，"
            "选择进入 chk2 的 chk1 checkpoint。"
        ),
        "generatedAt": GENERATED_AT,
        "cards": [
            {
                "id": "selected_checkpoint_card",
                "description": "同时满足最低 eval loss 与生成稳定性门禁。",
                "dataset": "headline",
                "sourceId": "selection_snapshot_source",
                "metrics": [
                    {
                        "label": "推荐 checkpoint",
                        "field": "selected_checkpoint",
                        "format": "number",
                    }
                ],
            },
            {
                "id": "best_eval_card",
                "description": "整个 255-step run 的最低 periodic eval loss。",
                "dataset": "headline",
                "sourceId": "loss_snapshot_source",
                "metrics": [
                    {
                        "label": "cp200 eval loss",
                        "field": "selected_eval_loss",
                        "format": "number",
                    }
                ],
            },
            {
                "id": "repetition_card",
                "description": "cp200 固定 8-case token 4-gram 平均重复率。",
                "dataset": "headline",
                "sourceId": "probe_snapshot_source",
                "metrics": [
                    {
                        "label": "cp200 mean repetition",
                        "field": "selected_mean_repetition",
                        "format": "number",
                    }
                ],
            },
            {
                "id": "post200_card",
                "description": "cp255 相对 cp200 的 eval loss 变化；正值表示变差。",
                "dataset": "headline",
                "sourceId": "loss_snapshot_source",
                "metrics": [
                    {
                        "label": "cp200→255 eval Δ",
                        "field": "post200_eval_change",
                        "format": "number",
                        "signed": True,
                    }
                ],
            },
        ],
        "charts": [
            {
                "id": "loss_trend_chart",
                "title": "Train 与 eval loss",
                "subtitle": (
                    "steps 10–255；train 为 trailing-10 optimizer-step 均值，"
                    "eval 为 199 条 validation 样本。"
                ),
                "intent": "trend",
                "question": "训练后半程是否继续改善验证损失，或出现分叉？",
                "rationale": "26 个有序评估点足以识别收敛平台和持续反弹。",
                "type": "line",
                "dataset": "loss_series",
                "sourceId": "loss_snapshot_source",
                "encodings": {
                    "x": {
                        "field": "step",
                        "type": "quantitative",
                        "label": "Optimizer step",
                    },
                    "y": {
                        "field": "value",
                        "type": "quantitative",
                        "label": "Loss",
                    },
                    "color": {
                        "field": "series",
                        "type": "nominal",
                        "label": "Series",
                    },
                    "tooltip": [
                        {
                            "field": "epoch",
                            "type": "quantitative",
                            "label": "Epoch",
                        }
                    ],
                },
                "layout": "full",
            },
            {
                "id": "repetition_chart",
                "title": "固定探针平均重复率",
                "subtitle": (
                    "冻结的 8 条 prompts 与 seeds；较低更好，"
                    "所有 chk1 checkpoint 均通过硬门禁。"
                ),
                "intent": "comparison",
                "question": "哪个通过 loss 门槛的 checkpoint 具有更低生成重复风险？",
                "rationale": "五个可比模型适合使用直接类别柱状比较。",
                "type": "bar",
                "dataset": "repetition_rows",
                "sourceId": "probe_snapshot_source",
                "encodings": {
                    "x": {
                        "field": "checkpoint_label",
                        "type": "nominal",
                        "label": "Model checkpoint",
                    },
                    "y": {
                        "field": "mean_repetition",
                        "type": "quantitative",
                        "label": "Mean token 4-gram repetition",
                    },
                    "tooltip": [
                        {
                            "field": "max_repetition",
                            "type": "quantitative",
                            "label": "Max repetition",
                        },
                        {
                            "field": "eval_loss",
                            "type": "quantitative",
                            "label": "Eval loss",
                        },
                    ],
                },
                "layout": "full",
            },
        ],
        "tables": [
            {
                "id": "selection_table",
                "title": "Checkpoint 决策对照",
                "subtitle": (
                    "排名同时考虑 validation loss、硬门禁与相对 chk0 重复率。"
                ),
                "dataset": "selection_rows",
                "sourceId": "selection_snapshot_source",
                "defaultSort": {"field": "rank", "direction": "asc"},
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "rank", "label": "Rank", "format": "number"},
                    {
                        "field": "checkpoint",
                        "label": "Checkpoint",
                        "format": "number",
                    },
                    {
                        "field": "eval_loss",
                        "label": "Eval loss",
                        "format": "number",
                    },
                    {
                        "field": "eval_delta_vs_cp200",
                        "label": "Eval Δ vs cp200",
                        "format": "number",
                        "movement": True,
                    },
                    {
                        "field": "mean_repetition",
                        "label": "Mean repetition",
                        "format": "number",
                    },
                    {
                        "field": "max_case_delta_vs_chk0",
                        "label": "Max case Δ vs chk0",
                        "format": "number",
                    },
                    {"field": "hard_gate", "label": "Gate", "type": "text"},
                    {"field": "decision", "label": "Decision", "type": "text"},
                ],
            },
            {
                "id": "recent_eval_table",
                "title": "收敛平台期评估",
                "subtitle": "steps 170–255；用于判断 cp200 后是否仍有验证收益。",
                "dataset": "recent_eval",
                "sourceId": "loss_snapshot_source",
                "defaultSort": {"field": "step", "direction": "asc"},
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "step", "label": "Step", "format": "number"},
                    {"field": "epoch", "label": "Epoch", "format": "number"},
                    {
                        "field": "train_loss_rolling20",
                        "label": "Train loss（trailing 20）",
                        "format": "number",
                    },
                    {
                        "field": "eval_loss",
                        "label": "Eval loss",
                        "format": "number",
                    },
                    {
                        "field": "is_final_eval",
                        "label": "Final eval",
                        "type": "text",
                    },
                ],
            },
        ],
        "sources": manifest_sources,
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "body": "# chk1 checkpoint 选择：推荐 checkpoint-200",
            },
            {
                "id": "technical_summary",
                "type": "markdown",
                "body": (
                    "## checkpoint-200 在拟合收益与生成稳定性之间最优\n\n"
                    "**建议将 `checkpoint-200` 作为 chk1 merge 输入并进入 chk2。** "
                    "它取得全程最低 eval loss `1.686099`，固定 8-case 探针全部通过；"
                    "继续训练到 checkpoint-255 没有带来 validation 收益，平均重复率却从 "
                    "`0.1347` 升至 `0.1410`。\n\n"
                    "这不是显著过拟合：cp200→cp255 的 eval 变化只有 "
                    "`+0.000032`（`+0.0019%`）。更准确的判断是训练在 cp200 后进入平台，"
                    "继续更新只增加了软性生成风险，没有可测的拟合回报。\n\n"
                    "**质量选择已经明确，但当前尚不能直接启动 chk2。** 这次 SFT 使用的"
                    "语义 override 收据明确限定为 chk1，`downstream_stages_allowed=[]`；"
                    "clean 数据语义审核也仍为 failed。进入 chk2 前必须获得新的下游例外授权，"
                    "或先让正常语义门禁通过。"
                ),
            },
            {
                "id": "headline_metrics",
                "type": "metric-strip",
                "cardIds": [
                    "selected_checkpoint_card",
                    "best_eval_card",
                    "repetition_card",
                    "post200_card",
                ],
            },
            {
                "id": "loss_finding",
                "type": "markdown",
                "sourceId": "loss_snapshot_source",
                "body": (
                    "## Eval loss 在 step 200 达到最低点，之后只有噪声级波动\n\n"
                    "每轮平均 train loss 为 `1.8727 → 1.7477 → 1.7132`，训练目标持续改善。"
                    "Eval loss 则在 step 200 达到 `1.686099`；step 190 到最终评估的完整范围"
                    "仅为 `0.000153`（约 `0.0091%`）。因此没有持续的 train/eval 分叉，"
                    "但也没有理由为 cp200 之后的 55 次更新支付额外行为风险。"
                ),
            },
            {
                "id": "loss_chart_block",
                "type": "chart",
                "chartId": "loss_trend_chart",
                "layout": "full",
            },
            {
                "id": "plateau_table_block",
                "type": "table",
                "tableId": "recent_eval_table",
                "layout": "full",
            },
            {
                "id": "behavior_finding",
                "type": "markdown",
                "sourceId": "probe_snapshot_source",
                "body": (
                    "## 四个 checkpoint 都没有硬退化，但 cp255 的软性重复风险最高\n\n"
                    "cp80、cp150、cp200、cp255 均为 8/8 EOS、8/8 合同有效，且没有触顶、"
                    "周期尾部或 catastrophic case。cp200 的平均 repetition 为 `0.1347`，"
                    "低于 cp255 的 `0.1410`；相对 chk0 的平均增量分别为 `+0.0519` 和 "
                    "`+0.0583`。这不足以判定 cp255 退化，却在 loss 已无收益时支持更早停止。"
                ),
            },
            {
                "id": "repetition_chart_block",
                "type": "chart",
                "chartId": "repetition_chart",
                "layout": "full",
            },
            {
                "id": "selection_finding",
                "type": "markdown",
                "sourceId": "selection_snapshot_source",
                "body": (
                    "## checkpoint-200 是主选，checkpoint-150 是保守备选\n\n"
                    "cp200 同时拥有最低 eval loss和通过的生成门禁。cp150 的最坏单样本重复"
                    "增量更低，但 eval loss 高 `0.01268`，只有在极端重视最坏单例风险时才"
                    "值得回退。cp80 明显欠拟合；cp255 与 cp200 验证表现近似相同，但没有"
                    "更优的行为指标。"
                ),
            },
            {
                "id": "selection_table_block",
                "type": "table",
                "tableId": "selection_table",
                "layout": "full",
            },
            {
                "id": "definitions",
                "type": "markdown",
                "body": (
                    "## 范围与指标定义\n\n"
                    "- 训练为两张 A30、LR `1e-6`、gradient accumulation `8`、3 epochs，"
                    "共 255 optimizer steps。\n"
                    "- Train loss 图使用每个评估点之前 10 个 optimizer steps 的算术均值；"
                    "候选表同时保留 trailing-20 均值用于降低 batch 噪声。\n"
                    "- Eval loss 来自完整 199-row validation split；cp255 使用训练结束后的"
                    "最终独立 eval。\n"
                    "- 退化探针固定 6 条 sampled 与 2 条 greedy 输出、相同 prompts、seeds、"
                    "chk0 baseline 和 3072-token 上限。"
                ),
            },
            {
                "id": "methodology",
                "type": "markdown",
                "body": (
                    "## 选择方法\n\n"
                    "先用完整 loss 历史确定拟合前沿和平台起点，再只在已运行固定生成探针的"
                    "checkpoint 间比较。硬门禁要求 finite、EOS、合法 `</think>`/answer、"
                    "无触顶、无周期尾部、无 catastrophic repetition；通过后再用 eval loss、"
                    "平均 repetition 和最大逐样本增量做软排序。"
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## 限制与稳健性\n\n"
                    "**结论对已知无限重复模式具有直接证据，但不是全面质量认证。** 探针只有"
                    "8 条，validation 也只有一个 split；train 与 eval 运行模式不同，二者"
                    "绝对差值不应被当作泛化 gap。cp200 的优势主要是 Pareto 选择：最低 eval、"
                    "硬门禁通过，并且不承担 cp255 的额外无收益更新。\n\n"
                    "**下游合规限制是执行阻断，不是模型质量指标。** 当前 override 授权仅覆盖"
                    "chk1 SFT，且该训练缺少 canonical run/training/merge receipt；在解除下游"
                    "限制并补齐 lineage 前，不应把任何 checkpoint 当作正式 chk2 parent。"
                ),
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": (
                    "## 下一步\n\n"
                    "1. 先取得明确的 chk2 下游 override 授权，或完成正常 clean-release 语义审核。\n"
                    "2. 为 `checkpoint-200` 建立版本化 promotion/training receipt，绑定 adapter "
                    "SHA256 `7a515556…9b9a9`。\n"
                    "3. 以 `checkpoint-200` 和 chk0 做独立、非破坏性的 adapter merge，并验证"
                    "精确 tensor lineage 与 tokenizer/chat-template。\n"
                    "4. chk2 使用全新输出目录，从 merged cp200 启动；不要从旧 chk2 run resume。\n"
                    "5. 保留 checkpoint-150 作为保守回退，但不要让 cp255 仅因是最终 step 而"
                    "自动进入 chk2。"
                ),
            },
            {
                "id": "questions",
                "type": "markdown",
                "body": (
                    "## 进一步问题\n\n"
                    "chk2 GRPO 是否会在保持 cp200 终止稳定性的同时提高 grounded reward，"
                    "需要用同一组冻结生成探针和 v3 reward 监控继续验证。"
                ),
            },
        ],
    },
    "snapshot": {
        "version": 1,
        "generatedAt": GENERATED_AT,
        "status": "ready",
        "datasets": {
            "headline": headline,
            "loss_rows": loss_rows,
            "loss_series": loss_series,
            "recent_eval": [row for row in loss_rows if row["step"] >= 170],
            "epoch_rows": epoch_rows,
            "probe_rows": probe_rows,
            "repetition_rows": repetition_rows,
            "selection_rows": selection_rows,
        },
    },
    "sources": [
        {
            "id": "loss_snapshot_source",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": (
                    f"SELECT * FROM read_json_auto('{loss_snapshot_rel}') "
                    "ORDER BY step"
                ),
                "description": (
                    "Reads the reviewed full-run loss snapshot derived from Trainer state."
                ),
                "executed_at": GENERATED_AT,
                "tables_used": [loss_snapshot_rel, str(RUN_REL / "trainer_state.json")],
                "filters": [
                    "all periodic evaluations from step 10 through 250",
                    "final standalone evaluation at step 255",
                ],
                "metric_definitions": [
                    "Train rolling loss is the arithmetic mean of preceding optimizer-step losses.",
                    "Eval loss is computed over the complete 199-row validation split.",
                ],
            },
        },
        {
            "id": "probe_snapshot_source",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": (
                    f"SELECT * FROM read_json_auto('{probe_snapshot_rel}') "
                    "ORDER BY checkpoint"
                ),
                "description": "Reads reviewed fixed-generation probe summaries.",
                "executed_at": GENERATED_AT,
                "tables_used": [probe_snapshot_rel],
                "filters": [
                    "checkpoints 80, 150, 200, and 255",
                    "six sampled plus two greedy cases",
                    "3072-token generation cap",
                    "comparison against frozen chk0 baseline",
                ],
                "metric_definitions": [
                    "Mean repetition is mean token 4-gram repetition across eight cases.",
                    "Hard pass requires no cap, no periodic tail, no catastrophic case, and no contract-rate drop.",
                ],
            },
        },
        {
            "id": "selection_snapshot_source",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": (
                    f"SELECT * FROM read_json_auto('{selection_snapshot_rel}') "
                    "ORDER BY rank"
                ),
                "description": (
                    "Reads the reviewed checkpoint ranking that combines loss and probe evidence."
                ),
                "executed_at": GENERATED_AT,
                "tables_used": [selection_snapshot_rel],
                "filters": ["only checkpoints with completed fixed probes"],
                "metric_definitions": [
                    "Ranking prioritizes hard-gate pass, then eval loss and soft repetition risk."
                ],
            },
        },
    ],
}

write_json(OUT / "artifact.json", artifact)

notes = f"""# Source and QA notes

- Decision: use checkpoint-200 as the chk1 input to chk2.
- Execution blocker: the current semantic override authorization is chk1-only with `downstream_stages_allowed=[]`; semantic audit remains failed and canonical promotion/merge lineage is absent.
- Audience: technical.
- Delivery mode: portable HTML because no MCP report renderer is callable in this runtime.
- Loss source: `{RUN_REL / 'trainer_state.json'}` (SHA256 `{sha256(RUN / 'trainer_state.json')}`).
- Final metrics source: `{RUN_REL / 'eval_results.json'}` (SHA256 `{sha256(RUN / 'eval_results.json')}`).
- Probe sample manifest SHA256: `3675e72983f6f235015b80460c1895f662bb8dbfe3fdd28bb1ac4634ecf65246`.
- Frozen chk0 baseline results SHA256: `26063038ea3258fb17ba7500a3af321389a6b39a394694a15c7c982229d72391`.
- Chart map:
  - Loss section: multi-series line; x=step, y=loss, series=train trailing-10/eval; supports convergence-platform claim.
  - Behavior section: category bar; x=model checkpoint, y=mean 4-gram repetition; supports soft-risk comparison.
- Palette: shared reader defaults; line uses meaningful series encoding, bar uses no redundant category legend.
- Required technical report sections are all present. Validation details are included under methodology and limitations.
- No chart was omitted. The final HTML must be packaged by the shared portable builder and its receipt retained.
"""
(OUT / "source_notes.md").write_text(notes, encoding="utf-8")

print(
    json.dumps(
        {
            "artifact": str(OUT / "artifact.json"),
            "best_step": best_row["step"],
            "best_eval_loss": best_row["eval_loss"],
            "selected_step": 200,
            "rows": {
                "loss": len(loss_rows),
                "probe": len(probe_rows),
                "selection": len(selection_rows),
            },
            "official_train_loss": train_results["train_loss"],
        },
        ensure_ascii=False,
        indent=2,
    )
)
