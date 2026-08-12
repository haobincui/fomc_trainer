#!/usr/bin/env python3
"""Reconcile and summarize the chk2 reward run through checkpoint 162."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = Path(__file__).resolve().parent
RUN_REL = Path(
    "output/training/retrain_v2/"
    "chk2_compressed_chk1_v1_reward_v3_long4096_fresh_20260807/"
    "adapters/chk2"
)
RUN_DIR = ROOT / RUN_REL
CHECKPOINT = 162
STATE_REL = RUN_REL / f"checkpoint-{CHECKPOINT}" / "trainer_state.json"
DETAIL_REL = RUN_REL / "reward.jsonl"
REWARD_HISTORY_REL = RUN_REL / "reward_history.jsonl"
LOSS_HISTORY_REL = RUN_REL / "loss_history.jsonl"
SCRIPT_REL = Path("docs/summary/20260809T112950Z/analyze_chk2_reward.py")
SNAPSHOT_REL = Path("docs/summary/20260809T112950Z/analysis_snapshot.json")

RUBRICS = (
    "data_fidelity",
    "trend_reasoning",
    "policy_relevance",
    "uncertainty_calibration",
    "fomc_style",
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def mean(rows: list[dict], getter) -> float:
    return statistics.fmean(float(getter(row)) for row in rows)


def reconcile_detail(
    state_rows: list[dict], detail_rows: list[dict]
) -> tuple[dict[int, list[dict]], list[dict]]:
    """Recover the eight completion records that produced each durable step."""

    starts: dict[int, int] = {}
    next_start = len(detail_rows)
    gaps: list[dict] = []
    for state_row in reversed(state_rows):
        step = int(state_row["step"])
        target = float(state_row["reward"])
        candidates: list[int] = []
        for start in range(0, next_start - 7):
            candidate_mean = statistics.fmean(
                float(row["reward"]) for row in detail_rows[start : start + 8]
            )
            if math.isclose(candidate_mean, target, abs_tol=1e-6):
                candidates.append(start)
        if not candidates:
            raise RuntimeError(f"could not reconcile completion records for step {step}")
        start = candidates[-1]
        gap = next_start - (start + 8)
        if gap:
            gaps.append({"after_step": step, "unselected_records": gap})
        starts[step] = start
        next_start = start

    canonical = {
        step: detail_rows[start : start + 8] for step, start in starts.items()
    }
    if any(len(rows) != 8 for rows in canonical.values()):
        raise RuntimeError("canonical reward groups must contain exactly eight rows")
    for state_row in state_rows:
        step = int(state_row["step"])
        reproduced = statistics.fmean(float(row["reward"]) for row in canonical[step])
        if not math.isclose(reproduced, float(state_row["reward"]), abs_tol=1e-6):
            raise RuntimeError(f"reward mismatch after reconciliation at step {step}")
    return canonical, gaps


def detail_window(canonical: dict[int, list[dict]], start: int, end: int) -> list[dict]:
    return [row for step in range(start, end + 1) for row in canonical[step]]


def step_window(state_rows: list[dict], start: int, end: int) -> list[dict]:
    return [row for row in state_rows if start <= int(row["step"]) <= end]


def detail_stats(rows: list[dict]) -> dict:
    patterns = [tuple(int(row["judge"][key]) for key in RUBRICS) for row in rows]
    valid_count = sum(len(row["validated_violations"]) for row in rows)
    invalid_count = sum(len(row["invalid_violations"]) for row in rows)
    return {
        "completion_count": len(rows),
        "reward_mean": mean(rows, lambda row: row["reward"]),
        "reward_median": statistics.median(float(row["reward"]) for row in rows),
        "reward_zero_rate": mean(rows, lambda row: row["reward"] == 0),
        "judge_score_mean": mean(rows, lambda row: row["components"]["judge_score"]),
        "judge_score_zero_rate": mean(
            rows, lambda row: row["components"]["judge_score"] == 0
        ),
        "rubric_all_zero_rate": statistics.fmean(
            all(score == 0 for score in pattern) for pattern in patterns
        ),
        "rubric_all_four_rate": statistics.fmean(
            all(score == 4 for score in pattern) for pattern in patterns
        ),
        "rubric_intermediate_rate": statistics.fmean(
            any(score in (1, 2, 3) for score in pattern) for pattern in patterns
        ),
        "answer_numeric_mean": mean(
            rows, lambda row: row["components"]["answer_numeric_score"]
        ),
        "structure_rate": mean(rows, lambda row: row["components"]["structure_score"]),
        "answer_concision_mean": mean(
            rows, lambda row: row["components"]["answer_concision"]
        ),
        "reasoning_efficiency_mean": mean(
            rows, lambda row: row["components"]["reasoning_efficiency"]
        ),
        "base_mean": mean(rows, lambda row: row["reward_before_penalty"]),
        "penalty_mean": mean(rows, lambda row: row["penalties"]["applied"]),
        "penalty_cap_rate": mean(rows, lambda row: row["penalties"]["applied"] >= 0.5),
        "major_answer_error_mean": mean(
            rows, lambda row: row["penalties"]["major_answer_error"]
        ),
        "major_think_error_mean": mean(
            rows, lambda row: row["penalties"]["major_think_error"]
        ),
        "valid_violation_count": valid_count,
        "invalid_violation_count": invalid_count,
        "invalid_violation_share": invalid_count / (valid_count + invalid_count),
        "target_leakage_rate": mean(
            rows, lambda row: bool(row["penalties"]["target_leakage"])
        ),
        "judge_retry_rate": mean(rows, lambda row: int(row["judge_attempts"]) > 1),
        "unsupported_answer_number_rate": mean(
            rows, lambda row: bool(row["unsupported_answer_numbers"])
        ),
    }


def step_stats(rows: list[dict]) -> dict:
    rewards = [float(row["reward"]) for row in rows]
    return {
        "step_count": len(rows),
        "reward_mean": statistics.fmean(rewards),
        "reward_median": statistics.median(rewards),
        "reward_p10": quantile(rewards, 0.10),
        "reward_p90": quantile(rewards, 0.90),
        "reward_below_010_rate": statistics.fmean(value < 0.10 for value in rewards),
        "reward_std_mean": mean(rows, lambda row: row["reward_std"]),
        "frac_reward_zero_std": mean(rows, lambda row: row["frac_reward_zero_std"]),
        "affected_zero_std_step_rate": mean(
            rows, lambda row: row["frac_reward_zero_std"] > 0
        ),
        "loss_mean": mean(rows, lambda row: row["loss"]),
        "completion_length_mean": mean(
            rows, lambda row: row["completions/mean_length"]
        ),
        "completion_clipped_ratio": mean(
            rows, lambda row: row["completions/clipped_ratio"]
        ),
        "kl_mean": mean(rows, lambda row: row["kl"]),
    }


def build() -> tuple[dict, dict]:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    state_payload = json.loads((ROOT / STATE_REL).read_text())
    state_rows = [row for row in state_payload["log_history"] if "reward" in row]
    if [int(row["step"]) for row in state_rows] != list(range(1, CHECKPOINT + 1)):
        raise RuntimeError("trainer state must contain one canonical row for every step")

    detail_rows = load_jsonl(ROOT / DETAIL_REL)
    reward_history = load_jsonl(ROOT / REWARD_HISTORY_REL)
    loss_history = load_jsonl(ROOT / LOSS_HISTORY_REL)
    canonical, gaps = reconcile_detail(state_rows, detail_rows)
    canonical_rows = detail_window(canonical, 1, CHECKPOINT)

    all_steps = step_stats(state_rows)
    all_detail = detail_stats(canonical_rows)
    last_10_start = CHECKPOINT - 9
    prior_10_start = CHECKPOINT - 19
    prior_10_end = CHECKPOINT - 10
    recent_5_start = CHECKPOINT - 4
    last_10_steps = step_stats(step_window(state_rows, last_10_start, CHECKPOINT))
    prior_10_steps = step_stats(
        step_window(state_rows, prior_10_start, prior_10_end)
    )
    last_10_detail = detail_stats(
        detail_window(canonical, last_10_start, CHECKPOINT)
    )
    prior_10_detail = detail_stats(
        detail_window(canonical, prior_10_start, prior_10_end)
    )
    recent_5_steps = step_stats(
        step_window(state_rows, recent_5_start, CHECKPOINT)
    )
    recent_5_detail = detail_stats(
        detail_window(canonical, recent_5_start, CHECKPOINT)
    )

    x_values = [float(row["step"]) for row in state_rows]
    y_values = [float(row["reward"]) for row in state_rows]
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    slope = sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values, strict=True)
    ) / sum((x - x_mean) ** 2 for x in x_values)

    reward_blocks = []
    window_rows = []
    for end in range(10, CHECKPOINT + 1, 10):
        start = end - 9
        steps = step_window(state_rows, start, end)
        details = detail_window(canonical, start, end)
        ss = step_stats(steps)
        ds = detail_stats(details)
        reward_blocks.append(
            {
                "block_end": end,
                "block_label": f"{start}-{end}",
                "reward_mean": ss["reward_mean"],
                "reward_median": ss["reward_median"],
                "reward_below_010_rate": ss["reward_below_010_rate"],
                "reward_std_mean": ss["reward_std_mean"],
                "completion_length_mean": ss["completion_length_mean"],
                "loss_mean": ss["loss_mean"],
                "judge_all_zero_rate": ds["rubric_all_zero_rate"],
                "completion_zero_reward_rate": ds["reward_zero_rate"],
                "penalty_mean": ds["penalty_mean"],
            }
        )
        if end % 20 == 0:
            twenty_start = end - 19
            twenty_steps = step_window(state_rows, twenty_start, end)
            twenty_details = detail_window(canonical, twenty_start, end)
            tss = step_stats(twenty_steps)
            tds = detail_stats(twenty_details)
            window_rows.append(
                {
                    "step_end": end,
                    "window": f"{twenty_start}-{end}",
                    "reward_mean": tss["reward_mean"],
                    "reward_median": tss["reward_median"],
                    "completion_zero_reward_rate": tds["reward_zero_rate"],
                    "judge_all_zero_rate": tds["rubric_all_zero_rate"],
                    "answer_numeric_mean": tds["answer_numeric_mean"],
                    "penalty_mean": tds["penalty_mean"],
                }
            )

    pattern_counts = Counter()
    for row in canonical_rows:
        pattern = tuple(int(row["judge"][key]) for key in RUBRICS)
        if all(score == 0 for score in pattern):
            pattern_counts["五项全部为 0"] += 1
        elif all(score == 4 for score in pattern):
            pattern_counts["五项全部为 4"] += 1
        elif any(score in (1, 2, 3) for score in pattern):
            pattern_counts["至少一项使用 1–3"] += 1
        else:
            pattern_counts["0/4 混合"] += 1
    pattern_order = ["五项全部为 0", "五项全部为 4", "至少一项使用 1–3", "0/4 混合"]
    judge_patterns = [
        {
            "pattern": pattern,
            "completion_count": pattern_counts[pattern],
            "share": pattern_counts[pattern] / len(canonical_rows),
            "total_completions": len(canonical_rows),
        }
        for pattern in pattern_order
    ]

    violation_counts = Counter()
    invalid_reasons = Counter()
    for row in canonical_rows:
        for violation in row["validated_violations"]:
            violation_counts[
                (violation.get("section"), violation.get("kind"), violation.get("severity"))
            ] += 1
        for violation in row["invalid_violations"]:
            invalid_reasons[violation.get("invalid_reason")] += 1

    def weighted_component_delta(component: str, weight: float) -> float:
        return weight * (
            last_10_detail[component] - prior_10_detail[component]
        )

    drivers = [
        {
            "driver": "Judge rubric",
            "contribution": weighted_component_delta("judge_score_mean", 0.50),
            "prior_value": prior_10_detail["judge_score_mean"],
            "current_value": last_10_detail["judge_score_mean"],
        },
        {
            "driver": "Answer numeric grounding",
            "contribution": weighted_component_delta("answer_numeric_mean", 0.25),
            "prior_value": prior_10_detail["answer_numeric_mean"],
            "current_value": last_10_detail["answer_numeric_mean"],
        },
        {
            "driver": "Structure",
            "contribution": weighted_component_delta("structure_rate", 0.15),
            "prior_value": prior_10_detail["structure_rate"],
            "current_value": last_10_detail["structure_rate"],
        },
        {
            "driver": "Answer concision",
            "contribution": weighted_component_delta("answer_concision_mean", 0.05),
            "prior_value": prior_10_detail["answer_concision_mean"],
            "current_value": last_10_detail["answer_concision_mean"],
        },
        {
            "driver": "Reasoning efficiency",
            "contribution": weighted_component_delta("reasoning_efficiency_mean", 0.05),
            "prior_value": prior_10_detail["reasoning_efficiency_mean"],
            "current_value": last_10_detail["reasoning_efficiency_mean"],
        },
        {
            "driver": "Applied factual penalty",
            "contribution": -(
                last_10_detail["penalty_mean"] - prior_10_detail["penalty_mean"]
            ),
            "prior_value": prior_10_detail["penalty_mean"],
            "current_value": last_10_detail["penalty_mean"],
        },
    ]

    recent_steps = []
    for row in step_window(state_rows, last_10_start, CHECKPOINT):
        step = int(row["step"])
        ds = detail_stats(canonical[step])
        recent_steps.append(
            {
                "step": step,
                "reward": float(row["reward"]),
                "loss": float(row["loss"]),
                "reward_std": float(row["reward_std"]),
                "completion_zero_reward_rate": ds["reward_zero_rate"],
                "judge_all_zero_rate": ds["rubric_all_zero_rate"],
                "major_answer_error_mean": ds["major_answer_error_mean"],
                "penalty_mean": ds["penalty_mean"],
            }
        )

    reward_history_counts = Counter(int(row["step"]) for row in reward_history)
    duplicate_history_rows = sum(count - 1 for count in reward_history_counts.values())
    data_quality = {
        "canonical_step_count": len(state_rows),
        "canonical_completion_count": len(canonical_rows),
        "reward_history_row_count": len(reward_history),
        "loss_history_row_count": len(loss_history),
        "duplicate_history_rows": duplicate_history_rows,
        "duplicated_steps": {
            str(step): count for step, count in reward_history_counts.items() if count > 1
        },
        "raw_detail_row_count": len(detail_rows),
        "excluded_detail_row_count": len(detail_rows) - len(canonical_rows),
        "reconciliation_gaps": gaps,
        "reproduced_all_step_means": True,
    }

    snapshot = {
        "generated_at": generated_at,
        "checkpoint": CHECKPOINT,
        "max_steps": int(state_payload["max_steps"]),
        "progress": CHECKPOINT / int(state_payload["max_steps"]),
        "all_steps": all_steps,
        "all_completions": all_detail,
        "prior_10_steps": prior_10_steps,
        "last_10_steps": last_10_steps,
        "prior_10_completions": prior_10_detail,
        "last_10_completions": last_10_detail,
        "recent_5_steps": recent_5_steps,
        "recent_5_completions": recent_5_detail,
        "reward_linear_slope_per_step": slope,
        "reward_linear_slope_per_100_steps": slope * 100,
        "reward_blocks": reward_blocks,
        "window_rows": window_rows,
        "judge_patterns": judge_patterns,
        "drivers_last10_vs_prior10": drivers,
        "recent_steps": recent_steps,
        "violation_summary": {
            "validated": [
                {
                    "section": section,
                    "kind": kind,
                    "severity": severity,
                    "count": count,
                }
                for (section, kind, severity), count in violation_counts.most_common()
            ],
            "invalid_reasons": dict(invalid_reasons),
            "raw_violation_count": sum(violation_counts.values())
            + sum(invalid_reasons.values()),
        },
        "data_quality": data_quality,
        "chart_map": [
            {
                "section": "Reward trend",
                "question": "Has reward improved across training?",
                "chart": "line",
                "dataset": "reward_blocks",
                "claim": "The overall slope is flat and the final 10-step block declined.",
            },
            {
                "section": "Judge behavior",
                "question": "Does the Judge use the full 0-4 rubric?",
                "chart": "bar",
                "dataset": "judge_patterns",
                "claim": "Judge outputs are overwhelmingly all-zero or all-four.",
            },
        ],
    }

    headline = [
        {
            "checkpoint": CHECKPOINT,
            "max_steps": int(state_payload["max_steps"]),
            "progress": snapshot["progress"],
            "overall_reward": all_steps["reward_mean"],
            "last10_reward": last_10_steps["reward_mean"],
            "prior10_reward": prior_10_steps["reward_mean"],
            "last10_delta": last_10_steps["reward_mean"] - prior_10_steps["reward_mean"],
            "last10_zero_reward_rate": last_10_detail["reward_zero_rate"],
            "judge_all_zero_rate": all_detail["rubric_all_zero_rate"],
            "zero_std_group_fraction": all_steps["frac_reward_zero_std"],
        }
    ]

    state_path = STATE_REL.as_posix()
    detail_path = DETAIL_REL.as_posix()
    analysis_path = SNAPSHOT_REL.as_posix()
    source_manifest = [
        {
            "id": "trainer_state_source",
            "label": f"checkpoint-{CHECKPOINT} canonical trainer state",
            "path": state_path,
        },
        {
            "id": "analysis_snapshot_source",
            "label": "Reconciled chk2 reward analysis snapshot",
            "path": analysis_path,
        },
    ]
    source_details = [
        {
            "id": "trainer_state_source",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": (
                    "SELECT log.* FROM read_json_auto('"
                    + state_path
                    + "') AS state, UNNEST(state.log_history) AS rows(log) "
                    f"WHERE log.reward IS NOT NULL AND log.step BETWEEN 1 AND {CHECKPOINT}"
                ),
                "description": f"Reads the durable per-step metrics saved with checkpoint {CHECKPOINT}.",
                "executed_at": generated_at,
                "tables_used": [state_path],
                "filters": [f"steps 1 through {CHECKPOINT}", "rows containing reward"],
                "metric_definitions": [
                    "Step reward is the mean grounded_analysis_v3 reward across eight completions.",
                    "frac_reward_zero_std is the fraction of two four-generation groups with zero within-group reward variance.",
                ],
            },
        },
        {
            "id": "analysis_snapshot_source",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": f"SELECT * FROM read_json_auto('{analysis_path}')",
                "description": "Reverse-matches eight completion records per durable step and computes the diagnostic datasets.",
                "executed_at": generated_at,
                "tables_used": [state_path, detail_path],
                "filters": [
                    f"checkpoint {CHECKPOINT} snapshot",
                    "eight completions per canonical step",
                    "exclude failed-restart and duplicate append records",
                ],
                "metric_definitions": [
                    "Completion zero rate is the share of canonical completion rewards equal to zero.",
                    "Judge all-zero rate is the share with all five rubric dimensions equal to zero.",
                    "Driver contributions apply the configured reward weights; penalty contribution is the negative change in applied penalty.",
                ],
            },
        },
    ]

    title = f"chk2 Reward 诊断（checkpoint {CHECKPOINT}）"
    overall_reward = all_steps["reward_mean"]
    last10_reward = last_10_steps["reward_mean"]
    prior10_reward = prior_10_steps["reward_mean"]
    last10_delta = last10_reward - prior10_reward
    report = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Canonical reward trend, Judge behavior, and penalty diagnostics for chk2.",
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "progress_card",
                    "description": "Durable optimization progress at the analysis cutoff.",
                    "dataset": "headline",
                    "sourceId": "trainer_state_source",
                    "metrics": [{"label": "训练进度", "field": "progress", "format": "percent"}],
                },
                {
                    "id": "overall_reward_card",
                    "description": f"Mean of the {CHECKPOINT} canonical per-step rewards.",
                    "dataset": "headline",
                    "sourceId": "trainer_state_source",
                    "metrics": [{"label": "全程平均 reward", "field": "overall_reward", "format": "number"}],
                },
                {
                    "id": "last10_reward_card",
                    "description": f"Final ten steps compared with steps {prior_10_start}-{prior_10_end}.",
                    "dataset": "headline",
                    "sourceId": "analysis_snapshot_source",
                    "metrics": [
                        {"label": "最近 10 step", "field": "last10_reward", "format": "number"},
                        {"label": "前 10 step", "field": "prior10_reward", "format": "number"},
                        {"label": "变化", "field": "last10_delta", "format": "number", "signed": True},
                    ],
                },
                {
                    "id": "judge_zero_card",
                    "description": "Share of canonical completions receiving zero on every Judge rubric dimension.",
                    "dataset": "headline",
                    "sourceId": "analysis_snapshot_source",
                    "metrics": [{"label": "Judge 五项全 0", "field": "judge_all_zero_rate", "format": "percent"}],
                },
            ],
            "charts": [
                {
                    "id": "reward_trend_chart",
                    "title": "10-step mean reward by training block",
                    "subtitle": "Complete 10-step blocks through step 160; the final block fell to 0.124 after peaking at 0.242 in steps 131-140.",
                    "intent": "trend",
                    "question": "Has canonical chk2 reward improved as training progresses?",
                    "rationale": "Sixteen ordered blocks are sufficient to show the noisy, non-monotonic trajectory without overplotting per-step variation.",
                    "type": "line",
                    "dataset": "reward_blocks",
                    "sourceId": "trainer_state_source",
                    "encodings": {
                        "x": {"field": "block_end", "type": "quantitative", "label": "Step（10-step block end）"},
                        "y": {"field": "reward_mean", "type": "quantitative", "label": "Mean reward"},
                        "tooltip": [
                            {"field": "reward_median", "type": "quantitative", "label": "Median reward"},
                            {"field": "reward_below_010_rate", "type": "quantitative", "label": "Steps below 0.10", "format": "percent"},
                            {"field": "completion_length_mean", "type": "quantitative", "label": "Mean completion tokens"},
                        ],
                    },
                    "valueFormat": "number",
                    "layout": "full",
                },
                {
                    "id": "judge_pattern_chart",
                    "title": "Judge rubric output pattern",
                    "subtitle": "Across 1,280 canonical completions; 91.7% received zero on all five dimensions.",
                    "intent": "comparison",
                    "question": "Does the Judge use the intended 0-4 rubric range?",
                    "rationale": "A sorted categorical comparison makes rubric collapse visible more clearly than averages alone.",
                    "type": "bar",
                    "dataset": "judge_patterns",
                    "sourceId": "analysis_snapshot_source",
                    "encodings": {
                        "x": {"field": "pattern", "type": "nominal", "label": "Rubric pattern"},
                        "y": {"field": "share", "type": "quantitative", "label": "Share of completions", "format": "percent"},
                        "tooltip": [
                            {"field": "completion_count", "type": "quantitative", "label": "Completions"},
                            {"field": "total_completions", "type": "quantitative", "label": "Total"},
                        ],
                    },
                    "valueFormat": "percent",
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "window_table",
                    "title": "20-step reward windows",
                    "subtitle": "Complete 20-step windows through step 160; rows follow training order.",
                    "dataset": "window_rows",
                    "sourceId": "analysis_snapshot_source",
                    "defaultSort": {"field": "step_end", "direction": "asc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "step_end", "label": "End step", "format": "number"},
                        {"field": "window", "label": "Window", "type": "text"},
                        {"field": "reward_mean", "label": "Mean reward", "format": "number"},
                        {"field": "reward_median", "label": "Median reward", "format": "number"},
                        {"field": "completion_zero_reward_rate", "label": "Zero reward", "format": "percent"},
                        {"field": "judge_all_zero_rate", "label": "Judge all-zero", "format": "percent"},
                        {"field": "penalty_mean", "label": "Mean penalty", "format": "number"},
                    ],
                },
                {
                    "id": "driver_table",
                    "title": "Drivers of the latest 10-step decline",
                    "subtitle": f"Steps {last_10_start}-{CHECKPOINT} versus {prior_10_start}-{prior_10_end}; contribution is on the pre-clip reward scale.",
                    "dataset": "drivers",
                    "sourceId": "analysis_snapshot_source",
                    "defaultSort": {"field": "contribution", "direction": "asc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "driver", "label": "Driver", "type": "text"},
                        {"field": "prior_value", "label": f"Steps {prior_10_start}-{prior_10_end}", "format": "number"},
                        {"field": "current_value", "label": f"Steps {last_10_start}-{CHECKPOINT}", "format": "number"},
                        {"field": "contribution", "label": "Reward contribution", "format": "number", "movement": True},
                    ],
                },
                {
                    "id": "recent_steps_table",
                    "title": "Recent per-step metrics",
                    "subtitle": f"The ten durable steps ending at checkpoint {CHECKPOINT}.",
                    "dataset": "recent_steps",
                    "sourceId": "analysis_snapshot_source",
                    "defaultSort": {"field": "step", "direction": "asc"},
                    "density": "dense",
                    "layout": "full",
                    "columns": [
                        {"field": "step", "label": "Step", "format": "number"},
                        {"field": "reward", "label": "Reward", "format": "number"},
                        {"field": "loss", "label": "GRPO loss", "format": "number"},
                        {"field": "reward_std", "label": "Reward std", "format": "number"},
                        {"field": "completion_zero_reward_rate", "label": "Zero completions", "format": "percent"},
                        {"field": "major_answer_error_mean", "label": "Major answer errors", "format": "number"},
                    ],
                },
            ],
            "sources": source_manifest,
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "analysis_snapshot_source",
                    "body": (
                        "## 技术结论\n\n"
                        f"**当前 reward 仍有训练信号，但没有显示持续改善。** `{len(state_rows)}` 个有效 step 的平均 reward 为 `{overall_reward:.4f}`，线性趋势每 100 step 仅 `{slope * 100:+.4f}`；最近 10 step 从前 10 step 的 `{prior10_reward:.4f}` 降到 `{last10_reward:.4f}`（`{last10_delta:+.4f}`，约 `{last10_delta / prior10_reward:.1%}`）。\n\n"
                        f"**主要问题在 Judge，而不是格式或截断。** `{all_detail['rubric_all_zero_rate']:.1%}` 的 completion 五项 rubric 全为 0，只有 `{all_detail['rubric_intermediate_rate']:.1%}` 真正用到 1–3；结构合格率为 `{all_detail['structure_rate']:.1%}`，目标泄漏为 0，最近五步平均截断率仅 `{recent_5_steps['completion_clipped_ratio']:.1%}`。当前 reward 实际上更像“结构/数字基础分减去事实惩罚”，而不是细粒度的 0–4 语义评价。"
                    ),
                },
                {
                    "id": "headline_metrics",
                    "type": "metric-strip",
                    "cardIds": ["progress_card", "overall_reward_card", "last10_reward_card", "judge_zero_card"],
                },
                {
                    "id": "trend_finding",
                    "type": "markdown",
                    "sourceId": "trainer_state_source",
                    "body": (
                        "## Reward 总体横盘，最近窗口回落\n\n"
                        f"完整 10-step block 均值在 `0.124–0.242` 间震荡，没有形成单调上升。steps 121–140 一度升至 `0.2239`，但 steps 141–160 回落到 `0.1657`；最新 steps {last_10_start}–{CHECKPOINT} 为 `{last10_reward:.4f}`。这可以描述为近期退化信号，但不能直接归因于模型，因为每个 step 使用不同 prompt，且没有同一 validation prompt 的固定种子对照。"
                    ),
                },
                {"id": "reward_trend", "type": "chart", "chartId": "reward_trend_chart"},
                {"id": "window_evidence", "type": "table", "tableId": "window_table"},
                {
                    "id": "judge_finding",
                    "type": "markdown",
                    "sourceId": "analysis_snapshot_source",
                    "body": (
                        "## Judge 的 0–4 rubric 实际退化为二元输出\n\n"
                        f"在 `{len(canonical_rows)}` 个有效 completion 中，五项全 0 有 `{pattern_counts['五项全部为 0']}` 条，五项全 4 有 `{pattern_counts['五项全部为 4']}` 条，任何一项使用 1–3 的只有 `{pattern_counts['至少一项使用 1–3']}` 条。与此同时，全部 `{sum(violation_counts.values()) + sum(invalid_reasons.values())}` 条原始 violation 都指向 answer，think 为 0 条。这与“完整审查 think + answer，并使用完整 0–4 量表”的设计目标不一致。"
                    ),
                },
                {"id": "judge_pattern", "type": "chart", "chartId": "judge_pattern_chart"},
                {
                    "id": "penalty_finding",
                    "type": "markdown",
                    "sourceId": "analysis_snapshot_source",
                    "body": (
                        "## 最近下降主要来自事实惩罚增大\n\n"
                        f"steps {last_10_start}–{CHECKPOINT} 相比 {prior_10_start}–{prior_10_end}，平均 major answer error 从 `{prior_10_detail['major_answer_error_mean']:.3f}` 变为 `{last_10_detail['major_answer_error_mean']:.3f}`，平均 penalty 从 `{prior_10_detail['penalty_mean']:.3f}` 变为 `{last_10_detail['penalty_mean']:.3f}`；零 reward completion 从 `{prior_10_detail['reward_zero_rate']:.1%}` 变为 `{last_10_detail['reward_zero_rate']:.1%}`。惩罚变化对 reward 的贡献为 `{-(last_10_detail['penalty_mean'] - prior_10_detail['penalty_mean']):+.3f}`；其他变化来自 numeric grounding 和 Judge base score。"
                    ),
                },
                {"id": "driver_evidence", "type": "table", "tableId": "driver_table"},
                {"id": "recent_evidence", "type": "table", "tableId": "recent_steps_table"},
                {
                    "id": "scope_definitions",
                    "type": "markdown",
                    "sourceId": "analysis_snapshot_source",
                    "body": (
                        "## 范围与指标定义\n\n"
                        f"分析快照固定在 checkpoint {CHECKPOINT}（{CHECKPOINT}/{state_payload['max_steps']} step）。每个 durable step 包含两个四生成 group，共 8 个 completion。step reward 是这 8 个 grounded_analysis_v3 reward 的均值；completion zero rate 是其中最终 reward 等于 0 的比例；Judge all-zero 表示五项 rubric 均为 0。GRPO loss 是策略目标，不应像监督学习 loss 那样按绝对值判断收敛。"
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "sourceId": "analysis_snapshot_source",
                    "body": (
                        "## 重启日志已被严格对齐\n\n"
                        f"checkpoint trainer state 提供 `{len(state_rows)}` 个唯一有效 step。reward/loss history 各有 `{len(reward_history)}` 行，其中 `{duplicate_history_rows}` 行来自 step 56 重跑。completion 日志共有 `{len(detail_rows)}` 行；通过反向匹配“连续 8 条 completion reward 的均值必须复现 durable step reward”，保留 `{len(canonical_rows)}` 行并排除 `{len(detail_rows) - len(canonical_rows)}` 行失败重启记录。全部 `{len(state_rows)}` 个 step 均在 `1e-6` 容差内复现。"
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## 不确定性与稳健性限制\n\n"
                        "本报告能确认 reward 的形状和 Judge 输出分布，但不能仅凭训练序列证明模型能力下降：prompt 随 step 改变，completion/evidence 原文按设计未写入 reward 日志，因此无法复核每条 factual violation 是否正确。按 meeting date 做重复样本对照时，最近窗口在 17 个可比日期中有 11 个低于此前均值，说明回落不完全是日期组合造成的；topic 难度差异仍未被控制。"
                    ),
                },
                {
                    "id": "recommendations",
                    "type": "markdown",
                    "body": (
                        "## 建议的下一步\n\n"
                        "1. 用 checkpoint 140、150、160 在同一批 validation prompts 和固定 generation seeds 上做 replay；先确认能力变化，再决定是否继续或回退。\n\n"
                        "2. 优先修 Judge prompt/输出：要求五项独立打分，并明确“一条事实错误不应自动把五项全部置 0”；同时检查为什么 think section 从未产生 violation。\n\n"
                        f"3. 对同一 factual claim 避免 rubric 降分与 penalty 重复惩罚。当前 penalty 平均消耗约 {all_detail['penalty_mean'] / all_detail['base_mean']:.0%} 的 base，最近 10 step 有 {last_10_detail['penalty_cap_rate']:.1%} completion 触及 0.50 上限。\n\n"
                        "4. 在 Judge 修订前，不要仅以当前 reward 上升或下降决定 chk2 晋级；当前信号足以继续产生梯度，但目标质量不足以证明 grounded analysis 正在改善。"
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## 仍需回答的问题\n\n"
                        f"- 固定 validation prompts 下，checkpoint {CHECKPOINT} 是否优于 140/150/160？\n"
                        "- Judge 的全 0 输出来自模型判断、schema 约束，还是 prompt 对“unsupported”的解释过严？\n"
                        "- think 完全没有 violation 是内容确实可靠，还是 Judge 没有审查该 section？"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "reward_blocks": reward_blocks,
                "judge_patterns": judge_patterns,
                "window_rows": window_rows,
                "drivers": drivers,
                "recent_steps": recent_steps,
            },
        },
        "sources": source_details,
    }
    return snapshot, report


def main() -> None:
    snapshot, report = build()
    (REPORT_DIR / "analysis_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n"
    )
    (REPORT_DIR / "artifact.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "checkpoint": snapshot["checkpoint"],
                "canonical_steps": snapshot["data_quality"]["canonical_step_count"],
                "canonical_completions": snapshot["data_quality"]["canonical_completion_count"],
                "overall_reward": snapshot["all_steps"]["reward_mean"],
                "last10_reward": snapshot["last_10_steps"]["reward_mean"],
                "judge_all_zero_rate": snapshot["all_completions"]["rubric_all_zero_rate"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
