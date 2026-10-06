"""Render the paper-chk2 Core8 leave-one-out technical report.

The renderer is intentionally read-only with respect to the evaluation bundle:
it validates the sealed score artifacts, projects them into a self-contained
HTML report, a readable Markdown report, and a review-ready LaTeX table, then
publishes those files create-only under ``<run-root>/report``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import importlib.metadata
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt

from jobs.eval import score_paper_chk2_core8_loo_vllm_k10 as scorer


DEFAULT_OUTPUT_ROOT = scorer.DEFAULT_OUTPUT_ROOT
REPORT_SCHEMA = "paper-chk2-core8-loo-report-manifest-v1"
REPORT_ID = "paper-chk2-cp50-core8-loo-k10-b10000-technical-report-v1"

TOPIC_LABELS = {
    "Consumer-Price-Index-(CPI)": "Consumer Price Index (CPI)",
    "GDP-Growth": "GDP Growth",
    "Government-Purchases": "Government Purchases",
    "Housing-Starts": "Housing Starts",
    "Industrial-Production": "Industrial Production",
    "Labour-Market": "Labour Market",
    "Money-Supply": "Money Supply",
    "Unemployment-Rate": "Unemployment Rate",
}
ARM_LABELS = {
    "exact_deletion": "Exact deletion",
    "token_matched_neutral": "Token-matched neutral replacement",
}
METRIC_LABELS = {"mpnet_cosine": "MPNet cosine", "bertscore_f1": "BERTScore F1"}

SCOPE_DISCLOSURE = (
    "This evaluation measures target-relative prompt sensitivity within the selected "
    "paper-level Model chk-2 checkpoint; it is not a causal estimate of topic importance."
)
REFERENCE_DISCLOSURE = (
    "Every Full and intervened output is compared with the same frozen, source-grounded "
    "synthetic Minutes reference for its meeting, not with official FOMC Minutes."
)
GATE_DISCLOSURE = (
    "Generation hard gates are reported as a diagnostic funnel; no nonempty generation "
    "is removed or zero-penalised because of a gate failure in the primary semantic estimand."
)
OOD_DISCLOSURE = (
    "Core8 is a multi-block 1993–2008 stress test of a model trained for atomic-analysis-to-"
    "paragraph rewriting; the results do not establish information sufficiency or meeting-level effects."
)
PANEL_REUSE_DISCLOSURE = (
    "The same frozen Core8 panel was scored in recorded experiments on 2026-08-14, "
    "2026-08-17, and 2026-08-24; repeated-holdout and model-development reuse cannot "
    "be ruled out. This checkpoint-50 rerun is therefore not a newly untouched or "
    "prospective holdout."
)


class PaperChk2Core8ReportError(RuntimeError):
    """The score-to-report contract or create-only publication failed."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise PaperChk2Core8ReportError(f"required regular file is missing: {resolved}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": _sha_file(resolved)}


def _read_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperChk2Core8ReportError(f"invalid report source {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise PaperChk2Core8ReportError(f"report source must be an object: {resolved}")
    return value


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.pop("manifest_sha256", None)
    return {**body, "manifest_sha256": hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()}


def _validate_seal(value: Mapping[str, Any]) -> None:
    body = dict(value)
    observed = body.pop("manifest_sha256", None)
    expected = hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()
    if observed != expected:
        raise PaperChk2Core8ReportError("report manifest seal mismatch")


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise PaperChk2Core8ReportError(f"{label} is not numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PaperChk2Core8ReportError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise PaperChk2Core8ReportError(f"{label} is not finite")
    return number


def _source(
    *, source_id: str, label: str, filename: str, binding: Mapping[str, Any], executed_at: str
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": (
                f"SELECT * FROM read_csv_auto('{filename}')"
                if filename.endswith(".csv")
                else f"SELECT * FROM read_json_auto('{filename}')"
            ),
            "description": f"Validated {filename}; SHA-256 {binding['sha256']}, {binding['bytes']} bytes.",
            "executed_at": executed_at,
            "tables_used": [filename],
        },
    }


def _project_cell(row: Mapping[str, Any]) -> dict[str, Any]:
    bootstrap = row["hierarchical_bootstrap"]
    test = row["meeting_level_t_test"]
    adequacy = row["k_adequacy"]
    estimate = _finite(row["point_estimate"], label="point estimate")
    low = _finite(bootstrap["ci_lower"], label="CI lower")
    high = _finite(bootstrap["ci_upper"], label="CI upper")
    adjusted = _finite(test["holm_adjusted_p"], label="Holm p")
    t_value = test.get("t_statistic")
    return {
        "topic": row["topic"],
        "topic_label": TOPIC_LABELS[str(row["topic"])],
        "arm": row["arm"],
        "arm_label": ARM_LABELS[str(row["arm"])],
        "metric": row["metric"],
        "metric_label": METRIC_LABELS[str(row["metric"])],
        "series_label": f"{ARM_LABELS[str(row['arm'])]} · {METRIC_LABELS[str(row['metric'])]}",
        "estimate": estimate,
        "estimate_display": f"{estimate:+.6f}",
        "ci_lower": low,
        "ci_upper": high,
        "ci_95": f"[{low:+.6f}, {high:+.6f}]",
        "t_statistic": None if t_value is None else _finite(t_value, label="t statistic"),
        "t_display": "undefined" if t_value is None else f"{float(t_value):.3f}",
        "holm_adjusted_p": adjusted,
        "holm_adjusted_p_display": "<0.0001" if adjusted < 0.0001 else f"{adjusted:.4f}",
        "significance_marker": test["significance_marker"],
        "holm_reject_005": bool(test["holm_reject_005"]),
        "positive_meetings": int(row["positive_meetings"]),
        "negative_meetings": int(row["negative_meetings"]),
        "increase_k_recommended": bool(adequacy["increase_k_reasons"]),
        "increase_k_reasons": ", ".join(adequacy["increase_k_reasons"]) or "None",
        "decoding_variance_share": adequacy["estimated_decoding_variance_share"],
    }


def load_report_model(run_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    scorer.validate_score_bundle(run_root)
    score_dir = run_root / "score"
    results = _read_json(score_dir / "bootstrap_results.json")
    score_manifest = _read_json(score_dir / "score_manifest.json")
    source_bindings = {
        "cell_statistics.csv": _binding(score_dir / "cell_statistics.csv"),
        "bootstrap_results.json": _binding(score_dir / "bootstrap_results.json"),
        "score_manifest.json": _binding(score_dir / "score_manifest.json"),
        "generation_diagnostic_failures.jsonl": _binding(score_dir / "generation_diagnostic_failures.jsonl"),
    }
    cells = [_project_cell(row) for row in results["cells"]]
    expected_order = [
        (topic, arm, metric)
        for topic in scorer.TOPICS for arm in scorer.ARMS for metric in scorer.METRICS
    ]
    if [(row["topic"], row["arm"], row["metric"]) for row in cells] != expected_order:
        raise PaperChk2Core8ReportError("cell ordering drift")
    funnel = copy.deepcopy(results["diagnostic_funnel"])
    funnel_rows = [
        {
            "stage": "Requested and semantically scored",
            "count": int(funnel["requested_rows"]),
            "share": 1.0,
            "share_display": "100.0%",
        },
        {
            "stage": "Nonempty answer",
            "count": int(funnel["nonempty_answer_rows"]),
            "share": int(funnel["nonempty_answer_rows"]) / int(funnel["requested_rows"]),
            "share_display": f"{100 * int(funnel['nonempty_answer_rows']) / int(funnel['requested_rows']):.1f}%",
        },
        {
            "stage": "Joint core gate PASS (diagnostic)",
            "count": int(funnel["preregistered_core_valid_rows"]),
            "share": int(funnel["preregistered_core_valid_rows"]) / int(funnel["requested_rows"]),
            "share_display": f"{100 * int(funnel['preregistered_core_valid_rows']) / int(funnel['requested_rows']):.1f}%",
        },
        {
            "stage": "Rows retained in primary analysis",
            "count": int(funnel["raw_semantic_scored_rows"]),
            "share": 1.0,
            "share_display": "100.0%",
        },
    ]
    generated_at = str(results.get("created_at_utc") or score_manifest.get("created_at_utc"))
    if not generated_at:
        raise PaperChk2Core8ReportError("stable score timestamp missing")
    return {
        "run_root": str(run_root),
        "generated_at_utc": generated_at,
        "cells": cells,
        "funnel": funnel,
        "funnel_rows": funnel_rows,
        "coverage": copy.deepcopy(results["coverage"]),
        "bootstrap_contract": copy.deepcopy(results["bootstrap_contract"]),
        "multiplicity": copy.deepcopy(results["multiplicity"]),
        "k_adequacy": copy.deepcopy(results["k_adequacy"]),
        "limitations": list(results["limitations"]),
        "source_bindings": source_bindings,
    }


def _chart(
    *, chart_id: str, title: str, subtitle: str, dataset: str, source_id: str,
    x_field: str, y_field: str, y_label: str, x_label: str = "Indicator",
    color_field: str | None = None, color_label: str | None = None,
    reference_zero: bool = False, tooltip: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    encoding: dict[str, Any] = {
        "x": {"field": x_field, "type": "nominal", "label": x_label},
        "y": {"field": y_field, "type": "quantitative", "label": y_label},
        "tooltip": [dict(value) for value in (tooltip or (
            {"field": "estimate_display", "type": "nominal", "label": "Full − intervention"},
            {"field": "ci_95", "type": "nominal", "label": "Hierarchical 95% CI"},
            {"field": "holm_adjusted_p_display", "type": "nominal", "label": "Holm p"},
        ))],
    }
    if color_field:
        encoding["color"] = {"field": color_field, "type": "nominal", "label": color_label or color_field}
    chart = {
        "id": chart_id,
        "title": title,
        "subtitle": subtitle,
        "intent": "comparison",
        "question": title,
        "rationale": "A grouped native bar chart shows intervention heterogeneity; exact intervals and adjusted p-values remain in the adjacent table.",
        "type": "bar",
        "dataset": dataset,
        "sourceId": source_id,
        "encodings": encoding,
        "palette": {"kind": "categorical", "name": "blue"},
        "labels": {"values": "auto"},
        "valueFormat": "number",
        "layout": "full",
    }
    if reference_zero:
        chart["referenceLines"] = [
            {"axis": "y", "value": 0, "label": "Zero", "color": "neutral", "lineStyle": "dashed"}
        ]
    return chart


def _table(
    *, table_id: str, title: str, subtitle: str, dataset: str, source_id: str,
    columns: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "id": table_id,
        "title": title,
        "subtitle": subtitle,
        "dataset": dataset,
        "sourceId": source_id,
        "density": "dense",
        "layout": "full",
        "columns": [dict(value) for value in columns],
    }


def build_canonical_artifact(model: Mapping[str, Any]) -> dict[str, Any]:
    cells = copy.deepcopy(list(model["cells"]))
    funnel_rows = copy.deepcopy(list(model["funnel_rows"]))
    generated_at = str(model["generated_at_utc"])
    significant = sum(bool(row["holm_reject_005"]) for row in cells)
    largest = max(cells, key=lambda row: abs(float(row["estimate"])))
    k_status = str(model["k_adequacy"]["aggregate_status"])
    sources = [
        _source(
            source_id="cell_source", label="Core8 cell statistics", filename="cell_statistics.csv",
            binding=model["source_bindings"]["cell_statistics.csv"], executed_at=generated_at,
        ),
        _source(
            source_id="result_source", label="Hierarchical bootstrap results", filename="bootstrap_results.json",
            binding=model["source_bindings"]["bootstrap_results.json"], executed_at=generated_at,
        ),
        _source(
            source_id="manifest_source", label="Sealed score manifest", filename="score_manifest.json",
            binding=model["source_bindings"]["score_manifest.json"], executed_at=generated_at,
        ),
    ]
    charts = [
        _chart(
            chart_id="mpnet_delta", title="Full-minus-intervention MPNet cosine by Core8 indicator",
            subtitle="Positive bars mean the intervention reduced similarity to the frozen synthetic target; exact 95% intervals are in tooltips and the table.",
            dataset="mpnet_cells", source_id="cell_source", x_field="topic_label", y_field="estimate",
            y_label="MPNet cosine difference", color_field="arm_label", color_label="Intervention", reference_zero=True,
        ),
        _chart(
            chart_id="bert_delta", title="Full-minus-intervention BERTScore F1 by Core8 indicator",
            subtitle="Bars are equal-meeting K10 estimates; the same paired Full output and target are used within every comparison.",
            dataset="bert_cells", source_id="cell_source", x_field="topic_label", y_field="estimate",
            y_label="BERTScore F1 difference", color_field="arm_label", color_label="Intervention", reference_zero=True,
        ),
        _chart(
            chart_id="diagnostic_funnel", title="Generation diagnostic funnel",
            subtitle="All requested rows remain in the primary semantic analysis even when a preregistered generation gate fails.",
            dataset="funnel", source_id="result_source", x_field="stage", y_field="count", y_label="Generation rows",
            x_label="Diagnostic stage",
            tooltip=(
                {"field": "count", "type": "quantitative", "label": "Rows"},
                {"field": "share_display", "type": "nominal", "label": "Share"},
            ),
        ),
    ]
    cell_columns = [
        {"field": "topic_label", "label": "Indicator", "type": "text"},
        {"field": "arm_label", "label": "Intervention", "type": "text"},
        {"field": "metric_label", "label": "Metric", "type": "text"},
        {"field": "estimate_display", "label": "Full − intervention", "type": "text"},
        {"field": "ci_95", "label": "Hierarchical 95% CI", "type": "text"},
        {"field": "t_display", "label": "Meeting t", "type": "text"},
        {"field": "holm_adjusted_p_display", "label": "Holm p", "type": "text"},
        {"field": "significance_marker", "label": "Marker", "type": "text"},
    ]
    tables = [
        _table(
            table_id="cell_table", title="All 32 target-relative Core8 sensitivity cells",
            subtitle="Bootstrap intervals are primary; meeting-level t tests are supplemental and jointly Holm-adjusted.",
            dataset="cells", source_id="cell_source", columns=cell_columns,
        ),
        _table(
            table_id="adequacy_table", title="K=10 replicate-adequacy diagnostics",
            subtitle="A flagged cell does not invalidate the estimate; it indicates that more stochastic replicates are recommended under the frozen thresholds.",
            dataset="adequacy", source_id="result_source",
            columns=[
                {"field": "topic_label", "label": "Indicator", "type": "text"},
                {"field": "arm_label", "label": "Intervention", "type": "text"},
                {"field": "metric_label", "label": "Metric", "type": "text"},
                {"field": "increase_k_recommended", "label": "Increase k", "type": "boolean"},
                {"field": "increase_k_reasons", "label": "Reason(s)", "type": "text"},
            ],
        ),
        _table(
            table_id="funnel_table", title="Generation diagnostic counts",
            subtitle="Gate validity is descriptive and never a primary-analysis exclusion rule.",
            dataset="funnel", source_id="result_source",
            columns=[
                {"field": "stage", "label": "Stage", "type": "text"},
                {"field": "count", "label": "Rows", "format": "number"},
                {"field": "share_display", "label": "Share", "type": "text"},
            ],
        ),
    ]
    blocks = [
        {"id": "title", "type": "markdown", "body": "# Core8 Indicators Leave-One-Out Sensitivity Results"},
        {"id": "scope", "type": "markdown", "sourceId": "manifest_source", "body": f"## Scope\n\n**{SCOPE_DISCLOSURE}** {REFERENCE_DISCLOSURE}"},
        {"id": "summary", "type": "markdown", "sourceId": "cell_source", "body": (
            "## Technical summary\n\n"
            f"The fresh paper-chk2 cp50 panel contains 128 meetings, 17 prompt variants, and 10 paired stochastic replicates (21,760 outputs). "
            f"{significant} of 32 supplemental meeting-level tests remain below 0.05 after joint Holm correction. "
            f"The largest absolute target-relative change is {largest['topic_label']} / {largest['arm_label']} / {largest['metric_label']} "
            f"at {largest['estimate_display']} (95% CI {largest['ci_95']}). The frozen K-adequacy status is `{k_status}`."
        )},
        {"id": "metric_intro", "type": "markdown", "sourceId": "result_source", "body": (
            "## Target-relative sensitivity estimates\n\nThe estimand is the raw semantic score from the Full prompt minus the score after exact deletion or token-matched neutral replacement. "
            "Ten paired generations are averaged within a meeting; the 128 meetings then receive equal weight."
        )},
        {"id": "mpnet", "type": "chart", "chartId": "mpnet_delta", "layout": "full"},
        {"id": "bert", "type": "chart", "chartId": "bert_delta", "layout": "full"},
        {"id": "cell_table_block", "type": "table", "tableId": "cell_table", "layout": "full"},
        {"id": "methods", "type": "markdown", "sourceId": "result_source", "body": (
            "## Statistical methods\n\nThe primary 95% intervals use 10,000 shared-index hierarchical paired-bootstrap draws: meetings are resampled with replacement and ten paired replicate indices are resampled within every sampled meeting occurrence. "
            "The same random indices are used for all 32 topic–arm–metric cells. Supplemental two-sided one-sample t tests use 128 meeting means (127 degrees of freedom), and all 32 p-values are adjusted as one Holm family."
        )},
        {"id": "adequacy_intro", "type": "markdown", "sourceId": "result_source", "body": (
            "## K adequacy\n\nThe frozen rule inspects K10-minus-K9 point and interval drift, leave-one-replicate drift, the estimated decoding-variance share, and sign/classification stability across K8–K10. "
            "It authorizes only aggregate topic–arm–metric interpretation and never meeting-specific inference."
        )},
        {"id": "adequacy_table_block", "type": "table", "tableId": "adequacy_table", "layout": "full"},
        {"id": "funnel_intro", "type": "markdown", "sourceId": "result_source", "body": f"## Diagnostic funnel\n\n{GATE_DISCLOSURE}"},
        {"id": "funnel_chart_block", "type": "chart", "chartId": "diagnostic_funnel", "layout": "full"},
        {"id": "funnel_table_block", "type": "table", "tableId": "funnel_table", "layout": "full"},
        {"id": "limitations", "type": "markdown", "sourceId": "manifest_source", "body": (
            "## Limitations\n\n"
            + "\n".join(f"- {item}" for item in dict.fromkeys([
                SCOPE_DISCLOSURE, REFERENCE_DISCLOSURE, GATE_DISCLOSURE, OOD_DISCLOSURE,
                PANEL_REUSE_DISCLOSURE, *model["limitations"]
            ]))
        )},
        {"id": "next", "type": "markdown", "sourceId": "manifest_source", "body": (
            "## Appropriate interpretation and next checks\n\nUse the panel to describe which prompt interventions are associated with reduced similarity to the fixed synthetic target for the selected checkpoint. "
            "Do not interpret the magnitude as a causal contribution or as evidence that a block is sufficient. If K adequacy recommends more replicates, extend the paired seed panel before relying on borderline cells; factual fidelity and official-Minutes style require separate evaluations."
        )},
    ]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Core8 Indicators Leave-One-Out Sensitivity Results",
            "description": "Fresh paper-chk2 cp50 target-relative Core8 K10 sensitivity diagnostic.",
            "generatedAt": generated_at,
            "cards": [],
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "cells": cells,
                "mpnet_cells": [row for row in cells if row["metric"] == "mpnet_cosine"],
                "bert_cells": [row for row in cells if row["metric"] == "bertscore_f1"],
                "adequacy": cells,
                "funnel": funnel_rows,
            },
        },
        "sources": copy.deepcopy(sources),
    }


def build_markdown_report(model: Mapping[str, Any]) -> str:
    cells = list(model["cells"])
    significant = [row for row in cells if row["holm_reject_005"]]
    largest = max(cells, key=lambda row: abs(float(row["estimate"])))
    funnel = model["funnel"]
    lines = [
        "# Core8 Indicators Leave-One-Out Sensitivity Results",
        "",
        "## Answer-first summary",
        "",
        (
            f"The fresh paper-level Model chk-2 checkpoint-50 evaluation covers 128 regular FOMC "
            f"meetings from 1993–2008, 17 variants per meeting, and 10 paired stochastic generations "
            f"per variant (21,760 outputs). {len(significant)} of 32 supplemental meeting-level tests "
            f"remain below 0.05 after joint Holm correction. The largest absolute target-relative "
            f"difference is {largest['topic_label']} under {largest['arm_label']} for "
            f"{largest['metric_label']}: {largest['estimate_display']} with hierarchical 95% CI "
            f"{largest['ci_95']}."
        ),
        "",
        f"**{SCOPE_DISCLOSURE}** {REFERENCE_DISCLOSURE}",
        "",
        "## Cell results",
        "",
        "| Indicator | Intervention | Metric | Full − intervention | Hierarchical 95% CI | t(127) | Holm p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in cells:
        marker = row["significance_marker"]
        lines.append(
            f"| {row['topic_label']} | {row['arm_label']} | {row['metric_label']} | "
            f"{row['estimate_display']}{marker} | {row['ci_95']} | {row['t_display']} | "
            f"{row['holm_adjusted_p_display']} |"
        )
    lines.extend(
        [
            "",
            "A positive difference means that the intervention reduced semantic similarity to the fixed synthetic target. Bootstrap intervals are the primary uncertainty summary. The t statistics are calculated from 128 meeting means after averaging K=10 paired generations within each meeting; their 32 two-sided p-values form one Holm family.",
            "",
            "## Generation diagnostic funnel",
            "",
            f"- Requested and semantically scored: {funnel['requested_rows']:,}",
            f"- Nonempty answers: {funnel['nonempty_answer_rows']:,}",
            f"- Joint core-gate PASS: {funnel['preregistered_core_valid_rows']:,}",
            f"- Joint core-gate FAIL: {funnel['preregistered_core_invalid_rows']:,}",
            f"- Rows removed from the primary analysis: {funnel['rows_removed_from_primary']:,}",
            "",
            GATE_DISCLOSURE,
            "",
            "## Statistical design",
            "",
            "For each bootstrap draw, 128 meetings are sampled with replacement. Ten paired replicate indices are then sampled with replacement within each sampled meeting occurrence. One shared index plan is used across all 32 topic–arm–metric cells, preserving the paired design. The primary interval is the 2.5th–97.5th percentile range over 10,000 draws. A meeting-only bootstrap is retained as a sensitivity check.",
            "",
            "## K adequacy",
            "",
            f"The frozen aggregate status is `{model['k_adequacy']['aggregate_status']}`. The rule checks K10-minus-K9 point drift (>0.005), maximum CI-endpoint drift (>0.010), leave-one-replicate drift (>0.010), decoding-variance share (>0.20), and decisive sign/classification stability across K8–K10. It does not authorize meeting-specific inference.",
            "",
            "## Limitations",
            "",
        ]
    )
    for limitation in dict.fromkeys(
        [SCOPE_DISCLOSURE, REFERENCE_DISCLOSURE, GATE_DISCLOSURE, OOD_DISCLOSURE,
         PANEL_REUSE_DISCLOSURE, *model["limitations"]]
    ):
        lines.append(f"- {limitation}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These results can describe checkpoint-specific sensitivity to exact deletion and token-matched neutral replacement. They cannot establish causal topic importance, information sufficiency, official-Minutes similarity, or factual correctness. Borderline cells should be re-estimated with additional paired stochastic replicates whenever the K-adequacy rule recommends increasing K.",
            "",
        ]
    )
    return "\n".join(lines)


def _tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _latex_value(row: Mapping[str, Any]) -> str:
    estimate = float(row["estimate"])
    marker = str(row["significance_marker"])
    t_display = str(row["t_display"])
    marker_tex = f"^{{{marker}}}" if marker else ""
    t_tex = r"\text{undefined}" if t_display == "undefined" else t_display
    return rf"\makecell{{\({estimate:+.6f}{marker_tex}\)\\\( ({t_tex}) \)}}"


def build_latex_table(model: Mapping[str, Any]) -> str:
    lookup = {
        (row["topic"], row["arm"], row["metric"]): row for row in model["cells"]
    }
    status = _tex_escape(str(model["k_adequacy"]["aggregate_status"]))
    rows: list[str] = []
    for index, topic in enumerate(scorer.TOPICS):
        values = [
            _latex_value(lookup[(topic, arm, metric)])
            for arm in scorer.ARMS for metric in scorer.METRICS
        ]
        rows.append(
            "            "
            + _tex_escape(TOPIC_LABELS[topic])
            + "\n            & "
            + "\n            & ".join(values)
            + r" \\" 
        )
        if index != len(scorer.TOPICS) - 1:
            rows.append("\n            \\addlinespace\n")
    body = "\n".join(rows)
    return rf"""\begin{{table}}[htbp]
    \centering
    \footnotesize
    \setlength{{\tabcolsep}}{{3pt}}
    \renewcommand{{\arraystretch}}{{1.00}}
    \caption{{Core8 Indicators Leave-One-Out Sensitivity Results for
    \textit{{Model chk-2}} (Checkpoint 50).}}
    \label{{tab:ch2:paper_chk2_core8_loo_ttest}}
    \begin{{threeparttable}}
        \begin{{tabularx}}{{\textwidth}}{{
            @{{}}
            >{{\raggedright\arraybackslash}}p{{0.21\textwidth}}
            >{{\centering\arraybackslash}}X
            >{{\centering\arraybackslash}}X
            >{{\centering\arraybackslash}}X
            >{{\centering\arraybackslash}}X
            @{{}}
        }}
            \toprule
            & \multicolumn{{2}}{{c}}{{\textbf{{Full \( - \) Exact deletion}}}}
            & \multicolumn{{2}}{{c}}{{\textbf{{Full \( - \) Neutral replacement}}}} \\
            \cmidrule(lr){{2-3}}
            \cmidrule(lr){{4-5}}

            \textbf{{Indicator}}
            & \makecell{{\textbf{{MPNet}}\\\textbf{{cosine}}}}
            & \makecell{{\textbf{{BERTScore}}\\\textbf{{F1}}}}
            & \makecell{{\textbf{{MPNet}}\\\textbf{{cosine}}}}
            & \makecell{{\textbf{{BERTScore}}\\\textbf{{F1}}}} \\
            \midrule

{body}

            \bottomrule
        \end{{tabularx}}

        \begin{{tablenotes}}[flushleft]
            \tiny
            \item \textit{{Notes:}} Each point estimate is the raw semantic-similarity
            score for the Full Core8 prompt minus the corresponding score after
            intervention. A positive value indicates that deleting or neutralizing
            the indicator reduces similarity to the fixed source-grounded synthetic
            Minutes reference; the reference is not official FOMC Minutes.

            \item Values in parentheses are paired \(t\)-statistics calculated from
            128 meeting-level differences after first averaging the \(K=10\) paired
            stochastic replicates within each meeting. Meetings receive equal weight,
            and all tests have 127 degrees of freedom.

            \item The 32 two-sided \(p\)-values are adjusted jointly using Holm's
            procedure: \(^{{*}}p_{{\mathrm{{Holm}}}}<0.05\),
            \(^{{**}}p_{{\mathrm{{Holm}}}}<0.01\), and
            \(^{{***}}p_{{\mathrm{{Holm}}}}<0.001\).

            \item The paired \(t\)-tests are supplemental. The primary uncertainty
            analysis is the 10,000-draw shared-index hierarchical paired bootstrap;
            its percentile intervals are unadjusted for multiplicity.

            \item Generation validity gates are diagnostic only and do not filter or
            zero-penalize nonempty outputs in this target-relative semantic analysis.
            The frozen K-adequacy status is \texttt{{{status}}}; the table does not
            support meeting-specific inference or causal topic attribution.
        \end{{tablenotes}}
    \end{{threeparttable}}
\end{{table}}
"""


def build_latex_results(model: Mapping[str, Any]) -> str:
    cells = list(model["cells"])
    significant = [row for row in cells if row["holm_reject_005"]]
    exact = sum(row["arm"] == "exact_deletion" for row in significant)
    neutral = sum(row["arm"] == "token_matched_neutral" for row in significant)
    mpnet = sum(row["metric"] == "mpnet_cosine" for row in significant)
    bert = sum(row["metric"] == "bertscore_f1" for row in significant)
    bootstrap_decisive = sum(float(row["ci_lower"]) > 0 or float(row["ci_upper"]) < 0 for row in cells)
    largest = max(cells, key=lambda row: abs(float(row["estimate"])))
    status = _tex_escape(str(model["k_adequacy"]["aggregate_status"]))
    flagged = sum(bool(row["increase_k_recommended"]) for row in cells)
    return rf"""Table~\ref{{tab:ch2:paper_chk2_core8_loo_ttest}} reports the
target-relative leave-one-out sensitivity results for the selected
\textit{{Model chk-2}} checkpoint. Of the 32 supplemental meeting-level
comparisons, {len(significant)} remain significant after joint Holm adjustment:
{exact} under exact deletion and {neutral} under token-matched neutral
replacement, comprising {mpnet} MPNet-cosine and {bert} BERTScore-F1
comparisons. The primary 10,000-draw hierarchical paired-bootstrap interval
excludes zero for {bootstrap_decisive} cells. The largest absolute point
estimate is observed for {_tex_escape(str(largest['topic_label']))} under
{_tex_escape(str(largest['arm_label']).lower())} using
{_tex_escape(str(largest['metric_label']))}
(\(\Delta={float(largest['estimate']):+.6f}\), 95\% CI
\([{float(largest['ci_lower']):+.6f}, {float(largest['ci_upper']):+.6f}]\)).

These estimates compare every generated paragraph with the same fixed,
source-grounded synthetic Minutes reference for its meeting. A positive
\(\Delta\) therefore means that the intervention reduced target similarity;
it does not measure causal topic importance, factual correctness, or similarity
to official FOMC Minutes. Generation hard gates are diagnostic and do not
filter the semantic estimand. The frozen replicate-adequacy verdict is
\texttt{{{status}}}, with {flagged} of 32 cells flagged by at least one
prespecified K-adequacy rule, so the results do not authorize meeting-specific
inference. Finally, the same 1993--2008 Core8 panel was scored in recorded
experiments on 2026-08-14, 2026-08-17, and 2026-08-24; repeated-holdout and
model-development reuse cannot be ruled out. Although its meeting dates do not
overlap the task-specific 2009--2025 Minutes-rewriting training meetings, this
rerun is not an evaluation on a newly untouched or prospective holdout.
"""


def _write_new_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_new_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _intended_binding(staged: Path, final: Path) -> dict[str, Any]:
    value = _binding(staged)
    value["path"] = str(final.expanduser().resolve())
    return value


def _package_static_html(
    *, markdown_path: Path, report_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Render the frozen Markdown report without a mutable plugin-cache dependency."""

    markdown_source = markdown_path.read_text(encoding="utf-8")
    renderer = MarkdownIt(
        "commonmark",
        {"html": False, "linkify": False, "typographer": False},
    ).enable("table")
    body = renderer.render(markdown_source)
    title = "Core8 Indicators Leave-One-Out Sensitivity Results"
    document = f"""<!doctype html>
<html lang="en" data-paper-chk2-core8-report="true">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
<meta name="referrer" content="no-referrer">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light dark; --bg:#ffffff; --ink:#18212b; --muted:#5f6b76; --rule:#d8dee4; --accent:#205ea6; --soft:#f6f8fa; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161b22; --ink:#e6edf3; --muted:#9da7b1; --rule:#30363d; --accent:#79c0ff; --soft:#21262d; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.58 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ width:min(1180px, calc(100% - 40px)); margin:0 auto; padding:52px 0 80px; }}
h1 {{ margin:0 0 30px; font-size:clamp(28px,4vw,43px); line-height:1.08; letter-spacing:-0.025em; }}
h2 {{ margin:44px 0 14px; padding-top:10px; border-top:1px solid var(--rule); font-size:22px; }}
h3 {{ margin:28px 0 10px; font-size:18px; }}
p,li {{ max-width:88ch; }}
strong {{ color:var(--ink); }}
code {{ padding:.12em .35em; border-radius:4px; background:var(--soft); font-family:ui-monospace,SFMono-Regular,Consolas,monospace; }}
table {{ width:100%; margin:18px 0 32px; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
th,td {{ padding:9px 10px; border-bottom:1px solid var(--rule); text-align:left; vertical-align:top; }}
th {{ position:sticky; top:0; background:var(--soft); font-size:13px; }}
tr:hover td {{ background:var(--soft); }}
blockquote {{ margin:20px 0; padding:2px 18px; border-left:4px solid var(--accent); color:var(--muted); }}
hr {{ border:0; border-top:1px solid var(--rule); margin:38px 0; }}
a {{ color:var(--accent); }}
@media (max-width:760px) {{ main {{ width:min(100% - 24px,1180px); padding-top:28px; overflow-x:auto; }} table {{ min-width:820px; }} }}
@media print {{ :root {{ color-scheme:light; --bg:#fff; --ink:#000; --muted:#444; --rule:#bbb; --soft:#f4f4f4; }} body {{ font-size:10pt; }} main {{ width:100%; padding:0; }} th {{ position:static; }} h2 {{ break-after:avoid; }} tr {{ break-inside:avoid; }} }}
</style>
</head>
<body><main>{body}</main></body>
</html>
"""
    _write_new_text(report_path, document)
    package_version = importlib.metadata.version("markdown-it-py")
    receipt = {
        "ok": True,
        "status": "complete",
        "renderer": "repository-owned-static-html-v1",
        "input_sha256": _sha_file(markdown_path),
        "output_sha256": _sha_file(report_path),
        "network_dependencies": False,
        "self_contained": True,
    }
    builder = {
        "engine": "markdown-it-py",
        "version": package_version,
        "module": _binding(Path(__import__("markdown_it").__file__)),
        "stylesheet": "embedded",
        "javascript": "none",
    }
    return receipt, builder


def validate_report(
    run_root: Path = DEFAULT_OUTPUT_ROOT, report_root: Path | None = None
) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    output = (report_root or run_root / "report").expanduser().resolve()
    if not output.is_dir() or output.is_symlink():
        raise PaperChk2Core8ReportError(f"report directory is missing or unsafe: {output}")
    expected_names = {
        "artifact.json", "report.html", "report.md", "core8_loo_table.tex",
        "core8_loo_results.tex", "report_manifest.json"
    }
    if {path.name for path in output.iterdir()} != expected_names:
        raise PaperChk2Core8ReportError("report artifact inventory drift")
    manifest = _read_json(output / "report_manifest.json")
    _validate_seal(manifest)
    if (
        manifest.get("schema_version") != REPORT_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("immutable") is not True
        or manifest.get("report_id") != REPORT_ID
    ):
        raise PaperChk2Core8ReportError("report role/status drift")
    artifacts = manifest.get("artifacts")
    expected_paths = {
        "canonical_artifact": output / "artifact.json",
        "portable_html": output / "report.html",
        "markdown_report": output / "report.md",
        "latex_table": output / "core8_loo_table.tex",
        "latex_results": output / "core8_loo_results.tex",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(expected_paths):
        raise PaperChk2Core8ReportError("report manifest artifact inventory drift")
    for role, path in expected_paths.items():
        observed = _binding(path)
        expected = artifacts[role]
        if not isinstance(expected, Mapping) or any(expected.get(field) != observed[field] for field in ("path", "bytes", "sha256")):
            raise PaperChk2Core8ReportError(f"report artifact binding drift: {role}")
    renderer = manifest.get("renderer")
    if not isinstance(renderer, Mapping) or any(renderer.get(field) != _binding(Path(__file__))[field] for field in ("path", "bytes", "sha256")):
        raise PaperChk2Core8ReportError("report renderer source drift")
    model = load_report_model(run_root)
    if _read_json(output / "artifact.json") != build_canonical_artifact(model):
        raise PaperChk2Core8ReportError("canonical report differs from deterministic replay")
    if (output / "report.md").read_text(encoding="utf-8") != build_markdown_report(model):
        raise PaperChk2Core8ReportError("Markdown report differs from deterministic replay")
    if (output / "core8_loo_table.tex").read_text(encoding="utf-8") != build_latex_table(model):
        raise PaperChk2Core8ReportError("LaTeX table differs from deterministic replay")
    if (output / "core8_loo_results.tex").read_text(encoding="utf-8") != build_latex_results(model):
        raise PaperChk2Core8ReportError("LaTeX results prose differs from deterministic replay")
    artifact_text = _canonical(build_canonical_artifact(model))
    markdown_text = build_markdown_report(model)
    for disclosure in (
        SCOPE_DISCLOSURE, REFERENCE_DISCLOSURE, GATE_DISCLOSURE, OOD_DISCLOSURE,
        PANEL_REUSE_DISCLOSURE,
    ):
        if disclosure not in artifact_text or disclosure not in markdown_text:
            raise PaperChk2Core8ReportError("required interpretation disclosure is missing")
    html = (output / "report.html").read_text(encoding="utf-8")
    required_html = (
        "Core8 Indicators Leave-One-Out Sensitivity Results",
        SCOPE_DISCLOSURE,
        REFERENCE_DISCLOSURE,
        GATE_DISCLOSURE,
        OOD_DISCLOSURE,
        PANEL_REUSE_DISCLOSURE,
        'data-paper-chk2-core8-report="true"',
    )
    if any(value not in html for value in required_html):
        raise PaperChk2Core8ReportError("self-contained HTML content drift")
    if "<script" in html or "default-src 'none'" not in html:
        raise PaperChk2Core8ReportError("self-contained HTML security contract drift")
    builder = manifest.get("report_builder")
    if (
        not isinstance(builder, Mapping)
        or builder.get("engine") != "markdown-it-py"
        or builder.get("javascript") != "none"
    ):
        raise PaperChk2Core8ReportError("report builder contract drift")
    receipt = manifest.get("delivery_receipt")
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("ok") is not True
        or receipt.get("self_contained") is not True
        or receipt.get("network_dependencies") is not False
    ):
        raise PaperChk2Core8ReportError("report delivery receipt drift")
    return copy.deepcopy(manifest)


def render_report(
    run_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    report_root: Path | None = None,
    node_bin: Path | None = None,
    plugin_root: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    if node_bin is not None or plugin_root is not None:
        raise PaperChk2Core8ReportError(
            "external report packager overrides are not supported by the sealed "
            "repository-owned renderer"
        )
    run_root = run_root.expanduser().resolve()
    output = (report_root or run_root / "report").expanduser().resolve()
    if output.exists() or output.is_symlink():
        if resume and output.is_dir() and not output.is_symlink():
            return validate_report(run_root, output)
        raise PaperChk2Core8ReportError(f"report output already exists: {output}")
    model = load_report_model(run_root)
    artifact = build_canonical_artifact(model)
    markdown = build_markdown_report(model)
    latex = build_latex_table(model)
    latex_results = build_latex_results(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)).resolve()
    published = False
    try:
        artifact_path = staging / "artifact.json"
        html_path = staging / "report.html"
        markdown_path = staging / "report.md"
        latex_path = staging / "core8_loo_table.tex"
        latex_results_path = staging / "core8_loo_results.tex"
        _write_new_json(artifact_path, artifact)
        _write_new_text(markdown_path, markdown)
        _write_new_text(latex_path, latex)
        _write_new_text(latex_results_path, latex_results)
        try:
            receipt, builder = _package_static_html(
                markdown_path=markdown_path,
                report_path=html_path,
            )
        except Exception as exc:
            raise PaperChk2Core8ReportError(f"self-contained HTML packaging failed: {exc}") from exc
        refreshed = load_report_model(run_root)
        if refreshed != model:
            raise PaperChk2Core8ReportError("score bundle changed during report rendering")
        manifest = _seal(
            {
                "schema_version": REPORT_SCHEMA,
                "status": "complete",
                "immutable": True,
                "report_id": REPORT_ID,
                "created_at_utc": model["generated_at_utc"],
                "operation": "render_only_no_generation_semantic_scoring_or_bootstrap_recomputation",
                "score_manifest": copy.deepcopy(model["source_bindings"]["score_manifest.json"]),
                "artifacts": {
                    "canonical_artifact": {
                        **_intended_binding(artifact_path, output / "artifact.json"),
                        "media_type": "application/json",
                    },
                    "portable_html": {
                        **_intended_binding(html_path, output / "report.html"),
                        "media_type": "text/html; charset=utf-8",
                        "self_contained": True,
                    },
                    "markdown_report": {
                        **_intended_binding(markdown_path, output / "report.md"),
                        "media_type": "text/markdown; charset=utf-8",
                    },
                    "latex_table": {
                        **_intended_binding(latex_path, output / "core8_loo_table.tex"),
                        "media_type": "application/x-tex; charset=utf-8",
                    },
                    "latex_results": {
                        **_intended_binding(latex_results_path, output / "core8_loo_results.tex"),
                        "media_type": "application/x-tex; charset=utf-8",
                    },
                },
                "report_builder": builder,
                "delivery_receipt": receipt,
                "renderer": _binding(Path(__file__)),
                "report_contract": {
                    "audience": "technical_and_paper_review",
                    "fresh_paper_chk2_cp50_only": True,
                    "target_relative_not_causal": True,
                    "synthetic_reference_not_official_minutes": True,
                    "diagnostic_gates_do_not_filter_primary": True,
                    "hierarchical_bootstrap_draws": scorer.BOOTSTRAP_DRAWS,
                    "joint_holm_family_size": 32,
                    "latex_not_auto_inserted_into_chapter2": True,
                },
            }
        )
        _write_new_json(staging / "report_manifest.json", manifest)
        os.rename(staging, output)
        published = True
        return validate_report(run_root, output)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("render", "validate"))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--node-bin", type=Path)
    parser.add_argument("--plugin-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "render":
        result = render_report(
            args.run_root,
            report_root=args.report_root,
            node_bin=args.node_bin,
            plugin_root=args.plugin_root,
            resume=args.resume,
        )
    else:
        result = validate_report(args.run_root, args.report_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
