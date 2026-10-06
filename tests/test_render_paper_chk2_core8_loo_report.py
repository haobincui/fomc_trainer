from __future__ import annotations

from pathlib import Path

import pytest

from jobs.eval import render_paper_chk2_core8_loo_report as target
from jobs.eval import score_paper_chk2_core8_loo_vllm_k10 as scorer


def _model() -> dict:
    cells = []
    position = 0
    for topic in scorer.TOPICS:
        for arm in scorer.ARMS:
            for metric in scorer.METRICS:
                estimate = 0.001 * (position + 1)
                marker = "***" if position == 0 else "*" if position == 1 else ""
                cells.append(
                    {
                        "topic": topic,
                        "topic_label": target.TOPIC_LABELS[topic],
                        "arm": arm,
                        "arm_label": target.ARM_LABELS[arm],
                        "metric": metric,
                        "metric_label": target.METRIC_LABELS[metric],
                        "series_label": f"{target.ARM_LABELS[arm]} · {target.METRIC_LABELS[metric]}",
                        "estimate": estimate,
                        "estimate_display": f"{estimate:+.6f}",
                        "ci_lower": estimate - 0.002,
                        "ci_upper": estimate + 0.002,
                        "ci_95": f"[{estimate - 0.002:+.6f}, {estimate + 0.002:+.6f}]",
                        "t_statistic": 1.0 + position,
                        "t_display": f"{1.0 + position:.3f}",
                        "holm_adjusted_p": 0.001 if marker else 0.2,
                        "holm_adjusted_p_display": "0.0010" if marker else "0.2000",
                        "significance_marker": marker,
                        "holm_reject_005": bool(marker),
                        "positive_meetings": 100,
                        "negative_meetings": 28,
                        "increase_k_recommended": position % 2 == 0,
                        "increase_k_reasons": "decoding_variance_share_above_threshold" if position % 2 == 0 else "None",
                        "decoding_variance_share": 0.3 if position % 2 == 0 else 0.1,
                    }
                )
                position += 1
    source = {"path": "/tmp/source", "bytes": 100, "sha256": "a" * 64}
    return {
        "generated_at_utc": "2026-09-01T00:00:00Z",
        "cells": cells,
        "funnel": {
            "requested_rows": 21_760,
            "raw_semantic_scored_rows": 21_760,
            "nonempty_answer_rows": 21_700,
            "empty_answer_rows": 60,
            "preregistered_core_valid_rows": 2_000,
            "preregistered_core_invalid_rows": 19_760,
            "rows_removed_from_primary": 0,
        },
        "funnel_rows": [
            {"stage": "Requested and semantically scored", "count": 21_760, "share": 1.0, "share_display": "100.0%"},
            {"stage": "Nonempty answer", "count": 21_700, "share": 21_700 / 21_760, "share_display": "99.7%"},
            {"stage": "Joint core gate PASS (diagnostic)", "count": 2_000, "share": 2_000 / 21_760, "share_display": "9.2%"},
            {"stage": "Rows retained in primary analysis", "count": 21_760, "share": 1.0, "share_display": "100.0%"},
        ],
        "coverage": {"meetings": 128, "generation_rows": 21_760},
        "bootstrap_contract": {"draws": 10_000},
        "multiplicity": {"family_size": 32, "adjustment": "Holm"},
        "k_adequacy": {
            "aggregate_status": "increase_k_recommended",
            "increase_k_recommended": True,
        },
        "limitations": ["The intervals are unadjusted."],
        "source_bindings": {
            "cell_statistics.csv": dict(source),
            "bootstrap_results.json": dict(source),
            "score_manifest.json": dict(source),
            "generation_diagnostic_failures.jsonl": dict(source),
        },
    }


def test_artifact_uses_native_blocks_and_preserves_all_disclosures():
    artifact = target.build_canonical_artifact(_model())
    assert artifact["surface"] == "report"
    assert len(artifact["snapshot"]["datasets"]["cells"]) == 32
    assert len(artifact["manifest"]["charts"]) == 3
    assert len(artifact["manifest"]["tables"]) == 3
    text = target._canonical(artifact)
    for disclosure in (
        target.SCOPE_DISCLOSURE,
        target.REFERENCE_DISCLOSURE,
        target.GATE_DISCLOSURE,
        target.OOD_DISCLOSURE,
        target.PANEL_REUSE_DISCLOSURE,
    ):
        assert disclosure in text


def test_markdown_is_answer_first_and_discloses_repeated_panel():
    text = target.build_markdown_report(_model())
    assert text.startswith("# Core8 Indicators Leave-One-Out Sensitivity Results")
    assert "## Answer-first summary" in text
    assert "| Indicator | Intervention | Metric |" in text
    assert "Rows removed from the primary analysis: 0" in text
    assert target.PANEL_REUSE_DISCLOSURE in text


def test_latex_table_matches_frozen_four_cell_column_layout():
    text = target.build_latex_table(_model())
    assert r"\label{tab:ch2:paper_chk2_core8_loo_ttest}" in text
    assert r"\textbf{Full \( - \) Exact deletion}" in text
    assert r"\textbf{Full \( - \) Neutral replacement}" in text
    assert text.count(r"\addlinespace") == 7
    assert r"+0.001000^{***}" in text
    assert "10,000-draw shared-index hierarchical paired bootstrap" in text
    assert r"\texttt{increase\_k\_recommended}" in text


def test_latex_results_are_dynamic_and_caveated():
    text = target.build_latex_results(_model())
    assert "Of the 32 supplemental meeting-level" in text
    assert "2 remain significant after joint Holm adjustment" in text
    assert "repeated-holdout" in text
    assert "2026-08-14" in text
    assert "newly untouched or prospective holdout" in text
    assert "do not authorize meeting-specific" in text


def test_tex_escape_handles_report_identifiers():
    assert target._tex_escape("increase_k&50%") == r"increase\_k\&50\%"


def test_static_html_packager_is_self_contained(tmp_path: Path):
    markdown = tmp_path / "report.md"
    html = tmp_path / "report.html"
    target._write_new_text(
        markdown,
        "# Core8 Indicators Leave-One-Out Sensitivity Results\n\n"
        f"{target.SCOPE_DISCLOSURE}\n\n"
        f"{target.REFERENCE_DISCLOSURE}\n\n"
        f"{target.GATE_DISCLOSURE}\n\n"
        f"{target.OOD_DISCLOSURE}\n\n"
        f"{target.PANEL_REUSE_DISCLOSURE}\n\n"
        "| Metric | Value |\n|---|---:|\n| MPNet | 0.1 |\n",
    )
    receipt, builder = target._package_static_html(
        markdown_path=markdown,
        report_path=html,
    )

    rendered = html.read_text(encoding="utf-8")
    assert receipt["ok"] is True
    assert receipt["network_dependencies"] is False
    assert builder["engine"] == "markdown-it-py"
    assert builder["javascript"] == "none"
    assert '<table>' in rendered
    assert '<script' not in rendered
    assert "default-src 'none'" in rendered
    assert 'data-paper-chk2-core8-report="true"' in rendered


def test_external_packager_override_is_rejected_before_io(tmp_path: Path):
    with pytest.raises(
        target.PaperChk2Core8ReportError,
        match="external report packager overrides",
    ):
        target.render_report(tmp_path, node_bin=tmp_path / "node")
