from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from jobs.eval import render_paper_chk2_text_similarity_report as report


def _score_row(model: str, metric: str, estimate: float) -> dict[str, object]:
    return {
        "bootstrap_scheme": report.PRIMARY_SCHEME,
        "scoring_policy": report.PRIMARY_POLICY,
        "policy_label": report.POLICY_LABELS[report.PRIMARY_POLICY],
        "model_id": model,
        "model": report.MODEL_LABELS[model],
        "metric": metric,
        "metric_label": report.METRIC_LABELS[metric],
        "estimate": estimate,
        "estimate_display": f"{estimate:.4f}",
        "ci_lower": estimate - 0.01,
        "ci_upper": estimate + 0.01,
        "ci_95": f"[{estimate - 0.01:.4f}, {estimate + 0.01:.4f}]",
        "samples": 391,
        "meetings": 124,
        "replicates": 10,
        "bootstrap_draws": 1000,
    }


def _report_model(tmp_path: Path) -> dict[str, object]:
    primary = []
    for metric_offset, metric in enumerate(report.METRIC_ORDER):
        for model_index, model in enumerate(report.MODEL_ORDER):
            primary.append(_score_row(model, metric, 0.60 + metric_offset * 0.1 + model_index * 0.02))

    per_k = []
    for row in primary:
        for replicate in range(10):
            item = dict(row)
            item.update({"replicate_id": replicate, "k": replicate + 1, "k_label": f"k={replicate + 1}"})
            per_k.append(item)

    splits = []
    for row in primary:
        for split in report.SPLIT_ORDER:
            item = dict(row)
            item.update({"split": split, "split_label": split.title()})
            splits.append(item)

    policies = []
    for row in primary:
        for policy in report.POLICY_ORDER:
            item = dict(row)
            item.update({"scoring_policy": policy, "policy_label": report.POLICY_LABELS[policy]})
            policies.append(item)

    contrasts = []
    for metric in report.METRIC_ORDER:
        for contrast_id, before, after in (
            ("chk1_minus_chk0", "chk0", "chk1"),
            ("chk2_minus_chk1", "chk1", "chk2"),
            ("chk2_minus_chk0", "chk0", "chk2"),
        ):
            contrasts.append(
                {
                    "bootstrap_scheme": report.PRIMARY_SCHEME,
                    "scoring_policy": report.PRIMARY_POLICY,
                    "aggregate_scope": "pooled_k10",
                    "replicate_id": None,
                    "contrast_id": contrast_id,
                    "contrast": f"{report.MODEL_LABELS[after]} − {report.MODEL_LABELS[before]}",
                    "metric": metric,
                    "metric_label": report.METRIC_LABELS[metric],
                    "series_label": f"{after} − {before} · {report.METRIC_LABELS[metric]}",
                    "estimate": 0.02,
                    "estimate_display": "+0.0200",
                    "ci_lower": 0.01,
                    "ci_upper": 0.03,
                    "ci_95": "[+0.0100, +0.0300]",
                    "p_value": 0.01,
                    "p_value_display": "0.0100",
                    "holm_adjusted_p": 0.03,
                    "holm_adjusted_p_display": "0.0300",
                    "holm_reject_005": True,
                    "p_value_method": "paired_sign_flip",
                    "bootstrap_draws": 1000,
                }
            )

    bindings = {
        filename: {"path": str(tmp_path / filename), "bytes": 1, "sha256": "a" * 64}
        for filename in report.REQUIRED_INPUTS
    }
    return {
        "generated_at_utc": "2026-09-01T00:00:00Z",
        "score_root": str(tmp_path),
        "source_bindings": bindings,
        "coverage": {
            "samples": 391,
            "meetings": 124,
            "replicates": 10,
            "generation_rows": 11730,
            "split_rows": {"train": 305, "validation": 42, "test": 44},
            "split_meetings": {"train": 98, "validation": 13, "test": 13},
        },
        "bootstrap_contract": {"draws": 1000},
        "multiple_testing": {"method": "Holm", "tests": 6},
        "primary_summary": primary,
        "primary_per_k": per_k,
        "primary_splits": splits,
        "policy_summary": policies,
        "primary_contrasts": contrasts,
        "delivery": [
            {
                "model_id": model,
                "model": report.MODEL_LABELS[model],
                "generations": 3910,
                "delivery_valid": 3900,
                "delivery_invalid": 10,
                "recovered_nonempty": 3905,
                "recovered_empty": 5,
                "delivery_valid_rate": 3900 / 3910,
                "delivery_valid_rate_display": "99.74%",
                "recovered_nonempty_rate_display": "99.87%",
                "failure_counts": '{"format": 10}',
            }
            for model in report.MODEL_ORDER
        ],
        "limitations": [
            "All 391 samples come from the training-only paper-CHK2 release.",
            "The ten stochastic replicates are repeated generations, not independent datasets.",
        ],
    }


def test_builds_native_technical_report_artifact(tmp_path: Path) -> None:
    artifact = report.build_canonical_artifact(_report_model(tmp_path))

    assert artifact["surface"] == "report"
    assert artifact["snapshot"]["status"] == "ready"
    assert set(artifact) == {"surface", "manifest", "snapshot", "sources"}
    chart_types = {chart["type"] for chart in artifact["manifest"]["charts"]}
    assert {"bar", "line"} <= chart_types
    assert len(artifact["manifest"]["tables"]) == 5
    assert artifact["snapshot"]["datasets"]["coverage"] == [
        {
            "samples": 391,
            "meetings": 124,
            "replicates": 10,
            "generation_rows": 11730,
        }
    ]
    assert len(artifact["snapshot"]["datasets"]["per_k_mpnet"]) == 30
    assert len(artifact["snapshot"]["datasets"]["split_all"]) == 18
    text = json.dumps(artifact, ensure_ascii=False)
    assert report.SCOPE_DISCLOSURE in text
    assert report.REFERENCE_DISCLOSURE in text
    assert report.METRIC_DISCLOSURE in text
    assert "Holm" in text


def test_cli_accepts_root_aliases_and_resume(tmp_path: Path) -> None:
    args = report.build_parser().parse_args(
        [
            "--score-root",
            str(tmp_path / "scores"),
            "--report-root",
            str(tmp_path / "report"),
            "--resume",
        ]
    )
    assert args.score_dir == tmp_path / "scores"
    assert args.report_dir == tmp_path / "report"
    assert args.resume is True


def test_portable_renderer_accepts_artifact(tmp_path: Path) -> None:
    if not report.DEFAULT_PLUGIN_ROOT.joinpath(report.PACKAGER_RELATIVE).is_file():
        pytest.skip("installed Data Analytics portable renderer unavailable")
    artifact_path = tmp_path / "artifact.json"
    html_path = tmp_path / "report.html"
    artifact_path.write_text(
        json.dumps(report.build_canonical_artifact(_report_model(tmp_path)), ensure_ascii=False),
        encoding="utf-8",
    )

    receipt, _bindings = report._package_report(
        artifact_path=artifact_path,
        report_path=html_path,
        node_bin=None,
        plugin_root=None,
    )

    assert receipt["ok"] is True
    assert html_path.stat().st_size > 10_000
    assert "in-sample training-release reconstruction diagnostic" in html_path.read_text(encoding="utf-8")


def test_resume_deep_validates_and_returns_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not report.DEFAULT_PLUGIN_ROOT.joinpath(report.PACKAGER_RELATIVE).is_file():
        pytest.skip("installed Data Analytics portable renderer unavailable")
    model = _report_model(tmp_path)
    monkeypatch.setattr(report, "load_report_model", lambda _score: copy.deepcopy(model))
    report_dir = tmp_path / "sealed-report"

    first = report.render_and_seal_report(
        score_dir=tmp_path / "scores", report_dir=report_dir
    )
    second = report.render_and_seal_report(
        score_dir=tmp_path / "scores", report_dir=report_dir, resume=True
    )

    assert second == first
    assert (report_dir / "artifact.json").is_file()
    assert (report_dir / "report.html").is_file()
