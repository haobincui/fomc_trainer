from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from jobs.eval import render_chk3_stochastic_bootstrap_report as subject
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


def _binding(label: str) -> dict:
    return {
        "path": f"/sealed/{label}",
        "sha256": (label.encode().hex() + "0" * 64)[:64],
        "bytes": 100,
    }


def _view(view_id: str) -> dict:
    denominators = {
        "full": (190, 171, 13),
        "row_disjoint": (178, 160, 13),
        "strict_meeting_disjoint": (59, 52, 4),
    }
    generation_n, semantic_n, meetings = denominators[view_id]
    estimates = {}
    for model_index, model_id in enumerate(subject.MODEL_ORDER):
        estimates[model_id] = {}
        for metric_index, metric in enumerate(subject.METRIC_ORDER):
            point = 0.8 + model_index * 0.02 + metric_index * 0.001
            observations = (
                semantic_n * 5 if metric in ("mpnet_cosine", "bertscore_f1") else generation_n * 5
            )
            estimates[model_id][metric] = {
                "point_estimate": point,
                "ci_lower": point - 0.02,
                "ci_upper": point + 0.02,
                "zero_or_fail_generation_count": model_index,
                "observations": observations,
            }
    contrasts = []
    for contrast_id, before, after, family in (
        ("chk3_minus_chk1", "chk1", "chk3", "primary"),
        ("chk1_minus_chk0", "chk0", "chk1", "background"),
        ("chk3_minus_chk0", "chk0", "chk3", "background"),
    ):
        difference = {"chk3_minus_chk1": 0.02, "chk1_minus_chk0": 0.02, "chk3_minus_chk0": 0.04}[contrast_id]
        for metric in subject.METRIC_ORDER:
            contrasts.append(
                {
                    "contrast_id": contrast_id,
                    "before_model": before,
                    "after_model": after,
                    "family": family,
                    "metric": metric,
                    "point_difference": difference,
                    "ci_lower": difference - 0.01,
                    "ci_upper": difference + 0.01,
                    "p_value": 0.01,
                    "holm_adjusted_p": 0.04,
                    "sign_flip_combinations": 16 if meetings == 4 else 8192,
                }
            )
    return {
        "view": {
            "view_id": view_id,
            "generation_prompts": generation_n,
            "semantic_prompts": semantic_n,
            "meetings": meetings,
            "inferential_conclusion_authorized": view_id == "row_disjoint",
        },
        "estimates": estimates,
        "contrasts": contrasts,
        "interpretation": {
            "verdict": "semantic_increment_under_random_decoding",
            "hard_gate_regressions": [],
            "semantic_increment_supported": True,
            "conclusion_eligible": view_id == "row_disjoint",
        },
        "bootstrap_index_plan": {"sha256": (view_id.encode().hex() + "0" * 64)[:64]},
    }


def _results() -> dict:
    return {
        "bootstrap_contract": {
            "draws": 2000,
            "seed": 20260812,
            "meeting_equal_weighting": True,
        },
        "views": {view_id: _view(view_id) for view_id in subject.VIEW_ORDER},
        "limitations": [
            "This is post-selection robustness: cp318 used N12 for selection.",
            "Bootstrap cannot remove checkpoint-selection bias.",
        ],
    }


def _greedy() -> dict:
    return {
        "summaries": {
            model_id: {
                "cohorts": {
                    "all_n12": {
                        "six_metrics": {
                            metric: {"score": 0.7 + model_index * 0.05}
                            for metric in subject.METRIC_ORDER
                        }
                    }
                }
            }
            for model_index, model_id in enumerate(subject.MODEL_ORDER)
        }
    }


def _failure() -> dict:
    return {
        "schema_version": subject.FAILURE_SCHEMA_VERSION,
        "model_id": "chk3",
        "sample_id": "sample-001",
        "meeting_id": "2024-01-31",
        "replicate_id": 2,
        "failed_generation_metrics": ["numeric_fidelity"],
        "preregistered_core_failures": ["numeric_multiset_not_preserved"],
        "tuple_key": {
            "model_id": "chk3",
            "sample_id": "sample-001",
            "replicate_id": 2,
        },
    }


def test_html_contains_three_views_anchor_tests_failures_and_post_selection_caveat() -> None:
    bindings = {
        "score_manifest": _binding("score"),
        "bootstrap_results": _binding("results"),
        "failure_samples": _binding("failures"),
        "bootstrap_draws": _binding("draws"),
        "greedy_n12_anchor": _binding("greedy"),
    }
    report = subject.render_report_html(
        score={},
        results=_results(),
        greedy=_greedy(),
        failures=[_failure()],
        bindings=bindings,
        title="Synthetic bootstrap report",
    )

    assert report.startswith("<!doctype html>")
    assert report.count('class="forest"') == 3
    assert 'id="full"' in report
    assert 'id="row_disjoint"' in report
    assert 'id="strict_meeting_disjoint"' in report
    assert "Greedy N12 指标" in report
    assert "Holm p" in report
    assert report.count("Bootstrap index plan SHA-256") == 3
    assert "8192" in report or "8,192" in report
    assert "numeric_multiset_not_preserved" in report
    assert "post-selection" in report
    assert "不能消除 cp318 的选点偏差" in report
    assert "http://" not in report
    assert "https://" not in report


def test_formal_matrix_and_draw_validators_require_exact_2850_and_6000() -> None:
    metrics = {metric: 1.0 for metric in sorted(subject.METRIC_ORDER)}
    rows = [
        {
            "schema_version": subject.ROW_SCHEMA_VERSION,
            "model_id": model_id,
            "sample_id": f"sample-{sample_index:03d}",
            "replicate_id": replicate_id,
            "six_metrics": metrics,
        }
        for model_id in subject.MODEL_ORDER
        for sample_index in range(190)
        for replicate_id in range(5)
    ]
    subject._validate_row_scores(rows)
    duplicate = copy.deepcopy(rows)
    duplicate[-1]["sample_id"] = duplicate[-2]["sample_id"]
    duplicate[-1]["replicate_id"] = duplicate[-2]["replicate_id"]
    with pytest.raises(subject.BootstrapReportError, match="duplicate"):
        subject._validate_row_scores(duplicate)

    draws = [
        {
            "schema_version": subject.DRAW_SCHEMA_VERSION,
            "view_id": view_id,
            "draw_id": draw_id,
            "model_metrics": {},
        }
        for view_id in subject.VIEW_ORDER
        for draw_id in range(2000)
    ]
    subject._validate_draws(draws)
    with pytest.raises(subject.BootstrapReportError, match="6000"):
        subject._validate_draws(draws[:-1])


def test_report_publication_is_self_contained_and_sealed(
    monkeypatch, tmp_path: Path
) -> None:
    score_path = tmp_path / "score.json"
    score_path.write_text("{}\n", encoding="utf-8")
    score_binding = subject._file_binding(score_path)
    artifact_bindings = {
        "bootstrap_results": _binding("results"),
        "row_scores": _binding("rows"),
        "failure_samples": _binding("failures"),
        "bootstrap_draws": _binding("draws"),
    }
    score = {
        "artifacts": artifact_bindings,
        "inputs": {
            "generation_suite_manifest": _binding("suite"),
            "sample_manifest": _binding("n190"),
            "selection_n12_sample_manifest": _binding("n12"),
            "semantic_model_manifest": _binding("semantic"),
        },
    }
    bundle = {
        "score": score,
        "score_binding": score_binding,
        "results": _results(),
        "greedy": _greedy(),
        "greedy_binding": _binding("greedy"),
        "failures": [_failure()],
    }
    monkeypatch.setattr(subject, "load_and_validate_bundle", lambda *_args: bundle)
    output = tmp_path / "report"

    manifest = subject.render_and_seal_report(
        score_manifest=score_path,
        score_manifest_sha256=score_binding["sha256"],
        output_dir=output,
        title="Synthetic sealed report",
    )

    assert validate_manifest_integrity(manifest) == manifest["integrity"][
        "payload_sha256"
    ]
    report_path = output / "report.html"
    assert report_path.is_file()
    assert manifest["report"]["sha256"] == subject._sha256_file(report_path)
    assert manifest["report"]["self_contained"] is True
    assert (
        manifest["report_contract"]["structural_validation"][
            "external_network_references"
        ]
        == 0
    )
    assert manifest["report_contract"]["greedy_anchor_displayed_separately"] is True
    assert (output / "manifest.json").is_file()
    with pytest.raises(subject.BootstrapReportError, match="reuse"):
        subject.render_and_seal_report(
            score_manifest=score_path,
            score_manifest_sha256=score_binding["sha256"],
            output_dir=output,
            title="Duplicate",
        )


def test_formal_bundle_loader_deep_validates_all_report_inputs(tmp_path: Path) -> None:
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    metrics = {metric: 1.0 for metric in subject.METRIC_ORDER}
    row_scores = [
        {
            "schema_version": subject.ROW_SCHEMA_VERSION,
            "model_id": model_id,
            "sample_id": f"sample-{sample_index:03d}",
            "replicate_id": replicate_id,
            "six_metrics": metrics,
        }
        for model_id in subject.MODEL_ORDER
        for sample_index in range(190)
        for replicate_id in range(5)
    ]
    draws = [
        {
            "schema_version": subject.DRAW_SCHEMA_VERSION,
            "view_id": view_id,
            "draw_id": draw_id,
            "model_metrics": {
                model_id: metrics for model_id in subject.MODEL_ORDER
            },
        }
        for view_id in subject.VIEW_ORDER
        for draw_id in range(2000)
    ]
    failures = [_failure()]

    def write_jsonl(name: str, rows: list[dict]) -> Path:
        path = score_dir / name
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    row_path = write_jsonl("row_scores.jsonl", row_scores)
    draw_path = write_jsonl("bootstrap_draws.jsonl", draws)
    failure_path = write_jsonl("failure_samples.jsonl", failures)
    result_payload = seal_manifest(
        {
            "schema_version": subject.RESULT_SCHEMA_VERSION,
            "status": "complete",
            "model_order": list(subject.MODEL_ORDER),
            "metric_order": list(subject.METRIC_ORDER),
            "view_order": list(subject.VIEW_ORDER),
            **_results(),
        }
    )
    result_path = score_dir / "bootstrap_results.json"
    result_path.write_text(json.dumps(result_payload, sort_keys=True), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    greedy_path = (
        root
        / "output/evaluation/main/chk3_checkpoint_selection_cp318_20260811_v1/six_metrics/chk0_chk1_chk3_cp318_n12_cpu_v1.json"
    )
    artifacts = {
        "row_scores": {**subject._file_binding(row_path), "rows": 2850},
        "failure_samples": {**subject._file_binding(failure_path), "rows": 1},
        "bootstrap_draws": {**subject._file_binding(draw_path), "rows": 6000},
        "bootstrap_results": subject._file_binding(result_path, sealed=True),
    }
    score_payload = seal_manifest(
        {
            "schema_version": subject.SCORE_MANIFEST_SCHEMA_VERSION,
            "status": "complete",
            "model_order": list(subject.MODEL_ORDER),
            "metric_order": list(subject.METRIC_ORDER),
            "coverage": {
                "generation_rows": 2850,
                "paired_tuple_coverage_identical": True,
                "input_truncation_rows": 0,
                "hard_gate_failure_rows": 1,
            },
            "scoring_contract": {
                "weighted_composite_calculated": False,
                "bootstrap_results_payload_sha256": result_payload["integrity"][
                    "payload_sha256"
                ],
                "bootstrap_draws_sha256": artifacts["bootstrap_draws"]["sha256"],
            },
            "artifacts": artifacts,
            "inputs": {
                "greedy_n12_six_metric_anchor": subject._file_binding(
                    greedy_path, sealed=True
                )
            },
        }
    )
    score_path = score_dir / "manifest.json"
    score_path.write_text(json.dumps(score_payload, sort_keys=True), encoding="utf-8")

    bundle = subject.load_and_validate_bundle(
        score_path, subject._sha256_file(score_path)
    )

    assert len(bundle["row_scores"]) == 2850
    assert len(bundle["draws"]) == 6000
    assert bundle["results"]["views"]["row_disjoint"]["view"][
        "generation_prompts"
    ] == 178
    assert bundle["greedy_binding"]["payload_sha256"] == (
        subject.EXPECTED_GREEDY_PAYLOAD_SHA256
    )
