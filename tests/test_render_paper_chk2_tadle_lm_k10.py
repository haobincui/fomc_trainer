from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from jobs.eval import render_paper_chk2_tadle_lm_k10 as report


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _absolute_binding(path: Path, *, rows: int | None = None) -> dict:
    value = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": report.sha256_file(path),
    }
    if rows is not None:
        value["rows"] = rows
    return value


def _relative_binding(root: Path, path: Path, *, rows: int | None = None) -> dict:
    return report.binding(path, relative_to=root, rows=rows)


def _fixture(root: Path) -> None:
    points, contrasts, distances, gains = [], [], [], []
    for h_index, horizon in enumerate(report.HORIZONS, 1):
        official = -0.1 * h_index
        for a_index, arm in enumerate(report.ARMS):
            post = official + 0.01 * a_index
            points.append(
                {
                    "schema_version": f"{report.STATISTICS_SCHEMA}:coefficient-row",
                    "backend_id": report.BACKEND_ID,
                    "convention": report.CONVENTION,
                    "specification": report.SPECIFICATION,
                    "horizon_label": horizon,
                    "horizon_months": int(horizon[2:]),
                    "arm": arm,
                    "paper_label": report.LABELS[arm],
                    "nobs": report.EXPECTED_NOBS[horizon],
                    "beta_pre_basis_points": post - 0.02,
                    "hc1_se_pre_basis_points": 0.01,
                    "beta_interaction_basis_points": 0.02,
                    "hc1_se_interaction_basis_points": 0.01,
                    "beta_post_basis_points": post,
                    "hc1_se_post_basis_points": 0.02,
                    "r_squared": 0.1,
                    "condition_number": 2.0,
                }
            )
        for estimand in ("pre", "interaction", "post"):
            family = (
                "derived_post_marginal_12"
                if estimand == "post"
                else "primitive_pre_and_interaction_24"
            )
            for arm in report.GENERATED_ARMS:
                contrasts.append(
                    {
                        "schema_version": f"{report.STATISTICS_SCHEMA}:model-minus-official-row",
                        "backend_id": report.BACKEND_ID,
                        "convention": report.CONVENTION,
                        "specification": report.SPECIFICATION,
                        "family": family,
                        "estimand": estimand,
                        "horizon_label": horizon,
                        "horizon_months": int(horizon[2:]),
                        "arm": arm,
                        "paper_label": report.LABELS[arm],
                        "estimate_basis_points": 0.01,
                        "ci_95_low_basis_points": -0.02,
                        "ci_95_high_basis_points": 0.04,
                        "bootstrap_p_raw": 0.2,
                        "holm_p": 0.5,
                        "bootstrap_draws": 10_000,
                    }
                )
        for a_index, arm in enumerate(report.GENERATED_ARMS, 1):
            distances.append(
                {
                    "schema_version": f"{report.STATISTICS_SCHEMA}:absolute-distance-row",
                    "backend_id": report.BACKEND_ID,
                    "convention": report.CONVENTION,
                    "specification": report.SPECIFICATION,
                    "estimand": "post",
                    "horizon_label": horizon,
                    "horizon_months": int(horizon[2:]),
                    "arm": arm,
                    "paper_label": report.LABELS[arm],
                    "estimate_basis_points": 0.01 * a_index,
                    "ci_95_low_basis_points": 0.0,
                    "ci_95_high_basis_points": 0.05,
                    "bootstrap_draws": 10_000,
                }
            )
        for comparator in ("chk0", "chk1"):
            gains.append(
                {
                    "schema_version": f"{report.STATISTICS_SCHEMA}:distance-gain-row",
                    "backend_id": report.BACKEND_ID,
                    "convention": report.CONVENTION,
                    "specification": report.SPECIFICATION,
                    "family": "chk2_distance_gains_8",
                    "estimand": "post",
                    "horizon_label": horizon,
                    "horizon_months": int(horizon[2:]),
                    "comparator_arm": comparator,
                    "comparator_label": report.LABELS[comparator],
                    "target_arm": "paper_chk2_cp50",
                    "target_label": report.LABELS["paper_chk2_cp50"],
                    "positive_means_chk2_closer": True,
                    "estimate_basis_points": -0.02,
                    "ci_95_low_basis_points": -0.05,
                    "ci_95_high_basis_points": 0.01,
                    "bootstrap_p_raw": 0.3,
                    "holm_p": 0.8,
                    "bootstrap_draws": 10_000,
                }
            )

    estimation = root / "estimation"
    data = {
        "coefficient_estimates": (estimation / "coefficient_estimates.jsonl", points),
        "model_minus_official_contrasts": (
            estimation / "model_minus_official_contrasts.jsonl",
            contrasts,
        ),
        "absolute_distances": (estimation / "absolute_distances.jsonl", distances),
        "distance_gains": (estimation / "distance_gains.jsonl", gains),
    }
    for path, rows in data.values():
        _write_jsonl(path, rows)

    train = [f"train-{index:02d}" for index in range(47)]
    empty = {"union_84": [], "standardization_83": [], "estimable_current_82": []}
    overlap = {
        "schema_version": report.OVERLAP_SCHEMA,
        "status": "complete",
        "overlap_by_split": {
            "train": {
                "union_84": train,
                "standardization_83": train,
                "estimable_current_82": train[:46],
            },
            "validation": empty,
            "test": empty,
        },
        "counts": {
            "train": {
                "union_84": 47,
                "standardization_83": 47,
                "estimable_current_82": 46,
            },
            "validation": {
                "union_84": 0,
                "standardization_83": 0,
                "estimable_current_82": 0,
            },
            "test": {"union_84": 0, "standardization_83": 0, "estimable_current_82": 0},
        },
    }
    overlap_path = root / "preparation/training_overlap_audit.json"
    _write_json(overlap_path, overlap)
    ledger_path = root / "preparation/atomic_prompt_ledger.jsonl"
    _write_jsonl(ledger_path, ({"row": index} for index in range(672)))
    _write_json(
        root / "preparation/evaluation_manifest.json",
        {
            "schema_version": report.PREPARATION_SCHEMA,
            "status": "prepared",
            "immutable": True,
            "evaluation_id": report.EVALUATION_ID,
            "population": {
                "meeting_union": 84,
                "estimable_current_events": 82,
                "standardization_meetings": 83,
                "topics_per_meeting": 8,
                "atomic_prompts": 672,
            },
            "generation": {
                "models": list(report.GENERATED_ARMS),
                "replicate_seeds": list(report.REPLICATE_SEEDS),
                "replicates": 10,
                "total_generation_rows": 20_160,
                "generated_documents": 2_520,
            },
            "inputs": {
                "atomic_prompt_ledger": _absolute_binding(ledger_path, rows=672),
                "training_overlap_audit": _absolute_binding(overlap_path),
            },
        },
    )

    generation_rows = root / "generation/generation_rows.jsonl"
    _write_jsonl(generation_rows, ({"row": index} for index in range(20_160)))
    models = {
        arm: {
            "schema_version": report.MODEL_GENERATION_SCHEMA,
            "status": "complete",
            "mode": "formal",
            "model_id": arm,
            "generation_rows": 6_720,
            "meetings": 84,
            "replicates": 10,
            "gates": {"complete": True},
        }
        for arm in report.GENERATED_ARMS
    }
    _write_json(
        root / "generation/validation.json",
        {
            "schema_version": report.GENERATION_SCHEMA,
            "status": "complete",
            "mode": "formal",
            "evaluation_id": report.EVALUATION_ID,
            "generation_rows": 20_160,
            "rows_per_model": 6_720,
            "paired_tuples": 6_720,
            "models": models,
            "combined_generation_rows": _absolute_binding(generation_rows, rows=20_160),
            "delivery_funnel": {"accepted": 20_160},
            "gates": {"complete": True},
            "diagnostic_failure_counts": {},
        },
    )

    post_rows = [
        {
            "arm": "official",
            "horizon_label": "FF1",
            "is_minutes_release_event": True,
            "post_2011": 1,
            "event_meeting_id": train[index],
            "lag_meeting_id": f"lag-{index:02d}",
        }
        for index in range(30)
    ]
    panel = estimation / "analysis_panel.jsonl"
    _write_jsonl(panel, [*post_rows, *({"row": index} for index in range(40_670))])
    scales = estimation / "standardization_scales.json"
    _write_json(
        scales, {"schema_version": f"{report.STATISTICS_SCHEMA}:standardization-scales"}
    )
    leave_one_event_out = estimation / "leave_one_event_out.jsonl"
    _write_jsonl(
        leave_one_event_out,
        (
            {
                "schema_version": f"{report.STATISTICS_SCHEMA}:leave-one-event-out-row",
                "delta_pre_basis_points": 0.01,
                "delta_interaction_basis_points": -0.02,
                "delta_post_basis_points": 0.03,
            }
            for _ in range(1_272)
        ),
    )
    leave_one_year_out = estimation / "leave_one_year_out.jsonl"
    _write_jsonl(
        leave_one_year_out,
        (
            {
                "schema_version": f"{report.STATISTICS_SCHEMA}:leave-one-year-out-row",
                "delta_pre_basis_points": -0.04,
                "delta_interaction_basis_points": 0.05,
                "delta_post_basis_points": -0.06,
            }
            for _ in range(172)
        ),
    )
    point_diagnostics = estimation / "point_diagnostics.json"
    _write_json(
        point_diagnostics,
        {
            "schema_version": f"{report.STATISTICS_SCHEMA}:point-diagnostics",
            "point_max_condition_number": 2.0,
            "max_abs_leave_one_event_delta_basis_points": {
                "pre": 0.01,
                "interaction": 0.02,
                "post": 0.03,
            },
            "max_abs_leave_one_year_delta_basis_points": {
                "pre": 0.04,
                "interaction": 0.05,
                "post": 0.06,
            },
        },
    )
    stability = root / "sentiment/k_stability.jsonl"
    _write_jsonl(
        stability,
        (
            {
                "schema_version": f"{report.STATISTICS_SCHEMA}:k-stability-row",
                "arm": arm,
                "k": k,
                "pearson_correlation_with_kmax": 0.9 + 0.01 * k,
                "mean_absolute_drift_from_kmax": 0.01 * (10 - k),
            }
            for arm in report.GENERATED_ARMS
            for k in range(1, 11)
        ),
    )
    documents_manifest = root / "documents/manifest.json"
    _write_json(
        documents_manifest,
        {"schema_version": "fixture-documents", "status": "complete"},
    )
    document_scores = root / "sentiment/document_scores.jsonl"
    _write_jsonl(document_scores, ({"row": index} for index in range(2_688)))
    removal_ledger = root / "sentiment/official_statement_removal_ledger.jsonl"
    _write_jsonl(removal_ledger, ({"row": index} for index in range(84)))
    dictionary_manifest = root / "sentiment/dictionary_manifest.json"
    _write_json(dictionary_manifest, {"schema_version": "fixture-dictionary"})
    sentiment_artifacts = {
        "document_scores": (document_scores, 2_688),
        "statement_removal_ledger": (removal_ledger, 84),
        "dictionary_manifest": (dictionary_manifest, None),
        "k_stability": (stability, 30),
    }
    _write_json(
        root / "sentiment/manifest.json",
        {
            "schema_version": f"{report.STATISTICS_SCHEMA}:sentiment-manifest",
            "status": "complete",
            "phase": "sentiment",
            "created_at_utc": "2026-09-02T00:00:00Z",
            "evaluation_id": report.EVALUATION_ID,
            "backend_id": report.BACKEND_ID,
            "replicates": 10,
            "atomic_generation_rows": 20_160,
            "generation_meetings": 84,
            "estimable_current_meeting_ids": 82,
            "lag_only_meeting_ids": ["lag-only-a", "lag-only-b"],
            "broader_standardization_roster": 83,
            "generated_document_rows": 2_520,
            "document_score_rows": 2_688,
            "statement_removal_rows": 84,
            "replicate_count": 10,
            "k_stability_rows": 30,
            "formal_documents_lineage_verified": True,
            "unsealed_custom_generated_documents": False,
            "inputs": {"documents_manifest": _absolute_binding(documents_manifest)},
            "artifacts": {
                name: _relative_binding(root, path, rows=rows)
                for name, (path, rows) in sentiment_artifacts.items()
            },
        },
    )
    bootstrap = estimation / "bootstrap_draws.jsonl"
    _write_jsonl(bootstrap, ({"draw_index": index} for index in range(10_000)))
    plan = estimation / "bootstrap_plan.json"
    _write_json(
        plan,
        {
            "schema_version": f"{report.STATISTICS_SCHEMA}:bootstrap-plan",
            "draws": 10_000,
            "replicate_count": 10,
            "plan_sha256": "plan",
        },
    )
    arrays = estimation / "bootstrap_plan_arrays.npz"
    arrays.write_bytes(b"fixture-npz")
    validation = estimation / "validation_report.json"
    _write_json(
        validation,
        {
            "schema_version": f"{report.STATISTICS_SCHEMA}:validation-report",
            "status": "passed",
            "checks": {"complete": True},
        },
    )
    artifacts = {
        "bootstrap_plan": (plan, None),
        "bootstrap_plan_arrays": (arrays, None),
        "bootstrap_draws": (bootstrap, 10_000),
        "model_minus_official_contrasts": (
            data["model_minus_official_contrasts"][0],
            36,
        ),
        "absolute_distances": (data["absolute_distances"][0], 12),
        "distance_gains": (data["distance_gains"][0], 8),
        "validation_report": (validation, None),
    }
    estimate_artifacts = {
        "standardization_scales": (scales, None),
        "analysis_panel": (panel, 40_700),
        "coefficient_estimates": (data["coefficient_estimates"][0], 16),
        "leave_one_event_out": (leave_one_event_out, 1_272),
        "leave_one_year_out": (leave_one_year_out, 172),
        "point_diagnostics": (point_diagnostics, None),
    }
    estimate_receipt = estimation / "estimate_receipt.json"
    _write_json(
        estimate_receipt,
        {
            "schema_version": f"{report.STATISTICS_SCHEMA}:estimate-manifest",
            "status": "complete",
            "created_at_utc": "2026-09-02T00:00:00Z",
            "evaluation_id": report.EVALUATION_ID,
            "backend_id": report.BACKEND_ID,
            "convention": report.CONVENTION,
            "specification": report.SPECIFICATION,
            "replicates": 10,
            "atomic_generation_rows": 20_160,
            "generated_documents": 2_520,
            "generation_meetings": 84,
            "estimable_current_meeting_ids": 82,
            "lag_only_meeting_ids": ["lag-only-a", "lag-only-b"],
            "broader_standardization_roster": 83,
            "analysis_panel_rows": 40_700,
            "coefficient_rows": 16,
            "leave_one_event_out_rows": 1_272,
            "leave_one_year_out_rows": 172,
            "inputs": {},
            "artifacts": {
                name: _relative_binding(root, path, rows=rows)
                for name, (path, rows) in estimate_artifacts.items()
            },
        },
    )
    _write_json(
        estimation / "statistics_manifest.json",
        {
            "schema_version": f"{report.STATISTICS_SCHEMA}:statistics-manifest",
            "status": "complete",
            "created_at_utc": "2026-09-02T00:00:00Z",
            "evaluation_id": report.EVALUATION_ID,
            "backend_id": report.BACKEND_ID,
            "convention": report.CONVENTION,
            "specification": report.SPECIFICATION,
            "replicates": 10,
            "atomic_generation_rows": 20_160,
            "generated_documents": 2_520,
            "bootstrap_draws": 10_000,
            "bootstrap_seed": 20_260_902,
            "plan_sha256": "plan",
            "coefficient_rows": 16,
            "contrast_rows": 36,
            "absolute_distance_rows": 12,
            "distance_gain_rows": 8,
            "generation_meetings": 84,
            "estimable_current_meeting_ids": 82,
            "broader_standardization_roster": 83,
            "exact_tadle_regression_form": True,
            "exact_tadle_2022_replication": False,
            "lag_only_meeting_ids": ["lag-only-a", "lag-only-b"],
            "interpretation": "mixed-scope historical document-source association diagnostic",
            "limitations": ["Synthetic fixture; not substantive evidence."],
            "inputs": {
                "analysis_panel": _absolute_binding(panel, rows=40_700),
                "coefficient_estimates": _absolute_binding(
                    data["coefficient_estimates"][0], rows=16
                ),
                "estimate_receipt": _absolute_binding(estimate_receipt),
            },
            "artifacts": {
                name: _relative_binding(root, path, rows=rows)
                for name, (path, rows) in artifacts.items()
            },
        },
    )


def test_report_create_and_exact_resume(tmp_path: Path) -> None:
    _fixture(tmp_path)
    manifest = report.render(tmp_path)
    assert manifest["status"] == "complete"
    table = (tmp_path / "report/post_beta_table.tex").read_text(encoding="utf-8")
    assert r"\label{tab:ch2:tadle_post_beta}" in table
    assert "per one-unit constructed news shock" in table
    distance = (tmp_path / "report/distance_and_gain_table.tex").read_text(
        encoding="utf-8"
    )
    assert r"\label{tab:ch2:tadle_beta_distance}" in distance
    assert "closest generated source" in (tmp_path / "report/results.tex").read_text(
        encoding="utf-8"
    )
    markdown = (tmp_path / "report/report.md").read_text(encoding="utf-8")
    assert "Point-estimate condition numbers range" in markdown
    assert "Maximum absolute leave-one-event deltas" in markdown
    assert "K=1--10 sentiment stability" in markdown
    assert "Generation delivery funnel" in markdown
    assert "NS itself is not re-standardized" in markdown
    assert set(manifest) >= {"sentiment_manifest", "estimate_receipt"}
    artifact = json.loads(
        (tmp_path / "report/artifact.json").read_text(encoding="utf-8")
    )
    assert set(artifact["sources"]) >= {"sentiment_manifest", "estimate_receipt"}
    assert set(artifact["sources"]) >= {
        "point_diagnostics",
        "leave_one_event_out",
        "leave_one_year_out",
        "k_stability",
    }
    assert set(artifact["diagnostics"]) == {
        "condition_number_range",
        "leave_one_out_max_abs_delta_basis_points",
        "k_stability_k1_to_k10",
    }
    assert report.render(tmp_path, resume=True) == manifest


def test_report_resume_recovers_empty_directory(tmp_path: Path) -> None:
    _fixture(tmp_path)
    (tmp_path / "report").mkdir()
    manifest = report.render(tmp_path, resume=True)
    assert manifest["created_at_utc"] == "2026-09-02T00:00:00Z"
    assert report.render(tmp_path, resume=True) == manifest


def test_report_resume_recovers_exact_partial_directory(tmp_path: Path) -> None:
    _fixture(tmp_path)
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    payloads = report._build_payloads(tmp_path, created_at="2026-09-02T00:00:00Z")
    for name in ("artifact.json", "methods.tex", "results.tex"):
        report._write_new_bytes(report_dir / name, payloads[name])
    manifest = report.render(tmp_path, resume=True)
    assert set(manifest["artifacts"]) == set(payloads)


def test_report_resume_rejects_wrong_partial_directory(tmp_path: Path) -> None:
    _fixture(tmp_path)
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "results.tex").write_bytes(b"wrong")
    with pytest.raises(report.ReportError, match="partial report drift"):
        report.render(tmp_path, resume=True)


def test_report_resume_rejects_output_tamper(tmp_path: Path) -> None:
    _fixture(tmp_path)
    report.render(tmp_path)
    with (tmp_path / "report/results.tex").open("a", encoding="utf-8") as handle:
        handle.write("tamper")
    with pytest.raises(report.ReportError, match="replay drift"):
        report.render(tmp_path, resume=True)


def test_report_rejects_generation_tamper(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "generation/validation.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["status"] = "running"
    _write_json(path, value)
    with pytest.raises(
        report.ReportError, match="generation validation contract drift"
    ):
        report.render(tmp_path)


def test_report_rejects_sentiment_artifact_tamper(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "sentiment/document_scores.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"tamper":true}\n')
    with pytest.raises(report.ReportError, match="sentiment phase validation failed"):
        report.render(tmp_path)


def test_report_rejects_symlinked_json_input(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    _write_json(target, {"value": 1})
    link = tmp_path / "input.json"
    link.symlink_to(target)
    with pytest.raises(report.ReportError, match="missing/unsafe JSON input"):
        report.read_json(link)


def test_report_rejects_statistics_schema_and_nonstandard_json_numbers(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    manifest_path = tmp_path / "estimation/statistics_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "wrong"
    _write_json(manifest_path, manifest)
    with pytest.raises(report.ReportError, match="statistics manifest contract drift"):
        report.render(tmp_path)
    manifest["schema_version"] = f"{report.STATISTICS_SCHEMA}:statistics-manifest"
    points_path = tmp_path / "estimation/coefficient_estimates.jsonl"
    points = report.read_jsonl(points_path)
    points[0]["beta_pre_basis_points"] = float("nan")
    _write_jsonl(points_path, points)
    manifest["inputs"]["coefficient_estimates"] = _absolute_binding(
        points_path, rows=16
    )
    estimate_path = tmp_path / "estimation/estimate_receipt.json"
    estimate = json.loads(estimate_path.read_text(encoding="utf-8"))
    estimate["artifacts"]["coefficient_estimates"] = _relative_binding(
        tmp_path, points_path, rows=16
    )
    _write_json(estimate_path, estimate)
    manifest["inputs"]["estimate_receipt"] = _absolute_binding(estimate_path)
    _write_json(manifest_path, manifest)
    with pytest.raises(report.ReportError, match="invalid JSONL input"):
        report.render(tmp_path)


def test_concurrent_publish_never_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture(tmp_path)
    barrier = threading.Barrier(2)
    original = report._build_payloads

    def synchronized(root: Path, *, created_at: str):
        payloads = original(root, created_at=created_at)
        barrier.wait(timeout=10)
        return payloads

    monkeypatch.setattr(report, "_build_payloads", synchronized)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(report.render, tmp_path) for _ in range(2)]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except FileExistsError:
            outcomes.append("exists")
    assert sum(item == "exists" for item in outcomes) == 1
    assert sum(isinstance(item, dict) for item in outcomes) == 1
    monkeypatch.setattr(report, "_build_payloads", original)
    assert report.render(tmp_path, resume=True)["status"] == "complete"
