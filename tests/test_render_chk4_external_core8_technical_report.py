from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobs.retrain_v2 import render_chk4_external_core8_technical_report as subject


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _matrix(
    *,
    cut: tuple[int, int, int, int],
    hold: tuple[int, int, int, int],
    hike: tuple[int, int, int, int],
) -> dict:
    labels = subject.PREDICTIONS
    return {
        "cut": dict(zip(labels, cut, strict=True)),
        "hold": dict(zip(labels, hold, strict=True)),
        "hike": dict(zip(labels, hike, strict=True)),
    }


def _score(matrix: dict, *, supported_only: bool) -> dict:
    classes = {}
    for direction in subject.DIRECTIONS:
        support = sum(matrix[direction].values())
        correct = matrix[direction][direction]
        classes[direction] = {
            "support": support,
            "correct": correct,
            "recall": correct / support if support else None,
        }
    supported = [
        direction
        for direction in subject.DIRECTIONS
        if classes[direction]["support"]
    ]
    supported_ba = sum(classes[direction]["recall"] for direction in supported) / len(
        supported
    )
    mechanical_ba = sum(
        classes[direction]["recall"] or 0.0 for direction in subject.DIRECTIONS
    ) / 3
    fixed_ba = mechanical_ba if len(supported) == len(subject.DIRECTIONS) else None
    correct = sum(matrix[direction][direction] for direction in subject.DIRECTIONS)
    cases = sum(classes[direction]["support"] for direction in subject.DIRECTIONS)
    return {
        "cases": cases,
        "direction_correct": correct,
        "direction_accuracy": correct / cases,
        "per_class": classes,
        "confusion_matrix": matrix,
        "supported_classes": supported,
        "supported_class_balanced_accuracy": supported_ba,
        "fixed_three_class_balanced_accuracy": fixed_ba,
        "mechanical_zero_insertion_balanced_accuracy": mechanical_ba,
        "mechanical_zero_insertion_is_estimand": fixed_ba is not None,
        "primary_balanced_accuracy_definition": (
            "supported_classes" if supported_only else "fixed_three_class"
        ),
        "primary_balanced_accuracy": supported_ba if supported_only else fixed_ba,
    }


def _pair(
    model_a: str,
    model_b: str,
    *,
    a_only: int,
    b_only: int,
    both: int,
    neither: int,
) -> dict:
    return {
        "model_a": model_a,
        "model_a_display": subject.MODEL_DISPLAY[model_a],
        "model_b": model_b,
        "model_b_display": subject.MODEL_DISPLAY[model_b],
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "both_correct": both,
        "neither_correct": neither,
        "discordant_pairs": a_only + b_only,
        "test": "exact_two_sided_mcnemar",
        "p_value_raw": subject._exact_mcnemar_p(a_only, b_only),
    }


def _holm_family(rows: list[dict], *, panel: str) -> list[dict]:
    adjusted = subject._holm_adjusted([row["p_value_raw"] for row in rows])
    return [
        {
            **row,
            "p_value_holm": adjusted[index],
            "holm_family": subject.HOLM_FAMILY,
            "holm_family_panel": panel,
            "holm_family_size": subject.HOLM_FAMILY_SIZE,
        }
        for index, row in enumerate(rows)
    ]


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    evaluation_root = tmp_path / "output/evaluation/external"
    results_path = evaluation_root / "run/results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("{}\n" * subject.EXPECTED_RESULT_ROWS, encoding="utf-8")

    topic_order = [
        "CPI",
        "GDP Growth",
        "Government Purchases",
        "Housing Starts",
        "Industrial Production",
        "Labour Market",
        "Money Supply",
        "Unemployment Rate",
    ]
    input_contract = {
        "name": "Deterministic Core8 Source-Analysis Decision Diagnostic",
        "version": subject.INPUT_CONTRACT_VERSION,
        "topic_order": topic_order,
        "blocks_per_meeting": 8,
        "separator": "two_newlines",
        "information_cutoff": "official_meeting_start_date_minus_one_calendar_day",
        "meeting_identity_in_prompt": False,
        "gold_in_prompt": False,
        "official_text_in_prompt": False,
        "distinct_from_teacher_compressed_n13": True,
        "byte_comparable_to_teacher_compressed_n13": False,
    }
    manifest = subject._sealed(
        {
            "schema_version": subject.EVALUATION_MANIFEST_SCHEMA,
            "status": "prepared",
            "purpose": "separate_panel_greedy_decision_evaluation_without_retraining",
            "population": {
                "panels": subject.PANEL_N,
                "class_counts": subject.PANEL_CLASS_COUNTS,
                "direct_decision_release_overlap": 0,
                "training_performed": False,
            },
            "input_contract": input_contract,
            "generation": {
                "mode": "greedy",
                "completions_per_meeting_model": 1,
                "expected_rows": subject.EXPECTED_RESULT_ROWS,
            },
            "reporting": {
                "panels_reported_separately": True,
                "pairwise_test": "exact_two_sided_mcnemar_direction_correctness",
                "pairwise_multiplicity": (
                    "Holm adjustment within each panel's family of three "
                    "model-pair comparisons"
                ),
                "pooled_result_prohibited": True,
            },
            "models": [
                {
                    "label": model,
                    "display_name": subject.MODEL_DISPLAY[model],
                    "training_state": subject.MODEL_TRAINING_STATE[model],
                }
                for model in subject.MODEL_ORDER
            ],
        }
    )
    manifest_path = evaluation_root / "evaluation_manifest.json"
    _write_json(manifest_path, manifest)

    historical_models = {
        "model_chk1_cp200": _score(
            _matrix(
                cut=(2, 2, 0, 0),
                hold=(1, 8, 1, 0),
                hike=(0, 1, 4, 0),
            ),
            supported_only=False,
        ),
        "model_chk3_sft_cp38": _score(
            _matrix(
                cut=(3, 1, 0, 0),
                hold=(2, 7, 1, 0),
                hike=(0, 2, 3, 0),
            ),
            supported_only=False,
        ),
        "model_chk3_grpo_cp450": _score(
            _matrix(
                cut=(2, 2, 0, 0),
                hold=(0, 9, 1, 0),
                hike=(1, 2, 2, 0),
            ),
            supported_only=False,
        ),
    }
    post_models = {
        "model_chk1_cp200": _score(
            _matrix(
                cut=(1, 2, 0, 0),
                hold=(2, 7, 0, 0),
                hike=(0, 0, 0, 0),
            ),
            supported_only=True,
        ),
        "model_chk3_sft_cp38": _score(
            _matrix(
                cut=(2, 1, 0, 0),
                hold=(2, 6, 1, 0),
                hike=(0, 0, 0, 0),
            ),
            supported_only=True,
        ),
        "model_chk3_grpo_cp450": _score(
            _matrix(
                cut=(1, 2, 0, 0),
                hold=(1, 8, 0, 0),
                hike=(0, 0, 0, 0),
            ),
            supported_only=True,
        ),
    }
    a, b, c = subject.MODEL_ORDER
    summary = subject._sealed(
        {
            "schema_version": subject.SUMMARY_SCHEMA,
            "status": "complete",
            "created_at_utc": "2026-08-24T12:00:00Z",
            "evaluation_manifest_sha256": subject._sha256_file(manifest_path),
            "results": {
                "path": "run/results.jsonl",
                "bytes": results_path.stat().st_size,
                "sha256": subject._sha256_file(results_path),
                "rows": subject.EXPECTED_RESULT_ROWS,
            },
            "panels": {
                "historical_n19": {
                    "population": {
                        "meetings": 19,
                        "class_counts": {"cut": 4, "hold": 10, "hike": 5},
                    },
                    "models": historical_models,
                    "pairwise": _holm_family(
                        [
                            _pair(a, b, a_only=4, b_only=3, both=10, neither=2),
                            _pair(a, c, a_only=3, b_only=2, both=11, neither=3),
                            _pair(b, c, a_only=3, b_only=3, both=10, neither=3),
                        ],
                        panel="historical_n19",
                    ),
                },
                "postcutoff_n12": {
                    "population": {
                        "meetings": 12,
                        "class_counts": {"cut": 3, "hold": 9},
                    },
                    "models": post_models,
                    "pairwise": _holm_family(
                        [
                            _pair(a, b, a_only=3, b_only=3, both=5, neither=1),
                            _pair(a, c, a_only=2, b_only=3, both=6, neither=1),
                            _pair(b, c, a_only=1, b_only=2, both=7, neither=2),
                        ],
                        panel="postcutoff_n12",
                    ),
                },
            },
            "pooled_result": None,
            "interpretation": {
                "input_contract": input_contract,
                "historical_n19": "historical external diagnostic",
                "postcutoff_n12": "retrospective diagnostic; no hike support",
                "teacher_compressed_n13": "separate opened diagnostic; not pooled",
            },
        }
    )
    summary_path = evaluation_root / "report/summary.json"
    _write_json(summary_path, summary)

    selection_path = tmp_path / "sources/selection.json"
    _write_json(
        selection_path,
        {
            "selected_checkpoint_step": None,
            "best_observed_checkpoint_step": 450,
            "status": "provisional_no_candidate_passed_all_gates",
        },
    )
    final_status_path = tmp_path / "sources/FINAL_STATUS.md"
    final_status_path.write_text(
        "Verdict: `quality_not_demonstrated`; checkpoint 450 was not automatically gate-selected.\n",
        encoding="utf-8",
    )
    return {
        "summary": summary_path,
        "manifest": manifest_path,
        "results": results_path,
        "selection": selection_path,
        "final_status": final_status_path,
    }


def test_builds_separate_table_first_report_and_portable_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    model = subject.load_report_model(
        paths["summary"],
        selection_path=paths["selection"],
        final_status_path=paths["final_status"],
    )

    markdown = subject.build_markdown_report(model)
    assert "Historical N=19" in markdown
    assert "Post-cutoff N=12" in markdown
    assert "Fixed-three-class balanced accuracy is **not estimable**" in markdown
    assert "selected_checkpoint_step = null" in markdown
    assert "quality_not_demonstrated" in markdown
    assert "performed no retraining" in markdown
    assert "teacher-compressed meeting brief" in markdown
    assert "retrospective, not prospectively sealed" in markdown
    assert "performance evidence remains table-first" in markdown
    assert "only one minimal grouped bar chart" in markdown
    assert "No N=31" in markdown
    assert "Exact McNemar p (raw)" in markdown
    assert "Holm-adjusted p" in markdown
    assert "Paper-facing inference uses the Holm-adjusted values" in markdown
    assert "no p-values are pooled or adjusted across panels" in markdown

    artifact = subject.build_canonical_artifact(model)
    assert artifact["surface"] == "report"
    assert artifact["manifest"]["title"] == subject.REPORT_TITLE
    assert len(artifact["manifest"]["charts"]) == 1
    chart = artifact["manifest"]["charts"][0]
    assert chart["id"] == "class_support_chart"
    assert chart["settings"]["groupMode"] == "grouped"
    assert chart["encodings"]["color"]["field"] == "panel"
    assert chart["referenceLines"][0]["value"] == 0
    assert chart["labels"]["values"] == "all"
    assert len(artifact["snapshot"]["datasets"]["class_support_chart"]) == 6
    post_hike = next(
        row
        for row in artifact["snapshot"]["datasets"]["class_support_chart"]
        if row["panel_id"] == "postcutoff_n12" and row["direction"] == "Hike"
    )
    assert post_hike["meeting_count"] == 0
    assert post_hike["panel_denominator"] == 12
    assert len(artifact["snapshot"]["datasets"]["historical_metrics"]) == 3
    assert len(artifact["snapshot"]["datasets"]["postcutoff_metrics"]) == 3
    assert len(artifact["snapshot"]["datasets"]["historical_mcnemar"]) == 3
    assert len(artifact["snapshot"]["datasets"]["postcutoff_mcnemar"]) == 3
    for panel, dataset in (
        ("historical_n19", "historical_mcnemar"),
        ("postcutoff_n12", "postcutoff_mcnemar"),
    ):
        for row in artifact["snapshot"]["datasets"][dataset]:
            assert row["holm_family"] == subject.HOLM_FAMILY
            assert row["holm_family_panel"] == panel
            assert row["holm_family_size"] == 3
            assert row["p_value_holm"] >= row["p_value_raw"]
            assert row["p_value_raw_display"]
            assert row["p_value_holm_display"]
    assert all(
        row["fixed_three_class_balanced_accuracy_display"]
        == "Not estimable (no hike support)"
        for row in artifact["snapshot"]["datasets"]["postcutoff_metrics"]
    )
    assert artifact["manifest"]["blocks"][0] == {
        "id": "title",
        "type": "markdown",
        "body": f"# {subject.REPORT_TITLE}",
    }
    assert all(not Path(source["path"]).is_absolute() for source in artifact["sources"])
    summary_source = next(
        source
        for source in artifact["sources"]
        if source["id"] == "external_summary_source"
    )
    assert summary_source["query"]["sql"].startswith(
        "SELECT * FROM read_json_auto("
    )


def test_render_is_atomic_create_only_and_seals_final_output_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "reports/external_decision"
    receipt = subject.render_report(
        paths["summary"],
        output,
        selection_path=paths["selection"],
        final_status_path=paths["final_status"],
    )

    assert receipt["status"] == "complete_unpacked"
    assert receipt["html_packaged"] is False
    assert (output / "technical_report.md").is_file()
    assert (output / "artifact.json").is_file()
    report_manifest = json.loads((output / "report_manifest.json").read_text())
    subject._verify_sealed_payload(
        report_manifest,
        schema=subject.REPORT_MANIFEST_SCHEMA,
        label="report manifest",
    )
    assert report_manifest["outputs"]["technical_report"]["path"] == str(
        (output / "technical_report.md").resolve()
    )
    assert report_manifest["outputs"]["artifact"]["path"] == str(
        (output / "artifact.json").resolve()
    )
    assert report_manifest["renderer_provenance"]["implementation"]["sha256"] == (
        subject._sha256_file(Path(subject.__file__))
    )
    assert report_manifest["renderer_provenance"]["focused_test"]["sha256"] == (
        subject._sha256_file(Path(__file__))
    )
    assert report_manifest["native_chart_count"] == 1
    assert report_manifest["performance_chart_omission_reason"]
    assert report_manifest["chart_map"][0]["type"] == "grouped_bar"
    assert "direct panel-and-count labels" in report_manifest["chart_map"][0][
        "non_color_plan"
    ]
    assert report_manifest["portable_render_qa"] == (
        "deferred_until_real_summary_exists"
    )
    with pytest.raises(subject.ExternalDecisionReportError, match="output already exists"):
        subject.render_report(
            paths["summary"],
            output,
            selection_path=paths["selection"],
            final_status_path=paths["final_status"],
        )


def test_rejects_tampered_summary_missing_hike_support_and_selected_cp450(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    raw = json.loads(paths["summary"].read_text())
    raw["panels"]["postcutoff_n12"]["models"]["model_chk1_cp200"]["per_class"]["hike"]["support"] = 1
    tampered = tmp_path / "tampered.json"
    _write_json(tampered, raw)
    with pytest.raises(subject.ExternalDecisionReportError, match="payload SHA-256 drift"):
        subject._validate_summary(subject._read_json(tampered, label="tampered"))

    selected = json.loads(paths["selection"].read_text())
    selected["selected_checkpoint_step"] = 450
    _write_json(paths["selection"], selected)
    with pytest.raises(subject.ExternalDecisionReportError, match="unexpectedly selected"):
        subject.load_report_model(
            paths["summary"],
            selection_path=paths["selection"],
            final_status_path=paths["final_status"],
        )


def test_rejects_resealed_nonzero_postcutoff_hike_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    raw = json.loads(paths["summary"].read_text())
    raw.pop("integrity")
    block = raw["panels"]["postcutoff_n12"]["models"]["model_chk1_cp200"]
    block["per_class"]["hike"] = {"support": 1, "correct": 0, "recall": 0.0}
    block["confusion_matrix"]["hike"]["hold"] = 1
    raw["panels"]["postcutoff_n12"]["population"]["class_counts"]["hike"] = 1
    resealed = subject._sealed(raw)
    with pytest.raises(subject.ExternalDecisionReportError, match="class distribution drift"):
        subject._validate_summary(resealed)


def test_rejects_resealed_incorrect_within_panel_holm_adjustment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    raw = json.loads(paths["summary"].read_text())
    raw.pop("integrity")
    comparison = raw["panels"]["historical_n19"]["pairwise"][0]
    comparison["p_value_holm"] = 0.5
    resealed = subject._sealed(raw)
    with pytest.raises(
        subject.ExternalDecisionReportError,
        match="Holm-adjusted McNemar p",
    ):
        subject._validate_summary(resealed)


def test_holm_adjustment_is_step_down_and_preserves_original_order() -> None:
    assert subject._holm_adjusted([0.20, 0.01, 0.04]) == pytest.approx(
        [0.20, 0.03, 0.08]
    )
