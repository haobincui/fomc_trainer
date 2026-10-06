from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from jobs.eval import paper_chk2_tadle_lm_k10_statistics as stats
from open_r1.validator.loo_generation_spec import seal_manifest


ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = (
    ROOT / "output/evaluation/main/"
    "tadle_interaction_wgan_dictionary_ff_futures_2004_2015_v1"
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(stats.canonical(row) + "\n" for row in rows), encoding="utf-8"
    )


def _document_rows(*, meeting: str = "2004-09-21") -> list[dict]:
    rows: list[dict] = []
    seeds = (101, 202)
    for arm in stats.GENERATED_ARMS:
        for replicate, seed in enumerate(seeds):
            text = f"{arm} meeting {meeting} replicate {replicate}."
            rows.append(
                {
                    "schema_version": "test",
                    "document_id": f"{arm}::{meeting}::{replicate}",
                    "arm": arm,
                    "meeting_id": meeting,
                    "meeting_end_date": meeting,
                    "replicate_id": replicate,
                    "replicate_seed": seed,
                    "section_count": 8,
                    "topic_order": list(stats.TOPIC_ORDER),
                    "assembly_separator": "two_newlines_exact_section_answers",
                    "document_text": text,
                    "document_text_sha256": stats.sha256_bytes(text.encode()),
                }
            )
    return rows


def test_generated_document_contract_closes_cartesian_product(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    rows = _document_rows()
    _write_jsonl(path, rows)
    source = {"needed_ids": ("2004-09-21",)}
    observed = stats.validate_generated_documents(
        path,
        source,
        replicate_count=2,
        replicate_seeds=(101, 202),
    )
    assert len(observed) == 6

    rows[-1]["topic_order"] = list(reversed(stats.TOPIC_ORDER))
    _write_jsonl(tmp_path / "tampered.jsonl", rows)
    with pytest.raises(stats.StatisticsError, match="topic-order drift"):
        stats.validate_generated_documents(
            tmp_path / "tampered.jsonl",
            source,
            replicate_count=2,
            replicate_seeds=(101, 202),
        )


def test_null_centered_p_value_and_holm_families() -> None:
    assert stats.null_centered_bootstrap_p_value(
        [1.0, 2.0, 3.0, 4.0], 2.0
    ) == pytest.approx(0.4)
    assert stats.null_centered_bootstrap_p_value([-1.0, 0.0, 1.0], 0.0) == 1.0

    arm_points = {
        "official": 0.0,
        "chk0": 3.0,
        "chk1": 2.0,
        "paper_chk2_cp50": 1.0,
    }
    points = []
    for horizon in stats.HORIZONS:
        for arm in stats.ARMS:
            value = arm_points[arm] + horizon / 100.0
            points.append(
                {
                    "horizon_months": horizon,
                    "arm": arm,
                    "beta_pre_basis_points": value,
                    "beta_interaction_basis_points": 0.0,
                    "beta_post_basis_points": value,
                }
            )
    draws = []
    offsets = {
        "official": [0.0, 0.0, 0.0, 0.0, 0.0],
        "chk0": [2.5, 3.0, 3.5, 2.8, 3.2],
        "chk1": [1.5, 2.0, 2.5, 1.8, 2.2],
        "paper_chk2_cp50": [0.5, 1.0, 1.5, 0.8, 1.2],
    }
    for draw_index in range(5):
        betas = {}
        for horizon in stats.HORIZONS:
            betas[f"FF{horizon}"] = {}
            for arm in stats.ARMS:
                value = offsets[arm][draw_index] + horizon / 100.0
                betas[f"FF{horizon}"][arm] = {
                    "pre_bp": value,
                    "interaction_bp": 0.0,
                    "post_bp": value,
                }
        draws.append({"betas": betas})

    contrasts, distances, gains = stats.summarize_bootstrap(points, draws)
    assert len(contrasts) == 36
    assert (
        len(
            [
                row
                for row in contrasts
                if row["family"] == "primitive_pre_and_interaction_24"
            ]
        )
        == 24
    )
    assert (
        len([row for row in contrasts if row["family"] == "derived_post_marginal_12"])
        == 12
    )
    assert len(distances) == 12
    assert len(gains) == 8
    ff1_chk0 = next(
        row
        for row in gains
        if row["horizon_months"] == 1 and row["comparator_arm"] == "chk0"
    )
    assert ff1_chk0["estimate_basis_points"] == pytest.approx(2.0)
    assert ff1_chk0["positive_means_chk2_closer"] is True
    assert all(0.0 <= row["holm_p"] <= 1.0 for row in [*contrasts, *gains])


def test_k_stability_uses_cumulative_replicates_and_closes_at_kmax() -> None:
    meetings = ("m0", "m1")
    raw = {
        "generated": {
            arm: {
                "m0": {0: 0.0, 1: 2.0},
                "m1": {0: 4.0, 1: 0.0},
            }
            for arm in stats.GENERATED_ARMS
        }
    }
    rows = stats.k_stability_rows(
        {"needed_ids": meetings}, {"raw": raw}, replicate_count=2
    )
    assert len(rows) == 6
    terminal = [row for row in rows if row["k"] == 2]
    assert all(row["mean_absolute_drift_from_kmax"] == 0.0 for row in terminal)
    assert all(row["pearson_correlation_with_kmax"] == 1.0 for row in terminal)


def _validation_fixture():
    point_values = {
        "official": 0.0,
        "chk0": 3.0,
        "chk1": 2.0,
        "paper_chk2_cp50": 1.0,
    }
    points = []
    for horizon in stats.HORIZONS:
        for arm in stats.ARMS:
            value = point_values[arm] + horizon / 100.0
            points.append(
                {
                    "horizon_months": horizon,
                    "arm": arm,
                    "nobs": stats.EXPECTED_NOBS[horizon],
                    "beta_pre_basis_points": value,
                    "hc1_se_pre_basis_points": 0.1,
                    "beta_interaction_basis_points": 0.0,
                    "hc1_se_interaction_basis_points": 0.1,
                    "beta_post_basis_points": value,
                    "hc1_se_post_basis_points": 0.1,
                    "r_squared": 0.1,
                    "condition_number": 10.0,
                }
            )
    offsets = {
        "official": [0.0, 0.0, 0.0, 0.0, 0.0],
        "chk0": [2.5, 3.0, 3.5, 2.8, 3.2],
        "chk1": [1.5, 2.0, 2.5, 1.8, 2.2],
        "paper_chk2_cp50": [0.5, 1.0, 1.5, 0.8, 1.2],
    }
    draws = []
    for draw_index in range(5):
        betas = {}
        for horizon in stats.HORIZONS:
            betas[f"FF{horizon}"] = {}
            for arm in stats.ARMS:
                value = offsets[arm][draw_index] + horizon / 100.0
                betas[f"FF{horizon}"][arm] = {
                    "pre_bp": value,
                    "interaction_bp": 0.0,
                    "post_bp": value,
                }
        draws.append(
            {"draw_index": draw_index, "plan_sha256": "test-plan", "betas": betas}
        )
    contrasts, distances, gains = stats.summarize_bootstrap(points, draws)
    source = {
        "needed_ids": tuple(f"m{i}" for i in range(84)),
        "scale_ids": tuple(f"m{i}" for i in range(1, 84)),
        "current_ids": tuple(f"m{i}" for i in range(2, 84)),
        "lag_only_ids": ("m0", "m1"),
    }
    plan = {"draws": 5, "plan_sha256": "test-plan"}
    return source, points, draws, contrasts, distances, gains, plan


def test_statistics_validation_rejects_nonfinite_bad_ci_and_nobs_drift() -> None:
    values = _validation_fixture()
    stats.validate_statistics(
        source=values[0],
        points=values[1],
        draws=values[2],
        contrasts=values[3],
        distances=values[4],
        gains=values[5],
        plan=values[6],
    )

    for mutation in ("nan", "reversed_ci", "negative_distance_ci", "nobs"):
        source, points, draws, contrasts, distances, gains, plan = copy.deepcopy(values)
        if mutation == "nan":
            points[0]["beta_pre_basis_points"] = float("nan")
        elif mutation == "reversed_ci":
            contrasts[0]["ci_95_low_basis_points"] = 10.0
            contrasts[0]["ci_95_high_basis_points"] = -10.0
        elif mutation == "negative_distance_ci":
            distances[0]["ci_95_low_basis_points"] = -0.01
        else:
            points[1]["nobs"] -= 1
        with pytest.raises(stats.StatisticsError, match="validation failed"):
            stats.validate_statistics(
                source=source,
                points=points,
                draws=draws,
                contrasts=contrasts,
                distances=distances,
                gains=gains,
                plan=plan,
            )


def test_partial_phase_resume_is_exact_and_tamper_fails(tmp_path: Path) -> None:
    root = tmp_path / "evaluation"
    artifact_a = root / "sentiment/a.txt"
    artifact_b = root / "sentiment/b.txt"
    artifact_a.parent.mkdir(parents=True)
    artifact_a.write_bytes(b"alpha\n")
    inputs: dict[str, dict] = {}
    manifest = stats._write_phase(
        root,
        phase="unit",
        manifest_path=root / "sentiment/manifest.json",
        inputs=inputs,
        artifacts={
            "a": (artifact_a, b"alpha\n", None),
            "b": (artifact_b, b"beta\n", None),
        },
        metadata={"test": True},
        resume=True,
    )
    assert manifest["status"] == "complete"
    assert artifact_b.read_bytes() == b"beta\n"
    artifact_a.write_bytes(b"tampered\n")
    with pytest.raises(stats.StatisticsError, match="artifact binding drift"):
        stats.verify_phase_manifest(
            root,
            root / "sentiment/manifest.json",
            phase="unit",
            inputs=inputs,
        )

    root2 = tmp_path / "tampered"
    bad = root2 / "sentiment/a.txt"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"wrong\n")
    with pytest.raises(stats.StatisticsError, match="differs on resume"):
        stats._write_phase(
            root2,
            phase="unit",
            manifest_path=root2 / "sentiment/manifest.json",
            inputs={},
            artifacts={"a": (bad, b"alpha\n", None)},
            metadata={},
            resume=True,
        )


def test_file_binding_rejects_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real.txt"
    real.write_text("safe", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(real)
    with pytest.raises(stats.StatisticsError, match="symlink"):
        stats.file_binding(link)
    with pytest.raises(stats.StatisticsError, match="symlink"):
        stats.read_json(link)
    with pytest.raises(stats.StatisticsError, match="symlink"):
        list(stats.iter_jsonl(link))
    with pytest.raises(stats.StatisticsError, match="symlink"):
        stats.sha256_file(link)


def test_sentiment_worker_budgets_are_byte_equivalent() -> None:
    positive = {"gain", "strong"}
    negative = {"loss", "weak"}
    statement = "The vote encompassed approval gain votes for this action: end."
    official = (
        "Context weak. The vote encompassed approval gain votes for this action: "
        "end. Further context strong."
    )
    tasks: list[tuple] = [
        ("official_pair", "m0", statement, official),
        ("official_pair", "m1", statement, official),
    ]
    tasks.extend(
        ("generated", "chk0", f"m{index}", index, 100 + index, "gain weak")
        for index in range(16)
    )
    expected = stats._run_score_tasks(tasks, positive, negative, workers=1)
    for workers in (2, 6, 32):
        observed = stats._run_score_tasks(tasks, positive, negative, workers=workers)
        assert stats.canonical(observed) == stats.canonical(expected)


def test_estimate_diagnostic_worker_budgets_are_byte_equivalent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stats, "HORIZONS", (1,))
    monkeypatch.setattr(stats, "ARMS", ("official",))
    monkeypatch.setattr(stats, "EXPECTED_NOBS", {1: 48})
    monkeypatch.setattr(stats, "EXPECTED_ESTIMABLE_EVENTS_BY_HORIZON", {1: 2})
    panel = []
    for index in range(48):
        year = 2004 + index // 4
        news = math.sin(index + 0.25) + 0.1 * (index % 3)
        vix = math.cos(index + 0.5) + 0.03 * index
        post = int(year > 2011 or (year == 2011 and index % 4 >= 2))
        panel.append(
            {
                "horizon_months": 1,
                "arm": "official",
                "trading_date": f"{year}-{1 + index % 4:02d}-15",
                "release_year": str(year),
                "event_meeting_id": f"m{index % 2}",
                "futures_return_log_percent": (
                    0.17 * news
                    + 0.09 * post * news
                    + 0.04 * vix
                    + 0.01 * math.sin(index * 0.7)
                ),
                "news_shock": news,
                "vix_log_percent_change": vix,
                "post_2011": post,
            }
        )
    fitted = stats._fit_panel_rows(panel)
    points = [
        {
            "horizon_months": 1,
            "arm": "official",
            "beta_pre_basis_points": 100.0 * fitted["beta_pre"],
            "beta_interaction_basis_points": 100.0 * fitted["beta_interaction"],
            "beta_post_basis_points": 100.0 * fitted["beta_post"],
        }
    ]
    expected = stats.sensitivity_diagnostics(panel, points, workers=1)
    for workers in (2, 6, 32):
        observed = stats.sensitivity_diagnostics(panel, points, workers=workers)
        assert stats.canonical(observed) == stats.canonical(expected)


def test_worker_budget_fails_closed() -> None:
    with pytest.raises(stats.StatisticsError, match="positive integer"):
        stats._worker_count(0, 1, label="test")
    with pytest.raises(stats.StatisticsError, match="available logical CPUs"):
        stats._worker_count((stats.os.cpu_count() or 1) + 1, 1, label="test")


def test_documents_manifest_closes_generation_lineage_and_detects_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluation"
    documents = root / "documents/generated_documents.jsonl"
    generation_rows = root / "generation/generation_rows.jsonl"
    generation_validation = root / "generation/validation.json"
    _write_jsonl(documents, [{"row": index} for index in range(2_520)])
    _write_jsonl(generation_rows, [{"row": index} for index in range(20_160)])
    generation_validation.write_text('{"status":"complete"}\n', encoding="utf-8")
    manifest = seal_manifest(
        {
            "schema_version": "paper-chk2-tadle-lm-k10-document-manifest-v1",
            "status": "complete",
            "evaluation_id": stats.EVALUATION_ID,
            "source_generation_validation": stats.file_binding(generation_validation),
            "source_generation_rows": stats.file_binding(generation_rows, rows=20_160),
            "generated_documents": stats.file_binding(documents, rows=2_520),
            "coverage": {
                "models": list(stats.GENERATED_ARMS),
                "meetings": 84,
                "replicates": 10,
                "topics_per_document": 8,
                "documents_per_model": 840,
                "documents_total": 2_520,
            },
        }
    )
    manifest_path = root / "documents/manifest.json"
    manifest_path.write_text(json_dumps(manifest), encoding="utf-8")
    assert stats.validate_documents_manifest(root, documents)["status"] == "complete"

    with documents.open("ab") as handle:
        handle.write(b'{"tampered":true}\n')
    with pytest.raises(stats.StatisticsError, match="row-count drift|binding drift"):
        stats.validate_documents_manifest(root, documents)


def json_dumps(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


@pytest.mark.skipif(
    not (LEGACY_ROOT / "estimation/analysis_panel.jsonl").is_file(),
    reason="sealed legacy replay artifact is unavailable",
)
def test_legacy_k5_panel_exactly_replays_point_estimates() -> None:
    panel = stats.read_jsonl(LEGACY_ROOT / "estimation/analysis_panel.jsonl")
    for row in panel:
        if row["arm"] == "chk3":
            row["arm"] = "paper_chk2_cp50"
    observed = stats.point_estimates(panel)
    expected = stats.read_jsonl(LEGACY_ROOT / "estimation/coefficient_estimates.jsonl")
    expected_lookup = {
        (
            int(row["horizon_months"]),
            "paper_chk2_cp50" if row["arm"] == "chk3" else row["arm"],
        ): row
        for row in expected
    }
    fields = (
        "beta_pre_basis_points",
        "beta_interaction_basis_points",
        "beta_post_basis_points",
        "hc1_se_pre_basis_points",
        "hc1_se_interaction_basis_points",
        "hc1_se_post_basis_points",
    )
    for row in observed:
        reference = expected_lookup[(int(row["horizon_months"]), str(row["arm"]))]
        for field in fields:
            assert row[field] == pytest.approx(reference[field], abs=1e-11)


@pytest.mark.skipif(
    not (LEGACY_ROOT / "sentiment/document_scores.jsonl").is_file(),
    reason="sealed legacy replay artifact is unavailable",
)
def test_bootstrap_worker_counts_are_numerically_identical() -> None:
    source = stats.load_source_skeleton()
    score_rows = stats.read_jsonl(LEGACY_ROOT / "sentiment/document_scores.jsonl")
    for row in score_rows:
        if row["arm"] == "chk3":
            row["arm"] = "paper_chk2_cp50"
    bundle = stats._score_bundle_from_rows(score_rows)
    semantics = stats.construct_semantics(source, bundle, replicate_count=5)
    plan = stats.make_bootstrap_plan(source, draws=3, seed=991, replicate_count=5)
    single = stats.run_bootstrap_draws(source, semantics, bundle, plan, workers=1)
    for workers in (2, 6, 32):
        parallel = stats.run_bootstrap_draws(
            source, semantics, bundle, plan, workers=workers
        )
        assert stats.canonical(single) == stats.canonical(parallel)
