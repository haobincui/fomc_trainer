from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from jobs.eval import estimate_chk3_sentiment_score_closeness_v1 as estimator
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _release_rows() -> list[dict]:
    return _jsonl(estimator.DEFAULT_RELEASE_LEDGER)


def _real_backend(directory: str) -> dict:
    root = estimator.DEFAULT_SENTIMENT_SUITE.parent / directory
    return {
        "manifest": json.loads((root / "manifest.json").read_text(encoding="utf-8")),
        "meeting_scores": _jsonl(root / "meeting_scores.jsonl"),
    }


def test_frozen_contract_and_paper_chk2_mapping() -> None:
    assert estimator.ARTIFACT_EXPERIMENTAL_ARM_ID == "chk3"
    assert estimator.PAPER_EXPERIMENTAL_STAGE_ID == "chk2"
    assert "chk2" not in estimator.ARM_ORDER
    assert estimator.BACKEND_ORDER == (
        estimator.sentiment.DISTIL_BACKEND,
        estimator.sentiment.FINBERT_BACKEND,
    )
    assert estimator.VIEW_ORDER == (
        "pre_external",
        "full",
        "post",
        "full_excl_current",
        "post_excl_current",
    )
    assert estimator.METRIC_ORDER == (
        "pearson_correlation",
        "spearman_correlation",
        "signed_bias",
        "mean_absolute_error",
        "root_mean_squared_error",
    )
    assert estimator.PRIMARY_SCALE == "pre_external_reference_sd"
    assert estimator.BOOTSTRAP_DRAWS == 10_000
    assert estimator.BOOTSTRAP_SEED == 20_260_817


def test_sentiment_only_views_have_exact_n_and_release_year_counts() -> None:
    views = estimator._views(estimator._ordered_release_rows(_release_rows()))
    assert {key: value.n_meetings for key, value in views.items()} == {
        "pre_external": 128,
        "full": 256,
        "post": 128,
        "full_excl_current": 247,
        "post_excl_current": 119,
    }
    assert {key: len(value.block_ids) for key, value in views.items()} == {
        "pre_external": 17,
        "full": 33,
        "post": 17,
        "full_excl_current": 32,
        "post_excl_current": 16,
    }
    # The closeness analysis does not inherit the market-regression N=126 lag loss.
    assert views["pre_external"].n_meetings != 126


def test_real_shared_complete_inventory_and_primary_point_metrics() -> None:
    release_rows = estimator._ordered_release_rows(_release_rows())
    scores = estimator._backend_scores(_real_backend("distilbert_fomc_neural_v2"))
    estimator._validate_release_score_alignment(release_rows, scores)
    counts = [len(value) for value in scores.paired_replicates_by_meeting.values()]
    assert counts.count(5) == 250
    assert counts.count(4) == 6
    view = estimator._views(release_rows)["pre_external"]
    assert (
        sum(
            len(scores.paired_replicates_by_meeting[meeting_id]) == 4
            for meeting_id in view.meeting_ids
        )
        == 4
    )
    scale = estimator._reference_scale(scores, release_rows)
    reference, matrices, inventory = estimator._score_arrays(
        view, scores, "shared_complete_core8"
    )
    means = estimator._point_means(view, matrices, inventory)
    metrics = estimator._metric_bundles(
        reference=reference,
        synthetic=means,
        standard_deviation=scale.standard_deviation,
    )
    raw_metrics = metrics["raw_score"]["chk3"]
    standardized = metrics["pre_external_reference_sd"]["chk3"]
    metrics = raw_metrics
    assert metrics["pearson_correlation"] == pytest.approx(
        0.061190268215911905, abs=1e-15
    )
    assert metrics["spearman_correlation"] == pytest.approx(
        0.003908395898187146, abs=1e-15
    )
    assert metrics["signed_bias"] == pytest.approx(-0.11017612104855934, abs=1e-15)
    assert metrics["mean_absolute_error"] == pytest.approx(
        0.11070963408533051, abs=1e-15
    )
    assert metrics["root_mean_squared_error"] == pytest.approx(
        0.12148428413808725, abs=1e-15
    )
    for metric_id in ("pearson_correlation", "spearman_correlation"):
        assert standardized[metric_id] == raw_metrics[metric_id]
    for metric_id in (
        "signed_bias",
        "mean_absolute_error",
        "root_mean_squared_error",
    ):
        assert standardized[metric_id] == (
            raw_metrics[metric_id] / scale.standard_deviation
        )


def test_scale_transform_is_exact_and_correlations_are_identical() -> None:
    raw = {
        "pearson_correlation": 0.25,
        "spearman_correlation": -0.125,
        "signed_bias": -0.4,
        "mean_absolute_error": 0.5,
        "root_mean_squared_error": 0.75,
    }
    sigma = 0.2
    standardized = estimator._scale_metrics(raw, sigma)
    assert standardized["pearson_correlation"] is raw["pearson_correlation"]
    assert standardized["spearman_correlation"] is raw["spearman_correlation"]
    for metric_id in (
        "signed_bias",
        "mean_absolute_error",
        "root_mean_squared_error",
    ):
        assert standardized[metric_id] == raw[metric_id] / sigma
    with pytest.raises(estimator.ClosenessEstimatorError, match="positive sigma"):
        estimator._scale_metrics(raw, 0.0)


def test_spearman_is_reranked_after_repeated_block_rows_expand() -> None:
    reference = np.asarray([1.0, 2.0, 3.0])
    model = np.asarray([1.0, 3.0, 2.0])
    expanded = np.asarray([0, 0, 1, 2], dtype=np.int64)
    observed = estimator._raw_metrics(reference[expanded], model[expanded])[
        "spearman_correlation"
    ]
    expected = estimator._pearson(
        estimator._midranks(reference[expanded]),
        estimator._midranks(model[expanded]),
    )
    incorrectly_preranked = estimator._pearson(
        estimator._midranks(reference)[expanded],
        estimator._midranks(model)[expanded],
    )
    assert observed == expected
    assert observed != incorrectly_preranked


def test_positive_closeness_gain_directions_include_absolute_bias() -> None:
    metrics = {
        "chk0": {
            "pearson_correlation": 0.1,
            "spearman_correlation": 0.2,
            "signed_bias": -0.7,
            "mean_absolute_error": 0.8,
            "root_mean_squared_error": 0.9,
        },
        "chk1": {
            "pearson_correlation": 0.3,
            "spearman_correlation": 0.4,
            "signed_bias": 0.5,
            "mean_absolute_error": 0.6,
            "root_mean_squared_error": 0.7,
        },
        "chk3": {
            "pearson_correlation": 0.5,
            "spearman_correlation": 0.6,
            "signed_bias": -0.2,
            "mean_absolute_error": 0.3,
            "root_mean_squared_error": 0.4,
        },
    }
    gains = estimator._gain_values(metrics)
    vs_chk1 = gains["chk3_closeness_gain_vs_chk1"]
    assert vs_chk1 == {
        "pearson_correlation": pytest.approx(0.2),
        "spearman_correlation": pytest.approx(0.2),
        "signed_bias": pytest.approx(0.3),
        "mean_absolute_error": pytest.approx(0.3),
        "root_mean_squared_error": pytest.approx(0.3),
    }
    assert estimator.GAIN_METRIC_IDS["signed_bias"] == ("absolute_bias_closeness_gain")


def test_midrank_uses_average_rank_for_ties() -> None:
    assert estimator._midranks(np.asarray([10.0, 10.0, 20.0, 30.0])).tolist() == [
        1.5,
        1.5,
        3.0,
        4.0,
    ]


def _fake_backend(release_rows: list[dict], *, backend_id: str) -> dict:
    rows: list[dict] = []
    missing_meetings = {str(row["generation_meeting_id"]) for row in release_rows[:6]}
    for index, release in enumerate(release_rows):
        meeting_id = str(release["generation_meeting_id"])
        exposed = bool(release["cp318_selection_exposed"])
        reference = index / 100.0 + (index % 7) / 1000.0
        rows.append(
            {
                "meeting_id": meeting_id,
                "arm": "reference",
                "replicate_id": None,
                "complete_core8": True,
                "score": reference,
                "neutral_imputed_score": reference,
                "cp318_selection_exposed": exposed,
            }
        )
        for arm_index, arm in enumerate(estimator.SYNTHETIC_ARMS, start=1):
            for replicate_id in range(5):
                incomplete = (
                    arm == "chk3"
                    and meeting_id in missing_meetings
                    and replicate_id == 0
                )
                value = (
                    reference * (1.0 + arm_index / 100.0)
                    + arm_index / 1000.0
                    + replicate_id / 10000.0
                )
                rows.append(
                    {
                        "meeting_id": meeting_id,
                        "arm": arm,
                        "replicate_id": replicate_id,
                        "complete_core8": not incomplete,
                        "score": None if incomplete else value,
                        "neutral_imputed_score": 0.0 if incomplete else value,
                        "cp318_selection_exposed": exposed,
                    }
                )
    return {
        "manifest": {
            "backend": backend_id,
            "construct": (
                "monetary_policy_stance_hawkish_minus_dovish"
                if backend_id == estimator.sentiment.DISTIL_BACKEND
                else "financial_valence_positive_minus_negative"
            ),
        },
        "meeting_scores": rows,
    }


def _fake_inputs() -> dict:
    release_rows = _release_rows()
    sentiment_manifest = json.loads(
        estimator.DEFAULT_SENTIMENT_SUITE.read_text(encoding="utf-8")
    )
    market_manifest_path = estimator.DEFAULT_RELEASE_LEDGER.parent / "manifest.json"
    market_manifest = json.loads(market_manifest_path.read_text(encoding="utf-8"))
    generation = json.loads(
        (
            estimator.DEFAULT_SENTIMENT_SUITE.parent / "preparation/manifest.json"
        ).read_text(encoding="utf-8")
    )["source_generation_suite"]
    return {
        "sentiment": {
            "manifest": sentiment_manifest,
            "manifest_binding": estimator._binding(
                estimator.DEFAULT_SENTIMENT_SUITE,
                payload_sha256=validate_manifest_integrity(sentiment_manifest),
            ),
            "inventory": {"manifest": {"source_generation_suite": generation}},
            "backends": {
                backend_id: _fake_backend(release_rows, backend_id=backend_id)
                for backend_id in estimator.BACKEND_ORDER
            },
        },
        "release_rows": release_rows,
        "sentiment_manifest_binding": estimator._binding(
            estimator.DEFAULT_SENTIMENT_SUITE,
            payload_sha256=validate_manifest_integrity(sentiment_manifest),
        ),
        "release_ledger_binding": estimator._binding(
            estimator.DEFAULT_RELEASE_LEDGER, rows=256
        ),
        "market_manifest_binding": estimator._binding(
            market_manifest_path,
            payload_sha256=validate_manifest_integrity(market_manifest),
        ),
    }


def _unseal_for_cleanup(root: Path) -> None:
    if not root.exists():
        return
    root.chmod(0o755)
    for path in root.iterdir():
        path.chmod(0o644)


def _set_runtime_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in estimator.REQUIRED_RUNTIME_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)


def test_create_only_seal_report_model_and_full_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_runtime_environment(monkeypatch)
    fake = _fake_inputs()
    monkeypatch.setattr(estimator, "_load_inputs", lambda **_: copy.deepcopy(fake))
    output = tmp_path / "closeness"
    try:
        manifest = estimator.estimate(output_dir=output, bootstrap_draws=3)
        assert manifest["coverage"] == {
            "backends": 2,
            "views": 5,
            "score_policies": 2,
            "scales": 2,
            "metrics": 5,
            "synthetic_arms": 3,
            "analysis_cells": 20,
            "meeting_aggregate_rows": 4096,
            "estimate_rows": 600,
            "contrast_rows": 400,
            "bootstrap_draw_rows": 60,
            "bootstrap_draws_per_analysis": 3,
        }
        assert oct(output.stat().st_mode & 0o777) == "0o555"
        assert all(
            oct(path.stat().st_mode & 0o777) == "0o444" for path in output.iterdir()
        )
        loaded = estimator.load_and_validate_closeness(output / "manifest.json")
        assert loaded["bootstrap_draw_rows"] == 60
        report = loaded["report_model"]
        assert report["primary_artifact_arm"] == "chk3"
        assert report["primary_paper_stage_id"] == "chk2"
        assert len(report["headline_estimates"]) == 5
        assert len(report["headline_contrasts"]) == 10
        assert len(report["scatter_rows"]) == 128
        with pytest.raises(estimator.ClosenessEstimatorError, match="fresh"):
            estimator.estimate(output_dir=output, bootstrap_draws=1)
    finally:
        _unseal_for_cleanup(output)


def test_deep_replay_rejects_resealed_draw_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_runtime_environment(monkeypatch)
    fake = _fake_inputs()
    monkeypatch.setattr(estimator, "_load_inputs", lambda **_: copy.deepcopy(fake))
    output = tmp_path / "closeness_tamper"
    try:
        estimator.estimate(output_dir=output, bootstrap_draws=2)
        output.chmod(0o755)
        draw_path = output / estimator.DRAWS_FILENAME
        manifest_path = output / "manifest.json"
        draw_path.chmod(0o644)
        manifest_path.chmod(0o644)
        lines = draw_path.read_text(encoding="utf-8").splitlines()
        first = json.loads(lines[0])
        first["metrics"]["raw_score"]["chk3"]["signed_bias"] += 0.125
        payload = dict(first)
        payload.pop("draw_row_sha256")
        first["draw_row_sha256"] = estimator._record_sha256(payload)
        lines[0] = estimator._canonical(first)
        draw_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        draw_path.chmod(0o444)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"]["bootstrap_draws"] = estimator._binding(
            draw_path, rows=manifest["coverage"]["bootstrap_draw_rows"]
        )
        manifest = seal_manifest(
            {key: value for key, value in manifest.items() if key != "integrity"}
        )
        manifest_path.write_text(
            estimator._canonical(manifest) + "\n", encoding="utf-8"
        )
        manifest_path.chmod(0o444)
        output.chmod(0o555)
        with pytest.raises(
            estimator.ClosenessEstimatorError,
            match="bootstrap deterministic replay drift: 0",
        ):
            estimator.load_and_validate_closeness(manifest_path)
    finally:
        _unseal_for_cleanup(output)


def test_import_source_guard_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    altered = copy.deepcopy(estimator._IMPLEMENTATION_IMPORT_BINDINGS)
    altered["statistical_primitives"]["sha256"] = "0" * 64
    monkeypatch.setattr(estimator, "_IMPLEMENTATION_IMPORT_BINDINGS", altered)
    with pytest.raises(estimator.ClosenessEstimatorError, match="changed after import"):
        estimator._implementation_source_bindings()


def test_runtime_environment_gate_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_runtime_environment(monkeypatch)
    assert estimator._require_runtime_environment() == (
        estimator.REQUIRED_RUNTIME_ENVIRONMENT
    )
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    with pytest.raises(estimator.ClosenessEstimatorError, match="runtime environment"):
        estimator._require_runtime_environment()
