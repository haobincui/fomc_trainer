from __future__ import annotations

import copy
import json
import os
from datetime import date, timedelta
from pathlib import Path

import pytest

from jobs.eval import estimate_chk3_sentiment_score_closeness_v1 as estimator
from jobs.eval import render_chk3_sentiment_score_closeness_report_v1 as subject
from open_r1.validator.loo_generation_spec import validate_manifest_integrity


BACKENDS = {
    subject.PRIMARY_BACKEND_ID: (
        "DistilBERT FOMC stance",
        "hawkish-minus-dovish stance",
    ),
    subject.ROBUSTNESS_BACKEND_ID: (
        "ProsusAI FinBERT financial valence",
        "positive-minus-negative financial valence",
    ),
}
VIEWS = (
    ("pre_external", "Pre-2009 external panel", 128),
    ("full", "Full panel", 256),
    ("post", "Post-2008 panel", 128),
    ("full_excl_current", "Full excluding selection exposure", 247),
    ("post_excl_current", "Post excluding selection exposure", 119),
)
POLICIES = ("shared_complete_core8", "neutral_imputed_fixed_k5")
SCALES = ("raw_score", subject.PRIMARY_SCALE_ID)
REFERENCE_SDS = {
    subject.PRIMARY_BACKEND_ID: 0.001254246515242593,
    subject.ROBUSTNESS_BACKEND_ID: 0.030700216259587926,
}
METRIC_IDS = {
    "pearson": "pearson_r",
    "spearman": "spearman_rho",
    "bias": "bias",
    "mae": "mae",
    "rmse": "rmse",
}
POINTS = {
    "reference": {
        "pearson": 1.0,
        "spearman": 1.0,
        "bias": 0.0,
        "mae": 0.0,
        "rmse": 0.0,
    },
    "chk0": {
        "pearson": -0.046966,
        "spearman": -0.118826,
        "bias": -0.122698,
        "mae": 0.123634,
        "rmse": 0.139095,
    },
    "chk1": {
        "pearson": -0.104650,
        "spearman": -0.164731,
        "bias": -0.119416,
        "mae": 0.119826,
        "rmse": 0.134707,
    },
    "chk3": {
        "pearson": 0.061190,
        "spearman": 0.003908,
        "bias": -0.110176,
        "mae": 0.110710,
        "rmse": 0.121484,
    },
}


def _source_bindings(root: Path) -> dict[str, dict]:
    names = (
        "estimator_manifest",
        "meeting_aggregates",
        "estimates",
        "contrasts",
        "diagnostics",
        "bootstrap_draws",
        "sentiment_suite_manifest",
        "generation_suite_manifest",
        "official_release_ledger",
        "market_manifest_seal_anchor",
    )
    result = {}
    for index, name in enumerate(names):
        path = root / "sealed" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"source": name, "index": index}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result[name] = subject._binding(path)
    return result


def _point(
    *, backend_id: str, arm: str, metric: str, scale_id: str, offset: float
) -> float:
    raw = POINTS[arm][metric]
    if backend_id == subject.ROBUSTNESS_BACKEND_ID and arm != "reference":
        raw += offset
    if scale_id == subject.PRIMARY_SCALE_ID and metric in {"bias", "mae", "rmse"}:
        return raw / REFERENCE_SDS[backend_id]
    return raw


def _gain(
    *,
    backend_id: str,
    metric: str,
    baseline: str,
    scale_id: str,
    offset: float,
    focal: str = "chk3",
) -> float:
    before = _point(
        backend_id=backend_id,
        arm=baseline,
        metric=metric,
        scale_id=scale_id,
        offset=offset,
    )
    after = _point(
        backend_id=backend_id,
        arm=focal,
        metric=metric,
        scale_id=scale_id,
        offset=offset,
    )
    if metric in {"pearson", "spearman"}:
        return after - before
    if metric == "bias":
        return abs(before) - abs(after)
    return before - after


def _estimate(
    *,
    backend_id: str,
    view_id: str,
    view_label: str,
    n: int,
    policy: str,
    scale_id: str,
    arm: str,
    metric: str,
    offset: float,
) -> dict:
    point = _point(
        backend_id=backend_id,
        arm=arm,
        metric=metric,
        scale_id=scale_id,
        offset=offset,
    )
    return {
        "schema_version": "chk3-sentiment-score-closeness-estimate-v1",
        "estimate_id": (
            f"{backend_id}::{view_id}::{policy}::{scale_id}::"
            f"{arm}::{METRIC_IDS[metric]}"
        ),
        "analysis_id": f"{backend_id}::{view_id}::{policy}",
        "backend_id": backend_id,
        "backend_label": BACKENDS[backend_id][0],
        "construct": BACKENDS[backend_id][1],
        "view_id": view_id,
        "view_label": view_label,
        "score_policy": policy,
        "scale_id": scale_id,
        "scale_units": (
            "backend_signed_score_units"
            if scale_id == "raw_score"
            else "pre_external_reference_standard_deviations"
        ),
        "arm": arm,
        "arm_label": subject._display_arm(arm),
        "paper_stage_id": "chk2" if arm == "chk3" else None,
        "metric_id": METRIC_IDS[metric],
        "metric_label": subject.METRIC_SHORT_LABELS[metric],
        "closeness_direction": (
            "higher_is_closer"
            if metric in {"pearson", "spearman"}
            else "zero_is_closer"
            if metric == "bias"
            else "lower_is_closer"
        ),
        "point_estimate": point,
        "ci_lower": point - 0.025,
        "ci_upper": point + 0.025,
        "bootstrap_interval": "paired_percentile_95pct",
        "n_meetings": n,
        "successful_bootstrap_draws": 10_000,
        "failed_bootstrap_draws": 0,
        "bootstrap_plan_sha256": "0" * 64,
    }


def _contrast(
    *,
    backend_id: str,
    view_id: str,
    view_label: str,
    n: int,
    policy: str,
    scale_id: str,
    baseline: str,
    metric: str,
    offset: float,
) -> dict:
    gain = _gain(
        backend_id=backend_id,
        metric=metric,
        baseline=baseline,
        scale_id=scale_id,
        offset=offset,
    )
    formula = {
        "pearson": "pearson_chk3 - pearson_baseline",
        "spearman": "spearman_chk3 - spearman_baseline",
        "bias": "abs(bias_baseline) - abs(bias_chk3)",
        "mae": "mae_baseline - mae_chk3",
        "rmse": "rmse_baseline - rmse_chk3",
    }[metric]
    interval_radius = abs(gain) + 0.04
    return {
        "schema_version": "chk3-sentiment-score-closeness-contrast-v1",
        "contrast_row_id": (
            f"{backend_id}::{view_id}::{policy}::{scale_id}::"
            f"chk3_vs_{baseline}::{METRIC_IDS[metric]}"
        ),
        "contrast_id": f"chk3_closeness_gain_vs_{baseline}",
        "analysis_id": f"{backend_id}::{view_id}::{policy}",
        "backend_id": backend_id,
        "backend_label": BACKENDS[backend_id][0],
        "construct": BACKENDS[backend_id][1],
        "view_id": view_id,
        "view_label": view_label,
        "score_policy": policy,
        "scale_id": scale_id,
        "scale_units": (
            "backend_signed_score_units"
            if scale_id == "raw_score"
            else "pre_external_reference_standard_deviations"
        ),
        "baseline_arm": baseline,
        "after_arm": "chk3",
        "paper_after_stage_id": "chk2",
        "metric_id": METRIC_IDS[metric],
        "metric_label": subject.METRIC_SHORT_LABELS[metric],
        "gain_metric_id": {
            "pearson": "pearson_correlation_gain",
            "spearman": "spearman_correlation_gain",
            "bias": "absolute_bias_closeness_gain",
            "mae": "mean_absolute_error_closeness_gain",
            "rmse": "root_mean_squared_error_closeness_gain",
        }[metric],
        "gain_formula": formula,
        "positive_interpretation": "positive_means_chk3_is_closer",
        "point_estimate": gain,
        "ci_lower": gain - interval_radius,
        "ci_upper": gain + interval_radius,
        "bootstrap_interval": "paired_percentile_95pct",
        "n_meetings": n,
        "successful_bootstrap_draws": 10_000,
        "failed_bootstrap_draws": 0,
        "bootstrap_plan_sha256": "0" * 64,
    }


def _meeting_aggregates() -> list[dict]:
    result = []
    pre_start = date(1993, 1, 5)
    post_start = date(2009, 1, 5)
    for backend_index, (backend_id, (_label, construct)) in enumerate(BACKENDS.items()):
        for policy in POLICIES:
            for meeting_index in range(256):
                if meeting_index < subject.PRIMARY_N_MEETINGS:
                    meeting_date = pre_start + timedelta(days=meeting_index * 45)
                    era = "pre_external"
                else:
                    meeting_date = post_start + timedelta(
                        days=(meeting_index - subject.PRIMARY_N_MEETINGS) * 45
                    )
                    era = "post"
                release_date = meeting_date + timedelta(days=30)
                reference = -0.015 + meeting_index * 0.0002 + backend_index * 0.08
                for arm_index, arm in enumerate(subject.ARM_ORDER):
                    deterministic = arm == "reference"
                    incomplete = meeting_index in {0, 1, 2, 3, 128, 129}
                    if deterministic:
                        replicate_ids = []
                    elif policy == "neutral_imputed_fixed_k5":
                        replicate_ids = list(range(5))
                    else:
                        replicate_ids = list(range(4 if incomplete else 5))
                    raw = (
                        reference
                        if deterministic
                        else reference - 0.12 + arm_index * 0.01
                    )
                    result.append(
                        {
                            "schema_version": "chk3-sentiment-score-closeness-meeting-aggregate-v1",
                            "row_id": (
                                f"{backend_id}::{policy}::{meeting_index:03d}::{arm}"
                            ),
                            "backend_id": backend_id,
                            "construct": construct,
                            "score_policy": policy,
                            "meeting_id": f"meeting-{meeting_index:03d}",
                            "meeting_date": meeting_date.isoformat(),
                            "release_date": release_date.isoformat(),
                            "release_year": release_date.year,
                            "era": era,
                            "cp318_selection_exposed": False,
                            "arm": arm,
                            "arm_label": subject._display_arm(arm),
                            "paper_stage_id": "chk2" if arm == "chk3" else None,
                            "deterministic_reference": deterministic,
                            "replicate_ids": replicate_ids,
                            "replicate_count": len(replicate_ids),
                            "raw_score": raw,
                            "pre_external_reference_sd_score": raw / 0.02,
                        }
                    )
    return result


def _report_model(root: Path) -> dict:
    estimates = []
    contrasts = []
    for backend_index, backend_id in enumerate(BACKENDS):
        for view_id, view_label, n in VIEWS:
            for policy_index, policy in enumerate(POLICIES):
                offset = backend_index * 0.01 + policy_index * 0.001
                for scale_id in SCALES:
                    for arm in ("chk0", "chk1", "chk3"):
                        for metric in subject.METRIC_ORDER:
                            estimates.append(
                                _estimate(
                                    backend_id=backend_id,
                                    view_id=view_id,
                                    view_label=view_label,
                                    n=n,
                                    policy=policy,
                                    scale_id=scale_id,
                                    arm=arm,
                                    metric=metric,
                                    offset=offset,
                                )
                            )
                    for baseline in ("chk1", "chk0"):
                        for metric in subject.METRIC_ORDER:
                            contrasts.append(
                                _contrast(
                                    backend_id=backend_id,
                                    view_id=view_id,
                                    view_label=view_label,
                                    n=n,
                                    policy=policy,
                                    scale_id=scale_id,
                                    baseline=baseline,
                                    metric=metric,
                                    offset=offset,
                                )
                            )
    headline_estimates = [
        row["estimate_id"]
        for row in estimates
        if row["backend_id"] == subject.PRIMARY_BACKEND_ID
        and row["view_id"] == subject.PRIMARY_VIEW_ID
        and row["score_policy"] == subject.PRIMARY_SCORE_POLICY
        and row["scale_id"] == subject.PRIMARY_SCALE_ID
        and row["arm"] == "chk3"
    ]
    headline_contrasts = [
        row["contrast_row_id"]
        for row in contrasts
        if row["backend_id"] == subject.PRIMARY_BACKEND_ID
        and row["view_id"] == subject.PRIMARY_VIEW_ID
        and row["score_policy"] == subject.PRIMARY_SCORE_POLICY
        and row["scale_id"] == subject.PRIMARY_SCALE_ID
    ]
    return {
        "generated_at_utc": "2026-08-18T12:00:00Z",
        "coverage": {
            "meeting_aggregate_rows": 4096,
            "estimate_rows": len(estimates),
            "contrast_rows": len(contrasts),
        },
        "view_specs": [
            {"view_id": view_id, "label": label, "n_meetings": n}
            for view_id, label, n in VIEWS
        ],
        "methodology": {
            "estimand": "meeting-level score closeness to the fixed synthetic reference",
            "meeting_aggregation": "mean of available K=5 stochastic replicates",
            "bootstrap_unit": "release calendar year",
            "within_meeting_resampling": "paired replicate resampling shared across arms",
            "bootstrap_draws": 10_000,
            "bootstrap_seed": 20_260_818,
            "interval": "paired percentile 95% interval",
            "undefined_correlation_draws": "counted as failed; never redrawn",
        },
        "limitations": [
            "K=5 leaves finite Monte Carlo noise in each generated-arm meeting mean.",
            "The DistilBERT pre-external reference SD is small, so standardized error units can be numerically large.",
        ],
        "reference_scales": copy.deepcopy(REFERENCE_SDS),
        "headline_estimate_row_ids": headline_estimates,
        "headline_contrast_row_ids": headline_contrasts,
        "meeting_aggregates": _meeting_aggregates(),
        "estimates": estimates,
        "contrasts": contrasts,
        "diagnostics": {
            "bootstrap_draws": 10_000,
            "failed_correlation_draws": 0,
            "replicate_pooling": False,
        },
        "source_bindings": _source_bindings(root),
    }


def _make_writable(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        child.chmod(0o755 if child.is_dir() else 0o600)
    path.chmod(0o755)


def test_artifact_is_answer_first_meeting_level_and_source_backed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    model = _report_model(tmp_path)

    artifact = subject.build_canonical_artifact(model)

    assert artifact["surface"] == "report"
    assert artifact["manifest"]["surface"] == "report"
    assert artifact["snapshot"]["status"] == "ready"
    assert len(artifact["snapshot"]["datasets"]["distil_raw_scatter"]) == 128
    assert len(artifact["snapshot"]["datasets"]["finbert_raw_scatter"]) == 128
    first = artifact["snapshot"]["datasets"]["distil_raw_scatter"][0]
    assert first["reference_raw_score"] == pytest.approx(-0.015)
    assert first["chk2_raw_score"] == pytest.approx(-0.105)
    assert first["focal_replicate_count"] == 4
    assert {
        row["focal_replicate_count"]
        for row in artifact["snapshot"]["datasets"]["distil_raw_scatter"]
    } == {4, 5}

    rendered_estimates = [
        row
        for name, rows in artifact["snapshot"]["datasets"].items()
        if name.startswith("all_estimates_page_")
        for row in rows
        if row["backend_id"] == subject.PRIMARY_BACKEND_ID
        and row["artifact_arm"] == "chk3"
        and row["view"] == subject.PRIMARY_VIEW_ID
        and row["score_policy"] == subject.PRIMARY_SCORE_POLICY
        and row["metric_id"] == "mae"
    ]
    raw_mae = next(row for row in rendered_estimates if row["scale"] == "raw_score")
    scaled_mae = next(
        row for row in rendered_estimates if row["scale"] == subject.PRIMARY_SCALE_ID
    )
    assert scaled_mae["estimate"] == pytest.approx(
        raw_mae["estimate"] / REFERENCE_SDS[subject.PRIMARY_BACKEND_ID]
    )
    assert scaled_mae["estimate"] > 80.0
    scale_comparison = artifact["snapshot"]["datasets"]["primary_focal_scale_metrics"]
    assert len(scale_comparison) == 10
    assert {row["scale"] for row in scale_comparison} == {
        "raw_score",
        subject.PRIMARY_SCALE_ID,
    }
    scale_rows = artifact["snapshot"]["datasets"]["reference_scales"]
    distil_scale = next(
        row for row in scale_rows if row["backend_id"] == subject.PRIMARY_BACKEND_ID
    )
    assert distil_scale["reference_sd_ddof1"] == pytest.approx(
        REFERENCE_SDS[subject.PRIMARY_BACKEND_ID]
    )
    support = artifact["snapshot"]["datasets"]["support_diagnostics"]
    assert all(row["raw_score_sd"] > 0.0 for row in support)
    assert all(row["variance_ratio_vs_reference"] > 0.0 for row in support)
    assert all(row["raw_score_min"] <= row["raw_score_p05"] for row in support)
    assert all(row["raw_score_p95"] <= row["raw_score_max"] for row in support)

    charts = {chart["id"]: chart for chart in artifact["manifest"]["charts"]}
    assert set(charts) == {"distil_raw_scatter_chart", "finbert_raw_scatter_chart"}
    assert all(chart["type"] == "scatter" for chart in charts.values())
    assert all("referenceLines" not in chart for chart in charts.values())
    assert charts["distil_raw_scatter_chart"]["encodings"]["x"]["field"] == (
        "reference_raw_score"
    )
    assert charts["distil_raw_scatter_chart"]["encodings"]["y"]["field"] == (
        "chk2_raw_score"
    )

    body = "\n".join(block.get("body", "") for block in artifact["manifest"]["blocks"])
    assert subject.PAPER_STAGE_DISCLOSURE in body
    assert subject.MEETING_GRAIN_DISCLOSURE in body
    assert subject.IDENTITY_LINE_DISCLOSURE in body
    assert subject.NONPOOLED_DISCLOSURE in body
    assert "N=128" in body
    assert "N=126" not in body
    assert "gain = paper CHK2 minus baseline" in body
    assert "|baseline bias| minus |paper-CHK2 bias|" in body
    assert (
        "All five paired gain intervals include zero; point gains are not stable "
        "evidence of improvement."
    ) in body
    assert "Pearson and Spearman are pattern-alignment diagnostics." in body
    assert "Signed bias, MAE, and RMSE are level-distance diagnostics." in body
    assert "small denominator" in body
    assert "Raw and reference-SD values are displayed together" in body
    assert "fidelity" not in body.lower()
    assert "significant" not in body.lower()

    block_ids = [block["id"] for block in artifact["manifest"]["blocks"]]
    assert block_ids.index("technical_summary_interpretation") < block_ids.index(
        "technical_summary_direct"
    )

    cards = artifact["manifest"]["cards"]
    assert len(cards) == 12
    direct = [card for card in cards if card["id"].endswith("_direct_card")]
    gains = [card for card in cards if card["id"].endswith("_gain_card")]
    assert len(direct) == len(gains) == 5
    assert {card["sourceId"] for card in direct} == {"estimates_source"}
    assert {card["sourceId"] for card in gains} == {"contrasts_source"}

    tables = {table["id"] for table in artifact["manifest"]["tables"]}
    assert {
        "primary_arm_metrics_table",
        "primary_gains_table",
        "primary_scale_comparison_table",
        "reference_scales_table",
        "support_diagnostics_table",
        "backend_robustness_table",
        "view_robustness_table",
        "policy_robustness_table",
    } <= tables
    source_ids = {source["id"] for source in artifact["sources"]}
    reachable = {
        item["sourceId"]
        for collection in (
            artifact["manifest"]["cards"],
            artifact["manifest"]["charts"],
            artifact["manifest"]["tables"],
            artifact["manifest"]["blocks"],
        )
        for item in collection
        if "sourceId" in item
    }
    assert source_ids <= reachable
    assert all(not source["path"].startswith("/") for source in artifact["sources"])


def test_model_rejects_n126_draw_drift_and_replicate_pooling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    model = _report_model(tmp_path)

    wrong_n = copy.deepcopy(model)
    next(
        row
        for row in wrong_n["estimates"]
        if row["estimate_id"] in wrong_n["headline_estimate_row_ids"]
    )["n_meetings"] = 126
    with pytest.raises(subject.SentimentClosenessReportError, match="coverage"):
        subject.validate_report_model(wrong_n)

    bad_draws = copy.deepcopy(model)
    bad_draws["contrasts"][0]["successful_bootstrap_draws"] = 9_999
    with pytest.raises(subject.SentimentClosenessReportError, match="10,000"):
        subject.validate_report_model(bad_draws)

    pooled = copy.deepcopy(model)
    focal = next(row for row in pooled["meeting_aggregates"] if row["arm"] == "chk3")
    focal["replicate_count"] = 640
    with pytest.raises(subject.SentimentClosenessReportError, match="replicate count"):
        subject.validate_report_model(pooled)


def test_summary_zero_interval_conclusion_is_derived_from_validated_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    model = _report_model(tmp_path)
    pearson_gain = next(
        row
        for row in model["contrasts"]
        if row["backend_id"] == subject.PRIMARY_BACKEND_ID
        and row["view_id"] == subject.PRIMARY_VIEW_ID
        and row["score_policy"] == subject.PRIMARY_SCORE_POLICY
        and row["scale_id"] == subject.PRIMARY_SCALE_ID
        and row["baseline_arm"] == subject.BASELINE_ARM
        and row["metric_id"] == "pearson_r"
    )
    pearson_gain["ci_lower"] = 0.01
    pearson_gain["ci_upper"] = max(0.30, float(pearson_gain["point_estimate"]))

    artifact = subject.build_canonical_artifact(model)
    body = "\n".join(block.get("body", "") for block in artifact["manifest"]["blocks"])

    assert "All five paired gain intervals include zero" not in body
    assert "The five paired gain intervals do not all include zero" in body
    assert "Pearson correlation gain" in body


def test_normalized_adapter_uses_frozen_loader_surface(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    expected = _report_model(tmp_path)
    report_model = {
        key: copy.deepcopy(value)
        for key, value in expected.items()
        if key not in {"generated_at_utc", "source_bindings"}
    }
    report_model["source_manifest_binding"] = expected["source_bindings"][
        "estimator_manifest"
    ]
    report_model["source_bindings"] = expected["source_bindings"]
    loaded = {
        "manifest": {"created_at_utc": expected["generated_at_utc"]},
        "manifest_binding": expected["source_bindings"]["estimator_manifest"],
        "report_model": report_model,
    }

    actual = subject._normalize_estimator(loaded)

    assert actual == expected
    assert actual["headline_estimate_row_ids"] == expected["headline_estimate_row_ids"]


def test_authoritative_estimator_report_model_integrates_with_renderer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    expected = _report_model(tmp_path)
    bindings = expected["source_bindings"]
    manifest = {
        "created_at_utc": expected["generated_at_utc"],
        "primary_selectors": {
            "backend_id": subject.PRIMARY_BACKEND_ID,
            "view_id": subject.PRIMARY_VIEW_ID,
            "score_policy": subject.PRIMARY_SCORE_POLICY,
            "scale_id": subject.PRIMARY_SCALE_ID,
            "arm": subject.FOCAL_ARM,
            "paper_stage_id": "chk2",
        },
        "reporting": {
            "headline_estimate_row_ids": expected["headline_estimate_row_ids"],
            "headline_contrast_row_ids": expected["headline_contrast_row_ids"],
            "arm_display_alias": "CHK2 (artifact CHK3/cp318)",
        },
        "analysis_contract": expected["methodology"],
        "limitations": expected["limitations"],
        "coverage": expected["coverage"],
        "reference_scales": expected["reference_scales"],
        "view_specs": expected["view_specs"],
        "artifacts": {
            name: bindings[name]
            for name in (
                "meeting_aggregates",
                "estimates",
                "contrasts",
                "bootstrap_draws",
                "diagnostics",
            )
        },
        "inputs": {
            name: bindings[name]
            for name in (
                "sentiment_suite_manifest",
                "official_release_ledger",
                "market_manifest_seal_anchor",
                "generation_suite_manifest",
            )
        },
    }
    authoritative = estimator._report_model(
        manifest=manifest,
        manifest_binding=bindings["estimator_manifest"],
        meeting_aggregates=expected["meeting_aggregates"],
        estimates=expected["estimates"],
        contrasts=expected["contrasts"],
        diagnostics=expected["diagnostics"],
    )
    loaded = {
        "manifest": manifest,
        "manifest_binding": bindings["estimator_manifest"],
        "report_model": authoritative,
    }

    actual = subject._normalize_estimator(loaded)

    assert actual == expected
    assert "manifest" not in actual["source_bindings"]
    assert (
        actual["source_bindings"]["estimator_manifest"]
        == bindings["estimator_manifest"]
    )


def test_incomplete_model_leaves_no_final_or_staging_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    model = _report_model(tmp_path)
    model["headline_estimate_row_ids"].pop()
    estimator_binding = model["source_bindings"]["estimator_manifest"]
    monkeypatch.setattr(
        subject,
        "load_report_model",
        lambda *_args, **_kwargs: (copy.deepcopy(model), estimator_binding),
    )
    output = tmp_path / "sentiment_score_closeness_report_v1"

    with pytest.raises(subject.SentimentClosenessReportError, match="selector"):
        subject.render_and_seal_report(
            estimator_manifest=Path(estimator_binding["path"]),
            estimator_manifest_sha256=estimator_binding["sha256"],
            output_dir=output,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".sentiment_score_closeness_report_v1.staging-*"))


def test_complete_fixture_packages_once_and_seals_exact_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    node = repository_root / ".cache/report_node/bin/node"
    if not node.is_file():
        pytest.skip("frozen local Node runtime unavailable")
    if not subject._delivery.DEFAULT_PLUGIN_ROOT.is_dir():
        pytest.skip("packaged Data Analytics report builder unavailable")
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    model = _report_model(tmp_path)
    estimator_binding = model["source_bindings"]["estimator_manifest"]
    monkeypatch.setattr(
        subject,
        "load_report_model",
        lambda *_args, **_kwargs: (copy.deepcopy(model), estimator_binding),
    )
    output = tmp_path / "sentiment_score_closeness_report_v1"

    manifest = subject.render_and_seal_report(
        estimator_manifest=Path(estimator_binding["path"]),
        estimator_manifest_sha256=estimator_binding["sha256"],
        output_dir=output,
        node_bin=node,
        plugin_root=subject._delivery.DEFAULT_PLUGIN_ROOT,
    )

    assert (
        validate_manifest_integrity(manifest) == manifest["integrity"]["payload_sha256"]
    )
    assert manifest["status"] == "complete"
    assert manifest["report_contract"]["surface_count"] == 1
    assert manifest["report_contract"]["primary_meeting_n"] == 128
    assert manifest["delivery_receipt"]["stages"]["validation"] == "passed"
    assert manifest["delivery_receipt"]["stages"]["package"] == "passed"
    assert manifest["delivery_receipt"]["stages"]["verification"] in {
        "passed",
        "structural_only",
    }
    assert {path.name for path in output.iterdir()} == {
        "artifact.json",
        "report.html",
        "manifest.json",
    }
    assert not os.access(output / "artifact.json", os.W_OK)
    assert output.stat().st_mode & 0o222 == 0
    assert (
        subject._sha256_file(output / "report.html")
        == manifest["artifacts"]["portable_report"]["sha256"]
    )
    html = (output / "report.html").read_text(encoding="utf-8")
    assert "data-analytics-portable-artifact-payload-source" in html
    assert subject.PAPER_STAGE_DISCLOSURE in html
    assert subject.IDENTITY_LINE_DISCLOSURE in html
    assert "N=128" in html
    assert "N=126" not in html
    assert (
        "All five paired gain intervals include zero; point gains are not stable "
        "evidence of improvement."
    ) in html
    assert "Raw and reference-SD values are displayed together" in html
    _make_writable(output)
