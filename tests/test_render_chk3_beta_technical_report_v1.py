from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from jobs.eval import estimate_chk3_beta_time_block_bootstrap_v1 as estimator
from jobs.eval import render_chk3_beta_technical_report_v1 as subject
from open_r1.validator.loo_generation_spec import validate_manifest_integrity


PRIMARY_SPECIFICATION = (
    "daily_release_association__dgs2_release_minus_previous_bp__"
    "current_lag_sentiment__vix__cyclic_month__hac4__"
    "shared_complete_core8__pre_external_reference_sd"
)


def _source_bindings(root: Path) -> dict[str, dict]:
    names = (
        "estimator_manifest",
        "estimates",
        "contrasts",
        "bootstrap_draws",
        "diagnostics",
        "market_panel_manifest",
        "sentiment_suite_manifest",
        "generation_suite_manifest",
    )
    result = {}
    for index, name in enumerate(names):
        path = root / "sealed" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"name": name, "index": index}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result[name] = subject._binding(path)
    return result


def _estimate(arm: str, point: float) -> dict:
    return {
        "estimate_id": f"primary::{arm}",
        "specification_id": PRIMARY_SPECIFICATION,
        "sample_id": "pre_external",
        "outcome_id": "dgs2_release_minus_previous_bp",
        "outcome_label": "2-year Treasury yield change (bp)",
        "backend_id": "distilbert_fomc_9c061b4_v1",
        "backend_label": "DistilBERT FOMC stance",
        "scale_id": "pre_external_reference_sd",
        "arm": arm,
        "point_estimate": point,
        "ci_lower": point - 0.02,
        "ci_upper": point + 0.02,
        "nobs": 126,
        "successful_bootstrap_draws": 10_000,
        "failed_bootstrap_draws": 0,
    }


def _contrast(
    contrast_id: str,
    family: str,
    before_arm: str,
    after_arm: str,
    point: float,
    *,
    low: float,
    high: float,
) -> dict:
    return {
        "contrast_row_id": f"primary::{contrast_id}",
        "contrast_id": contrast_id,
        "family": family,
        "multiplicity_family_id": (
            "distilbert_fomc_9c061b4_v1::pre_external__"
            "dgs2_release_minus_previous_bp__shared_complete_core8::"
            f"pre_external_reference_sd::{family}"
        ),
        "family_size": {
            "primary": 1,
            "background": 2,
            "reference_context": 4,
        }[family],
        "before_arm": before_arm,
        "after_arm": after_arm,
        "specification_id": PRIMARY_SPECIFICATION,
        "sample_id": "pre_external",
        "outcome_id": "dgs2_release_minus_previous_bp",
        "outcome_label": "2-year Treasury yield change (bp)",
        "backend_id": "distilbert_fomc_9c061b4_v1",
        "backend_label": "DistilBERT FOMC stance",
        "scale_id": "pre_external_reference_sd",
        "point_estimate": point,
        "ci_lower": low,
        "ci_upper": high,
        "p_value": 0.04,
        "holm_adjusted_p": 0.08,
        "nobs": 126,
        "successful_bootstrap_draws": 10_000,
        "failed_bootstrap_draws": 0,
    }


def _report_model(root: Path) -> dict:
    estimates = [
        _estimate("reference", -0.010000),
        _estimate("chk0", -0.020000),
        _estimate("chk1", -0.022940),
        _estimate("chk3", -0.018991),
    ]
    contrasts = [
        _contrast(
            "chk3_minus_chk1",
            "primary",
            "chk1",
            "chk3",
            0.003949,
            low=-0.005000,
            high=0.012898,
        ),
        _contrast(
            "chk1_minus_chk0",
            "background",
            "chk0",
            "chk1",
            -0.002940,
            low=-0.015000,
            high=0.009120,
        ),
        _contrast(
            "chk3_minus_chk0",
            "background",
            "chk0",
            "chk3",
            0.001009,
            low=-0.012000,
            high=0.014018,
        ),
        _contrast(
            "chk0_minus_reference",
            "reference_context",
            "reference",
            "chk0",
            -0.010000,
            low=-0.025000,
            high=0.005000,
        ),
        _contrast(
            "chk1_minus_reference",
            "reference_context",
            "reference",
            "chk1",
            -0.012940,
            low=-0.028000,
            high=0.002120,
        ),
        _contrast(
            "chk3_minus_reference",
            "reference_context",
            "reference",
            "chk3",
            -0.008991,
            low=-0.024000,
            high=0.006018,
        ),
        _contrast(
            "reference_distance_gain_chk3_vs_chk1",
            "reference_context",
            "chk1",
            "chk3",
            0.003949,
            low=-0.006000,
            high=0.013898,
        ),
    ]
    view_nobs = {
        "pre_external": 126,
        "full": 254,
        "post": 127,
        "full_excluding_cp318_selection": 243,
        "post_excluding_cp318_selection": 116,
    }
    backend_labels = {
        "distilbert_fomc_9c061b4_v1": "DistilBERT FOMC stance",
        "prosus_finbert_4556d13_v1": "ProsusAI FinBERT financial valence",
    }
    for backend_index, (backend_id, backend_label) in enumerate(backend_labels.items()):
        for view_index, (view_id, nobs) in enumerate(view_nobs.items()):
            if backend_id == "distilbert_fomc_9c061b4_v1" and view_id == "pre_external":
                continue
            chk1_point = 0.02 + backend_index * 0.01 + view_index * 0.001
            difference = 0.001 + backend_index * 0.002 + view_index * 0.0001
            chk3_point = chk1_point + difference
            prefix = f"summary::{backend_id}::{view_id}"
            for arm, point in (("chk1", chk1_point), ("chk3", chk3_point)):
                estimates.append(
                    {
                        "estimate_id": f"{prefix}::{arm}",
                        "specification_id": PRIMARY_SPECIFICATION,
                        "sample_id": view_id,
                        "outcome_id": "dgs2_release_minus_previous_bp",
                        "outcome_label": "2-year Treasury yield change (bp)",
                        "backend_id": backend_id,
                        "backend_label": backend_label,
                        "scale_id": "pre_external_reference_sd",
                        "arm": arm,
                        "point_estimate": point,
                        "ci_lower": point - 0.02,
                        "ci_upper": point + 0.02,
                        "nobs": nobs,
                        "successful_bootstrap_draws": 10_000,
                        "failed_bootstrap_draws": 0,
                    }
                )
            contrasts.append(
                {
                    "contrast_row_id": f"{prefix}::chk3_minus_chk1",
                    "contrast_id": "chk3_minus_chk1",
                    "family": "primary",
                    "multiplicity_family_id": (
                        f"{backend_id}::{view_id}__dgs2_release_minus_previous_bp__"
                        "shared_complete_core8::pre_external_reference_sd::primary"
                    ),
                    "family_size": 1,
                    "before_arm": "chk1",
                    "after_arm": "chk3",
                    "specification_id": PRIMARY_SPECIFICATION,
                    "sample_id": view_id,
                    "outcome_id": "dgs2_release_minus_previous_bp",
                    "outcome_label": "2-year Treasury yield change (bp)",
                    "backend_id": backend_id,
                    "backend_label": backend_label,
                    "scale_id": "pre_external_reference_sd",
                    "point_estimate": difference,
                    "ci_lower": difference - 0.01,
                    "ci_upper": difference + 0.01,
                    "p_value": 0.25,
                    "holm_adjusted_p": 0.25,
                    "nobs": nobs,
                    "successful_bootstrap_draws": 10_000,
                    "failed_bootstrap_draws": 0,
                }
            )
    return {
        "generated_at_utc": "2026-08-17T12:00:00Z",
        "study": {
            "study_id": "chk3-beta-core8-estimator-v1",
            "title": "CHK3 Sentiment–Treasury Association Study",
            "meeting_date_start": "1993-02-03",
            "meeting_date_end": "2024-12-18",
            "release_date_start": "1993-03-26",
            "release_date_end": "2025-02-19",
            "arm_order": list(subject.ARM_ORDER),
            "primary_outcome_id": "dgs2_release_minus_previous_bp",
            "primary_backend_id": "distilbert_fomc_9c061b4_v1",
            "primary_sample_id": "pre_external",
            "primary_scale_id": "pre_external_reference_sd",
        },
        "backend_rows": [
            {
                "backend_id": "distilbert_fomc_9c061b4_v1",
                "backend_label": "DistilBERT FOMC stance",
                "role": "primary",
                "regression_eligible": True,
                "reference_scale_sd": 0.001254246515242593,
                "exclusion_reason": None,
            },
            {
                "backend_id": "prosus_finbert_4556d13_v1",
                "backend_label": "ProsusAI FinBERT financial valence",
                "role": "robustness_nonpooled_construct",
                "regression_eligible": True,
                "reference_scale_sd": 0.030700216259587926,
                "exclusion_reason": None,
            },
            {
                "backend_id": "lucca_trebbi_hawk_dove_lexicon_v1",
                "backend_label": "Lucca–Trebbi-inspired hawk/dove lexicon",
                "role": "diagnostic_only",
                "regression_eligible": False,
                "reference_scale_sd": 0.0,
                "exclusion_reason": "zero_reference_variance",
            },
        ],
        "reporting": {
            "headline_estimate_id": "primary::chk3",
            "headline_contrast_row_id": "primary::chk3_minus_chk1",
            "primary_estimate_ids": [
                "primary::reference",
                "primary::chk0",
                "primary::chk1",
                "primary::chk3",
            ],
            "primary_contrast_row_ids": ["primary::chk3_minus_chk1"],
            "background_contrast_row_ids": [
                "primary::chk1_minus_chk0",
                "primary::chk3_minus_chk0",
            ],
            "reference_contrast_row_ids": [
                "primary::chk0_minus_reference",
                "primary::chk1_minus_reference",
                "primary::chk3_minus_reference",
                "primary::reference_distance_gain_chk3_vs_chk1",
            ],
        },
        "methodology": {
            "estimand": (
                "Change in the outcome associated with a one-unit increase in "
                "meeting-level sentiment."
            ),
            "regression_formula": (
                "dgs2_release_minus_previous_bp ~ 1 + current_sentiment + "
                "lag_sentiment + vix_log_change + month_sin + month_cos"
            ),
            "outcome_definition": (
                "release-day DGS2 close minus the previous valid business-day close, "
                "in basis points"
            ),
            "sentiment_definition": (
                "equal-weight Core8 meeting score; synthetic arms average five "
                "paired stochastic replicates"
            ),
            "covariance_estimator": "Newey–West HAC",
            "hac_lag": 4,
            "bootstrap_unit": "calendar-year time block",
            "within_meeting_resampling": (
                "observed-size n_i paired replicate draws for shared-complete and "
                "fixed K=5 for neutral imputation, shared by CHK0, CHK1, and CHK3"
            ),
            "duplicate_calendar_block_replicate_semantics": (
                "duplicate occurrences of a sampled year reuse the same meeting-level "
                "replicate resample within the draw"
            ),
            "bootstrap_draws": 10_000,
            "bootstrap_seed": 20_260_817,
            "same_sample_design": True,
            "lag_construction": (
                "lagged sentiment is computed once in original chronology before "
                "calendar-year block resampling"
            ),
            "seasonality_controls": [
                "sin(2*pi*release_month/12)",
                "cos(2*pi*release_month/12)",
            ],
            "bootstrap_p_value": (
                "plus-one-corrected two-sided descriptive bootstrap sign-tail "
                "probability around zero, not a null-centered hypothesis test; Holm "
                "adjustment separately within backend x analysis cell x score scale "
                "x frozen contrast family; robustness inference is exploratory"
            ),
            "sensitivity_designs": [
                "DGS5 and DGS10 same-window outcomes",
                "release close to next-business-day close",
                "pre-release placebo window",
            ],
        },
        "sample_rows": [
            {
                "sample_id": sample_id,
                "label": label,
                "role": role,
                "meeting_rows": meeting_rows,
                "regression_nobs": nobs,
                "meeting_date_start": start,
                "meeting_date_end": end,
                "exclusion_count": excluded,
            }
            for sample_id, label, role, meeting_rows, nobs, start, end, excluded in (
                (
                    "pre_external",
                    "Pre-2009 external panel",
                    "primary exploratory",
                    127,
                    126,
                    "1993-02-03",
                    "2008-12-16",
                    1,
                ),
                (
                    "full",
                    "Full common panel",
                    "descriptive",
                    255,
                    254,
                    "1993-02-03",
                    "2024-12-18",
                    1,
                ),
                (
                    "post",
                    "Post-2008 panel",
                    "internal sensitivity",
                    128,
                    127,
                    "2009-01-28",
                    "2024-12-18",
                    0,
                ),
                (
                    "full_excluding_cp318_selection",
                    "Full excluding cp318 selection exposure",
                    "post-selection sensitivity",
                    244,
                    243,
                    "1993-02-03",
                    "2024-12-18",
                    12,
                ),
                (
                    "post_excluding_cp318_selection",
                    "Post excluding cp318 selection exposure",
                    "post-selection sensitivity",
                    117,
                    116,
                    "2009-01-28",
                    "2024-12-18",
                    12,
                ),
            )
        ],
        "estimates": estimates,
        "contrasts": contrasts,
        "limitations": [
            "Daily data cannot isolate intraday Minutes-release surprises.",
            "The full sample reuses nine cp318-selection-exposed meetings.",
        ],
        "source_bindings": _source_bindings(root),
    }


def _make_writable(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        child.chmod(0o755 if child.is_dir() else 0o600)
    path.chmod(0o755)


def test_canonical_artifact_is_answer_first_source_backed_and_noncausal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    model = _report_model(tmp_path)

    artifact = subject.build_canonical_artifact(model)

    assert artifact["surface"] == "report"
    assert artifact["snapshot"]["status"] == "ready"
    assert artifact["snapshot"]["datasets"]["headline"][0][
        "headline_beta"
    ] == pytest.approx(-0.018991)
    assert artifact["snapshot"]["datasets"]["headline"][0][
        "headline_difference"
    ] == pytest.approx(0.003949)
    assert (
        artifact["snapshot"]["datasets"]["headline"][0]["headline_beta_display"]
        == "β=-0.018991"
    )
    assert (
        artifact["snapshot"]["datasets"]["headline"][0]["headline_difference_display"]
        == "Δβ=+0.003949"
    )
    body = "\n".join(block.get("body", "") for block in artifact["manifest"]["blocks"])
    assert subject.NON_CAUSAL_DISCLOSURE in body
    assert subject.REFERENCE_DISCLOSURE in body
    assert subject.EVENT_WINDOW_DISCLOSURE in body
    assert subject.POST_SELECTION_DISCLOSURE in body
    assert subject.LEXICON_DISCLOSURE in body
    assert "not official FOMC Minutes" in body
    assert "caused the observed Treasury-yield changes" in body
    assert all(not source["path"].startswith("/") for source in artifact["sources"])
    assert (
        artifact["snapshot"]["datasets"]["backend_audit"][-1]["reference_scale_sd"]
        == 0.0
    )
    backend_audit = artifact["snapshot"]["datasets"]["backend_audit"]
    assert backend_audit[0]["reference_scale_sd_display"] == "SD=0.00125424651524"
    assert backend_audit[-1]["reference_scale_sd_display"] == (
        "SD=0 (exact zero; excluded)"
    )

    five_view = artifact["snapshot"]["datasets"]["five_view_primary_contrasts"]
    assert len(five_view) == 10
    assert {row["sample"] for row in five_view} == set(subject.EXPECTED_VIEW_NOBS)
    assert {row["backend_id"] for row in five_view} == set(
        subject.REGRESSION_BACKEND_IDS
    )
    assert {row["score_policy"] for row in five_view} == {"shared_complete_core8"}
    assert next(
        row
        for row in five_view
        if row["backend_id"] == subject.PRIMARY_BACKEND_ID and row["sample"] == "full"
    )["difference"] == pytest.approx(0.0011)

    estimate_pages = [
        rows
        for name, rows in artifact["snapshot"]["datasets"].items()
        if name.startswith("all_estimates_page_")
    ]
    contrast_pages = [
        rows
        for name, rows in artifact["snapshot"]["datasets"].items()
        if name.startswith("all_contrasts_page_")
    ]
    assert estimate_pages and all(
        1 <= len(page) <= subject.APPENDIX_PAGE_SIZE for page in estimate_pages
    )
    assert contrast_pages and all(
        1 <= len(page) <= subject.APPENDIX_PAGE_SIZE for page in contrast_pages
    )
    assert {row["estimate_id"] for page in estimate_pages for row in page} == {
        row["estimate_id"] for row in model["estimates"]
    }
    assert {row["contrast_row_id"] for page in contrast_pages for row in page} == {
        row["contrast_row_id"] for row in model["contrasts"]
    }

    blocks = artifact["manifest"]["blocks"]
    block_ids = [block["id"] for block in blocks]
    assert block_ids.index("primary_estimate_finding") + 1 == block_ids.index(
        "primary_beta_chart_block"
    )
    assert block_ids.index("primary_beta_chart_block") + 1 == block_ids.index(
        "primary_estimates_table_block"
    )
    by_id = {block["id"]: block for block in blocks}
    assert by_id["technical_summary_estimate"]["sourceId"] == "estimates_source"
    assert by_id["technical_summary_contrast"]["sourceId"] == "contrasts_source"
    assert all(
        block.get("sourceId") == "sentiment_suite_source"
        for block in blocks
        if subject.LEXICON_DISCLOSURE in block.get("body", "")
    )
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
    assert "not a null-centered formal hypothesis test" in body
    assert body.count(subject.LEXICON_DISCLOSURE) == 2


def test_report_model_rejects_bad_interval_draw_accounting_and_contrast_math(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    model = _report_model(tmp_path)

    bad_interval = copy.deepcopy(model)
    bad_interval["estimates"][0]["ci_lower"] = 0.5
    with pytest.raises(subject.BetaTechnicalReportError, match="unordered"):
        subject.validate_report_model(bad_interval)

    bad_draws = copy.deepcopy(model)
    bad_draws["contrasts"][0]["successful_bootstrap_draws"] = 9_999
    with pytest.raises(subject.BetaTechnicalReportError, match="10,000"):
        subject.validate_report_model(bad_draws)

    bad_difference = copy.deepcopy(model)
    bad_difference["contrasts"][0]["point_estimate"] = 0.005
    with pytest.raises(subject.BetaTechnicalReportError, match="does not reconcile"):
        subject.validate_report_model(bad_difference)

    bad_distance = copy.deepcopy(model)
    next(
        row
        for row in bad_distance["contrasts"]
        if row["contrast_id"] == "reference_distance_gain_chk3_vs_chk1"
    )["point_estimate"] = -0.002
    with pytest.raises(subject.BetaTechnicalReportError, match="does not reconcile"):
        subject.validate_report_model(bad_distance)

    incomplete_primary = copy.deepcopy(model)
    incomplete_primary["reporting"]["primary_estimate_ids"].remove("primary::chk0")
    with pytest.raises(subject.BetaTechnicalReportError, match="all four text arms"):
        subject.validate_report_model(incomplete_primary)

    incomplete_reference = copy.deepcopy(model)
    incomplete_reference["reporting"]["reference_contrast_row_ids"].pop()
    with pytest.raises(subject.BetaTechnicalReportError, match="all frozen contextual"):
        subject.validate_report_model(incomplete_reference)

    lexicon_beta = copy.deepcopy(model)
    lexicon_beta["estimates"][0]["backend_id"] = "lucca_trebbi_hawk_dove_lexicon_v1"
    with pytest.raises(subject.BetaTechnicalReportError, match="identity drift"):
        subject.validate_report_model(lexicon_beta)

    lexicon_scale_drift = copy.deepcopy(model)
    lexicon_scale_drift["backend_rows"][-1]["reference_scale_sd"] = 0.01
    with pytest.raises(subject.BetaTechnicalReportError, match="eligibility/scale"):
        subject.validate_report_model(lexicon_scale_drift)


def test_authoritative_report_model_adapter_is_strict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    model = _report_model(tmp_path)

    assert subject._normalize_estimator({"report_model": model}) == model

    incomplete = copy.deepcopy(model)
    incomplete.pop("backend_rows")
    with pytest.raises(subject.BetaTechnicalReportError, match="inventory drift"):
        subject._normalize_estimator({"report_model": incomplete})


def test_estimator_runtime_preflight_rejects_numpy_drift_before_deep_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "runtime": {
                    "python_version": "3.11.5",
                    "numpy_version": "1.24.3",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        subject,
        "_current_runtime_versions",
        lambda: {"python_version": "3.11.5", "numpy_version": "1.26.4"},
    )

    with pytest.raises(
        subject.BetaTechnicalReportError, match="numpy_version runtime drift"
    ):
        subject.load_report_model(manifest_path, subject._sha256_file(manifest_path))


def test_estimator_runtime_preflight_rejects_thread_drift_before_deep_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "runtime": {
                    "python_version": "3.11.5",
                    "numpy_version": "1.24.3",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        subject,
        "_current_runtime_versions",
        lambda: {"python_version": "3.11.5", "numpy_version": "1.24.3"},
    )
    observed = dict(subject.REQUIRED_THREAD_ENV)
    observed["OPENBLAS_NUM_THREADS"] = "32"
    monkeypatch.setattr(subject, "_current_thread_runtime", lambda: observed)

    with pytest.raises(subject.BetaTechnicalReportError, match="thread runtime drift"):
        subject.load_report_model(manifest_path, subject._sha256_file(manifest_path))


def test_estimator_report_model_contract_integrates_with_renderer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    expected = _report_model(tmp_path)
    sample_specs = [
        {
            "sample_id": row["sample_id"],
            "label": row["label"],
            "role": row["role"],
            "market_outcome_rows": row["meeting_rows"],
            "regression_nobs": row["regression_nobs"],
            "meeting_date_start": row["meeting_date_start"],
            "meeting_date_end": row["meeting_date_end"],
            "release_date_start": "1993-03-26",
            "release_date_end": "2025-02-19",
            "exclusion_count": row["exclusion_count"],
        }
        for row in expected["sample_rows"]
    ]
    raw_estimates = [
        {**row, "view_id": row["sample_id"]} for row in expected["estimates"]
    ]
    raw_contrasts = [
        {**row, "view_id": row["sample_id"]} for row in expected["contrasts"]
    ]
    source_bindings = expected["source_bindings"]
    manifest = {
        "created_at_utc": expected["generated_at_utc"],
        "sample_specs": sample_specs,
        "reporting": expected["reporting"],
        "limitations": expected["limitations"],
        "reference_scales": {
            estimator.sentiment.DISTIL_BACKEND: {"standard_deviation": 0.21},
            estimator.sentiment.FINBERT_BACKEND: {"standard_deviation": 0.13},
        },
        "analysis_contract": {
            "regression_formula": expected["methodology"]["regression_formula"],
            "lag_sentiment_source": expected["methodology"]["lag_construction"],
            "seasonality_controls": expected["methodology"]["seasonality_controls"],
            "excluded_lexicon": {
                "pre_external_reference_score_standard_deviation_ddof1": 0.0
            },
        },
        "artifacts": {
            name: source_bindings[name]
            for name in ("estimates", "contrasts", "bootstrap_draws", "diagnostics")
        },
        "inputs": {
            "market_panel_manifest": source_bindings["market_panel_manifest"],
            "sentiment_suite_manifest": source_bindings["sentiment_suite_manifest"],
            "generation_suite_manifest": source_bindings["generation_suite_manifest"],
        },
    }

    actual = estimator._report_model(
        manifest=manifest,
        manifest_binding=source_bindings["estimator_manifest"],
        estimates=raw_estimates,
        contrasts=raw_contrasts,
    )

    validated = subject._normalize_estimator({"report_model": actual})
    assert validated["study"]["primary_backend_id"] == subject.PRIMARY_BACKEND_ID
    assert validated["backend_rows"][-1]["reference_scale_sd"] == 0.0


def test_incomplete_model_does_not_create_final_report_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    model = _report_model(tmp_path)
    model["reporting"].pop("headline_estimate_id")
    estimator_binding = model["source_bindings"]["estimator_manifest"]
    monkeypatch.setattr(
        subject,
        "load_report_model",
        lambda *_args, **_kwargs: (copy.deepcopy(model), estimator_binding),
    )
    output = tmp_path / "technical_report_v1"

    with pytest.raises(subject.BetaTechnicalReportError, match="selector contract"):
        subject.render_and_seal_report(
            estimator_manifest=Path(estimator_binding["path"]),
            estimator_manifest_sha256=estimator_binding["sha256"],
            output_dir=output,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".technical_report_v1.staging-*"))


def test_complete_fixture_packages_once_and_seals_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    node = repository_root / ".cache/report_node/bin/node"
    if not node.is_file():
        pytest.skip("frozen local Node runtime unavailable")
    if not subject.DEFAULT_PLUGIN_ROOT.is_dir():
        pytest.skip("packaged Data Analytics report builder unavailable")
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    model = _report_model(tmp_path)
    estimator_binding = model["source_bindings"]["estimator_manifest"]
    monkeypatch.setattr(
        subject,
        "load_report_model",
        lambda *_args, **_kwargs: (copy.deepcopy(model), estimator_binding),
    )
    output = tmp_path / "technical_report_v1"

    manifest = subject.render_and_seal_report(
        estimator_manifest=Path(estimator_binding["path"]),
        estimator_manifest_sha256=estimator_binding["sha256"],
        output_dir=output,
        node_bin=node,
        plugin_root=subject.DEFAULT_PLUGIN_ROOT,
    )

    assert (
        validate_manifest_integrity(manifest) == manifest["integrity"]["payload_sha256"]
    )
    assert manifest["status"] == "complete"
    assert manifest["delivery_receipt"]["stages"]["validation"] == "passed"
    assert manifest["delivery_receipt"]["stages"]["package"] == "passed"
    assert manifest["delivery_receipt"]["stages"]["verification"] in {
        "passed",
        "structural_only",
    }
    assert (output / "artifact.json").is_file()
    assert (output / "report.html").is_file()
    assert (output / "manifest.json").is_file()
    assert not os.access(output / "artifact.json", os.W_OK)
    assert output.stat().st_mode & 0o222 == 0
    assert (
        subject._sha256_file(output / "report.html")
        == manifest["artifacts"]["portable_report"]["sha256"]
    )
    html = (output / "report.html").read_text(encoding="utf-8")
    assert "data-analytics-portable-artifact-payload-source" in html
    assert subject.NON_CAUSAL_DISCLOSURE in html
    assert subject.REFERENCE_DISCLOSURE in html
    assert "β=-0.018991" in html
    assert "Δβ=+0.003949" in html
    assert "SD=0.00125424651524" in html
    assert "SD=0 (exact zero; excluded)" in html
    assert all(view_id in html for view_id in subject.EXPECTED_VIEW_NOBS)
    assert "Showing first" not in html
    _make_writable(output)
