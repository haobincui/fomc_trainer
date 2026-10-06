from __future__ import annotations

import copy

import numpy as np
import pytest

from jobs.eval import estimate_chk3_beta_time_block_bootstrap_v1 as estimator


def _release(
    meeting_id: str,
    release_date: str,
    *,
    era: str = "pre_external",
    exposed: bool = False,
) -> dict:
    return {
        "generation_meeting_id": meeting_id,
        "meeting_end_date": meeting_id,
        "release_date": release_date,
        "era": era,
        "cp318_selection_exposed": exposed,
    }


def _market(release: dict, value: float) -> dict:
    return {
        **release,
        "in_full_sample": True,
        "in_pre_sample": release["era"] == "pre_external",
        "in_post_sample": release["era"] == "post",
        "dgs2_release_minus_previous_bp": value,
        "dgs5_release_minus_previous_bp": value + 1.0,
        "dgs10_release_minus_previous_bp": value + 2.0,
        "dgs2_next_minus_release_bp": value + 3.0,
        "dgs2_placebo_previous_minus_previous_2_bp": value + 4.0,
        "vix_release_minus_previous_log_pct": value / 10.0,
        "vix_next_minus_release_log_pct": value / 11.0,
        "vix_placebo_previous_minus_previous_2_log_pct": value / 12.0,
    }


def test_frozen_regression_contract_excludes_zero_variance_lexicon() -> None:
    assert estimator.PRIMARY_BACKEND == estimator.sentiment.DISTIL_BACKEND
    assert estimator.REGRESSION_BACKEND_ORDER == (
        estimator.sentiment.DISTIL_BACKEND,
        estimator.sentiment.FINBERT_BACKEND,
    )
    assert estimator.sentiment.LEXICON_BACKEND not in estimator.REGRESSION_BACKEND_ORDER
    assert estimator.BOOTSTRAP_DRAWS == 10_000
    assert estimator.BOOTSTRAP_SEED == 20_260_817
    assert estimator.HAC_LAG == 4
    assert len(estimator._analysis_inventory()) == 14


def test_true_previous_fomc_lag_survives_missing_market_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # B is the 2004-09-21 analogue: no market row, but it must remain C's lag.
    releases = [
        _release("A", "2004-09-01"),
        _release("B", "2004-11-11"),
        _release("C", "2004-12-16"),
        _release("D", "2005-02-01"),
    ]
    market_rows = [_market(releases[index], float(index)) for index in (0, 2, 3)]
    monkeypatch.setattr(estimator.sentiment.data_contract, "EXPECTED_MEETINGS", 4)
    monkeypatch.setitem(estimator.EXPECTED_VIEW_NOBS, "pre_external", 2)
    view = estimator._make_design_view(
        market_rows=market_rows,
        release_rows=releases,
        view_id="pre_external",
    )
    assert view.current_meeting_ids == ("C", "D")
    assert view.lag_meeting_ids == ("B", "C")
    assert view.meeting_date_start == "C"
    assert view.nobs == 2


def test_selection_exclusion_uses_true_current_and_lag_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    releases = [
        _release("A", "2022-01-01"),
        _release("B", "2022-02-01", exposed=True),
        _release("C", "2022-03-01"),
        _release("D", "2022-04-01"),
        _release("E", "2022-05-01"),
    ]
    market_rows = [_market(row, float(index)) for index, row in enumerate(releases)]
    monkeypatch.setattr(estimator.sentiment.data_contract, "EXPECTED_MEETINGS", 5)
    monkeypatch.setitem(
        estimator.EXPECTED_VIEW_NOBS, "full_excluding_cp318_selection", 2
    )
    view = estimator._make_design_view(
        market_rows=market_rows,
        release_rows=releases,
        view_id="full_excluding_cp318_selection",
    )
    assert view.lag_meeting_ids == ("C", "D")
    assert view.current_meeting_ids == ("D", "E")


def test_window_specific_vix_controls_and_cyclic_seasonality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    releases = [_release(f"M{index}", f"2000-{index + 1:02d}-15") for index in range(8)]
    market_rows = [_market(row, float(index + 1)) for index, row in enumerate(releases)]
    monkeypatch.setattr(estimator.sentiment.data_contract, "EXPECTED_MEETINGS", 8)
    monkeypatch.setitem(estimator.EXPECTED_VIEW_NOBS, "pre_external", 7)
    view = estimator._make_design_view(
        market_rows=market_rows,
        release_rows=releases,
        view_id="pre_external",
        outcome_id="dgs2_next_minus_release_bp",
    )
    assert view.vix_control_id == "vix_next_minus_release_log_pct"
    scale = estimator.ReferenceScale("backend", 0.0, 1.0, 128, 1, tuple())
    matrix, columns = estimator._design_matrix(
        view,
        np.arange(7, dtype=np.float64),
        np.arange(7, dtype=np.float64) - 1.0,
        scale=scale,
    )
    assert matrix.shape == (7, 6)
    assert columns[-2:] == ("release_month_sin", "release_month_cos")
    assert not any("month_0" in column for column in columns)


def _meeting_score_rows() -> list[dict]:
    rows: list[dict] = [
        {
            "meeting_id": "M",
            "arm": "reference",
            "replicate_id": None,
            "complete_core8": True,
            "score": 0.25,
            "neutral_imputed_score": 0.25,
            "cp318_selection_exposed": False,
        }
    ]
    for arm in estimator.SYNTHETIC_ARMS:
        for replicate_id in range(5):
            incomplete = arm == "chk3" and replicate_id == 2
            rows.append(
                {
                    "meeting_id": "M",
                    "arm": arm,
                    "replicate_id": replicate_id,
                    "complete_core8": not incomplete,
                    "score": None if incomplete else replicate_id / 10.0,
                    "neutral_imputed_score": (
                        0.0 if incomplete else replicate_id / 10.0
                    ),
                    "cp318_selection_exposed": False,
                }
            )
    return rows


def test_shared_complete_and_neutral_fixed_k5_are_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(estimator.sentiment.data_contract, "EXPECTED_MEETINGS", 1)
    loaded = {
        "manifest": {
            "backend": estimator.sentiment.DISTIL_BACKEND,
            "construct": "stance",
        },
        "meeting_scores": _meeting_score_rows(),
    }
    scores = estimator._backend_scores(loaded)
    assert scores.paired_replicates_by_meeting["M"] == (0, 1, 3, 4)
    assert set(scores.neutral_synthetic_by_meeting["chk3"]["M"]) == set(range(5))


def test_reference_distance_gain_is_direct_closeness_not_after_minus_before() -> None:
    betas = {"reference": 1.0, "chk0": -2.0, "chk1": 4.0, "chk3": 2.0}
    contrasts = estimator._contrast_values(betas)
    assert contrasts["chk3_minus_chk1"] == -2.0
    assert contrasts[estimator.DERIVED_CONTRAST_ID] == 2.0


def test_lexicon_exclusion_requires_exact_zero_variance_evidence() -> None:
    evidence = {
        "backend_id": estimator.sentiment.LEXICON_BACKEND,
        "used_for_regression": False,
        "reason": "zero_reference_variance",
        "reference_topic_rows": 2_048,
        "reference_meeting_rows": 256,
        "pre_external_reference_meeting_rows": 128,
        "reference_matched_topic_rows": 0,
        "reference_total_matches": 0,
        "pre_external_reference_score_standard_deviation_ddof1": 0.0,
        "diagnostic_only": True,
    }
    loaded = {"excluded_backends": {estimator.sentiment.LEXICON_BACKEND: evidence}}
    assert estimator._lexicon_exclusion(loaded) == evidence
    altered = copy.deepcopy(loaded)
    altered["excluded_backends"][estimator.sentiment.LEXICON_BACKEND][
        "pre_external_reference_score_standard_deviation_ddof1"
    ] = 0.01
    with pytest.raises(estimator.BetaEstimatorError, match="zero-variance"):
        estimator._lexicon_exclusion(altered)


def test_exact_bootstrap_replay_rejects_consistently_resealed_tamper() -> None:
    expected = {
        "schema_version": estimator.DRAW_SCHEMA,
        "analysis_id": "analysis",
        "backend_id": estimator.sentiment.DISTIL_BACKEND,
        "view_id": "pre_external",
        "outcome_id": estimator.OUTCOME_ID,
        "score_policy": estimator.PRIMARY_SCORE_POLICY,
        "draw_index": 0,
        "plan_sha256": "a" * 64,
        "status": "ok",
        "failure_reason": None,
        "sampled_regression_rows": 126,
        "betas": {
            estimator.PRIMARY_SCALE: {
                "reference": 1.0,
                "chk0": 2.0,
                "chk1": 3.0,
                "chk3": 4.0,
            },
            "raw_score": {
                "reference": 0.1,
                "chk0": 0.2,
                "chk1": 0.3,
                "chk3": 0.4,
            },
        },
    }
    expected["contrasts"] = {
        scale_id: estimator._contrast_values(betas)
        for scale_id, betas in expected["betas"].items()
    }
    tampered = copy.deepcopy(expected)
    tampered["betas"][estimator.PRIMARY_SCALE]["chk3"] = 4.25
    tampered["contrasts"][estimator.PRIMARY_SCALE] = estimator._contrast_values(
        tampered["betas"][estimator.PRIMARY_SCALE]
    )

    # A validator that only checked summaries/arithmetic could accept this row after
    # all dependent artifacts were consistently rebuilt and re-sealed.  Exact replay
    # from the frozen plan/input data must still reject it.
    with pytest.raises(
        estimator.BetaEstimatorError, match="bootstrap deterministic replay drift"
    ):
        estimator._require_exact_bootstrap_replay(
            [tampered],
            [expected],
            analysis_key=(estimator.sentiment.DISTIL_BACKEND, "analysis"),
        )

    estimator._require_exact_bootstrap_replay(
        [expected],
        [copy.deepcopy(expected)],
        analysis_key=(estimator.sentiment.DISTIL_BACKEND, "analysis"),
    )


def test_import_time_source_binding_rejects_runtime_drift() -> None:
    binding = estimator._binding(estimator.Path(estimator.statistics.__file__))
    altered = copy.deepcopy(binding)
    altered["sha256"] = "0" * 64
    with pytest.raises(
        estimator.BetaEstimatorError, match="implementation source changed after import"
    ):
        estimator._require_import_binding_unchanged("statistical_primitives", altered)
