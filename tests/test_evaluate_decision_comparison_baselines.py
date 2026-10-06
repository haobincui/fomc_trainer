from __future__ import annotations

import copy
from collections import Counter
from datetime import date

import pytest

from jobs.eval import evaluate_decision_comparison_baselines as baselines


def _frozen_config() -> dict[str, object]:
    return baselines.load_config(baselines.DEFAULT_CONFIG)


def _feature_row() -> dict[str, object]:
    retrieval = "2026-08-27T00:00:00Z"
    return {
        "meeting_id": "synthetic-meeting",
        "meeting_start_date": "2024-01-31",
        "evidence_cutoff": "2024-01-30",
        "market_observation_cutoff": "2024-01-29",
        "features": {
            "tbill6m_minus_effr_pp": -0.25,
            "unemployment_rate_pct": 3.7,
            "real_gdp_growth_annualized_pct": 2.1,
        },
        "observations": {
            "UNRATE": {
                "observation_date": "2023-12-01",
                "availability_as_of_date": "2024-01-30",
                "source_sha256": "unrate-source",
                "retrieved_at_utc": retrieval,
            },
            "GDPC1": {
                "observation_dates": ["2023-04-01", "2023-07-01"],
                "availability_as_of_date": "2024-01-30",
                "source_sha256": "gdpc1-source",
                "retrieved_at_utc": retrieval,
            },
            "DTB6": {
                "observation_date": "2024-01-26",
                "availability_as_of_date": "2024-01-29",
                "source_sha256": "dtb6-source",
                "retrieved_at_utc": retrieval,
            },
            "DFF": {
                "observation_date": "2024-01-26",
                "availability_as_of_date": "2024-01-29",
                "source_sha256": "dff-source",
                "retrieved_at_utc": retrieval,
            },
        },
        "hamilton_jorda_state": {
            "coverage_status": "unavailable",
            "unavailable_reason": "DFEDTAR_single_target_history_ends_before_evidence_cutoff:2008-12-15",
        },
    }


def _synthetic_panel_meetings() -> list[dict[str, object]]:
    meetings: list[dict[str, object]] = []
    panel_directions = {
        "historical_n19": ["cut"] * 4 + ["hold"] * 10 + ["hike"] * 5,
        "postcutoff_n12": ["cut"] * 3 + ["hold"] * 9,
    }
    for panel, directions in panel_directions.items():
        for index, direction in enumerate(directions):
            meetings.append(
                {
                    "sample_id": f"{panel}-{index:02d}",
                    "panel": panel,
                    "meeting_id": f"{panel}-meeting-{index:02d}",
                    "meeting_start_date": "2024-01-31",
                    "meeting_end_date": "2024-01-31",
                    "evidence_cutoff": "2024-01-30",
                    "direction": direction,
                    "magnitude_bp": 0 if direction == "hold" else 25,
                }
            )
    return meetings


def _signed_target_action(meeting: dict[str, object]) -> int:
    magnitude = int(meeting["magnitude_bp"])
    if meeting["direction"] == "cut":
        return -magnitude
    if meeting["direction"] == "hike":
        return magnitude
    return 0


def _synthetic_predictions() -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []
    for index, meeting in enumerate(_synthetic_panel_meetings()):
        invalid_count = 2 if index == 0 else 0
        invalid_probability = invalid_count / 10
        predictions.append(
            baselines._available_prediction(
                meeting,
                model="model_chk0",
                comparison_contract="matched_roster",
                fit_contract="frozen_paper_checkpoint_k10",
                action_probabilities={
                    _signed_target_action(meeting): 1.0 - invalid_probability
                },
                invalid_probability=invalid_probability,
                source_hashes={"synthetic_paper_source": "hash"},
                extra={
                    "generation_count": 10,
                    "generation_invalid_count": invalid_count,
                    "generation_invalid_rate": invalid_probability,
                    "generation_invalid_definition": "synthetic extractability contract",
                    "strict_contract_invalid_count": invalid_count,
                    "strict_contract_invalid_rate": invalid_probability,
                    "strict_contract_invalid_definition": "synthetic strict contract",
                    "invalid_is_coverage_failure": False,
                    "prediction_seed": None,
                    "prediction_seed_source": "frozen_paper_result_rows",
                },
            )
        )
        if index == 0:
            predictions.append(
                baselines._unavailable_prediction(
                    meeting,
                    model="always_hold",
                    comparison_contract="expanding_window",
                    fit_contract="no_fit_constant",
                    reason="insufficient_prior_classes",
                )
            )
        else:
            predictions.append(
                baselines._available_prediction(
                    meeting,
                    model="always_hold",
                    comparison_contract="expanding_window",
                    fit_contract="no_fit_constant",
                    action_probabilities={0: 1.0},
                )
            )
    return predictions


def test_frozen_action_roster_closes_and_lag1_uses_the_full_chronology() -> None:
    roster = baselines.build_action_roster(_frozen_config())

    assert len(roster) == len({row["meeting_id"] for row in roster}) == 268
    assert Counter(row["role"] for row in roster) == {
        "train": 211,
        "validation": 13,
        "test": 13,
        "historical_n19": 19,
        "postcutoff_n12": 12,
    }
    assert Counter(row["direction"] for row in roster if row["role"] == "train") == {
        "cut": 23,
        "hold": 156,
        "hike": 32,
    }
    assert Counter(row["panel"] for row in roster if row["panel"] is not None) == {
        "historical_n19": 19,
        "postcutoff_n12": 12,
    }

    chronology = [(row["meeting_start_date"], row["meeting_id"]) for row in roster]
    assert chronology == sorted(chronology)
    assert roster[0]["lag1_meeting_id"] is None
    assert roster[0]["lag1_direction"] is None
    assert roster[0]["lag1_magnitude_bp"] is None
    for previous, current in zip(roster[:-1], roster[1:], strict=True):
        assert current["lag1_meeting_id"] == previous["meeting_id"]
        assert current["lag1_direction"] == previous["direction"]
        assert current["lag1_magnitude_bp"] == previous["magnitude_bp"]

    assert all(row["magnitude_bp"] == 0 for row in roster if row["direction"] == "hold")


@pytest.mark.parametrize("leaked_date", ["2024-01-31", "2024-02-01"])
def test_feature_information_set_rejects_same_day_and_future_observations(
    leaked_date: str,
) -> None:
    row = copy.deepcopy(_feature_row())
    row["observations"]["UNRATE"]["observation_date"] = leaked_date

    with pytest.raises(
        baselines.DecisionComparisonError,
        match="same-day/future UNRATE observation",
    ):
        baselines.validate_feature_information_set([row])


@pytest.mark.parametrize("leaked_date", ["2024-01-31", "2024-02-01"])
def test_feature_information_set_rejects_same_day_and_future_availability(
    leaked_date: str,
) -> None:
    row = copy.deepcopy(_feature_row())
    row["observations"]["UNRATE"]["availability_as_of_date"] = leaked_date

    with pytest.raises(
        baselines.DecisionComparisonError,
        match="future UNRATE availability",
    ):
        baselines.validate_feature_information_set([row])


def test_hamilton_jorda_state_uses_complete_prior_week_target_clock() -> None:
    meeting = {
        "meeting_start_date": "2020-01-28",
        "evidence_cutoff": "2020-01-27",
    }
    fred = {
        "DFEDTAR": [
            ("2020-01-01", 2.00, "target-hash"),
            ("2020-01-08", 1.75, "target-hash"),
            ("2020-01-15", 1.50, "target-hash"),
            ("2020-01-22", 1.50, "target-hash"),
            ("2020-01-27", 1.50, "target-hash"),
        ],
        "DFF": [
            ("2020-01-16", 1.50, "dff-hash"),
            ("2020-01-17", 1.50, "dff-hash"),
            ("2020-01-18", 99.00, "dff-hash"),
            ("2020-01-19", 99.00, "dff-hash"),
            ("2020-01-21", 1.50, "dff-hash"),
            ("2020-01-22", 1.50, "dff-hash"),
        ],
        "DTB6": [
            ("2020-01-16", 2.00, "dtb6-hash"),
            ("2020-01-17", 2.00, "dtb6-hash"),
            ("2020-01-18", 77.00, "dtb6-hash"),
            ("2020-01-19", 77.00, "dtb6-hash"),
            ("2020-01-21", 2.00, "dtb6-hash"),
            ("2020-01-22", 2.00, "dtb6-hash"),
        ],
    }
    state = baselines.build_hamilton_jorda_state(
        meeting,
        fred,
        market_lag_days=1,
    )
    assert state["coverage_status"] == "available"
    assert state["prior_week_end"] == "2020-01-22"
    assert state["previous_change_week_end"] == "2020-01-15"
    assert state["penultimate_change_week_end"] == "2020-01-08"
    assert state["previous_duration_weeks"] == 1
    assert state["previous_change_pp"] == pytest.approx(-0.25)
    assert state["spread_pp"] == pytest.approx(0.5)
    assert state["source_hashes"] == {
        "DFEDTAR": "target-hash",
        "DTB6": "dtb6-hash",
        "DFF": "dff-hash",
    }


def test_hamilton_jorda_weekly_spread_matches_archive_separate_means() -> None:
    """Reproduce the archive's -0.379 spread for the 1998-11-17 meeting week."""

    start = date(1998, 11, 5)
    end = date(1998, 11, 11)
    dff = baselines._weekly_mean(
        [
            ("1998-11-05", 5.05, "dff-hash"),
            ("1998-11-06", 4.64, "dff-hash"),
            ("1998-11-07", 4.64, "dff-hash"),
            ("1998-11-08", 4.64, "dff-hash"),
            ("1998-11-09", 4.95, "dff-hash"),
            ("1998-11-10", 4.84, "dff-hash"),
            ("1998-11-11", 4.84, "dff-hash"),
        ],
        start=start,
        end=end,
        availability_cutoff=end,
        series_id="DFF",
    )
    dtb6 = baselines._weekly_mean(
        [
            ("1998-11-05", 4.47, "dtb6-hash"),
            ("1998-11-06", 4.53, "dtb6-hash"),
            ("1998-11-09", 4.50, "dtb6-hash"),
            ("1998-11-10", 4.44, "dtb6-hash"),
        ],
        start=start,
        end=end,
        availability_cutoff=end,
        series_id="DTB6",
    )

    assert dff[0] == pytest.approx(4.864)
    assert dtb6[0] == pytest.approx(4.485)
    assert dtb6[0] - dff[0] == pytest.approx(-0.379)


def test_hamilton_jorda_state_is_unavailable_after_single_target_series_ends() -> None:
    state = baselines.build_hamilton_jorda_state(
        {
            "meeting_start_date": "2024-01-31",
            "evidence_cutoff": "2024-01-30",
        },
        {
            "DFEDTAR": [("2008-12-15", 1.0, "target-hash")],
            "DFF": [("2024-01-01", 1.0, "dff-hash")],
            "DTB6": [("2024-01-01", 1.0, "dtb6-hash")],
        },
        market_lag_days=1,
    )
    assert state["coverage_status"] == "unavailable"
    assert "DFEDTAR_single_target_history_ends" in state["unavailable_reason"]


def test_frozen_paper_generations_close_to_four_by_thirty_one_probabilities() -> None:
    predictions = baselines.aggregate_paper_predictions(_frozen_config())
    baselines.validate_prediction_rows(predictions)

    assert len(predictions) == 4 * 31
    assert Counter(row["model"] for row in predictions) == {
        "model_chk0": 31,
        "model_chk1_cp200": 31,
        "model_chk3_sft_cp38": 31,
        "model_chk3_grpo_cp450": 31,
    }
    assert len({row["meeting_id"] for row in predictions}) == 31
    assert len({(row["model"], row["meeting_id"]) for row in predictions}) == 124
    assert all(row["generation_count"] == 10 for row in predictions)
    for row in predictions:
        assert set(row["direction_probabilities"]) == {
            "cut",
            "hold",
            "hike",
            "invalid",
        }
        assert sum(row["direction_probabilities"].values()) == pytest.approx(1.0)
        assert sum(row["magnitude_probabilities"].values()) == pytest.approx(1.0)


def test_frozen_paper_exact_action_scores_close_to_generation_counts() -> None:
    predictions = baselines.aggregate_paper_predictions(_frozen_config())
    summary = baselines.score_predictions(
        predictions, bootstrap_draws=0, seed=20260827
    )
    expected = {
        "model_chk0": {"historical_n19": (40, 190), "postcutoff_n12": (16, 120)},
        "model_chk1_cp200": {
            "historical_n19": (31, 190),
            "postcutoff_n12": (17, 120),
        },
        "model_chk3_sft_cp38": {
            "historical_n19": (39, 190),
            "postcutoff_n12": (21, 120),
        },
        "model_chk3_grpo_cp450": {
            "historical_n19": (36, 190),
            "postcutoff_n12": (22, 120),
        },
    }
    for model, panels in expected.items():
        for panel, (correct, total) in panels.items():
            metric = summary["matched_roster"][panel]["models"][model][
                "magnitude_metrics"
            ]["mean_true_action_probability"]
            assert metric == pytest.approx(correct / total)


def test_frozen_paper_exact_action_scores_close_by_realized_direction() -> None:
    predictions = baselines.aggregate_paper_predictions(_frozen_config())
    summary = baselines.score_predictions(
        predictions, bootstrap_draws=0, seed=20260827
    )
    expected = {
        "model_chk0": {
            "historical_n19": {
                "cut": (3, 40),
                "hold": (24, 100),
                "hike": (13, 50),
            },
            "postcutoff_n12": {"cut": (2, 30), "hold": (14, 90)},
        },
        "model_chk1_cp200": {
            "historical_n19": {
                "cut": (3, 40),
                "hold": (18, 100),
                "hike": (10, 50),
            },
            "postcutoff_n12": {"cut": (7, 30), "hold": (10, 90)},
        },
        "model_chk3_sft_cp38": {
            "historical_n19": {
                "cut": (4, 40),
                "hold": (16, 100),
                "hike": (19, 50),
            },
            "postcutoff_n12": {"cut": (3, 30), "hold": (18, 90)},
        },
        "model_chk3_grpo_cp450": {
            "historical_n19": {
                "cut": (3, 40),
                "hold": (17, 100),
                "hike": (16, 50),
            },
            "postcutoff_n12": {"cut": (6, 30), "hold": (16, 90)},
        },
    }
    for model, panels in expected.items():
        for panel, directions in panels.items():
            model_metrics = summary["matched_roster"][panel]["models"][model]
            per_direction = model_metrics["magnitude_metrics"][
                "per_target_direction_mean_true_action_probability"
            ]
            for direction, (correct, total) in directions.items():
                assert per_direction[direction] == pytest.approx(correct / total)
            if panel == "postcutoff_n12":
                assert per_direction["hike"] is None


def test_frozen_paper_direction_only_scores_ignore_magnitude() -> None:
    predictions = baselines.aggregate_paper_predictions(_frozen_config())
    summary = baselines.score_predictions(
        predictions, bootstrap_draws=0, seed=20260827
    )
    expected = {
        "model_chk0": {
            "historical_n19": {
                "cut": (5, 40),
                "hold": (24, 100),
                "hike": (24, 50),
            },
            "postcutoff_n12": {"cut": (2, 30), "hold": (14, 90)},
        },
        "model_chk1_cp200": {
            "historical_n19": {
                "cut": (5, 40),
                "hold": (18, 100),
                "hike": (18, 50),
            },
            "postcutoff_n12": {"cut": (7, 30), "hold": (10, 90)},
        },
        "model_chk3_sft_cp38": {
            "historical_n19": {
                "cut": (7, 40),
                "hold": (16, 100),
                "hike": (33, 50),
            },
            "postcutoff_n12": {"cut": (3, 30), "hold": (18, 90)},
        },
        "model_chk3_grpo_cp450": {
            "historical_n19": {
                "cut": (4, 40),
                "hold": (17, 100),
                "hike": (27, 50),
            },
            "postcutoff_n12": {"cut": (6, 30), "hold": (16, 90)},
        },
    }
    for model, panels in expected.items():
        for panel, directions in panels.items():
            model_metrics = summary["matched_roster"][panel]["models"][model]
            per_direction = model_metrics["probability_metrics"][
                "per_target_direction_mean_true_direction_probability"
            ]
            generation_accounting = model_metrics["probability_metrics"][
                "per_target_direction_generation_accounting"
            ]
            for direction, (correct, total) in directions.items():
                assert per_direction[direction] == pytest.approx(correct / total)
                assert generation_accounting[direction][
                    "correct_direction_generation_count"
                ] == correct
                assert generation_accounting[direction]["generation_count"] == total
            if panel == "postcutoff_n12":
                assert per_direction["hike"] is None
                assert generation_accounting["hike"]["generation_count"] is None


def test_target_direction_coverage_uses_all_rows_not_only_available_rows() -> None:
    summary = baselines.score_predictions(
        _synthetic_predictions(), bootstrap_draws=0, seed=314159
    )
    historical = summary["expanding_window"]["historical_n19"]["models"][
        "always_hold"
    ]
    cut_coverage = historical["coverage"]["target_class_coverage"]["cut"]
    assert cut_coverage == {
        "total_meetings": 4,
        "available_meetings": 3,
        "unavailable_meetings": 1,
        "coverage_rate": 0.75,
    }
    postcutoff_hike = summary["expanding_window"]["postcutoff_n12"]["models"][
        "always_hold"
    ]["coverage"]["target_class_coverage"]["hike"]
    assert postcutoff_hike == {
        "total_meetings": 0,
        "available_meetings": 0,
        "unavailable_meetings": 0,
        "coverage_rate": None,
    }


def test_score_predictions_keeps_contracts_panels_and_failure_types_separate() -> None:
    summary = baselines.score_predictions(
        _synthetic_predictions(), bootstrap_draws=5, seed=314159
    )

    assert summary["schema_version"] == "decision-comparison-summary-v4"
    assert summary["pooled_result"] is None
    assert summary["pooling_prohibited"] is True
    for contract in ("matched_roster", "expanding_window"):
        assert set(summary[contract]) == {"historical_n19", "postcutoff_n12"}
        for panel, expected in {"historical_n19": 19, "postcutoff_n12": 12}.items():
            block = summary[contract][panel]
            assert block["expected_meetings"] == expected
            assert block["status"] == "scored"
            assert block["pooled_with_other_panel"] is False
            assert len(block["models"]) == 1
            metrics = next(iter(block["models"].values()))
            assert metrics["coverage"]["total_meetings"] == expected

    matched = summary["matched_roster"]["historical_n19"]["models"]["model_chk0"][
        "coverage"
    ]
    assert matched["available_meetings"] == 19
    assert matched["unavailable_meetings"] == 0
    assert matched["invalid_generation_count"] == 2
    assert matched["generation_count"] == 190
    matched_magnitude = summary["matched_roster"]["historical_n19"]["models"][
        "model_chk0"
    ]["magnitude_metrics"]
    assert matched_magnitude["mean_true_action_probability"] == pytest.approx(
        18.8 / 19
    )
    assert "mean_true_action_probability" in summary["matched_roster"][
        "historical_n19"
    ]["models"]["model_chk0"]["bootstrap_intervals"]

    expanding = summary["expanding_window"]["historical_n19"]["models"]["always_hold"][
        "coverage"
    ]
    assert expanding["available_meetings"] == 18
    assert expanding["unavailable_meetings"] == 1
    assert expanding["unavailable_reasons"] == {"insufficient_prior_classes": 1}
    assert expanding["invalid_generation_count"] == 0

    comparison = {
        (row["comparison_contract"], row["panel"], row["model"]): row
        for row in baselines.comparison_rows(summary)
    }
    matched_row = comparison[("matched_roster", "historical_n19", "model_chk0")]
    assert matched_row["cut_mean_true_action_probability"] == pytest.approx(0.95)
    assert matched_row["cut_mean_true_direction_probability"] == pytest.approx(
        0.95
    )
    assert matched_row["cut_available_meetings"] == 4
    assert matched_row["cut_total_meetings"] == 4
    assert matched_row["cut_score_ci_valid_draws"] == 5
    assert (
        matched_row[
            "cut_mean_true_direction_probability_ci_valid_draws"
        ]
        == 5
    )
    n12_row = comparison[("matched_roster", "postcutoff_n12", "model_chk0")]
    assert n12_row["hike_mean_true_action_probability"] is None
    assert n12_row["hike_mean_true_direction_probability"] is None
    assert n12_row["hike_available_meetings"] == 0
    assert n12_row["hike_total_meetings"] == 0
    assert n12_row["hike_score_ci_valid_draws"] is None

    for contract, model in (
        ("matched_roster", "model_chk0"),
        ("expanding_window", "always_hold"),
    ):
        n12 = summary[contract]["postcutoff_n12"]["models"][model]
        assert (
            n12["point_metrics"]["fixed_three_class_balanced_accuracy_secondary"]
            is None
        )
        assert (
            n12["probability_metrics"][
                "fixed_three_class_expected_balanced_accuracy_secondary"
            ]
            is None
        )


def test_mean_true_action_probability_does_not_use_a_modal_action() -> None:
    meeting = {
        **_synthetic_panel_meetings()[0],
        "direction": "hike",
        "magnitude_bp": 50,
    }
    row = baselines._available_prediction(
        meeting,
        model="always_hold",
        comparison_contract="matched_roster",
        fit_contract="synthetic_probability_distribution",
        action_probabilities={25: 0.7, 50: 0.3},
    )

    metrics = baselines._metric_block([row], panel="historical_n19")

    assert row["predicted_magnitude_bp"] == 25
    assert metrics["magnitude_metrics"]["exact_action_accuracy"] == 0.0
    assert metrics["magnitude_metrics"][
        "mean_true_action_probability"
    ] == pytest.approx(0.3)
    assert metrics["magnitude_metrics"][
        "per_target_direction_mean_true_action_probability"
    ] == {"cut": None, "hold": None, "hike": pytest.approx(0.3)}
    assert metrics["probability_metrics"][
        "per_target_direction_mean_true_direction_probability"
    ] == {"cut": None, "hold": None, "hike": pytest.approx(1.0)}


def test_direction_only_score_keeps_invalid_generations_in_k10_denominator() -> None:
    meeting = {
        **_synthetic_panel_meetings()[0],
        "direction": "hike",
        "magnitude_bp": 50,
    }
    row = baselines._available_prediction(
        meeting,
        model="model_chk0",
        comparison_contract="matched_roster",
        fit_contract="synthetic_k10_direction_distribution",
        action_probabilities={25: 0.4, -25: 0.3},
        invalid_probability=0.3,
        extra={
            "generation_count": 10,
            "generation_invalid_count": 3,
            "strict_contract_invalid_count": 3,
        },
    )

    metrics = baselines._metric_block([row], panel="historical_n19")

    assert metrics["probability_metrics"][
        "per_target_direction_mean_true_direction_probability"
    ]["hike"] == pytest.approx(0.4)
    assert metrics["probability_metrics"][
        "per_target_direction_generation_accounting"
    ]["hike"] == {
        "generation_count": 10,
        "correct_direction_generation_count": 4,
        "invalid_generation_count": 3,
        "strict_contract_invalid_generation_count": 3,
    }


def test_expanding_training_rows_excludes_current_and_future_actions() -> None:
    roster = baselines.build_action_roster(_frozen_config())
    targets = [row for row in roster if row["panel"] is not None]
    for target in targets:
        training = baselines.expanding_training_rows(roster, target)
        assert all(row["meeting_id"] != target["meeting_id"] for row in training)
        assert all(
            row["meeting_end_date"] < target["meeting_start_date"] for row in training
        )
    earliest = min(targets, key=lambda row: row["meeting_start_date"])
    eligibility = baselines.model_utils.expanding_window_eligibility(
        [
            row["direction"]
            for row in baselines.expanding_training_rows(roster, earliest)
        ]
    )
    assert not eligibility.eligible


def test_prediction_validator_cross_checks_direction_magnitude_and_coverage() -> None:
    meeting = _synthetic_panel_meetings()[0]
    row = baselines._available_prediction(
        meeting,
        model="always_hold",
        comparison_contract="matched_roster",
        fit_contract="no_fit_constant",
        action_probabilities={0: 1.0},
    )
    row["direction_probabilities"] = {
        "cut": 1.0,
        "hold": 0.0,
        "hike": 0.0,
        "invalid": 0.0,
    }
    with pytest.raises(baselines.DecisionComparisonError, match="probability mismatch"):
        baselines.validate_prediction_rows([row])

    unavailable = baselines._unavailable_prediction(
        meeting,
        model="always_hold",
        comparison_contract="matched_roster",
        fit_contract="no_fit_constant",
        reason="synthetic",
    )
    unavailable["predicted_magnitude_bp"] = 25
    with pytest.raises(baselines.DecisionComparisonError, match="carries a prediction"):
        baselines.validate_prediction_rows([unavailable])


def _futures_meeting(day: str, cutoff: str) -> dict[str, object]:
    return {
        "sample_id": f"futures-{day}",
        "panel": "historical_n19",
        "meeting_id": f"futures-meeting-{day}",
        "meeting_start_date": day,
        "meeting_end_date": day,
        "evidence_cutoff": cutoff,
        "direction": "hold",
        "magnitude_bp": 0,
    }


def _futures_feature() -> dict[str, object]:
    return {
        "feature_sha256": "feature-hash",
        "observations": {"DFF": {"value": 4.75}},
    }


def test_futures_prediction_handles_weekend_gap_month_end_and_gap_gate() -> None:
    source = {
        "available": True,
        "contract_by_month": {"2024-01": "jan", "2024-02": "feb"},
        "settlements": {
            "jan": [("2024-01-12", 95.0)],
            "feb": [("2024-01-30", 95.0)],
        },
        "source_hashes": {"wrds_manifest_sha256": "manifest-hash"},
    }
    regular = baselines._futures_prediction(
        _futures_meeting("2024-01-15", "2024-01-14"),
        _futures_feature(),
        source,
        comparison_contract="matched_roster",
        maximum_gap_days=4,
    )
    assert regular["coverage_status"] == "available"
    assert regular["settlement_gap_calendar_days"] == 2
    assert regular["month_end_next_contract"] is False
    assert regular["restricted_raw_quote_persisted"] is False
    assert "settlement" not in regular and "futcode" not in regular

    month_end = baselines._futures_prediction(
        _futures_meeting("2024-01-31", "2024-01-30"),
        _futures_feature(),
        source,
        comparison_contract="matched_roster",
        maximum_gap_days=4,
    )
    assert month_end["coverage_status"] == "available"
    assert month_end["contract_month"] == "2024-02"
    assert month_end["month_end_next_contract"] is True
    assert month_end["magnitude_probabilities"]["25"] == pytest.approx(1.0)

    stale_source = copy.deepcopy(source)
    stale_source["settlements"]["jan"] = [("2024-01-09", 95.0)]
    stale = baselines._futures_prediction(
        _futures_meeting("2024-01-15", "2024-01-14"),
        _futures_feature(),
        stale_source,
        comparison_contract="matched_roster",
        maximum_gap_days=4,
    )
    assert stale["coverage_status"] == "unavailable"
    assert stale["unavailable_reason"].startswith("settlement_gap_exceeds_4_days")


def test_create_only_resume_rejects_output_drift(tmp_path) -> None:
    output = tmp_path / "rows.jsonl"
    baselines.write_jsonl(output, [{"value": 1}])
    assert baselines.write_jsonl(output, [{"value": 1}], resume=True) == 1
    with pytest.raises(baselines.DecisionComparisonError, match="resume output drift"):
        baselines.write_jsonl(output, [{"value": 2}], resume=True)
    with pytest.raises(FileExistsError, match="create-only"):
        baselines.write_jsonl(output, [{"value": 1}], resume=False)


def test_status_snapshot_rejects_bound_input_hash_drift(tmp_path) -> None:
    config = _frozen_config()
    paths = baselines.runtime_paths(config, tmp_path / "release")
    paths.output_root.mkdir(parents=True)
    bound_input = tmp_path / "bound-input.json"
    bound_input.write_text('{"version":1}\n', encoding="utf-8")
    output_paths = (
        paths.action_roster,
        paths.features,
        paths.predictions,
        paths.summary,
        paths.comparison_csv,
    )
    for index, path in enumerate(output_paths):
        path.write_text(f"sealed-output-{index}\n", encoding="utf-8")
    manifest = {
        "schema_version": "decision-comparison-baselines-manifest-v1",
        "status": "complete",
        "panel_contract": {
            "historical_n19": 19,
            "postcutoff_n12": 12,
            "pooled_result": None,
            "pooling_prohibited": True,
        },
        "training_contracts": ["matched_roster"],
        "baseline_models": ["always_hold"],
        "inputs": {"config": baselines._artifact_record(bound_input)},
        "outputs": {
            path.name: baselines._artifact_record(
                path, relative_to=paths.output_root
            )
            for path in output_paths
        },
    }
    baselines.write_json(paths.manifest, manifest)

    valid = baselines._status_snapshot(paths)
    assert valid["effective_state"] == "release_complete"
    assert valid["release"]["inputs_verified"] == 1

    bound_input.write_text('{"version":2}\n', encoding="utf-8")
    invalid = baselines._status_snapshot(paths)
    assert invalid["effective_state"] == "release_invalid"
    assert "release input binding drift" in invalid["release"]["error"]


def test_macro_http_transport_requests_csv_only(monkeypatch) -> None:
    observed = {}

    class FakeResponse:
        status = 200
        headers = {"Content-Type": "text/csv"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read():
            return b"observation_date,UNRATE\n2024-01-01,3.7\n"

    def fake_urlopen(request, *, timeout):
        observed["accept"] = request.get_header("Accept")
        observed["user_agent"] = request.get_header("User-Agent")
        observed["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(baselines.urllib.request, "urlopen", fake_urlopen)
    body, headers = baselines._http_get(
        "https://example.invalid/series.csv",
        timeout_seconds=180.0,
        retries=1,
        backoff_initial_seconds=2.0,
        backoff_max_seconds=30.0,
    )

    assert body.startswith(b"observation_date")
    assert headers["content-type"] == "text/csv"
    assert observed == {
        "accept": "text/csv",
        "user_agent": None,
        "timeout": 180.0,
    }


def test_macro_acquisition_is_mocked_sealed_and_recovers_raw_only_partial(
    tmp_path, monkeypatch
) -> None:
    config = copy.deepcopy(_frozen_config())
    roster = [
        {
            "meeting_id": "synthetic-macro-meeting",
            "meeting_start_date": "2024-01-31",
            "meeting_end_date": "2024-01-31",
            "evidence_cutoff": "2024-01-30",
        }
    ]

    observed_http_options = []

    def fake_http_get(url: str, **options):
        from urllib.parse import parse_qs, urlparse

        observed_http_options.append(options)
        query = parse_qs(urlparse(url).query)
        series_id = query["id"][0].split(",")[0]
        if "alfred" in url:
            compact = query["vintage_date"][0].replace("-", "")
            body = (
                f"observation_date,{series_id}_{compact}\n"
                f"2023-10-01,{3.7 if series_id == 'UNRATE' else 22000.0}\n"
            ).encode()
        else:
            body = (
                f"observation_date,{series_id}\n2024-01-26,5.0\n2024-01-29,5.1\n"
            ).encode()
        return body, {"content-type": "text/csv"}

    monkeypatch.setattr(baselines, "_http_get", fake_http_get)
    macro_root = tmp_path / "macro"
    manifest = baselines.acquire_macro_sources(config, roster, macro_root, resume=False)
    baselines.validate_macro_manifest(
        macro_root,
        manifest,
        expected_contract=baselines._macro_acquisition_contract(config, roster),
    )
    assert manifest["status"] == "complete"
    assert manifest["meeting_count"] == 1
    assert observed_http_options
    assert all(
        options
        == {
            "timeout_seconds": 180.0,
            "retries": 6,
            "backoff_initial_seconds": 2.0,
            "backoff_max_seconds": 30.0,
        }
        for options in observed_http_options
    )
    assert manifest["http_transport"] == config["macro_sources"]["http"]
    assert (
        baselines.acquire_macro_sources(config, roster, macro_root, resume=True)
        == manifest
    )

    partial_root = tmp_path / "partial-macro"
    real_atomic_write = baselines._atomic_write
    failed = False

    def fail_first_normalized(path, content, *, exclusive):
        nonlocal failed
        if path.name == "normalized.csv" and not failed:
            failed = True
            raise RuntimeError("synthetic interruption")
        return real_atomic_write(path, content, exclusive=exclusive)

    monkeypatch.setattr(baselines, "_atomic_write", fail_first_normalized)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        baselines.acquire_macro_sources(config, roster, partial_root, resume=False)
    assert len(list(partial_root.glob("alfred/*/*/response.bin"))) == 1
    assert not list(partial_root.glob("alfred/*/*/normalized.csv"))

    monkeypatch.setattr(baselines, "_atomic_write", real_atomic_write)
    resumed = baselines.acquire_macro_sources(config, roster, partial_root, resume=True)
    baselines.validate_macro_manifest(
        partial_root,
        resumed,
        expected_contract=baselines._macro_acquisition_contract(config, roster),
    )
