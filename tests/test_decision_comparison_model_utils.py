from __future__ import annotations

import builtins

import numpy as np
import pytest

from open_r1.utils import decision_comparison_baselines as baselines


def _model_sample(seed: int = 7) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    signal = np.linspace(-2.5, 2.5, 90)
    features = np.column_stack(
        (
            signal + rng.normal(scale=0.15, size=len(signal)),
            np.sin(signal) + rng.normal(scale=0.1, size=len(signal)),
        )
    )
    labels = [baselines.CUT] * 30 + [baselines.HOLD] * 30 + [baselines.HIKE] * 30
    return features, labels


def _ordered_model_sample(seed: int = 19) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(600, 2))
    latent = (
        1.4 * features[:, 0]
        - 0.4 * features[:, 1]
        + rng.normal(scale=1.0, size=len(features))
    )
    labels = np.where(
        latent < -0.6,
        baselines.CUT,
        np.where(latent > 0.6, baselines.HIKE, baselines.HOLD),
    ).tolist()
    return features, labels


def test_probability_validation_and_default_tie_policy() -> None:
    row = baselines.validate_probability_matrix([0.1, 0.7, 0.2])
    assert np.array_equal(row, np.asarray([0.1, 0.7, 0.2]))
    assert baselines.deterministic_argmax(row) == baselines.HOLD
    assert baselines.deterministic_argmax([0.5, 0.5, 0.0]) is None
    assert (
        baselines.deterministic_argmax(
            [0.5, 0.5, 0.0],
            tie_priority=(baselines.HOLD, baselines.CUT, baselines.HIKE),
        )
        == baselines.HOLD
    )

    with pytest.raises(baselines.ProbabilityValidationError, match="sum to one"):
        baselines.validate_probability_matrix([0.2, 0.2, 0.2])
    with pytest.raises(baselines.ProbabilityValidationError, match=r"\[0, 1\]"):
        baselines.validate_probability_matrix([-0.1, 0.6, 0.5])
    with pytest.raises(baselines.ProbabilityValidationError, match="non-finite"):
        baselines.validate_probability_matrix([np.nan, 0.5, 0.5])


def test_deterministic_argmax_rows_preserves_unresolved_ties() -> None:
    predictions = baselines.deterministic_argmax_rows(
        [[0.8, 0.1, 0.1], [0.0, 0.5, 0.5], [0.1, 0.2, 0.7]]
    )
    assert predictions == (baselines.CUT, None, baselines.HIKE)


def test_fixed_expanding_window_eligibility_boundaries() -> None:
    eligible_labels = [baselines.CUT] * 3 + [baselines.HOLD] * 24 + [baselines.HIKE] * 3
    result = baselines.expanding_window_eligibility(eligible_labels)
    assert result.eligible
    assert result.n_observations == 30
    assert result.class_counts == {"cut": 3, "hold": 24, "hike": 3}
    assert result.reasons == ()

    too_short = baselines.expanding_window_eligibility(eligible_labels[:-1])
    assert not too_short.eligible
    assert any("requires at least 30" in reason for reason in too_short.reasons)

    rare_hike = baselines.expanding_window_eligibility(
        [baselines.CUT] * 4 + [baselines.HOLD] * 24 + [baselines.HIKE] * 2
    )
    assert not rare_hike.eligible
    assert any("class 'hike'" in reason for reason in rare_hike.reasons)


def test_kauppi_mnl_uses_canonical_probability_order() -> None:
    pytest.importorskip("sklearn")
    features, labels = _model_sample()
    probabilities = baselines.kauppi_mnl_predict_proba(
        features,
        labels,
        np.asarray([[-2.2, -0.8], [0.0, 0.0], [2.2, 0.8]]),
    )
    assert probabilities.shape == (3, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert baselines.deterministic_argmax_rows(probabilities) == (
        baselines.CUT,
        baselines.HOLD,
        baselines.HIKE,
    )


def test_kauppi_mnl_passes_the_fixed_estimator_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    scaler_calls: dict[str, np.ndarray] = {}

    class FakeStandardScaler:
        def fit_transform(self, features):
            scaler_calls["fit_transform"] = features.copy()
            return features + 10.0

        def transform(self, features):
            scaler_calls["transform"] = features.copy()
            return features - 10.0

    class FakeLogisticRegression:
        classes_ = np.asarray([2, 0, 1])

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self, features, labels):
            assert features.shape == (3, 1)
            assert np.array_equal(features[:, 0], [9.0, 10.0, 11.0])
            assert labels.tolist() == [0, 1, 2]

        def predict_proba(self, features):
            assert features.shape == (1, 1)
            assert np.array_equal(features[:, 0], [-9.5])
            # Estimator columns are encoded classes (hike, cut, hold).
            return np.asarray([[0.1, 0.7, 0.2]])

    monkeypatch.setattr(
        baselines,
        "_require_sklearn_standard_scaler",
        lambda: FakeStandardScaler,
    )
    monkeypatch.setattr(
        baselines,
        "_require_sklearn_logistic_regression",
        lambda: FakeLogisticRegression,
    )
    probabilities = baselines.kauppi_mnl_predict_proba(
        [[-1.0], [0.0], [1.0]],
        [baselines.CUT, baselines.HOLD, baselines.HIKE],
        [[0.5]],
    )
    assert captured == {"C": 1.0, "solver": "lbfgs", "max_iter": 5000}
    assert np.array_equal(scaler_calls["fit_transform"], [[-1.0], [0.0], [1.0]])
    assert np.array_equal(scaler_calls["transform"], [[0.5]])
    assert np.allclose(probabilities, [[0.7, 0.2, 0.1]])


def test_ordered_probit_maps_ordered_levels_to_canonical_columns() -> None:
    pytest.importorskip("statsmodels")
    features, labels = _ordered_model_sample()
    probabilities = baselines.ordered_probit_predict_proba(
        features,
        labels,
        np.asarray([[-2.2, -0.8], [0.0, 0.0], [2.2, 0.8]]),
    )
    assert probabilities.shape == (3, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.argmax(probabilities[0]) == 0
    assert np.argmax(probabilities[1]) == 1
    assert np.argmax(probabilities[2]) == 2


def test_ordered_probit_rejects_nonconvergence(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResult:
        mle_retvals = {"converged": False}

    class FakeOrderedModel:
        def __init__(self, endog, exog, distr):
            assert distr == "probit"

        def fit(self, **kwargs):
            assert kwargs == {"method": "bfgs", "maxiter": 5000, "disp": False}
            return FakeResult()

    monkeypatch.setattr(
        baselines, "_require_statsmodels_ordered_model", lambda: FakeOrderedModel
    )
    features, labels = _model_sample()
    with pytest.raises(baselines.ModelConvergenceError, match="did not report"):
        baselines.ordered_probit_predict_proba(features, labels, features[:1])


def test_hamilton_jorda_ach_op_matches_official_replication_equations() -> None:
    no_meeting_hazard = baselines.hamilton_jorda_ach_hazard(
        previous_duration_weeks=1.74,
        spread_pp=0.495,
        fomc_meeting=False,
    )
    meeting_hazard = baselines.hamilton_jorda_ach_hazard(
        previous_duration_weeks=1.74,
        spread_pp=0.495,
        fomc_meeting=True,
    )
    assert no_meeting_hazard == pytest.approx(0.0378146188546886)
    assert meeting_hazard == pytest.approx(0.29419793405921607)

    conditional_marks = baselines.hamilton_jorda_ordered_probit_mark_probabilities(
        previous_change_pp=0.25,
        spread_pp=0.495,
    )
    assert tuple(conditional_marks) == (-50, -25, 0, 25, 50)
    assert sum(conditional_marks.values()) == pytest.approx(1.0)
    assert conditional_marks[25] == pytest.approx(0.5485669075415229)

    actions = baselines.hamilton_jorda_ach_op_action_probabilities(
        previous_change_pp=0.25,
        previous_duration_weeks=1.74,
        spread_pp=0.495,
        fomc_meeting=True,
    )
    assert sum(actions.values()) == pytest.approx(1.0)
    assert actions[0] == pytest.approx(1.0 - meeting_hazard + meeting_hazard * conditional_marks[0])
    assert actions[25] == pytest.approx(meeting_hazard * conditional_marks[25])


def test_hamilton_jorda_pasted_hazard_and_input_validation() -> None:
    # A sufficiently large absolute spread sends the raw index below the
    # boundary; the official smooth paste keeps the hazard strictly below one.
    assert baselines.hamilton_jorda_ach_hazard(
        previous_duration_weeks=1,
        spread_pp=10,
        fomc_meeting=True,
    ) == pytest.approx(1.0 / 1.0001)
    with pytest.raises(ValueError, match="must be positive"):
        baselines.hamilton_jorda_ach_hazard(
            previous_duration_weeks=0,
            spread_pp=0,
        )
    with pytest.raises(ValueError, match="finite"):
        baselines.hamilton_jorda_ordered_probit_mark_probabilities(
            previous_change_pp=float("nan"),
            spread_pp=0,
        )


def test_missing_statsmodels_error_names_the_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("statsmodels"):
            raise ImportError("blocked for test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match="ordered probit requires statsmodels"):
        baselines._require_statsmodels_ordered_model()


def test_two_point_isotonic_projection_and_class_recovery() -> None:
    q1, q2 = baselines.project_cumulative_probabilities([0.2, 0.8], [0.6, 0.3])
    assert np.allclose(q1, [0.4, 0.8])
    assert np.allclose(q2, [0.4, 0.3])
    probabilities = baselines.cumulative_to_decision_probabilities(
        [0.2, 0.8], [0.6, 0.3]
    )
    assert np.allclose(probabilities, [[0.6, 0.0, 0.4], [0.2, 0.5, 0.3]])


def test_cumulative_ordinal_rf_uses_requested_adjacent_seeds_and_fixed_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_arguments: list[dict[str, object]] = []

    class FakeRandomForestClassifier:
        classes_ = np.asarray([0, 1])

        def __init__(self, **kwargs):
            constructor_arguments.append(kwargs)
            self.instance_index = len(constructor_arguments) - 1

        def fit(self, features, labels):
            assert features.shape == (3, 1)
            assert set(labels.tolist()) == {0, 1}

        def predict_proba(self, features):
            assert features.shape == (1, 1)
            positive = 0.2 if self.instance_index == 0 else 0.6
            return np.asarray([[1.0 - positive, positive]])

    monkeypatch.setattr(
        baselines,
        "_require_sklearn_random_forest_classifier",
        lambda: FakeRandomForestClassifier,
    )
    probabilities = baselines.cumulative_ordinal_rf_predict_proba(
        [[-1.0], [0.0], [1.0]],
        [baselines.CUT, baselines.HOLD, baselines.HIKE],
        [[0.5]],
        random_state=41,
    )
    assert constructor_arguments == [
        {
            "n_estimators": 500,
            "min_samples_leaf": 5,
            "class_weight": "balanced_subsample",
            "max_features": "sqrt",
            "random_state": 41,
        },
        {
            "n_estimators": 500,
            "min_samples_leaf": 5,
            "class_weight": "balanced_subsample",
            "max_features": "sqrt",
            "random_state": 42,
        },
    ]
    assert np.allclose(probabilities, [[0.6, 0.0, 0.4]])


def test_cumulative_ordinal_rf_returns_valid_canonical_probabilities() -> None:
    pytest.importorskip("sklearn")
    features, labels = _model_sample()
    first = baselines.cumulative_ordinal_rf_predict_proba(
        features,
        labels,
        np.asarray([[-2.2, -0.8], [0.0, 0.0], [2.2, 0.8]]),
        random_state=20260827,
    )
    second = baselines.cumulative_ordinal_rf_predict_proba(
        features,
        labels,
        np.asarray([[-2.2, -0.8], [0.0, 0.0], [2.2, 0.8]]),
        random_state=20260827,
    )
    probabilities = first
    assert probabilities.shape == (3, 3)
    assert np.all(probabilities >= 0.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.array_equal(first, second)
    assert baselines.deterministic_argmax_rows(probabilities) == (
        baselines.CUT,
        baselines.HOLD,
        baselines.HIKE,
    )


def test_monthly_futures_identity_and_move_grid_interpolation() -> None:
    assert baselines.fed_funds_futures_implied_rate(95.0) == pytest.approx(5.0)
    post_rate = baselines.monthly_implied_post_rate(
        futures_price=95.0,
        current_rate=4.75,
        meeting_day=15,
        days_in_month=31,
    )
    assert post_rate == pytest.approx((31 * 5.0 - 15 * 4.75) / 16)

    grid_probabilities = baselines.interpolate_futures_move_probabilities(-12.5)
    assert grid_probabilities.shape == (9,)
    assert grid_probabilities[3] == pytest.approx(0.5)
    assert grid_probabilities[4] == pytest.approx(0.5)
    assert grid_probabilities.sum() == pytest.approx(1.0)
    assert np.allclose(
        baselines.futures_direction_probabilities(-12.5), [0.5, 0.5, 0.0]
    )

    assert baselines.interpolate_futures_move_probabilities(-500.0)[0] == 1.0
    assert baselines.interpolate_futures_move_probabilities(500.0)[-1] == 1.0
    with pytest.raises(ValueError, match="meeting_day"):
        baselines.monthly_implied_post_rate(95.0, 4.75, 31, 31)
