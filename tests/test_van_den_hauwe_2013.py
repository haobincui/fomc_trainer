from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtr
from scipy.stats import truncnorm

from open_r1.utils import van_den_hauwe_2013 as vdh
from open_r1.utils.van_den_hauwe_2013 import (
    CUT,
    HIKE,
    HOLD,
    ImportanceSamplingConfig,
    MCMCConfig,
    VanDenHauweModelError,
    fit_van_den_hauwe_2013,
    latent_truncation_bounds,
    ordered_direction_probabilities,
    recursive_importance_forecast,
)


def _synthetic_monthly_panel(
    *, seed: int = 42, n_months: int = 54
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a dynamic ordered panel with one signal and one noise input."""

    rng = np.random.default_rng(seed)
    predictors = rng.normal(size=(n_months, 2))
    target_previous = np.full(n_months, 2.0)
    phi = 0.55
    disturbance = np.empty(n_months)
    disturbance[0] = rng.normal(scale=1.0 / np.sqrt(1.0 - phi**2))
    for t in range(1, n_months):
        disturbance[t] = phi * disturbance[t - 1] + rng.normal()
    offset = 2.0 * predictors[:, 0] + disturbance
    decisions = np.where(
        offset <= 0.0,
        CUT,
        np.where(offset <= 0.9, HOLD, HIKE),
    ).astype(float)
    # Main-specification non-meeting months: missing, never recoded to Hold.
    missing_months = np.asarray([4, 9, 14, 21, 29, 37, 45])
    decisions[missing_months[missing_months < n_months]] = np.nan
    assert all(np.any(decisions == category) for category in (CUT, HOLD, HIKE))
    return predictors, decisions, target_previous


@pytest.fixture(scope="module")
def synthetic_fit():
    predictors, decisions, target_previous = _synthetic_monthly_panel()
    return fit_van_den_hauwe_2013(
        predictors,
        decisions,
        target_previous,
        config=MCMCConfig(draws=120, burn_in=120, seed=7),
    )


def test_ordered_probability_formula_is_normalized_and_uses_prior_target() -> None:
    probabilities = ordered_direction_probabilities(
        mean=2.0,
        standard_deviation=1.0,
        target_previous=2.0,
        upper_threshold=1.0,
    )
    expected = np.asarray([0.5, ndtr(1.0) - 0.5, 1.0 - ndtr(1.0)])
    assert probabilities.shape == (3,)
    assert np.allclose(probabilities, expected)
    assert np.isclose(probabilities.sum(), 1.0)

    vectorized = ordered_direction_probabilities(
        mean=np.asarray([1.0, 2.0, 3.0]),
        standard_deviation=1.0,
        target_previous=2.0,
        upper_threshold=np.asarray([0.5, 1.0, 1.5]),
    )
    assert vectorized.shape == (3, 3)
    assert np.all(vectorized >= 0.0)
    assert np.allclose(vectorized.sum(axis=1), 1.0)


def test_missing_decision_month_has_no_latent_truncation() -> None:
    assert latent_truncation_bounds(np.nan, 2.0, 0.75) == (-np.inf, np.inf)
    assert latent_truncation_bounds(CUT, 2.0, 0.75) == (-np.inf, 2.0)
    assert latent_truncation_bounds(HOLD, 2.0, 0.75) == (2.0, 2.75)
    assert latent_truncation_bounds(HIKE, 2.0, 0.75) == (2.75, np.inf)


def test_sampler_preserves_missing_months_and_observed_constraints(
    synthetic_fit,
) -> None:
    assert synthetic_fit.diagnostics.missing_y_months == 7
    assert synthetic_fit.diagnostics.observed_constraint_violations == 0
    assert synthetic_fit.diagnostics.finite_draws
    assert np.all(np.isfinite(synthetic_fit.draws.latent_rate))

    probabilities = synthetic_fit.smoothed_class_probabilities()
    assert probabilities.shape == (54, 3)
    assert np.all(probabilities >= 0.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_kuo_mallick_selection_identifies_strong_synthetic_predictor(
    synthetic_fit,
) -> None:
    inclusion = synthetic_fit.posterior_inclusion_probability
    assert inclusion.shape == (2,)
    assert inclusion[0] > 0.90
    assert inclusion[0] > inclusion[1]
    assert set(np.unique(synthetic_fit.draws.gamma)).issubset({0, 1})
    assert np.all((synthetic_fit.draws.pi > 0.0) & (synthetic_fit.draws.pi < 1.0))


def test_phi_is_stationary_and_reported_thresholds_remain_ordered(
    synthetic_fit,
) -> None:
    assert np.all(synthetic_fit.draws.phi > -1.0)
    assert np.all(synthetic_fit.draws.phi < 1.0)
    assert synthetic_fit.diagnostics.phi_min > -1.0
    assert synthetic_fit.diagnostics.phi_max < 1.0
    original_thresholds = synthetic_fit.draws.original_thresholds
    assert np.all(original_thresholds[:, 0] < original_thresholds[:, 1])


def test_first_month_probability_uses_stationary_initial_variance(
    synthetic_fit,
) -> None:
    beta = synthetic_fit.draws.beta
    initial_mean = (
        synthetic_fit.draws.intercept
        + beta @ synthetic_fit.standardized_predictors[0]
    )
    initial_sd = 1.0 / np.sqrt(1.0 - synthetic_fit.draws.phi**2)
    manual = ordered_direction_probabilities(
        mean=initial_mean,
        standard_deviation=initial_sd,
        target_previous=synthetic_fit.target_previous[0],
        upper_threshold=synthetic_fit.draws.upper_threshold,
    ).mean(axis=0)
    assert np.allclose(synthetic_fit.smoothed_class_probabilities()[0], manual)


def test_one_step_posterior_prediction_is_a_probability_vector(synthetic_fit) -> None:
    prediction = synthetic_fit.predict_next([0.25, -0.5], target_previous=2.0)
    assert prediction.shape == (3,)
    assert np.all(prediction >= 0.0)
    assert np.isclose(prediction.sum(), 1.0)
    summary = synthetic_fit.posterior_summary()
    assert summary["model"] == "van_den_hauwe_2013_reduced_predictor"
    assert summary["architecture_faithful"] is True
    assert summary["full_33_predictor_replication"] is False


def test_fixed_seed_reproduces_all_retained_draws() -> None:
    predictors, decisions, target_previous = _synthetic_monthly_panel(
        seed=11, n_months=36
    )
    config = MCMCConfig(draws=40, burn_in=40, thin=1, seed=99)
    first = fit_van_den_hauwe_2013(
        predictors, decisions, target_previous, config=config
    )
    second = fit_van_den_hauwe_2013(
        predictors, decisions, target_previous, config=config
    )
    assert np.array_equal(first.draws.phi, second.draws.phi)
    assert np.array_equal(first.draws.upper_threshold, second.draws.upper_threshold)
    assert np.array_equal(first.draws.gamma, second.draws.gamma)
    assert np.array_equal(first.draws.psi, second.draws.psi)
    assert np.array_equal(first.draws.latent_rate, second.draws.latent_rate)
    assert first.diagnostics == second.diagnostics


def test_flat_upper_threshold_prior_requires_at_least_one_hike() -> None:
    predictors = np.arange(16, dtype=float).reshape(8, 2)
    decisions = np.asarray([CUT, HOLD, HOLD, np.nan, CUT, HOLD, HOLD, HOLD])
    with pytest.raises(VanDenHauweModelError, match="at least one Hike"):
        fit_van_den_hauwe_2013(
            predictors,
            decisions,
            np.full(8, 2.0),
            config=MCMCConfig(draws=2, burn_in=0),
        )


def _future_panel() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictors = np.asarray(
        [
            [0.20, -0.10],
            [-0.35, 0.40],
            [0.75, 0.15],
            [-0.60, -0.25],
            [0.10, 0.55],
        ]
    )
    decisions = np.asarray([HOLD, np.nan, HIKE, CUT, HOLD], dtype=float)
    target_previous = np.full(len(decisions), 2.0)
    return predictors, decisions, target_previous


def test_recursive_importance_probabilities_normalize_and_match_likelihood(
    synthetic_fit,
) -> None:
    predictors, decisions, target_previous = _future_panel()
    forecast = recursive_importance_forecast(
        synthetic_fit,
        predictors,
        decisions,
        target_previous,
        config=ImportanceSamplingConfig(seed=123),
    )
    assert forecast.probabilities.shape == (5, 3)
    assert np.all(forecast.probabilities >= 0.0)
    assert np.allclose(forecast.probabilities.sum(axis=1), 1.0)
    assert np.all(forecast.pre_update_ess > 0.0)
    assert np.all(forecast.post_update_ess > 0.0)
    assert np.all(forecast.post_update_ess_ratio <= 1.0 + 1e-12)

    for t, decision in enumerate(decisions):
        if not np.isnan(decision):
            assert np.isclose(
                np.exp(forecast.log_normalizer_increment[t]),
                forecast.probabilities[t, int(decision)],
            )
    assert np.allclose(
        forecast.cumulative_log_normalizer,
        np.cumsum(forecast.log_normalizer_increment),
    )
    diagnostics = forecast.diagnostics()
    assert diagnostics["resampling"] is False
    assert diagnostics["rejuvenation"] is False


def test_recursive_prediction_does_not_read_current_or_future_decisions(
    synthetic_fit,
) -> None:
    predictors, decisions, target_previous = _future_panel()
    baseline = recursive_importance_forecast(
        synthetic_fit,
        predictors,
        decisions,
        target_previous,
        config=ImportanceSamplingConfig(seed=314),
    )

    changed_current = decisions.copy()
    changed_current[2] = CUT
    counterfactual = recursive_importance_forecast(
        synthetic_fit,
        predictors,
        changed_current,
        target_previous,
        config=ImportanceSamplingConfig(seed=314),
    )
    # The changed label at t=2 cannot affect forecasts at t=0, t=1, or t=2.
    assert np.array_equal(
        baseline.probabilities[:3], counterfactual.probabilities[:3]
    )

    changed_final = decisions.copy()
    changed_final[-1] = HIKE
    final_counterfactual = recursive_importance_forecast(
        synthetic_fit,
        predictors,
        changed_final,
        target_previous,
        config=ImportanceSamplingConfig(seed=314),
    )
    # A label observed after the last probability is formed changes no forecast.
    assert np.array_equal(
        baseline.probabilities, final_counterfactual.probabilities
    )


def test_missing_future_month_propagates_state_without_weight_update(
    synthetic_fit,
) -> None:
    predictors, decisions, target_previous = _future_panel()
    forecast = recursive_importance_forecast(
        synthetic_fit,
        predictors,
        decisions,
        target_previous,
        config=ImportanceSamplingConfig(seed=77),
    )
    missing_t = 1
    assert np.isnan(decisions[missing_t])
    assert not forecast.observed_weight_update[missing_t]
    assert forecast.log_normalizer_increment[missing_t] == 0.0
    assert forecast.pre_update_ess[missing_t] == forecast.post_update_ess[missing_t]
    assert np.array_equal(
        forecast.normalized_log_weight_history[missing_t],
        forecast.normalized_log_weight_history[missing_t + 1],
    )
    assert np.all(np.isfinite(forecast.latent_rate_history[missing_t]))


def test_recursive_importance_forecast_is_reproducible(synthetic_fit) -> None:
    predictors, decisions, target_previous = _future_panel()
    config = ImportanceSamplingConfig(seed=808)
    first = recursive_importance_forecast(
        synthetic_fit, predictors, decisions, target_previous, config=config
    )
    second = recursive_importance_forecast(
        synthetic_fit, predictors, decisions, target_previous, config=config
    )
    assert np.array_equal(first.probabilities, second.probabilities)
    assert np.array_equal(first.pre_update_ess, second.pre_update_ess)
    assert np.array_equal(first.post_update_ess, second.post_update_ess)
    assert np.array_equal(
        first.normalized_log_weight_history,
        second.normalized_log_weight_history,
    )
    assert np.array_equal(first.latent_rate_history, second.latent_rate_history)
    assert np.array_equal(
        first.cumulative_log_normalizer,
        second.cumulative_log_normalizer,
    )


@pytest.mark.parametrize(
    ("mean", "sd", "lower", "upper"),
    [
        (0.0, 1.0, -np.inf, np.inf),
        (0.0, 1.0, -1.25, 2.50),
        (2.0, 0.4, 1.80, np.inf),
        (-2.0, 0.4, -np.inf, -1.80),
        (0.0, 1.0, 40.0, np.inf),
        (0.0, 1.0, -np.inf, -40.0),
        (0.0, 1.0, 40.0, 40.01),
        (0.0, 1.0, -40.01, -40.0),
    ],
)
def test_scalar_truncated_normal_respects_finite_and_extreme_bounds(
    mean: float,
    sd: float,
    lower: float,
    upper: float,
) -> None:
    rng = np.random.default_rng(1729)
    values = np.asarray(
        [
            vdh._sample_truncated_normal(mean, sd, lower, upper, rng)
            for _ in range(500)
        ]
    )
    assert np.all(np.isfinite(values))
    if np.isfinite(lower):
        assert np.all(values > lower)
    if np.isfinite(upper):
        assert np.all(values <= upper)


def test_scalar_truncated_normal_matches_analytic_moments() -> None:
    mean, sd, lower, upper = 1.2, 2.0, -0.3, 2.7
    standardized_lower = (lower - mean) / sd
    standardized_upper = (upper - mean) / sd
    expected_mean, expected_variance = truncnorm.stats(
        standardized_lower,
        standardized_upper,
        loc=mean,
        scale=sd,
        moments="mv",
    )
    rng = np.random.default_rng(2501)
    values = np.asarray(
        [
            vdh._sample_truncated_normal(mean, sd, lower, upper, rng)
            for _ in range(30_000)
        ]
    )
    assert abs(values.mean() - expected_mean) < 0.015
    assert abs(values.var() - expected_variance) < 0.02


def test_scalar_truncated_normal_extreme_tails_have_symmetric_moments() -> None:
    right_rng = np.random.default_rng(8080)
    left_rng = np.random.default_rng(8080)
    right = np.asarray(
        [
            vdh._sample_truncated_normal(0.0, 1.0, 40.0, np.inf, right_rng)
            for _ in range(2_000)
        ]
    )
    left = np.asarray(
        [
            vdh._sample_truncated_normal(0.0, 1.0, -np.inf, -40.0, left_rng)
            for _ in range(2_000)
        ]
    )
    assert 40.0 < right.mean() < 40.05
    assert -40.05 < left.mean() < -40.0
    assert abs(right.mean() + left.mean()) < 0.003


def test_scalar_truncated_normal_is_seed_reproducible() -> None:
    first_rng = np.random.default_rng(909)
    second_rng = np.random.default_rng(909)
    first = np.asarray(
        [
            vdh._sample_truncated_normal(0.5, 1.3, 3.0, np.inf, first_rng)
            for _ in range(1_000)
        ]
    )
    second = np.asarray(
        [
            vdh._sample_truncated_normal(0.5, 1.3, 3.0, np.inf, second_rng)
            for _ in range(1_000)
        ]
    )
    assert np.array_equal(first, second)


def test_inverse_cdf_preserves_scipy_uniform_stream_mapping() -> None:
    cases = (
        (0.2, 1.0, -np.inf, 0.0),
        (0.4, 1.0, 0.0, 0.8),
        (-0.3, 1.0, 0.8, np.inf),
        (0.1, 1.0, -np.inf, np.inf),
        (0.0, 1.0, 8.0, np.inf),
        (0.0, 1.0, -np.inf, -8.0),
    )
    scipy_rng = np.random.default_rng(1234)
    inverse_rng = np.random.default_rng(1234)
    scipy_values: list[float] = []
    inverse_values: list[float] = []
    for i in range(600):
        mean, sd, lower, upper = cases[i % len(cases)]
        if np.isneginf(lower) and np.isposinf(upper):
            scipy_values.append(float(scipy_rng.normal(mean, sd)))
        else:
            scipy_values.append(
                float(
                    truncnorm.rvs(
                        (lower - mean) / sd,
                        (upper - mean) / sd,
                        loc=mean,
                        scale=sd,
                        random_state=scipy_rng,
                    )
                )
            )
        inverse_values.append(
            vdh._sample_truncated_normal(
                mean, sd, lower, upper, inverse_rng
            )
        )
    # Both implementations consume one variate per draw and apply the same CDF
    # orientation.  Tail-specific special functions differ only at roundoff.
    assert np.allclose(scipy_values, inverse_values, rtol=0.0, atol=2e-15)
