"""Reduced-predictor dynamic ordered probit of van den Hauwe et al. (2013).

This module implements the *architecture* of the monthly Bayesian model in
van den Hauwe, Paap, and van Dijk (2013), while leaving the number and identity
of candidate predictors generic.  In particular, it preserves

* a monthly latent desired target rate ``r_star``;
* thresholds applied to ``r_star[t] - target_previous[t]``;
* an AR(1) disturbance with a stationary initial distribution;
* genuinely missing decision months (their latent state is not truncated);
* Kuo--Mallick predictor inclusion and Bayesian model averaging; and
* the paper's Gibbs/Metropolis--Hastings posterior simulation scheme.

The sampler uses the paper's computational reparameterization: the lower
threshold is fixed at zero and an always-included intercept is estimated.  If
``a`` is that intercept and ``alpha_2_star`` the sampled upper threshold, the
thresholds in the no-intercept presentation of the model are ``(-a,
alpha_2_star - a)``.  Candidate-predictor coefficients, unlike the intercept,
have the Kuo--Mallick representation ``beta_k = gamma_k * psi_k``.

Only NumPy and SciPy are required.  Inputs should form a complete monthly
calendar.  A meeting-free month must have ``y[t] = numpy.nan``; coding it as a
hold changes the model and reproduces the alternative specification rejected
for the paper's main analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, log_ndtr, logsumexp, ndtr, ndtri, ndtri_exp


CUT = 0
HOLD = 1
HIKE = 2
N_CLASSES = 3

# Prior hyperparameters used in the paper's main specification.
PHI_PRIOR_MEAN = 0.0
PHI_PRIOR_VARIANCE = 10.0
PSI_PRIOR_VARIANCE = 16.0
PI_PRIOR_A = 1.0
PI_PRIOR_B = 1.0


class VanDenHauweModelError(ValueError):
    """Raised when the monthly panel or MCMC state violates the model."""


@dataclass(frozen=True)
class MCMCConfig:
    """Simulation controls for :func:`fit_van_den_hauwe_2013`.

    ``draws`` is the number of retained draws.  A draw is retained every
    ``thin`` iterations after ``burn_in``.  The prior is intentionally not
    configurable here: the module constants freeze the paper specification.
    """

    draws: int = 1_000
    burn_in: int = 1_000
    thin: int = 1
    seed: int = 20260830
    phi_proposal_sd: float = 0.12
    standardize_predictors: bool = True

    def validate(self) -> None:
        if self.draws <= 0:
            raise VanDenHauweModelError("draws must be positive")
        if self.burn_in < 0:
            raise VanDenHauweModelError("burn_in cannot be negative")
        if self.thin <= 0:
            raise VanDenHauweModelError("thin must be positive")
        if not np.isfinite(self.phi_proposal_sd) or self.phi_proposal_sd <= 0:
            raise VanDenHauweModelError("phi_proposal_sd must be positive")

    @property
    def total_iterations(self) -> int:
        return self.burn_in + self.draws * self.thin


@dataclass(frozen=True)
class ImportanceSamplingConfig:
    """Controls for recursive posterior forecasting without particle refresh."""

    seed: int = 20260830


@dataclass(frozen=True)
class SamplerDiagnostics:
    """Auditable single-chain diagnostics and posterior selection summaries."""

    iterations: int
    retained_draws: int
    burn_in: int
    thin: int
    seed: int
    phi_acceptance_rate: float
    phi_effective_sample_size: float
    upper_threshold_effective_sample_size: float
    pi_effective_sample_size: float
    gamma_effective_sample_size: tuple[float, ...]
    posterior_inclusion_probability: tuple[float, ...]
    phi_min: float
    phi_max: float
    observed_constraint_violations: int
    missing_y_months: int
    constant_predictor_indices: tuple[int, ...]
    finite_draws: bool
    chains: int = 1
    rhat: None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable diagnostics record."""

        return asdict(self)


@dataclass(frozen=True)
class RecursiveImportanceForecast:
    """Sequential probabilities and particle-degeneracy audit trail.

    ``normalized_log_weight_history[0]`` contains the equal initial weights.
    Row ``t + 1`` contains weights after processing future month ``t``.  The
    latent-state history is similarly recorded after propagation in each
    future month.  No particle is resampled or rejuvenated.
    """

    probabilities: NDArray[np.float64]
    pre_update_ess: NDArray[np.float64]
    post_update_ess: NDArray[np.float64]
    pre_update_ess_ratio: NDArray[np.float64]
    post_update_ess_ratio: NDArray[np.float64]
    log_normalizer_increment: NDArray[np.float64]
    cumulative_log_normalizer: NDArray[np.float64]
    observed_weight_update: NDArray[np.bool_]
    normalized_log_weight_history: NDArray[np.float64]
    latent_rate_history: NDArray[np.float64]
    decisions: NDArray[np.float64]
    config: ImportanceSamplingConfig

    @property
    def n_months(self) -> int:
        return int(self.probabilities.shape[0])

    @property
    def n_particles(self) -> int:
        return int(self.normalized_log_weight_history.shape[1])

    def diagnostics(self) -> dict[str, Any]:
        """Return a JSON-safe forecast and weight-degeneracy summary."""

        return {
            "method": "recursive_importance_sampling",
            "seed": self.config.seed,
            "n_months": self.n_months,
            "n_particles": self.n_particles,
            "resampling": False,
            "rejuvenation": False,
            "observed_updates": int(self.observed_weight_update.sum()),
            "missing_decision_months": int((~self.observed_weight_update).sum()),
            "minimum_post_update_ess": float(self.post_update_ess.min()),
            "minimum_post_update_ess_ratio": float(
                self.post_update_ess_ratio.min()
            ),
            "final_ess": float(self.post_update_ess[-1]),
            "final_ess_ratio": float(self.post_update_ess_ratio[-1]),
            "total_log_normalizer": float(self.cumulative_log_normalizer[-1]),
            "log_normalizer_increment": self.log_normalizer_increment.tolist(),
            "cumulative_log_normalizer": self.cumulative_log_normalizer.tolist(),
        }


@dataclass(frozen=True)
class PosteriorDraws:
    """Retained draws in the sampler's identified parameterization."""

    upper_threshold: NDArray[np.float64]
    phi: NDArray[np.float64]
    pi: NDArray[np.float64]
    intercept: NDArray[np.float64]
    psi: NDArray[np.float64]
    gamma: NDArray[np.int8]
    latent_rate: NDArray[np.float64]

    @property
    def beta(self) -> NDArray[np.float64]:
        """Kuo--Mallick predictor effects ``gamma * psi`` for every draw."""

        return self.gamma * self.psi

    @property
    def original_thresholds(self) -> NDArray[np.float64]:
        """Threshold draws in the equivalent no-intercept parameterization."""

        return np.column_stack(
            (
                -self.intercept,
                self.upper_threshold - self.intercept,
            )
        )


@dataclass(frozen=True)
class VanDenHauwe2013Result:
    """Posterior draws, transformations, diagnostics, and prediction methods."""

    draws: PosteriorDraws
    diagnostics: SamplerDiagnostics
    config: MCMCConfig
    predictor_mean: NDArray[np.float64]
    predictor_scale: NDArray[np.float64]
    standardized_predictors: NDArray[np.float64]
    decisions: NDArray[np.float64]
    target_previous: NDArray[np.float64]

    @property
    def n_predictors(self) -> int:
        return int(self.standardized_predictors.shape[1])

    @property
    def posterior_inclusion_probability(self) -> NDArray[np.float64]:
        return self.draws.gamma.mean(axis=0)

    def transform_predictors(self, predictors: ArrayLike) -> NDArray[np.float64]:
        """Apply the training-only location and scale transformation."""

        values = np.asarray(predictors, dtype=float)
        one_row = values.ndim == 1
        if one_row:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.n_predictors:
            raise VanDenHauweModelError(
                f"predictors must have {self.n_predictors} columns"
            )
        if not np.all(np.isfinite(values)):
            raise VanDenHauweModelError("predictors contain non-finite values")
        transformed = (values - self.predictor_mean) / self.predictor_scale
        return transformed[0] if one_row else transformed

    def smoothed_class_probabilities(self) -> NDArray[np.float64]:
        """Return posterior mean probabilities for every fitted month.

        For ``t > 0`` the probability in each retained draw conditions on that
        draw's ``r_star[t-1]``.  At ``t = 0`` it integrates over the stationary
        initial state.  These are posterior-smoothed in-sample quantities, not
        chronological forecasts.  Missing-y months remain valid rows.
        """

        n_draws, n_months = self.draws.latent_rate.shape
        probabilities = np.zeros((n_months, N_CLASSES), dtype=float)
        for draw_index in range(n_draws):
            beta = self.draws.beta[draw_index]
            intercept = self.draws.intercept[draw_index]
            phi = self.draws.phi[draw_index]
            eta = intercept + self.standardized_predictors @ beta

            initial_sd = 1.0 / np.sqrt(1.0 - phi * phi)
            probabilities[0] += ordered_direction_probabilities(
                mean=eta[0],
                standard_deviation=initial_sd,
                target_previous=self.target_previous[0],
                upper_threshold=self.draws.upper_threshold[draw_index],
            )
            if n_months > 1:
                means = eta[1:] + phi * (
                    self.draws.latent_rate[draw_index, :-1] - eta[:-1]
                )
                probabilities[1:] += ordered_direction_probabilities(
                    mean=means,
                    standard_deviation=1.0,
                    target_previous=self.target_previous[1:],
                    upper_threshold=self.draws.upper_threshold[draw_index],
                )
        return _normalize_probabilities(probabilities / n_draws)

    def predict_next(
        self,
        predictors: ArrayLike,
        target_previous: float,
    ) -> NDArray[np.float64]:
        """Bayesian one-month-ahead direction probabilities.

        The fit must end in the month immediately before the prediction month.
        The method integrates over parameter uncertainty, predictor-selection
        uncertainty, and the last latent state using the retained posterior
        draws.  It implements equation (6) and the normal-CDF expression in
        Appendix B of van den Hauwe et al. (2013), without recursive importance
        reweighting because the supplied draws already condition on the entire
        training panel.
        """

        x_next = self.transform_predictors(predictors)
        if x_next.ndim != 1:
            raise VanDenHauweModelError("predict_next accepts exactly one month")
        if not np.isfinite(target_previous):
            raise VanDenHauweModelError("target_previous must be finite")

        beta = self.draws.beta
        eta_previous = self.draws.intercept + beta @ self.standardized_predictors[-1]
        eta_next = self.draws.intercept + beta @ x_next
        means = eta_next + self.draws.phi * (
            self.draws.latent_rate[:, -1] - eta_previous
        )
        per_draw = ordered_direction_probabilities(
            mean=means,
            standard_deviation=1.0,
            target_previous=float(target_previous),
            upper_threshold=self.draws.upper_threshold,
        )
        return _normalize_probabilities(per_draw.mean(axis=0))

    def posterior_summary(self) -> dict[str, Any]:
        """Return compact JSON-safe posterior summaries for manifests."""

        thresholds = self.draws.original_thresholds
        return {
            "model": "van_den_hauwe_2013_reduced_predictor",
            "architecture_faithful": True,
            "full_33_predictor_replication": False,
            "n_predictors": self.n_predictors,
            "prior": {
                "phi": "N(0,10) truncated to (-1,1)",
                "psi": "N(0,16 I_K)",
                "pi": "Beta(1,1)",
                "threshold": "flat ordered; lower threshold fixed at zero internally",
                "intercept": "flat (computational reparameterization)",
            },
            "posterior_mean": {
                "phi": float(np.mean(self.draws.phi)),
                "pi": float(np.mean(self.draws.pi)),
                "original_lower_threshold": float(np.mean(thresholds[:, 0])),
                "original_upper_threshold": float(np.mean(thresholds[:, 1])),
                "inclusion_probability": self.posterior_inclusion_probability.tolist(),
            },
            "diagnostics": self.diagnostics.to_dict(),
        }


def latent_truncation_bounds(
    decision: float,
    target_previous: float,
    upper_threshold: float,
) -> tuple[float, float]:
    """Return the latent-rate bounds for one month.

    ``numpy.nan`` denotes a month with no FOMC decision and therefore returns
    ``(-inf, inf)``.  This behavior is central to the paper's main model.
    """

    if not np.isfinite(target_previous):
        raise VanDenHauweModelError("target_previous must be finite")
    if not np.isfinite(upper_threshold) or upper_threshold <= 0:
        raise VanDenHauweModelError("upper_threshold must be positive")
    if np.isnan(decision):
        return -np.inf, np.inf
    if decision == CUT:
        return -np.inf, float(target_previous)
    if decision == HOLD:
        return float(target_previous), float(target_previous + upper_threshold)
    if decision == HIKE:
        return float(target_previous + upper_threshold), np.inf
    raise VanDenHauweModelError("decision must be 0, 1, 2, or NaN")


def ordered_direction_probabilities(
    *,
    mean: ArrayLike,
    standard_deviation: ArrayLike,
    target_previous: ArrayLike,
    upper_threshold: ArrayLike,
) -> NDArray[np.float64]:
    """Evaluate Cut/Hold/Hike probabilities from the identified model.

    With internal thresholds ``(0, alpha_2_star)``, the formula is

    ``P(cut)  = Phi((r_prev - mu) / sigma)``
    ``P(hold) = Phi((r_prev + alpha_2_star - mu) / sigma) - P(cut)``
    ``P(hike) = 1 - Phi((r_prev + alpha_2_star - mu) / sigma)``.

    NumPy broadcasting is supported; the final dimension of the return value
    is always ordered as Cut, Hold, Hike.
    """

    log_probabilities = _ordered_direction_log_probabilities(
        mean=mean,
        standard_deviation=standard_deviation,
        target_previous=target_previous,
        upper_threshold=upper_threshold,
    )
    return np.exp(log_probabilities)


def fit_van_den_hauwe_2013(
    predictors: ArrayLike,
    decisions: Sequence[float] | NDArray[np.float64],
    target_previous: ArrayLike,
    *,
    config: MCMCConfig | None = None,
) -> VanDenHauwe2013Result:
    """Fit the reduced-predictor Bayesian dynamic ordered probit.

    Parameters
    ----------
    predictors:
        Complete monthly ``T x K`` predictor matrix.  Values must respect the
        desired real-time information contract before they reach this method.
    decisions:
        Monthly direction codes (0 Cut, 1 Hold, 2 Hike).  Use ``numpy.nan`` in
        every month without an FOMC decision.
    target_previous:
        Prevailing target rate at the end of the preceding month, in the same
        rate units as the desired latent target and thresholds.
    config:
        Deterministic single-chain simulation controls.

    Notes
    -----
    Predictors are standardized using this fitting panel only by default.  The
    fixed priors are ``phi ~ N(0,10) I(-1,1)``, ``psi ~ N(0,16 I_K)``, and
    ``pi ~ Beta(1,1)``.  The intercept has the improper flat prior implied by
    the paper's threshold/intercept reparameterization.
    """

    active_config = config or MCMCConfig()
    active_config.validate()
    x, y, previous = _validate_inputs(predictors, decisions, target_previous)
    n_months, n_predictors = x.shape

    predictor_mean = x.mean(axis=0)
    raw_scale = x.std(axis=0, ddof=0)
    constant_indices = tuple(np.flatnonzero(raw_scale <= np.finfo(float).eps).tolist())
    predictor_scale = np.where(raw_scale <= np.finfo(float).eps, 1.0, raw_scale)
    if active_config.standardize_predictors:
        standardized = (x - predictor_mean) / predictor_scale
    else:
        predictor_mean = np.zeros(n_predictors, dtype=float)
        predictor_scale = np.ones(n_predictors, dtype=float)
        standardized = x.copy()

    rng = np.random.default_rng(active_config.seed)
    upper_threshold = 1.0
    phi = 0.0
    pi = 0.5
    gamma = np.ones(n_predictors, dtype=np.int8)
    psi = np.zeros(n_predictors, dtype=float)

    initial_offset = np.zeros(n_months, dtype=float)
    initial_offset[y == CUT] = -0.5
    initial_offset[y == HOLD] = 0.5
    initial_offset[y == HIKE] = 1.5
    latent = previous + initial_offset
    intercept = float(np.mean(latent))

    retained_upper = np.empty(active_config.draws, dtype=float)
    retained_phi = np.empty(active_config.draws, dtype=float)
    retained_pi = np.empty(active_config.draws, dtype=float)
    retained_intercept = np.empty(active_config.draws, dtype=float)
    retained_psi = np.empty((active_config.draws, n_predictors), dtype=float)
    retained_gamma = np.empty(
        (active_config.draws, n_predictors), dtype=np.int8
    )
    retained_latent = np.empty((active_config.draws, n_months), dtype=float)

    phi_accepts = 0
    retained_index = 0
    for iteration in range(active_config.total_iterations):
        upper_threshold = _sample_upper_threshold(latent, y, previous, rng)

        beta = gamma * psi
        eta = intercept + standardized @ beta
        residual_state = latent - eta
        phi, accepted = _sample_phi(
            phi,
            residual_state,
            active_config.phi_proposal_sd,
            rng,
        )
        phi_accepts += int(accepted)

        pi = float(
            rng.beta(
                PI_PRIOR_A + int(gamma.sum()),
                PI_PRIOR_B + n_predictors - int(gamma.sum()),
            )
        )

        latent = _sample_latent_path(
            latent=latent,
            predictors=standardized,
            decisions=y,
            target_previous=previous,
            intercept=intercept,
            psi=psi,
            gamma=gamma,
            phi=phi,
            upper_threshold=upper_threshold,
            rng=rng,
        )

        transformed_y, transformed_x = _ar1_regression_transform(
            latent, standardized, phi
        )
        gamma = _sample_gamma(
            transformed_y,
            transformed_x,
            intercept,
            psi,
            gamma,
            pi,
            rng,
        )
        intercept, psi = _sample_coefficients(
            transformed_y,
            transformed_x,
            gamma,
            rng,
        )

        if iteration >= active_config.burn_in and (
            iteration - active_config.burn_in
        ) % active_config.thin == 0:
            retained_upper[retained_index] = upper_threshold
            retained_phi[retained_index] = phi
            retained_pi[retained_index] = pi
            retained_intercept[retained_index] = intercept
            retained_psi[retained_index] = psi
            retained_gamma[retained_index] = gamma
            retained_latent[retained_index] = latent
            retained_index += 1

    if retained_index != active_config.draws:  # pragma: no cover - defensive
        raise RuntimeError("internal retained-draw accounting error")

    posterior = PosteriorDraws(
        upper_threshold=retained_upper,
        phi=retained_phi,
        pi=retained_pi,
        intercept=retained_intercept,
        psi=retained_psi,
        gamma=retained_gamma,
        latent_rate=retained_latent,
    )
    violations = _count_constraint_violations(posterior, y, previous)
    finite_draws = all(
        np.all(np.isfinite(array))
        for array in (
            retained_upper,
            retained_phi,
            retained_pi,
            retained_intercept,
            retained_psi,
            retained_latent,
        )
    )
    diagnostics = SamplerDiagnostics(
        iterations=active_config.total_iterations,
        retained_draws=active_config.draws,
        burn_in=active_config.burn_in,
        thin=active_config.thin,
        seed=active_config.seed,
        phi_acceptance_rate=phi_accepts / active_config.total_iterations,
        phi_effective_sample_size=_effective_sample_size(retained_phi),
        upper_threshold_effective_sample_size=_effective_sample_size(retained_upper),
        pi_effective_sample_size=_effective_sample_size(retained_pi),
        gamma_effective_sample_size=tuple(
            _effective_sample_size(retained_gamma[:, k].astype(float))
            for k in range(n_predictors)
        ),
        posterior_inclusion_probability=tuple(
            retained_gamma.mean(axis=0).astype(float).tolist()
        ),
        phi_min=float(retained_phi.min()),
        phi_max=float(retained_phi.max()),
        observed_constraint_violations=violations,
        missing_y_months=int(np.isnan(y).sum()),
        constant_predictor_indices=constant_indices,
        finite_draws=finite_draws,
    )
    if not finite_draws:
        raise RuntimeError("MCMC produced non-finite retained draws")
    if violations:
        raise RuntimeError("MCMC retained latent draws outside observed categories")

    return VanDenHauwe2013Result(
        draws=posterior,
        diagnostics=diagnostics,
        config=active_config,
        predictor_mean=predictor_mean,
        predictor_scale=predictor_scale,
        standardized_predictors=standardized,
        decisions=y,
        target_previous=previous,
    )


def recursive_importance_forecast(
    fitted: VanDenHauwe2013Result,
    future_predictors: ArrayLike,
    future_decisions: Sequence[float] | NDArray[np.float64],
    future_target_previous: ArrayLike,
    *,
    config: ImportanceSamplingConfig | None = None,
) -> RecursiveImportanceForecast:
    """Generate recursive one-month-ahead forecasts by importance sampling.

    The retained MCMC draws in ``fitted`` form an equally weighted sample from
    the posterior through month ``t0``.  For every subsequent calendar month,
    this function performs the Appendix-B sequence:

    1. evaluate and weight the Cut/Hold/Hike probability *before* consulting
       that month's decision;
    2. if the decision is observed, multiply particle weights by its predictive
       likelihood and draw ``r_star`` from the corresponding truncated normal;
    3. if the month has no decision (``NaN``), leave weights unchanged and draw
       ``r_star`` from its untruncated transition distribution.

    Parameters are not redrawn, particles are not resampled, and no MCMC
    rejuvenation is performed.  The returned ESS path therefore exposes the
    importance-weight degeneracy that can accumulate over a long horizon.
    ``future_predictors`` must start in the month immediately after the fitted
    panel and contain every calendar month, including non-meeting months.
    """

    active_config = config or ImportanceSamplingConfig()
    x_future = np.asarray(future_predictors, dtype=float)
    y_future = np.asarray(future_decisions, dtype=float)
    previous_future = np.asarray(future_target_previous, dtype=float)
    if x_future.ndim != 2:
        raise VanDenHauweModelError(
            "future_predictors must be a two-dimensional matrix"
        )
    n_months = x_future.shape[0]
    if n_months < 1:
        raise VanDenHauweModelError("at least one future month is required")
    if x_future.shape[1] != fitted.n_predictors:
        raise VanDenHauweModelError(
            f"future_predictors must have {fitted.n_predictors} columns"
        )
    if y_future.shape != (n_months,):
        raise VanDenHauweModelError(
            "future_decisions length must match future predictor rows"
        )
    if previous_future.shape != (n_months,):
        raise VanDenHauweModelError(
            "future_target_previous length must match future predictor rows"
        )
    if not np.all(np.isfinite(x_future)):
        raise VanDenHauweModelError("future_predictors contain non-finite values")
    if not np.all(np.isfinite(previous_future)):
        raise VanDenHauweModelError(
            "future_target_previous contains non-finite values"
        )
    observed = y_future[~np.isnan(y_future)]
    if not np.all(np.isin(observed, (CUT, HOLD, HIKE))):
        raise VanDenHauweModelError(
            "future_decisions must contain only 0, 1, 2, or NaN"
        )

    standardized_future = fitted.transform_predictors(x_future)
    n_particles = len(fitted.draws.phi)
    if n_particles < 1:  # pragma: no cover - impossible for a valid fit
        raise VanDenHauweModelError("fitted result contains no posterior particles")
    rng = np.random.default_rng(active_config.seed)

    probabilities = np.empty((n_months, N_CLASSES), dtype=float)
    pre_ess = np.empty(n_months, dtype=float)
    post_ess = np.empty(n_months, dtype=float)
    log_increment = np.zeros(n_months, dtype=float)
    cumulative_log_normalizer = np.empty(n_months, dtype=float)
    observed_update = ~np.isnan(y_future)
    log_weight_history = np.empty((n_months + 1, n_particles), dtype=float)
    latent_history = np.empty((n_months, n_particles), dtype=float)

    log_weights = np.full(n_particles, -np.log(n_particles), dtype=float)
    log_weight_history[0] = log_weights
    latent_previous = fitted.draws.latent_rate[:, -1].copy()
    beta = fitted.draws.beta
    eta_previous = (
        fitted.draws.intercept
        + beta @ fitted.standardized_predictors[-1]
    )
    cumulative = 0.0

    for t in range(n_months):
        eta_current = (
            fitted.draws.intercept + beta @ standardized_future[t]
        )
        transition_mean = eta_current + fitted.draws.phi * (
            latent_previous - eta_previous
        )
        particle_log_probabilities = _ordered_direction_log_probabilities(
            mean=transition_mean,
            standard_deviation=1.0,
            target_previous=previous_future[t],
            upper_threshold=fitted.draws.upper_threshold,
        )

        # The prediction is computed before y_future[t] is read below.  The
        # weights at this point contain information only through month t - 1.
        aggregate_log_probability = logsumexp(
            log_weights[:, None] + particle_log_probabilities,
            axis=0,
        )
        aggregate_log_probability -= logsumexp(aggregate_log_probability)
        probabilities[t] = np.exp(aggregate_log_probability)
        pre_ess[t] = _importance_effective_sample_size(log_weights)

        if np.isnan(y_future[t]):
            latent_current = rng.normal(transition_mean, 1.0)
            # Integrating over an unobserved category contributes likelihood
            # one, hence a zero log-normalizer increment and identical weights.
            log_increment[t] = 0.0
        else:
            category = int(y_future[t])
            unnormalized_log_weights = (
                log_weights + particle_log_probabilities[:, category]
            )
            increment = float(logsumexp(unnormalized_log_weights))
            if not np.isfinite(increment):  # pragma: no cover - defensive
                raise RuntimeError(
                    "all importance particles assign zero probability to the decision"
                )
            log_increment[t] = increment
            log_weights = unnormalized_log_weights - increment
            latent_current = np.empty(n_particles, dtype=float)
            for particle in range(n_particles):
                lower, upper = latent_truncation_bounds(
                    y_future[t],
                    previous_future[t],
                    fitted.draws.upper_threshold[particle],
                )
                latent_current[particle] = _sample_truncated_normal(
                    float(transition_mean[particle]),
                    1.0,
                    lower,
                    upper,
                    rng,
                )

        cumulative += log_increment[t]
        cumulative_log_normalizer[t] = cumulative
        post_ess[t] = _importance_effective_sample_size(log_weights)
        log_weight_history[t + 1] = log_weights
        latent_history[t] = latent_current
        latent_previous = latent_current
        eta_previous = eta_current

    return RecursiveImportanceForecast(
        probabilities=probabilities,
        pre_update_ess=pre_ess,
        post_update_ess=post_ess,
        pre_update_ess_ratio=pre_ess / n_particles,
        post_update_ess_ratio=post_ess / n_particles,
        log_normalizer_increment=log_increment,
        cumulative_log_normalizer=cumulative_log_normalizer,
        observed_weight_update=observed_update,
        normalized_log_weight_history=log_weight_history,
        latent_rate_history=latent_history,
        decisions=y_future,
        config=active_config,
    )


def _validate_inputs(
    predictors: ArrayLike,
    decisions: Sequence[float] | NDArray[np.float64],
    target_previous: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    x = np.asarray(predictors, dtype=float)
    y = np.asarray(decisions, dtype=float)
    previous = np.asarray(target_previous, dtype=float)
    if x.ndim != 2:
        raise VanDenHauweModelError("predictors must be a two-dimensional matrix")
    if x.shape[0] < 2:
        raise VanDenHauweModelError("at least two monthly rows are required")
    if x.shape[1] < 1:
        raise VanDenHauweModelError("at least one candidate predictor is required")
    if y.shape != (x.shape[0],):
        raise VanDenHauweModelError("decisions length must match predictor rows")
    if previous.shape != (x.shape[0],):
        raise VanDenHauweModelError("target_previous length must match predictor rows")
    if not np.all(np.isfinite(x)):
        raise VanDenHauweModelError("predictors contain non-finite values")
    if not np.all(np.isfinite(previous)):
        raise VanDenHauweModelError("target_previous contains non-finite values")
    observed = y[~np.isnan(y)]
    if observed.size == 0:
        raise VanDenHauweModelError("at least one observed decision is required")
    if not np.all(np.isin(observed, (CUT, HOLD, HIKE))):
        raise VanDenHauweModelError("decisions must be 0, 1, 2, or NaN")
    # With a flat ordered-threshold prior the upper threshold has an improper
    # conditional posterior if no Hike observation supplies a finite bound.
    if not np.any(observed == HIKE):
        raise VanDenHauweModelError(
            "at least one Hike observation is required by the flat threshold prior"
        )
    return x, y, previous


def _sample_upper_threshold(
    latent: NDArray[np.float64],
    decisions: NDArray[np.float64],
    previous: NDArray[np.float64],
    rng: np.random.Generator,
) -> float:
    offset = latent - previous
    holds = offset[decisions == HOLD]
    hikes = offset[decisions == HIKE]
    lower = max(0.0, float(np.max(holds)) if holds.size else 0.0)
    upper = float(np.min(hikes))
    if not np.isfinite(upper) or not lower < upper:
        raise RuntimeError(
            "invalid ordered-threshold conditional bounds; check latent state"
        )
    return float(rng.uniform(lower, upper))


def _sample_phi(
    current: float,
    residual_state: NDArray[np.float64],
    proposal_sd: float,
    rng: np.random.Generator,
) -> tuple[float, bool]:
    proposal = _sample_truncated_normal(current, proposal_sd, -1.0, 1.0, rng)
    log_ratio = _phi_log_posterior(proposal, residual_state) - _phi_log_posterior(
        current, residual_state
    )
    # Truncated random-walk proposals are asymmetric near the stationarity
    # bounds.  The Gaussian kernels cancel, leaving this normalizer ratio.
    log_ratio += _log_rw_normalizer(current, proposal_sd) - _log_rw_normalizer(
        proposal, proposal_sd
    )
    if np.log(rng.uniform()) < min(0.0, log_ratio):
        return proposal, True
    return current, False


def _phi_log_posterior(
    phi: float, residual_state: NDArray[np.float64]
) -> float:
    if not -1.0 < phi < 1.0:
        return -np.inf
    stationary_precision = 1.0 - phi * phi
    value = 0.5 * np.log(stationary_precision)
    value -= 0.5 * stationary_precision * residual_state[0] ** 2
    innovations = residual_state[1:] - phi * residual_state[:-1]
    value -= 0.5 * float(innovations @ innovations)
    value -= 0.5 * (phi - PHI_PRIOR_MEAN) ** 2 / PHI_PRIOR_VARIANCE
    return float(value)


def _log_rw_normalizer(center: float, standard_deviation: float) -> float:
    upper = ndtr((1.0 - center) / standard_deviation)
    lower = ndtr((-1.0 - center) / standard_deviation)
    return float(np.log(max(float(upper - lower), np.finfo(float).tiny)))


def _sample_latent_path(
    *,
    latent: NDArray[np.float64],
    predictors: NDArray[np.float64],
    decisions: NDArray[np.float64],
    target_previous: NDArray[np.float64],
    intercept: float,
    psi: NDArray[np.float64],
    gamma: NDArray[np.int8],
    phi: float,
    upper_threshold: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    sampled = latent.copy()
    eta = intercept + predictors @ (gamma * psi)
    n_months = len(sampled)
    for t in range(n_months):
        if n_months == 1:  # input validation currently rules this out
            mean = eta[0]
            variance = 1.0 / (1.0 - phi * phi)
        elif t == 0:
            # The stationary initial density and the t=2 transition combine
            # to precision one (Appendix A.2 of the paper).
            mean = eta[0] + phi * (sampled[1] - eta[1])
            variance = 1.0
        elif t == n_months - 1:
            mean = eta[t] + phi * (sampled[t - 1] - eta[t - 1])
            variance = 1.0
        else:
            forward_mean = eta[t] + phi * (sampled[t - 1] - eta[t - 1])
            backward_signal = sampled[t + 1] - eta[t + 1] + phi * eta[t]
            variance = 1.0 / (1.0 + phi * phi)
            mean = variance * (forward_mean + phi * backward_signal)
        lower, upper = latent_truncation_bounds(
            decisions[t], target_previous[t], upper_threshold
        )
        sampled[t] = _sample_truncated_normal(
            float(mean), float(np.sqrt(variance)), lower, upper, rng
        )
    return sampled


def _ar1_regression_transform(
    latent: NDArray[np.float64],
    predictors: NDArray[np.float64],
    phi: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    design = np.column_stack((np.ones(len(predictors)), predictors))
    transformed_y = np.empty_like(latent)
    transformed_x = np.empty_like(design)
    stationary_scale = np.sqrt(1.0 - phi * phi)
    transformed_y[0] = stationary_scale * latent[0]
    transformed_x[0] = stationary_scale * design[0]
    transformed_y[1:] = latent[1:] - phi * latent[:-1]
    transformed_x[1:] = design[1:] - phi * design[:-1]
    return transformed_y, transformed_x


def _sample_gamma(
    transformed_y: NDArray[np.float64],
    transformed_x: NDArray[np.float64],
    intercept: float,
    psi: NDArray[np.float64],
    gamma: NDArray[np.int8],
    pi: float,
    rng: np.random.Generator,
) -> NDArray[np.int8]:
    sampled = gamma.copy()
    predictor_design = transformed_x[:, 1:]
    log_prior_odds = np.log(np.clip(pi, 1e-15, 1.0)) - np.log(
        np.clip(1.0 - pi, 1e-15, 1.0)
    )
    for k in rng.permutation(len(sampled)):
        other_effect = predictor_design @ (sampled * psi)
        other_effect -= predictor_design[:, k] * sampled[k] * psi[k]
        base_residual = (
            transformed_y - transformed_x[:, 0] * intercept - other_effect
        )
        residual_zero = base_residual
        residual_one = base_residual - predictor_design[:, k] * psi[k]
        log_likelihood_odds = -0.5 * (
            float(residual_one @ residual_one)
            - float(residual_zero @ residual_zero)
        )
        probability = float(expit(log_prior_odds + log_likelihood_odds))
        sampled[k] = int(rng.uniform() < probability)
    return sampled


def _sample_coefficients(
    transformed_y: NDArray[np.float64],
    transformed_x: NDArray[np.float64],
    gamma: NDArray[np.int8],
    rng: np.random.Generator,
) -> tuple[float, NDArray[np.float64]]:
    # The intercept is always included and has a flat prior.  Multiplication
    # of predictor columns by gamma implements beta_k = gamma_k * psi_k while
    # retaining N(0,16) draws for excluded psi_k values.
    effective_design = transformed_x.copy()
    effective_design[:, 1:] *= gamma
    n_coefficients = effective_design.shape[1]
    prior_precision = np.zeros(n_coefficients, dtype=float)
    prior_precision[1:] = 1.0 / PSI_PRIOR_VARIANCE
    precision = effective_design.T @ effective_design
    precision.flat[:: n_coefficients + 1] += prior_precision
    information = effective_design.T @ transformed_y
    try:
        chol = np.linalg.cholesky(precision)
        mean = np.linalg.solve(chol.T, np.linalg.solve(chol, information))
        noise = np.linalg.solve(chol.T, rng.normal(size=n_coefficients))
    except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
        raise RuntimeError("coefficient posterior precision is not positive definite") from exc
    draw = mean + noise
    return float(draw[0]), draw[1:]


def _sample_truncated_normal(
    mean: float,
    standard_deviation: float,
    lower: float,
    upper: float,
    rng: np.random.Generator,
) -> float:
    if not lower < upper:
        raise RuntimeError("truncated-normal lower bound must be below upper bound")
    if np.isneginf(lower) and np.isposinf(upper):
        return float(rng.normal(mean, standard_deviation))
    standardized_lower = (lower - mean) / standard_deviation
    standardized_upper = (upper - mean) / standard_deviation
    # Generator.random() is half-open and may very rarely return exactly zero;
    # replace only that endpoint so an infinite truncation bound cannot be
    # selected.  Every truncated draw still consumes exactly one uniform.
    uniform = max(float(rng.random()), float(np.nextafter(0.0, 1.0)))

    if standardized_lower >= 0.0:
        # In the right tail, ordinary CDF values both round to one.  Sample a
        # survival probability q uniformly between Q(b) and Q(a), retain it in
        # log space, then use Phi^{-1}(q) with reflection symmetry.
        log_survival_lower = float(log_ndtr(-standardized_lower))
        log_survival_upper = float(log_ndtr(-standardized_upper))
        relative_mass = float(
            -np.expm1(log_survival_upper - log_survival_lower)
        )
        # This orientation matches ordinary inverse-CDF interpolation:
        # p = Phi(a) + u[Phi(b)-Phi(a)], hence q = Q(a)-u[Q(a)-Q(b)].
        log_probability = log_survival_lower + np.log1p(
            -uniform * relative_mass
        )
        standardized_draw = -float(ndtri_exp(log_probability))
    elif standardized_upper <= 0.0:
        # Symmetric left-tail calculation using log CDFs.  ndtri_exp accepts
        # log probabilities far below the range representable by exp(log_p).
        log_cdf_lower = float(log_ndtr(standardized_lower))
        log_cdf_upper = float(log_ndtr(standardized_upper))
        log_mass = _log_difference(log_cdf_upper, log_cdf_lower)
        log_probability = np.logaddexp(
            log_cdf_lower,
            np.log(uniform) + log_mass,
        )
        standardized_draw = float(ndtri_exp(log_probability))
    else:
        # An interval crossing zero has at least half of one tail available, so
        # direct CDF interpolation is well conditioned.
        cdf_lower = float(ndtr(standardized_lower))
        cdf_upper = float(ndtr(standardized_upper))
        probability = cdf_lower + uniform * (cdf_upper - cdf_lower)
        standardized_draw = float(ndtri(probability))

    draw = float(mean + standard_deviation * standardized_draw)
    # Protect the exact category constraints from a rare endpoint returned due
    # to floating-point rounding in extreme tails.
    if np.isfinite(lower) and draw <= lower:
        draw = float(np.nextafter(lower, upper))
    if np.isfinite(upper) and draw > upper:
        draw = float(upper)
    return draw


def _log_difference(log_larger: float, log_smaller: float) -> float:
    """Return ``log(exp(log_larger) - exp(log_smaller))`` stably."""

    if np.isneginf(log_smaller):
        return log_larger
    difference = min(log_smaller - log_larger, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = log_larger + np.log(-np.expm1(difference))
    if np.isfinite(result):
        return float(result)
    raise RuntimeError(
        "truncated-normal interval probability is numerically indistinguishable"
    )


def _ordered_direction_log_probabilities(
    *,
    mean: ArrayLike,
    standard_deviation: ArrayLike,
    target_previous: ArrayLike,
    upper_threshold: ArrayLike,
) -> NDArray[np.float64]:
    """Numerically stable log probabilities in Cut/Hold/Hike order."""

    mean_array = np.asarray(mean, dtype=float)
    sd_array = np.asarray(standard_deviation, dtype=float)
    target_array = np.asarray(target_previous, dtype=float)
    threshold_array = np.asarray(upper_threshold, dtype=float)
    if np.any(~np.isfinite(mean_array)):
        raise VanDenHauweModelError("mean contains non-finite values")
    if np.any(~np.isfinite(sd_array)) or np.any(sd_array <= 0):
        raise VanDenHauweModelError(
            "standard_deviation must be finite and positive"
        )
    if np.any(~np.isfinite(target_array)):
        raise VanDenHauweModelError(
            "target_previous contains non-finite values"
        )
    if np.any(~np.isfinite(threshold_array)) or np.any(threshold_array <= 0):
        raise VanDenHauweModelError(
            "upper_threshold must be finite and positive"
        )

    standardized_lower = (target_array - mean_array) / sd_array
    standardized_upper = (
        target_array + threshold_array - mean_array
    ) / sd_array
    standardized_lower, standardized_upper = np.broadcast_arrays(
        standardized_lower, standardized_upper
    )
    log_cut = log_ndtr(standardized_lower)
    log_hike = log_ndtr(-standardized_upper)

    # For a positive interval use survival functions; otherwise use CDFs.
    # This avoids catastrophic cancellation when both CDF values round to one.
    positive_interval = standardized_lower > 0.0
    hold_larger = np.where(
        positive_interval,
        log_ndtr(-standardized_lower),
        log_ndtr(standardized_upper),
    )
    hold_smaller = np.where(
        positive_interval,
        log_ndtr(-standardized_upper),
        log_ndtr(standardized_lower),
    )
    difference = np.minimum(hold_smaller - hold_larger, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_hold = hold_larger + np.log(-np.expm1(difference))

    values = np.stack((log_cut, log_hold, log_hike), axis=-1)
    # The three analytic probabilities sum to one; this final normalization
    # removes only floating-point drift and keeps sequential log weights stable.
    return values - logsumexp(values, axis=-1, keepdims=True)


def _normalize_probabilities(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(probabilities, dtype=float)
    values = np.clip(values, 0.0, 1.0)
    totals = values.sum(axis=-1, keepdims=True)
    if np.any(~np.isfinite(values)) or np.any(totals <= 0):
        raise RuntimeError("invalid ordered-probit probability values")
    return values / totals


def _importance_effective_sample_size(
    normalized_log_weights: NDArray[np.float64],
) -> float:
    """Compute ``1 / sum(w**2)`` without exponentiating tiny weights."""

    return float(np.exp(-logsumexp(2.0 * normalized_log_weights)))


def _effective_sample_size(values: NDArray[np.float64]) -> float:
    series = np.asarray(values, dtype=float)
    n = len(series)
    if n < 3:
        return float(n)
    centered = series - series.mean()
    variance = float(centered @ centered) / n
    if variance <= np.finfo(float).eps:
        return float(n)
    autocorrelation_sum = 0.0
    # Initial-positive-sequence estimate; sufficient as a transparent warning
    # diagnostic for this deliberately single-chain implementation.
    for lag in range(1, n):
        covariance = float(centered[:-lag] @ centered[lag:]) / (n - lag)
        autocorrelation = covariance / variance
        if autocorrelation <= 0:
            break
        autocorrelation_sum += autocorrelation
    ess = n / (1.0 + 2.0 * autocorrelation_sum)
    return float(np.clip(ess, 1.0, n))


def _count_constraint_violations(
    draws: PosteriorDraws,
    decisions: NDArray[np.float64],
    target_previous: NDArray[np.float64],
) -> int:
    violations = 0
    observed_indices = np.flatnonzero(~np.isnan(decisions))
    for draw_index, upper_threshold in enumerate(draws.upper_threshold):
        offset = draws.latent_rate[draw_index] - target_previous
        for t in observed_indices:
            if decisions[t] == CUT and offset[t] > 1e-12:
                violations += 1
            elif decisions[t] == HOLD and not (
                offset[t] > -1e-12 and offset[t] <= upper_threshold + 1e-12
            ):
                violations += 1
            elif decisions[t] == HIKE and offset[t] <= upper_threshold - 1e-12:
                violations += 1
    return violations


__all__ = [
    "CUT",
    "HOLD",
    "HIKE",
    "ImportanceSamplingConfig",
    "MCMCConfig",
    "PosteriorDraws",
    "RecursiveImportanceForecast",
    "SamplerDiagnostics",
    "VanDenHauwe2013Result",
    "VanDenHauweModelError",
    "fit_van_den_hauwe_2013",
    "latent_truncation_bounds",
    "ordered_direction_probabilities",
    "recursive_importance_forecast",
]
