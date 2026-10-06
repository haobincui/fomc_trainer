"""Pure model helpers for FOMC decision-comparison baselines.

All probability matrices in this module use the fixed column order
``(cut, hold, hike)``.  The functions accept in-memory arrays and do not read or
write project artifacts, which keeps the information-set and expanding-window
bookkeeping in the calling job.

The statistical dependencies are imported lazily.  This module can therefore
be imported by data-preparation and reporting code even when an optional model
dependency is unavailable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import numpy as np


class DecisionClass(str, Enum):
    """Canonical three-way FOMC decision labels."""

    CUT = "cut"
    HOLD = "hold"
    HIKE = "hike"


CUT = DecisionClass.CUT.value
HOLD = DecisionClass.HOLD.value
HIKE = DecisionClass.HIKE.value
DECISION_CLASSES = (CUT, HOLD, HIKE)
DECISION_CLASS_TO_INDEX = {label: index for index, label in enumerate(DECISION_CLASSES)}

MIN_EXPANDING_TRAINING_OBSERVATIONS = 30
MIN_EXPANDING_CLASS_COUNT = 3

KAUPPI_MNL_C = 1.0
KAUPPI_MNL_SOLVER = "lbfgs"
KAUPPI_MNL_MAX_ITER = 5_000

ORDERED_PROBIT_METHOD = "bfgs"
ORDERED_PROBIT_MAX_ITER = 5_000

# Hamilton and Jorda's authors-revised 2001 weekly ACH(0,1) and ordered-probit
# estimates.  The ACH constant below is the denominator-scale value reported
# in table 5 (the replication program stores this value minus one before its
# pasted-link transformation).  These fixed estimates make this a published-
# coefficient operational application, not a re-estimation on the task roster.
HJ_ACH_LAG_DURATION = 0.067460919
HJ_ACH_CONSTANT = 30.390652
HJ_ACH_FOMC_WEEK = -23.045727
HJ_ACH_ABSOLUTE_SPREAD = -8.2087573
HJ_ACH_PASTING_DELTA = 0.1
HJ_ACH_PASTING_EPSILON = 0.0001
HJ_OP_PREVIOUS_CHANGE = 2.5449149
HJ_OP_SPREAD = 0.54142729
HJ_OP_THRESHOLDS = (-1.8948826, -0.42001991, -0.005251548, 1.5173916)
HJ_OP_MARKS_BP = (-50, -25, 0, 25, 50)

ORDINAL_RF_N_ESTIMATORS = 500
ORDINAL_RF_MIN_SAMPLES_LEAF = 5
ORDINAL_RF_CLASS_WEIGHT = "balanced_subsample"
ORDINAL_RF_MAX_FEATURES = "sqrt"
ORDINAL_RF_RANDOM_STATE = 20_260_827

FUTURES_MOVE_GRID_BP = tuple(range(-100, 101, 25))


class ProbabilityValidationError(ValueError):
    """Raised when values do not form valid cut/hold/hike probabilities."""


class ModelConvergenceError(RuntimeError):
    """Raised when an estimator reports that optimization did not converge."""


@dataclass(frozen=True)
class ExpandingWindowEligibility:
    """Auditable result of the fixed expanding-window training gate."""

    eligible: bool
    n_observations: int
    class_counts: dict[str, int]
    reasons: tuple[str, ...]


def _coerce_decision_label(value: Any) -> str:
    if isinstance(value, DecisionClass):
        return value.value
    if isinstance(value, str):
        label = value.strip().lower()
        if label in DECISION_CLASS_TO_INDEX:
            return label
    raise ValueError(
        f"invalid decision label {value!r}; expected one of {DECISION_CLASSES}"
    )


def _coerce_labels(
    labels: Sequence[Any], *, expected_rows: int | None = None
) -> np.ndarray:
    try:
        raw = list(labels)
    except TypeError as exc:
        raise ValueError("decision labels must be a one-dimensional sequence") from exc
    normalized = np.asarray(
        [_coerce_decision_label(value) for value in raw], dtype=object
    )
    if normalized.ndim != 1:
        raise ValueError("decision labels must be one-dimensional")
    if expected_rows is not None and len(normalized) != expected_rows:
        raise ValueError(
            f"label count {len(normalized)} does not match training rows {expected_rows}"
        )
    return normalized


def _encode_labels(labels: Sequence[Any], *, expected_rows: int) -> np.ndarray:
    normalized = _coerce_labels(labels, expected_rows=expected_rows)
    present = set(normalized.tolist())
    missing = [label for label in DECISION_CLASSES if label not in present]
    if missing:
        raise ValueError(
            "training labels must contain all three decision classes; "
            f"missing {missing}"
        )
    return np.asarray(
        [DECISION_CLASS_TO_INDEX[label] for label in normalized], dtype=int
    )


def _as_feature_matrix(values: Any, *, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric two-dimensional matrix") from exc
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one row and one feature")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains a non-finite value")
    return matrix


def _coerce_model_inputs(
    x_train: Any, y_train: Sequence[Any], x_test: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = _as_feature_matrix(x_train, name="x_train")
    test = _as_feature_matrix(x_test, name="x_test")
    if train.shape[1] != test.shape[1]:
        raise ValueError(
            "x_train and x_test must contain the same number of feature columns"
        )
    encoded_labels = _encode_labels(y_train, expected_rows=train.shape[0])
    return train, encoded_labels, test


def validate_probability_matrix(
    probabilities: Any, *, atol: float = 1e-8
) -> np.ndarray:
    """Validate and copy one row or a matrix of three-class probabilities.

    A one-dimensional input is returned as a one-dimensional array.  This
    function validates rather than repairs: materially invalid bounds or row
    sums raise :class:`ProbabilityValidationError`.
    """

    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be a finite non-negative number")
    try:
        values = np.asarray(probabilities, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ProbabilityValidationError("probabilities must be numeric") from exc
    if values.ndim not in (1, 2):
        raise ProbabilityValidationError(
            "probabilities must be one row or a two-dimensional matrix"
        )
    if values.shape[-1] != len(DECISION_CLASSES):
        raise ProbabilityValidationError(
            f"probabilities must have {len(DECISION_CLASSES)} columns in "
            f"{DECISION_CLASSES} order"
        )
    if values.ndim == 2 and values.shape[0] == 0:
        raise ProbabilityValidationError("probability matrix must contain a row")
    if not np.isfinite(values).all():
        raise ProbabilityValidationError("probabilities contain a non-finite value")
    if np.any(values < -atol) or np.any(values > 1.0 + atol):
        raise ProbabilityValidationError("probabilities must lie in [0, 1]")
    row_sums = values.sum(axis=-1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=atol):
        raise ProbabilityValidationError("each probability row must sum to one")
    return values.copy()


def _validate_tie_priority(
    tie_priority: Sequence[str | DecisionClass],
) -> tuple[str, ...]:
    normalized = tuple(_coerce_decision_label(value) for value in tie_priority)
    if len(normalized) != len(DECISION_CLASSES) or set(normalized) != set(
        DECISION_CLASSES
    ):
        raise ValueError(
            "tie_priority must list each of cut, hold, and hike exactly once"
        )
    return normalized


def deterministic_argmax(
    probabilities: Any,
    *,
    tie_priority: Sequence[str | DecisionClass] | None = None,
    tie_atol: float = 0.0,
) -> str | None:
    """Return the unique maximum class, or ``None`` for an unresolved tie.

    No implicit tie breaker is used.  A caller that wants to resolve ties must
    explicitly provide a complete ``tie_priority`` permutation.
    """

    row = validate_probability_matrix(probabilities)
    if row.ndim != 1:
        raise ProbabilityValidationError(
            "deterministic_argmax expects exactly one probability row"
        )
    if not np.isfinite(tie_atol) or tie_atol < 0:
        raise ValueError("tie_atol must be a finite non-negative number")
    maximum = float(np.max(row))
    winners = np.flatnonzero(np.isclose(row, maximum, rtol=0.0, atol=tie_atol))
    if len(winners) == 1:
        return DECISION_CLASSES[int(winners[0])]
    if tie_priority is None:
        return None
    priority = _validate_tie_priority(tie_priority)
    tied_labels = {DECISION_CLASSES[int(index)] for index in winners}
    return next(label for label in priority if label in tied_labels)


def deterministic_argmax_rows(
    probabilities: Any,
    *,
    tie_priority: Sequence[str | DecisionClass] | None = None,
    tie_atol: float = 0.0,
) -> tuple[str | None, ...]:
    """Apply :func:`deterministic_argmax` to every row of a probability matrix."""

    matrix = validate_probability_matrix(probabilities)
    if matrix.ndim != 2:
        raise ProbabilityValidationError(
            "deterministic_argmax_rows expects a two-dimensional matrix"
        )
    return tuple(
        deterministic_argmax(row, tie_priority=tie_priority, tie_atol=tie_atol)
        for row in matrix
    )


def expanding_window_eligibility(
    labels: Sequence[str | DecisionClass],
) -> ExpandingWindowEligibility:
    """Evaluate the fixed ``n >= 30`` and ``min(class count) >= 3`` gate."""

    normalized = _coerce_labels(labels)
    counts = {
        label: int(np.count_nonzero(normalized == label)) for label in DECISION_CLASSES
    }
    reasons: list[str] = []
    if len(normalized) < MIN_EXPANDING_TRAINING_OBSERVATIONS:
        reasons.append(
            "training window has "
            f"{len(normalized)} observations; requires at least "
            f"{MIN_EXPANDING_TRAINING_OBSERVATIONS}"
        )
    for label in DECISION_CLASSES:
        if counts[label] < MIN_EXPANDING_CLASS_COUNT:
            reasons.append(
                f"class {label!r} has {counts[label]} observations; requires at "
                f"least {MIN_EXPANDING_CLASS_COUNT}"
            )
    return ExpandingWindowEligibility(
        eligible=not reasons,
        n_observations=len(normalized),
        class_counts=counts,
        reasons=tuple(reasons),
    )


def _require_sklearn_logistic_regression():
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Kauppi MNL requires scikit-learn; install scikit-learn>=1.5"
        ) from exc
    return LogisticRegression


def _require_sklearn_standard_scaler():
    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Kauppi MNL standardization requires scikit-learn; install "
            "scikit-learn>=1.5"
        ) from exc
    return StandardScaler


def _require_sklearn_random_forest_classifier():
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "cumulative ordinal RF requires scikit-learn; install scikit-learn>=1.5"
        ) from exc
    return RandomForestClassifier


def _require_statsmodels_ordered_model():
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "ordered probit requires statsmodels; install statsmodels>=0.14"
        ) from exc
    return OrderedModel


def _map_encoded_class_probabilities(
    probabilities: Any, estimator_classes: Sequence[Any]
) -> np.ndarray:
    raw = np.asarray(probabilities, dtype=float)
    if raw.ndim != 2:
        raise ProbabilityValidationError("estimator probabilities must be a matrix")
    classes = np.asarray(estimator_classes)
    if raw.shape[1] != len(classes):
        raise ProbabilityValidationError(
            "estimator probability columns do not match estimator classes"
        )
    output = np.zeros((raw.shape[0], len(DECISION_CLASSES)), dtype=float)
    seen: set[int] = set()
    for column, raw_class in enumerate(classes):
        try:
            encoded_class = int(raw_class)
        except (TypeError, ValueError) as exc:
            raise ProbabilityValidationError(
                f"unexpected estimator class {raw_class!r}"
            ) from exc
        if encoded_class not in range(len(DECISION_CLASSES)) or encoded_class in seen:
            raise ProbabilityValidationError(
                f"unexpected estimator classes {classes.tolist()!r}"
            )
        output[:, encoded_class] = raw[:, column]
        seen.add(encoded_class)
    if seen != set(range(len(DECISION_CLASSES))):
        raise ProbabilityValidationError(
            "estimator did not produce all cut/hold/hike probability columns"
        )
    return validate_probability_matrix(output)


def kauppi_mnl_predict_proba(
    x_train: Any, y_train: Sequence[str | DecisionClass], x_test: Any
) -> np.ndarray:
    """Fit the fixed sklearn Kauppi-style MNL and predict class probabilities.

    Every feature is standardized using parameters fitted only on ``x_train``;
    ``x_test`` is transformed with that frozen training-window scaler.
    """

    train, encoded_labels, test = _coerce_model_inputs(x_train, y_train, x_test)
    StandardScaler = _require_sklearn_standard_scaler()
    scaler = StandardScaler()
    standardized_train = scaler.fit_transform(train)
    standardized_test = scaler.transform(test)
    LogisticRegression = _require_sklearn_logistic_regression()
    estimator = LogisticRegression(
        C=KAUPPI_MNL_C,
        solver=KAUPPI_MNL_SOLVER,
        max_iter=KAUPPI_MNL_MAX_ITER,
    )
    estimator.fit(standardized_train, encoded_labels)
    return _map_encoded_class_probabilities(
        estimator.predict_proba(standardized_test), estimator.classes_
    )


def ordered_probit_predict_proba(
    x_train: Any, y_train: Sequence[str | DecisionClass], x_test: Any
) -> np.ndarray:
    """Fit a probit ``OrderedModel`` and return cut/hold/hike probabilities.

    Statsmodels ordered models must not receive an explicit or implicit constant
    feature.  A non-converged optimizer result is rejected rather than silently
    used for prediction.
    """

    train, encoded_labels, test = _coerce_model_inputs(x_train, y_train, x_test)
    OrderedModel = _require_statsmodels_ordered_model()
    model = OrderedModel(encoded_labels, train, distr="probit")
    result = model.fit(
        method=ORDERED_PROBIT_METHOD,
        maxiter=ORDERED_PROBIT_MAX_ITER,
        disp=False,
    )
    convergence = getattr(result, "mle_retvals", {}).get("converged")
    if convergence is not True and not (
        isinstance(convergence, np.bool_) and bool(convergence)
    ):
        raise ModelConvergenceError(
            "statsmodels OrderedModel probit did not report convergence"
        )
    raw = model.predict(result.params, exog=test)
    return _map_encoded_class_probabilities(raw, model.labels)


def _finite_scalar(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite and numeric")
    return result


def hamilton_jorda_ach_hazard(
    *,
    previous_duration_weeks: float,
    spread_pp: float,
    fomc_meeting: bool = True,
) -> float:
    """Apply the authors-revised weekly ACH(0,1) published coefficients.

    ``spread_pp`` is the six-month Treasury-bill rate minus the effective
    federal funds rate, in percentage points.  The smooth pasted link is the
    one used by Hamilton and Jorda's official GAUSS replication program.  A
    target meeting is evaluated as an FOMC week.
    """

    duration = _finite_scalar(
        previous_duration_weeks, name="previous_duration_weeks"
    )
    spread = _finite_scalar(spread_pp, name="spread_pp")
    if duration <= 0.0:
        raise ValueError("previous_duration_weeks must be positive")
    if not isinstance(fomc_meeting, (bool, np.bool_)):
        raise ValueError("fomc_meeting must be boolean")
    index = (
        HJ_ACH_CONSTANT
        + HJ_ACH_LAG_DURATION * duration
        + HJ_ACH_FOMC_WEEK * float(bool(fomc_meeting))
        + HJ_ACH_ABSOLUTE_SPREAD * abs(spread)
    )
    if index <= 1.0:
        denominator = 1.0 + HJ_ACH_PASTING_EPSILON
    elif index <= 1.0 + HJ_ACH_PASTING_DELTA:
        offset = index - 1.0
        denominator = (
            1.0
            + HJ_ACH_PASTING_EPSILON
            + 2.0
            * HJ_ACH_PASTING_DELTA
            * offset**2
            / (HJ_ACH_PASTING_DELTA**2 + offset**2)
        )
    else:
        denominator = index + HJ_ACH_PASTING_EPSILON
    hazard = 1.0 / denominator
    if not 0.0 < hazard < 1.0:
        raise ProbabilityValidationError("Hamilton-Jorda ACH hazard is outside (0, 1)")
    return hazard


def hamilton_jorda_ordered_probit_mark_probabilities(
    *, previous_change_pp: float, spread_pp: float
) -> dict[int, float]:
    """Return conditional probabilities for the paper's five target-change bins."""

    previous_change = _finite_scalar(previous_change_pp, name="previous_change_pp")
    spread = _finite_scalar(spread_pp, name="spread_pp")
    eta = HJ_OP_PREVIOUS_CHANGE * previous_change + HJ_OP_SPREAD * spread

    def normal_cdf(value: float) -> float:
        return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

    cumulative = [normal_cdf(threshold - eta) for threshold in HJ_OP_THRESHOLDS]
    probabilities = np.asarray(
        [
            cumulative[0],
            cumulative[1] - cumulative[0],
            cumulative[2] - cumulative[1],
            cumulative[3] - cumulative[2],
            1.0 - cumulative[3],
        ],
        dtype=float,
    )
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < -1e-12)
        or not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-10)
    ):
        raise ProbabilityValidationError(
            "Hamilton-Jorda ordered-probit mark probabilities are invalid"
        )
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum()
    return {
        mark: float(probability)
        for mark, probability in zip(HJ_OP_MARKS_BP, probabilities, strict=True)
    }


def hamilton_jorda_ach_op_action_probabilities(
    *,
    previous_change_pp: float,
    previous_duration_weeks: float,
    spread_pp: float,
    fomc_meeting: bool = True,
) -> dict[int, float]:
    """Combine the fixed ACH hazard and conditional mark model.

    The zero-action mass is ``1-h`` plus the ACH-event mass assigned to the
    ordered-probit zero bin, matching the paper's marked-point forecast.  The
    returned keys are signed basis-point actions.
    """

    hazard = hamilton_jorda_ach_hazard(
        previous_duration_weeks=previous_duration_weeks,
        spread_pp=spread_pp,
        fomc_meeting=fomc_meeting,
    )
    conditional = hamilton_jorda_ordered_probit_mark_probabilities(
        previous_change_pp=previous_change_pp,
        spread_pp=spread_pp,
    )
    probabilities = {
        mark: hazard * probability for mark, probability in conditional.items()
    }
    probabilities[0] += 1.0 - hazard
    if any(value < 0.0 or value > 1.0 for value in probabilities.values()) or not math.isclose(
        sum(probabilities.values()), 1.0, abs_tol=1e-10
    ):
        raise ProbabilityValidationError(
            "Hamilton-Jorda ACH-OP action probabilities are invalid"
        )
    return probabilities


def project_cumulative_probabilities(
    probability_gt_cut: Any, probability_gt_hold: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Project two cumulative probabilities onto ``q_cut >= q_hold``.

    The pointwise Euclidean projection for a violation replaces both values by
    their mean.  This is the two-point isotonic regression solution.
    """

    try:
        q1, q2 = np.broadcast_arrays(
            np.asarray(probability_gt_cut, dtype=float),
            np.asarray(probability_gt_hold, dtype=float),
        )
    except (TypeError, ValueError) as exc:
        raise ProbabilityValidationError(
            "cumulative probabilities must be numeric and broadcast-compatible"
        ) from exc
    if not np.isfinite(q1).all() or not np.isfinite(q2).all():
        raise ProbabilityValidationError(
            "cumulative probabilities contain a non-finite value"
        )
    if np.any(q1 < 0.0) or np.any(q1 > 1.0) or np.any(q2 < 0.0) or np.any(q2 > 1.0):
        raise ProbabilityValidationError("cumulative probabilities must lie in [0, 1]")
    violations = q1 < q2
    pooled = (q1 + q2) / 2.0
    projected_q1 = np.where(violations, pooled, q1)
    projected_q2 = np.where(violations, pooled, q2)
    return projected_q1.copy(), projected_q2.copy()


def cumulative_to_decision_probabilities(
    probability_gt_cut: Any, probability_gt_hold: Any
) -> np.ndarray:
    """Project cumulative estimates and recover cut/hold/hike probabilities."""

    q1, q2 = project_cumulative_probabilities(probability_gt_cut, probability_gt_hold)
    if q1.ndim == 0:
        probabilities = np.asarray([1.0 - q1, q1 - q2, q2], dtype=float)
    else:
        probabilities = np.stack((1.0 - q1, q1 - q2, q2), axis=-1)
    return validate_probability_matrix(probabilities)


def _positive_class_probability(estimator: Any, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(features), dtype=float)
    classes = np.asarray(estimator.classes_)
    positive_columns = np.flatnonzero(classes == 1)
    if probabilities.ndim != 2 or len(positive_columns) != 1:
        raise ProbabilityValidationError(
            "binary forest did not expose exactly one positive-class column"
        )
    return probabilities[:, int(positive_columns[0])]


def cumulative_ordinal_rf_predict_proba(
    x_train: Any,
    y_train: Sequence[str | DecisionClass],
    x_test: Any,
    *,
    random_state: int = ORDINAL_RF_RANDOM_STATE,
) -> np.ndarray:
    """Fit the fixed two-forest cumulative ordinal approximation.

    This is an ordinal approximation, not an exact replication of Yoon and Fan
    (2024).  The two raw cumulative estimates are isotonic-projected before the
    three class probabilities are recovered.
    """

    train, encoded_labels, test = _coerce_model_inputs(x_train, y_train, x_test)
    if isinstance(random_state, bool) or not isinstance(
        random_state, (int, np.integer)
    ):
        raise ValueError("random_state must be an integer")
    seed = int(random_state)
    if seed < 0 or seed >= np.iinfo(np.uint32).max:
        raise ValueError("random_state must satisfy 0 <= seed < 2**32 - 1")
    RandomForestClassifier = _require_sklearn_random_forest_classifier()
    common_parameters = {
        "n_estimators": ORDINAL_RF_N_ESTIMATORS,
        "min_samples_leaf": ORDINAL_RF_MIN_SAMPLES_LEAF,
        "class_weight": ORDINAL_RF_CLASS_WEIGHT,
        "max_features": ORDINAL_RF_MAX_FEATURES,
    }
    gt_cut_estimator = RandomForestClassifier(**common_parameters, random_state=seed)
    gt_hold_estimator = RandomForestClassifier(
        **common_parameters, random_state=seed + 1
    )
    gt_cut_estimator.fit(train, (encoded_labels > 0).astype(int))
    gt_hold_estimator.fit(train, (encoded_labels > 1).astype(int))
    q1 = _positive_class_probability(gt_cut_estimator, test)
    q2 = _positive_class_probability(gt_hold_estimator, test)
    return cumulative_to_decision_probabilities(q1, q2)


def fed_funds_futures_implied_rate(futures_price: float) -> float:
    """Convert a 30-day Fed Funds futures price to an annualized percent rate."""

    try:
        price = float(futures_price)
    except (TypeError, ValueError) as exc:
        raise ValueError("futures_price must be finite and numeric") from exc
    if not np.isfinite(price):
        raise ValueError("futures_price must be finite and numeric")
    return 100.0 - price


def implied_post_meeting_rate(
    monthly_implied_rate: float,
    current_rate: float,
    meeting_day: int,
    days_in_month: int,
) -> float:
    """Recover the implied post-meeting rate from a monthly-average rate.

    ``meeting_day`` counts days up to and including the meeting, matching
    ``monthly = (d / D) * current + ((D - d) / D) * post``.  Rates are in
    percentage points, not basis points.
    """

    try:
        monthly = float(monthly_implied_rate)
        current = float(current_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "monthly_implied_rate and current_rate must be numeric"
        ) from exc
    if not np.isfinite(monthly) or not np.isfinite(current):
        raise ValueError("monthly_implied_rate and current_rate must be finite")
    if isinstance(meeting_day, bool) or not isinstance(meeting_day, (int, np.integer)):
        raise ValueError("meeting_day must be an integer")
    if isinstance(days_in_month, bool) or not isinstance(
        days_in_month, (int, np.integer)
    ):
        raise ValueError("days_in_month must be an integer")
    day = int(meeting_day)
    total_days = int(days_in_month)
    if total_days <= 1 or day < 1 or day >= total_days:
        raise ValueError("meeting_day must satisfy 1 <= meeting_day < days_in_month")
    return (total_days * monthly - day * current) / (total_days - day)


def monthly_implied_post_rate(
    futures_price: float,
    current_rate: float,
    meeting_day: int,
    days_in_month: int,
) -> float:
    """Convert a futures price directly to its implied post-meeting rate."""

    return implied_post_meeting_rate(
        fed_funds_futures_implied_rate(futures_price),
        current_rate,
        meeting_day,
        days_in_month,
    )


def interpolate_futures_move_probabilities(expected_move_bp: float) -> np.ndarray:
    """Linearly interpolate an expected move on the fixed 25bp outcome grid.

    Values outside ``[-100, 100]`` are winsorized to the nearest endpoint.  The
    returned nine probabilities correspond to :data:`FUTURES_MOVE_GRID_BP`.
    """

    try:
        move = float(expected_move_bp)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected_move_bp must be finite and numeric") from exc
    if not np.isfinite(move):
        raise ValueError("expected_move_bp must be finite and numeric")
    grid = np.asarray(FUTURES_MOVE_GRID_BP, dtype=float)
    clipped = float(np.clip(move, grid[0], grid[-1]))
    probabilities = np.zeros(len(grid), dtype=float)
    upper_index = int(np.searchsorted(grid, clipped, side="left"))
    if upper_index == 0:
        probabilities[0] = 1.0
    elif upper_index == len(grid):
        probabilities[-1] = 1.0
    elif clipped == grid[upper_index]:
        probabilities[upper_index] = 1.0
    else:
        lower_index = upper_index - 1
        upper_weight = (clipped - grid[lower_index]) / (
            grid[upper_index] - grid[lower_index]
        )
        probabilities[lower_index] = 1.0 - upper_weight
        probabilities[upper_index] = upper_weight
    return probabilities


def futures_direction_probabilities(expected_move_bp: float) -> np.ndarray:
    """Aggregate interpolated move-grid mass into cut/hold/hike probabilities."""

    move_probabilities = interpolate_futures_move_probabilities(expected_move_bp)
    grid = np.asarray(FUTURES_MOVE_GRID_BP)
    probabilities = np.asarray(
        [
            move_probabilities[grid < 0].sum(),
            move_probabilities[grid == 0].sum(),
            move_probabilities[grid > 0].sum(),
        ],
        dtype=float,
    )
    return validate_probability_matrix(probabilities)


__all__ = [
    "CUT",
    "HOLD",
    "HIKE",
    "DECISION_CLASSES",
    "DECISION_CLASS_TO_INDEX",
    "DecisionClass",
    "ExpandingWindowEligibility",
    "FUTURES_MOVE_GRID_BP",
    "HJ_ACH_ABSOLUTE_SPREAD",
    "HJ_ACH_CONSTANT",
    "HJ_ACH_FOMC_WEEK",
    "HJ_ACH_LAG_DURATION",
    "HJ_ACH_PASTING_DELTA",
    "HJ_ACH_PASTING_EPSILON",
    "HJ_OP_MARKS_BP",
    "HJ_OP_PREVIOUS_CHANGE",
    "HJ_OP_SPREAD",
    "HJ_OP_THRESHOLDS",
    "KAUPPI_MNL_C",
    "KAUPPI_MNL_MAX_ITER",
    "KAUPPI_MNL_SOLVER",
    "MIN_EXPANDING_CLASS_COUNT",
    "MIN_EXPANDING_TRAINING_OBSERVATIONS",
    "ModelConvergenceError",
    "ORDERED_PROBIT_MAX_ITER",
    "ORDERED_PROBIT_METHOD",
    "ORDINAL_RF_CLASS_WEIGHT",
    "ORDINAL_RF_MAX_FEATURES",
    "ORDINAL_RF_MIN_SAMPLES_LEAF",
    "ORDINAL_RF_N_ESTIMATORS",
    "ORDINAL_RF_RANDOM_STATE",
    "ProbabilityValidationError",
    "cumulative_ordinal_rf_predict_proba",
    "cumulative_to_decision_probabilities",
    "deterministic_argmax",
    "deterministic_argmax_rows",
    "expanding_window_eligibility",
    "fed_funds_futures_implied_rate",
    "futures_direction_probabilities",
    "hamilton_jorda_ach_hazard",
    "hamilton_jorda_ach_op_action_probabilities",
    "hamilton_jorda_ordered_probit_mark_probabilities",
    "implied_post_meeting_rate",
    "interpolate_futures_move_probabilities",
    "kauppi_mnl_predict_proba",
    "monthly_implied_post_rate",
    "ordered_probit_predict_proba",
    "project_cumulative_probabilities",
    "validate_probability_matrix",
]
