"""Statistical primitives for the CHK3 sentiment--market beta study.

The module is deliberately independent of generation, sentiment, and market
artifact schemas.  Those stages produce sealed ledgers; this module supplies
the small, auditable numerical core used by the final estimator:

* OLS with a Newey--West/HAC covariance matrix;
* paired calendar-block resampling shared by every text arm; and
* within-meeting paired replicate resampling for stochastic generations.

The analysis is descriptive association preservation.  Nothing here supports
a causal interpretation of synthetic text that was never publicly released.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_260_817
DEFAULT_HAC_LAG = 4
SYNTHETIC_ARMS = ("chk0", "chk1", "chk3")


class BetaStatisticsError(RuntimeError):
    """A statistical input or reproducibility contract failed closed."""


def canonical_json(value: Any) -> str:
    """Return the canonical JSON encoding used for index-plan hashes."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise BetaStatisticsError(f"non-canonical payload: {exc}") from exc


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OlsHacResult:
    coefficients: np.ndarray
    standard_errors: np.ndarray
    covariance: np.ndarray
    residuals: np.ndarray
    fitted: np.ndarray
    nobs: int
    rank: int
    dof_resid: int
    r_squared: float
    adjusted_r_squared: float
    hac_lag: int


def _finite_vector(name: str, value: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise BetaStatisticsError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise BetaStatisticsError(f"{name} contains non-finite values")
    return array


def _finite_matrix(
    name: str, value: Sequence[Sequence[float]] | np.ndarray
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise BetaStatisticsError(f"{name} must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise BetaStatisticsError(f"{name} contains non-finite values")
    return array


def ols_hac(
    y: Sequence[float] | np.ndarray,
    x: Sequence[Sequence[float]] | np.ndarray,
    *,
    hac_lag: int = DEFAULT_HAC_LAG,
    finite_sample: bool = True,
) -> OlsHacResult:
    """Fit OLS and a Bartlett-kernel Newey--West covariance matrix.

    ``x`` must already include an intercept when desired.  Rows are assumed to
    be in chronological order.  The function rejects rank-deficient designs
    instead of silently using unidentified coefficients.
    """

    y_array = _finite_vector("y", y)
    x_array = _finite_matrix("x", x)
    if x_array.shape[0] != len(y_array):
        raise BetaStatisticsError("x/y row counts differ")
    nobs, columns = x_array.shape
    if nobs <= columns:
        raise BetaStatisticsError("OLS requires more observations than columns")
    if not isinstance(hac_lag, int) or hac_lag < 0 or hac_lag >= nobs:
        raise BetaStatisticsError("hac_lag must be an integer in [0, nobs)")

    coefficients, _, rank, _ = np.linalg.lstsq(x_array, y_array, rcond=None)
    if int(rank) != columns:
        raise BetaStatisticsError(
            f"rank-deficient design: rank={int(rank)}, columns={columns}"
        )
    fitted = x_array @ coefficients
    residuals = y_array - fitted

    xtx_inverse = np.linalg.inv(x_array.T @ x_array)
    xu = x_array * residuals[:, None]
    meat = xu.T @ xu
    for lag in range(1, hac_lag + 1):
        weight = 1.0 - lag / (hac_lag + 1.0)
        cross = xu[lag:].T @ xu[:-lag]
        meat += weight * (cross + cross.T)
    covariance = xtx_inverse @ meat @ xtx_inverse
    if finite_sample:
        covariance *= nobs / (nobs - columns)
    covariance = (covariance + covariance.T) / 2.0
    diagonal = np.diag(covariance)
    if np.any(diagonal < -1e-12):
        raise BetaStatisticsError("HAC covariance has a materially negative diagonal")
    standard_errors = np.sqrt(np.maximum(diagonal, 0.0))

    centered = y_array - float(y_array.mean())
    total_ss = float(centered @ centered)
    residual_ss = float(residuals @ residuals)
    r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else float("nan")
    adjusted = (
        1.0 - (1.0 - r_squared) * (nobs - 1) / (nobs - columns)
        if math.isfinite(r_squared)
        else float("nan")
    )
    return OlsHacResult(
        coefficients=coefficients,
        standard_errors=standard_errors,
        covariance=covariance,
        residuals=residuals,
        fitted=fitted,
        nobs=nobs,
        rank=int(rank),
        dof_resid=nobs - columns,
        r_squared=r_squared,
        adjusted_r_squared=adjusted,
        hac_lag=hac_lag,
    )


@dataclass(frozen=True)
class PairedBlockBootstrapPlan:
    """Shared calendar-block and within-meeting replicate draws."""

    block_ids: tuple[str, ...]
    meeting_ids: tuple[str, ...]
    replicate_ids_by_meeting: tuple[tuple[int, ...], ...]
    replicate_draw_counts_by_meeting: tuple[int, ...]
    sampled_block_indices: np.ndarray
    sampled_replicate_positions: np.ndarray
    draws: int
    seed: int
    replicates_per_draw: int
    sha256: str


def make_paired_block_bootstrap_plan(
    *,
    block_ids: Sequence[str],
    meeting_ids: Sequence[str],
    meeting_block_ids: Sequence[str],
    replicate_ids_by_meeting: Mapping[str, Sequence[int]],
    replicate_draw_counts_by_meeting: Mapping[str, int] | None = None,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    replicates_per_draw: int = 5,
) -> PairedBlockBootstrapPlan:
    """Create model-shared calendar-block and replicate resampling indices.

    ``sampled_replicate_positions[d, m, k]`` indexes the ordered *available*
    replicate inventory of meeting ``m``.  ``replicate_draw_counts_by_meeting``
    freezes how many leading positions contribute to each meeting mean.  This
    supports an observed-size bootstrap (n_i draws from n_i paired-complete
    replicates) while retaining one rectangular, hash-bound index tensor.
    """

    blocks = tuple(str(value) for value in block_ids)
    meetings = tuple(str(value) for value in meeting_ids)
    meeting_blocks = tuple(str(value) for value in meeting_block_ids)
    if not blocks or len(set(blocks)) != len(blocks):
        raise BetaStatisticsError("block_ids must be non-empty and unique")
    if not meetings or len(set(meetings)) != len(meetings):
        raise BetaStatisticsError("meeting_ids must be non-empty and unique")
    if len(meeting_blocks) != len(meetings) or any(
        block not in set(blocks) for block in meeting_blocks
    ):
        raise BetaStatisticsError("meeting_block_ids do not match block inventory")
    if draws < 1 or replicates_per_draw < 1:
        raise BetaStatisticsError("bootstrap dimensions must be positive")

    inventories: list[tuple[int, ...]] = []
    draw_counts: list[int] = []
    for meeting in meetings:
        raw = replicate_ids_by_meeting.get(meeting)
        if raw is None:
            raise BetaStatisticsError(f"missing paired replicate inventory: {meeting}")
        values = tuple(int(value) for value in raw)
        if (
            not values
            or len(set(values)) != len(values)
            or any(value < 0 for value in values)
        ):
            raise BetaStatisticsError(f"invalid paired replicate inventory: {meeting}")
        inventories.append(values)
        raw_count = (
            replicates_per_draw
            if replicate_draw_counts_by_meeting is None
            else replicate_draw_counts_by_meeting.get(meeting)
        )
        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count < 1
            or raw_count > replicates_per_draw
        ):
            raise BetaStatisticsError(
                f"invalid replicate draw count: {meeting}:{raw_count}"
            )
        draw_counts.append(raw_count)
    if replicate_draw_counts_by_meeting is not None and set(
        replicate_draw_counts_by_meeting
    ) != set(meetings):
        raise BetaStatisticsError("replicate draw-count inventory drift")

    rng = np.random.default_rng(seed)
    block_indices = rng.integers(
        0,
        len(blocks),
        size=(draws, len(blocks)),
        dtype=np.uint16,
    )
    replicate_positions = np.empty(
        (draws, len(meetings), replicates_per_draw), dtype=np.uint8
    )
    for meeting_index, inventory in enumerate(inventories):
        replicate_positions[:, meeting_index, :] = rng.integers(
            0,
            len(inventory),
            size=(draws, replicates_per_draw),
            dtype=np.uint8,
        )

    metadata = {
        "schema_version": "chk3-beta-paired-calendar-block-bootstrap-plan-v1",
        "block_ids": list(blocks),
        "meeting_ids": list(meetings),
        "meeting_block_ids": list(meeting_blocks),
        "replicate_ids_by_meeting": {
            meeting: list(inventories[index]) for index, meeting in enumerate(meetings)
        },
        "replicate_draw_counts_by_meeting": {
            meeting: draw_counts[index] for index, meeting in enumerate(meetings)
        },
        "draws": draws,
        "seed": seed,
        "replicates_per_draw": replicates_per_draw,
        "sampled_block_indices_shape": list(block_indices.shape),
        "sampled_replicate_positions_shape": list(replicate_positions.shape),
        "sampled_block_indices_dtype": "uint16-le",
        "sampled_replicate_positions_dtype": "uint8",
        "numpy_version": np.__version__,
        "numpy_bit_generator": type(rng.bit_generator).__name__,
    }
    digest = hashlib.sha256()
    digest.update(canonical_json(metadata).encode("utf-8"))
    digest.update(block_indices.astype("<u2", copy=False).tobytes(order="C"))
    digest.update(replicate_positions.tobytes(order="C"))
    return PairedBlockBootstrapPlan(
        block_ids=blocks,
        meeting_ids=meetings,
        replicate_ids_by_meeting=tuple(inventories),
        replicate_draw_counts_by_meeting=tuple(draw_counts),
        sampled_block_indices=block_indices,
        sampled_replicate_positions=replicate_positions,
        draws=draws,
        seed=seed,
        replicates_per_draw=replicates_per_draw,
        sha256=digest.hexdigest(),
    )


def resampled_meeting_means(
    values_by_arm: Mapping[str, np.ndarray],
    plan: PairedBlockBootstrapPlan,
    *,
    draw_index: int,
) -> dict[str, np.ndarray]:
    """Apply one shared within-meeting replicate draw to every synthetic arm.

    Each arm array must have shape ``(meetings, max_replicate_id + 1)``.  Only
    the paired-complete replicate IDs frozen in the plan can be selected.
    """

    if not 0 <= draw_index < plan.draws:
        raise BetaStatisticsError("draw_index is outside the bootstrap plan")
    if set(values_by_arm) != set(SYNTHETIC_ARMS):
        raise BetaStatisticsError(
            f"synthetic arm inventory must be {list(SYNTHETIC_ARMS)}"
        )
    arrays = {
        arm: np.asarray(values_by_arm[arm], dtype=np.float64) for arm in SYNTHETIC_ARMS
    }
    expected_rows = len(plan.meeting_ids)
    widths = {array.shape[1] for array in arrays.values() if array.ndim == 2}
    if (
        any(
            array.ndim != 2 or array.shape[0] != expected_rows
            for array in arrays.values()
        )
        or len(widths) != 1
    ):
        raise BetaStatisticsError("synthetic score arrays have incompatible shapes")

    result = {arm: np.empty(expected_rows, dtype=np.float64) for arm in SYNTHETIC_ARMS}
    positions = plan.sampled_replicate_positions[draw_index]
    for meeting_index, inventory_tuple in enumerate(plan.replicate_ids_by_meeting):
        inventory = np.asarray(inventory_tuple, dtype=np.int64)
        draw_count = plan.replicate_draw_counts_by_meeting[meeting_index]
        selected = inventory[positions[meeting_index, :draw_count].astype(np.int64)]
        for arm in SYNTHETIC_ARMS:
            values = arrays[arm][meeting_index, selected]
            if not np.all(np.isfinite(values)):
                raise BetaStatisticsError(
                    f"paired replicate score is non-finite: {arm}/{plan.meeting_ids[meeting_index]}"
                )
            result[arm][meeting_index] = float(values.mean())
    return result


def sampled_meeting_row_indices(
    plan: PairedBlockBootstrapPlan,
    meeting_block_ids: Sequence[str],
    *,
    draw_index: int,
) -> np.ndarray:
    """Expand one sampled block sequence to chronological meeting row indices."""

    if not 0 <= draw_index < plan.draws:
        raise BetaStatisticsError("draw_index is outside the bootstrap plan")
    blocks = tuple(str(value) for value in meeting_block_ids)
    if len(blocks) != len(plan.meeting_ids):
        raise BetaStatisticsError("meeting block vector length drift")
    rows_by_block = {
        block: np.asarray(
            [index for index, value in enumerate(blocks) if value == block],
            dtype=np.int64,
        )
        for block in plan.block_ids
    }
    if any(not len(rows_by_block[block]) for block in plan.block_ids):
        raise BetaStatisticsError("a declared calendar block has no meetings")
    selected = [
        rows_by_block[plan.block_ids[int(index)]]
        for index in plan.sampled_block_indices[draw_index]
    ]
    return np.concatenate(selected)


def percentile_interval(
    values: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    array = _finite_vector("bootstrap values", values)
    if not 0.0 < confidence < 1.0:
        raise BetaStatisticsError("confidence must lie strictly between zero and one")
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(array, [tail, 1.0 - tail], method="linear")
    return float(low), float(high)


def two_sided_bootstrap_p_value(values: Sequence[float] | np.ndarray) -> float:
    """Return a plus-one-corrected two-sided sign probability around zero."""

    array = _finite_vector("bootstrap contrast", values)
    non_positive = int(np.count_nonzero(array <= 0.0))
    non_negative = int(np.count_nonzero(array >= 0.0))
    tail = min(non_positive, non_negative)
    return float(min(1.0, 2.0 * (tail + 1.0) / (len(array) + 1.0)))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    indexed: list[tuple[int, float]] = []
    for index, value in enumerate(p_values):
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise BetaStatisticsError("Holm p-values must be finite in [0,1]")
        indexed.append((index, numeric))
    indexed.sort(key=lambda item: item[1])
    adjusted = [0.0] * len(indexed)
    running = 0.0
    total = len(indexed)
    for rank, (original_index, value) in enumerate(indexed):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[original_index] = running
    return adjusted
