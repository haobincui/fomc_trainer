from __future__ import annotations

import numpy as np
import pytest

from jobs.eval.chk3_beta_statistics import (
    BetaStatisticsError,
    holm_adjust,
    make_paired_block_bootstrap_plan,
    ols_hac,
    percentile_interval,
    resampled_meeting_means,
    sampled_meeting_row_indices,
    two_sided_bootstrap_p_value,
)


def _plan(*, draws: int = 20):
    return make_paired_block_bootstrap_plan(
        block_ids=("1993", "1994"),
        meeting_ids=("m0", "m1", "m2", "m3"),
        meeting_block_ids=("1993", "1993", "1994", "1994"),
        replicate_ids_by_meeting={
            "m0": (0, 1, 2, 3, 4),
            "m1": (0, 1, 3, 4),
            "m2": (0, 1, 2, 3, 4),
            "m3": (0, 2, 3, 4),
        },
        draws=draws,
        seed=20260817,
        replicates_per_draw=5,
    )


def test_ols_hac_recovers_exact_linear_coefficients() -> None:
    current = np.linspace(-2.0, 2.0, 80)
    control = np.sin(np.arange(80, dtype=np.float64))
    x = np.column_stack([np.ones(80), current, control])
    y = 1.25 + 2.5 * current - 0.75 * control
    result = ols_hac(y, x, hac_lag=4)
    assert result.coefficients == pytest.approx([1.25, 2.5, -0.75], abs=1e-12)
    assert result.standard_errors == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    assert result.r_squared == pytest.approx(1.0)
    assert result.nobs == 80
    assert result.rank == 3


def test_ols_hac_rejects_rank_deficiency_and_bad_lag() -> None:
    x = np.column_stack([np.ones(10), np.ones(10)])
    with pytest.raises(BetaStatisticsError, match="rank-deficient"):
        ols_hac(np.arange(10), x)
    with pytest.raises(BetaStatisticsError, match="hac_lag"):
        ols_hac(
            np.arange(10), np.column_stack([np.ones(10), np.arange(10)]), hac_lag=10
        )


def test_paired_plan_is_reproducible_and_model_shared() -> None:
    first = _plan()
    second = _plan()
    assert first.sha256 == second.sha256
    assert np.array_equal(first.sampled_block_indices, second.sampled_block_indices)
    assert np.array_equal(
        first.sampled_replicate_positions, second.sampled_replicate_positions
    )

    base = np.arange(20, dtype=np.float64).reshape(4, 5)
    scores = {"chk0": base, "chk1": base + 100.0, "chk3": base + 200.0}
    means = resampled_meeting_means(scores, first, draw_index=0)
    assert means["chk1"] - means["chk0"] == pytest.approx(np.full(4, 100.0))
    assert means["chk3"] - means["chk1"] == pytest.approx(np.full(4, 100.0))


def test_paired_plan_never_selects_incomplete_replicates() -> None:
    plan = _plan(draws=100)
    scores = {
        arm: np.arange(20, dtype=np.float64).reshape(4, 5) + offset
        for arm, offset in zip(("chk0", "chk1", "chk3"), (0.0, 10.0, 20.0))
    }
    for array in scores.values():
        array[1, 2] = np.nan
        array[3, 1] = np.nan
    for draw in range(plan.draws):
        result = resampled_meeting_means(scores, plan, draw_index=draw)
        assert all(np.all(np.isfinite(value)) for value in result.values())


def test_paired_plan_uses_observed_replicate_count_per_meeting() -> None:
    plan = make_paired_block_bootstrap_plan(
        block_ids=("1993",),
        meeting_ids=("complete", "incomplete"),
        meeting_block_ids=("1993", "1993"),
        replicate_ids_by_meeting={
            "complete": (0, 1, 2, 3, 4),
            "incomplete": (0, 1, 3, 4),
        },
        replicate_draw_counts_by_meeting={"complete": 5, "incomplete": 4},
        draws=3,
        seed=20260817,
        replicates_per_draw=5,
    )
    assert plan.replicate_draw_counts_by_meeting == (5, 4)
    values = np.asarray([[0, 1, 2, 3, 4], [10, 20, np.nan, 30, 40]], dtype=float)
    scores = {arm: values.copy() for arm in ("chk0", "chk1", "chk3")}
    observed = resampled_meeting_means(scores, plan, draw_index=0)["chk0"]
    positions = plan.sampled_replicate_positions[0]
    complete_inventory = np.asarray((0, 1, 2, 3, 4))
    incomplete_inventory = np.asarray((0, 1, 3, 4))
    assert observed[0] == pytest.approx(
        values[0, complete_inventory[positions[0, :5]]].mean()
    )
    assert observed[1] == pytest.approx(
        values[1, incomplete_inventory[positions[1, :4]]].mean()
    )


def test_paired_plan_hash_binds_replicate_draw_counts() -> None:
    fixed_k5 = _plan(draws=3)
    observed_size = make_paired_block_bootstrap_plan(
        block_ids=fixed_k5.block_ids,
        meeting_ids=fixed_k5.meeting_ids,
        meeting_block_ids=("1993", "1993", "1994", "1994"),
        replicate_ids_by_meeting={
            meeting_id: inventory
            for meeting_id, inventory in zip(
                fixed_k5.meeting_ids,
                fixed_k5.replicate_ids_by_meeting,
                strict=True,
            )
        },
        replicate_draw_counts_by_meeting={
            meeting_id: len(inventory)
            for meeting_id, inventory in zip(
                fixed_k5.meeting_ids,
                fixed_k5.replicate_ids_by_meeting,
                strict=True,
            )
        },
        draws=3,
        seed=20260817,
        replicates_per_draw=5,
    )
    assert fixed_k5.sha256 != observed_size.sha256


def test_sampled_blocks_retain_whole_calendar_clusters() -> None:
    plan = _plan(draws=4)
    blocks = ("1993", "1993", "1994", "1994")
    for draw in range(plan.draws):
        indexes = sampled_meeting_row_indices(plan, blocks, draw_index=draw)
        assert len(indexes) == 4
        for offset in (0, 2):
            pair = indexes[offset : offset + 2].tolist()
            assert pair in ([0, 1], [2, 3])


def test_intervals_p_values_and_holm() -> None:
    values = np.arange(1.0, 101.0)
    low, high = percentile_interval(values)
    assert low == pytest.approx(3.475)
    assert high == pytest.approx(97.525)
    assert two_sided_bootstrap_p_value(values) == pytest.approx(2.0 / 101.0)
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_plan_rejects_missing_or_duplicate_inventory() -> None:
    with pytest.raises(BetaStatisticsError, match="unique"):
        make_paired_block_bootstrap_plan(
            block_ids=("1993", "1993"),
            meeting_ids=("m0",),
            meeting_block_ids=("1993",),
            replicate_ids_by_meeting={"m0": (0,)},
        )
    with pytest.raises(BetaStatisticsError, match="missing"):
        make_paired_block_bootstrap_plan(
            block_ids=("1993",),
            meeting_ids=("m0",),
            meeting_block_ids=("1993",),
            replicate_ids_by_meeting={},
        )
