from __future__ import annotations

import pytest

from jobs.retrain_v2 import compare_chk4_all237_three_models as comparison


def _row(sample_id: str, target: str, predicted: str, exact: bool = False) -> dict:
    return {
        "sample_id": sample_id,
        "target_direction": target,
        "target_magnitude_bp": 0,
        "decision_prediction": {"direction": predicted, "magnitude_bp": 0},
        "decision_direction_correct": target == predicted,
        "decision_exact": exact,
        "decision_dense_v3_reward": 0.0,
        "delivery_valid": True,
        "strict_json": True,
        "hit_eos": True,
        "cap_reached": False,
        "strict_periodic_tail": False,
        "response_format": "strict_json",
        "decision_rejection_reason": None,
        "completion_token_count": 10,
    }


def test_population_has_237_unique_meetings_and_expected_splits() -> None:
    tokenizer = comparison._load_tokenizer(comparison.MODELS[-1])
    samples, prompts = comparison.load_population(tokenizer)
    assert len(samples) == len(prompts) == 237
    assert len({sample["sample_id"] for sample in samples}) == 237
    assert {
        split: sum(sample["split"] == split for sample in samples)
        for split in comparison.SPLITS
    } == comparison.EXPECTED_UNIQUE_COUNTS


def test_batch_contract_has_31_batches_per_model_and_93_total() -> None:
    tokenizer = comparison._load_tokenizer(comparison.MODELS[-1])
    samples, _ = comparison.load_population(tokenizer)
    contracts = comparison.batch_contracts(samples)
    assert len(contracts) == 93
    assert all(
        sum(contract["model_label"] == model["label"] for contract in contracts) == 31
        for model in comparison.MODELS
    )
    assert sum(len(contract["sample_ids"]) for contract in contracts) == 711


def test_extended_metrics_distinguishes_fixed_and_supported_balanced_accuracy() -> None:
    rows = [
        _row("a", "hold", "hold", True),
        _row("b", "hold", "hold", True),
        _row("c", "hike", "hold", False),
    ]
    metrics = comparison._extended_metrics(rows)
    assert metrics["balanced_accuracy"] == 1 / 3
    assert metrics["balanced_accuracy_supported_classes"] == 0.5
    assert metrics["supported_directions"] == ["hold", "hike"]


def test_paired_comparison_uses_model_b_minus_model_a() -> None:
    rows_a = [
        _row("cut", "cut", "hold"),
        _row("hold", "hold", "hold", True),
        _row("hike", "hike", "hold"),
    ]
    rows_b = [
        _row("cut", "cut", "cut", True),
        _row("hold", "hold", "hold", True),
        _row("hike", "hike", "hike", True),
    ]
    result = comparison.paired_comparison(rows_a, rows_b, draws=50, seed=7)
    assert result["delta_definition"] == "model_b_minus_model_a"
    assert result["deltas"]["direction_accuracy"] == pytest.approx(2 / 3)
    assert result["paired_direction_correctness"] == {
        "a_only_direction_correct": 0,
        "b_only_direction_correct": 2,
        "both_direction_correct": 1,
        "neither_direction_correct": 0,
    }
