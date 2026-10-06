from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobs.retrain_v2 import chk4_cp450_final_evaluation as subject


def _row(
    target: str,
    prediction: str | None,
    *,
    target_bp: int = 0,
    prediction_bp: int = 0,
    delivery: bool = True,
    exact: bool | None = None,
) -> dict:
    if exact is None:
        exact = prediction == target and prediction_bp == target_bp
    return {
        "target_direction": target,
        "target_magnitude_bp": target_bp,
        "decision_prediction": (
            None
            if prediction is None
            else {"direction": prediction, "magnitude_bp": prediction_bp}
        ),
        "decision_exact": exact,
        "decision_dense_v3_reward": 1.0 if exact else 0.0,
        "delivery_valid": delivery,
        "strict_json": delivery,
        "hit_eos": delivery,
        "cap_reached": not delivery,
        "strict_periodic_tail": False,
    }


def test_metric_block_keeps_invalid_predictions_in_denominator() -> None:
    rows = [
        _row("cut", "cut", target_bp=25, prediction_bp=25),
        _row("hold", "hold"),
        _row("hike", None, target_bp=25, delivery=False),
    ]

    metrics = subject.metric_block(rows)

    assert metrics["direction_accuracy"] == pytest.approx(2 / 3)
    assert metrics["balanced_accuracy"] == pytest.approx(2 / 3)
    assert metrics["macro_f1"] == pytest.approx(2 / 3)
    assert metrics["exact_direction_magnitude_accuracy"] == pytest.approx(2 / 3)
    assert metrics["parseable_count"] == 2
    assert metrics["delivery_valid_rate"] == pytest.approx(2 / 3)
    assert metrics["confusion_matrix"]["hike"]["invalid"] == 1


def test_metric_block_uses_signed_basis_points() -> None:
    metrics = subject.metric_block(
        [_row("cut", "hike", target_bp=25, prediction_bp=25, exact=False)]
    )
    assert metrics["signed_bp_mae_parseable"] == 50


def test_batch_contract_is_fixed_and_within_sequence_limit() -> None:
    samples = [{"sample_id": f"sample-{index}"} for index in range(13)]
    batches = subject._batch_contract(samples)

    assert [(row["mode"], len(row["sample_ids"])) for row in batches] == [
        ("greedy", 8),
        ("greedy", 5),
        ("sampled", 2),
        ("sampled", 2),
        ("sampled", 2),
        ("sampled", 2),
        ("sampled", 2),
        ("sampled", 2),
        ("sampled", 1),
    ]
    assert all(
        len(row["sample_ids"]) * row["num_return_sequences"]
        <= subject.MAX_SEQUENCES_PER_BATCH
        for row in batches
    )
    assert len({row["seed"] for row in batches}) == len(batches)


def test_authorization_integrity_binds_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "evaluation_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")

    value = subject._authorization(manifest, "a" * 64)
    unsigned = dict(value)
    integrity = unsigned.pop("integrity")

    assert value["scope"]["allowed"] == [
        "cpu_non_destructive_cp450_merge",
        "gpu1_one_shot_sealed_test",
        "exact_manifest_resume",
    ]
    assert "checkpoint_reselection" in value["scope"]["forbidden"]
    assert integrity["payload_sha256"] == subject._canonical_payload_sha(unsigned)


def test_jsonl_record_includes_release_row_count(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a":1}\n\n{"a":2}\n', encoding="utf-8")

    assert subject._jsonl_record(path, relative_to=tmp_path) == {
        "path": "rows.jsonl",
        "bytes": path.stat().st_size,
        "rows": 2,
        "sha256": subject.sha256_file(path),
    }


def test_open_receipt_is_create_only_replayable(tmp_path: Path) -> None:
    samples = [
        {
            "sample_id": f"sample-{index}",
            "meeting_date": f"2020-01-{index + 1:02d}",
            "direction": "hold",
            "magnitude_bp": 0,
            "prompt_sha256": f"{index:064x}",
            "prompt_token_count": 100 + index,
        }
        for index in range(13)
    ]
    path = tmp_path / "test_open_receipt.json"
    value = subject._open_receipt("b" * 64, samples)
    path.write_text(json.dumps(value), encoding="utf-8")

    observed = subject._validate_open_receipt(path, "b" * 64, samples)

    assert observed["sample_count"] == 13
    assert len(observed["batches"]) == 9


def test_sample_frequency_brier_penalizes_invalid_mass() -> None:
    sample = {
        "sample_id": "one",
        "direction": "cut",
        "magnitude_bp": 25,
    }
    rows = []
    for index, prediction in enumerate(("cut", "cut", None, None), 1):
        rows.append(
            {
                **_row(
                    "cut",
                    prediction,
                    target_bp=25,
                    prediction_bp=25,
                    delivery=prediction is not None,
                ),
                "sample_id": "one",
                "generation_mode": "sampled",
                "generation_index": index,
            }
        )

    result = subject._sample_frequency_metrics(rows, [sample])

    # p(cut)=0.5 and p(invalid)=0.5 against a one-hot cut target.
    assert result["brier_four_class"] == pytest.approx(0.5)
    assert result["unanimous_prediction_rate"] == 0


def test_validate_batch_rows_rejects_manifest_drift() -> None:
    contract = {
        "mode": "greedy",
        "batch_index": 0,
        "sample_ids": ["one"],
        "num_return_sequences": 1,
        "seed": 7,
    }
    row = {
        "schema_version": subject.RESULT_SCHEMA,
        "sample_id": "one",
        "generation_index": 0,
        "generation_mode": "greedy",
        "batch_index": 0,
        "batch_seed": 7,
        "evaluation_manifest_sha256": "wrong",
        "merged_model_sha256": "m" * 64,
    }

    with pytest.raises(subject.FinalEvaluationError, match="batch manifest drift"):
        subject._validate_batch_rows([row], contract, "a" * 64, "m" * 64)
