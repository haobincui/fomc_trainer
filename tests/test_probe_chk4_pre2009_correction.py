from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobs.retrain_v2 import probe_chk4_pre2009_correction as probe


def _row(
    *,
    panel: str,
    direction: str,
    reward: float,
    correct: bool,
    batch_index: int | None = None,
    generation_mode: str = "sampled",
) -> dict[str, object]:
    return {
        "panel": panel,
        "batch_index": batch_index,
        "target_direction": direction,
        "generation_mode": generation_mode,
        "decision_dense_v3_reward": reward,
        "decision_direction_correct": correct,
        "cap_reached": False,
        "think_boundary_count": 1,
        "strict_periodic_tail": False,
    }


def _direction_rows(
    *, panel: str, direction: str, rewards: list[float], batch_index: int
) -> list[dict[str, object]]:
    return [
        _row(
            panel=panel,
            direction=direction,
            reward=reward,
            correct=reward > 0,
            batch_index=batch_index,
        )
        for reward in rewards
    ]


def _selection_rows() -> list[dict[str, object]]:
    return [
        *_direction_rows(
            panel="selection",
            direction="hold",
            rewards=[0.0, 0.25, 1.0, 0.0],
            batch_index=0,
        ),
        *_direction_rows(
            panel="selection",
            direction="hike",
            rewards=[0.0, 0.25, 0.0, 0.0],
            batch_index=0,
        ),
        *_direction_rows(
            panel="selection",
            direction="hold",
            rewards=[0.0, 0.25, 0.0, 0.0],
            batch_index=1,
        ),
        *_direction_rows(
            panel="selection",
            direction="cut",
            rewards=[0.0, 0.25, 0.0, 0.0],
            batch_index=1,
        ),
    ]


def test_selection_contract_is_original_smoke_layout_only():
    assert probe.SELECTION_BATCH_IDS == (
        (
            "dec-bb7d9c61358a339a9a1f4aa5",
            "dec-4dfab939b0a910c949931475",
        ),
        (
            "dec-29de02cb43945c20f838fcf7",
            "dec-8b8d55ea065b662a19cefe88",
        ),
    )
    assert probe.SELECTION_SEED == 31416
    assert probe.MAX_NEW_TOKENS == 1536
    assert probe.NUM_RETURN_SEQUENCES == 4
    assert probe.PRE2009_HOLD_ID not in sum(probe.SELECTION_BATCH_IDS, ())
    assert probe.PRE2009_HIKE_ID not in sum(probe.SELECTION_BATCH_IDS, ())


def test_selection_summary_passes_only_with_all_direction_and_delivery_gates():
    manifest = {
        "purpose": probe.SELECTION_PURPOSE,
        "quality_gate": {"sealed": True},
    }
    summary = probe.summarize_results(
        _selection_rows(), manifest=manifest, provenance={"test": True}
    )

    assert summary["quality_status"] == "passed"
    assert summary["quality_gate"]["reasons"] == []
    assert summary["panels"]["step1"]["directions"]["hold"][
        "correct_nonzero_count"
    ] == 2

    hike_zero = _selection_rows()
    for row in hike_zero:
        if row["batch_index"] == 0 and row["target_direction"] == "hike":
            row["decision_dense_v3_reward"] = 0.0
            row["decision_direction_correct"] = False
    failed = probe.summarize_results(
        hike_zero, manifest=manifest, provenance={"test": True}
    )
    assert failed["quality_status"] == "failed"
    assert failed["quality_gate"]["reasons"] == [
        "step1:hike_correct_nonzero_lt_1",
        "step1:hike_reward_zero_std",
    ]


def test_selection_safety_regression_is_visible_separately():
    rows = _selection_rows()
    for row in rows[:3]:
        row["cap_reached"] = True
    summary = probe.summarize_results(
        rows,
        manifest={
            "purpose": probe.SELECTION_PURPOSE,
            "quality_gate": {"sealed": True},
        },
        provenance={},
    )
    assert "step1:cap_count_gt_2" in summary["quality_gate"]["reasons"]


def _confirmation_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    correct_counts = {"hold": 3, "hike": 1, "cut": 1}
    for direction in ("hold", "hike", "cut"):
        rows.append(
            _row(
                panel="legacy_stratified",
                direction=direction,
                reward=0.0,
                correct=False,
                generation_mode="greedy",
            )
        )
        for index in range(4):
            reward = 0.25 if index < correct_counts[direction] else 0.0
            rows.append(
                _row(
                    panel="legacy_stratified",
                    direction=direction,
                    reward=reward,
                    correct=reward > 0,
                )
            )
    for direction, correct_count in (("hold", 2), ("hike", 1)):
        for index in range(4):
            reward = 0.25 if index < correct_count else 0.0
            rows.append(
                _row(
                    panel="pre2009_blind",
                    direction=direction,
                    reward=reward,
                    correct=reward > 0,
                )
            )
    return rows


def test_blind_confirmation_contract_passes_without_checkpoint_fallback():
    summary = probe.summarize_results(
        _confirmation_rows(),
        manifest={
            "purpose": probe.CONFIRMATION_PURPOSE,
            "quality_gate": {"sealed": True},
        },
        provenance={},
    )
    assert summary["cases"] == 23
    assert summary["quality_status"] == "passed"
    assert summary["panels"]["legacy_stratified"]["directions"]["hold"][
        "correct_nonzero_count"
    ] == 3
    assert summary["panels"]["pre2009_blind"]["directions"]["hold"][
        "correct_nonzero_count"
    ] == 2


def test_probe_artifact_writer_is_create_only_and_seals_payload(tmp_path: Path):
    path = tmp_path / "manifest.json"
    probe._write_exclusive_json(path, {"fixed": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"fixed": True}
    assert path.stat().st_mode & 0o777 == 0o444
    with pytest.raises(probe.CorrectionProbeError, match="overwrite"):
        probe._write_exclusive_json(path, {"fixed": False})
