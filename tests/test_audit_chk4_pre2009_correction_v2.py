from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobs.retrain_v2 import audit_chk4_pre2009_correction_v2 as audit
from jobs.retrain_v2 import probe_chk4_pre2009_correction_v2 as probe
from open_r1.validator.loo_generation_spec import seal_manifest


def _block(
    direction: str,
    correct: int,
    *,
    strict: int = 1,
    reward_std: float = 0.1,
    boundary: int = 4,
    cap: int = 0,
    periodic: int = 0,
) -> dict[str, object]:
    return {
        "target_direction": direction,
        "target_magnitude_bp": 0 if direction == "hold" else 25,
        "cases": 4,
        "correct_nonzero_count": correct,
        "strict_json_exact_count": strict,
        "reward_mean": 0.2,
        "reward_std": reward_std,
        "boundary_count": boundary,
        "cap_count": cap,
        "periodic_tail_count": periodic,
        "rewards_finite": True,
    }


def _metrics(
    *,
    hold: tuple[int, int],
    hike: tuple[int, int],
    cut: tuple[int, int],
) -> dict[str, object]:
    prompts = {
        "h0": _block("hold", hold[0]),
        "h1": _block("hold", hold[1]),
        "u0": _block("hike", hike[0]),
        "u1": _block("hike", hike[1]),
        "d0": _block("cut", cut[0]),
        "d1": _block("cut", cut[1]),
    }
    return {
        "prompts": prompts,
        "aggregate_correct_nonzero": {
            "hold": sum(hold),
            "hike": sum(hike),
            "cut": sum(cut),
        },
    }


def test_selection_gate_combines_absolute_and_baseline_relative_floors() -> None:
    baseline = _metrics(hold=(3, 3), hike=(1, 0), cut=(2, 1))
    candidate = _metrics(hold=(3, 2), hike=(1, 1), cut=(2, 1))
    result = audit.evaluate_selection_or_blind(
        stage=probe.SELECTION_STAGE,
        baseline_metrics=baseline,
        candidate_metrics=candidate,
    )
    assert result["quality_status"] == "passed"
    assert result["reasons"] == []
    assert result["aggregate_thresholds"] == {"hold": 5, "hike": 2, "cut": 3}

    candidate["prompts"]["u1"]["correct_nonzero_count"] = 0
    candidate["aggregate_correct_nonzero"]["hike"] = 1
    failed = audit.evaluate_selection_or_blind(
        stage=probe.SELECTION_STAGE,
        baseline_metrics=baseline,
        candidate_metrics=candidate,
    )
    assert failed["quality_status"] == "failed"
    assert "selection:u1:correct_nonzero_lt_1" in failed["reasons"]
    assert "selection:aggregate_hike_correct_nonzero_lt_2" in failed["reasons"]


def test_zero_std_is_allowed_only_for_four_strict_exact() -> None:
    baseline = _metrics(hold=(2, 2), hike=(0, 1), cut=(1, 1))
    candidate = _metrics(hold=(2, 2), hike=(1, 1), cut=(1, 1))
    candidate["prompts"]["u0"].update({"reward_std": 0.0, "strict_json_exact_count": 4})
    passed = audit.evaluate_selection_or_blind(
        stage=probe.BLIND_STAGE,
        baseline_metrics=baseline,
        candidate_metrics=candidate,
    )
    assert passed["quality_status"] == "passed"

    candidate["prompts"]["u0"]["strict_json_exact_count"] = 3
    failed = audit.evaluate_selection_or_blind(
        stage=probe.BLIND_STAGE,
        baseline_metrics=baseline,
        candidate_metrics=candidate,
    )
    assert "blind:u0:reward_zero_std_without_4_strict_exact" in failed["reasons"]


def test_retention_gate_never_regresses_per_prompt_baseline() -> None:
    baseline = _metrics(hold=(0, 0), hike=(2, 3), cut=(2, 1))
    candidate = _metrics(hold=(0, 0), hike=(2, 3), cut=(2, 1))
    # Retention contains action prompts only in production; discard synthetic holds.
    for metrics in (baseline, candidate):
        del metrics["prompts"]["h0"]
        del metrics["prompts"]["h1"]
    candidate["prompts"]["u0"]["strict_json_exact_count"] = 2
    baseline["prompts"]["u0"]["strict_json_exact_count"] = 2
    passed = audit.evaluate_retention(
        baseline_metrics=baseline, candidate_metrics=candidate
    )
    assert passed["quality_status"] == "passed"

    candidate["prompts"]["u0"]["correct_nonzero_count"] = 1
    candidate["prompts"]["u0"]["strict_json_exact_count"] = 1
    failed = audit.evaluate_retention(
        baseline_metrics=baseline, candidate_metrics=candidate
    )
    assert failed["quality_status"] == "failed"
    assert any(
        "correct_nonzero_lt_baseline_floor_2" in item for item in failed["reasons"]
    )
    assert any("strict_json_exact_lt_baseline_2" in item for item in failed["reasons"])


def test_selection_locks_earliest_pass_and_rejects_later_screen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline_metrics = _metrics(hold=(3, 3), hike=(0, 0), cut=(1, 1))
    failed_metrics = _metrics(hold=(2, 2), hike=(0, 0), cut=(1, 1))
    passed_metrics = _metrics(hold=(3, 2), hike=(1, 1), cut=(1, 1))
    manifest = {"methodology": {"checkpoint_selection": "earliest_all_pass"}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(probe, "validate_manifest", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(
        audit,
        "_descriptor",
        lambda path: {"path": str(path), "sha256": "a" * 64, "bytes": 3},
    )
    monkeypatch.setattr(audit, "_implementation", lambda _root: {})

    def fake_replay(**kwargs: object) -> dict[str, object]:
        role = kwargs["role"]
        output = Path(str(kwargs["output_dir"])).name
        if role == "baseline":
            metrics = baseline_metrics
        elif output == "cp2":
            metrics = failed_metrics
        else:
            metrics = passed_metrics
        return {"metrics": metrics, "output": output}

    monkeypatch.setattr(audit, "replay_output", fake_replay)
    decision = audit.selection_decision(
        repo_root=tmp_path,
        manifest_path=manifest_path,
        manifest_sha256="a" * 64,
        baseline_output=tmp_path / "baseline",
        candidates=((2, tmp_path / "cp2"), (4, tmp_path / "cp4")),
    )
    assert decision["status"] == "selection_passed"
    assert decision["selected_checkpoint"] == 4
    assert decision["evaluated_candidate_prefix"] == [2, 4]

    with pytest.raises(audit.CorrectionV2AuditError, match="later candidate"):
        audit.selection_decision(
            repo_root=tmp_path,
            manifest_path=manifest_path,
            manifest_sha256="a" * 64,
            baseline_output=tmp_path / "baseline",
            candidates=(
                (2, tmp_path / "cp2"),
                (4, tmp_path / "cp4"),
                (6, tmp_path / "cp6"),
            ),
        )


def test_retention_auditor_requires_blind_exact_selection_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_sha = "a" * 64
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    selection = seal_manifest(
        {
            "schema_version": audit.SELECTION_RECEIPT_SCHEMA,
            "status": "selection_passed",
            "selected_checkpoint": 4,
            "manifest": {"sha256": manifest_sha},
            "implementation": {},
        }
    )
    original_selection = tmp_path / "selection-original.json"
    other_selection = tmp_path / "selection-other.json"
    for path in (original_selection, other_selection):
        path.write_text(json.dumps(selection, sort_keys=True) + "\n", encoding="utf-8")
    blind = seal_manifest(
        {
            "schema_version": audit.CONFIRMATION_RECEIPT_SCHEMA,
            "status": "confirmation_passed",
            "stage": probe.BLIND_STAGE,
            "locked_checkpoint": 4,
            "manifest": {"sha256": manifest_sha},
            "selection_receipt": {
                "path": str(original_selection.resolve()),
                "sha256": audit.sha256_file(original_selection),
            },
            "implementation": {},
        }
    )
    blind_path = tmp_path / "blind.json"
    blind_path.write_text(json.dumps(blind, sort_keys=True) + "\n", encoding="utf-8")
    metrics = _metrics(hold=(0, 0), hike=(1, 1), cut=(1, 1))
    del metrics["prompts"]["h0"]
    del metrics["prompts"]["h1"]
    monkeypatch.setattr(probe, "validate_manifest", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(audit, "verify_selection_receipt", lambda **_kwargs: selection)
    monkeypatch.setattr(audit, "verify_confirmation_receipt", lambda **_kwargs: blind)
    monkeypatch.setattr(
        audit,
        "replay_output",
        lambda **_kwargs: {"metrics": metrics},
    )
    monkeypatch.setattr(audit, "_implementation", lambda _root: {})

    with pytest.raises(
        audit.CorrectionV2AuditError,
        match="exact selection receipt",
    ):
        audit.confirmation_decision(
            repo_root=tmp_path,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha,
            stage=probe.RETENTION_STAGE,
            selection_receipt_path=other_selection,
            predecessor_receipt_path=blind_path,
            baseline_output=tmp_path / "baseline",
            candidate_output=tmp_path / "candidate",
        )
