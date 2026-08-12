from __future__ import annotations

from pathlib import Path

from jobs.retrain_v2 import audit_chk4_pre2009_correction_selection as audit


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_two_offline_replay_is_auditable_but_not_selected():
    result = audit.audit_step(REPO_ROOT, 2)

    assert result["quality_status"] == "failed"
    assert result["outcome"] == "advance_allowed_action_direction_insufficiency"
    assert result["advance_allowed"] is True
    assert result["gate_reasons"] == [
        "step1:hike_correct_nonzero_lt_1",
        "step1:hike_reward_zero_std",
        "step2:cut_correct_nonzero_lt_1",
    ]
    assert result["instrumentation_replay"]["generation_not_rerun"] is True
    assert result["instrumentation_replay"]["invalid_v2_diagnosis"]["status"] == (
        "invalid_instrumentation_repeated_eos_padding"
    )
    assert result["summary"]["panels"]["step1"]["directions"]["hold"][
        "correct_nonzero_count"
    ] == 3
    assert result["summary"]["panels"]["step1"]["directions"]["hike"][
        "correct_nonzero_count"
    ] == 0
    assert result["summary"]["panels"]["static_step2_prior"]["directions"][
        "cut"
    ]["correct_nonzero_count"] == 0


def test_advance_rule_never_treats_hold_or_delivery_failure_as_action_only():
    assert "step1:hold_correct_nonzero_lt_2" not in audit.ACTION_ONLY_REASONS
    assert "step1:cap_count_gt_2" not in audit.ACTION_ONLY_REASONS
    assert "step1:boundary_count_lt_6" not in audit.ACTION_ONLY_REASONS
    assert "step1:periodic_tail_present" not in audit.ACTION_ONLY_REASONS
    assert "step1:hike_correct_nonzero_lt_1" in audit.ACTION_ONLY_REASONS
    assert "step2:cut_correct_nonzero_lt_1" in audit.ACTION_ONLY_REASONS
