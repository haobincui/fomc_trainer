from __future__ import annotations

from pathlib import Path

from jobs.retrain_v2 import audit_chk4_pre2009_correction_cp6_final as final_audit


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_real_cp6_artifacts_replay_to_terminal_selection_failure() -> None:
    result = final_audit.audit(REPO_ROOT)
    assert result["status"] == "selection_failed"
    assert result["outcome"] == "no_preregistered_checkpoint_passed_full_gate"
    assert result["selected_checkpoint"] is None
    assert result["candidate_order_completed"] == [2, 4, 6]
    assert result["advance_allowed"] is False
    assert result["terminal"] is True
    assert result["quality_gate"]["reasons"] == [
        "step1:hike_correct_nonzero_lt_1",
        "step1:hike_reward_zero_std",
    ]
    assert "adapter_merge" in result["prohibited"]
    assert "grpo_smoke" in result["prohibited"]


def test_cp6_final_evidence_retains_delivery_and_other_direction_passes() -> None:
    result = final_audit.audit(REPO_ROOT)
    step1 = result["panels"]["step1"]
    step2 = result["panels"]["static_step2_prior"]
    assert step1["directions"]["hold"]["correct_nonzero_count"] == 3
    assert step1["directions"]["hold"]["reward_std"] > 0
    assert step1["directions"]["hike"]["correct_nonzero_count"] == 0
    assert step1["delivery"]["cap_count"] <= 2
    assert step1["delivery"]["boundary_count"] >= 6
    assert step2["directions"]["hold"]["correct_nonzero_count"] == 1
    assert step2["directions"]["cut"]["correct_nonzero_count"] == 1
    assert step2["directions"]["hold"]["reward_std"] > 0
    assert step2["directions"]["cut"]["reward_std"] > 0


def test_cp6_final_binds_all_three_decision_stages() -> None:
    result = final_audit.audit(REPO_ROOT)
    assert result["checkpoint_2_decision"]["sha256"]
    assert result["checkpoint_4_decision"]["sha256"]
    assert result["checkpoint_6_authorization"]["sha256"] == (
        final_audit.AUTHORIZATION_SHA256
    )
    assert result["checkpoint_6_screen_artifacts"]["results"]["rows"] == 16
