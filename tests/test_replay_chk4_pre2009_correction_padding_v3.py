from __future__ import annotations

from pathlib import Path

from jobs.retrain_v2 import replay_chk4_pre2009_correction_padding_v3 as replay


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_truncate_early_stop_with_repeated_eos_padding():
    trimmed, evidence = replay.truncate_at_first_eos(
        [11, 12, replay.EOS_TOKEN_ID, replay.EOS_TOKEN_ID, replay.EOS_TOKEN_ID],
        {replay.EOS_TOKEN_ID},
    )

    assert trimmed == [11, 12, replay.EOS_TOKEN_ID]
    assert evidence["first_eos_index"] == 2
    assert evidence["discarded_after_first_eos_count"] == 2
    assert evidence["all_discarded_tokens_are_bound_eos"] is True


def test_no_eos_cap_sequence_is_not_trimmed():
    trimmed, evidence = replay.truncate_at_first_eos(
        [11, 12, 13, 14], {replay.EOS_TOKEN_ID}
    )

    assert trimmed == [11, 12, 13, 14]
    assert evidence["first_eos_index"] is None
    assert evidence["discarded_after_first_eos_count"] == 0
    assert evidence["all_discarded_tokens_are_bound_eos"] is False


def test_first_eos_is_authoritative_even_if_followed_by_non_eos():
    trimmed, evidence = replay.truncate_at_first_eos(
        [11, replay.EOS_TOKEN_ID, 99, 100], {replay.EOS_TOKEN_ID}
    )

    assert trimmed == [11, replay.EOS_TOKEN_ID]
    assert evidence["discarded_unique_token_ids"] == [99, 100]
    assert evidence["all_discarded_tokens_are_bound_eos"] is False


def test_real_v2_diagnosis_finds_thirteen_batched_padding_rows():
    diagnosis = replay.build_diagnosis(REPO_ROOT)

    assert diagnosis["status"] == "invalid_instrumentation_repeated_eos_padding"
    assert diagnosis["scope"]["invalid_for_checkpoint_selection"] is True
    assert diagnosis["scope"]["model_failure_claim_prohibited"] is True
    assert diagnosis["diagnosis"]["affected_rows"] == 13
    assert diagnosis["diagnosis"]["unaffected_rows"] == 3
    assert diagnosis["diagnosis"]["all_discarded_tokens_are_bound_eos"] is True
