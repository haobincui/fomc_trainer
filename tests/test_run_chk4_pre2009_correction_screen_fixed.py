from __future__ import annotations

import json
from pathlib import Path

from jobs.retrain_v2 import run_chk4_pre2009_correction_screen_fixed as fixed


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_fixed_runner_normalizes_at_first_eos_without_changing_sampling():
    trimmed, evidence = fixed.normalize_generated_ids(
        [10, 11, 128001, 128001, 128001], {128001}
    )

    assert trimmed == [10, 11, 128001]
    assert evidence["method"] == "truncate_at_first_eos_inclusive"
    assert evidence["generation_not_changed"] is True
    assert evidence["discarded_after_first_eos_count"] == 2


def test_checkpoint_four_authorization_binds_both_sft_receipts_and_cp2_decision():
    # Before launch, replay the proposed authorization.  After the create-only
    # screen has landed, inspect the immutable authorization instead: the
    # preflight correctly rejects recomputation because fresh_output is false.
    if fixed.output_path(REPO_ROOT, 4).exists():
        authorization = json.loads(
            fixed.authorization_path(REPO_ROOT, 4).read_text(encoding="utf-8")
        )
    else:
        authorization = fixed.authorization_payload(REPO_ROOT, 4)

    assert authorization["status"] == "authorized"
    assert authorization["scope"]["gpu"] == 1
    assert authorization["scope"]["fresh_output"] is True
    assert authorization["lineage"]["original_sft_authorization"]["sha256"] == (
        fixed.ORIGINAL_AUTH_SHA256
    )
    assert authorization["lineage"]["supplemental_post_run_attestation"][
        "sha256"
    ] == fixed.SUPPLEMENTAL_SHA256
    assert authorization["lineage"]["checkpoint_2_advance_decision"][
        "sha256"
    ] == fixed.CP2_DECISION_SHA256
    assert authorization["generation"]["seed"] == 31416
    assert authorization["generation"]["max_new_tokens"] == 1536
    assert authorization["generation"]["padding_postprocessor"] == (
        "truncate_at_first_bound_eos_inclusive_before_decode"
    )
    assert authorization["methodology"]["full_original_gate_required_for_selection"]
