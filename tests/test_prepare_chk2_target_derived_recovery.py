from __future__ import annotations

from jobs.generation.prepare_chk2_target_derived_recovery import (
    classify_recovery_action,
    generation_reason_families,
)


def _prepared(words: int = 100) -> dict:
    return {"official_minutes_paragraph": "word " * words}


def _verification(*, semantic_good: bool) -> dict:
    return {
        "parsed_verification": {
            "target_claims": [{"verdict": "entailed"}],
            "analysis_claims": [{"verdict": "supported"}],
            "reasoning_compatible": semantic_good,
            "bidirectional_entailment": semantic_good,
            "issues": [] if semantic_good else ["issue"],
            "overall_pass": semantic_good,
        }
    }


def test_generation_family_mapping_covers_structural_prefixes() -> None:
    families = generation_reason_families(
        [
            "analysis_insufficiently_destylized",
            "analysis_numeric_multiset_mismatch",
            "atomic_claim_4:target_span_must_be_nonempty_text",
        ]
    )
    assert families == ("destylization", "numeric", "structure_schema")


def test_generation_retry_policy_uses_breadth_and_length() -> None:
    one_family = {
        "rejection_stage": "generation_gate",
        "rejection_reasons": ["reasoning_attribution_set_mismatch"],
    }
    two_families = {
        "rejection_stage": "generation_gate",
        "rejection_reasons": [
            "reasoning_attribution_set_mismatch",
            "reasoning_numeric_multiset_mismatch",
        ],
    }
    three_families = {
        "rejection_stage": "generation_gate",
        "rejection_reasons": [
            "reasoning_attribution_set_mismatch",
            "reasoning_numeric_multiset_mismatch",
            "analysis_insufficiently_destylized",
        ],
    }

    assert classify_recovery_action(
        rejection=one_family, prepared=_prepared(250), verification=None
    )[0] == "regenerate_and_reverify"
    assert classify_recovery_action(
        rejection=two_families, prepared=_prepared(200), verification=None
    )[0] == "regenerate_and_reverify"
    assert classify_recovery_action(
        rejection=two_families, prepared=_prepared(201), verification=None
    )[0] == "defer_targeted_repair"
    assert classify_recovery_action(
        rejection=three_families, prepared=_prepared(80), verification=None
    )[0] == "defer_targeted_repair"


def test_verification_retry_policy_separates_output_and_semantics() -> None:
    rejection = {
        "rejection_stage": "verification_gate",
        "rejection_reasons": ["verifier_overall_fail"],
    }
    assert classify_recovery_action(
        rejection=rejection,
        prepared=_prepared(),
        verification=_verification(semantic_good=True),
    )[0] == "reverify_only"
    assert classify_recovery_action(
        rejection=rejection,
        prepared=_prepared(),
        verification=_verification(semantic_good=False),
    )[0] == "regenerate_and_reverify"
    assert classify_recovery_action(
        rejection=rejection, prepared=_prepared(), verification=None
    )[0] == "reverify_only"
