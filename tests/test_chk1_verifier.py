from __future__ import annotations

import pytest

from jobs.retrain_v2.chk1.contracts import render_generator_payload
from jobs.retrain_v2.chk1.verifier import (
    critic_accepts,
    ngram_overlap,
    select_candidate,
    validate_fact_card,
    validate_critic,
    verify_candidate,
)


SOURCE_HASH = "a" * 64


def _fact_card() -> dict:
    return {
        "sample_id": "2024-01-31::Unemployment-Rate",
        "meeting_date": "2024-01-31",
        "atomic_topic": "Unemployment Rate",
        "cutoff_ts": "2024-01-30T23:59:59Z",
        "evidence": [
            {
                "evidence_id": "E1",
                "value": "4.0",
                "units": "Percent",
                "observation_date": "2023-12-01",
                "release_ts": "2024-01-05T13:30:00Z",
                "availability_upper_bound_ts": None,
                "availability_basis": "actual_release_ts",
                "source_sha256": SOURCE_HASH,
            },
            {
                "evidence_id": "E2",
                "value": "3.7",
                "units": "Percent",
                "observation_date": "2023-11-01",
                "release_ts": "2023-12-08T13:30:00Z",
                "availability_upper_bound_ts": None,
                "availability_basis": "actual_release_ts",
                "source_sha256": SOURCE_HASH,
            },
        ],
    }


def _critic(style: int = 4) -> dict:
    return {
        "grounded": True,
        "unsupported_claims": [],
        "style_score": style,
        "reasoning_consistency": True,
    }


def test_grounded_candidate_passes_and_builds_deepseek_boundary() -> None:
    result = verify_candidate(
        {
            "reasoning": "The latest reading was 4.0 percent, compared with 3.7 percent previously.",
            "final_analysis": "The measure increased over the two available observations.",
            "evidence_ids": ["E1", "E2"],
        },
        fact_card=_fact_card(),
    )

    assert result.passed is True
    assert result.candidate is not None
    assert result.candidate.response.count("</think>") == 1
    assert result.details["unsupported_numeric_claims"] == []


def test_unsupported_number_post_cutoff_and_causal_claim_fail_closed() -> None:
    card = _fact_card()
    card["evidence"][0]["release_ts"] = "2024-02-02T00:00:00Z"
    result = verify_candidate(
        {
            "reasoning": "The rate reached 9.9 percent because of a recession.",
            "final_analysis": "The measure increased.",
            "evidence_ids": ["E1"],
        },
        fact_card=card,
    )

    assert result.passed is False
    assert "fact_card_post_cutoff_availability" in result.error_codes
    assert "candidate_unsupported_numeric_claim" in result.error_codes
    assert "candidate_external_policy_or_event" in result.error_codes
    assert "candidate_unlicensed_causal_claim" in result.error_codes


def test_sealed_alfred_upper_bound_and_string_formula_match_source_schema() -> None:
    card = _fact_card()
    card["evidence"].append(
        {
            "evidence_id": "E3",
            "value": "0.3",
            "units": "Percentage points",
            "observation_date": "2023-12-01",
            "formula": "E1 - E2",
            "operand_evidence_ids": ["E1", "E2"],
            "release_ts": None,
            "availability_upper_bound_ts": "2024-01-30T23:59:59Z",
            "availability_basis": "sealed_alfred_d1_upper_bound",
            "source_sha256": SOURCE_HASH,
        }
    )

    assert validate_fact_card(card) == []
    result = verify_candidate(
        {
            "reasoning": "The deterministic change was 0.3 percentage points.",
            "final_analysis": "The available observations indicate an increase.",
            "evidence_ids": ["E3"],
        },
        fact_card=card,
    )
    assert result.passed is True
    assert result.details["unsupported_numeric_claims"] == []


def test_actual_release_may_retain_a_valid_auxiliary_alfred_upper_bound() -> None:
    card = _fact_card()
    card["evidence"][0]["availability_upper_bound_ts"] = (
        "2024-01-30T23:59:59Z"
    )

    assert validate_fact_card(card) == []


@pytest.mark.parametrize(
    ("updates", "expected_error"),
    [
        (
            {"availability_upper_bound_ts": None},
            "fact_card_missing_release_or_availability",
        ),
        (
            {"release_ts": "2024-01-05T13:30:00Z"},
            "fact_card_inconsistent_availability",
        ),
        (
            {"availability_upper_bound_ts": "2024-02-02T00:00:00Z"},
            "fact_card_post_cutoff_availability",
        ),
        (
            {"availability_basis": "unverified_snapshot"},
            "fact_card_invalid_availability_basis",
        ),
    ],
)
def test_sealed_availability_proof_fails_closed_on_invalid_variants(
    updates: dict[str, object], expected_error: str
) -> None:
    card = _fact_card()
    evidence = card["evidence"][0]
    evidence.update(
        {
            "release_ts": None,
            "availability_upper_bound_ts": "2024-01-30T23:59:59Z",
            "availability_basis": "sealed_alfred_d1_upper_bound",
        }
    )
    evidence.update(updates)

    assert expected_error in validate_fact_card(card)


def test_same_sample_minutes_are_used_only_for_eight_token_rejection() -> None:
    copied = "Conditions in labor markets remained tight but showed signs of easing."
    result = verify_candidate(
        {
            "reasoning": "Available evidence was reviewed without adding external context.",
            "final_analysis": copied,
            "evidence_ids": ["E1"],
        },
        fact_card=_fact_card(),
        same_sample_minutes=f"Staff noted that {copied} Other material followed.",
    )

    assert len(ngram_overlap(copied, copied, n=8)) >= 1
    assert "candidate_minutes_8token_overlap" in result.error_codes


def test_prompt_renderer_rejects_decision_or_legacy_fields() -> None:
    card = _fact_card()
    card["current_rate"] = 5.25
    try:
        render_generator_payload(
            fact_card=card,
            atomic_topic="Unemployment Rate",
            style_guide={"section_style_id": "participants_views"},
        )
    except ValueError as exc:
        assert "current_rate" in str(exc)
    else:
        raise AssertionError("current_rate must never enter a teacher payload")


def test_critic_contract_and_candidate_tie_break() -> None:
    first = verify_candidate(
        {
            "reasoning": "The latest reading was 4.0 percent.",
            "final_analysis": "The measure increased.",
            "evidence_ids": ["E1"],
        },
        fact_card=_fact_card(),
    )
    second = verify_candidate(
        {
            "reasoning": "The latest reading was 4.0 percent, after 3.7 percent.",
            "final_analysis": "The two observations indicate an increase.",
            "evidence_ids": ["E1", "E2"],
        },
        fact_card=_fact_card(),
    )
    assert critic_accepts(validate_critic(_critic())) is True
    assert select_candidate([(first, _critic()), (second, _critic())]) == 1
