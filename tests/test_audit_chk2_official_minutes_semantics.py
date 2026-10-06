from __future__ import annotations

import json

import pytest

from jobs.generation.audit_chk2_official_minutes_semantics import (
    SemanticAuditError,
    _failure_reasons,
    _validate_verdicts,
)


def test_validate_verdicts_preserves_strict_identity_and_types() -> None:
    content = json.dumps(
        {
            "audits": [
                {
                    "sample_id": "sample-1",
                    "target_supported": True,
                    "same_topic": True,
                    "reasoning_compatible": True,
                    "confidence": 0.95,
                    "note": "All three semantic axes pass.",
                }
            ]
        }
    )

    verdicts = _validate_verdicts(content, expected_ids=["sample-1"])

    assert verdicts == [
        {
            "sample_id": "sample-1",
            "target_supported": True,
            "same_topic": True,
            "reasoning_compatible": True,
            "confidence": 0.95,
            "note": "All three semantic axes pass.",
        }
    ]


def test_validate_verdicts_fails_closed_on_changed_identity() -> None:
    content = json.dumps(
        {
            "audits": [
                {
                    "sample_id": "wrong-id",
                    "target_supported": True,
                    "same_topic": True,
                    "reasoning_compatible": True,
                    "confidence": 0.9,
                    "note": "Identity changed.",
                }
            ]
        }
    )

    with pytest.raises(SemanticAuditError, match="identity changed"):
        _validate_verdicts(content, expected_ids=["sample-1"])


def test_machine_pass_requires_all_axes_and_confidence() -> None:
    passing = {
        "target_supported": True,
        "same_topic": True,
        "reasoning_compatible": True,
        "confidence": 0.9,
    }
    low_confidence = {**passing, "confidence": 0.7}
    topic_failure = {**passing, "same_topic": False}

    assert _failure_reasons(passing, confidence_threshold=0.8) == []
    assert _failure_reasons(low_confidence, confidence_threshold=0.8) == [
        "confidence_below_threshold"
    ]
    assert _failure_reasons(topic_failure, confidence_threshold=0.8) == [
        "same_topic"
    ]
