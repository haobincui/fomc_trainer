from __future__ import annotations

import json

import pytest

from jobs.generation.materialize_chk3_training_data import (
    MaterializationError,
    _fact_card_supports_numeric_surface,
    _validate_clean_analysis,
    is_transport_wrapped,
    project_internal_evidence_citations,
    project_valid_transport_analysis,
)


def test_valid_transport_answer_is_projected() -> None:
    answer = "Credit conditions remained mixed, while uncertainty persisted."
    wrapped = json.dumps({"answer": answer, "evidence_ids": ["ev-abcdef123456"]})

    assert is_transport_wrapped(wrapped)
    assert project_valid_transport_analysis(wrapped) == answer


def test_nested_transport_answer_is_projected() -> None:
    answer = "Inflation eased in May, but the outlook remained uncertain."
    wrapped = json.dumps({"content": json.dumps({"answer": answer})})

    assert project_valid_transport_analysis(wrapped) == answer


@pytest.mark.parametrize(
    "value",
    [
        "{",
        '{"answer":"The answer was cut off',
        '```json\n{"answer":"The answer was cut off"}',
    ],
)
def test_truncated_transport_requires_source_recovery(value: str) -> None:
    assert project_valid_transport_analysis(value) is None


def test_plain_analysis_is_unchanged() -> None:
    analysis = "Labor-market conditions remained solid, although uncertainty increased."

    assert not is_transport_wrapped(analysis)
    assert project_valid_transport_analysis(analysis) == analysis


def test_semantic_projection_removes_short_internal_evidence_ids() -> None:
    analysis = (
        "Credit grew to 5,600 in November (ev-5600), while the rate eased "
        "to 4.8 percent (ev-a0e7)."
    )

    projected = project_internal_evidence_citations(analysis)

    assert "ev-5600" not in projected
    assert "ev-a0e7" not in projected
    assert "5,600" in projected
    assert "4.8 percent" in projected


def test_clean_analysis_rejects_transport_or_tiny_fragments() -> None:
    with pytest.raises(MaterializationError, match="transport wrapper"):
        _validate_clean_analysis('{"answer":"still wrapped"}', sample_id="sample-a")
    with pytest.raises(MaterializationError, match="implausibly short"):
        _validate_clean_analysis("fragment", sample_id="sample-b")


def test_fact_card_numeric_surface_accepts_exact_dates_and_rejects_rounding() -> None:
    fact_card = {
        "meeting_date": "2009-08-12",
        "evidence": [
            {
                "metric": "10-Year Treasury Rate",
                "value": "96.9476",
                "units": "Index",
            }
        ],
    }

    assert _fact_card_supports_numeric_surface("2009", fact_card)
    assert _fact_card_supports_numeric_surface("10", fact_card)
    assert _fact_card_supports_numeric_surface("96.9476", fact_card)
    assert not _fact_card_supports_numeric_surface("96.95", fact_card)
    assert not _fact_card_supports_numeric_surface("1.6 percent", fact_card)
