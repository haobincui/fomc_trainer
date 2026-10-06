from __future__ import annotations

import pytest

from jobs.eval import paper_chk2_text_similarity_common as subject


def test_reference_is_unique_think_suffix() -> None:
    response = "faithful plan\n</think>\nOne formal Minutes paragraph."
    assert subject.extract_reference(response) == "One formal Minutes paragraph."


@pytest.mark.parametrize(
    "response,match",
    [
        ("no boundary", "one </think>"),
        ("plan</think>a</think>b", "one </think>"),
        ("plan</think>   ", "reference is empty"),
        ("plan</think>one\n\ntwo", "not one paragraph"),
    ],
)
def test_reference_contract_fails_closed(response: str, match: str) -> None:
    with pytest.raises(subject.SimilarityPreparationError, match=match):
        subject.extract_reference(response)


def test_fixed_population_and_generation_contract() -> None:
    assert sum(subject.EXPECTED_SPLIT_ROWS.values()) == 391
    assert sum(subject.EXPECTED_SPLIT_MEETINGS.values()) == 124
    assert len(subject.REPLICATE_SEEDS) == len(set(subject.REPLICATE_SEEDS)) == 10
