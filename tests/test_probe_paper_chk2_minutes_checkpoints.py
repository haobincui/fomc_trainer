from __future__ import annotations

import json

import pytest

from jobs.retrain_v2 import probe_paper_chk2_minutes_checkpoints as probe
from jobs.retrain_v2 import judge_paper_chk2_minutes_probe as judge
from jobs.retrain_v2 import select_paper_chk2_minutes_probe_checkpoint as selector


SOURCE = (
    "In March 2024, participants observed that inflation was 2.5 percent "
    "and economic growth remained moderate."
)
ANSWER = (
    "Participants observed that, in March 2024, inflation was 2.5 percent "
    "and economic growth remained moderate, while presenting the assessment "
    "in neutral terms and without adding any further substantive conclusion."
)


def _completion(answer: str = ANSWER) -> str:
    return (
        "I will preserve the date, quantity, attribution, and direction while "
        "using formal institutional wording.\n</think>\n" + answer
    )


def _ids() -> list[int]:
    return [*range(10, 130), 999]


def test_extract_source_analysis_requires_strict_single_field_json() -> None:
    prompt = (
        "Rewrite the following analysis as formal FOMC Minutes prose:\n\n"
        + json.dumps({"analysis": SOURCE})
    )
    assert probe.extract_source_analysis(prompt) == SOURCE
    with pytest.raises(probe.PaperChk2ProbeError):
        probe.extract_source_analysis(
            "Rewrite the following analysis as formal FOMC Minutes prose:\n\n"
            + json.dumps({"analysis": SOURCE, "target": "forbidden"})
        )


def test_valid_completion_passes_deterministic_hard_gates() -> None:
    result = probe.analyze_student_completion(
        completion=_completion(),
        generated_token_ids=_ids(),
        eos_token_ids={999},
        max_new_tokens=2048,
        prompt_token_count=300,
        source_analysis=SOURCE,
    )
    assert result["deterministic_hard_gate_pass"] is True
    assert result["deterministic_hard_gate_failures"] == []
    assert result["numeric_multiset_preserved"] is True
    assert result["date_set_preserved"] is True
    assert result["attribution_set_preserved"] is True
    assert result["semantic_factual_fidelity_status"] == "PENDING_FRESH_VALIDATOR_A"
    assert result["minutes_style_status"] == "PENDING_FRESH_VALIDATOR_B"


def test_signed_numeric_flip_fails_deterministic_fidelity() -> None:
    source = SOURCE.replace("2.5 percent", "-2.5 percent")
    answer = ANSWER.replace("2.5 percent", "+2.5 percent")
    result = probe.analyze_student_completion(
        completion=_completion(answer),
        generated_token_ids=_ids(),
        eos_token_ids={999},
        max_new_tokens=2048,
        prompt_token_count=300,
        source_analysis=source,
    )
    assert result["numeric_multiset_preserved"] is False
    assert "numeric_multiset_not_preserved" in result["deterministic_hard_gate_failures"]


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        (ANSWER.replace("2.5 percent", "3.0 percent"), "numeric_multiset_not_preserved"),
        (ANSWER.replace("March 2024", "April 2024"), "date_set_not_preserved"),
        (ANSWER.replace("Participants observed", "It was observed"), "attribution_set_not_preserved"),
        (ANSWER + "\nA second paragraph follows.", "final_answer_not_single_paragraph"),
    ],
)
def test_deterministic_fidelity_and_paragraph_failures(
    answer: str, reason: str
) -> None:
    result = probe.analyze_student_completion(
        completion=_completion(answer),
        generated_token_ids=_ids(),
        eos_token_ids=999,
        max_new_tokens=2048,
        prompt_token_count=300,
        source_analysis=SOURCE,
    )
    assert result["deterministic_hard_gate_pass"] is False
    assert reason in result["deterministic_hard_gate_failures"]


def test_missing_eos_and_periodic_tail_fail_closed() -> None:
    unit = list(range(1, 17))
    generated = unit * 4
    result = probe.analyze_student_completion(
        completion=_completion(),
        generated_token_ids=generated,
        eos_token_ids=999,
        max_new_tokens=2048,
        prompt_token_count=300,
        source_analysis=SOURCE,
    )
    assert result["deterministic_hard_gate_pass"] is False
    assert "missing_terminal_eos" in result["deterministic_hard_gate_failures"]
    assert "strict_periodic_tail" in result["deterministic_hard_gate_failures"]


def _candidate(index: int) -> dict:
    return {
        "sample_id": f"sample-{index:02d}",
        "line_number": index + 1,
        "meeting_date": f"2024-{index + 1:02d}-01",
        "atomic_topic": f"topic-{index % 5}",
        "section_style_id": f"style-{index % 2}",
        "prompt_sha256": f"p{index}",
        "source_analysis_sha256": f"s{index}",
        "prompt_token_count": 100 + index * 10,
        "source_features": {
            "numeric_occurrences": index,
            "date_values": index % 4,
            "attribution_categories": ["participants"] if index % 3 == 0 else [],
        },
    }


def test_eight_row_selection_is_unique_deterministic_and_diverse() -> None:
    candidates = [_candidate(index) for index in range(16)]
    first = probe.select_probe_rows(candidates, 8)
    second = probe.select_probe_rows(candidates, 8)
    assert first == second
    assert len(first) == len({row["sample_id"] for row in first}) == 8
    assert {row["section_style_id"] for row in first} == {"style-0", "style-1"}
    assert len({row["meeting_date"] for row in first}) == 8
    assert {"length_min", "length_max", "numeric_heavy", "date_heavy"} <= {
        row["selection_tag"] for row in first
    }


def test_full_validation_selection_preserves_line_order() -> None:
    candidates = list(reversed([_candidate(index) for index in range(5)]))
    selected = probe.select_probe_rows(candidates, 5)
    assert [row["line_number"] for row in selected] == [1, 2, 3, 4, 5]
    assert {row["selection_tag"] for row in selected} == {"full_validation"}


@pytest.mark.parametrize(
    ("deterministic", "a_pass", "a_unresolved", "b_pass", "b_unresolved", "expected"),
    [
        (False, None, False, None, False, "FAIL_DETERMINISTIC"),
        (True, None, True, None, False, "UNRESOLVED_JUDGE_FAILURE"),
        (True, False, False, None, False, "FAIL_VALIDATOR_A"),
        (True, True, False, None, True, "UNRESOLVED_JUDGE_FAILURE"),
        (True, True, False, False, False, "FAIL_VALIDATOR_B"),
        (True, True, False, True, False, "PASS_ALL_HARD_GATES"),
    ],
)
def test_fresh_judge_ordering_classification(
    deterministic: bool,
    a_pass: bool | None,
    a_unresolved: bool,
    b_pass: bool | None,
    b_unresolved: bool,
    expected: str,
) -> None:
    assert (
        judge.classify_gate_result(
            deterministic_pass=deterministic,
            validator_a_pass=a_pass,
            validator_a_unresolved=a_unresolved,
            validator_b_pass=b_pass,
            validator_b_unresolved=b_unresolved,
        )
        == expected
    )


def test_relaxed_checkpoint_policy_tolerates_two_factual_misses() -> None:
    generation_rows = []
    judge_rows = []
    for index in range(8):
        factual_failure = index >= 6
        generation_rows.append(
            {
                "sample_id": f"sample-{index}",
                "metrics": {
                    "deterministic_hard_gate_failures": (
                        ["numeric_multiset_not_preserved"] if factual_failure else []
                    )
                },
            }
        )
        if factual_failure:
            judge_rows.append(
                {
                    "sample_id": f"sample-{index}",
                    "deterministic_pass": False,
                    "validator_a": {"status": "NOT_RUN"},
                    "validator_b": {"status": "NOT_RUN"},
                    "hard_gate_status": "FAIL_DETERMINISTIC",
                }
            )
        elif index == 5:
            judge_rows.append(
                {
                    "sample_id": f"sample-{index}",
                    "deterministic_pass": True,
                    "validator_a": {"status": "FAIL"},
                    "validator_b": {"status": "NOT_RUN"},
                    "hard_gate_status": "FAIL_VALIDATOR_A",
                }
            )
        else:
            b_pass = index < 3
            judge_rows.append(
                {
                    "sample_id": f"sample-{index}",
                    "deterministic_pass": True,
                    "validator_a": {"status": "PASS"},
                    "validator_b": {
                        "status": "PASS" if b_pass else "FAIL",
                        "mean_score": 7.0 if b_pass else 5.0,
                    },
                    "hard_gate_status": (
                        "PASS_ALL_HARD_GATES" if b_pass else "FAIL_VALIDATOR_B"
                    ),
                }
            )
    result = selector.evaluate_checkpoint(
        generation_rows, judge_rows, checkpoint=50
    )
    assert result["strict_all_rows_pass"] is False
    assert result["relaxed_probe_pass"] is True
    assert result["metrics"]["deterministic_factual_pass_rate"] == 0.75
    assert result["metrics"]["full_chain_pass_rate"] == 0.375
