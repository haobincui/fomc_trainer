from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobs.retrain_v2.decision_rollout_pilot import (
    GENERATIONS_PER_PROMPT,
    PILOT_PROMPTS,
    DecisionPilotError,
    PilotCompletion,
    _load_config,
    evaluate_pilot,
    select_pilot_rows,
)


VALID = '<think>brief</think><answer>{"direction":"hold","magnitude_bp":0}</answer>'
INVALID = '<think>brief</think><answer>{"direction":"wait","magnitude_bp":0}</answer>'


def _rows(count: int = 13):
    return [
        {"sample_id": f"sample-{index:02d}", "prompt": f"synthetic prompt {index}"}
        for index in range(count)
    ]


def _selected():
    return select_pilot_rows(_rows())


def test_selection_is_deterministic_and_does_not_return_plain_ids():
    first = select_pilot_rows(_rows())
    second = select_pilot_rows(list(reversed(_rows())))
    assert [item["row_id_sha256"] for item in first] == [
        item["row_id_sha256"] for item in second
    ]
    assert len(first) == PILOT_PROMPTS
    assert all(len(item["row_id_sha256"]) == 64 for item in first)


def test_selection_rejects_too_few_or_duplicate_rows():
    with pytest.raises(DecisionPilotError, match="at least"):
        select_pilot_rows(_rows(PILOT_PROMPTS - 1))
    rows = _rows()
    rows[-1] = dict(rows[0])
    with pytest.raises(DecisionPilotError, match="duplicate"):
        select_pilot_rows(rows)


@pytest.mark.parametrize(
    ("truncated", "valid", "expected"),
    [
        (3, 20, "passed"),
        (4, 32, "failed"),
        (0, 19, "failed"),
    ],
)
def test_pilot_thresholds_are_strict(truncated: int, valid: int, expected: str):
    emitted = 0

    def generate(_prompt: str, _seed: int):
        nonlocal emitted
        completions = []
        for _ in range(GENERATIONS_PER_PROMPT):
            index = emitted
            emitted += 1
            completions.append(
                PilotCompletion(
                    text=VALID if index < valid else INVALID,
                    completion_tokens=64,
                    truncated=index < truncated,
                )
            )
        return 200, completions

    result = evaluate_pilot(_selected(), generate=generate)
    assert result["status"] == expected
    assert result["summary"]["valid_completions"] == valid
    assert result["summary"]["truncated_completions"] == truncated
    serialized = json.dumps(result)
    assert "synthetic prompt" not in serialized
    assert "<think>" not in serialized


def test_pilot_rejects_wrong_generation_count_or_prompt_overflow():
    with pytest.raises(DecisionPilotError, match="exactly four"):
        evaluate_pilot(
            _selected(),
            generate=lambda _prompt, _seed: (
                100,
                [PilotCompletion(VALID, 10, False)],
            ),
        )
    with pytest.raises(DecisionPilotError, match="exceeds"):
        evaluate_pilot(
            _selected(),
            generate=lambda _prompt, _seed: (
                2561,
                [PilotCompletion(VALID, 10, False)] * GENERATIONS_PER_PROMPT,
            ),
        )


def test_config_contract_is_exact(tmp_path: Path):
    config = {
        "dataset_prompt_column": "prompt",
        "dataset_test_split": "validation",
        "load_in_4bit": True,
        "max_completion_length": 512,
        "num_generations": 4,
        "temperature": 0.7,
        "top_p": 0.9,
        "mask_truncated_completions": True,
        "system_prompt": "synthetic",
    }
    path = tmp_path / "chk4.yaml"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert _load_config(path)["num_generations"] == 4
    config["num_generations"] = 8
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(DecisionPilotError, match="num_generations"):
        _load_config(path)
