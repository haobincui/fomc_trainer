from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from open_r1.configs import GRPOScriptArguments
from open_r1.trainer.rewards.reward_funcs.decision_reward_v2 import (
    decision_dense_reward_v2,
)
from open_r1.trainer.rewards.reward_funcs.decision_reward_v3 import (
    FENCED_JSON,
    FENCED_SEMANTIC_DISCOUNT,
    INVALID,
    STRICT_JSON,
    decision_dense_reward_v3,
    parse_decision_response_v3,
)
from open_r1.trainer.rewards.reward_register import get_reward_funcs


def _completion(answer: str, reasoning: str = "Evidence supports the decision."):
    return [{"content": f"{reasoning}\n</think>\n{answer}"}]


def _fence(value: object) -> str:
    return "```json\n" + json.dumps(value, separators=(",", ":")) + "\n```"


def test_strict_json_keeps_v2_exact_maximum() -> None:
    answer = json.dumps({"direction": "cut", "magnitude_bp": 50})

    assert decision_dense_reward_v2(
        [_completion(answer)], direction=["cut"], magnitude_bp=[50]
    ) == [1.0]
    assert decision_dense_reward_v3(
        [_completion(answer)], direction=["cut"], magnitude_bp=[50]
    ) == [1.0]
    parsed = parse_decision_response_v3(_completion(answer)[0]["content"])
    assert parsed.response_format == STRICT_JSON
    assert parsed.format_valid is True
    assert parsed.fenced_recovered is False
    assert parsed.rejection_reason is None


def test_strict_json_keeps_v2_dense_partial_score() -> None:
    answer = json.dumps({"direction": "hike", "magnitude_bp": 25})

    v2 = decision_dense_reward_v2(
        [_completion(answer)], direction=["hike"], magnitude_bp=[75]
    )[0]
    v3 = decision_dense_reward_v3(
        [_completion(answer)], direction=["hike"], magnitude_bp=[75]
    )[0]

    assert v2 == pytest.approx(0.65)
    assert v3 == pytest.approx(v2)


def test_exact_fenced_json_recovers_prediction_with_fixed_discount() -> None:
    answer = _fence({"direction": "hold", "magnitude_bp": 0})
    parsed = parse_decision_response_v3(_completion(answer)[0]["content"])
    reward = decision_dense_reward_v3(
        [_completion(answer)], direction=["hold"], magnitude_bp=[0]
    )[0]

    assert FENCED_SEMANTIC_DISCOUNT == 0.25
    assert parsed.prediction == ("hold", 0)
    assert parsed.response_format == FENCED_JSON
    assert parsed.format_valid is False
    assert parsed.fenced_recovered is True
    assert reward == pytest.approx(0.2375)
    # V2 is deliberately untouched and still rejects the fence.
    assert decision_dense_reward_v2(
        [_completion(answer)], direction=["hold"], magnitude_bp=[0]
    ) == [0.0]


def test_fenced_partial_and_wrong_direction_scores() -> None:
    partial = _fence({"direction": "hike", "magnitude_bp": 50})
    wrong = _fence({"direction": "cut", "magnitude_bp": 25})

    rewards = decision_dense_reward_v3(
        [_completion(partial), _completion(wrong)],
        direction=["hike", "hike"],
        magnitude_bp=[75, 25],
    )

    # 0.25 * (0.45 direction + 0.30 * 0.75 magnitude), no format point.
    assert rewards[0] == pytest.approx(0.16875)
    assert rewards[1] == 0.0


def test_optional_leading_think_and_outer_whitespace_are_accepted() -> None:
    text = (
        "  <think>Evidence is balanced.\n</think>\n  "
        '{"direction":"hold","magnitude_bp":0}  '
    )

    parsed = parse_decision_response_v3(text)

    assert parsed.prediction == ("hold", 0)
    assert parsed.response_format == STRICT_JSON


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "empty_completion"),
        ('{"direction":"hold","magnitude_bp":0}', "missing_think_boundary"),
        ('</think>{"direction":"hold","magnitude_bp":0}', "empty_reasoning"),
        ("reasoning</think>   ", "empty_answer"),
        (
            'reasoning</think>{"direction":"hold","magnitude_bp":0}</think>',
            "multiple_think_boundaries",
        ),
        (
            'reasoning</think>result: {"direction":"hold","magnitude_bp":0}',
            "answer_is_not_plain_or_fenced_json",
        ),
        (
            'reasoning</think>{"direction":"hold","magnitude_bp":0} trailing',
            "extra_text_after_json",
        ),
        (
            'reasoning</think>{"direction":"hold","magnitude_bp":0}'
            '{"direction":"hold","magnitude_bp":0}',
            "multiple_json_values",
        ),
        (
            'reasoning</think>{"direction":"hold"}',
            "decision_schema_keys_not_exact",
        ),
        (
            'reasoning</think>{"direction":"hold","magnitude_bp":0,"extra":1}',
            "decision_schema_keys_not_exact",
        ),
        (
            'reasoning</think>{"direction":"hold","magnitude_bp":"0"}',
            "magnitude_type_invalid",
        ),
        (
            'reasoning</think>{"direction":"hold","magnitude_bp":true}',
            "magnitude_type_invalid",
        ),
        (
            'reasoning</think>{"direction":"hold","magnitude_bp":25}',
            "hold_magnitude_invalid",
        ),
        (
            'reasoning</think>{"direction":"cut","magnitude_bp":0}',
            "action_magnitude_invalid",
        ),
        (
            'reasoning</think>{"direction":"pause","magnitude_bp":0}',
            "direction_domain_invalid",
        ),
        (
            'reasoning</think>{"direction":"hold","magnitude_bp":',
            "invalid_or_incomplete_json",
        ),
        (
            'reasoning</think>```json\n{"direction":"hold","magnitude_bp":0}',
            "invalid_or_incomplete_json_fence",
        ),
        (
            'reasoning</think>```json\n{"direction":"hold","magnitude_bp":0}\n``` extra',
            "invalid_or_incomplete_json_fence",
        ),
        (
            'reasoning</think>```json\n{"direction":"hold","magnitude_bp":0}'
            '\n{"direction":"hold","magnitude_bp":0}\n```',
            "fenced_multiple_json_values",
        ),
        (
            'reasoning</think>```json\n{"direction":"hold","magnitude_bp":"0"}\n```',
            "magnitude_type_invalid",
        ),
        (
            'reasoning</think>```JSON\n{"direction":"hold","magnitude_bp":0}\n```',
            "answer_is_not_plain_or_fenced_json",
        ),
    ],
)
def test_all_non_contract_shapes_stay_zero(text: str, reason: str) -> None:
    parsed = parse_decision_response_v3(text)
    reward = decision_dense_reward_v3(
        [[{"content": text}]], direction=["hold"], magnitude_bp=[0]
    )[0]

    assert parsed.prediction is None
    assert parsed.response_format == INVALID
    assert parsed.format_valid is False
    assert parsed.fenced_recovered is False
    assert parsed.rejection_reason == reason
    assert reward == 0.0


def test_audit_records_distinguish_strict_fenced_and_invalid(tmp_path) -> None:
    save_path = tmp_path / "reward.jsonl"
    strict = json.dumps({"direction": "hold", "magnitude_bp": 0})
    fenced = _fence({"direction": "cut", "magnitude_bp": 25})
    invalid = "```json\n{}\n```"

    rewards = decision_dense_reward_v3(
        [_completion(strict), _completion(fenced), _completion(invalid)],
        direction=["hold", "cut", "hike"],
        magnitude_bp=[0, 25, 25],
        meeting_date=["a", "b", "c"],
        save_path=str(save_path),
    )
    records = [json.loads(line) for line in save_path.read_text().splitlines()]

    assert rewards == pytest.approx([1.0, 0.2375, 0.0])
    assert [record["response_format"] for record in records] == [
        STRICT_JSON,
        FENCED_JSON,
        INVALID,
    ]
    assert records[0]["format_valid"] is True
    assert records[0]["strict_format_valid"] is True
    assert records[0]["fenced_recovered"] is False
    assert records[0]["recovery_mode"] == "none"
    assert records[0]["semantic_discount"] == 1.0
    assert records[1]["format_valid"] is False
    assert records[1]["strict_format_valid"] is False
    assert records[1]["fenced_recovered"] is True
    assert records[1]["recovery_mode"] == FENCED_JSON
    assert records[1]["semantic_discount"] == 0.25
    assert records[1]["prediction"] == {"direction": "cut", "magnitude_bp": 25}
    assert records[2]["prediction"] is None
    assert records[2]["rejection_reason"] == "decision_schema_keys_not_exact"
    # Audits contain decisions and parser facts, never completion/reasoning text.
    serialized = json.dumps(records)
    assert "Evidence supports the decision" not in serialized
    assert "```json" not in serialized


def test_rate_change_target_and_target_validation_match_v2() -> None:
    answer = json.dumps({"direction": "cut", "magnitude_bp": 25})
    assert decision_dense_reward_v3(
        [_completion(answer)], rate_change=["cut 25 bp"]
    ) == [1.0]
    with pytest.raises(ValueError, match="invalid decision target"):
        decision_dense_reward_v3(
            [_completion(answer)], direction=["cut"], magnitude_bp=[0]
        )


def test_registry_exposes_v3_and_injects_audit_path(tmp_path) -> None:
    args = GRPOScriptArguments(
        dataset_name="dummy",
        reward_funcs=["decision_dense_v3"],
        save_reward=True,
    )
    reward_func = get_reward_funcs(args, SimpleNamespace(output_dir=str(tmp_path)))[0]
    answer = _fence({"direction": "hike", "magnitude_bp": 75})

    assert reward_func.__name__ == "decision_dense_reward_v3"
    assert reward_func(
        completions=[_completion(answer)], direction=["hike"], magnitude_bp=[75]
    ) == [pytest.approx(0.2375)]
    record = json.loads((tmp_path / "reward.jsonl").read_text())
    assert record["type"] == "decision_dense_v3"
    assert record["response_format"] == FENCED_JSON
    assert record["recovery_mode"] == FENCED_JSON
