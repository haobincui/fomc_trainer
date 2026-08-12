from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from open_r1.configs import GRPOScriptArguments
from open_r1.trainer.rewards.reward_funcs import analysis_reward_v3_deepseek_low as low_module
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v3_deepseek_high import (
    grounded_analysis_reward_v3_deepseek_high,
)
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v3_deepseek_low import (
    DEEPSEEK_TIMEOUT600_SECONDS,
    DEEPSEEK_TIMEOUT_SECONDS,
    build_deepseek_low_batch_request_v3,
    grounded_analysis_reward_v3_deepseek_low,
    grounded_analysis_reward_v3_deepseek_low_timeout600,
    numeric_grounding_v3_deepseek_low,
)
from open_r1.trainer.rewards.reward_register import get_reward_funcs


_RUBRICS = (
    "data_fidelity",
    "trend_reasoning",
    "policy_relevance",
    "uncertainty_calibration",
    "fomc_style",
)


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(range(max(1, len(text.split()))))


class _Responses:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        payload = {
            candidate_id: {
                **{key: 4 for key in _RUBRICS},
                "violations": [],
            }
            for candidate_id in ("candidate_0", "candidate_1", "candidate_2", "candidate_3")
        }
        visible = json.dumps(payload, separators=(",", ":"))
        return SimpleNamespace(
            status="completed",
            model="deepseek-v4-flash",
            id=f"response-{len(self.calls)}",
            output_text=visible,
            output=[
                SimpleNamespace(
                    type="reasoning",
                    content=[SimpleNamespace(type="reasoning_text", text="private reasoning")],
                ),
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text=visible)],
                ),
            ],
            usage=SimpleNamespace(
                input_tokens=600,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
                output_tokens=300,
                output_tokens_details=SimpleNamespace(reasoning_tokens=200),
                total_tokens=900,
            ),
        )


class _Client:
    def __init__(self) -> None:
        self.responses = _Responses()


def _evidence() -> str:
    return json.dumps(
        {
            "atomic_topic": "Federal Funds Rate",
            "evidence": [
                {
                    "metric": "Federal Funds Effective Rate",
                    "observation_date": "2025-01-01",
                    "units": "Percent",
                    "value": "4.25",
                }
            ],
        },
        sort_keys=True,
    )


def _completions() -> list[list[dict[str, str]]]:
    return [
        [
            {
                "content": (
                    f"Evidence shows a 4.25 percent rate for candidate {index}."
                    "\n</think>\nThe effective rate was 4.25 percent."
                )
            }
        ]
        for index in range(4)
    ]


def test_low_reward_is_additively_registered() -> None:
    args = GRPOScriptArguments(
        dataset_name="dummy",
        reward_funcs=[
            "grounded_analysis_v3_deepseek_high",
            "grounded_analysis_v3_deepseek_low",
            "grounded_analysis_v3_deepseek_low_timeout600",
        ],
    )

    rewards = get_reward_funcs(args)

    assert [reward.__name__ for reward in rewards] == [
        "grounded_analysis_reward_v3_deepseek_high",
        "grounded_analysis_reward_v3_deepseek_low",
        "grounded_analysis_reward_v3_deepseek_low_timeout600",
    ]
    assert rewards[1].keywords["max_completion_tokens"] == 16_384
    assert rewards[1].keywords["timeout"] == DEEPSEEK_TIMEOUT_SECONDS
    assert rewards[1].keywords["max_retries"] == 2
    assert rewards[2].keywords["timeout"] == DEEPSEEK_TIMEOUT600_SECONDS
    assert rewards[2].keywords["max_completion_tokens"] == 16_384
    assert rewards[2].keywords["max_retries"] == 2


def test_low_request_locks_effort_budget_and_four_candidate_schema() -> None:
    request = build_deepseek_low_batch_request_v3(
        evidence=_evidence(),
        candidates=["candidate"] * 4,
    )

    assert request["model"] == "deepseek-v4-flash"
    assert request["reasoning"] == {"effort": "low"}
    assert request["max_output_tokens"] == 16_384
    output_format = request["text"]["format"]
    assert output_format["strict"] is True
    assert set(output_format["schema"]["properties"]) == {
        "candidate_0",
        "candidate_1",
        "candidate_2",
        "candidate_3",
    }
    assert output_format["schema"]["additionalProperties"] is False


def test_low_cache_and_logs_are_isolated_and_math_matches_high(tmp_path: Path) -> None:
    completions = _completions()
    evidence = [_evidence()] * 4
    tokenizer = _Tokenizer()
    low_client = _Client()
    high_client = _Client()
    low_path = tmp_path / "low" / "reward.jsonl"
    high_path = tmp_path / "high" / "reward.jsonl"

    low_rewards = grounded_analysis_reward_v3_deepseek_low(
        completions,
        evidence,
        meeting_date=[""] * 4,
        save_path=str(low_path),
        judge_tokenizer=tokenizer,
        client=low_client,
    )
    high_rewards = grounded_analysis_reward_v3_deepseek_high(
        completions,
        evidence,
        meeting_date=[""] * 4,
        save_path=str(high_path),
        judge_tokenizer=tokenizer,
        client=high_client,
    )

    assert low_rewards == pytest.approx(high_rewards, abs=1e-12)
    assert low_client.responses.calls[0]["reasoning"] == {"effort": "low"}
    assert high_client.responses.calls[0]["reasoning"] == {"effort": "high"}
    assert (low_path.parent / "deepseek_low_reward_contract.json").is_file()
    assert (low_path.parent / "deepseek_low_reward_cache").is_dir()
    assert not (low_path.parent / "deepseek_reward_contract.json").exists()
    assert not (low_path.parent / "deepseek_reward_cache").exists()
    assert (high_path.parent / "deepseek_reward_contract.json").is_file()
    assert (high_path.parent / "deepseek_reward_cache").is_dir()
    low_rows = [json.loads(line) for line in low_path.read_text().splitlines()]
    high_rows = [json.loads(line) for line in high_path.read_text().splitlines()]
    assert {row["type"] for row in low_rows} == {
        "grounded_analysis_v3_deepseek_low"
    }
    assert {row["type"] for row in high_rows} == {
        "grounded_analysis_v3_deepseek_high"
    }
    assert all(row["provider"]["reasoning_effort"] == "low" for row in low_rows)

    cached = grounded_analysis_reward_v3_deepseek_low(
        completions,
        evidence,
        meeting_date=[""] * 4,
        save_path=str(low_path),
        judge_tokenizer=tokenizer,
        client=low_client,
    )
    assert cached == pytest.approx(low_rewards, abs=1e-12)
    assert len(low_client.responses.calls) == 1


def test_low_monetary_scale_math_matches_v3_extension() -> None:
    evidence = json.dumps(
        {
            "atomic_topic": "credit",
            "evidence": [
                {"metric": "TOTALSL", "value": "2502.7", "units": "Billions of Dollars"}
            ],
        }
    )

    assert numeric_grounding_v3_deepseek_low(
        "Credit was approximately 2.5027 trillion dollars.", evidence
    ) == (1.0, [])
    assert numeric_grounding_v3_deepseek_low(
        "Credit was 2502.7 million dollars.", evidence
    ) == (0.0, [2502.7])


def test_low_live_transport_accepts_600_and_rejects_other_timeouts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _Client()
    client_settings: list[dict[str, Any]] = []

    def fake_get_client(*, api_key: str, url: str, timeout: float) -> _Client:
        client_settings.append({"api_key": api_key, "url": url, "timeout": timeout})
        return client

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-secret")
    monkeypatch.setattr(low_module, "_get_client", fake_get_client)
    rewards = grounded_analysis_reward_v3_deepseek_low_timeout600(
        _completions(),
        [_evidence()] * 4,
        meeting_date=[""] * 4,
        save_path=str(tmp_path / "allowed" / "reward.jsonl"),
        judge_tokenizer=_Tokenizer(),
        timeout=600,
        max_retries=2,
    )

    assert len(rewards) == 4
    assert client_settings == [
        {
            "api_key": "test-only-secret",
            "url": "https://api.deepseek.com",
            "timeout": 600.0,
        }
    ]
    assert client.responses.calls[0]["reasoning"] == {"effort": "low"}
    assert client.responses.calls[0]["max_output_tokens"] == 16_384

    with pytest.raises(ValueError, match="exactly 600 seconds"):
        grounded_analysis_reward_v3_deepseek_low_timeout600(
            _completions(),
            [_evidence()] * 4,
            meeting_date=[""] * 4,
            save_path=str(tmp_path / "rejected-wrapper" / "reward.jsonl"),
            judge_tokenizer=_Tokenizer(),
            timeout=599,
        )
    with pytest.raises(ValueError, match="exactly one of"):
        grounded_analysis_reward_v3_deepseek_low(
            _completions(),
            [_evidence()] * 4,
            meeting_date=[""] * 4,
            save_path=str(tmp_path / "rejected-base" / "reward.jsonl"),
            judge_tokenizer=_Tokenizer(),
            timeout=601,
        )


def test_low_420_and_600_contracts_and_cache_keys_are_isolated(tmp_path: Path) -> None:
    inputs = (_completions(), [_evidence()] * 4)
    path420 = tmp_path / "timeout420" / "reward.jsonl"
    path600 = tmp_path / "timeout600" / "reward.jsonl"

    grounded_analysis_reward_v3_deepseek_low(
        *inputs,
        meeting_date=[""] * 4,
        save_path=str(path420),
        judge_tokenizer=_Tokenizer(),
        client=_Client(),
    )
    grounded_analysis_reward_v3_deepseek_low_timeout600(
        *inputs,
        meeting_date=[""] * 4,
        save_path=str(path600),
        judge_tokenizer=_Tokenizer(),
        client=_Client(),
    )

    contract420 = json.loads(
        (path420.parent / "deepseek_low_reward_contract.json").read_text()
    )
    contract600 = json.loads(
        (path600.parent / "deepseek_low_reward_contract.json").read_text()
    )
    cache420 = list((path420.parent / "deepseek_low_reward_cache").glob("*.json"))
    cache600 = list((path600.parent / "deepseek_low_reward_cache").glob("*.json"))
    assert contract420["timeout_seconds"] == 420.0
    assert contract600["timeout_seconds"] == 600.0
    assert contract420["request_contract_sha256"] != contract600[
        "request_contract_sha256"
    ]
    assert len(cache420) == len(cache600) == 1
    assert cache420[0].name != cache600[0].name
