from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from jobs.retrain_v2 import benchmark_chk2_deepseek_low_judge as benchmark


def _evaluation() -> dict[str, Any]:
    return {
        "data_fidelity": 4,
        "trend_reasoning": 4,
        "policy_relevance": 4,
        "uncertainty_calibration": 4,
        "fomc_style": 4,
        "violations": [],
    }


def _response(*, status: str = "completed", model: str = benchmark.MODEL) -> Any:
    visible = json.dumps(
        {candidate_id: _evaluation() for candidate_id in benchmark.CANDIDATE_IDS}
    )
    return SimpleNamespace(
        id="response-sensitive-id",
        status=status,
        model=model,
        output_text=visible,
        output=[
            SimpleNamespace(
                type="reasoning",
                content=[
                    SimpleNamespace(
                        type="reasoning_text", text="sensitive hidden reasoning"
                    )
                ],
            ),
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=visible)],
            ),
        ],
        usage=SimpleNamespace(
            input_tokens=500,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
            output_tokens=250,
            output_tokens_details=SimpleNamespace(reasoning_tokens=100),
            total_tokens=750,
        ),
    )


class _Responses:
    def __init__(self, actions: list[Any]) -> None:
        self.actions = list(actions)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.actions:
            raise AssertionError("unexpected provider call")
        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        return action


class _Client:
    def __init__(self, actions: list[Any]) -> None:
        self.responses = _Responses(actions)


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(range(max(1, len(text.split()))))


def _args(output_dir: Path, *, dry_run: bool = False) -> Namespace:
    return Namespace(
        completion_parquet=benchmark.DEFAULT_COMPLETION_PARQUET,
        dataset_dir=benchmark.DEFAULT_DATASET_DIR,
        high_reward_log=benchmark.DEFAULT_HIGH_REWARD_LOG,
        high_canary_reward_log=benchmark.DEFAULT_HIGH_CANARY_REWARD_LOG,
        tokenizer_path=benchmark.DEFAULT_TOKENIZER_PATH,
        output_dir=output_dir,
        dry_run=dry_run,
    )


def test_locked_request_uses_low_effort_and_four_candidates() -> None:
    candidates = [f"reasoning {index}\n</think>\nanswer {index}" for index in range(4)]
    request = benchmark.build_low_batch_request(
        evidence=json.dumps({"atomic_topic": "test", "evidence": []}),
        candidates=candidates,
    )

    assert request["model"] == "deepseek-v4-flash"
    assert request["reasoning"] == {"effort": "low"}
    assert request["max_output_tokens"] == 16_384
    payload = json.loads(request["input"])
    assert list(payload["candidates"]) == list(benchmark.CANDIDATE_IDS)
    assert list(payload["candidates"].values()) == candidates
    assert request["text"]["format"]["strict"] is True


def test_sources_materialize_five_groups_and_fixed_permutations() -> None:
    groups = benchmark.load_benchmark_groups(
        completion_parquet=benchmark.DEFAULT_COMPLETION_PARQUET,
        dataset_dir=benchmark.DEFAULT_DATASET_DIR,
        high_reward_log=benchmark.DEFAULT_HIGH_REWARD_LOG,
        high_canary_reward_log=benchmark.DEFAULT_HIGH_CANARY_REWARD_LOG,
    )

    assert len(groups) == 10
    assert len({group.base_group_id for group in groups}) == 5
    for base_group_id in {group.base_group_id for group in groups}:
        variants = {group.order_variant: group for group in groups if group.base_group_id == base_group_id}
        assert set(variants) == {"original", "permuted_2031"}
        original = variants["original"]
        permuted = variants["permuted_2031"]
        assert permuted.labels == tuple(
            original.labels[index] for index in benchmark.PERMUTATION
        )
        assert permuted.evidence == original.evidence
    assert sum(
        baseline is not None
        for group in groups
        if group.order_variant == "original"
        for baseline in group.high_baseline
    ) == 8


def test_provider_retries_incomplete_then_accepts_low_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "BACKOFF_SECONDS", 0.0)
    client = _Client([_response(status="incomplete"), _response()])
    request = benchmark.build_low_batch_request(
        evidence=json.dumps({"atomic_topic": "test", "evidence": []}),
        candidates=[f"r{index}\n</think>\na{index}" for index in range(4)],
    )

    evaluations, provider = benchmark._call_provider(client=client, request=request)

    assert len(evaluations) == 4
    assert provider["attempts"] == 2
    assert len(client.responses.calls) == 2
    assert all(call["reasoning"] == {"effort": "low"} for call in client.responses.calls)


def test_provider_model_mismatch_fails_without_retry() -> None:
    client = _Client([_response(model="wrong-model"), _response()])
    request = benchmark.build_low_batch_request(
        evidence=json.dumps({"atomic_topic": "test", "evidence": []}),
        candidates=[f"r{index}\n</think>\na{index}" for index in range(4)],
    )

    with pytest.raises(benchmark.LowJudgeProviderError, match="permanently"):
        benchmark._call_provider(client=client, request=request)

    assert len(client.responses.calls) == 1


def test_dry_run_is_network_free_and_writes_nothing(tmp_path: Path) -> None:
    output_dir = tmp_path / "dry-run-output"

    result = benchmark.run_benchmark(_args(output_dir, dry_run=True))

    assert result["status"] == "dry_run"
    assert result["network_requests"] == 0
    assert len(result["groups"]) == 10
    assert result["request"]["reasoning_effort"] == "low"
    assert not output_dir.exists()


def test_fake_run_is_sanitized_and_cache_resume_is_network_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "DEEPSEEK_TEST_SECRET_MUST_NOT_PERSIST"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    output_dir = tmp_path / "benchmark"
    first_client = _Client([_response() for _ in range(10)])

    first = benchmark.run_benchmark(
        _args(output_dir), client=first_client, tokenizer=_Tokenizer()
    )

    assert first["groups"] == 10
    assert first["candidates"] == 40
    assert len(first_client.responses.calls) == 10
    assert len(list((output_dir / "groups").glob("*.json"))) == 10
    assert len((output_dir / "benchmark_records.jsonl").read_text().splitlines()) == 40
    serialized = "".join(
        path.read_text(encoding="utf-8")
        for path in output_dir.rglob("*")
        if path.is_file()
    )
    assert secret not in serialized
    assert "sensitive hidden reasoning" not in serialized

    cached_client = _Client([])
    second = benchmark.run_benchmark(
        _args(output_dir), client=cached_client, tokenizer=_Tokenizer()
    )

    assert second["cache_hits"] == 10
    assert cached_client.responses.calls == []
