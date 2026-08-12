from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from open_r1.configs import GRPOScriptArguments
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v2 import (
    JudgeInfrastructureError,
)
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v3 import (
    grounded_analysis_reward_v3,
    numeric_grounding_v3,
)
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v3_deepseek_high import (
    build_deepseek_batch_request_v3,
    grounded_analysis_reward_v3_deepseek_high,
    numeric_grounding_v3_deepseek_high,
)
from open_r1.trainer.rewards.reward_register import get_reward_funcs


_MODEL = "deepseek-v4-flash"
_MAX_OUTPUT_TOKENS = 16_384
_SLOTS = tuple(f"candidate_{index}" for index in range(4))
_RUBRICS = (
    "data_fidelity",
    "trend_reasoning",
    "policy_relevance",
    "uncertainty_calibration",
    "fomc_style",
)


def test_deepseek_high_reward_is_additively_registered() -> None:
    args = GRPOScriptArguments(
        dataset_name="dummy",
        reward_funcs=[
            "grounded_analysis_v2",
            "grounded_analysis_v3",
            "grounded_analysis_v3_deepseek_high",
        ],
    )

    rewards = get_reward_funcs(args)
    assert [reward.__name__ for reward in rewards] == [
        "grounded_analysis_reward_v2",
        "grounded_analysis_reward_v3",
        "grounded_analysis_reward_v3_deepseek_high",
    ]
    assert rewards[-1].keywords["url"] == "https://api.deepseek.com"
    assert rewards[-1].keywords["model"] == _MODEL
    assert rewards[-1].keywords["max_completion_tokens"] == 16_384
    assert rewards[-1].keywords["timeout"] == 420
    assert rewards[-1].keywords["max_retries"] == 2
    assert rewards[-1].keywords["backoff_seconds"] == 2.0
    assert rewards[-1].keywords["api_key_env"] == "DEEPSEEK_API_KEY"


class _FixedJudgeTokenizer:
    def __init__(self, count: int = 64) -> None:
        self.count = count

    def apply_chat_template(self, _messages: Any, **_kwargs: Any) -> list[int]:
        return list(range(self.count))

    def encode(self, _text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(range(self.count))


class _FakeResponses:
    def __init__(self, *actions: Any) -> None:
        self.actions = list(actions)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.actions:
            raise AssertionError("unexpected DeepSeek Responses API call")
        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        if callable(action):
            return action(kwargs)
        return action


class _FakeClient:
    def __init__(self, *actions: Any) -> None:
        self.responses = _FakeResponses(*actions)


class _StatusError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"provider status {status_code}")
        self.status_code = status_code


@pytest.fixture(autouse=True)
def _clear_judge_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DEEPSEEK_API_KEY",
        "OPEN_R1_JUDGE_API_KEY",
        "OPEN_R1_JUDGE_URL",
        "OPEN_R1_JUDGE_MODEL",
        "FOMC_RETRAIN_JUDGE_MODEL_PATH",
        "FOMC_RETRAIN_JUDGE_MAX_MODEL_LEN",
        "OPENAI_LOG",
        "HTTPX_LOG_LEVEL",
    ):
        monkeypatch.delenv(name, raising=False)


def _tokenizer_dir(tmp_path: Path) -> Path:
    path = tmp_path / "judge-tokenizer"
    path.mkdir(exist_ok=True)
    return path


def _evaluation(
    *,
    data_fidelity: int = 4,
    violations: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "data_fidelity": data_fidelity,
        "trend_reasoning": 4,
        "policy_relevance": 4,
        "uncertainty_calibration": 4,
        "fomc_style": 4,
        "violations": list(violations or []),
    }


def _violation(
    section: str,
    kind: str,
    severity: str,
    quote: str,
    *,
    explanation: str = "The quoted claim is not supported by the evidence.",
) -> dict[str, str]:
    return {
        "section": section,
        "kind": kind,
        "severity": severity,
        "candidate_quote": quote,
        "explanation": explanation,
    }


def _group_payload(
    evaluations: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    rows = evaluations or [_evaluation() for _ in _SLOTS]
    assert len(rows) == 4
    return {slot: value for slot, value in zip(_SLOTS, rows, strict=True)}


def _duplicate_candidate_payload() -> str:
    encoded = json.dumps(_evaluation(), separators=(",", ":"))
    return "{" + ",".join(
        [
            f'"candidate_0":{encoded}',
            f'"candidate_0":{encoded}',
            f'"candidate_1":{encoded}',
            f'"candidate_2":{encoded}',
            f'"candidate_3":{encoded}',
        ]
    ) + "}"


def _response(
    payload: dict[str, Any] | str,
    *,
    model: str = _MODEL,
    hidden: str = "private high-effort reasoning",
    status: str = "completed",
) -> SimpleNamespace:
    visible = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        status=status,
        incomplete_details=None,
        output_text=visible,
        output=[
            SimpleNamespace(
                type="reasoning",
                content=[SimpleNamespace(type="reasoning_text", text=hidden)],
            ),
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=visible)],
            ),
        ],
        usage=SimpleNamespace(
            input_tokens=800,
            output_tokens=320,
            output_tokens_details=SimpleNamespace(reasoning_tokens=256),
            total_tokens=1120,
        ),
        model=model,
        id="response-test",
    )


def _completion(answer: str) -> list[dict[str, str]]:
    return [
        {
            "content": (
                "Evidence supports the conclusion.\n</think>\n" + answer
            )
        }
    ]


def _completions() -> list[list[dict[str, str]]]:
    return [
        _completion("Alpha is supported."),
        _completion("Beta is supported."),
        _completion("Gamma makes a major unsupported claim."),
        _completion("The target meeting voted to cut."),
    ]


def _evidence(marker: str = "stable") -> str:
    return json.dumps(
        {
            "atomic_topic": "financial conditions",
            "evidence": [{"metric": "Condition", "value": marker}],
        },
        sort_keys=True,
    )


def _monetary_evidence(
    value: str = "4864.88470", units: str = "Billions of Dollars"
) -> str:
    return json.dumps(
        {
            "atomic_topic": "money supply",
            "evidence": [
                {
                    "metric": "M2",
                    "series_id": "M2-4864.88470-billion",
                    "units": units,
                    "observation_date": "2025-04-01",
                    "value": value,
                }
            ],
        },
        sort_keys=True,
    )


def _candidate_texts() -> list[str]:
    return [
        "<think>\nEvidence supports the conclusion.\n</think>\n"
        f"<answer>\n{completion[0]['content'].split('</think>', 1)[1].strip()}\n</answer>"
        for completion in _completions()
    ]


@pytest.mark.parametrize(
    "candidate",
    [
        "The balance was approximately 4.8649 trillion.",
        "The balance was approximately $4.8649 trillion dollars.",
        "The balance was 4,864.88470 billion dollars.",
    ],
)
def test_deepseek_numeric_grounding_accepts_equivalent_monetary_scales(
    candidate: str,
) -> None:
    assert numeric_grounding_v3_deepseek_high(
        candidate, _monetary_evidence()
    ) == (1.0, [])


def test_deepseek_numeric_grounding_rejects_equal_raw_value_at_wrong_scale() -> None:
    evidence = _monetary_evidence(value="4")
    candidate = "The balance was 4 million dollars."

    # The v3 raw-number check sees 4 on both sides. DeepSeek High must also
    # validate the explicit scale, so 4 million cannot stand for 4 billion.
    assert numeric_grounding_v3(candidate, evidence) == (1.0, [])
    assert numeric_grounding_v3_deepseek_high(candidate, evidence) == (0.0, [4.0])


@pytest.mark.parametrize(
    "candidate",
    [
        "On 2025-04-01, series M2-4864.88470-billion was observed.",
        "The reported percentage was 4.8649%.",
        "The source identifier was M2-4864.88470-billion.",
    ],
)
def test_deepseek_monetary_extension_preserves_v3_non_scale_golden(
    candidate: str,
) -> None:
    evidence = _monetary_evidence()
    assert numeric_grounding_v3_deepseek_high(
        candidate, evidence
    ) == numeric_grounding_v3(candidate, evidence)


def test_deepseek_reward_uses_local_monetary_scale_validation(tmp_path: Path) -> None:
    completions = [
        _completion("The balance was approximately 4.8649 trillion dollars."),
        _completion("The balance was approximately 4.8649 billion dollars."),
        _completion("The balance was 4,864.88470 billion dollars."),
        _completion("The evidence supports the reported balance."),
    ]

    rewards = _run_reward(
        tmp_path,
        _FakeClient(_response(_group_payload())),
        completions=completions,
        evidence=_monetary_evidence(),
    )

    assert rewards == pytest.approx([1.0, 0.75, 1.0, 1.0])


def _evaluation_for_candidate(candidate: str) -> dict[str, Any]:
    if "Beta is supported" in candidate:
        return _evaluation(data_fidelity=0)
    if "Gamma makes a major unsupported claim" in candidate:
        return _evaluation(
            violations=[
                _violation(
                    "answer",
                    "factual",
                    "major",
                    "Gamma makes a major unsupported claim",
                )
            ]
        )
    if "target meeting voted to cut" in candidate:
        return _evaluation(
            violations=[
                _violation(
                    "answer",
                    "target_leakage",
                    "major",
                    "The target meeting voted to cut",
                )
            ]
        )
    return _evaluation()


def _mapped_response(kwargs: dict[str, Any]) -> SimpleNamespace:
    request_input = json.loads(kwargs["input"])
    candidates = request_input["candidates"]
    evaluations = [
        _evaluation_for_candidate(json.dumps(candidates[slot], ensure_ascii=False))
        for slot in _SLOTS
    ]
    return _response(_group_payload(evaluations))


def _run_reward(
    tmp_path: Path,
    client: _FakeClient,
    *,
    completions: list[list[dict[str, str]]] | None = None,
    evidence: str | None = None,
    save_path: Path | None = None,
    max_retries: int = 2,
) -> list[float]:
    evidence_text = evidence or _evidence()
    return grounded_analysis_reward_v3_deepseek_high(
        completions or _completions(),
        [evidence_text] * 4,
        meeting_date=["2025-01-01"] * 4,
        save_path=str(save_path) if save_path is not None else None,
        tokenizer_path=str(_tokenizer_dir(tmp_path)),
        judge_tokenizer=_FixedJudgeTokenizer(),
        max_model_len=131_072,
        max_completion_tokens=_MAX_OUTPUT_TOKENS,
        timeout=5,
        max_retries=max_retries,
        backoff_seconds=0,
        client=client,
    )


def test_batch_request_is_fixed_four_candidate_high_effort_strict_schema() -> None:
    candidates = _candidate_texts()
    evidence = _evidence()
    request = build_deepseek_batch_request_v3(
        candidates=candidates,
        evidence=evidence,
        model=_MODEL,
        max_output_tokens=_MAX_OUTPUT_TOKENS,
    )

    assert request["model"] == _MODEL
    assert request["reasoning"] == {"effort": "high"}
    assert request["max_output_tokens"] == 16_384
    assert "temperature" not in request
    assert "top_p" not in request
    response_format = request["text"]["format"]
    assert response_format["type"] == "json_schema"
    assert response_format["strict"] is True
    schema = response_format["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == list(_SLOTS)
    assert set(schema["properties"]) == set(_SLOTS)
    for slot in _SLOTS:
        candidate_schema = schema["properties"][slot]
        assert candidate_schema["additionalProperties"] is False
        assert set(candidate_schema["required"]) == {*_RUBRICS, "violations"}

    request_input = json.loads(request["input"])
    assert request_input["evidence"] == evidence
    assert list(request_input["candidates"]) == list(_SLOTS)
    assert [request_input["candidates"][slot] for slot in _SLOTS] == candidates


def test_batch_request_rejects_any_candidate_count_other_than_four() -> None:
    with pytest.raises(ValueError, match="4|four"):
        build_deepseek_batch_request_v3(
            candidates=_candidate_texts()[:3],
            evidence=_evidence(),
            model=_MODEL,
            max_output_tokens=_MAX_OUTPUT_TOKENS,
        )


def test_group_response_maps_each_slot_back_to_input_order(tmp_path: Path) -> None:
    client = _FakeClient(_mapped_response)

    rewards = _run_reward(tmp_path, client)

    assert rewards == pytest.approx([1.0, 0.85, 0.80, 0.0])
    assert len(client.responses.calls) == 1
    call = client.responses.calls[0]
    assert call["reasoning"] == {"effort": "high"}
    assert call["max_output_tokens"] == 16_384


def test_invalid_visible_json_is_retried_then_recovers(tmp_path: Path) -> None:
    client = _FakeClient(_response("{invalid"), _mapped_response)

    rewards = _run_reward(tmp_path, client, max_retries=2)

    assert rewards == pytest.approx([1.0, 0.85, 0.80, 0.0])
    assert len(client.responses.calls) == 2


def test_duplicate_candidate_id_is_retried_then_recovers(tmp_path: Path) -> None:
    client = _FakeClient(
        _response(_duplicate_candidate_payload()),
        _mapped_response,
    )

    rewards = _run_reward(tmp_path, client, max_retries=2)

    assert rewards == pytest.approx([1.0, 0.85, 0.80, 0.0])
    assert len(client.responses.calls) == 2


def test_duplicate_candidate_id_exhaustion_fails_closed(tmp_path: Path) -> None:
    duplicate = _duplicate_candidate_payload()
    client = _FakeClient(_response(duplicate), _response(duplicate))

    with pytest.raises(JudgeInfrastructureError):
        _run_reward(tmp_path, client, max_retries=2)

    assert len(client.responses.calls) == 2


def test_incomplete_response_is_retried_then_recovers(tmp_path: Path) -> None:
    client = _FakeClient(
        _response(_group_payload(), status="incomplete"),
        _mapped_response,
    )

    rewards = _run_reward(tmp_path, client, max_retries=2)

    assert rewards == pytest.approx([1.0, 0.85, 0.80, 0.0])
    assert len(client.responses.calls) == 2


def test_missing_high_effort_reasoning_is_retried(tmp_path: Path) -> None:
    client = _FakeClient(
        _response(_group_payload(), hidden=""),
        _mapped_response,
    )

    rewards = _run_reward(tmp_path, client, max_retries=2)

    assert rewards == pytest.approx([1.0, 0.85, 0.80, 0.0])
    assert len(client.responses.calls) == 2


def test_rate_limit_is_retried_but_auth_failure_is_not(tmp_path: Path) -> None:
    retry_client = _FakeClient(_StatusError(429), _mapped_response)
    rewards = _run_reward(tmp_path, retry_client, max_retries=2)
    assert rewards == pytest.approx([1.0, 0.85, 0.80, 0.0])
    assert len(retry_client.responses.calls) == 2

    auth_client = _FakeClient(_StatusError(401), _mapped_response)
    with pytest.raises(JudgeInfrastructureError, match="status=401"):
        _run_reward(tmp_path, auth_client, max_retries=2)
    assert len(auth_client.responses.calls) == 1


def test_invalid_responses_exhaust_retries_and_fail_closed(tmp_path: Path) -> None:
    client = _FakeClient(
        _response("{invalid-one"),
        _response("{invalid-two"),
    )

    with pytest.raises(JudgeInfrastructureError):
        _run_reward(tmp_path, client, max_retries=2)

    assert len(client.responses.calls) == 2


def test_returned_model_mismatch_is_not_retried(tmp_path: Path) -> None:
    client = _FakeClient(
        _response(_group_payload(), model="unexpected-model"),
        _mapped_response,
    )

    with pytest.raises(JudgeInfrastructureError, match="model"):
        _run_reward(tmp_path, client, max_retries=2)

    assert len(client.responses.calls) == 1


def test_missing_returned_model_is_not_filled_from_request(tmp_path: Path) -> None:
    client = _FakeClient(_response(_group_payload(), model=""), _mapped_response)

    with pytest.raises(JudgeInfrastructureError, match="model|provenance"):
        _run_reward(tmp_path, client, max_retries=2)

    assert len(client.responses.calls) == 1


def test_missing_boundaries_are_sent_but_locally_forced_to_zero(
    tmp_path: Path,
) -> None:
    client = _FakeClient(_response(_group_payload()))
    reward_log = tmp_path / "run" / "reward.jsonl"
    completions = [[{"content": f"Plain answer {name}."}] for name in "ABCD"]

    rewards = _run_reward(
        tmp_path,
        client,
        completions=completions,
        save_path=reward_log,
    )

    assert rewards == [0.0, 0.0, 0.0, 0.0]
    assert len(client.responses.calls) == 1
    records = [json.loads(line) for line in reward_log.read_text().splitlines()]
    assert len(records) == 4
    assert all(record["contract"]["accepted_for_judge"] is False for record in records)
    assert all(
        record["contract"]["rejection_reason"] == "missing_think_boundary"
        for record in records
    )


def test_mixed_evidence_group_is_rejected_before_provider_call(tmp_path: Path) -> None:
    client = _FakeClient(_mapped_response)
    evidence = [_evidence("one"), _evidence("one"), _evidence("two"), _evidence("one")]

    with pytest.raises(ValueError, match="share exact evidence"):
        grounded_analysis_reward_v3_deepseek_high(
            _completions(),
            evidence,
            meeting_date=["2025-01-01"] * 4,
            judge_tokenizer=_FixedJudgeTokenizer(),
            max_completion_tokens=_MAX_OUTPUT_TOKENS,
            timeout=5,
            max_retries=2,
            backoff_seconds=0,
            client=client,
        )

    assert client.responses.calls == []


def test_live_reward_refuses_sdk_debug_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_LOG", "debug")

    with pytest.raises(ValueError, match="unsafe OPENAI_LOG"):
        grounded_analysis_reward_v3_deepseek_high(
            _completions(),
            [_evidence()] * 4,
            meeting_date=["2025-01-01"] * 4,
            save_path=str(tmp_path / "run" / "reward.jsonl"),
            judge_tokenizer=_FixedJudgeTokenizer(),
            max_completion_tokens=_MAX_OUTPUT_TOKENS,
            timeout=420,
            max_retries=2,
            backoff_seconds=2,
            client=None,
        )


def test_quote_not_found_is_audited_without_factual_penalty(tmp_path: Path) -> None:
    payload = _group_payload(
        [
            _evaluation(
                violations=[
                    _violation(
                        "answer",
                        "factual",
                        "major",
                        "THIS QUOTE DOES NOT OCCUR",
                    )
                ]
            ),
            _evaluation(),
            _evaluation(),
            _evaluation(),
        ]
    )
    reward_log = tmp_path / "run" / "reward.jsonl"

    rewards = _run_reward(
        tmp_path,
        _FakeClient(_response(payload)),
        save_path=reward_log,
    )

    record = json.loads(reward_log.read_text().splitlines()[0])
    assert rewards[0] == pytest.approx(1.0)
    assert record["penalties"]["applied"] == 0.0
    assert record["validated_violations"] == []
    assert record["invalid_violations"][0]["invalid_reason"] == "quote_not_found"


def test_reward_and_cache_logs_never_persist_raw_text_or_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_key = "SENSITIVE_API_KEY_VALUE"
    evidence = _evidence("SENSITIVE_EVIDENCE_BODY")
    completions = _completions()
    completions[0] = _completion("SENSITIVE_CANDIDATE_BODY 987654321.125")
    explanation = "SENSITIVE_PROVIDER_EXPLANATION"
    hidden = "SENSITIVE_HIDDEN_REASONING"
    payload = _group_payload(
        [
            _evaluation(
                violations=[
                    _violation(
                        "answer",
                        "format",
                        "minor",
                        "SENSITIVE_CANDIDATE_BODY 987654321.125",
                        explanation=explanation,
                    )
                ]
            ),
            _evaluation(),
            _evaluation(),
            _evaluation(),
        ]
    )
    client = _FakeClient(_response(payload, hidden=hidden))
    reward_log = tmp_path / "run" / "reward.jsonl"
    monkeypatch.setenv("DEEPSEEK_API_KEY", api_key)

    _run_reward(
        tmp_path,
        client,
        completions=completions,
        evidence=evidence,
        save_path=reward_log,
    )

    cache_files = list((reward_log.parent / "deepseek_reward_cache").rglob("*.json"))
    assert len(cache_files) == 1
    serialized = reward_log.read_text() + cache_files[0].read_text()
    for forbidden in (
        api_key,
        "SENSITIVE_EVIDENCE_BODY",
        "SENSITIVE_CANDIDATE_BODY",
        "987654321.125",
        explanation,
        hidden,
    ):
        assert forbidden not in serialized
    cache = json.loads(cache_files[0].read_text())
    assert cache["schema_version"] == "grounded-analysis-v3-deepseek-high-cache-v1"
    assert reward_log.stat().st_mode & 0o777 == 0o600
    assert cache_files[0].stat().st_mode & 0o777 == 0o600


def test_persistent_cache_recovers_without_a_second_provider_call(
    tmp_path: Path,
) -> None:
    reward_log = tmp_path / "run" / "reward.jsonl"
    first_client = _FakeClient(_mapped_response)
    first = _run_reward(tmp_path, first_client, save_path=reward_log)
    assert len(first_client.responses.calls) == 1

    cached_client = _FakeClient()
    recovered = _run_reward(tmp_path, cached_client, save_path=reward_log)

    assert recovered == first
    assert cached_client.responses.calls == []
    records = [json.loads(line) for line in reward_log.read_text().splitlines()]
    assert len(records) == 8
    assert all(record["cache_hit"] is False for record in records[:4])
    assert all(record["cache_hit"] is True for record in records[4:])


def test_tampered_persistent_cache_fails_closed_without_provider_fallback(
    tmp_path: Path,
) -> None:
    reward_log = tmp_path / "run" / "reward.jsonl"
    first_client = _FakeClient(_mapped_response)
    _run_reward(tmp_path, first_client, save_path=reward_log)
    cache_path = next((reward_log.parent / "deepseek_reward_cache").rglob("*.json"))
    cache = json.loads(cache_path.read_text())
    cache["binding"]["evidence_sha256"] = "0" * 64
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    fallback_client = _FakeClient(_mapped_response)

    with pytest.raises(JudgeInfrastructureError, match="cache|binding"):
        _run_reward(tmp_path, fallback_client, save_path=reward_log)

    assert fallback_client.responses.calls == []


def test_cache_rubric_tamper_fails_payload_hash_check(tmp_path: Path) -> None:
    reward_log = tmp_path / "run" / "reward.jsonl"
    _run_reward(tmp_path, _FakeClient(_mapped_response), save_path=reward_log)
    cache_path = next((reward_log.parent / "deepseek_reward_cache").rglob("*.json"))
    cache = json.loads(cache_path.read_text())
    cache["evaluations"][0]["rubric"]["data_fidelity"] = 0
    cache_path.write_text(json.dumps(cache), encoding="utf-8")

    with pytest.raises(JudgeInfrastructureError, match="hash"):
        _run_reward(tmp_path, _FakeClient(_mapped_response), save_path=reward_log)


def test_cache_penalty_tamper_is_recomputed_even_with_updated_hash(
    tmp_path: Path,
) -> None:
    reward_log = tmp_path / "run" / "reward.jsonl"
    _run_reward(tmp_path, _FakeClient(_mapped_response), save_path=reward_log)
    cache_path = next((reward_log.parent / "deepseek_reward_cache").rglob("*.json"))
    cache = json.loads(cache_path.read_text())
    cache["evaluations"][2]["penalties"]["applied"] = 0.0
    core = {
        key: cache[key]
        for key in ("schema_version", "binding", "provider", "evaluations")
    }
    cache["payload_sha256"] = hashlib.sha256(
        json.dumps(
            core,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    cache_path.write_text(json.dumps(cache), encoding="utf-8")

    with pytest.raises(JudgeInfrastructureError, match="penalty"):
        _run_reward(tmp_path, _FakeClient(_mapped_response), save_path=reward_log)


def test_deepseek_transport_preserves_v3_reward_math(tmp_path: Path) -> None:
    deepseek_client = _FakeClient(_mapped_response)
    deepseek_rewards = _run_reward(tmp_path, deepseek_client)

    def local_judge(*, candidate: str, **_kwargs: Any) -> tuple[dict[str, Any], str, int]:
        return _evaluation_for_candidate(candidate), "{}", 1

    tokenizer_path = _tokenizer_dir(tmp_path)
    with patch(
        "open_r1.trainer.rewards.reward_funcs.analysis_reward_v3._judge_one_v3",
        side_effect=local_judge,
    ):
        v3_rewards = grounded_analysis_reward_v3(
            _completions(),
            [_evidence()] * 4,
            meeting_date=["2025-01-01"] * 4,
            tokenizer_path=str(tokenizer_path),
            judge_tokenizer=_FixedJudgeTokenizer(),
            max_model_len=131_072,
            max_completion_tokens=_MAX_OUTPUT_TOKENS,
            timeout=5,
            max_retries=1,
            backoff_seconds=0,
        )

    assert deepseek_rewards == pytest.approx(v3_rewards)
