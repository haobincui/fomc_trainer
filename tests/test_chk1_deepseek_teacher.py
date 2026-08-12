from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import jobs.retrain_v2.chk1.deepseek_teacher as deepseek_teacher_module
from jobs.retrain_v2.chk1.deepseek_teacher import (
    DeepSeekTeacherConfig,
    DeepSeekTeacherResponse,
    MockDeepSeekTeacherBackend,
    OpenAIDeepSeekBackend,
    TeacherResponseRejectedError,
    candidate_to_sft_response,
    run_deepseek_chk1_generation,
)
def _model_repo(root: Path) -> None:
    model = root / "models" / "DeepSeek-R1-Distill-Llama-8B"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"chk0")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")


def _fact_card() -> dict:
    return {
        "schema_version": "chk1-fact-card-v1",
        "sample_id": "sample-1",
        "canonical_key": {
            "meeting_date": "2020-01-01",
            "atomic_topic": "GDP Growth",
        },
        "meeting_date": "2020-01-01",
        "atomic_topic": "GDP Growth",
        "cutoff_ts": "2020-01-01T23:59:59Z",
        "evidence": [
            {
                "evidence_id": "E1",
                "source_id": "SERIES",
                "value": "1.0",
                "units": "Percent",
                "metric": "growth",
                "fact_kind": "latest",
                "observation_date": "2019-12-01",
                "formula": None,
                "operand_evidence_ids": [],
                "release_ts": "2019-12-15T12:00:00Z",
                "availability_upper_bound_ts": None,
                "availability_basis": "actual_release_ts",
                "requested_vintage_date": "2019-12-31",
                "cutoff_ts": "2020-01-01T23:59:59Z",
                "source_sha256": "a" * 64,
            }
        ],
    }


def _teacher_response() -> DeepSeekTeacherResponse:
    return DeepSeekTeacherResponse(
        analysis="The supplied observation was 1.0 percent.",
        answer="The available growth observation was 1.0 percent.",
        evidence_ids=("E1",),
        response_id="response-1",
        returned_model="deepseek-v4-pro",
        system_fingerprint="fp-test",
        finish_reason="stop",
        created=1,
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )


def test_teacher_contract_has_no_secret_and_no_qwen_role() -> None:
    contract = DeepSeekTeacherConfig().contract()
    serialized = json.dumps(contract, sort_keys=True)
    assert contract["schema_version"] == "chk1-deepseek-v4-thinking-teacher-v3"
    assert contract["model"] == "deepseek-v4-pro"
    assert contract["reasoning_effort"] == "high"
    assert contract["thinking"] == {"type": "enabled"}
    assert contract["analysis_source"] == "message.reasoning_content"
    assert contract["answer_source"] == "message.content.answer"
    assert contract["acceptance_contract"] == {
        "finish_reason": "stop",
        "content_required_keys": ["answer", "evidence_ids"],
        "content_additional_properties": False,
        "reasoning_non_empty": True,
        "answer_format": "plain_text_without_inline_evidence_ids",
        "evidence_ids_separate_non_empty_unique_list": True,
        "evidence_ids_must_reference_fact_card": True,
    }
    assert "api_key" not in contract
    assert contract["api_key_env"] == "DEEPSEEK_API_KEY"
    assert "qwen" not in serialized.casefold()


def test_deepseek_analysis_and_answer_become_one_sft_completion(
    tmp_path: Path,
) -> None:
    _model_repo(tmp_path)
    teacher = MockDeepSeekTeacherBackend([_teacher_response()])
    result = run_deepseek_chk1_generation(
        prompt="teacher prompt",
        repo_root=tmp_path,
        fact_card=_fact_card(),
        same_sample_minutes="unrelated minutes reference",
        environment={},
        teacher_backend=teacher,
        generation_provenance_sha256="b" * 64,
        mock=True,
    )
    assert result["status"] == "accepted"
    assert result["teacher_output"] == {
        "analysis": "The supplied observation was 1.0 percent.",
        "answer": "The available growth observation was 1.0 percent.",
        "evidence_ids": ["E1"],
    }
    assert result["selected_response"] == (
        "The supplied observation was 1.0 percent.\n</think>\n"
        "The available growth observation was 1.0 percent."
    )
    assert "verifier" not in result
    assert "critic" not in result
    assert len(teacher.requests) == 1


def test_deepseek_content_is_not_rejected_by_legacy_deterministic_rules(
    tmp_path: Path,
) -> None:
    _model_repo(tmp_path)
    duplicated = "The supplied evidence was reviewed before preparing the assessment."
    response = DeepSeekTeacherResponse(
        analysis=(
            "The prompt mentions 2021 and 9.9 because those tokens require review. "
            f"{duplicated}"
        ),
        answer=duplicated,
        evidence_ids=("E1",),
        response_id="response-no-verifier",
        returned_model="deepseek-v4-pro",
        system_fingerprint="fp-test",
        finish_reason="stop",
        created=1,
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )
    teacher = MockDeepSeekTeacherBackend([response])
    result = run_deepseek_chk1_generation(
        prompt="teacher prompt",
        repo_root=tmp_path,
        fact_card=_fact_card(),
        same_sample_minutes=f"Prefix {duplicated} suffix",
        environment={},
        teacher_backend=teacher,
        generation_provenance_sha256="b" * 64,
        mock=True,
    )

    assert result["status"] == "accepted"
    assert result["selected_candidate"]["reasoning"] == response.analysis
    assert "verifier" not in result
    assert "critic" not in result


def test_prior_accepted_teacher_cache_is_reused_after_policy_revision(
    tmp_path: Path,
) -> None:
    _model_repo(tmp_path)
    teacher = MockDeepSeekTeacherBackend([_teacher_response()])
    cache = tmp_path / "teacher-cache"
    first = run_deepseek_chk1_generation(
        prompt="teacher prompt",
        repo_root=tmp_path,
        fact_card=_fact_card(),
        same_sample_minutes="unrelated minutes reference",
        cache_dir=cache,
        environment={},
        teacher_backend=teacher,
        generation_provenance_sha256="b" * 64,
        mock=True,
    )
    recovered = run_deepseek_chk1_generation(
        prompt="teacher prompt",
        repo_root=tmp_path,
        fact_card=_fact_card(),
        same_sample_minutes="unrelated minutes reference",
        cache_dir=cache,
        environment={},
        teacher_backend=teacher,
        generation_provenance_sha256="c" * 64,
        mock=True,
    )

    assert recovered["teacher_output"] == first["teacher_output"]
    assert recovered["reused_from_cache_key"] == first["cache_key"]
    assert len(teacher.requests) == 1


def test_v1_historical_cache_is_left_untouched(tmp_path: Path) -> None:
    _model_repo(tmp_path)
    cache = tmp_path / "teacher-cache"
    cache.mkdir()
    legacy_contract = DeepSeekTeacherConfig().contract()
    legacy_contract["schema_version"] = "chk1-deepseek-v4-thinking-teacher-v1"
    legacy_path = cache / "legacy-v1.json"
    legacy_path.write_text(
        json.dumps(
            {
                "schema_version": "chk1-deepseek-teacher-cache-v3",
                "status": "accepted",
                "prompt_sha256": deepseek_teacher_module.sha256_text(
                    "teacher prompt"
                ),
                "teacher_contract": legacy_contract,
            },
            sort_keys=True,
        )
        + "\n"
    )
    legacy_bytes = legacy_path.read_bytes()

    result = run_deepseek_chk1_generation(
        prompt="teacher prompt",
        repo_root=tmp_path,
        fact_card=_fact_card(),
        cache_dir=cache,
        environment={},
        teacher_backend=MockDeepSeekTeacherBackend([_teacher_response()]),
        generation_provenance_sha256="b" * 64,
        mock=True,
    )

    assert result["status"] == "accepted"
    assert legacy_path.read_bytes() == legacy_bytes
    assert len(list(cache.rglob("*.json"))) == 2


def test_missing_reasoning_content_is_rejected() -> None:
    response = _teacher_response()
    broken = DeepSeekTeacherResponse(
        analysis="",
        answer=response.answer,
        evidence_ids=response.evidence_ids,
        response_id=response.response_id,
        returned_model=response.returned_model,
        system_fingerprint=response.system_fingerprint,
        finish_reason=response.finish_reason,
        created=response.created,
        usage=response.usage,
    )
    backend = MockDeepSeekTeacherBackend([broken])
    with pytest.raises(TeacherResponseRejectedError) as captured:
        backend.generate(
            config=DeepSeekTeacherConfig(),
            system_prompt="system",
            user_prompt="user",
            environment={},
        )
    assert captured.value.payload["rejection_error_codes"] == ["reasoning_empty"]


@pytest.mark.parametrize(
    ("response", "error_code"),
    [
        (
            replace(_teacher_response(), finish_reason="length"),
            "finish_reason_not_stop",
        ),
        (replace(_teacher_response(), answer=""), "answer_empty"),
        (replace(_teacher_response(), evidence_ids=()), "evidence_ids_empty"),
        (
            replace(
                _teacher_response(),
                answer="Growth was 1.0 percent (ev-deadbeef).",
            ),
            "answer_inline_evidence_id",
        ),
        (
            replace(_teacher_response(), answer="The JSON schema contains an answer."),
            "answer_schema_meta",
        ),
        (
            replace(_teacher_response(), answer="<answer>Growth was stable.</answer>"),
            "answer_control_markup",
        ),
        (
            replace(_teacher_response(), answer="**Growth was stable.**"),
            "answer_markdown",
        ),
        (
            replace(_teacher_response(), evidence_ids=("E1", "E1")),
            "evidence_ids_duplicate",
        ),
    ],
)
def test_structural_pollution_is_rejected(
    response: DeepSeekTeacherResponse, error_code: str
) -> None:
    backend = MockDeepSeekTeacherBackend([response])
    with pytest.raises(TeacherResponseRejectedError) as captured:
        backend.generate(
            config=DeepSeekTeacherConfig(),
            system_prompt="system",
            user_prompt="user",
            environment={},
        )
    assert captured.value.payload["rejection_error_codes"] == [error_code]


@pytest.mark.parametrize(
    ("content", "error_code"),
    [
        ("not JSON", "content_malformed_json"),
        (
            json.dumps(
                {
                    "content": {
                        "answer": "Growth was stable.",
                        "evidence_ids": ["E1"],
                    }
                }
            ),
            "content_schema_invalid",
        ),
        (
            json.dumps(
                {
                    "answer": "Growth was stable.",
                    "evidence_ids": ["E1"],
                    "schema": "wrapper",
                }
            ),
            "content_schema_invalid",
        ),
        (
            '{"answer":"first","answer":"second","evidence_ids":["E1"]}',
            "content_duplicate_key",
        ),
    ],
)
def test_raw_content_never_falls_back_to_answer(
    content: str, error_code: str
) -> None:
    response = replace(_teacher_response(), raw_content=content)
    backend = MockDeepSeekTeacherBackend([response])
    with pytest.raises(TeacherResponseRejectedError) as captured:
        backend.generate(
            config=DeepSeekTeacherConfig(),
            system_prompt="system",
            user_prompt="user",
            environment={},
        )
    attempt = captured.value.payload["attempts"][0]
    assert captured.value.payload["rejection_error_codes"] == [error_code]
    assert attempt["provider_raw"]["content"] == content


def test_contract_failure_retries_then_accepts(tmp_path: Path) -> None:
    _model_repo(tmp_path)
    cache = tmp_path / "teacher-cache"
    backend = MockDeepSeekTeacherBackend(
        [replace(_teacher_response(), finish_reason="length"), _teacher_response()]
    )

    result = run_deepseek_chk1_generation(
        prompt="teacher prompt",
        repo_root=tmp_path,
        fact_card=_fact_card(),
        cache_dir=cache,
        environment={},
        teacher_backend=backend,
        generation_provenance_sha256="b" * 64,
        mock=True,
    )

    assert result["status"] == "accepted"
    assert len(backend.requests) == 2
    assert [item["status"] for item in result["attempts"]] == [
        "rejected",
        "accepted",
    ]
    assert result["attempts"][0]["error_code"] == "finish_reason_not_stop"
    assert result["attempts"][0]["provider_raw"]["content"]

    cached_backend = MockDeepSeekTeacherBackend([])
    cached = run_deepseek_chk1_generation(
        prompt="teacher prompt",
        repo_root=tmp_path,
        fact_card=_fact_card(),
        cache_dir=cache,
        environment={},
        teacher_backend=cached_backend,
        generation_provenance_sha256="b" * 64,
        mock=True,
    )
    assert cached["cache_hit"] is True
    assert cached["attempts"] == result["attempts"]
    assert not cached_backend.requests


def test_unknown_fact_card_evidence_id_retries_then_accepts(tmp_path: Path) -> None:
    _model_repo(tmp_path)
    backend = MockDeepSeekTeacherBackend(
        [
            replace(_teacher_response(), evidence_ids=("NOT-IN-FACT-CARD",)),
            _teacher_response(),
        ]
    )

    result = run_deepseek_chk1_generation(
        prompt="teacher prompt",
        repo_root=tmp_path,
        fact_card=_fact_card(),
        environment={},
        teacher_backend=backend,
        generation_provenance_sha256="b" * 64,
        mock=True,
    )

    assert result["status"] == "accepted"
    assert len(backend.requests) == 2
    assert result["attempts"][0]["error_code"] == "evidence_id_not_in_fact_card"


def test_answer_may_not_inline_a_fact_card_evidence_id(tmp_path: Path) -> None:
    _model_repo(tmp_path)
    polluted = replace(
        _teacher_response(),
        answer="The available growth observation was 1.0 percent according to E1.",
    )
    backend = MockDeepSeekTeacherBackend([polluted for _ in range(4)])

    with pytest.raises(TeacherResponseRejectedError) as captured:
        run_deepseek_chk1_generation(
            prompt="teacher prompt",
            repo_root=tmp_path,
            fact_card=_fact_card(),
            environment={},
            teacher_backend=backend,
            generation_provenance_sha256="b" * 64,
            mock=True,
        )

    assert captured.value.payload["rejection_error_codes"] == [
        "answer_inline_evidence_id"
    ]


@pytest.mark.parametrize("binding", ["prompt_sha256", "teacher_contract"])
def test_cache_load_rejects_rewritten_request_bindings(
    tmp_path: Path, binding: str
) -> None:
    _model_repo(tmp_path)
    cache = tmp_path / "teacher-cache"
    run_deepseek_chk1_generation(
        prompt="teacher prompt",
        repo_root=tmp_path,
        fact_card=_fact_card(),
        cache_dir=cache,
        environment={},
        teacher_backend=MockDeepSeekTeacherBackend([_teacher_response()]),
        generation_provenance_sha256="b" * 64,
        mock=True,
    )
    path = next(cache.rglob("*.json"))
    path.chmod(0o600)
    payload = json.loads(path.read_text())
    if binding == "prompt_sha256":
        payload[binding] = "0" * 64
    else:
        payload[binding] = {**payload[binding], "revision": "tampered"}
    payload = deepseek_teacher_module._signed_cache_payload(payload)
    path.write_text(
        deepseek_teacher_module.canonical_json(payload) + "\n", encoding="utf-8"
    )

    with pytest.raises(
        deepseek_teacher_module.CacheIntegrityError,
        match="cache binding mismatch",
    ):
        run_deepseek_chk1_generation(
            prompt="teacher prompt",
            repo_root=tmp_path,
            fact_card=_fact_card(),
            cache_dir=cache,
            environment={},
            teacher_backend=MockDeepSeekTeacherBackend([]),
            generation_provenance_sha256="b" * 64,
            mock=True,
        )


def test_contract_retry_exhaustion_writes_rejected_audit_cache(
    tmp_path: Path,
) -> None:
    _model_repo(tmp_path)
    cache = tmp_path / "teacher-cache"
    backend = MockDeepSeekTeacherBackend(
        [replace(_teacher_response(), finish_reason="length") for _ in range(4)]
    )

    with pytest.raises(TeacherResponseRejectedError) as captured:
        run_deepseek_chk1_generation(
            prompt="teacher prompt",
            repo_root=tmp_path,
            fact_card=_fact_card(),
            cache_dir=cache,
            environment={},
            teacher_backend=backend,
            generation_provenance_sha256="b" * 64,
            mock=True,
        )

    assert len(backend.requests) == 4
    payload = captured.value.payload
    assert payload["status"] == "rejected"
    assert payload["rejection_error_codes"] == ["finish_reason_not_stop"]
    assert len(payload["attempts"]) == 4
    assert "selected_candidate" not in payload
    paths = list(cache.rglob("*.json"))
    assert len(paths) == 1
    stored = json.loads(paths[0].read_text())
    assert stored["status"] == "rejected"
    assert stored["attempts"] == payload["attempts"]
    assert "cache_hit" not in stored

    cached_backend = MockDeepSeekTeacherBackend([])
    with pytest.raises(TeacherResponseRejectedError) as cached:
        run_deepseek_chk1_generation(
            prompt="teacher prompt",
            repo_root=tmp_path,
            fact_card=_fact_card(),
            cache_dir=cache,
            environment={},
            teacher_backend=cached_backend,
            generation_provenance_sha256="b" * 64,
            mock=True,
        )
    assert cached.value.payload["cache_hit"] is True
    assert not cached_backend.requests


def test_accepted_provider_raw_is_cached_losslessly(tmp_path: Path) -> None:
    _model_repo(tmp_path)
    raw_reasoning = "  The supplied observation was reviewed.\n"
    raw_content = (
        ' \n{ "answer": "Growth was 1.0 percent.", '
        '"evidence_ids": ["E1"] }\n'
    )
    response = replace(
        _teacher_response(),
        analysis=raw_reasoning,
        answer="Growth was 1.0 percent.",
        raw_content=raw_content,
    )
    result = run_deepseek_chk1_generation(
        prompt="teacher prompt",
        repo_root=tmp_path,
        fact_card=_fact_card(),
        cache_dir=tmp_path / "teacher-cache",
        environment={},
        teacher_backend=MockDeepSeekTeacherBackend([response]),
        generation_provenance_sha256="b" * 64,
        mock=True,
    )

    assert result["provider_raw"] == {
        "reasoning_content": raw_reasoning,
        "content": raw_content,
    }
    assert result["teacher_output"]["analysis"] == raw_reasoning.strip()
    stored = json.loads(next((tmp_path / "teacher-cache").rglob("*.json")).read_text())
    assert stored["provider_raw"] == result["provider_raw"]
    stored_path = next((tmp_path / "teacher-cache").rglob("*.json"))
    assert stored_path.stat().st_mode & 0o777 == 0o400
    assert stored["cache_payload_sha256"] == deepseek_teacher_module.sha256_text(
        deepseek_teacher_module.canonical_json(
            {key: value for key, value in stored.items() if key != "cache_payload_sha256"}
        )
    )


def test_openai_backend_retries_truncated_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = SimpleNamespace(
        id="invalid",
        model="deepseek-v4-pro",
        system_fingerprint="fp",
        created=1,
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="length",
                message=SimpleNamespace(
                    reasoning_content="partial reasoning",
                    content='{"answer":"partial","evidence_ids":["E1"]}',
                ),
            )
        ],
    )
    valid = SimpleNamespace(
        id="valid",
        model="deepseek-v4-pro",
        system_fingerprint="fp",
        created=2,
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    reasoning_content="complete reasoning",
                    content=(
                        '{"answer":"Growth was stable.",'
                        '"evidence_ids":["E1"]}'
                    ),
                ),
            )
        ],
    )
    completions = [invalid, valid]
    calls: list[dict] = []

    class FakeOpenAI:
        def __init__(self, **_kwargs) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        @staticmethod
        def _create(**kwargs):
            calls.append(kwargs)
            return completions.pop(0)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setattr(deepseek_teacher_module.time, "sleep", lambda _delay: None)

    observed = OpenAIDeepSeekBackend().generate(
        config=DeepSeekTeacherConfig(),
        system_prompt="system",
        user_prompt="user",
        environment={"DEEPSEEK_API_KEY": "test-only"},
    )

    assert observed.response_id == "valid"
    assert observed.finish_reason == "stop"
    assert len(observed.rejected_attempts) == 1
    assert observed.rejected_attempts[0]["error_code"] == "finish_reason_not_stop"
    assert len(calls) == 2


def test_candidate_response_uses_analysis_answer_boundary() -> None:
    response = candidate_to_sft_response(
        {
            "reasoning": "analysis",
            "final_analysis": "answer",
            "evidence_ids": ["E1"],
        }
    )
    assert response == "analysis\n</think>\nanswer"
