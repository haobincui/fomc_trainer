from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobs.retrain_v2.chk1.local_models import (
    CACHE_SCHEMA_VERSION,
    CHK0_CRITIC_JSON_SCHEMA,
    CacheIntegrityError,
    CriticRejectedError,
    GpuSafetyError,
    LocalModelPathError,
    MockGenerationBackend,
    ModelOutputContractError,
    NetworkSafetyError,
    assert_gpu_safe,
    assert_no_deepseek_api_access,
    build_cache_key,
    build_generation_plan,
    build_local_configs,
    inspect_gpu_safety,
    offline_environment,
    parse_critic_verdict,
    parse_teacher_candidate,
    qwen_to_deepseek_response,
    resolve_local_model_path,
    run_local_chk1_generation,
    store_cache_entry,
)


GPU_OUTPUT = """0, GPU-aaa, NVIDIA A30, 24576
1, GPU-bbb, NVIDIA A30, 24576
"""
SOURCE_HASH = "a" * 64


def _model_repo(root: Path) -> None:
    for relative in (
        "models/Qwen3.5-9B",
        "models/DeepSeek-R1-Distill-Llama-8B",
    ):
        model = root / relative
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}\n", encoding="utf-8")
        (model / "model.safetensors").write_bytes(relative.encode("utf-8"))
        (model / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
        (model / "tokenizer.json").write_text(
            json.dumps({"model": relative}) + "\n", encoding="utf-8"
        )


def _fact_card() -> dict:
    return {
        "sample_id": "2024-01-31::Unemployment-Rate",
        "meeting_date": "2024-01-31",
        "atomic_topic": "Unemployment Rate",
        "cutoff_ts": "2024-01-30T23:59:59Z",
        "evidence": [
            {
                "evidence_id": "E1",
                "value": "4.0",
                "unit": "Percent",
                "observation_date": "2023-12-01",
                "availability_ts": "2024-01-05T13:30:00Z",
                "source_sha256": SOURCE_HASH,
            },
            {
                "evidence_id": "E2",
                "value": "3.7",
                "unit": "Percent",
                "observation_date": "2023-11-01",
                "availability_ts": "2023-12-08T13:30:00Z",
                "source_sha256": SOURCE_HASH,
            },
        ],
    }


def _teacher_payload(
    *,
    reasoning: str,
    final_analysis: str,
    evidence_ids: list[str],
) -> str:
    return json.dumps(
        {
            "reasoning": reasoning,
            "final_analysis": final_analysis,
            "evidence_ids": evidence_ids,
        }
    )


def _candidate_one() -> str:
    return _teacher_payload(
        reasoning="The latest reading was 4.0 percent.",
        final_analysis="The available measure was elevated.",
        evidence_ids=["E1"],
    )


def _candidate_two() -> str:
    return _teacher_payload(
        reasoning="The latest reading was 4.0 percent, after 3.7 percent.",
        final_analysis="The two available observations indicate an increase.",
        evidence_ids=["E1", "E2"],
    )


def _critic_payload(
    *,
    grounded: bool = True,
    unsupported_claims: list[str] | None = None,
    style_score: int = 4,
    reasoning_consistency: bool = True,
) -> str:
    return json.dumps(
        {
            "grounded": grounded,
            "unsupported_claims": (
                [] if unsupported_claims is None else unsupported_claims
            ),
            "style_score": style_score,
            "reasoning_consistency": reasoning_consistency,
        }
    )


def test_gpu_preflight_requires_two_a30s_and_no_external_compute_processes() -> None:
    commands: list[tuple[str, ...]] = []

    def safe_runner(command):
        commands.append(tuple(command))
        return GPU_OUTPUT if "--query-gpu" in command[1] else ""

    report = assert_gpu_safe(command_runner=safe_runner)
    assert report.safe is True
    assert [item.index for item in report.devices] == [0, 1]
    assert all(command[0] == "nvidia-smi" for command in commands)
    assert not any(
        "kill" in part or "terminate" in part
        for command in commands
        for part in command
    )

    def occupied_runner(command):
        if "--query-gpu" in command[1]:
            return GPU_OUTPUT
        return "GPU-aaa, 9912, python, 2048\n"

    report = inspect_gpu_safety(command_runner=occupied_runner)
    assert report.safe is False
    assert report.external_processes[0].pid == 9912
    with pytest.raises(GpuSafetyError, match="external GPU compute"):
        assert_gpu_safe(command_runner=occupied_runner)


def test_gpu_preflight_fails_closed_on_probe_error_or_wrong_inventory() -> None:
    def one_gpu(command):
        return "0, GPU-aaa, NVIDIA A30, 24576\n" if "--query-gpu" in command[1] else ""

    with pytest.raises(GpuSafetyError, match="expected exactly 2 GPUs"):
        assert_gpu_safe(command_runner=one_gpu)

    def broken(_command):
        raise OSError("nvidia-smi unavailable")

    report = inspect_gpu_safety(command_runner=broken)
    assert report.safe is False
    assert len(report.errors) >= 2


def test_remote_credentials_and_deepseek_hosts_are_blocked() -> None:
    with pytest.raises(NetworkSafetyError, match="credentials"):
        assert_no_deepseek_api_access(environment={"DEEPSEEK_API_KEY": "secret"})
    with pytest.raises(NetworkSafetyError, match="credentials"):
        assert_no_deepseek_api_access(environment={"OPENAI_API_KEY": "secret"})
    with pytest.raises(NetworkSafetyError, match="remote"):
        assert_no_deepseek_api_access(
            endpoint="https://api.deepseek.com/v1", environment={}
        )
    assert_no_deepseek_api_access(endpoint="http://127.0.0.1:8000/v1", environment={})

    sanitized = offline_environment(
        {
            "DEEPSEEK_API_KEY": "secret",
            "OPENAI_BASE_URL": "https://api.deepseek.com",
            "KEEP_ME": "yes",
        }
    )
    assert "DEEPSEEK_API_KEY" not in sanitized
    assert "OPENAI_BASE_URL" not in sanitized
    assert sanitized["TRANSFORMERS_OFFLINE"] == "1"
    assert sanitized["KEEP_ME"] == "yes"


def test_only_the_two_pinned_local_model_paths_are_accepted(tmp_path: Path) -> None:
    _model_repo(tmp_path)
    resolved = resolve_local_model_path(
        repo_root=tmp_path,
        model_path="models/Qwen3.5-9B",
        expected_relative_path="models/Qwen3.5-9B",
    )
    assert resolved == (tmp_path / "models/Qwen3.5-9B").resolve()

    with pytest.raises(LocalModelPathError, match="remote model"):
        resolve_local_model_path(
            repo_root=tmp_path,
            model_path="https://huggingface.co/Qwen/Qwen3.5-9B",
            expected_relative_path="models/Qwen3.5-9B",
        )
    other = tmp_path / "models/other"
    other.mkdir()
    with pytest.raises(LocalModelPathError, match="must be"):
        resolve_local_model_path(
            repo_root=tmp_path,
            model_path=other,
            expected_relative_path="models/Qwen3.5-9B",
        )


def test_qwen_contract_is_pinned_to_two_seeds_4bit_thinking_and_sampling(
    tmp_path: Path,
) -> None:
    _model_repo(tmp_path)
    plan = build_generation_plan(
        prompt="evidence",
        repo_root=tmp_path,
        fact_card=_fact_card(),
    )
    assert plan["status"] == "dry_run"
    assert plan["qwen"]["seeds"] == [42, 43]
    assert plan["qwen"]["quantization"] == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
    }
    assert plan["qwen"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert plan["qwen"]["generation"] == {
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_new_tokens": 1536,
    }
    assert plan["critic"]["response_schema"]["strict"] is True
    assert plan["critic"]["response_schema"]["schema"]["additionalProperties"] is False
    assert CHK0_CRITIC_JSON_SCHEMA["schema"]["required"] == [
        "grounded",
        "unsupported_claims",
        "style_score",
        "reasoning_consistency",
    ]
    assert plan["verification"]["maximum_repairs"] == 1
    assert plan["verification"]["critic_acceptance"]["minimum_style_score"] == 4


def test_qwen_tokens_are_cleaned_and_converted_to_deepseek_completion() -> None:
    raw = (
        "<|im_start|>assistant\n<think>Check the table carefully.</think>\n"
        "The indicator rose modestly.<|im_end|>"
    )
    converted = qwen_to_deepseek_response(raw)
    assert converted == (
        "Check the table carefully.\n</think>\nThe indicator rose modestly."
    )
    assert converted.count("</think>") == 1
    assert "<|" not in converted

    channel = (
        "<|channel|>analysis<|message|>Compare 1.0 with 2.0."
        "<|channel|>final<|message|>The value increased.<|end|>"
    )
    assert qwen_to_deepseek_response(channel) == (
        "Compare 1.0 with 2.0.\n</think>\nThe value increased."
    )
    with pytest.raises(ModelOutputContractError, match="thinking boundary"):
        qwen_to_deepseek_response("answer without thinking")

    strict = parse_teacher_candidate(
        f"<think>native model thought</think>\n{_candidate_two()}<|im_end|>"
    )
    assert set(strict) == {"reasoning", "final_analysis", "evidence_ids"}
    assert strict["evidence_ids"] == ["E1", "E2"]
    with pytest.raises(ModelOutputContractError, match="keys must be exactly"):
        parse_teacher_candidate(json.dumps({**strict, "extra": "forbidden"}))


def test_chk0_critic_accepts_only_exact_json_contract() -> None:
    verdict = parse_critic_verdict(f"reasoning\n</think>\n{_critic_payload()}")
    assert verdict.accepted is True
    assert verdict.to_dict() == {
        "grounded": True,
        "unsupported_claims": [],
        "style_score": 4,
        "reasoning_consistency": True,
    }

    extra = json.loads(_critic_payload())
    extra["comment"] = "not allowed"
    with pytest.raises(ModelOutputContractError, match="critic_schema_keys"):
        parse_critic_verdict(json.dumps(extra))
    with pytest.raises(ModelOutputContractError, match="Markdown"):
        parse_critic_verdict(f"```json\n{_critic_payload()}\n```")

    unsupported = parse_critic_verdict(
        _critic_payload(unsupported_claims=["invented number"])
    )
    assert unsupported.accepted is False
    assert parse_critic_verdict(_critic_payload(style_score=3)).accepted is False


def test_cache_key_is_deterministic_and_sensitive_to_prompt(tmp_path: Path) -> None:
    _model_repo(tmp_path)
    qwen, critic = build_local_configs(repo_root=tmp_path)
    first = build_cache_key(
        prompt="same prompt",
        qwen_config=qwen,
        critic_config=critic,
        repo_root=tmp_path,
        fact_card=_fact_card(),
    )
    second = build_cache_key(
        prompt="same prompt",
        qwen_config=qwen,
        critic_config=critic,
        repo_root=tmp_path,
        fact_card=_fact_card(),
    )
    changed = build_cache_key(
        prompt="changed prompt",
        qwen_config=qwen,
        critic_config=critic,
        repo_root=tmp_path,
        fact_card=_fact_card(),
    )
    minutes_changed = build_cache_key(
        prompt="same prompt",
        qwen_config=qwen,
        critic_config=critic,
        repo_root=tmp_path,
        fact_card=_fact_card(),
        same_sample_minutes="a separately held leakage reference",
    )
    assert first == second
    assert first != changed
    assert first != minutes_changed
    assert len(first) == 64

    qwen_weights = tmp_path / "models" / "Qwen3.5-9B" / "model.safetensors"
    qwen_weights.write_bytes(b"changed-weights")
    weight_changed = build_cache_key(
        prompt="same prompt",
        qwen_config=qwen,
        critic_config=critic,
        repo_root=tmp_path,
        fact_card=_fact_card(),
    )
    qwen_tokenizer = tmp_path / "models" / "Qwen3.5-9B" / "tokenizer.json"
    qwen_tokenizer.write_text('{"changed":true}\n', encoding="utf-8")
    tokenizer_changed = build_cache_key(
        prompt="same prompt",
        qwen_config=qwen,
        critic_config=critic,
        repo_root=tmp_path,
        fact_card=_fact_card(),
    )
    assert weight_changed != first
    assert tokenizer_changed != weight_changed


def test_dry_run_and_mock_execution_never_load_models_and_reuse_cache(
    tmp_path: Path,
) -> None:
    _model_repo(tmp_path)

    def forbidden_gpu_probe(_command):
        raise AssertionError("dry-run/mock must not probe or initialize GPUs")

    dry = run_local_chk1_generation(
        prompt="reference-free evidence",
        repo_root=tmp_path,
        fact_card=_fact_card(),
        dry_run=True,
        environment={},
        gpu_command_runner=forbidden_gpu_probe,
    )
    assert dry["status"] == "dry_run"

    backend = MockGenerationBackend(
        {
            ("qwen_teacher", 42): _candidate_two(),
            ("qwen_teacher", 43): _candidate_one(),
            ("chk0_critic", 42): _critic_payload(),
            ("chk0_critic", 43): _critic_payload(),
        }
    )
    cache = tmp_path / "cache"
    generated = run_local_chk1_generation(
        prompt="reference-free evidence",
        repo_root=tmp_path,
        fact_card=_fact_card(),
        cache_dir=cache,
        backend=backend,
        mock=True,
        environment={},
        gpu_command_runner=forbidden_gpu_probe,
    )
    assert generated["cache_hit"] is False
    assert generated["status"] == "accepted"
    assert generated["selected_from"] == "42"
    assert generated["selected_response"] == (
        "The latest reading was 4.0 percent, after 3.7 percent.\n"
        "</think>\n"
        "The two available observations indicate an increase."
    )
    assert [request.role for request in backend.requests] == [
        "qwen_teacher",
        "qwen_teacher",
        "chk0_critic",
        "chk0_critic",
    ]

    no_call_backend = MockGenerationBackend({})
    cached = run_local_chk1_generation(
        prompt="reference-free evidence",
        repo_root=tmp_path,
        fact_card=_fact_card(),
        cache_dir=cache,
        backend=no_call_backend,
        mock=True,
        environment={},
        gpu_command_runner=forbidden_gpu_probe,
    )
    assert cached["cache_hit"] is True
    assert cached["selected_response"] == generated["selected_response"]
    assert no_call_backend.requests == []


def test_equal_coverage_candidates_use_shorter_output_as_tiebreak(
    tmp_path: Path,
) -> None:
    _model_repo(tmp_path)
    longer = _teacher_payload(
        reasoning=(
            "The latest reading was 4.0 percent, after the prior available "
            "reading of 3.7 percent, providing two observations for comparison."
        ),
        final_analysis="The available observations indicate an increase.",
        evidence_ids=["E1", "E2"],
    )
    backend = MockGenerationBackend(
        {
            ("qwen_teacher", 42): longer,
            ("qwen_teacher", 43): _candidate_two(),
            ("chk0_critic", 42): _critic_payload(),
            ("chk0_critic", 43): _critic_payload(),
        }
    )

    generated = run_local_chk1_generation(
        prompt="reference-free evidence",
        repo_root=tmp_path,
        fact_card=_fact_card(),
        backend=backend,
        mock=True,
        environment={},
    )

    assert generated["selected_from"] == "43"


def test_deterministic_failure_never_reaches_critic_when_other_candidate_passes(
    tmp_path: Path,
) -> None:
    _model_repo(tmp_path)
    unsupported = _teacher_payload(
        reasoning="The latest reading was 9.9 percent.",
        final_analysis="The measure was elevated.",
        evidence_ids=["E1"],
    )
    backend = MockGenerationBackend(
        {
            ("qwen_teacher", 42): unsupported,
            ("qwen_teacher", 43): _candidate_two(),
            ("chk0_critic", 43): _critic_payload(),
        }
    )

    generated = run_local_chk1_generation(
        prompt="reference-free evidence",
        repo_root=tmp_path,
        fact_card=_fact_card(),
        backend=backend,
        mock=True,
        environment={},
    )

    assert generated["selected_from"] == "43"
    assert generated["critics"]["42"] is None
    assert (
        "candidate_unsupported_numeric_claim"
        in generated["verifications"]["42"]["error_codes"]
    )
    assert [request.role for request in backend.requests] == [
        "qwen_teacher",
        "qwen_teacher",
        "chk0_critic",
    ]


def test_both_originals_rejected_trigger_exactly_one_successful_qwen_repair(
    tmp_path: Path,
) -> None:
    _model_repo(tmp_path)
    backend = MockGenerationBackend(
        {
            ("qwen_teacher", 42): _candidate_one(),
            ("qwen_teacher", 43): _candidate_two(),
            ("chk0_critic", 42): _critic_payload(style_score=3),
            ("chk0_critic", 43): _critic_payload(
                grounded=False,
                unsupported_claims=["unsupported direction"],
            ),
            ("qwen_repair", 42): _candidate_two(),
            ("chk0_critic", "repair"): _critic_payload(),
        }
    )

    generated = run_local_chk1_generation(
        prompt="reference-free evidence",
        repo_root=tmp_path,
        fact_card=_fact_card(),
        backend=backend,
        mock=True,
        environment={},
    )

    roles = [request.role for request in backend.requests]
    assert roles.count("qwen_repair") == 1
    assert roles.count("chk0_critic") == 3
    assert generated["selected_from"] == "repair"
    assert generated["repair"]["verification"]["passed"] is True
    assert generated["repair"]["critic"]["style_score"] == 4


def test_failed_single_repair_is_quarantined_and_cached_rejection_raises(
    tmp_path: Path,
) -> None:
    _model_repo(tmp_path)
    cache = tmp_path / "cache"
    backend = MockGenerationBackend(
        {
            ("qwen_teacher", 42): _candidate_one(),
            ("qwen_teacher", 43): _candidate_two(),
            ("chk0_critic", 42): _critic_payload(style_score=3),
            ("chk0_critic", 43): _critic_payload(style_score=3),
            ("qwen_repair", 42): _candidate_two(),
            ("chk0_critic", "repair"): _critic_payload(reasoning_consistency=False),
        }
    )

    with pytest.raises(CriticRejectedError) as first_error:
        run_local_chk1_generation(
            prompt="reference-free evidence",
            repo_root=tmp_path,
            fact_card=_fact_card(),
            cache_dir=cache,
            backend=backend,
            mock=True,
            environment={},
        )
    assert first_error.value.payload is not None
    assert first_error.value.payload["status"] == "rejected"
    assert first_error.value.payload["selected_response"] is None
    assert [request.role for request in backend.requests].count("qwen_repair") == 1

    no_call_backend = MockGenerationBackend({})
    with pytest.raises(CriticRejectedError) as cached_error:
        run_local_chk1_generation(
            prompt="reference-free evidence",
            repo_root=tmp_path,
            fact_card=_fact_card(),
            cache_dir=cache,
            backend=no_call_backend,
            mock=True,
            environment={},
        )
    assert cached_error.value.payload is not None
    assert cached_error.value.payload["cache_hit"] is True
    assert no_call_backend.requests == []


def test_immutable_cache_rejects_conflicting_content(tmp_path: Path) -> None:
    key = "a" * 64
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": key,
        "generation_provenance_sha256": "b" * 64,
        "selected_response": "reason\n</think>\nanswer",
    }
    store_cache_entry(tmp_path, payload)
    store_cache_entry(tmp_path, payload)
    with pytest.raises(CacheIntegrityError, match="collision"):
        store_cache_entry(
            tmp_path, {**payload, "selected_response": "different\n</think>\nanswer"}
        )
