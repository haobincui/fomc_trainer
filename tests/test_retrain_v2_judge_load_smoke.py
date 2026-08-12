import threading
import time

from jobs.retrain_v2.judge_load_smoke import (
    build_synthetic_evidence,
    count_chat_tokens,
    run_load_smoke,
    size_synthetic_prompt,
)


class _WordTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["add_generation_prompt"] is True
        return " ".join(message["content"] for message in messages).split()


class _BatchEncodingTokenizer(_WordTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": [super().apply_chat_template(messages, **kwargs)]}


def test_synthetic_evidence_is_deterministic_and_explicitly_fake():
    evidence = build_synthetic_evidence(3, request_id=2)
    assert evidence == build_synthetic_evidence(3, request_id=2)
    assert "SYNTHETIC_LOAD_SMOKE=1" in evidence
    assert "not real data" in evidence
    assert evidence.count("record_id=SYN-") == 3


def test_prompt_sizing_stays_at_or_below_target():
    evidence, candidate, prompt_tokens, rows = size_synthetic_prompt(
        _WordTokenizer(), request_id=0, target_prompt_tokens=300
    )
    assert evidence
    assert candidate
    assert rows > 0
    assert 256 <= prompt_tokens <= 300


def test_token_count_supports_transformers_batch_encoding_return():
    count = count_chat_tokens(_BatchEncodingTokenizer(), "fake evidence", "fake candidate")
    assert count > 2


def test_load_smoke_runs_requests_concurrently_and_aggregates_results():
    barrier = threading.Barrier(4)

    def judge(**kwargs):
        assert kwargs["max_retries"] == 1
        assert kwargs["backoff_seconds"] == 0
        barrier.wait(timeout=2)
        time.sleep(0.01)
        return {
            "data_fidelity": 4.0,
            "trend_reasoning": 4.0,
            "policy_relevance": 4.0,
            "uncertainty_calibration": 4.0,
            "fomc_style": 4.0,
            "unsupported_claims": [],
        }, "{}", 1

    result = run_load_smoke(
        url="http://127.0.0.1:8000/v1/chat/completions",
        model="Qwen3.5-9B",
        tokenizer=_WordTokenizer(),
        concurrency=4,
        target_prompt_tokens=300,
        timeout=5,
        judge_fn=judge,
        gpu_index=0,
        max_gpu_memory_mib=22_000,
        memory_poll_interval=0.001,
        memory_reader=lambda index: 17_625 if index == 0 else 99_999,
    )

    assert result["status"] == "passed"
    assert result["passed"] == 4
    assert result["success_rate"] == 1.0
    assert result["strict_json_schema"] is True
    assert result["enable_thinking"] is False
    assert result["input_kind"] == "in_memory_synthetic_only"
    assert result["peak_gpu_memory_mib"] == 17_625
    assert result["gpu_memory_gate_passed"] is True
    assert len(result["requests"]) == 4
