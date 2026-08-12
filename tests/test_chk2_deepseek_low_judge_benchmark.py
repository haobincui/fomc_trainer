from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from jobs.retrain_v2 import benchmark_chk2_deepseek_low_judge as benchmark
from open_r1.structured_response import parse_structured_response


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(range(max(1, len(text.split()))))


class _FakeResponses:
    def __init__(self, handler: Any, *, retry_first: bool = False) -> None:
        self.handler = handler
        self.retry_first = retry_first
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.retry_first and len(self.calls) == 1:
            raise TimeoutError("synthetic timeout")
        payload = self.handler(kwargs)
        visible = json.dumps(payload, separators=(",", ":"))
        return SimpleNamespace(
            status="completed",
            model=benchmark.MODEL,
            id=f"fake-response-{len(self.calls)}",
            output_text=visible,
            output=[
                SimpleNamespace(
                    type="reasoning",
                    content=[
                        SimpleNamespace(type="reasoning_text", text="private fake reasoning")
                    ],
                ),
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text=visible)],
                ),
            ],
            usage=SimpleNamespace(
                input_tokens=1600,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
                output_tokens=400,
                output_tokens_details=SimpleNamespace(reasoning_tokens=256),
                total_tokens=2000,
            ),
        )


class _FakeClient:
    def __init__(self, handler: Any, *, retry_first: bool = False) -> None:
        self.responses = _FakeResponses(handler, retry_first=retry_first)


def _args(tmp_path: Path, *, dry_run: bool = False) -> Namespace:
    return Namespace(
        completion_parquet=benchmark.DEFAULT_COMPLETION_PARQUET,
        dataset_dir=benchmark.DEFAULT_DATASET_DIR,
        high_reward_log=benchmark.DEFAULT_HIGH_REWARD_LOG,
        high_canary_reward_log=benchmark.DEFAULT_HIGH_CANARY_REWARD_LOG,
        tokenizer_path=benchmark.DEFAULT_TOKENIZER_PATH,
        output_dir=tmp_path / "low-benchmark",
        dry_run=dry_run,
    )


def _violation(
    *, section: str, severity: str, quote: str, kind: str = "factual"
) -> dict[str, str]:
    return {
        "section": section,
        "kind": kind,
        "severity": severity,
        "candidate_quote": quote,
        "explanation": "The quoted statement is not supported by the evidence.",
    }


def _quotes(candidate: str, section: str, count: int) -> list[str]:
    if count == 0:
        return []
    parsed = parse_structured_response(candidate)
    text = parsed.reasoning if section == "think" else parsed.answer
    unique: list[str] = []
    for word in text.split():
        if word not in unique:
            unique.append(word)
        if len(unique) == count:
            break
    assert len(unique) == count
    return unique


def _fake_handler(groups: list[benchmark.BenchmarkGroup]):
    metadata: dict[str, tuple[set[str], dict[str, Any] | None]] = {}
    for group in groups:
        for completion, label, baseline in zip(
            group.completions, group.labels, group.high_baseline, strict=True
        ):
            candidate = benchmark._candidate_for_judge(completion)
            digest = hashlib.sha256(candidate.encode()).hexdigest()
            labels, existing = metadata.setdefault(digest, (set(), baseline and dict(baseline)))
            labels.add(label)
            if existing is None and baseline is not None:
                metadata[digest] = (labels, dict(baseline))

    def handler(kwargs: dict[str, Any]) -> dict[str, Any]:
        assert kwargs["model"] == benchmark.MODEL
        assert kwargs["reasoning"] == {"effort": "low"}
        assert kwargs["max_output_tokens"] == 16_384
        payload = json.loads(kwargs["input"])
        candidates = payload["candidates"]
        result: dict[str, Any] = {}
        # Deliberately return a non-schema insertion order. Binding is by ID,
        # never by JSON object order.
        for candidate_id in ("candidate_3", "candidate_1", "candidate_0", "candidate_2"):
            candidate = candidates[candidate_id]
            digest = hashlib.sha256(candidate.encode()).hexdigest()
            labels, baseline = metadata[digest]
            rubric = (
                dict(baseline["judge"])
                if baseline is not None
                else {key: 4 for key in benchmark._RUBRIC_KEYS}
            )
            violations: list[dict[str, str]] = []
            if baseline is not None:
                for section in ("think", "answer"):
                    for severity in ("major", "minor"):
                        count = int(
                            baseline["penalties"].get(
                                f"{severity}_{section}_error", 0
                            )
                        )
                        violations.extend(
                            _violation(
                                section=section,
                                severity=severity,
                                quote=quote,
                            )
                            for quote in _quotes(candidate, section, count)
                        )
            elif "totalsl_wrong_millions" in labels or "totalsl_wrong_billions" in labels:
                violations.append(
                    _violation(
                        section="answer",
                        severity="major",
                        quote=_quotes(candidate, "answer", 1)[0],
                        kind="numerical",
                    )
                )
                rubric["data_fidelity"] = 1
            elif "unemployment_unsupported_cause" in labels:
                violations.append(
                    _violation(
                        section="think",
                        severity="major",
                        quote="caused by restrictive monetary policy",
                        kind="causal",
                    )
                )
                rubric["data_fidelity"] = 1
            elif "unemployment_wrong_number" in labels:
                violations.append(
                    _violation(
                        section="answer",
                        severity="major",
                        quote="8.8 percent",
                        kind="numerical",
                    )
                )
                rubric["data_fidelity"] = 1
            elif "unemployment_meta_style" in labels:
                violations.append(
                    _violation(
                        section="answer",
                        severity="minor",
                        quote="Based on the provided data",
                        kind="style",
                    )
                )
                rubric["fomc_style"] = 2
            elif "ffr_think_target_leakage" in labels:
                violations.append(
                    _violation(
                        section="think",
                        severity="major",
                        quote="target meeting Minutes",
                        kind="target_leakage",
                    )
                )
                rubric["data_fidelity"] = 0
            elif "ffr_answer_target_leakage" in labels:
                violations.append(
                    _violation(
                        section="answer",
                        severity="major",
                        quote="target meeting Minutes",
                        kind="target_leakage",
                    )
                )
                rubric["data_fidelity"] = 0
            result[candidate_id] = {**rubric, "violations": violations}
        return result

    return handler


def test_low_request_is_locked_to_effort_and_strict_v3_schema() -> None:
    evidence = json.dumps(
        {"atomic_topic": "test", "evidence": [{"value": "1"}]},
        separators=(",", ":"),
    )
    request = benchmark.build_low_batch_request(
        evidence=evidence,
        candidates=["candidate"] * 4,
    )

    assert request["model"] == "deepseek-v4-flash"
    assert request["reasoning"] == {"effort": "low"}
    assert request["max_output_tokens"] == 16_384
    schema = request["text"]["format"]
    assert schema["strict"] is True
    assert set(schema["schema"]["properties"]) == set(benchmark.CANDIDATE_IDS)
    assert schema["schema"]["additionalProperties"] is False


def test_frozen_groups_have_exact_permutation_and_high8_binding(tmp_path: Path) -> None:
    args = _args(tmp_path, dry_run=True)
    groups = benchmark.load_benchmark_groups(
        completion_parquet=args.completion_parquet,
        dataset_dir=args.dataset_dir,
        high_reward_log=args.high_reward_log,
        high_canary_reward_log=args.high_canary_reward_log,
    )

    assert len(groups) == 10
    assert sum(len(group.completions) for group in groups) == 40
    assert len({group.base_group_id for group in groups}) == 5
    for base in {group.base_group_id for group in groups}:
        pair = {group.order_variant: group for group in groups if group.base_group_id == base}
        assert set(pair) == {"original", "permuted_2031"}
        assert pair["permuted_2031"].labels == tuple(
            pair["original"].labels[index] for index in benchmark.PERMUTATION
        )
    comparable = [
        baseline
        for group in groups
        if group.order_variant == "original"
        for baseline in group.high_baseline
        if baseline is not None
    ]
    assert len(comparable) == 8


def test_fake_retry_resume_privacy_and_summary_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    args = _args(tmp_path)
    groups = benchmark.load_benchmark_groups(
        completion_parquet=args.completion_parquet,
        dataset_dir=args.dataset_dir,
        high_reward_log=args.high_reward_log,
        high_canary_reward_log=args.high_canary_reward_log,
    )
    client = _FakeClient(_fake_handler(groups), retry_first=True)

    summary = benchmark.run_benchmark(args, client=client, tokenizer=_Tokenizer())

    assert len(client.responses.calls) == 11
    assert summary["groups"] == 10
    assert summary["candidates"] == 40
    assert summary["high8_comparison"]["candidates"] == 8
    assert summary["permutation_consistency"]["fixed_permutation"] == [2, 0, 3, 1]
    assert summary["provider"]["reasoning_effort"] == "low"
    assert summary["provider"]["max_output_tokens"] == 16_384
    assert all(summary["gates"].values())
    assert summary["suitable_for_chk2_trial"] is True

    records = (args.output_dir / "benchmark_records.jsonl").read_text(encoding="utf-8")
    assert len(records.splitlines()) == 40
    protected = [
        *(group.evidence for group in groups),
        *(completion for group in groups for completion in group.completions),
        "private fake reasoning",
    ]
    for path in args.output_dir.rglob("*"):
        if path.is_file():
            raw = path.read_text(encoding="utf-8")
            assert all(value not in raw for value in protected)
            assert '"candidate_quote":' not in raw
            assert '"explanation"' not in raw

    no_calls = _FakeClient(lambda _kwargs: pytest.fail("cache should avoid provider"))
    resumed = benchmark.run_benchmark(args, client=no_calls, tokenizer=_Tokenizer())
    assert no_calls.responses.calls == []
    assert resumed["cache_hits"] == 10
    assert resumed["suitable_for_chk2_trial"] is True


def test_decode_maps_by_candidate_id_not_object_order() -> None:
    evaluation = {
        **{key: 4 for key in benchmark._RUBRIC_KEYS},
        "violations": [],
    }
    visible = json.dumps(
        {
            "candidate_2": evaluation,
            "candidate_0": evaluation,
            "candidate_3": evaluation,
            "candidate_1": evaluation,
        }
    )

    decoded = benchmark._decode_batch_output(visible)

    assert len(decoded) == 4
    assert all(value["data_fidelity"] == 4 for value in decoded)
