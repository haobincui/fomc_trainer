from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import jobs.generation.generate_chk2_target_derived_minutes as target_job
from jobs.generation.generate_chk2_target_derived_minutes import (
    GENERATION_SYSTEM_PROMPT,
    API_KEY_ENV,
    STUDENT_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
    ModelDriftError,
    OpenAICompatibleDeepSeekBackend,
    PreparedRow,
    ProviderIdentityGuard,
    TargetDerivedDataError,
    TeacherConfig,
    TeacherResponse,
    canonical_json,
    prepare_official_targets,
    run_pipeline,
    sha256_text,
    validate_generation,
    validate_verification,
)


TARGET = (
    "In March 2024, staff reported that business investment increased 2 "
    "percent, while hiring remained unchanged amid uncertainty across surveyed "
    "industries during the month."
)
ANALYSIS = (
    "According to staff, firms expanded business capital spending by 2 percent "
    "in March 2024. Hiring activity, however, did not change, and uncertainty "
    "continued to surround conditions across the industries surveyed that month."
)
REASONING = (
    "Staff evidence for March 2024 links a 2 percent increase specifically to "
    "business capital investment. Labor demand is separately characterized as "
    "unchanged, while uncertainty remains present. The causal structure is "
    "limited to coexistence rather than an asserted cause, and the scope stays "
    "confined to the reported activities and surveyed industries."
)


class FakeTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(
        self, messages, *, tokenize: bool, add_generation_prompt: bool, **kwargs
    ):
        del kwargs
        assert add_generation_prompt is True
        assert messages[0] == {"role": "system", "content": STUDENT_SYSTEM_PROMPT}
        rendered = (
            "".join(f"<{item['role']}>{item['content']}" for item in messages)
            + "<assistant><think>\n"
        )
        return rendered.split() if tokenize else rendered

    def __call__(self, *, text: str):
        return {"input_ids": list(range(len(text.split())))}


def _generation_content(*, analysis: str = ANALYSIS, reasoning: str = REASONING) -> str:
    return canonical_json(
        {
            "atomic_claim_card": [
                {
                    "claim_id": "c1",
                    "target_span": TARGET,
                    "proposition": (
                        "Staff described a March 2024 rise in business investment "
                        "of 2 percent, unchanged hiring, and ongoing uncertainty "
                        "across surveyed industries."
                    ),
                }
            ],
            "analysis": analysis,
            "fidelity_reasoning": reasoning,
            "claim_alignment": [{"claim_id": "c1", "analysis_evidence": analysis}],
        }
    )


def _verification_content(*, passed: bool = True) -> str:
    return canonical_json(
        {
            "target_claims": [
                {
                    "target_span": TARGET,
                    "analysis_evidence": ANALYSIS,
                    "verdict": "entailed" if passed else "unsupported",
                }
            ],
            "analysis_claims": [
                {
                    "analysis_span": ANALYSIS,
                    "target_evidence": TARGET,
                    "verdict": "supported" if passed else "unsupported",
                }
            ],
            "reasoning_compatible": passed,
            "bidirectional_entailment": passed,
            "issues": [] if passed else ["The analysis omits a target claim."],
            "overall_pass": passed,
        }
    )


def _provider_response(
    *, response_id: str, content: str, fingerprint: str = "fp-fixed"
) -> TeacherResponse:
    usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    return TeacherResponse(
        raw_reasoning="Provider-native reasoning retained for audit only.",
        raw_content=content,
        response_id=response_id,
        returned_model="deepseek-v4-flash",
        system_fingerprint=fingerprint,
        finish_reason="stop",
        created=1,
        usage=usage,
        attempts=(
            {
                "attempt": 1,
                "status": "success",
                "response_id": response_id,
                "returned_model": "deepseek-v4-flash",
                "system_fingerprint": fingerprint,
                "finish_reason": "stop",
                "usage": usage,
            },
        ),
    )


def _prepared(sample_id: str = "official-test-1") -> PreparedRow:
    prompt = target_job._generation_user_prompt(TARGET)
    return PreparedRow(
        sample_id=sample_id,
        split="train",
        source_split="train",
        source_index=0,
        meeting_date="2024-03-20",
        section_name="Staff Review of the Economic Situation",
        section_category="core_economic_financial",
        topic="Business Investment",
        line_id="7",
        official_minutes_raw=TARGET,
        official_minutes_paragraph=TARGET,
        official_minutes_raw_sha256=sha256_text(TARGET),
        official_minutes_sha256=sha256_text(TARGET),
        official_normalization_repairs=(),
        official_source_match={
            "path": "official.csv",
            "line_id": "7",
            "section_name": "Staff Review of the Economic Situation",
            "raw_text_sha256": sha256_text(TARGET),
            "projected_text_sha256": sha256_text(TARGET),
            "normalization_repairs": [],
        },
        official_source_files=(
            {"path": "official.csv", "sha256": "a" * 64, "bytes": 1},
            {"path": "official.xlsx", "sha256": "b" * 64, "bytes": 1},
        ),
        legacy_sample_ids=("legacy-1",),
        legacy_source_rows=(
            {
                "sample_id": "legacy-1",
                "source_row_index": 1,
                "manifest_path": "legacy.jsonl",
                "manifest_line": 1,
            },
        ),
        duplicate_cluster_size=1,
        generation_user_prompt=prompt,
        generation_prompt_sha256=sha256_text(prompt),
    )


class ParallelBackend:
    def __init__(
        self,
        *,
        delay: float = 0.02,
        forbid_calls: bool = False,
        verification_passed: bool = True,
    ):
        self.delay = delay
        self.forbid_calls = forbid_calls
        self.verification_passed = verification_passed
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self.system_prompts: list[str] = []

    def generate(self, *, config, system_prompt, user_prompt, environment):
        del config, user_prompt, environment
        if self.forbid_calls:
            raise AssertionError("resume unexpectedly called provider")
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls += 1
            self.system_prompts.append(system_prompt)
            response_id = f"resp-{self.calls}"
        try:
            time.sleep(self.delay)
            if system_prompt == GENERATION_SYSTEM_PROMPT:
                content = _generation_content()
            elif system_prompt == VERIFICATION_SYSTEM_PROMPT:
                content = _verification_content(passed=self.verification_passed)
            else:
                raise AssertionError("unexpected system prompt")
            return _provider_response(response_id=response_id, content=content)
        finally:
            with self._lock:
                self.active -= 1


class TruncatedGenerationBackend:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0

    def generate(self, *, config, system_prompt, user_prompt, environment):
        del config, user_prompt, environment
        assert system_prompt == GENERATION_SYSTEM_PROMPT
        with self._lock:
            self.calls += 1
            response_id = f"truncated-{self.calls}"
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 4096,
            "total_tokens": 4196,
        }
        return TeacherResponse(
            raw_reasoning="Provider thinking consumed the output allowance.",
            raw_content="",
            response_id=response_id,
            returned_model="deepseek-v4-flash",
            system_fingerprint="fp-fixed",
            finish_reason="length",
            created=1,
            usage=usage,
            attempts=(
                {
                    "attempt": 1,
                    "status": "success",
                    "response_id": response_id,
                    "returned_model": "deepseek-v4-flash",
                    "system_fingerprint": "fp-fixed",
                    "finish_reason": "length",
                    "usage": usage,
                },
            ),
        )


def _preparation(count: int) -> dict:
    return {
        "schema_version": "test-preparation",
        "total_prepared": count,
        "prepared_counts": {"train": count},
    }


def test_source_preparation_uses_only_official_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_row = {
        "sample_id": "legacy-1",
        "split": "train",
        "meeting_date": "2024-03-20",
        "source_row_index": 9,
        "section_name": "Staff Review of the Economic Situation",
        "topic": "Business Investment",
        "quality_flags": [],
        "reference_excerpt": TARGET,
        "raw_analysis": "PRIVILEGED OLD ANALYSIS MUST NEVER BE REUSED",
        "teacher_rewrite_reasoning": "PRIVILEGED OLD REASONING",
        "teacher_rewrite_response": "PRIVILEGED SYNTHETIC ANSWER",
    }
    (source / "train_manifest.jsonl").write_text(
        canonical_json(source_row) + "\n", encoding="utf-8"
    )
    split_manifest = tmp_path / "meeting_split_manifest.json"
    split_manifest.write_text(
        json.dumps(
            [{"meeting_date": "2024-03-20", "split": "train", "sample_count": 1}]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        target_job,
        "_official_source_matches",
        lambda meeting_date, official: [
            {
                "path": "official.csv",
                "line_id": "7",
                "section_name": "Staff Review of the Economic Situation",
                "raw_text_sha256": sha256_text(official),
                "projected_text_sha256": sha256_text(official),
                "normalization_repairs": [],
            }
        ],
    )
    monkeypatch.setattr(
        target_job,
        "_source_file_records",
        lambda meeting_date: [
            {"path": f"{meeting_date}.csv", "sha256": "a" * 64, "bytes": 1},
            {"path": f"{meeting_date}.xlsx", "sha256": "b" * 64, "bytes": 1},
        ],
    )

    prepared, summary = prepare_official_targets(
        source_root=source,
        split_manifest_path=split_manifest,
        output_root=tmp_path / "output",
        selected_splits=("train",),
    )

    assert summary["total_prepared"] == 1
    row = prepared["train"][0]
    assert json.loads(row.generation_user_prompt.split("\n\n", 1)[1]) == {
        "official_minutes_paragraph": TARGET
    }
    assert "PRIVILEGED" not in row.generation_user_prompt
    assert row.legacy_sample_ids == ("legacy-1",)


def test_generation_builds_literal_official_completion() -> None:
    row = _prepared()
    normalized, diagnostics = validate_generation(
        row,
        _provider_response(response_id="generation", content=_generation_content()),
        tokenizer=FakeTokenizer(),
    )

    assert normalized["student_prompt"].endswith(canonical_json({"analysis": ANALYSIS}))
    reasoning, target = normalized["sft_completion"].split("\n</think>\n", 1)
    assert reasoning == REASONING
    assert target == TARGET
    assert diagnostics["analysis_numbers"] == {"2": 1}
    assert diagnostics["target_dates"] == ["2024", "march"]


def test_preflight_selection_covers_all_available_splits() -> None:
    rows = [replace(_prepared(f"train-{index}"), split="train") for index in range(4)]
    rows.extend(
        replace(_prepared(f"validation-{index}"), split="validation")
        for index in range(4)
    )
    rows.extend(replace(_prepared(f"test-{index}"), split="test") for index in range(4))

    selected = target_job._select_preflight_rows(rows, 6)

    assert [row.split for row in selected] == [
        "train",
        "validation",
        "test",
        "train",
        "validation",
        "test",
    ]


def test_generation_rejects_structured_fact_mismatch() -> None:
    bad_analysis = ANALYSIS.replace("2 percent", "3 percent")
    response = _provider_response(
        response_id="bad-generation",
        content=_generation_content(analysis=bad_analysis),
    )
    with pytest.raises(
        TargetDerivedDataError, match="analysis_numeric_multiset_mismatch"
    ):
        validate_generation(_prepared(), response, tokenizer=FakeTokenizer())


def test_verifier_fails_closed() -> None:
    row = _prepared()
    generation, _ = validate_generation(
        row,
        _provider_response(response_id="generation", content=_generation_content()),
        tokenizer=FakeTokenizer(),
    )
    verification, machine_pass, reasons = validate_verification(
        row,
        generation,
        _provider_response(
            response_id="verification", content=_verification_content(passed=False)
        ),
    )

    assert machine_pass is False
    assert verification["machine_pass"] is False
    assert "verifier_target_claim_not_entailed" in reasons
    assert "verifier_reported_issues" in reasons


def test_parallel_two_wave_pipeline_and_resume_without_calls(tmp_path: Path) -> None:
    rows = [_prepared(f"official-test-{index}") for index in range(4)]
    prepared = {"train": rows, "validation": [], "test": []}
    backend = ParallelBackend()

    summary = run_pipeline(
        prepared,
        preparation=_preparation(4),
        output_root=tmp_path / "release",
        tokenizer=FakeTokenizer(),
        backend=backend,
        environment={},
        concurrency=4,
        phase="all",
    )

    assert backend.calls == 8
    assert backend.max_active >= 2
    assert summary["status"] == "machine_screen_complete_human_review_pending"
    assert summary["total_machine_pass"] == 4
    assert summary["training_ready"] is False
    candidates = (
        (tmp_path / "release/sft_candidate/train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(candidates) == 4
    assert all(json.loads(line)["response"].endswith(TARGET) for line in candidates)

    fail_closed_backend = ParallelBackend(forbid_calls=True)
    with pytest.raises(TargetDerivedDataError, match="cache exists; pass --resume"):
        run_pipeline(
            prepared,
            preparation=_preparation(4),
            output_root=tmp_path / "release",
            tokenizer=FakeTokenizer(),
            backend=fail_closed_backend,
            environment={},
            concurrency=4,
            phase="all",
        )
    assert fail_closed_backend.calls == 0

    no_call_backend = ParallelBackend(forbid_calls=True)
    resumed = run_pipeline(
        prepared,
        preparation=_preparation(4),
        output_root=tmp_path / "release",
        tokenizer=FakeTokenizer(),
        backend=no_call_backend,
        environment={},
        concurrency=4,
        phase="all",
        resume=True,
    )
    assert no_call_backend.calls == 0
    assert resumed["total_machine_pass"] == 4


def test_end_to_end_preflight_precedes_full_unbudgeted_run(tmp_path: Path) -> None:
    rows = [_prepared(f"official-test-{index}") for index in range(10)]
    prepared = {"train": rows, "validation": [], "test": []}
    backend = ParallelBackend(delay=0)

    summary = run_pipeline(
        prepared,
        preparation=_preparation(10),
        output_root=tmp_path / "release",
        tokenizer=FakeTokenizer(),
        backend=backend,
        environment={},
        concurrency=4,
        preflight_rows=4,
        phase="all",
    )

    assert backend.calls == 20
    assert backend.system_prompts[:4] == [GENERATION_SYSTEM_PROMPT] * 4
    assert backend.system_prompts[4:8] == [VERIFICATION_SYSTEM_PROMPT] * 4
    assert backend.system_prompts[8:14] == [GENERATION_SYSTEM_PROMPT] * 6
    assert backend.system_prompts[14:] == [VERIFICATION_SYSTEM_PROMPT] * 6
    assert summary["total_machine_pass"] == 10
    preflight = json.loads(
        (tmp_path / "release/preflight.json").read_text(encoding="utf-8")
    )
    assert preflight["request_or_spending_budget"] is None
    assert preflight["generation"]["passed"] is True
    assert preflight["verification"]["passed"] is True


def test_generation_preflight_stops_truncated_contract_before_full_wave(
    tmp_path: Path,
) -> None:
    rows = [_prepared(f"official-test-{index}") for index in range(10)]
    prepared = {"train": rows, "validation": [], "test": []}
    backend = TruncatedGenerationBackend()

    with pytest.raises(TargetDerivedDataError, match="generation preflight failed"):
        run_pipeline(
            prepared,
            preparation=_preparation(10),
            output_root=tmp_path / "release",
            tokenizer=FakeTokenizer(),
            backend=backend,
            environment={},
            concurrency=4,
            preflight_rows=4,
            phase="all",
        )

    assert backend.calls == 4
    preflight = json.loads(
        (tmp_path / "release/preflight.json").read_text(encoding="utf-8")
    )
    assert preflight["generation"]["observed_passes"] == 0
    assert preflight["generation"]["passed"] is False
    assert preflight["verification"] is None
    assert not list((tmp_path / "release/cache/generation").glob("*.json"))
    assert (
        len(
            list(
                (tmp_path / "release/diagnostics/preflight_rejected").glob(
                    "*_generation_*/*.json"
                )
            )
        )
        == 4
    )

    retry_backend = ParallelBackend(delay=0)
    recovered = run_pipeline(
        prepared,
        preparation=_preparation(10),
        output_root=tmp_path / "release",
        tokenizer=FakeTokenizer(),
        backend=retry_backend,
        environment={},
        concurrency=4,
        preflight_rows=4,
        phase="all",
        resume=True,
    )
    assert retry_backend.calls == 20
    assert recovered["total_machine_pass"] == 10


def test_acquire_first_caches_every_raw_response_before_filtering(
    tmp_path: Path,
) -> None:
    rows = [_prepared(f"official-test-{index}") for index in range(10)]
    prepared = {"train": rows, "validation": [], "test": []}
    backend = TruncatedGenerationBackend()

    summary = run_pipeline(
        prepared,
        preparation=_preparation(10),
        output_root=tmp_path / "release",
        tokenizer=FakeTokenizer(),
        backend=backend,
        environment={},
        concurrency=4,
        preflight_rows=4,
        phase="generate",
        acquire_first=True,
    )

    assert backend.calls == 10
    assert len(list((tmp_path / "release/cache/generation").glob("*.json"))) == 10
    assert summary["status"] == "generation_complete_verification_pending"
    assert summary["acquisition_mode"] == "acquire_first"
    assert summary["quality_status"] == "deferred_post_acquisition"
    assert summary["split_counts"]["train"]["generation_responses"] == 10
    assert summary["split_counts"]["train"]["generation_gate_pass"] == 0
    preflight = json.loads(
        (tmp_path / "release/preflight.json").read_text(encoding="utf-8")
    )
    assert preflight["status"] == "skipped_for_acquire_first"
    assert preflight["total_generation_rows"] == 10


def test_rejected_verification_preflight_is_quarantined_and_retryable(
    tmp_path: Path,
) -> None:
    rows = [_prepared(f"official-test-{index}") for index in range(10)]
    prepared = {"train": rows, "validation": [], "test": []}
    rejected_backend = ParallelBackend(delay=0, verification_passed=False)

    with pytest.raises(TargetDerivedDataError, match="verification preflight failed"):
        run_pipeline(
            prepared,
            preparation=_preparation(10),
            output_root=tmp_path / "release",
            tokenizer=FakeTokenizer(),
            backend=rejected_backend,
            environment={},
            concurrency=4,
            preflight_rows=4,
            phase="all",
        )

    assert rejected_backend.calls == 8
    assert len(list((tmp_path / "release/cache/generation").glob("*.json"))) == 4
    assert not list((tmp_path / "release/cache/verification").glob("*.json"))
    assert (
        len(
            list(
                (tmp_path / "release/diagnostics/preflight_rejected").glob(
                    "*_verification_*/*.json"
                )
            )
        )
        == 4
    )

    retry_backend = ParallelBackend(delay=0)
    recovered = run_pipeline(
        prepared,
        preparation=_preparation(10),
        output_root=tmp_path / "release",
        tokenizer=FakeTokenizer(),
        backend=retry_backend,
        environment={},
        concurrency=4,
        preflight_rows=4,
        phase="all",
        resume=True,
    )
    assert retry_backend.calls == 16
    assert recovered["total_machine_pass"] == 10


def test_teacher_request_omits_client_side_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder: dict[str, object] = {}

    def create(**kwargs):
        recorder.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        reasoning_content="Native teacher reasoning.",
                        content="{}",
                    ),
                )
            ],
            id="response-no-cap",
            model="deepseek-v4-flash",
            system_fingerprint="fp-fixed",
            created=1,
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    backend = OpenAICompatibleDeepSeekBackend()
    monkeypatch.setattr(backend, "_client", lambda **kwargs: client)

    config = TeacherConfig()
    response = backend.generate(
        config=config,
        system_prompt=GENERATION_SYSTEM_PROMPT,
        user_prompt="{}",
        environment={API_KEY_ENV: "test-placeholder-key"},
    )

    assert config.contract()["max_tokens"] is None
    with pytest.raises(TypeError):
        TeacherConfig(max_tokens=4096)
    assert "max_tokens" not in recorder
    assert response.finish_reason == "stop"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--dry-run", "--preflight-rows", "0"], "preflight_rows"),
        (["--dry-run", "--concurrency", "0"], "concurrency"),
    ],
)
def test_dry_run_rejects_invalid_runtime_options(arguments, message) -> None:
    with pytest.raises(TargetDerivedDataError, match=message):
        target_job.main(arguments)


def test_provider_identity_guard_rejects_model_and_fingerprint_drift() -> None:
    guard = ProviderIdentityGuard()
    first = _provider_response(response_id="one", content=_generation_content())
    guard.bind(first)
    with pytest.raises(ModelDriftError, match="identity drift"):
        guard.bind(
            _provider_response(
                response_id="two",
                content=_generation_content(),
                fingerprint="fp-changed",
            )
        )
    wrong_model = TeacherResponse(
        **{**first.__dict__, "returned_model": "deepseek-chat", "response_id": "three"}
    )
    with pytest.raises(ModelDriftError, match="expected exactly"):
        ProviderIdentityGuard().bind(wrong_model)


def test_strict_json_rejects_duplicate_keys() -> None:
    response = _provider_response(
        response_id="duplicate",
        content=(
            '{"atomic_claim_card":[],"analysis":"one","analysis":"two",'
            '"fidelity_reasoning":"reason","claim_alignment":[]}'
        ),
    )
    with pytest.raises(TargetDerivedDataError, match="duplicate_json_keys"):
        validate_generation(_prepared(), response, tokenizer=FakeTokenizer())
