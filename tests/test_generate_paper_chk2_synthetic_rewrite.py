from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from jobs.generation.generate_paper_chk2_synthetic_rewrite import (
    API_KEY_ENV,
    DEFAULT_CONCURRENCY,
    DEFAULT_PREFLIGHT_ROWS,
    DEFAULT_SOURCE_ROOT,
    EXPECTED_SOURCE_SAMPLE_ID_SHA256,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_TOTAL,
    MODEL,
    ROLE_REWRITE_PRIMARY,
    ROLE_REWRITE_REPAIR,
    ROLE_VALIDATOR_A_PRIMARY,
    ROLE_VALIDATOR_A_REPAIR,
    ROLE_VALIDATOR_B,
    STUDENT_SYSTEM_PROMPT,
    TERMINAL_FIDELITY_REJECT,
    TERMINAL_GENERATION_REJECT,
    TERMINAL_PASS,
    TERMINAL_SCHEMA_VERSION,
    ContractError,
    PreparedRow,
    ProviderResponse,
    SyntheticRewriteError,
    _candidate_from_response,
    _repair_user_prompt,
    _validate_validator_a,
    _validate_validator_b,
    canonical_json,
    prepare_source_release,
    render_user_prompt,
    run_pipeline,
    sha256_text,
)


ANALYSIS = (
    "In March 2024, industrial production increased 2 percent, while "
    "uncertainty about the durability of the improvement remained elevated."
)
REASONING = (
    "March 2024 anchors the observation. Industrial production moves upward "
    "by 2 percent, and elevated uncertainty continues to qualify the "
    "durability of the improvement in the formal formulation."
)
REWRITE = (
    "In March 2024, industrial production was reported to have increased "
    "2 percent, while uncertainty surrounding the durability of the "
    "improvement remained elevated over the period."
)
OFFICIAL = (
    "Industrial production increased 2 percent in March 2024, while "
    "uncertainty about whether the improvement would endure remained elevated."
)


class FakeTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        del kwargs
        assert tokenize is False
        assert add_generation_prompt is True
        assert messages[0] == {"role": "system", "content": STUDENT_SYSTEM_PROMPT}
        return (
            "<bos>"
            + "".join(f"<{item['role']}>{item['content']}" for item in messages)
            + "<assistant><think>\n"
        )

    def __call__(self, *, text):
        return {"input_ids": list(range(len(text.split())))}


def _provider_response(
    *,
    response_id: str,
    content: dict,
    reasoning: str = "Independent audit reasoning.",
    fingerprint: str = "fp-fixed",
) -> ProviderResponse:
    return ProviderResponse(
        raw_reasoning=reasoning,
        raw_content=canonical_json(content),
        response_id=response_id,
        returned_model=MODEL,
        system_fingerprint=fingerprint,
        finish_reason="stop",
        created=1,
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )


def _validator_a_payload(user_prompt: str, *, passed: bool) -> dict:
    payload = json.loads(user_prompt.split("\n\n", 1)[1])
    source = payload["source_analysis"]
    reasoning = payload["teacher_response_analysis"]
    rewrite = payload["rewritten_minutes"]
    if passed:
        return {
            "source_claims": [
                {
                    "source_span": source,
                    "reasoning_evidence": reasoning,
                    "rewrite_evidence": rewrite,
                    "rewrite_verdict": "entailed",
                    "reasoning_verdict": "covered",
                }
            ],
            "rewrite_claims": [
                {
                    "rewrite_span": rewrite,
                    "source_evidence": source,
                    "verdict": "supported",
                }
            ],
            "reasoning_issues": [],
            "bidirectional_entailment": True,
            "reasoning_compatible": True,
            "issues": [],
            "overall_pass": True,
        }
    return {
        "source_claims": [
            {
                "source_span": source,
                "reasoning_evidence": None,
                "rewrite_evidence": None,
                "rewrite_verdict": "omitted",
                "reasoning_verdict": "not_covered",
            }
        ],
        "rewrite_claims": [
            {
                "rewrite_span": rewrite,
                "source_evidence": source,
                "verdict": "supported",
            }
        ],
        "reasoning_issues": [],
        "bidirectional_entailment": False,
        "reasoning_compatible": False,
        "issues": ["material source claim omitted"],
        "overall_pass": False,
    }


def _validator_b_payload(user_prompt: str, *, passed: bool) -> dict:
    payload = json.loads(user_prompt.split("\n\n", 1)[1])
    official = payload["official_minutes"]
    rewrite = payload["rewritten_minutes"]
    if passed:
        scores = {
            "factual_consistency": 30,
            "claim_coverage": 25,
            "absence_of_unsupported_content": 20,
            "fomc_minutes_style": 15,
            "clarity_and_coherence": 5,
            "independent_rewriting": 5,
        }
        official_verdict = "preserved"
        rewrite_verdict = "supported"
        rewrite_evidence: str | None = rewrite
        official_evidence: str | None = official
        critical_errors: list[str] = []
    else:
        scores = {
            "factual_consistency": 0,
            "claim_coverage": 0,
            "absence_of_unsupported_content": 0,
            "fomc_minutes_style": 10,
            "clarity_and_coherence": 2,
            "independent_rewriting": 0,
        }
        official_verdict = "omitted"
        rewrite_verdict = "unsupported"
        rewrite_evidence = None
        official_evidence = None
        critical_errors = ["reference mismatch"]
    return {
        "official_claims": [
            {
                "official_span": official,
                "rewrite_evidence": rewrite_evidence,
                "verdict": official_verdict,
            }
        ],
        "rewrite_claims": [
            {
                "rewrite_span": rewrite,
                "official_evidence": official_evidence,
                "verdict": rewrite_verdict,
            }
        ],
        "scores": scores,
        "critical_errors": critical_errors,
        "overall_score": sum(scores.values()),
        "overall_pass": passed,
    }


class ScenarioBackend:
    def __init__(
        self,
        *,
        validator_a_primary_pass: bool = True,
        validator_a_repair_pass: bool = True,
        reference_pass: bool = True,
        bad_primary: bool = False,
        bad_repair: bool = False,
        malformed_reference: bool = False,
        drift_role: str | None = None,
    ) -> None:
        self.validator_a_primary_pass = validator_a_primary_pass
        self.validator_a_repair_pass = validator_a_repair_pass
        self.reference_pass = reference_pass
        self.bad_primary = bad_primary
        self.bad_repair = bad_repair
        self.malformed_reference = malformed_reference
        self.drift_role = drift_role
        self.requests: list[dict] = []
        self._lock = threading.Lock()

    def generate(self, *, role, config, system_prompt, user_prompt, environment):
        del environment
        assert config.model == MODEL
        assert config.reasoning_effort == "high"
        contract = config.contract()
        assert contract["temperature"] is None
        assert contract["top_p"] is None
        assert contract["fallback"] == "forbidden"
        with self._lock:
            request_number = len(self.requests) + 1
            self.requests.append(
                {
                    "role": role,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )
        fingerprint = "fp-drift" if role == self.drift_role else "fp-fixed"
        if role in {ROLE_REWRITE_PRIMARY, ROLE_REWRITE_REPAIR}:
            payload = json.loads(user_prompt.split("\n\n", 1)[1])
            if role == ROLE_REWRITE_PRIMARY:
                assert set(payload) == {"analysis"}
                answer = REWRITE.replace("2 percent", "3 percent") if self.bad_primary else REWRITE
            else:
                assert set(payload) == {"analysis", "repair_feedback"}
                answer = REWRITE.replace("2 percent", "4 percent") if self.bad_repair else REWRITE
            return _provider_response(
                response_id=f"r-{request_number}",
                content={"answer": answer},
                reasoning=REASONING,
                fingerprint=fingerprint,
            )
        if role in {ROLE_VALIDATOR_A_PRIMARY, ROLE_VALIDATOR_A_REPAIR}:
            passed = (
                self.validator_a_repair_pass
                if role == ROLE_VALIDATOR_A_REPAIR
                else self.validator_a_primary_pass
            )
            return _provider_response(
                response_id=f"r-{request_number}",
                content=_validator_a_payload(user_prompt, passed=passed),
                fingerprint=fingerprint,
            )
        if role == ROLE_VALIDATOR_B:
            content = (
                {"unexpected": True}
                if self.malformed_reference
                else _validator_b_payload(user_prompt, passed=self.reference_pass)
            )
            return _provider_response(
                response_id=f"r-{request_number}",
                content=content,
                fingerprint=fingerprint,
            )
        raise AssertionError(f"unexpected role: {role}")


class NoCallBackend:
    def generate(self, **kwargs):  # pragma: no cover - assertion path
        raise AssertionError(f"resume unexpectedly called provider: {kwargs['role']}")


def _row(split: str, index: int = 0) -> PreparedRow:
    prompt = render_user_prompt(ANALYSIS)
    return PreparedRow(
        sample_id=f"{split}-{index}",
        split=split,
        source_index=index,
        meeting_date=f"2024-0{index + 1}-15-{split}",
        source_analysis=ANALYSIS,
        official_minutes=OFFICIAL,
        student_prompt=prompt,
        source_analysis_sha256=sha256_text(ANALYSIS),
        official_minutes_sha256=sha256_text(OFFICIAL),
        prompt_sha256=sha256_text(prompt),
        source_manifest_sha256="a" * 64,
        source_response_sha256="b" * 64,
    )


def _prepared() -> dict[str, list[PreparedRow]]:
    return {split: [_row(split)] for split in ("train", "validation", "test")}


def _preparation() -> dict:
    return {
        "root": "test-source",
        "summary_sha256": "1" * 64,
        "handoff_sha256": "2" * 64,
        "merge_receipt_sha256": "3" * 64,
        "sample_id_sha256": "4" * 64,
        "split_counts": {"train": 1, "validation": 1, "test": 1},
        "total_rows": 3,
        "artifacts": {},
    }


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_fixed_source_release_prepares_all_2083_rows(tmp_path: Path) -> None:
    prepared, summary = prepare_source_release(
        source_root=DEFAULT_SOURCE_ROOT,
        output_root=tmp_path / "prepared",
        enforce_pins=True,
    )
    assert summary["total_rows"] == EXPECTED_TOTAL
    assert summary["split_counts"] == EXPECTED_SPLIT_COUNTS
    assert summary["sample_id_sha256"] == EXPECTED_SOURCE_SAMPLE_ID_SHA256
    assert {split: len(rows) for split, rows in prepared.items()} == EXPECTED_SPLIT_COUNTS


def test_primary_pass_and_low_reference_score_remains_training_pass(
    tmp_path: Path,
) -> None:
    backend = ScenarioBackend(reference_pass=False)
    summary = run_pipeline(
        _prepared(),
        preparation=_preparation(),
        output_root=tmp_path / "run",
        tokenizer=FakeTokenizer(),
        backend=backend,
        environment={API_KEY_ENV: "not-persisted"},
        phase="all",
        concurrency=1,
        preflight_rows=3,
    )
    assert summary["quality_status"] == "passed"
    assert summary["total_training_pass"] == 3
    assert summary["unresolved_failure_count"] == 0
    assert summary["reference_diagnostic_affects_selection"] is False
    terminal = _jsonl(tmp_path / "run/terminal/train.jsonl")[0]
    assert terminal["schema_version"] == TERMINAL_SCHEMA_VERSION
    assert terminal["terminal_status"] == TERMINAL_PASS
    assert terminal["training_pass"] is True
    assert terminal["reference_diagnostic"]["complete"] is True
    assert terminal["reference_diagnostic"]["diagnostic_pass"] is False
    assert terminal["reference_diagnostic"]["warning"] == (
        "REFERENCE_DIAGNOSTIC_WARNING"
    )
    assert terminal["lineage"]["rewrite_teacher_input_fields"] == [
        "source_analysis"
    ]
    assert terminal["lineage"][
        "rewrite_teacher_received_official_validator_feedback"
    ] is False
    assert terminal["lineage"]["reference_validator_saw_official_target"] is True
    assert terminal["lineage"][
        "reference_validator_used_for_training_selection"
    ] is False
    assert terminal["lineage"]["reference_audit_only"] is True
    assert terminal["lineage"]["official_minutes_used_as_student_target"] is False
    assert terminal["lineage"]["suitable_for_leakage_safe_evaluation"] is False
    assert "official_minutes" not in terminal
    assert set(_jsonl(tmp_path / "run/sft_candidate/train.jsonl")[0]) == {
        "prompt",
        "response",
    }
    assert ROLE_REWRITE_REPAIR not in [request["role"] for request in backend.requests]

    projections = {
        terminal["generation"]["provider"]["request_projection"]["role"]:
        terminal["generation"]["provider"]["request_projection"],
        terminal["validator_a"]["provider"]["request_projection"]["role"]:
        terminal["validator_a"]["provider"]["request_projection"],
        terminal["reference_diagnostic"]["provider"]["request_projection"]["role"]:
        terminal["reference_diagnostic"]["provider"]["request_projection"],
    }
    assert projections[ROLE_REWRITE_PRIMARY]["allowed_fields"] == ["source_analysis"]
    assert projections[ROLE_VALIDATOR_A_PRIMARY]["official_target_policy"] == "forbidden"
    assert projections[ROLE_VALIDATOR_B]["allowed_fields"] == [
        "official_minutes",
        "rewritten_minutes",
    ]
    assert all(
        projection["reference_diagnostic_can_trigger_repair"] is False
        for projection in projections.values()
    )
    assert "not-persisted" not in (tmp_path / "run/terminal/train.jsonl").read_text()


def test_validator_a_failure_repairs_once_with_exact_nonofficial_evidence(
    tmp_path: Path,
) -> None:
    backend = ScenarioBackend(
        validator_a_primary_pass=False,
        validator_a_repair_pass=True,
        reference_pass=True,
    )
    summary = run_pipeline(
        _prepared(),
        preparation=_preparation(),
        output_root=tmp_path / "run",
        tokenizer=FakeTokenizer(),
        backend=backend,
        environment={API_KEY_ENV: "secret-test-value"},
        phase="all",
        concurrency=1,
        preflight_rows=3,
    )
    assert summary["total_training_pass"] == 3
    roles = [request["role"] for request in backend.requests]
    assert roles.count(ROLE_REWRITE_REPAIR) == 3
    assert roles.count(ROLE_VALIDATOR_A_REPAIR) == 3
    for request in backend.requests:
        if request["role"] != ROLE_REWRITE_REPAIR:
            continue
        payload = json.loads(request["user_prompt"].split("\n\n", 1)[1])
        assert set(payload) == {"analysis", "repair_feedback"}
        evidence = payload["repair_feedback"]["validator_a_exact_evidence"]
        assert evidence["source_claims"][0]["source_span"] == ANALYSIS
        assert evidence["rewrite_claims"][0]["rewrite_span"] == REWRITE
        assert "official_minutes" not in canonical_json(payload)
        assert OFFICIAL not in request["user_prompt"]
    terminal = _jsonl(tmp_path / "run/terminal/train.jsonl")[0]
    projection = terminal["generation"]["provider"]["request_projection"]
    assert projection["repair_trigger_source"] == "validator_a"
    assert projection["reference_diagnostic_can_trigger_repair"] is False


def test_deterministic_primary_and_repair_failure_is_terminal_generation_reject(
    tmp_path: Path,
) -> None:
    backend = ScenarioBackend(bad_primary=True, bad_repair=True)
    summary = run_pipeline(
        _prepared(),
        preparation=_preparation(),
        output_root=tmp_path / "run",
        tokenizer=FakeTokenizer(),
        backend=backend,
        environment={API_KEY_ENV: "test"},
        phase="all",
        concurrency=1,
        preflight_rows=3,
    )
    assert summary["total_training_pass"] == 0
    assert summary["quality_status"] == "failed"
    row = _jsonl(tmp_path / "run/terminal/train.jsonl")[0]
    assert row["terminal_status"] == TERMINAL_GENERATION_REJECT
    assert row["training_pass"] is False
    assert row["generation"]["repair_used"] is True
    assert row["generation"]["deterministic_validation"]["machine_pass"] is False
    assert row["generation"]["provider"]["request_projection"][
        "repair_trigger_source"
    ] == "deterministic_validation"
    assert row["validator_a"]["provider"] == {}
    assert row["reference_diagnostic"]["provider"] == {}
    assert ROLE_VALIDATOR_B not in [request["role"] for request in backend.requests]


def test_validator_a_fails_after_single_repair_is_input_fidelity_reject(
    tmp_path: Path,
) -> None:
    backend = ScenarioBackend(
        validator_a_primary_pass=False,
        validator_a_repair_pass=False,
    )
    summary = run_pipeline(
        _prepared(),
        preparation=_preparation(),
        output_root=tmp_path / "run",
        tokenizer=FakeTokenizer(),
        backend=backend,
        environment={API_KEY_ENV: "test"},
        phase="all",
        concurrency=1,
        preflight_rows=3,
    )
    assert summary["total_training_pass"] == 0
    row = _jsonl(tmp_path / "run/terminal/train.jsonl")[0]
    assert row["terminal_status"] == TERMINAL_FIDELITY_REJECT
    assert row["generation"]["repair_used"] is True
    assert row["validator_a"]["complete"] is True
    assert row["validator_a"]["machine_pass"] is False
    assert row["reference_diagnostic"]["complete"] is False
    assert ROLE_VALIDATOR_B not in [request["role"] for request in backend.requests]


def test_reference_contract_failure_is_unresolved_not_a_warning(tmp_path: Path) -> None:
    backend = ScenarioBackend(malformed_reference=True)
    with pytest.raises(SyntheticRewriteError, match="preflight has unresolved"):
        run_pipeline(
            _prepared(),
            preparation=_preparation(),
            output_root=tmp_path / "run",
            tokenizer=FakeTokenizer(),
            backend=backend,
            environment={API_KEY_ENV: "test"},
            phase="all",
            concurrency=1,
            preflight_rows=3,
        )
    assert not (tmp_path / "run/terminal/train.jsonl").exists()


def test_provider_fingerprint_is_fixed_across_roles(tmp_path: Path) -> None:
    backend = ScenarioBackend(drift_role=ROLE_VALIDATOR_B)
    with pytest.raises(SyntheticRewriteError, match="preflight has unresolved"):
        run_pipeline(
            _prepared(),
            preparation=_preparation(),
            output_root=tmp_path / "run",
            tokenizer=FakeTokenizer(),
            backend=backend,
            environment={API_KEY_ENV: "test"},
            phase="all",
            concurrency=1,
            preflight_rows=3,
        )


def test_resume_replays_terminal_identity_without_provider_calls(tmp_path: Path) -> None:
    output = tmp_path / "run"
    first = run_pipeline(
        _prepared(),
        preparation=_preparation(),
        output_root=output,
        tokenizer=FakeTokenizer(),
        backend=ScenarioBackend(),
        environment={API_KEY_ENV: "test"},
        phase="all",
        concurrency=1,
        preflight_rows=3,
    )
    resumed = run_pipeline(
        _prepared(),
        preparation=_preparation(),
        output_root=output,
        tokenizer=FakeTokenizer(),
        backend=NoCallBackend(),
        environment={API_KEY_ENV: "different-not-used"},
        phase="all",
        concurrency=1,
        preflight_rows=3,
        resume=True,
    )
    assert resumed["total_training_pass"] == first["total_training_pass"] == 3
    assert resumed["provider_identities"]["returned_model"] == MODEL
    assert resumed["provider_identities"]["system_fingerprint"] == "fp-fixed"
    assert set(resumed["provider_identities"]["roles_observed"]) == {
        ROLE_REWRITE_PRIMARY,
        ROLE_VALIDATOR_A_PRIMARY,
        ROLE_VALIDATOR_B,
    }


def test_raw_reasoning_meta_and_outer_whitespace_are_preserved() -> None:
    row = _row("train")
    raw_reasoning = (
        "\nWe are asked to return JSON, and I should check the answer field, "
        "transport contract, and length before preserving March 2024 and the "
        "2 percent increase in industrial production together with the stated "
        "elevated uncertainty about the durability of the improvement.\n"
    )
    response = _provider_response(
        response_id="meta",
        content={"answer": REWRITE},
        reasoning=raw_reasoning,
    )
    candidate = _candidate_from_response(
        row,
        response,
        attempt="primary",
        tokenizer=FakeTokenizer(),
    )
    diagnostics = candidate.deterministic_validation["diagnostics"]
    assert candidate.teacher_response_analysis == raw_reasoning
    assert candidate.sft_response.startswith(raw_reasoning + "\n</think>\n")
    assert diagnostics["reasoning_meta_categories"]
    assert diagnostics["reasoning_meta_used_for_rejection"] is False
    assert diagnostics["reasoning_sanitization"] == "none"


def test_deterministic_repair_preserves_exact_reason_codes() -> None:
    reasons = [
        "teacher_response_analysis_meta:answer_key,answering,json",
        "numeric_multiset_mismatch",
    ]
    prompt = _repair_user_prompt(
        _row("train"),
        reasons,
    )
    payload = json.loads(prompt.split("\n\n", 1)[1])
    assert payload["repair_feedback"]["reason_codes"] == reasons
    assert "answer_key" in prompt
    assert "teacher_response_analysis_meta" in prompt


def test_reasoning_over_4096_total_tokens_is_allowed_and_recorded() -> None:
    reasoning = " ".join(["fidelity"] * 4200)
    candidate = _candidate_from_response(
        _row("train"),
        _provider_response(
            response_id="long-cot",
            content={"answer": REWRITE},
            reasoning=reasoning,
        ),
        attempt="primary",
        tokenizer=FakeTokenizer(),
    )
    assert candidate.teacher_response_analysis == reasoning
    diagnostics = candidate.deterministic_validation["diagnostics"]
    assert diagnostics["reasoning_words"] == 4200
    assert diagnostics["total_tokens"] > 4096
    assert diagnostics["total_token_limit"] is None
    assert diagnostics["total_token_used_for_rejection"] is False


def test_verbatim_source_sentence_and_near_copy_are_allowed() -> None:
    source = (
        "In March 2024, industrial production increased 2 percent. "
        "Uncertainty about the durability of the improvement remained elevated."
    )
    reused = (
        "In March 2024, industrial production increased 2 percent. "
        "Uncertainty about whether the improvement would endure remained "
        "elevated throughout the period covered by the assessment."
    )
    base = _row("train")
    prompt = render_user_prompt(source)
    row = PreparedRow(
        sample_id=base.sample_id,
        split=base.split,
        source_index=base.source_index,
        meeting_date=base.meeting_date,
        source_analysis=source,
        official_minutes=base.official_minutes,
        student_prompt=prompt,
        source_analysis_sha256=sha256_text(source),
        official_minutes_sha256=base.official_minutes_sha256,
        prompt_sha256=sha256_text(prompt),
        source_manifest_sha256=base.source_manifest_sha256,
        source_response_sha256=base.source_response_sha256,
    )
    candidate = _candidate_from_response(
        row,
        _provider_response(
            response_id="near-copy",
            content={"answer": reused},
            reasoning=REASONING,
        ),
        attempt="primary",
        tokenizer=FakeTokenizer(),
    )
    assert candidate.rewritten_minutes == reused
    assert candidate.deterministic_validation["diagnostics"][
        "near_copy_used_for_rejection"
    ] is False
    assert candidate.deterministic_validation["diagnostics"][
        "source_sentence_reuse_count"
    ] == 1


def test_whole_answer_equal_to_source_remains_minimal_rewrite_failure() -> None:
    source = (
        ANALYSIS
        + " The qualification continued to shape the assessment during the period."
    )
    prompt = render_user_prompt(source)
    base = _row("train")
    row = PreparedRow(
        sample_id=base.sample_id,
        split=base.split,
        source_index=base.source_index,
        meeting_date=base.meeting_date,
        source_analysis=source,
        official_minutes=base.official_minutes,
        student_prompt=prompt,
        source_analysis_sha256=sha256_text(source),
        official_minutes_sha256=base.official_minutes_sha256,
        prompt_sha256=sha256_text(prompt),
        source_manifest_sha256=base.source_manifest_sha256,
        source_response_sha256=base.source_response_sha256,
    )
    with pytest.raises(
        ContractError, match="rewritten_minutes_exactly_copies_source_analysis"
    ):
        _candidate_from_response(
            row,
            _provider_response(
                response_id="exact-source",
                content={"answer": source},
                reasoning=REASONING,
            ),
            attempt="primary",
            tokenizer=FakeTokenizer(),
        )


def test_validator_a_legacy_meta_issue_is_nonblocking() -> None:
    row = _row("train")
    candidate = _candidate_from_response(
        row,
        _provider_response(
            response_id="candidate",
            content={"answer": REWRITE},
            reasoning=(
                "We are asked to return JSON and should verify the answer field "
                "and transport length. March 2024, the 2 percent increase in "
                "industrial production, and the elevated uncertainty about the "
                "durability of the improvement all need to remain in the rewrite."
            ),
        ),
        attempt="primary",
        tokenizer=FakeTokenizer(),
    )
    payload = _validator_a_payload(
        "Verify:\n\n"
        + canonical_json(
            {
                "source_analysis": row.source_analysis,
                "teacher_response_analysis": candidate.teacher_response_analysis,
                "rewritten_minutes": candidate.rewritten_minutes,
            }
        ),
        passed=True,
    )
    payload["reasoning_issues"] = [
        {
            "reasoning_span": "We are asked to return JSON",
            "issue_type": "meta_discussion",
        },
        {
            "reasoning_span": "answer field and transport length",
            "issue_type": "unsupported_claim",
        },
    ]
    payload["reasoning_compatible"] = False
    payload["issues"] = ["reasoning contains prompt and JSON meta-discussion"]
    payload["overall_pass"] = False
    normalized, machine_pass, reasons = _validate_validator_a(
        row,
        candidate,
        _provider_response(response_id="validator-a", content=payload),
    )
    assert machine_pass is True
    assert reasons == []
    assert normalized["ignored_operational_reasoning_issues"]
    assert normalized["ignored_operational_issues"]
    assert normalized["reasoning_meta_used_for_rejection"] is False


def test_validator_a_lexical_overlap_complaint_is_nonblocking() -> None:
    row = _row("train")
    candidate = _candidate_from_response(
        row,
        _provider_response(
            response_id="candidate-overlap",
            content={"answer": REWRITE},
            reasoning=REASONING,
        ),
        attempt="primary",
        tokenizer=FakeTokenizer(),
    )
    payload = _validator_a_payload(
        "Verify:\n\n"
        + canonical_json(
            {
                "source_analysis": row.source_analysis,
                "teacher_response_analysis": candidate.teacher_response_analysis,
                "rewritten_minutes": candidate.rewritten_minutes,
            }
        ),
        passed=True,
    )
    payload["issues"] = ["rewrite is too similar to the source and is a verbatim copy"]
    payload["overall_pass"] = False
    normalized, machine_pass, reasons = _validate_validator_a(
        row,
        candidate,
        _provider_response(response_id="validator-a-overlap", content=payload),
    )
    assert machine_pass is True
    assert reasons == []
    assert normalized["ignored_lexical_overlap_issues"] == payload["issues"]


def test_validator_b_official_exact_copy_and_copy_complaint_are_audit_only() -> None:
    row = _row("train")
    row = PreparedRow(
        sample_id=row.sample_id,
        split=row.split,
        source_index=row.source_index,
        meeting_date=row.meeting_date,
        source_analysis=row.source_analysis,
        official_minutes=REWRITE,
        student_prompt=row.student_prompt,
        source_analysis_sha256=row.source_analysis_sha256,
        official_minutes_sha256=sha256_text(REWRITE),
        prompt_sha256=row.prompt_sha256,
        source_manifest_sha256=row.source_manifest_sha256,
        source_response_sha256=row.source_response_sha256,
    )
    candidate = _candidate_from_response(
        row,
        _provider_response(
            response_id="candidate-copy",
            content={"answer": REWRITE},
            reasoning=REASONING,
        ),
        attempt="primary",
        tokenizer=FakeTokenizer(),
    )
    scores = {
        "factual_consistency": 30,
        "claim_coverage": 25,
        "absence_of_unsupported_content": 20,
        "fomc_minutes_style": 15,
        "clarity_and_coherence": 5,
        "independent_rewriting": 0,
    }
    payload = {
        "official_claims": [
            {
                "official_span": REWRITE,
                "rewrite_evidence": [REWRITE],
                "verdict": "supported",
            }
        ],
        "rewrite_claims": [
            {
                "rewrite_span": REWRITE,
                "official_evidence": f"{REWRITE}; {REWRITE}",
                "verdict": "supported",
            }
        ],
        "scores": scores,
        "critical_errors": ["verbatim copy"],
        "overall_score": sum(scores.values()),
        "overall_pass": False,
    }
    normalized, report_complete, diagnostic_pass, reasons = _validate_validator_b(
        row,
        candidate,
        _provider_response(response_id="validator-b", content=payload),
    )
    assert report_complete is True
    assert diagnostic_pass is True
    assert reasons == []
    assert normalized["normalized_exact_copy"] is True
    assert normalized["official_claims"][0]["verdict"] == "preserved"
    assert normalized["official_claims"][0]["reported_verdict"] == "supported"
    assert normalized["official_claims"][0]["rewrite_evidence"] == [REWRITE]
    assert normalized["rewrite_claims"][0]["official_evidence"] == [
        REWRITE,
        REWRITE,
    ]
    assert normalized["ignored_lexical_overlap_errors"] == ["verbatim copy"]
    assert normalized["lexical_overlap_used_for_warning"] is False


def test_default_runtime_contracts_are_eight_and_no_coverage_floor() -> None:
    assert DEFAULT_CONCURRENCY == 8
    assert DEFAULT_PREFLIGHT_ROWS == 8
