from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
import yaml

from jobs.generation.generate_chk3_sft_targets import (
    DEFAULT_OUTPUT_ROOT,
    EXPECTED_SPLIT_COUNTS,
    MANIFEST_SCHEMA_VERSION,
    MIN_REASONING_TOKENS,
    MAX_REASONING_TOKENS,
    PREPARED_SCHEMA_VERSION,
    PROMPT_CONTRACT_SCHEMA_VERSION,
    REPO_ROOT,
    STUDENT_SYSTEM_PROMPT,
    TEACHER_CACHE_SCHEMA_VERSION,
    TEACHER_REPAIR_SYSTEM_PROMPT,
    TEACHER_SYSTEM_PROMPT,
    Chk3DataError,
    DeepSeekTeacherResponse,
    OutputContractError,
    PreparedRow,
    ProviderIdentityGuard,
    build_sft_completion,
    canonical_json,
    generate_one_target,
    prepare_chk1_release,
    prompt_contract,
    render_user_prompt,
    run_generation,
    sha256_file,
    sha256_text,
    validate_teacher_target,
)


DEFAULT_TEST_REASONING = """\
The source reports a measured change and identifies its magnitude and
observation period. A faithful rewrite should retain the direction, numeric
quantities, temporal references, comparison structure, and degree of
uncertainty. Formal Minutes wording should remain neutral and concise while
avoiding unsupported causes, actors, policy implications, or decisions. The
prose should preserve the relationship among the supplied facts without
adding interpretation beyond the source, changing the stated values, or
weakening the source's uncertainty.
""".strip()


class FakeTokenizer:
    def apply_chat_template(
        self, messages, *, tokenize: bool, add_generation_prompt: bool
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        assert messages == [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": messages[1]["content"]},
        ]
        return (
            "<bos>"
            + "".join(f"<{item['role']}>{item['content']}" for item in messages)
            + "<assistant><think>\n"
        )

    def encode(self, text: str, *, add_special_tokens: bool):
        assert add_special_tokens is False
        return text.split()


class SequenceBackend:
    def __init__(self, responses: list[DeepSeekTeacherResponse]):
        self._responses = list(responses)
        self._lock = threading.Lock()
        self.requests: list[dict] = []

    def generate(self, *, config, system_prompt, user_prompt, environment):
        del environment
        with self._lock:
            self.requests.append(
                {
                    "contract": config.contract(),
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )
            if not self._responses:
                raise AssertionError("unexpected teacher request")
            return self._responses.pop(0)


def _response(
    *,
    response_id: str,
    reasoning: str = DEFAULT_TEST_REASONING,
    answer: str = "In 2021, activity increased 2 percent.",
    fingerprint: str = "fp-fixed",
    raw_content: str | None = None,
) -> DeepSeekTeacherResponse:
    content = (
        raw_content if raw_content is not None else canonical_json({"answer": answer})
    )
    return DeepSeekTeacherResponse(
        analysis=reasoning,
        answer=answer,
        evidence_ids=(),
        response_id=response_id,
        returned_model="deepseek-v4-pro",
        system_fingerprint=fingerprint,
        finish_reason="stop",
        created=1,
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        raw_content=content,
    )


def _prepared(sample_id: str = "sample-1", *, index: int = 0) -> PreparedRow:
    analysis = "Activity increased 2 percent in 2021."
    prompt = render_user_prompt(analysis)
    return PreparedRow(
        sample_id=sample_id,
        split="train",
        source_index=index,
        analysis=analysis,
        user_prompt=prompt,
        analysis_sha256=sha256_text(analysis),
        prompt_sha256=sha256_text(prompt),
        source_response_sha256=sha256_text(f"source-{sample_id}"),
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )


def _build_chk1_release(root: Path, counts: dict[str, int]) -> Path:
    release = root / "chk1-release"
    split_files: dict[str, dict] = {}
    manifest_files: dict[str, dict] = {}
    for split, count in counts.items():
        sft_rows: list[dict] = []
        manifests: list[dict] = []
        for index in range(count):
            sample_id = f"{split}-{index:05d}"
            response = (
                f"Reasoning for {sample_id}.\n</think>\n"
                f"Activity increased 2 percent in 2021 for sample {index}."
            )
            sft_rows.append(
                {
                    "prompt": f"privileged source prompt {sample_id}",
                    "provided_data": f"privileged fact card {sample_id}",
                    "response": response,
                }
            )
            manifests.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "response_sha256": sha256_text(response),
                }
            )
        sft_path = release / "sft" / f"{split}.jsonl"
        manifest_path = release / "manifests" / f"{split}.jsonl"
        _write_jsonl(sft_path, sft_rows)
        _write_jsonl(manifest_path, manifests)
        split_files[split] = {
            "path": f"sft/{split}.jsonl",
            "rows": count,
            "sha256": sha256_file(sft_path),
        }
        manifest_files[split] = {
            "path": f"manifests/{split}.jsonl",
            "rows": count,
            "sha256": sha256_file(manifest_path),
        }
    handoff = release / "handoff.json"
    handoff.write_text(
        json.dumps(
            {
                "schema_version": "chk1-local-data-handoff-v1",
                "release_id": "chk1-test",
                "quality_status": "passed",
                "immutable": True,
                "split_counts": counts,
                "split_files": split_files,
                "manifest_files": manifest_files,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return handoff


def test_full_chk1_population_is_projected_to_answer_only(tmp_path: Path) -> None:
    handoff = _build_chk1_release(tmp_path, dict(EXPECTED_SPLIT_COUNTS))
    output = tmp_path / "chk3"
    prepared, summary = prepare_chk1_release(handoff, output_root=output)

    assert summary["total_rows"] == 2072
    assert summary["split_counts"] == EXPECTED_SPLIT_COUNTS
    assert summary["chk2_dependency"] is False
    assert {
        split: len(rows) for split, rows in prepared.items()
    } == EXPECTED_SPLIT_COUNTS

    first = prepared["train"][0]
    assert first.analysis == "Activity increased 2 percent in 2021 for sample 0."
    assert json.loads(first.user_prompt.split("\n\n", 1)[1]) == {
        "analysis": first.analysis
    }
    assert "privileged source prompt" not in first.user_prompt
    assert "privileged fact card" not in first.user_prompt
    assert "provided_data" not in first.as_dict()
    assert not any("chk2" in key.casefold() for key in first.as_dict())


def test_native_deepseek_mapping_keeps_reasoning_and_omits_legacy_tags() -> None:
    analysis = "Activity increased 2 percent in 2021."
    response = _response(response_id="r1")
    target = validate_teacher_target(
        response=response,
        analysis=analysis,
        user_prompt=render_user_prompt(analysis),
        tokenizer=FakeTokenizer(),
    )

    assert target.completion == (
        f"{DEFAULT_TEST_REASONING}\n</think>\nIn 2021, activity increased 2 percent."
    )
    assert target.completion.count("</think>") == 1
    assert "<think>" not in target.completion
    assert "<answer>" not in target.completion
    assert target.reasoning_was_sanitized is False
    assert target.reasoning_removed_segments == 0
    assert build_sft_completion(target.reasoning, target.minutes) == target.completion


def test_new_number_or_date_is_rejected() -> None:
    analysis = "Activity increased 2 percent in 2021."
    response = _response(
        response_id="r-new-fact",
        answer="In March 2022, activity increased 3 percent.",
    )
    with pytest.raises(OutputContractError) as exc_info:
        validate_teacher_target(
            response=response,
            analysis=analysis,
            user_prompt=render_user_prompt(analysis),
            tokenizer=FakeTokenizer(),
        )
    assert "missing_numbers" in str(exc_info.value)
    assert "unsupported_numbers" in str(exc_info.value)
    assert "unsupported_dates" in str(exc_info.value)


def test_calendar_years_are_dates_and_may_repeat_without_number_errors() -> None:
    analysis = "Activity rose from December 2008 to February 2009."
    answer = (
        "Activity rose from December 2008 through February 2009 over the 2009 period."
    )
    target = validate_teacher_target(
        response=_response(response_id="r-repeat-year", answer=answer),
        analysis=analysis,
        user_prompt=render_user_prompt(analysis),
        tokenizer=FakeTokenizer(),
    )
    assert target.minutes == answer

    with pytest.raises(OutputContractError, match="missing_dates:2008"):
        validate_teacher_target(
            response=_response(
                response_id="r-missing-year",
                answer="Activity rose from December to February 2009.",
            ),
            analysis=analysis,
            user_prompt=render_user_prompt(analysis),
            tokenizer=FakeTokenizer(),
        )


def test_month_day_dates_may_repeat_but_cannot_be_added_or_omitted() -> None:
    analysis = "Balances were $718 billion on August 5 after rising on July 29."
    answer = (
        "Balances were $718 billion on August 5 after rising on July 29; "
        "the August 5 level was the latest observation."
    )
    target = validate_teacher_target(
        response=_response(response_id="r-repeat-month-day", answer=answer),
        analysis=analysis,
        user_prompt=render_user_prompt(analysis),
        tokenizer=FakeTokenizer(),
    )
    assert target.minutes == answer

    with pytest.raises(OutputContractError, match="missing_dates:august-5"):
        validate_teacher_target(
            response=_response(
                response_id="r-missing-month-day",
                answer="Balances were $718 billion in August after rising on July 29.",
            ),
            analysis=analysis,
            user_prompt=render_user_prompt(analysis),
            tokenizer=FakeTokenizer(),
        )

    with pytest.raises(OutputContractError, match="unsupported_dates:august-6"):
        validate_teacher_target(
            response=_response(
                response_id="r-added-month-day",
                answer="Balances were $718 billion on August 6 after rising on July 29.",
            ),
            analysis=analysis,
            user_prompt=render_user_prompt(analysis),
            tokenizer=FakeTokenizer(),
        )


@pytest.mark.parametrize(
    ("analysis", "answer"),
    [
        (
            "Housing starts were 510 thousand units.",
            "Housing starts were 510,000 units.",
        ),
        (
            "Housing starts were 1,256,000 units.",
            "Housing starts were 1.256 million units.",
        ),
        ("The rate was 0.30 percent.", "The rate was 30 basis points."),
        ("The rate was 5.25 percent.", "The rate was 5-1/4 percent."),
        ("The amount was $40,900 million.", "The amount was $40.9 billion."),
    ],
)
def test_exact_number_unit_and_form_conversions_are_allowed(
    analysis: str, answer: str
) -> None:
    target = validate_teacher_target(
        response=_response(response_id="r-convert", answer=answer),
        analysis=analysis,
        user_prompt=render_user_prompt(analysis),
        tokenizer=FakeTokenizer(),
    )
    assert target.minutes == answer


def test_rounded_derived_missing_and_repeated_numbers_are_rejected() -> None:
    cases = [
        (
            "Housing starts were 1.256 million units.",
            "Housing starts were 1.26 million units.",
            ("missing_numbers", "unsupported_numbers"),
        ),
        (
            "The rate rose from 5 percent to 5.25 percent.",
            "The rate rose from 5 percent to 5.25 percent, a 25 basis point increase.",
            ("unsupported_numbers",),
        ),
        (
            "The rate was 5 percent and later remained at 5 percent.",
            "The rate remained at 5 percent.",
            ("missing_numbers",),
        ),
    ]
    for index, (analysis, answer, expected_codes) in enumerate(cases):
        with pytest.raises(OutputContractError) as exc_info:
            validate_teacher_target(
                response=_response(response_id=f"r-bad-number-{index}", answer=answer),
                analysis=analysis,
                user_prompt=render_user_prompt(analysis),
                tokenizer=FakeTokenizer(),
            )
        for code in expected_codes:
            assert code in str(exc_info.value)


def test_evidence_id_digits_do_not_create_missing_numbers() -> None:
    analysis = "Activity increased 2 percent (ev-42c51d)."
    answer = "Activity increased 2 percent."
    target = validate_teacher_target(
        response=_response(response_id="r-short-citation", answer=answer),
        analysis=analysis,
        user_prompt=render_user_prompt(analysis),
        tokenizer=FakeTokenizer(),
    )
    assert target.minutes == answer


def test_evidence_id_digits_do_not_create_dates() -> None:
    analysis = "Inflation stayed contained in 2009 (ev-68309f2087d3c6e6c297d4cd)."
    answer = "Inflation stayed contained in 2009."
    target = validate_teacher_target(
        response=_response(response_id="r-evidence-year", answer=answer),
        analysis=analysis,
        user_prompt=render_user_prompt(analysis),
        tokenizer=FakeTokenizer(),
    )
    assert target.minutes == answer


def test_title_case_may_is_a_month_and_lowercase_may_is_modal() -> None:
    analysis = "Activity may slow in May."
    target = validate_teacher_target(
        response=_response(response_id="r-may-ok", answer="Activity may slow in May."),
        analysis=analysis,
        user_prompt=render_user_prompt(analysis),
        tokenizer=FakeTokenizer(),
    )
    assert target.minutes.endswith("May.")

    with pytest.raises(OutputContractError, match="missing_dates:may"):
        validate_teacher_target(
            response=_response(
                response_id="r-may-missing", answer="Activity may slow."
            ),
            analysis=analysis,
            user_prompt=render_user_prompt(analysis),
            tokenizer=FakeTokenizer(),
        )

    modal_only = "Activity may slow."
    with pytest.raises(OutputContractError, match="unsupported_dates:may"):
        validate_teacher_target(
            response=_response(
                response_id="r-may-added", answer="Activity slowed in May."
            ),
            analysis=modal_only,
            user_prompt=render_user_prompt(modal_only),
            tokenizer=FakeTokenizer(),
        )


@pytest.mark.parametrize(
    "answer",
    [
        "The staff noted that activity increased 2 percent in 2021.",
        "Participants observed that activity increased 2 percent in 2021.",
        "Information reviewed at the meeting showed activity increased 2 percent in 2021.",
        "The Committee decided that activity increased 2 percent in 2021.",
    ],
)
def test_new_attributions_are_rejected(answer: str) -> None:
    analysis = "Activity increased 2 percent in 2021."
    with pytest.raises(OutputContractError, match="unsupported_attributions"):
        validate_teacher_target(
            response=_response(response_id="r-attribution", answer=answer),
            analysis=analysis,
            user_prompt=render_user_prompt(analysis),
            tokenizer=FakeTokenizer(),
        )


def test_source_attribution_may_be_preserved() -> None:
    analysis = "The staff noted that activity increased 2 percent in 2021."
    target = validate_teacher_target(
        response=_response(
            response_id="r-attribution-ok",
            answer="The staff reported that activity increased 2 percent in 2021.",
        ),
        analysis=analysis,
        user_prompt=render_user_prompt(analysis),
        tokenizer=FakeTokenizer(),
    )
    assert target.minutes.startswith("The staff")


@pytest.mark.parametrize(
    "reasoning",
    [
        "Return JSON containing the rewritten observation.",
        "The API contract requires a response format.",
        "Use the answer key and answer field.",
        "The prompt asks for the final answer.",
        "The validation error identifies an output contract issue.",
        "We are asked to produce exactly one paragraph.",
        "We are given an analysis and need to rewrite it.",
        "Let's craft the answer.",
        "No reasoning in the content.",
        "The modal verb may is not a month.",
        "You are the DeepSeek V4 Pro reasoning teacher for this dataset.",
        "The user message says to preserve the quantities.",
        "The prompt does not specify a meeting date.",
        "The required_dates field lists May and June.",
        "Check required_quantity_occurrences before writing.",
        "The word count is safely under 200 words.",
        "The output should contain one paragraph.",
        "The reasoning should be separate from the answer.",
        "This is confusing because the answer is a separate field.",
    ],
)
def test_reasoning_transport_and_instruction_meta_is_rejected(reasoning: str) -> None:
    analysis = "Activity increased 2 percent in 2021."
    with pytest.raises(OutputContractError, match="empty_reasoning_content"):
        validate_teacher_target(
            response=_response(response_id="r-meta", reasoning=reasoning),
            analysis=analysis,
            user_prompt=render_user_prompt(analysis),
            tokenizer=FakeTokenizer(),
        )


def test_transport_chatter_is_removed_from_otherwise_useful_reasoning() -> None:
    analysis = "Activity increased 2 percent in 2021."
    raw_reasoning = (
        "We are asked to rewrite the analysis.\n\n"
        "The increase, its magnitude, and the observation year should remain "
        "unchanged in formal wording.\n\n"
        "Return JSON using the answer key."
    )
    target = validate_teacher_target(
        response=_response(response_id="r-sanitize", reasoning=raw_reasoning),
        analysis=analysis,
        user_prompt=render_user_prompt(analysis),
        tokenizer=FakeTokenizer(),
        min_reasoning_tokens=1,
    )
    assert target.reasoning == (
        "The increase, its magnitude, and the observation year should remain "
        "unchanged in formal wording."
    )
    assert target.reasoning_was_sanitized is True
    assert target.reasoning_removed_segments == 2


def test_economic_output_is_not_mistaken_for_reasoning_meta() -> None:
    analysis = "Economic output increased 2 percent in 2021."
    target = validate_teacher_target(
        response=_response(
            response_id="r-economic-output",
            reasoning="Economic output and its direction should remain unchanged.",
            answer="Economic output increased 2 percent in 2021.",
        ),
        analysis=analysis,
        user_prompt=render_user_prompt(analysis),
        tokenizer=FakeTokenizer(),
        min_reasoning_tokens=1,
    )
    assert target.reasoning.startswith("Economic output")


def test_student_loan_economic_content_is_not_mistaken_for_a_role() -> None:
    analysis = "Student loan balances increased 2 percent in 2021."
    target = validate_teacher_target(
        response=_response(
            response_id="r-student-loans",
            reasoning=(
                "Student loan balances, their direction, magnitude, and observation "
                "year should remain unchanged in formal wording."
            ),
            answer="Student loan balances increased 2 percent in 2021.",
        ),
        analysis=analysis,
        user_prompt=render_user_prompt(analysis),
        tokenizer=FakeTokenizer(),
        min_reasoning_tokens=1,
    )
    assert target.reasoning.startswith("Student loan balances")


def test_reasoning_has_an_independent_token_budget() -> None:
    row = _prepared()
    with pytest.raises(
        OutputContractError, match=rf"reasoning_tokens:5<{MIN_REASONING_TOKENS}"
    ):
        validate_teacher_target(
            response=_response(
                response_id="r-reasoning-short",
                reasoning="source relationship wording fidelity check",
            ),
            analysis=row.analysis,
            user_prompt=row.user_prompt,
            tokenizer=FakeTokenizer(),
        )

    with pytest.raises(OutputContractError, match="reasoning_tokens"):
        validate_teacher_target(
            response=_response(
                response_id="r-reasoning-long",
                reasoning="source relationship wording fidelity check",
            ),
            analysis=row.analysis,
            user_prompt=row.user_prompt,
            tokenizer=FakeTokenizer(),
            min_reasoning_tokens=1,
            max_reasoning_tokens=3,
        )


def test_reasoning_cannot_repeat_the_complete_analysis_or_final_draft() -> None:
    analysis = (
        "Activity increased 2 percent in 2021, while uncertainty about the "
        "durability of that increase remained elevated."
    )
    answer = (
        "Activity increased 2 percent in 2021, although uncertainty about "
        "the durability of the increase remained elevated."
    )
    sanitized = validate_teacher_target(
        response=_response(
            response_id="r-repeat-analysis",
            reasoning=analysis + " Formal wording should retain the uncertainty.",
            answer=answer,
        ),
        analysis=analysis,
        user_prompt=render_user_prompt(analysis),
        tokenizer=FakeTokenizer(),
        min_reasoning_tokens=1,
    )
    assert sanitized.reasoning == "Formal wording should retain the uncertainty."
    assert sanitized.reasoning_was_sanitized is True

    with pytest.raises(OutputContractError, match="reasoning_tokens"):
        validate_teacher_target(
            response=_response(
                response_id="r-repeat-draft",
                reasoning=f"{answer} Wording check. {answer}",
                answer=answer,
            ),
            analysis=analysis,
            user_prompt=render_user_prompt(analysis),
            tokenizer=FakeTokenizer(),
        )


def test_internal_evidence_ids_are_input_only_and_rejected_in_minutes() -> None:
    evidence_id = "ev-7daf73f5321fe1b5b3f753a1"
    analysis = f"Activity increased 2 percent in 2021 ({evidence_id})."
    response = _response(
        response_id="r-citation",
        answer=f"In 2021, activity increased 2 percent ({evidence_id}).",
    )
    with pytest.raises(
        OutputContractError, match="minutes_contains_internal_evidence_id"
    ):
        validate_teacher_target(
            response=response,
            analysis=analysis,
            user_prompt=render_user_prompt(analysis),
            tokenizer=FakeTokenizer(),
        )


def test_one_v4_pro_repair_maps_reasoning_and_answer(tmp_path: Path) -> None:
    row = _prepared()
    primary = _response(
        response_id="primary",
        reasoning="",
        raw_content=canonical_json(
            {"answer": "In 2021, activity increased 2 percent."}
        ),
    )
    repair = _response(response_id="repair")
    backend = SequenceBackend([primary, repair])
    code_sha = sha256_file(REPO_ROOT / "jobs/generation/generate_chk3_sft_targets.py")

    result = generate_one_target(
        row,
        output_root=tmp_path / "output",
        tokenizer=FakeTokenizer(),
        backend=backend,
        identity_guard=ProviderIdentityGuard(),
        environment={},
        code_sha256=code_sha,
    )

    assert result["status"] == "accepted"
    assert result["attempt"] == "repair"
    assert result["target"]["completion"].endswith(
        "</think>\nIn 2021, activity increased 2 percent."
    )
    assert len(backend.requests) == 2
    assert backend.requests[0]["system_prompt"] == TEACHER_SYSTEM_PROMPT
    assert backend.requests[1]["system_prompt"] == TEACHER_REPAIR_SYSTEM_PROMPT
    assert "output_contract_errors" in backend.requests[1]["user_prompt"]


def test_resume_uses_immutable_cache_without_teacher_call(tmp_path: Path) -> None:
    row = _prepared()
    prepared = {"train": [row], "eval": [], "test": []}
    output = tmp_path / "output"
    first_backend = SequenceBackend([_response(response_id="first")])
    first = run_generation(
        prepared,
        output_root=output,
        source_handoff_sha256="a" * 64,
        tokenizer=FakeTokenizer(),
        backend=first_backend,
        environment={},
        concurrency=1,
    )
    assert first["status"] == "complete"

    no_call_backend = SequenceBackend([])
    resumed = run_generation(
        prepared,
        output_root=output,
        source_handoff_sha256="a" * 64,
        tokenizer=FakeTokenizer(),
        backend=no_call_backend,
        environment={},
        concurrency=1,
        resume=True,
    )
    assert resumed["status"] == "complete"
    assert no_call_backend.requests == []
    sft = json.loads((output / "sft/train.jsonl").read_text().strip())
    assert set(sft) == {"prompt", "response"}


def test_provider_fingerprint_drift_stops_release(tmp_path: Path) -> None:
    rows = [_prepared("sample-1", index=0), _prepared("sample-2", index=1)]
    backend = SequenceBackend(
        [
            _response(response_id="first", fingerprint="fp-one"),
            _response(response_id="second", fingerprint="fp-two"),
        ]
    )
    summary = run_generation(
        {"train": rows, "eval": [], "test": []},
        output_root=tmp_path / "output",
        source_handoff_sha256="b" * 64,
        tokenizer=FakeTokenizer(),
        backend=backend,
        environment={},
        concurrency=1,
    )
    assert summary["status"] == "incomplete"
    assert summary["total_accepted"] == 1
    failures = [
        json.loads(line)
        for line in (tmp_path / "output/failures.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert "ModelDriftError" in failures[0]["error"]


def test_total_token_overflow_is_rejected_without_truncation() -> None:
    row = _prepared()
    response = _response(response_id="too-long", reasoning="word " * 50)
    with pytest.raises(OutputContractError, match="student_total_tokens"):
        validate_teacher_target(
            response=response,
            analysis=row.analysis,
            user_prompt=row.user_prompt,
            tokenizer=FakeTokenizer(),
            max_length=10,
        )


def test_invalid_chk1_split_counts_fail_closed(tmp_path: Path) -> None:
    handoff = _build_chk1_release(tmp_path, {"train": 1, "eval": 1, "test": 1})
    with pytest.raises(Chk3DataError, match="split counts"):
        prepare_chk1_release(handoff, output_root=tmp_path / "out")


def test_v2_contract_cache_and_default_output_are_isolated_from_v1() -> None:
    contract = prompt_contract(code_sha256="a" * 64)
    assert DEFAULT_OUTPUT_ROOT.name == "deepseek_v4_pro_v2"
    assert PREPARED_SCHEMA_VERSION.endswith("-v2")
    assert TEACHER_CACHE_SCHEMA_VERSION.endswith("-v2")
    assert MANIFEST_SCHEMA_VERSION.endswith("-v2")
    assert PROMPT_CONTRACT_SCHEMA_VERSION.endswith("-v2")
    assert contract["schema_version"] == PROMPT_CONTRACT_SCHEMA_VERSION
    assert contract["validation"]["min_reasoning_tokens"] == MIN_REASONING_TOKENS
    assert contract["validation"]["max_reasoning_tokens"] == MAX_REASONING_TOKENS
    assert contract["validation"]["truncation"] is False


def test_chk3_yaml_uses_exact_student_system_prompt() -> None:
    config_path = REPO_ROOT / "configs/retrain_v2/chk3_minutes_sft.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["system_prompt"] == STUDENT_SYSTEM_PROMPT
