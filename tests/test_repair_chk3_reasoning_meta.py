from __future__ import annotations

from jobs.generation.generate_chk3_sft_targets import (
    STUDENT_SYSTEM_PROMPT,
    DeepSeekTeacherResponse,
    canonical_json,
    render_user_prompt,
    sha256_text,
)
from jobs.generation.materialize_chk3_training_data import AcquisitionRow
from jobs.generation.repair_chk3_reasoning_meta import (
    PriorReleaseRow,
    _parse_analysis,
    identify_rows_to_reacquire,
)


class FakeTokenizer:
    def apply_chat_template(
        self, messages, *, tokenize: bool, add_generation_prompt: bool
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        assert messages[0] == {"role": "system", "content": STUDENT_SYSTEM_PROMPT}
        return "<bos>" + canonical_json(messages) + "<assistant><think>\n"

    def encode(self, text: str, *, add_special_tokens: bool):
        assert add_special_tokens is False
        return text.split()


def _teacher(reasoning: str, minutes: str, response_id: str) -> DeepSeekTeacherResponse:
    return DeepSeekTeacherResponse(
        analysis=reasoning,
        answer=minutes,
        evidence_ids=(),
        response_id=response_id,
        returned_model="deepseek-v4-pro",
        system_fingerprint="fp-fixed",
        finish_reason="stop",
        created=1,
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        raw_content=canonical_json({"answer": minutes}),
    )


def _rows(sample_id: str, reasoning: str) -> tuple[PriorReleaseRow, AcquisitionRow]:
    analysis = "Activity increased 2 percent in 2021."
    minutes = "Activity increased 2 percent in 2021."
    prompt = render_user_prompt(analysis)
    response = reasoning + "\n</think>\n" + minutes
    teacher = _teacher(reasoning, minutes, f"response-{sample_id}")
    manifest = {
        "analysis_mode": "unchanged",
        "target_mode": "acquisition_revalidated",
    }
    prior = PriorReleaseRow(
        sample_id=sample_id,
        source_split="train",
        source_index=0,
        analysis=analysis,
        prompt=prompt,
        reasoning=reasoning,
        minutes=minutes,
        response=response,
        manifest=manifest,
        teacher_response=teacher,
    )
    acquisition = AcquisitionRow(
        prepared={
            "sample_id": sample_id,
            "split": "train",
            "source_index": 0,
            "analysis": analysis,
            "source_response_sha256": sha256_text(f"source-{sample_id}"),
        },
        teacher={},
        sft={},
        manifest={},
    )
    return prior, acquisition


def test_parse_analysis_round_trips_structured_prompt() -> None:
    analysis = "Activity increased 2 percent in 2021."
    assert _parse_analysis(render_user_prompt(analysis), sample_id="sample") == analysis


def test_identify_rows_reacquires_meta_and_reuses_clean_reasoning() -> None:
    clean_reasoning = (
        "The source reports a measured increase and states its magnitude and year. "
        "Formal wording should preserve the direction, quantity, temporal reference, "
        "and neutral tone without adding a cause or attribution. The economic claim "
        "is a direct observation rather than an explanation, so the prose should keep "
        "the relationship factual and restrained. The stated increase remains the "
        "central movement, while the magnitude and observation year anchor its scope. "
        "No uncertainty, comparison with another period, named actor, or policy action "
        "appears in the source, and formal wording should not imply any of them."
    )
    meta_reasoning = (
        "You are the reasoning teacher and the required_dates field lists 2021. "
        "The final output should be one paragraph under 200 words."
    )
    clean, clean_source = _rows("clean", clean_reasoning)
    meta, meta_source = _rows("meta", meta_reasoning)
    meta = PriorReleaseRow(**{**meta.__dict__, "source_index": 1})
    meta_source = AcquisitionRow(
        prepared={**meta_source.prepared, "source_index": 1},
        teacher={},
        sft={},
        manifest={},
    )

    flagged, reasons = identify_rows_to_reacquire(
        {"train": [clean, meta], "eval": [], "test": []},
        source_by_id={"clean": clean_source, "meta": meta_source},
        tokenizer=FakeTokenizer(),
    )

    assert [row.sample_id for row in flagged["train"]] == ["meta"]
    assert set(reasons) == {"meta"}
    assert "roles" in reasons["meta"]["reasoning_meta_categories"]
    assert "inventory_contract" in reasons["meta"]["reasoning_meta_categories"]
    assert "length_contract" in reasons["meta"]["reasoning_meta_categories"]
