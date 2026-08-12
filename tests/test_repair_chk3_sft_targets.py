from __future__ import annotations

import json

from jobs.generation.generate_chk3_sft_targets import (
    PreparedRow,
    ProviderIdentityGuard,
    _cache_key,
    _validate_cached_payload,
    render_user_prompt,
    sha256_text,
)
from jobs.generation.repair_chk3_sft_targets import (
    REPAIR_SCHEMA,
    TARGETED_REPAIR_SYSTEM_PROMPT,
    repair_contract,
    repair_one,
    semantic_analysis,
    source_quantity_occurrences,
    targeted_user_prompt,
)
from jobs.retrain_v2.chk1.deepseek_teacher import (
    DeepSeekTeacherResponse,
    MockDeepSeekTeacherBackend,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return "\n".join(item["content"] for item in messages) + "\n<think>\n"

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()


def _row(analysis: str) -> PreparedRow:
    prompt = render_user_prompt(analysis)
    return PreparedRow(
        sample_id="chk3-residual-test",
        split="train",
        source_index=0,
        analysis=analysis,
        user_prompt=prompt,
        analysis_sha256=sha256_text(analysis),
        prompt_sha256=sha256_text(prompt),
        source_response_sha256="a" * 64,
    )


def test_quantity_inventory_excludes_calendar_and_evidence_digits() -> None:
    analysis = (
        "Activity rose 2 percent from $1.25 billion on August 5, 2021, "
        "to 1,250 million (ev-68309f2087d3c6e6c297d4cd)."
    )
    assert source_quantity_occurrences(analysis) == [
        "2 percent",
        "1.25 billion",
        "1,250 million",
    ]


def test_semantic_analysis_unwraps_provider_envelopes_and_short_evidence_ids() -> None:
    wrapped = json.dumps(
        {
            "reasoning_content": "Duplicate 2 percent planning.",
            "content": {
                "answer": "Activity increased 2 percent (ev-0316).",
                "evidence_ids": ["ev-0316"],
            },
        }
    )
    assert semantic_analysis(wrapped) == "Activity increased 2 percent."
    malformed = (
        '{"answer":"Activity increased 2 percent.",'
        '"evidence_ids":["ev-0'
    )
    assert semantic_analysis(malformed) == "Activity increased 2 percent."
    truncated = '{ "answer": "Activity increased 2 percent. The next'
    assert semantic_analysis(truncated) == "Activity increased 2 percent."


def test_targeted_prompt_has_occurrence_inventory_and_short_limits() -> None:
    row = _row("The rate was 5 percent and later remained at 5 percent in May 2021.")
    prompt = targeted_user_prompt(
        row, attempt=2, prior_errors=("missing_numbers:5",)
    )
    ledger = prompt.split("\n\n")[-1]
    assert 'Quantity occurrences: ["5 percent","5 percent"]' in ledger
    assert 'Date markers: ["2021","may"]' in ledger
    assert 'Silent correction notes for attempt 2: ["missing_numbers:5"]' in ledger
    assert "three to six concise prose sentences" in TARGETED_REPAIR_SYSTEM_PROMPT
    assert "required_dates" not in prompt
    assert "required_quantity_occurrences" not in prompt


def test_repair_contract_is_self_hashing() -> None:
    payload = repair_contract(
        code_sha256="b" * 64,
        original_contract_sha256="c" * 64,
        max_attempts=3,
    )
    assert payload["schema_version"] == REPAIR_SCHEMA
    assert len(payload["repair_contract_sha256"]) == 64


def test_targeted_acceptance_remains_valid_original_cache(tmp_path) -> None:
    analysis = "Activity increased 2 percent in 2021."
    row = _row(analysis)
    reasoning = " ".join(
        [
            "The increase, its exact magnitude, observation year, direction, and "
            "fidelity should remain unchanged in concise formal wording."
        ]
        * 7
    )
    response = DeepSeekTeacherResponse(
        analysis=reasoning,
        answer=analysis,
        evidence_ids=(),
        response_id="targeted-response-1",
        returned_model="deepseek-v4-pro",
        system_fingerprint="fp-test",
        finish_reason="stop",
        created=1,
        usage={},
        raw_content=json.dumps({"answer": analysis}),
    )
    backend = MockDeepSeekTeacherBackend([response])
    contract = repair_contract(
        code_sha256="b" * 64,
        original_contract_sha256="c" * 64,
        max_attempts=1,
    )
    original_code_sha = "d" * 64
    result = repair_one(
        row,
        output_root=tmp_path,
        tokenizer=FakeTokenizer(),
        backend=backend,
        identity_guard=ProviderIdentityGuard(),
        environment={},
        original_code_sha256=original_code_sha,
        repair_contract_payload=contract,
        max_attempts=1,
    )
    assert result.accepted is True
    assert result.payload["targeted_repair"]["schema_version"] == REPAIR_SCHEMA
    key = _cache_key(row, code_sha256=original_code_sha)
    validated = _validate_cached_payload(
        result.payload,
        row=row,
        cache_key=key,
        code_sha256=original_code_sha,
    )
    assert validated["sample_id"] == row.sample_id
