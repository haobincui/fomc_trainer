from __future__ import annotations

import json
from copy import deepcopy

import pytest

from jobs.retrain_v2.chk1.contracts import canonical_json
from jobs.retrain_v2.chk1.prompt_projection import (
    ANALYSIS_GRPO_SYSTEM_PROMPT,
    GRPO_PROMPT_TOKEN_LIMIT,
    GRPO_USER_PREAMBLE,
    PromptProjectionError,
    build_sft_token_auditor,
    project_prompt_to_budget,
    project_preteacher_inputs,
    projection_contract_sha256,
    render_grpo_user_prompt,
    safe_grpo_fact_card,
    safe_model_fact_card,
)
from jobs.retrain_v2.sft_training_budget import build_training_sft_token_auditor
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v2 import (
    validate_judge_evidence,
)


DIGEST = "a" * 64
CUTOFF = "2019-01-29T23:59:59Z"


def _evidence(
    evidence_id: str,
    *,
    series: str,
    kind: str,
    observation: str,
    operands: list[str] | None = None,
) -> dict:
    return {
        "evidence_id": evidence_id,
        "source_id": f"source-{series}",
        "series_id": series,
        "metric": f"metric-{series}",
        "fact_kind": kind,
        "value": "1.0",
        "units": "Percent",
        "observation_date": observation,
        "availability_upper_bound_ts": "2019-01-29T05:59:59Z",
        "cutoff_ts": CUTOFF,
        "source_sha256": DIGEST,
        "operand_evidence_ids": operands or [],
    }


def _lineage(item: dict) -> dict:
    return {
        "evidence_id": item["evidence_id"],
        "source_id": item["source_id"],
        "source_sha256": item["source_sha256"],
        "cutoff_ts": item["cutoff_ts"],
        "evidence_sha256": DIGEST,
        "raw_sha256": DIGEST,
        "request_id": DIGEST,
        "snapshot_manifest_payload_sha256": DIGEST,
        "registry_sha256": DIGEST,
    }


def _bundle() -> tuple[dict, list[dict]]:
    evidence = [
        _evidence(
            "E-old",
            series="SERIES_A",
            kind="recent_observation",
            observation="2018-10-01",
        ),
        _evidence(
            "E-latest-a",
            series="SERIES_A",
            kind="latest",
            observation="2018-12-01",
        ),
        _evidence(
            "E-derived",
            series="SERIES_A",
            kind="change_3m",
            observation="2018-12-01",
            operands=["E-old", "E-latest-a"],
        ),
        _evidence(
            "E-latest-b",
            series="SERIES_B",
            kind="latest",
            observation="2018-11-01",
        ),
    ]
    fact = {
        "schema_version": "chk1-point-in-time-fact-card-v1",
        "sample_id": "chk1-analysis-2019-01-29-synthetic-topic",
        "canonical_key": {
            "meeting_date": "2019-01-29",
            "atomic_topic": "Synthetic Topic",
        },
        "meeting_date": "2019-01-29",
        "atomic_topic": "Synthetic Topic",
        "cutoff_ts": CUTOFF,
        "evidence": evidence,
    }
    return fact, [_lineage(item) for item in evidence]


def _renderer(fact_card: dict, provided_data: str) -> str:
    assert provided_data == canonical_json(fact_card)
    return "Synthetic instruction.\n\n" + provided_data


def _character_counter(system: str, prompt: str) -> int:
    return len(system) + len(prompt)


def _project(
    fact: dict,
    lineage: list[dict],
    *,
    budget: int,
    required: tuple[str, ...] = (),
    preferred: tuple[str, ...] = (),
):
    return project_prompt_to_budget(
        projection_name="synthetic",
        fact_card=fact,
        evidence_lineage=lineage,
        system_prompt="Synthetic system.",
        max_prompt_tokens=budget,
        prompt_renderer=_renderer,
        chat_token_counter=_character_counter,
        required_evidence_ids=required,
        preferred_evidence_ids=preferred,
    )


def _rendered_size(fact: dict) -> int:
    provided = canonical_json(fact)
    return _character_counter("Synthetic system.", _renderer(fact, provided))


def test_full_projection_preserves_whole_evidence_and_exact_prompt_binding() -> None:
    fact, lineage = _bundle()
    result = _project(fact, lineage, budget=100_000)

    assert result.prompt.count(result.provided_data) == 1
    assert result.provided_data == canonical_json(result.fact_card)
    assert len(result.fact_card["evidence"]) == 4
    assert len(result.evidence_lineage) == 4
    assert result.attestation["input_truncated"] is False
    assert result.attestation["string_truncation"] is False
    assert result.attestation["projected_evidence_count"] == 4
    assert len(result.attestation["evidence_bindings"]) == 4
    assert {
        binding["evidence_id"] for binding in result.attestation["evidence_bindings"]
    } == {item["evidence_id"] for item in result.fact_card["evidence"]}
    assert all(
        len(binding["evidence_sha256"]) == 64 and len(binding["lineage_sha256"]) == 64
        for binding in result.attestation["evidence_bindings"]
    )
    assert len(projection_contract_sha256()) == 64


def test_required_derived_evidence_retains_transitive_operand_closure() -> None:
    fact, lineage = _bundle()
    required_ids = {"E-old", "E-latest-a", "E-derived"}
    required_fact = deepcopy(fact)
    required_fact["evidence"] = [
        item for item in fact["evidence"] if item["evidence_id"] in required_ids
    ]
    budget = _rendered_size(required_fact)

    result = _project(
        fact,
        lineage,
        budget=budget,
        required=("E-derived",),
    )

    selected = {item["evidence_id"] for item in result.fact_card["evidence"]}
    assert selected == required_ids
    assert {item["evidence_id"] for item in result.evidence_lineage} == required_ids
    assert result.prompt_tokens <= budget
    assert result.attestation["source_evidence_count"] == 4
    assert result.attestation["projected_evidence_count"] == 3


def test_required_closure_overflow_fails_closed() -> None:
    fact, lineage = _bundle()
    one = deepcopy(fact)
    one["evidence"] = [fact["evidence"][0]]
    with pytest.raises(PromptProjectionError, match="required evidence dependency"):
        _project(
            fact,
            lineage,
            budget=_rendered_size(one),
            required=("E-derived",),
        )


def test_unknown_required_id_and_dependency_cycle_fail_closed() -> None:
    fact, lineage = _bundle()
    with pytest.raises(PromptProjectionError, match="unknown evidence ID"):
        _project(fact, lineage, budget=100_000, required=("unknown",))

    cyclic = deepcopy(fact)
    cyclic["evidence"][0]["operand_evidence_ids"] = ["E-derived"]
    with pytest.raises(PromptProjectionError, match="dependency cycle"):
        _project(cyclic, lineage, budget=100)


def test_lineage_binding_mismatch_fails_closed() -> None:
    fact, lineage = _bundle()
    lineage[0]["source_sha256"] = "b" * 64
    with pytest.raises(PromptProjectionError, match="source_sha256 mismatch"):
        _project(fact, lineage, budget=100_000)


def test_renderer_must_embed_exact_provided_data_once() -> None:
    fact, lineage = _bundle()

    def bad_renderer(_fact: dict, _provided: str) -> str:
        return "No evidence is embedded."

    with pytest.raises(PromptProjectionError, match="byte-for-byte exactly once"):
        project_prompt_to_budget(
            projection_name="bad",
            fact_card=fact,
            evidence_lineage=lineage,
            system_prompt="system",
            max_prompt_tokens=10_000,
            prompt_renderer=bad_renderer,
            chat_token_counter=_character_counter,
        )


def test_no_complete_evidence_object_fit_fails_closed() -> None:
    fact, lineage = _bundle()
    with pytest.raises(PromptProjectionError, match="no complete evidence"):
        _project(fact, lineage, budget=1)


def test_projection_is_deterministic_under_source_list_permutation() -> None:
    fact, lineage = _bundle()
    reversed_fact = deepcopy(fact)
    reversed_fact["evidence"] = list(reversed(reversed_fact["evidence"]))
    first = _project(fact, lineage, budget=100_000)
    second = _project(reversed_fact, list(reversed(lineage)), budget=100_000)
    assert first.prompt == second.prompt
    assert first.provided_data == second.provided_data
    assert first.evidence_lineage == second.evidence_lineage


def test_grpo_projection_removes_meeting_identity_and_matches_reward_evidence() -> None:
    fact, lineage = _bundle()
    safe = safe_grpo_fact_card(fact)
    result = project_prompt_to_budget(
        projection_name="analysis_grpo",
        fact_card=safe,
        evidence_lineage=lineage,
        system_prompt=ANALYSIS_GRPO_SYSTEM_PROMPT,
        max_prompt_tokens=GRPO_PROMPT_TOKEN_LIMIT,
        prompt_renderer=render_grpo_user_prompt,
        chat_token_counter=lambda _system, _prompt: 100,
    )

    assert result.prompt.startswith(GRPO_USER_PREAMBLE)
    embedded = result.prompt[len(GRPO_USER_PREAMBLE) :]
    assert embedded == result.provided_data
    assert json.loads(embedded) == result.fact_card
    assert set(result.fact_card) == {
        "schema_version",
        "atomic_topic",
        "evidence",
    }
    assert "meeting_date" not in result.prompt
    assert "canonical_key" not in result.prompt
    assert "sample_id" not in result.prompt
    assert "chk1-analysis-2019-01-29-synthetic-topic" not in result.prompt
    assert validate_judge_evidence(result.provided_data) == result.provided_data


def test_teacher_safe_fact_card_removes_date_bearing_production_sample_id() -> None:
    fact, _lineage_rows = _bundle()
    safe = safe_model_fact_card(fact)

    assert set(safe) == {
        "schema_version",
        "atomic_topic",
        "evidence",
    }
    rendered = canonical_json(safe)
    assert "sample_id" not in rendered
    assert "meeting_date" not in rendered
    assert "canonical_key" not in rendered
    assert "cutoff_ts" not in rendered
    assert "availability_upper_bound_ts" not in rendered
    assert "source_id" not in rendered
    assert "chk1-analysis-2019-01-29-synthetic-topic" not in rendered
    assert "2019-01-29" not in rendered
    assert safe_grpo_fact_card(fact) == safe
    assert safe_grpo_fact_card(safe) == safe


def test_preteacher_projection_removes_identity_before_both_model_prompts() -> None:
    fact, lineage = _bundle()
    result = project_preteacher_inputs(
        fact_card=fact,
        evidence_lineage=lineage,
        atomic_topic=fact["atomic_topic"],
        style_guide={"section_style_id": "synthetic", "guide_text": "Be concise."},
        student_chat_token_counter=lambda _system, _prompt: 100,
        generator_prompt_token_counter=lambda _prompt: 200,
    )

    assert result.provided_data == canonical_json(result.fact_card)
    assert result.validation_fact_card["sample_id"].startswith("chk1-analysis-")
    assert result.validation_fact_card["cutoff_ts"] == CUTOFF
    assert result.generator_prompt.count(result.provided_data) == 1
    assert result.student_prompt.count(result.provided_data) == 1
    for forbidden in (
        "sample_id",
        "meeting_date",
        "canonical_key",
        "cutoff_ts",
        "chk1-analysis-2019-01-29-synthetic-topic",
    ):
        assert forbidden not in result.generator_prompt
        assert forbidden not in result.student_prompt
    assert result.attestation["generator_prompt_tokens"] == 200
    assert result.attestation["input_truncated"] is False


def test_preferred_evidence_is_retained_before_an_equivalent_unpreferred_row() -> None:
    fact, lineage = _bundle()
    latest_b = next(
        item for item in fact["evidence"] if item["evidence_id"] == "E-latest-b"
    )
    one = deepcopy(fact)
    one["evidence"] = [latest_b]
    result = _project(
        fact,
        lineage,
        budget=_rendered_size(one),
        preferred=("E-latest-b",),
    )
    assert [item["evidence_id"] for item in result.fact_card["evidence"]] == [
        "E-latest-b"
    ]


def test_sft_target_auditor_counts_continuation_and_eos_exactly() -> None:
    class CharacterTokenizer:
        eos_token = "<eos>"

        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt, **_kwargs
        ):
            assert tokenize is False
            assert add_generation_prompt is True
            return "".join(
                f"<{item['role']}>{item['content']}" for item in messages
            ) + "<assistant>"

        def __call__(self, *, text):
            return {"input_ids": list(range(len(text)))}

    audit = build_sft_token_auditor(CharacterTokenizer())("prompt", "response")

    assert audit["completion_tokens"] == len("response<eos>")
    assert audit["total_tokens"] == (
        audit["prompt_tokens"] + audit["completion_tokens"]
    )
    assert audit["passed"] is True
    assert audit["truncated"] is False


def test_training_sft_auditor_has_no_standalone_completion_ceiling() -> None:
    class CharacterTokenizer:
        bos_token = "<bos>"
        bos_token_id = 1
        eos_token = "<eos>"

        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt, **_kwargs
        ):
            assert add_generation_prompt is True
            rendered = self.bos_token + "".join(
                f"<{item['role']}>{item['content']}" for item in messages
            ) + "<assistant>"
            if not tokenize:
                return rendered
            return self._encode(rendered, add_special_tokens=False)

        def __call__(self, *, text):
            return {"input_ids": self._encode(text, add_special_tokens=True)}

        def _encode(self, text, *, add_special_tokens):
            ids = []
            if text.startswith(self.bos_token):
                ids.append(self.bos_token_id)
                text = text[len(self.bos_token) :]
            elif add_special_tokens:
                ids.append(self.bos_token_id)
            return ids + [ord(character) + 10 for character in text]

    response = "x" * 2000
    audit = build_training_sft_token_auditor(CharacterTokenizer())(
        "prompt", response
    )

    assert audit["completion_tokens"] > 1024
    assert audit["max_completion_tokens"] is None
    assert audit["max_total_tokens"] == 7168
    assert audit["passed"] is True
