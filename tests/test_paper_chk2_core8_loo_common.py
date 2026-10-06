from __future__ import annotations

import copy
from pathlib import Path
from types import MappingProxyType

import pytest

from jobs.eval import paper_chk2_core8_loo_common as subject


class _ConstantTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize is True
        assert add_generation_prompt is True
        assert [message["role"] for message in messages] == ["system", "user"]
        return [101, 102, 103]


class _NeutralMismatchTokenizer(_ConstantTokenizer):
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        values = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        if "token_matched_neutral" in messages[-1]["content"]:
            values.append(104)
        return values


def _synthetic_source() -> subject.FrozenSource:
    template = (
        "Rewrite the following analysis as formal FOMC Minutes prose:\n\n"
        '{"analysis":"[SOURCE_ANALYSIS]"}'
    )
    rows: list[dict[str, object]] = []
    full_ids: dict[str, str] = {}
    for meeting_rank in range(subject.EXPECTED_MEETINGS):
        meeting_id = f"meeting-{meeting_rank:03d}"
        full_id = f"core8-loo::{meeting_id}::full::none"
        full_ids[meeting_id] = full_id
        full_analysis_sha = subject.sha256_text(f"{meeting_id}:full:None")
        full_reference_sha = subject.sha256_text(f"reference:{meeting_id}")
        for variant_rank in range(subject.VARIANTS_PER_MEETING):
            arm, topic = subject._expected_variant(variant_rank)
            sample_id = f"core8-loo::{meeting_id}::{arm}::{topic or 'none'}"
            analysis = f"{meeting_id}:{arm}:{topic}"
            prompt = subject._render_user_prompt(template, analysis)
            neutral_proof = None
            if arm == "token_matched_neutral":
                neutral_proof = {
                    "meeting_id": meeting_id,
                    "topic": topic,
                    "replacement_sha256": "a" * 64,
                    "label_preserved": True,
                    "numeric_surfaces_removed": True,
                    "date_values_removed": True,
                }
            rows.append(
                {
                    "sample_id": sample_id,
                    "meeting_id": meeting_id,
                    "meeting_start_date": f"1993-01-{meeting_rank % 28 + 1:02d}",
                    "meeting_rank": meeting_rank,
                    "arm": arm,
                    "intervention_topic": topic,
                    "variant_rank": variant_rank,
                    "prompt": prompt,
                    "prompt_sha256": subject.sha256_text(prompt),
                    "source_analysis": analysis,
                    "source_analysis_sha256": subject.sha256_text(analysis),
                    "reference_minutes_sha256": full_reference_sha,
                    "full_source_analysis_sha256": full_analysis_sha,
                    "full_reference_sha256": full_reference_sha,
                    "neutral_proof": neutral_proof,
                }
            )
    typed_rows = tuple(dict(row) for row in rows)
    by_id = {str(row["sample_id"]): row for row in typed_rows}
    return subject.FrozenSource(
        source_release_manifest={},
        source_release_manifest_path=Path("release.json"),
        source_release_manifest_sha256="b" * 64,
        panel_path=Path("panel.jsonl"),
        panel_sha256="c" * 64,
        panel_rows=(),
        inputs_path=Path("inputs.jsonl"),
        inputs_sha256="d" * 64,
        legacy_samples_manifest_path=Path("samples.json"),
        legacy_samples_manifest_sha256="e" * 64,
        legacy_samples_manifest={},
        ordered_rows=typed_rows,
        by_sample_id=MappingProxyType(by_id),
        full_sample_id_by_meeting=MappingProxyType(full_ids),
        meeting_ids=tuple(full_ids),
        topics=subject.TOPICS,
    )


def _synthetic_bindings(tokenizer: object) -> subject.PaperChk2Bindings:
    template = (
        "Rewrite the following analysis as formal FOMC Minutes prose:\n\n"
        '{"analysis":"[SOURCE_ANALYSIS]"}'
    )
    return subject.PaperChk2Bindings(
        parent_model_path=Path("parent"),
        adapter_path=Path("adapter"),
        tokenizer_path=Path("adapter"),
        prompt_contract_path=Path("prompt.json"),
        prompt_contract={},
        system_prompt="system",
        user_template=template,
        system_prompt_sha256=subject.sha256_text("system"),
        user_template_sha256=subject.sha256_text(template),
        runtime_versions=MappingProxyType({}),
        file_bindings=MappingProxyType({}),
        tokenizer=tokenizer,
    )


def test_fixed_coverage_and_case_coordinates() -> None:
    assert subject.EXPECTED_PROMPTS == 128 * 17 == 2_176
    assert subject.EXPECTED_GENERATIONS == 2_176 * 10 == 21_760
    assert len(subject.REPLICATE_SEEDS) == len(set(subject.REPLICATE_SEEDS)) == 10
    assert (subject.EXPECTED_MIN_PROMPT_TOKENS, subject.EXPECTED_MAX_PROMPT_TOKENS) == (
        981,
        1_291,
    )
    assert subject.canonical_case_coordinates(0) == {
        "absolute_case_index": 0,
        "absolute_prompt_index": 0,
        "replicate_id": 0,
        "replicate_seed": subject.REPLICATE_SEEDS[0],
        "meeting_index": 0,
        "variant_rank": 0,
        "paired_block_index": 0,
    }
    last = subject.canonical_case_coordinates(subject.EXPECTED_GENERATIONS - 1)
    assert last["absolute_prompt_index"] == 2_175
    assert last["replicate_id"] == 9
    assert last["meeting_index"] == 127
    assert last["variant_rank"] == 16
    with pytest.raises(subject.PaperChk2Core8PreparationError, match="out of range"):
        subject.canonical_case_coordinates(subject.EXPECTED_GENERATIONS)


def test_frozen_source_matrix_is_complete_and_ordered() -> None:
    source = subject.load_and_validate_frozen_source()
    assert len(source.ordered_rows) == 2_176
    assert len(source.panel_rows) == 1_024
    assert len(source.meeting_ids) == 128
    assert len(source.full_sample_id_by_meeting) == 128
    assert source.ordered_rows[0]["variant_rank"] == 0
    assert source.ordered_rows[-1]["variant_rank"] == 16
    assert source.ordered_rows[0]["meeting_start_date"] == "1993-02-02"
    assert source.ordered_rows[-1]["meeting_start_date"] == "2008-12-15"


def test_recorded_lineage_holdout_audit_has_zero_exact_overlap_and_caveats() -> None:
    audit = subject.build_recorded_lineage_holdout_audit(
        subject.load_and_validate_frozen_source()
    )
    assert audit["status"] == "passed_with_required_caveats"
    assert set(audit["tests"].values()) == {0, True}
    assert audit["populations"]["parent_chk1"]["rows"] == 1_743
    assert audit["populations"]["paper_chk2"]["rows"] == 391
    assert (
        audit["populations"]["paper_chk2"]["sources_bound_to_parent_chk1_answers"]
        == 391
    )
    caveats = " ".join(audit["required_caveats"])
    assert "repeated-holdout" in caveats
    assert "pretraining" in caveats
    assert "not a prospective" in caveats


def test_exact_ledger_and_all_neutral_proofs_are_sealed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "EXPECTED_MIN_PROMPT_TOKENS", 3)
    monkeypatch.setattr(subject, "EXPECTED_MAX_PROMPT_TOKENS", 3)
    source = _synthetic_source()
    ledger = subject.build_exact_token_ledger(
        source, _synthetic_bindings(_ConstantTokenizer())
    )
    proofs = subject.build_neutral_token_match_proofs(source, ledger)
    assert len(ledger) == 2_176
    assert len(proofs) == 1_024
    assert {row["prompt_token_count"] for row in ledger} == {3}
    assert all(row["row_sha256"] == subject._row_digest(row, "row_sha256") for row in ledger)
    assert all(proof["token_count_equal"] is True for proof in proofs)
    first_seeds = {
        subject.row_seed_for_case(ledger[rank], 0)
        for rank in range(subject.VARIANTS_PER_MEETING)
    }
    assert len(first_seeds) == 1


def test_new_prompt_neutral_token_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "EXPECTED_MIN_PROMPT_TOKENS", 3)
    monkeypatch.setattr(subject, "EXPECTED_MAX_PROMPT_TOKENS", 4)
    source = _synthetic_source()
    ledger = subject.build_exact_token_ledger(
        source, _synthetic_bindings(_NeutralMismatchTokenizer())
    )
    with pytest.raises(
        subject.PaperChk2Core8PreparationError,
        match="neutral token-count mismatch",
    ):
        subject.build_neutral_token_match_proofs(source, ledger)


def test_global_prompt_token_range_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "EXPECTED_MIN_PROMPT_TOKENS", 2)
    monkeypatch.setattr(subject, "EXPECTED_MAX_PROMPT_TOKENS", 3)
    with pytest.raises(
        subject.PaperChk2Core8PreparationError,
        match="global prompt-token range drift",
    ):
        subject.build_exact_token_ledger(
            _synthetic_source(), _synthetic_bindings(_ConstantTokenizer())
        )


def test_ledger_token_or_row_tampering_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "EXPECTED_MIN_PROMPT_TOKENS", 3)
    monkeypatch.setattr(subject, "EXPECTED_MAX_PROMPT_TOKENS", 3)
    ledger = list(
        subject.build_exact_token_ledger(
            _synthetic_source(), _synthetic_bindings(_ConstantTokenizer())
        )
    )
    tampered = copy.deepcopy(ledger)
    tampered[0]["prompt_token_ids"][0] += 1
    with pytest.raises(subject.PaperChk2Core8PreparationError, match="token-ID hash"):
        subject._validate_ledger_rows(tampered)
