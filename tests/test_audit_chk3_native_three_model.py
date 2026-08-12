from __future__ import annotations

from jobs.eval.audit_chk3_native_three_model import (
    _answer_markup_failures,
    _copy_metrics,
    _file_binding_at,
    _ngram_repetition,
    _paired_contrast,
    _reasoning_metrics,
    summarize_rows,
)


def test_markup_and_reasoning_contract_proxies() -> None:
    assert _answer_markup_failures("Participants noted that inflation declined.") == []
    assert "markdown_or_list_markup" in _answer_markup_failures("- item")
    assert "json_or_schema_meta" in _answer_markup_failures('{"answer":"x"}')
    assert "citation_or_evidence_id" in _answer_markup_failures("See ev-abcdef12.")

    reasoning = "I need to inspect the prompt before writing the final answer."
    metrics = _reasoning_metrics(reasoning, "Inflation declined.", "Inflation declined.")
    assert metrics["reasoning_meta_free"] is False
    assert metrics["reasoning_first_person"] is True


def test_copy_normalizations_and_repetition() -> None:
    values = _copy_metrics(
        "The index was 15,164.01.",
        "The index was 15164.01.",
        "The index was 15,164.01.",
    )
    assert values["answer_source_raw_exact"] is False
    assert values["answer_source_punctuation_insensitive_exact"] is True
    assert values["answer_reference_raw_exact"] is True
    assert _ngram_repetition([1, 2, 3, 4, 1, 2, 3, 4], 4) > 0


def test_staging_binding_uses_post_rename_path(tmp_path) -> None:
    staging = tmp_path / "staging.json"
    staging.write_text("{}\n", encoding="utf-8")
    final = tmp_path / "published" / "artifact.json"
    binding = _file_binding_at(staging, final)
    assert binding["path"] == str(final.resolve())
    assert len(binding["sha256"]) == 64


def _minimal_row(stage: str, sample: str, rouge: float, core: bool) -> dict:
    row = {
        "stage_id": stage,
        "sample_id": sample,
        "length_bucket": "short",
        "analysis_reference_exact_identity": False,
        "delivery_valid": True,
        "degeneration_free": True,
        "fidelity_valid": True,
        "runner_quality_valid": True,
        "runner_native_structure_valid": True,
        "answer_format_valid": True,
        "core_native_valid": core,
        "extended_prompt_contract_valid": core,
        "rouge_eligible": True,
        "numeric_multiset_preserved": True,
        "date_set_preserved": True,
        "signed_numeric_surface_preserved": True,
        "attribution_no_unsupported": True,
        "reasoning_meta_free": True,
        "reasoning_first_person": False,
        "reasoning_contains_complete_source": False,
        "reasoning_contains_complete_answer": False,
        "answer_source_raw_exact": False,
        "answer_source_punctuation_insensitive_exact": False,
        "answer_reference_raw_exact": False,
        "answer_reference_punctuation_insensitive_exact": False,
        "strict_periodic_tail": False,
        "reasoning_tokens": 100,
        "answer_tokens": 20,
        "answer_reference_token_ratio": 1.0,
        "answer_word_trigram_repetition": 0.0,
        "reasoning_word_trigram_repetition": 0.0,
        "answer_token_4gram_repetition": 0.0,
        "reasoning_token_4gram_repetition": 0.0,
        "runner_full_token_4gram_repetition": 0.0,
        "runner_tail_token_4gram_repetition": 0.0,
        "answer_reference_rouge_l_f1": rouge,
        "answer_source_rouge_l_f1": rouge / 2,
        "reference_source_rouge_l_f1": 0.9,
        "audit_failures": [],
        "reasoning_meta_categories": [],
        "unsupported_attribution_categories": [],
        "dropped_attribution_categories": [],
    }
    return row


def test_summary_and_paired_contrast_keep_denominators() -> None:
    before = _minimal_row("chk1", "s1", 0.4, True)
    after = _minimal_row("chk3", "s1", 0.6, False)
    summary = summarize_rows([before])
    assert summary["cases"] == 1
    assert summary["core_native_valid_count"] == 1
    assert summary["answer_reference_rouge_l_f1"]["mean"] == 0.4

    indexed = {"chk1": {"s1": before}, "chk3": {"s1": after}}
    contrast = _paired_contrast(indexed, "chk1", "chk3")
    assert contrast["rouge_l"]["wins"] == 1
    assert contrast["rouge_l"]["mean_delta"] == 0.19999999999999996
    assert contrast["core_native_valid_regressed_cases"] == 1
