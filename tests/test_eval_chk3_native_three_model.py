from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from jobs.eval import eval_chk3_native_three_model as subject
from open_r1.validator.loo_generation_spec import seal_manifest, validate_manifest_integrity


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_MANIFEST = (
    ROOT
    / "docs/summary/20260811T003000Z/chk3_native_analysis_to_minutes_cp250_eval/samples_n12.json"
)
SAMPLE_MANIFEST_SHA = "371d29601e343acf98cac6173663d842ead9acaa4a454991d512b3f00bf5ee77"


class ExactDecodeTokenizer:
    def __init__(self, expected: str):
        self.expected = expected
        self.eos_token = "<eos>"

    def decode(self, token_ids, **_kwargs):
        assert token_ids == [10, 11]
        return self.expected


def _row(stage_id: str = "chk0", model_label: str = "base") -> dict:
    source = "In April 2024, the rate was -22.8 percent."
    prompt = subject.native_probe.USER_PROMPT_PREFIX + json.dumps(
        {"analysis": source}, separators=(",", ":")
    )
    reference = f"reference reasoning\n</think>\n{source}"
    generated = f"generation reasoning\n</think>\n{source}"
    sample = {
        "sample_id": "sample-0",
        "length_bucket": "short",
        "prompt_token_count": 42,
        "prompt_sha256": subject.native_probe.common_probe.sha256_text(prompt),
        "analysis_reference_exact_identity": True,
        "normalized_identity": True,
        "punctuation_insensitive_identity": True,
    }
    return subject.build_full_result(
        text=generated,
        generated_token_ids=[10, 11, 999],
        eos_token_ids=[999],
        max_new_tokens=512,
        tail_tokens=128,
        source_prompt=prompt,
        source_analysis=source,
        reference_response=reference,
        load_in_4bit=True,
        attn_implementation="sdpa",
        pad_token_id=999,
        stage_id=stage_id,
        model_label=model_label,
        sample_manifest_sha256="a" * 64,
        sample=sample,
        seed=7,
    )


def test_frozen_test_n12_selection_and_identity_are_exact() -> None:
    manifest, observed = subject._load_sample_manifest(
        SAMPLE_MANIFEST, SAMPLE_MANIFEST_SHA
    )
    rows = subject._load_bound_rows(manifest)

    assert observed == SAMPLE_MANIFEST_SHA
    assert [sample["line_number"] - 1 for sample in manifest["samples"]] == [
        142,
        146,
        155,
        166,
        113,
        72,
        176,
        131,
        73,
        188,
        7,
        24,
    ]
    assert len(rows) == 12
    assert sum(sample["normalized_identity"] for sample in manifest["samples"]) == 1


def test_full_row_persists_and_recomputes_all_text_token_and_signed_metrics() -> None:
    row = _row()
    source = row["source_analysis"]
    sample = {
        "sample_id": row["sample_id"],
        "length_bucket": row["length_bucket"],
        "prompt_token_count": row["prompt_token_count"],
        "prompt_sha256": row["source_prompt_sha256"],
        "analysis_reference_exact_identity": True,
        "normalized_identity": True,
        "punctuation_insensitive_identity": True,
    }
    subject.validate_full_result(
        row,
        stage_id="chk0",
        model_label="base",
        sample_manifest_sha256="a" * 64,
        sample=sample,
        source_analysis=source,
        reference_response=f"reference\n</think>\n{source}",
        tokenizer=ExactDecodeTokenizer(row["generated_text"]),
    )
    assert row["signed_numeric_surface_preserved"] is True
    assert row["source_prompt"] and row["reference_minutes"] and row["answer"]


def test_full_row_rejects_text_token_and_metric_tampering() -> None:
    row = _row()
    sample = {
        "sample_id": row["sample_id"],
        "length_bucket": row["length_bucket"],
        "prompt_token_count": row["prompt_token_count"],
        "prompt_sha256": row["source_prompt_sha256"],
        "analysis_reference_exact_identity": True,
        "normalized_identity": True,
        "punctuation_insensitive_identity": True,
    }
    row["generated_text"] += "tampered"
    with pytest.raises(subject.NativeThreeModelEvalError, match="SHA256"):
        subject.validate_full_result(
            row,
            stage_id="chk0",
            model_label="base",
            sample_manifest_sha256="a" * 64,
            sample=sample,
            source_analysis=row["source_analysis"],
            reference_response=f"ref\n</think>\n{row['source_analysis']}",
        )


def test_resealed_cherry_picked_manifest_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(SAMPLE_MANIFEST.read_text(encoding="utf-8"))
    payload.pop("integrity")
    payload["samples"][0]["sample_id"] = "cherry-picked"
    tampered = tmp_path / "samples.json"
    tampered.write_text(
        json.dumps(seal_manifest(payload), sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest, _ = subject._load_sample_manifest(
        tampered, subject.native_probe.common_probe.sha256_file(tampered)
    )
    with pytest.raises(subject.NativeThreeModelEvalError, match="frozen deterministic"):
        subject._load_bound_rows(manifest)


def test_comparison_requires_identical_contract_and_seals_n12() -> None:
    runs = {}
    for stage_id in subject.REQUIRED_STAGES:
        model_label = f"model-{stage_id}"
        rows = []
        for index in range(12):
            row = copy.deepcopy(_row(stage_id, model_label))
            row["sample_id"] = f"sample-{index}"
            row["length_bucket"] = ("short", "medium", "long")[index // 4]
            row["seed"] = 100 + index
            row["analysis_reference_exact_identity"] = index == 7
            row["normalized_identity"] = index == 7
            row["punctuation_insensitive_identity"] = index == 7
            rows.append(row)
        runs[stage_id] = {
            "stage_id": stage_id,
            "model_label": model_label,
            "results": rows,
            "summary": subject._summary(
                rows, model_label=model_label, sample_manifest_sha256="a" * 64
            ),
            "generation_contract": subject._generation_contract(rows),
            "manifest_binding": {
                "path": f"/{stage_id}/manifest.json",
                "sha256": stage_id.ljust(64, "0"),
                "payload_sha256": stage_id.ljust(64, "1"),
            },
        }
    comparison = subject.build_comparison(runs)
    assert validate_manifest_integrity(comparison) == comparison["integrity"][
        "payload_sha256"
    ]
    assert comparison["cohort_summaries"]["analysis_reference_identity"]["cases"] == 1

    runs["chk3"]["generation_contract"]["load_in_4bit"] = False
    with pytest.raises(subject.NativeThreeModelEvalError, match="contract drift"):
        subject.build_comparison(runs)
