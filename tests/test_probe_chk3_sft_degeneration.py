from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobs.retrain_v2 import probe_chk3_sft_degeneration as probe


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **_kwargs,
    ):
        assert tokenize is True
        assert add_generation_prompt is True
        words = " ".join(message["content"] for message in messages).split()
        return list(range(1, len(words) + 2))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _prompt(analysis: str) -> str:
    return probe.USER_PROMPT_PREFIX + json.dumps(
        {"analysis": analysis}, ensure_ascii=False, separators=(",", ":")
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, list[dict]]:
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    config = tmp_path / "chk3.yaml"
    config.write_text(
        "system_prompt: |\n  Rewrite faithfully as one paragraph.\n",
        encoding="utf-8",
    )
    word_counts = [2, 4, 6, 8, 10]
    rows = []
    for index, word_count in enumerate(word_counts, start=1):
        filler = " ".join(f"term-{index}-{item}" for item in range(word_count))
        analysis = (
            f"In January 202{index}, the index was {index}.5 percent. {filler}"
        )
        rows.append(
            {
                "prompt": _prompt(analysis),
                "response": (
                    f"plan {index}\n</think>\nIn January 202{index}, "
                    f"the index was {index}.5 percent."
                ),
            }
        )
    validation = tmp_path / "validation.jsonl"
    _write_jsonl(validation, rows)
    return validation, tokenizer, config, rows


def _result(
    sample_id: str,
    bucket: str,
    *,
    model_label: str,
    repetition: float = 0.05,
    tail_repetition: float = 0.06,
    quality_valid: bool = True,
    failures: list[str] | None = None,
) -> dict:
    return {
        "sample_id": sample_id,
        "length_bucket": bucket,
        "model_label": model_label,
        "sample_manifest_sha256": "a" * 64,
        "seed": 20260810 + int(sample_id.rsplit("-", 1)[-1]),
        "generation_mode": "greedy",
        "max_new_tokens": 3072,
        "tail_tokens": 1024,
        "prompt_token_count": 400,
        "quality_valid": quality_valid,
        "quality_failures": failures or [],
        "hit_eos": True,
        "cap_reached": False,
        "think_boundary_count": 1,
        "final_answer_single_paragraph": True,
        "numeric_multiset_preserved": True,
        "date_set_preserved": True,
        "strict_periodic_tail": False,
        "full_token_4gram_repetition": repetition,
        "tail_token_4gram_repetition": tail_repetition,
    }


def test_build_manifest_selects_fixed_validation_lengths_without_text(
    tmp_path: Path,
) -> None:
    validation, tokenizer_path, config, rows = _fixture(tmp_path)
    manifest = probe.build_sample_manifest(
        validation_data=validation,
        tokenizer=FakeTokenizer(),
        tokenizer_path=tokenizer_path,
        training_config=config,
    )

    assert manifest["schema_version"] == probe.MANIFEST_SCHEMA_VERSION
    assert [row["length_bucket"] for row in manifest["samples"]] == [
        "short",
        "medium",
        "long",
    ]
    selected_lines = [row["line_number"] for row in manifest["samples"]]
    assert selected_lines == [1, 3, 5]
    assert [row["prompt_token_count"] for row in manifest["samples"]] == sorted(
        row["prompt_token_count"] for row in manifest["samples"]
    )
    serialized = json.dumps(manifest, ensure_ascii=False)
    for row in rows:
        assert row["prompt"] not in serialized
        assert row["response"] not in serialized
        assert probe.extract_source_analysis(row["prompt"]) not in serialized
    assert "Rewrite faithfully as one paragraph" not in serialized


def test_bound_manifest_fails_closed_on_validation_drift(tmp_path: Path) -> None:
    validation, tokenizer_path, config, _rows = _fixture(tmp_path)
    manifest = probe.build_sample_manifest(
        validation_data=validation,
        tokenizer=FakeTokenizer(),
        tokenizer_path=tokenizer_path,
        training_config=config,
    )
    validation.write_text(validation.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(probe.Chk3ProbeError, match="dataset hash changed"):
        probe._load_bound_rows(manifest)


@pytest.mark.parametrize(
    "prompt",
    [
        "not a chk3 prompt",
        probe.USER_PROMPT_PREFIX + "not-json",
        probe.USER_PROMPT_PREFIX + '{"analysis":"ok","extra":1}',
        probe.USER_PROMPT_PREFIX + '{"analysis":""}',
    ],
)
def test_source_analysis_parser_is_strict(prompt: str) -> None:
    with pytest.raises(probe.Chk3ProbeError):
        probe.extract_source_analysis(prompt)


def test_analyze_completion_passes_all_task_aligned_gates() -> None:
    source = "In January 2024, the index was 1.5 percent."
    metrics = probe.analyze_completion(
        text=(
            "The rewrite should preserve the measured change.\n</think>\n"
            "In January 2024, the index was 1.5 percent."
        ),
        generated_token_ids=[*range(1, 80), 999],
        eos_token_ids=999,
        max_new_tokens=512,
        source_analysis=source,
    )

    assert metrics["quality_valid"] is True
    assert metrics["quality_failures"] == []
    assert metrics["hit_eos"] is True
    assert metrics["think_boundary_count"] == 1
    assert metrics["has_nonempty_final_answer"] is True
    assert metrics["final_answer_single_paragraph"] is True
    assert metrics["numeric_multiset_preserved"] is True
    assert metrics["date_set_preserved"] is True


def test_analyze_completion_detects_structure_cap_repetition_and_fidelity() -> None:
    periodic_ids = list(range(16)) * 3
    metrics = probe.analyze_completion(
        text=(
            "loop\n</think>\nIn February, the index changed.\n"
            "This is a second paragraph."
        ),
        generated_token_ids=periodic_ids,
        eos_token_ids=999,
        max_new_tokens=len(periodic_ids),
        source_analysis="In January 2024, the index was 1.5 percent.",
    )

    assert metrics["quality_valid"] is False
    assert metrics["cap_reached"] is True
    assert metrics["strict_periodic_tail"] is True
    assert metrics["final_answer_single_paragraph"] is False
    assert metrics["numeric_multiset_preserved"] is False
    assert metrics["date_set_preserved"] is False
    assert set(metrics["quality_failures"]) >= {
        "missing_terminal_eos",
        "completion_length_cap",
        "strict_periodic_tail",
        "final_answer_not_single_paragraph",
        "source_numeric_multiset_not_preserved",
        "source_date_set_not_preserved",
    }


def test_redacted_result_never_persists_prompt_source_target_or_completion() -> None:
    source = "In January 2024, the index was 1.5 percent."
    completion = "fidelity plan\n</think>\n" + source
    result = probe.build_redacted_result(
        text=completion,
        generated_token_ids=[*range(1, 60), 999],
        eos_token_ids=999,
        max_new_tokens=512,
        tail_tokens=128,
        source_analysis=source,
        model_label="chk1-base",
        sample_manifest_sha256="a" * 64,
        sample={
            "sample_id": "opaque-sample",
            "length_bucket": "short",
            "prompt_token_count": 42,
        },
        seed=7,
    )

    serialized = json.dumps(result, ensure_ascii=False)
    assert source not in serialized
    assert completion not in serialized
    assert "completion" not in result
    assert "prompt" not in result
    assert "source_analysis" not in result
    assert result["completion_sha256"] == probe.common_probe.sha256_text(completion)


def test_compare_passes_bounded_candidate_change() -> None:
    buckets = ["short", "medium", "long"]
    baseline = [
        _result(f"sample-{index}", bucket, model_label="chk1-base")
        for index, bucket in enumerate(buckets)
    ]
    candidate = [
        _result(
            f"sample-{index}",
            bucket,
            model_label="chk3-smoke",
            repetition=0.08,
            tail_repetition=0.09,
        )
        for index, bucket in enumerate(buckets)
    ]

    comparison = probe.compare_results(baseline, candidate)

    assert comparison["status"] == "passed"
    assert comparison["case_failures"] == []
    assert comparison["aggregate_failures"] == []


def test_compare_fails_candidate_absolute_quality_and_repetition() -> None:
    buckets = ["short", "medium", "long"]
    baseline = [
        _result(f"sample-{index}", bucket, model_label="chk1-base")
        for index, bucket in enumerate(buckets)
    ]
    candidate = [
        _result(
            f"sample-{index}",
            bucket,
            model_label="chk3-smoke",
            repetition=0.55 if index == 0 else 0.05,
            tail_repetition=0.70 if index == 0 else 0.06,
            quality_valid=index != 1,
            failures=(
                ["source_numeric_multiset_not_preserved"] if index == 1 else []
            ),
        )
        for index, bucket in enumerate(buckets)
    ]

    comparison = probe.compare_results(baseline, candidate)

    assert comparison["status"] == "failed"
    reasons = {
        reason
        for failure in comparison["case_failures"]
        for reason in failure["reasons"]
    }
    assert "full_4gram_repetition_ge_0.50" in reasons
    assert "source_numeric_multiset_not_preserved" in reasons
    assert "candidate_quality_valid_rate_below_one" in comparison[
        "aggregate_failures"
    ]


def test_compare_rejects_manifest_or_case_drift() -> None:
    buckets = ["short", "medium", "long"]
    baseline = [
        _result(f"sample-{index}", bucket, model_label="chk1-base")
        for index, bucket in enumerate(buckets)
    ]
    candidate = [
        _result(f"sample-{index}", bucket, model_label="chk3-smoke")
        for index, bucket in enumerate(buckets)
    ]
    candidate[0]["sample_manifest_sha256"] = "b" * 64
    with pytest.raises(probe.Chk3ProbeError, match="one sample manifest"):
        probe.compare_results(baseline, candidate)

    candidate[0]["sample_manifest_sha256"] = "a" * 64
    candidate[0]["sample_id"] = "different"
    with pytest.raises(probe.Chk3ProbeError, match="exactly three cases"):
        probe.compare_results(baseline, candidate)


def test_compare_rejects_generation_contract_drift() -> None:
    buckets = ["short", "medium", "long"]
    baseline = [
        _result(f"sample-{index}", bucket, model_label="chk1-base")
        for index, bucket in enumerate(buckets)
    ]
    candidate = [
        _result(f"sample-{index}", bucket, model_label="chk3-smoke")
        for index, bucket in enumerate(buckets)
    ]
    candidate[1]["max_new_tokens"] = 4096

    with pytest.raises(probe.Chk3ProbeError, match="generation contract drift"):
        probe.compare_results(baseline, candidate)
