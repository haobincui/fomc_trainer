from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobs.retrain_v2 import probe_chk1_sft_degeneration as probe


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEEPSEEK_TOKENIZER = REPO_ROOT / "models/DeepSeek-R1-Distill-Llama-8B"


class FakeChatTokenizer:
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


class FakeDeepSeekDecoder:
    eos_token = "<eos>"

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        values = {10: "reasoning", 128014: "</think>", 11: "answer", 999: "<eos>"}
        return " ".join(values[token_id] for token_id in token_ids)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _candidate_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    candidate = tmp_path / "candidate"
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    config = tmp_path / "chk1.yaml"
    config.write_text(
        "system_prompt: |\n  fixed system prompt\nuser_prompt_suffix: null\n",
        encoding="utf-8",
    )

    # Each changed state has four rows spanning both eval and test.  Prompt
    # lengths are deliberately distinct so min/median/max selection is stable.
    definitions = {
        "eval": [(False, 2), (False, 8), (True, 3), (True, 9)],
        "test": [(False, 5), (False, 12), (True, 6), (True, 13)],
    }
    repairs: list[dict] = []
    split_records: dict[str, dict] = {}
    counter = 0
    for split, items in definitions.items():
        rows: list[dict] = []
        for changed, words in items:
            counter += 1
            prompt_text = " ".join(f"word-{counter}-{index}" for index in range(words))
            provided_data = json.dumps({"value": counter}, sort_keys=True)
            response = f"reasoning {counter}\n</think>\nanswer {counter}"
            row = {
                "prompt": prompt_text,
                "provided_data": provided_data,
                "response": response,
            }
            rows.append(row)
            repairs.append(
                {
                    "sample_id": f"sample-{counter}",
                    "split": split,
                    "changed": changed,
                    "prompt_sha256": probe.sha256_text(prompt_text),
                    "provided_data_sha256": probe.sha256_text(provided_data),
                    "new_response_sha256": probe.sha256_text(response),
                }
            )
        split_path = candidate / "analysis_sft" / f"{split}.jsonl"
        _write_jsonl(split_path, rows)
        split_records[split] = {
            "path": f"analysis_sft/{split}.jsonl",
            "rows": len(rows),
            "sha256": probe.sha256_file(split_path),
        }
    repair_path = candidate / "audits/repair_manifest.jsonl"
    _write_jsonl(repair_path, repairs)
    manifest = {
        "schema_version": "chk1-clean-sft-candidate-v2",
        "candidate_id": "fixture-clean-candidate",
        "immutable_candidate": True,
        "quality_status": "pending_semantic_audit",
        "split_files": split_records,
        "repair_manifest": {
            "path": "audits/repair_manifest.jsonl",
            "rows": len(repairs),
            "sha256": probe.sha256_file(repair_path),
        },
    }
    manifest_path = candidate / "candidate_manifest.json"
    _write_json(manifest_path, manifest)
    return candidate, tokenizer, config


def _result(
    *,
    sample_id: str,
    repetition: float = 0.05,
    tail_repetition: float = 0.06,
    contract_valid: bool = True,
    catastrophic: list[str] | None = None,
    manifest: str = "a" * 64,
    model_label: str = "model",
) -> dict:
    return {
        "sample_id": sample_id,
        "mode": "sampled",
        "seed": 100,
        "model_label": model_label,
        "sample_manifest_sha256": manifest,
        "status": "ok",
        "hit_eos": True,
        "cap_reached": False,
        "contract_valid": contract_valid,
        "strict_periodic_tail": False,
        "catastrophic_reasons": catastrophic or [],
        "full_token_4gram_repetition": repetition,
        "tail_token_4gram_repetition": tail_repetition,
    }


def test_build_sample_manifest_is_hash_bound_balanced_and_text_free(
    tmp_path: Path,
) -> None:
    candidate, tokenizer_path, config = _candidate_fixture(tmp_path)
    candidate_manifest = candidate / "candidate_manifest.json"
    artifact = probe.build_sample_manifest(
        candidate_dir=candidate,
        candidate_manifest_sha256=probe.sha256_file(candidate_manifest),
        tokenizer=FakeChatTokenizer(),
        tokenizer_path=tokenizer_path,
        training_config=config,
        sample_seed=70,
        greedy_longest=2,
    )

    assert artifact["schema_version"] == probe.MANIFEST_SCHEMA_VERSION
    assert len(artifact["samples"]) == 6
    assert {row["changed"] for row in artifact["samples"]} == {False, True}
    for changed in (False, True):
        group = [row for row in artifact["samples"] if row["changed"] is changed]
        assert {row["length_bucket"] for row in group} == {
            "short",
            "median",
            "long",
        }
        assert {row["split"] for row in group} == {"eval", "test"}
    assert sum(row["run_greedy"] for row in artifact["samples"]) == 2
    assert [row["sampling_seed"] for row in artifact["samples"]] == list(range(70, 76))
    serialized = json.dumps(artifact, ensure_ascii=False)
    assert "word-" not in serialized
    assert "fixed system prompt" not in serialized
    assert "reasoning" not in serialized


def test_build_sample_manifest_fails_on_candidate_hash_drift(tmp_path: Path) -> None:
    candidate, tokenizer_path, config = _candidate_fixture(tmp_path)
    with pytest.raises(probe.ProbeError, match="candidate manifest SHA256 mismatch"):
        probe.build_sample_manifest(
            candidate_dir=candidate,
            candidate_manifest_sha256="0" * 64,
            tokenizer=FakeChatTokenizer(),
            tokenizer_path=tokenizer_path,
            training_config=config,
        )


def test_analyze_completion_accepts_one_boundary_and_eos() -> None:
    metrics = probe.analyze_completion(
        text="careful reasoning\n</think>\na grounded final answer",
        generated_token_ids=[*range(1, 101), 999],
        eos_token_ids=999,
        max_new_tokens=3072,
    )

    assert metrics["status"] == "ok"
    assert metrics["hit_eos"] is True
    assert metrics["cap_reached"] is False
    assert metrics["think_boundary_count"] == 1
    assert metrics["has_nonempty_answer"] is True
    assert metrics["contract_valid"] is True
    assert metrics["catastrophic_reasons"] == []
    assert metrics["full_token_4gram_repetition"] == 0.0


def test_decode_preserves_deepseek_boundary_and_removes_only_terminal_eos() -> None:
    text = probe.decode_completion_preserving_boundary(
        FakeDeepSeekDecoder(),
        [10, 128014, 11, 999],
        # This is the exact internal shape produced by the run path after it
        # normalizes GenerationConfig.eos_token_id for model.generate.
        eos_token_ids={999},
    )

    assert text == "reasoning </think> answer"
    assert "<eos>" not in text


def test_eos_normalization_is_idempotent_and_rejects_non_integer_iterables() -> None:
    assert probe._normalize_eos_ids(128001) == {128001}
    assert probe._normalize_eos_ids({128001}) == {128001}
    assert probe._normalize_eos_ids(frozenset({128001, 128009})) == {
        128001,
        128009,
    }
    with pytest.raises(probe.ProbeError, match="non-integer"):
        probe._normalize_eos_ids([128001, "128009"])
    with pytest.raises(probe.ProbeError, match="negative"):
        probe._normalize_eos_ids([-1])


@pytest.mark.skipif(
    not (LOCAL_DEEPSEEK_TOKENIZER / "tokenizer.json").is_file(),
    reason="local DeepSeek tokenizer is unavailable",
)
def test_real_deepseek_decode_preserves_native_boundary_without_eos() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(LOCAL_DEEPSEEK_TOKENIZER), local_files_only=True
    )
    content = tokenizer.encode(
        "reasoning\n</think>\nanswer", add_special_tokens=False
    )
    assert tokenizer.encode("</think>", add_special_tokens=False) == [128014]
    text = probe.decode_completion_preserving_boundary(
        tokenizer,
        [*content, tokenizer.eos_token_id],
        eos_token_ids=tokenizer.eos_token_id,
    )

    assert text.count("</think>") == 1
    assert text.split("</think>", 1)[1].strip() == "answer"
    assert tokenizer.eos_token not in text


def test_analyze_completion_fails_cap_and_strict_periodic_tail() -> None:
    periodic = list(range(16)) * 3
    metrics = probe.analyze_completion(
        text="looping output",
        generated_token_ids=periodic,
        eos_token_ids=999,
        max_new_tokens=len(periodic),
    )

    assert metrics["cap_reached"] is True
    assert metrics["strict_periodic_tail"] is True
    assert "completion_length_cap" in metrics["catastrophic_reasons"]
    assert "strict_periodic_tail" in metrics["catastrophic_reasons"]


def test_compare_probe_results_passes_small_bounded_change() -> None:
    baseline = [_result(sample_id=f"sample-{index}") for index in range(4)]
    candidate = [
        _result(
            sample_id=f"sample-{index}",
            repetition=0.08,
            tail_repetition=0.09,
            model_label="smoke",
        )
        for index in range(4)
    ]

    comparison = probe.compare_probe_results(baseline, candidate)

    assert comparison["status"] == "passed"
    assert comparison["case_failures"] == []
    assert comparison["aggregate_failures"] == []


def test_compare_probe_results_fails_catastrophe_and_material_delta() -> None:
    baseline = [_result(sample_id="sample-1")]
    candidate = [
        _result(
            sample_id="sample-1",
            repetition=0.55,
            tail_repetition=0.70,
            catastrophic=[
                "full_4gram_repetition_ge_0.50",
                "tail_4gram_repetition_ge_0.60",
            ],
            model_label="smoke",
        )
    ]

    comparison = probe.compare_probe_results(baseline, candidate)

    assert comparison["status"] == "failed"
    reasons = comparison["case_failures"][0]["reasons"]
    assert "full_4gram_repetition_ge_0.50" in reasons
    assert "material_full_repetition_increase" in reasons
    assert "material_mean_tail_repetition_increase" in comparison["aggregate_failures"]


def test_compare_recomputes_cap_failure_instead_of_trusting_reason_list() -> None:
    baseline = [_result(sample_id="sample-1")]
    capped = _result(sample_id="sample-1", model_label="smoke")
    capped["cap_reached"] = True
    capped["catastrophic_reasons"] = []

    comparison = probe.compare_probe_results(baseline, [capped])

    assert comparison["status"] == "failed"
    assert "completion_length_cap" in comparison["case_failures"][0]["reasons"]


def test_compare_probe_results_fails_mismatched_manifest_or_cases() -> None:
    baseline = [_result(sample_id="sample-1")]
    with pytest.raises(probe.ProbeError, match="not bound to one sample manifest"):
        probe.compare_probe_results(
            baseline,
            [_result(sample_id="sample-1", manifest="b" * 64)],
        )
    with pytest.raises(probe.ProbeError, match="cases do not match"):
        probe.compare_probe_results(
            baseline,
            [_result(sample_id="sample-2")],
        )


def test_summarize_rejects_duplicate_generation_cases() -> None:
    row = _result(sample_id="duplicate")
    with pytest.raises(probe.ProbeError, match="duplicate generation cases"):
        probe.summarize_probe_results(
            [row, row],
            model_label="model",
            sample_manifest_sha256="a" * 64,
        )
