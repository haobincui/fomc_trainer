from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from jobs.retrain_v2 import repair_chk1_sft_release as repair
from jobs.retrain_v2.validate_chk1_clean_sft import validate_clean_sft
from open_r1.trainer.dataset_release import sha256_file, verify_clean_sft_release


class FakeTokenizer:
    bos_token = "<bos>"
    bos_token_id = 1
    eos_token = "<eos>"
    eos_token_id = 2

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        if text.startswith(self.bos_token):
            ids.append(self.bos_token_id)
            text = text[len(self.bos_token) :]
        elif add_special_tokens:
            ids.append(self.bos_token_id)
        suffix: list[int] = []
        if text.endswith(self.eos_token):
            text = text[: -len(self.eos_token)]
            suffix = [self.eos_token_id]
        ids.extend(10 + sum(token.encode("utf-8")) % 10000 for token in text.split())
        return ids + suffix

    def __call__(self, *, text: str) -> dict[str, list[int]]:
        return {"input_ids": self.encode(text, add_special_tokens=True)}

    def apply_chat_template(
        self, messages, *, tokenize: bool, add_generation_prompt: bool, **_kwargs
    ):
        assert add_generation_prompt is True
        rendered = self.bos_token + " ".join(
            f"<{item['role']}>{item['content']}" for item in messages
        ) + " <assistant> "
        return (
            self.encode(rendered, add_special_tokens=False)
            if tokenize
            else rendered
        )


def _prose(label: str, words: int = 24) -> str:
    return " ".join([label] + [f"economic{index}" for index in range(words - 1)]) + "."


def _reasoning(label: str) -> str:
    return " ".join(f"{label}{index}" for index in range(520))


@pytest.mark.parametrize(
    ("raw_content", "raw_reasoning", "expected_method", "needle"),
    [
        (
            json.dumps(
                {
                    "reasoning_content": "provider reasoning",
                    "content": {"answer": _prose("nested"), "evidence_ids": []},
                }
            ),
            "unused",
            "recover_nested_content_answer",
            "nested",
        ),
        (
            '{"answer":' + json.dumps(_prose("content")) + ',"evidence_ids":[',
            "unused",
            "recover_content_answer_string",
            "content",
        ),
        (
            '{"answer":"truncated',
            "Planning. Final object: "
            + json.dumps({"answer": _prose("reasoning"), "evidence_ids": []}),
            "recover_reasoning_final_answer_json",
            "reasoning",
        ),
        (
            '{"answer":"truncated',
            "Planning complete.\n\nI'll write:\n\n" + json.dumps(_prose("quoted")),
            "recover_marked_final_answer_quote",
            "quoted",
        ),
    ],
)
def test_four_cache_recovery_paths(
    raw_content: str,
    raw_reasoning: str,
    expected_method: str,
    needle: str,
) -> None:
    method, answer = repair.recover_json_like_answer(raw_content, raw_reasoning)
    assert method == expected_method
    assert answer.startswith(needle)


def test_cache_recovery_fails_closed_on_missing_or_ambiguous_source() -> None:
    with pytest.raises(repair.RepairError, match="no unambiguous"):
        repair.recover_json_like_answer('{"answer":"cut', "no final prose here")

    two_answers = (
        json.dumps({"answer": _prose("first")})
        + "\n"
        + json.dumps({"answer": _prose("second")})
    )
    with pytest.raises(repair.RepairError, match="ambiguous"):
        repair.recover_json_like_answer(two_answers, "unused")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("The rate was (37.7 percent; ev-ab12).", "The rate was (37.7 percent)."),
        ("The rate was (ev-ab12; 37.7 percent).", "The rate was (37.7 percent)."),
        ("The reading was (x, ev-a1, ev-b2).", "The reading was (x)."),
    ],
)
def test_mixed_citation_cleanup_preserves_factual_content(
    source: str, expected: str
) -> None:
    cleaned, removed = repair.strip_inline_evidence_ids(source)
    assert cleaned == expected
    assert removed >= 1
    number_pattern = re.compile(r"(?<![a-z])\d+(?:\.\d+)?", re.I)
    source_without_ids = repair._EVIDENCE_ID_RE.sub("", source)
    assert number_pattern.findall(cleaned) == number_pattern.findall(source_without_ids)
    assert ("37.7 percent" in source) == ("37.7 percent" in cleaned)
    assert ("(x" in source) == ("(x" in cleaned)


def test_reasoning_meta_cleanup_and_exact_tail_removal() -> None:
    answer = _prose("answer")
    reasoning = (
        "The fact card supports the assessment. The provided data are bounded.\n\n"
        "Based on these observations, the fixed final answer is supported: activity improved.\n\n"
        f"On balance, {answer} Additional duplicated summary."
    )
    cleaned, methods = repair.clean_reasoning(reasoning, answer)
    assert set(methods) == {
        "replace_fact_card_meta",
        "replace_provided_data_meta",
        "rewrite_fixed_final_answer_meta",
        "drop_repeated_answer_tail",
    }
    assert "available evidence" in cleaned.lower()
    assert "fixed final answer" not in cleaned.lower()
    assert answer not in cleaned


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(repair.canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )


def _fixture_inputs(tmp_path: Path) -> tuple[repair.RepairInputs, repair.RepairExpectations]:
    source = tmp_path / "source"
    base = tmp_path / "base"
    generation = tmp_path / "generation"
    cache = tmp_path / "cache"
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    _write_json(source / "release_manifest.json", {"status": "passed"})
    _write_json(base / "base_release_manifest.json", {"status": "passed"})
    _write_json(generation / "generation_handoff.json", {"status": "complete"})

    methods = [
        "recover_nested_content_answer",
        "recover_content_answer_string",
        "recover_reasoning_final_answer_json",
        "recover_marked_final_answer_quote",
        "strip_inline_evidence_ids",
        "unchanged",
    ]
    source_rows: list[dict[str, str]] = []
    manifests: list[dict] = []
    for index, method in enumerate(methods):
        prompt = f"Prompt {index}"
        provided = repair.canonical_json({"observation": index})
        clean_answer = _prose(f"answer{index}")
        if method.startswith("recover_"):
            source_answer = '{"answer":"broken'
        elif method == "strip_inline_evidence_ids":
            source_answer = clean_answer + " (ev-a12)"
        else:
            source_answer = clean_answer
        response = _reasoning(f"r{index}-") + repair.BOUNDARY + source_answer
        row = {"prompt": prompt, "response": response, "provided_data": provided}
        source_rows.append(row)
        sample_id = f"sample-{index}"
        cache_key = repair.sha256_text(f"cache-{index}")
        generator_prompt_sha = repair.sha256_text(f"generator-prompt-{index}")
        provenance_sha = repair.sha256_text("generation-provenance")
        manifests.append(
            {
                "sample_id": sample_id,
                "split": "train",
                "prompt_sha256": repair.sha256_text(prompt),
                "provided_data_sha256": repair.sha256_text(provided),
                "response_sha256": repair.sha256_text(response),
                "final_analysis_sha256": repair.sha256_text(source_answer),
                "generation": {
                    "cache_key": cache_key,
                    "generation_provenance_sha256": provenance_sha,
                    "model_input_projection": {
                        "generator_prompt_sha256": generator_prompt_sha
                    },
                },
            }
        )
        if method.startswith("recover_"):
            if method == "recover_nested_content_answer":
                content = json.dumps(
                    {
                        "reasoning_content": "provider",
                        "content": {"answer": clean_answer, "evidence_ids": []},
                    }
                )
                provider_reasoning = "unused"
            elif method == "recover_content_answer_string":
                content = '{"answer":' + json.dumps(clean_answer) + ',"evidence_ids":['
                provider_reasoning = "unused"
            elif method == "recover_reasoning_final_answer_json":
                content = '{"answer":"cut'
                provider_reasoning = json.dumps(
                    {"answer": clean_answer, "evidence_ids": []}
                )
            else:
                content = '{"answer":"cut'
                provider_reasoning = "Final answer:\n\n" + json.dumps(clean_answer)
            _write_json(
                cache / cache_key[:2] / f"{cache_key}.json",
                {
                    "cache_key": cache_key,
                    "prompt_sha256": generator_prompt_sha,
                    "generation_provenance_sha256": provenance_sha,
                    "provider_raw": {
                        "content": content,
                        "reasoning_content": provider_reasoning,
                    },
                },
            )

    for split in repair.SPLITS:
        rows = source_rows if split == "train" else []
        _write_jsonl(source / "analysis_sft" / f"{split}.jsonl", rows)
        _write_jsonl(base / "analysis_sft" / f"{split}.jsonl", rows)
        _write_jsonl(
            generation / "manifests" / f"{split}.jsonl",
            manifests if split == "train" else [],
        )
    inputs = repair.RepairInputs(source, base, generation, cache, tokenizer)
    expectations = repair.RepairExpectations(
        split_counts={"train": 6, "eval": 0, "test": 0},
        answer_method_counts={method: 1 for method in methods},
        reasoning_repair_counts={},
        changed_rows=5,
    )
    return inputs, expectations


def test_candidate_replay_and_audited_noreplace_publication(tmp_path: Path) -> None:
    inputs, expectations = _fixture_inputs(tmp_path)
    tokenizer = FakeTokenizer()
    candidate = tmp_path / "candidate"
    manifest = repair.materialize_candidate(
        inputs=inputs,
        work_dir=candidate,
        tokenizer=tokenizer,
        expectations=expectations,
    )
    assert manifest["quality_status"] == "pending_semantic_audit"
    repair.verify_candidate(candidate, tokenizer=tokenizer, expectations=expectations)

    repair_path = candidate / "audits/repair_manifest.jsonl"
    records = [json.loads(line) for line in repair_path.read_text().splitlines()]
    changed = [
        row
        for row in records
        if row["old_response_sha256"] != row["new_response_sha256"]
    ]
    audit_dir = tmp_path / "semantic_audit"
    row_audits = [
        {
            "schema_version": repair.SEMANTIC_AUDIT_SCHEMA_VERSION,
            "sample_id": row["sample_id"],
            "split": row["split"],
            "candidate_sha256": row["new_response_sha256"],
            "evidence_sha256": row["provided_data_sha256"],
            "judge_attempts": 1,
            "judge_raw_sha256": repair.sha256_text(f"judge-{row['sample_id']}"),
            "rubric": {
                "data_fidelity": 4,
                "trend_reasoning": 4,
                "policy_relevance": 4,
                "uncertainty_calibration": 4,
                "fomc_style": 4,
            },
            "status": "passed",
            "validated_violations": [],
            "blocking_violations": [],
        }
        for row in changed
    ]
    row_path = audit_dir / "row_audits.jsonl"
    _write_jsonl(row_path, row_audits)
    summary_path = audit_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "schema_version": repair.SEMANTIC_AUDIT_SCHEMA_VERSION,
            "status": "passed",
            "repair_manifest_sha256": repair.sha256_file(repair_path),
            "counts": {
                "expected": 5,
                "completed": 5,
                "passed": 5,
                "failed": 0,
                "judge_errors": 0,
                "blocking_violations": 0,
            },
            "row_audit": {
                "path": "row_audits.jsonl",
                "rows": 5,
                "sha256": repair.sha256_file(row_path),
            },
            "errors": [],
            "judge": {
                "model": repair.SEMANTIC_JUDGE_MODEL,
                "health": {
                    "model": repair.SEMANTIC_JUDGE_MODEL,
                    "loaded_model_root": f"/models/{repair.SEMANTIC_JUDGE_MODEL}",
                    "status": "ready",
                    "tokenizer_parity": True,
                    "weight_attested": True,
                }
            },
        },
    )
    validation_path = tmp_path / "deterministic_validation.json"
    validation = validate_clean_sft(
        source_release=inputs.source_release,
        clean_release=candidate,
        tokenizer_path=inputs.tokenizer_path,
        max_length=4608,
        expected_split_counts={"train": 6, "eval": 0, "test": 0},
        tokenizer=tokenizer,
    )
    assert validation["status"] == "passed"
    _write_json(validation_path, dict(validation))
    release_dir = tmp_path / "release"
    release_manifest = repair.publish_release(
        work_dir=candidate,
        semantic_audit_summary=summary_path,
        deterministic_validation_report=validation_path,
        release_dir=release_dir,
        tokenizer=tokenizer,
        expectations=expectations,
    )
    assert release_manifest["schema_version"] == "chk1-clean-sft-release-v2"
    assert release_manifest["quality_status"] == "passed"
    assert release_manifest["immutable"] is True
    assert release_manifest["split_counts"] == {"train": 6, "eval": 0, "test": 0}
    assert "source_hashes" in release_manifest
    release_manifest_path = release_dir / "release_manifest.json"
    verified = verify_clean_sft_release(
        dataset_dir=(release_dir / "analysis_sft").resolve(),
        manifest_path=release_manifest_path.resolve(),
        expected_manifest_sha256=sha256_file(release_manifest_path),
    )
    assert verified["deterministic_validation"]["max_length"] == 4608
    with pytest.raises(repair.RepairError, match="release exists"):
        repair.publish_release(
            work_dir=candidate,
            semantic_audit_summary=summary_path,
            deterministic_validation_report=validation_path,
            release_dir=release_dir,
            tokenizer=tokenizer,
            expectations=expectations,
        )
    repair._remove_tree(release_dir)
    repair._remove_tree(candidate)


def test_publication_rejects_missing_or_drifted_deterministic_receipt(
    tmp_path: Path,
) -> None:
    inputs, expectations = _fixture_inputs(tmp_path)
    tokenizer = FakeTokenizer()
    candidate = tmp_path / "candidate"
    repair.materialize_candidate(
        inputs=inputs,
        work_dir=candidate,
        tokenizer=tokenizer,
        expectations=expectations,
    )
    candidate_manifest = repair._read_json(candidate / "candidate_manifest.json")
    validation = validate_clean_sft(
        source_release=inputs.source_release,
        clean_release=candidate,
        tokenizer_path=inputs.tokenizer_path,
        max_length=4608,
        expected_split_counts={"train": 6, "eval": 0, "test": 0},
        tokenizer=tokenizer,
    )
    report_path = tmp_path / "validation.json"
    _write_json(report_path, dict(validation))
    repair._validate_deterministic_receipt(
        report_path=report_path,
        candidate_dir=candidate,
        candidate_manifest=candidate_manifest,
    )

    drifted = dict(validation)
    drifted["counts"] = {**validation["counts"], "truncated_rows": 1}
    digest_payload = dict(drifted)
    digest_payload.pop("validation_sha256")
    drifted["validation_sha256"] = repair.sha256_text(
        repair.canonical_json(digest_payload)
    )
    _write_json(report_path, drifted)
    with pytest.raises(repair.RepairError, match="truncated_rows"):
        repair._validate_deterministic_receipt(
            report_path=report_path,
            candidate_dir=candidate,
            candidate_manifest=candidate_manifest,
        )
    repair._remove_tree(candidate)
    repair._remove_tree(candidate)


def test_production_population_matches_sealed_sources() -> None:
    if not repair.DEFAULT_SOURCE_RELEASE.exists():
        pytest.skip("sealed production chk1 release is not present")
    inputs = repair.RepairInputs(
        repair.DEFAULT_SOURCE_RELEASE,
        repair.DEFAULT_BASE_RELEASE,
        repair.DEFAULT_GENERATION_DIR,
        repair.DEFAULT_TEACHER_CACHE,
        repair.DEFAULT_TOKENIZER,
    ).resolved()
    tokenizer = repair._load_tokenizer(inputs.tokenizer_path)
    payload = repair.build_candidate_payload(
        inputs, tokenizer, expectations=repair.RepairExpectations.production()
    )
    assert payload.summary["answer_repair_method_counts"] == dict(
        sorted(repair.EXPECTED_ANSWER_METHOD_COUNTS.items())
    )
    assert payload.summary["reasoning_repair_counts"] == dict(
        sorted(repair.EXPECTED_REASONING_REPAIR_COUNTS.items())
    )
    assert payload.summary["changed_rows"] == 237
    assert len(payload.repair_records) == 1743
    assert len(payload.semantic_audit_rows) == 237
