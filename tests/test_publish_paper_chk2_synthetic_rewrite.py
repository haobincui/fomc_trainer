from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobs.generation import generate_paper_chk2_synthetic_rewrite as generator
from jobs.generation import publish_paper_chk2_synthetic_rewrite as publisher
from jobs.generation.publish_paper_chk2_synthetic_rewrite import (
    BOUNDARY,
    DATASET_ROLE,
    RELEASE_SCHEMA_VERSION,
    TERMINAL_SCHEMA_VERSION,
    PublicationError,
    publish_release,
    sha256_file,
    sha256_text,
    verify_release,
)


SPLITS = ("train", "validation", "test")
LINEAGE = {
    "analysis_source_lineage": "target_derived_from_official_minutes",
    "analysis_teacher_saw_official_target": True,
    "source_population_target_derived": True,
    "rewrite_teacher_input_fields": ["source_analysis"],
    "rewrite_teacher_received_official_validator_feedback": False,
    "target_is_teacher_synthetic_rewrite": True,
    "student_prompt_has_direct_target_field": False,
    "rewrite_teacher_saw_official_target": False,
    "input_fidelity_validator_saw_official_target": False,
    "official_reference_validator_saw_official_target": True,
    "official_reference_feedback_returned_to_rewrite_teacher": False,
    "official_reference_score_used_for_selection": False,
    "rewrite_stage_target_assisted_selection": False,
    "reference_validator_saw_official_target": True,
    "reference_validator_used_for_training_selection": False,
    "reference_audit_only": True,
    "official_minutes_used_as_student_target": False,
    "teacher_response_analysis_was_sanitized": False,
    "teacher_response_analysis_is_complete_native_cot": True,
    "teacher_response_analysis_operational_meta_allowed": True,
    "teacher_response_analysis_draft_deliberation_allowed": True,
    "source_wording_reuse_allowed": True,
    "near_copy_used_for_training_rejection": False,
    "exact_full_source_copy_used_for_training_rejection": True,
    "reference_lexical_overlap_used_for_warning": False,
    "token_length_used_for_training_rejection": False,
    "maximum_total_tokens": None,
    "training_only": True,
    "evaluation_eligible": False,
    "suitable_for_leakage_safe_evaluation": False,
    "human_review_required": False,
}


class FakeDeepSeekTokenizer:
    bos_token = "<BOS>"
    eos_token = "<EOS>"
    bos_token_id = 1
    eos_token_id = 2

    def _encode(self, text: str) -> list[int]:
        ids = [self.bos_token_id]
        if text.startswith(self.bos_token):
            text = text[len(self.bos_token) :]
        body = text
        has_eos = body.endswith(self.eos_token)
        if has_eos:
            body = body[: -len(self.eos_token)]
        ids.extend(10 + ord(character) for character in body)
        if has_eos:
            ids.append(self.eos_token_id)
        return ids

    def __call__(self, *, text: str):
        return {"input_ids": self._encode(text)}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        truncation: bool = False,
        return_dict: bool = False,
    ):
        assert add_generation_prompt is True
        assert truncation is False
        rendered = self.bos_token
        for message in messages:
            rendered += f"[{message['role']}]\n{message['content']}\n"
        rendered += "[assistant]\n<think>\n"
        return self._encode(rendered) if tokenize else rendered


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _descriptor(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "rows": len(path.read_text(encoding="utf-8").splitlines()),
    }


def _id_digest(values: list[str]) -> str:
    return sha256_text("".join(f"{value}\n" for value in sorted(values)))


def _prompt(analysis: str) -> str:
    return (
        "Rewrite the following analysis as formal FOMC Minutes prose:\n\n"
        + json.dumps({"analysis": analysis}, ensure_ascii=False, separators=(",", ":"))
    )


def _provider(
    role: str,
    allowed_fields: tuple[str, ...],
    wire_fields: tuple[str, ...],
    *,
    official_target_policy: str = "forbidden",
    repair_trigger_source=None,
) -> dict:
    return {
        "response_id": f"response-{role}",
        "returned_model": "deepseek-v4-flash",
        "system_fingerprint": "fixed-fingerprint",
        "finish_reason": "stop",
        "request_projection": {
            "role": role,
            "allowed_fields": list(allowed_fields),
            "wire_fields": list(wire_fields),
            "field_value_sha256": {
                field: sha256_text(f"value-{role}-{field}") for field in allowed_fields
            },
            "canonical_payload_sha256": sha256_text(f"payload-{role}"),
            "official_target_policy": official_target_policy,
            "repair_trigger_source": repair_trigger_source,
            "reference_diagnostic_can_trigger_repair": False,
            "api_key_env": "DEEPSEEK_API_KEY",
            "plaintext_credential_persisted": False,
        },
    }


def _source_fixture(root: Path) -> tuple[dict[str, list[dict]], str, str, str]:
    source_rows: dict[str, list[dict]] = {}
    artifacts: dict[str, dict] = {}
    all_ids: list[str] = []
    for split_number, split in enumerate(SPLITS):
        candidates: list[dict] = []
        manifests: list[dict] = []
        source_rows[split] = []
        for index in range(2):
            sample_id = f"{split}-{index}"
            meeting = f"2026-0{split_number + 1}-{index + 1:02d}"
            analysis = f"Economic activity in {split} source row {index} remained steady."
            prompt = _prompt(analysis)
            official = f"Official paragraph for {sample_id}."
            candidate = {"prompt": prompt, "response": f"old{BOUNDARY}{official}"}
            manifest = {
                "sample_id": sample_id,
                "split": split,
                "analysis": analysis,
                "analysis_sha256": sha256_text(analysis),
                "prompt_sha256": sha256_text(prompt),
                "official_minutes_sha256": sha256_text(official),
                "meeting_date": meeting,
            }
            candidates.append(candidate)
            manifests.append(manifest)
            source_rows[split].append(
                {"candidate": candidate, "manifest": manifest, "source_index": index}
            )
            all_ids.append(sample_id)
        _jsonl(root / f"sft_candidate/{split}.jsonl", candidates)
        _jsonl(root / f"manifests/{split}.jsonl", manifests)
        artifacts[split] = {
            "rows": 2,
            "sft_candidate": _descriptor(root, f"sft_candidate/{split}.jsonl"),
            "manifest": _descriptor(root, f"manifests/{split}.jsonl"),
        }
    summary = {"artifacts": artifacts, "total_training_rows": 6}
    _json(root / "summary.json", summary)
    _json(root / "merge_receipt.json", {"status": "passed", "rows": 6})
    summary_sha = sha256_file(root / "summary.json")
    _json(root / "handoff.json", {"summary": {"path": "summary.json", "sha256": summary_sha}})
    return source_rows, summary_sha, sha256_file(root / "handoff.json"), _id_digest(all_ids)


def _terminal_row(source: dict, *, passed: bool) -> dict:
    manifest = source["manifest"]
    analysis = manifest["analysis"]
    reasoning = "Preserve the direction, scope, attribution, and uncertainty in the source."
    answer = "Economic activity remained steady, according to the supplied analysis."
    prompt = source["candidate"]["prompt"]
    response = reasoning + BOUNDARY + answer
    if passed:
        status = "PASS"
        rejection_stage = None
        rejection_reasons: list[str] = []
        deterministic_pass = True
        validator_a_complete = True
        validator_a_pass = True
    else:
        status = "GENERATION_QUALITY_REJECT"
        rejection_stage = "deterministic_validation"
        rejection_reasons = ["numeric_multiset_mismatch"]
        deterministic_pass = False
        validator_a_complete = False
        validator_a_pass = False
    return {
        "schema_version": TERMINAL_SCHEMA_VERSION,
        "sample_id": manifest["sample_id"],
        "split": manifest["split"],
        "source_index": source["source_index"],
        "meeting_date": manifest["meeting_date"],
        "terminal_status": status,
        "training_pass": passed,
        "rejection_stage": rejection_stage,
        "rejection_reasons": rejection_reasons,
        "source_analysis": analysis,
        "teacher_response_analysis": reasoning if passed else None,
        "rewritten_minutes": answer if passed else None,
        "student_prompt": prompt if passed else None,
        "sft_response": response if passed else None,
        "source_analysis_sha256": sha256_text(analysis),
        "official_minutes_sha256": manifest["official_minutes_sha256"],
        "teacher_response_analysis_sha256": sha256_text(reasoning) if passed else None,
        "rewritten_minutes_sha256": sha256_text(answer) if passed else None,
        "prompt_sha256": sha256_text(prompt) if passed else None,
        "response_sha256": sha256_text(response) if passed else None,
        "generation": {
            "selected_attempt": "primary",
            "repair_used": False,
            "deterministic_validation": {
                "machine_pass": deterministic_pass,
                "reasons": [] if passed else rejection_reasons,
                "diagnostics": {},
            },
            "provider": _provider(
                "rewrite_primary", ("source_analysis",), ("analysis",)
            ),
        },
        "validator_a": {
            "complete": validator_a_complete,
            "machine_pass": validator_a_pass,
            "reasons": [],
            "result": {} if passed else None,
            "provider": (
                _provider(
                    "validator_a_primary",
                    (
                        "source_analysis",
                        "teacher_response_analysis",
                        "rewritten_minutes",
                    ),
                    (
                        "source_analysis",
                        "teacher_response_analysis",
                        "rewritten_minutes",
                    ),
                )
                if passed
                else {}
            ),
        },
        # Deliberately poor official-reference score: it must remain diagnostic.
        "reference_diagnostic": {
            "complete": passed,
            "diagnostic_pass": False if passed else None,
            "overall_score": 12 if passed else None,
            "warning": "REFERENCE_DIAGNOSTIC_WARNING" if passed else None,
            "reasons": ["style_distance"] if passed else [],
            "result": {} if passed else None,
            "provider": (
                _provider(
                    "validator_b",
                    ("official_minutes", "rewritten_minutes"),
                    ("official_minutes", "rewritten_minutes"),
                    official_target_policy=(
                        "required_exact_normalized_official_minutes"
                    ),
                )
                if passed
                else {}
            ),
        },
        "lineage": dict(LINEAGE),
    }


def _acquisition_fixture(
    root: Path,
    source_rows: dict[str, list[dict]],
    *,
    source_root: Path,
    source_summary_sha: str,
    source_handoff_sha: str,
    source_id_sha: str,
) -> None:
    contract = generator._prompt_contract(
        code_sha256=sha256_file(Path(generator.__file__).resolve()),
        config=generator.ProviderConfig(),
    )
    _json(root / "prompt_contract.json", contract)
    artifacts: dict[str, dict] = {}
    for split in SPLITS:
        # Put the rejection first to prove release_index and source_index differ.
        terminal = [
            _terminal_row(source_rows[split][0], passed=False),
            _terminal_row(source_rows[split][1], passed=True),
        ]
        candidate = [
            {
                "prompt": terminal[1]["student_prompt"],
                "response": terminal[1]["sft_response"],
            }
        ]
        manifest = [{"sample_id": terminal[1]["sample_id"]}]
        _jsonl(root / f"terminal/{split}.jsonl", terminal)
        _jsonl(root / f"sft_candidate/{split}.jsonl", candidate)
        _jsonl(root / f"manifests/{split}.jsonl", manifest)
        artifacts[split] = {
            "terminal": _descriptor(root, f"terminal/{split}.jsonl"),
            "sft_candidate": _descriptor(root, f"sft_candidate/{split}.jsonl"),
            "manifest": _descriptor(root, f"manifests/{split}.jsonl"),
        }
    source_summary = json.loads(
        (source_root / "summary.json").read_text(encoding="utf-8")
    )
    _json(
        root / "summary.json",
        {
            "quality_status": "passed",
            "unresolved_failure_count": 0,
            "prompt_contract_sha256": sha256_file(root / "prompt_contract.json"),
            "artifacts": artifacts,
            "source": {
                "root": str(source_root.resolve()),
                "summary_sha256": source_summary_sha,
                "handoff_sha256": source_handoff_sha,
                "merge_receipt_sha256": sha256_file(
                    source_root / "merge_receipt.json"
                ),
                "sample_id_sha256": source_id_sha,
                "split_counts": {split: 2 for split in SPLITS},
                "total_rows": 6,
                "artifacts": source_summary["artifacts"],
            },
        },
    )
    _json(
        root / "handoff.json",
        {
            "summary": {
                "path": "summary.json",
                "sha256": sha256_file(root / "summary.json"),
            }
        },
    )


def _fixture(tmp_path: Path):
    source = tmp_path / "source"
    acquisition = tmp_path / "acquisition"
    release = tmp_path / "release"
    source_rows, summary_sha, handoff_sha, id_sha = _source_fixture(source)
    _acquisition_fixture(
        acquisition,
        source_rows,
        source_root=source,
        source_summary_sha=summary_sha,
        source_handoff_sha=handoff_sha,
        source_id_sha=id_sha,
    )
    kwargs = {
        "source_root": source,
        "acquisition_root": acquisition,
        "release_root": release,
        "tokenizer": FakeDeepSeekTokenizer(),
        "tokenizer_path": tmp_path / "tokenizer",
        "expected_source_summary_sha256": summary_sha,
        "expected_source_handoff_sha256": handoff_sha,
        "expected_source_sample_id_sha256": id_sha,
        "expected_split_counts": {split: 2 for split in SPLITS},
    }
    return source_rows, acquisition, release, kwargs


def _refresh_acquisition(acquisition: Path) -> None:
    summary = json.loads((acquisition / "summary.json").read_text(encoding="utf-8"))
    for split in SPLITS:
        for name, directory in (
            ("terminal", "terminal"),
            ("sft_candidate", "sft_candidate"),
            ("manifest", "manifests"),
        ):
            summary["artifacts"][split][name] = _descriptor(
                acquisition, f"{directory}/{split}.jsonl"
            )
    _json(acquisition / "summary.json", summary)
    _json(
        acquisition / "handoff.json",
        {
            "summary": {
                "path": "summary.json",
                "sha256": sha256_file(acquisition / "summary.json"),
            }
        },
    )


def test_publishes_pass_only_partition_and_keeps_reference_score_diagnostic(
    tmp_path: Path,
) -> None:
    _, _, release, kwargs = _fixture(tmp_path)

    handoff = publish_release(**kwargs)

    assert handoff["dataset_role"] == DATASET_ROLE
    assert handoff["split_counts"] == {split: 1 for split in SPLITS}
    for split in SPLITS:
        rows = [
            json.loads(line)
            for line in (release / f"minutes_alignment/{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(rows) == 1
        assert set(rows[0]) == {"prompt", "response"}
        sidecar = json.loads(
            (release / f"minutes_alignment/manifests/{split}.jsonl")
            .read_text(encoding="utf-8")
        )
        assert sidecar["release_index"] == 0
        assert sidecar["source_index"] == 1
        assert sidecar["reference_diagnostic"] == {
            "complete": True,
            "diagnostic_pass": False,
            "warning": "REFERENCE_DIAGNOSTIC_WARNING",
            "overall_score": 12,
            "used_for_selection": False,
            "used_for_repair": False,
            "record_sha256": sidecar["reference_diagnostic"]["record_sha256"],
        }
    audit = json.loads((release / "audits/data_quality.json").read_text())
    assert audit["source_rows"] == 6
    assert audit["pass_rows"] == 3
    assert audit["reject_rows"] == 3
    assert audit["reference_score_used_for_selection"] is False
    assert audit["integrity"]["pass_reject_exact_source_partition"] is True
    assert len((release / "audits/rejections.jsonl").read_text().splitlines()) == 3
    assert len(
        (release / "audits/reference_diagnostics.jsonl").read_text().splitlines()
    ) == 3
    replay = json.loads((release / "audits/tokenizer_replay.json").read_text())
    assert replay["contract"]["single_bos"] is True
    assert replay["contract"]["reasoning_boundary_answer_eos_unmasked"] is True


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda row: row.update(training_pass=False),
            "PASS is not deterministic\\+A eligibility",
        ),
        (
            lambda row: row["reference_diagnostic"].update(complete=False),
            "reference diagnostic.*incomplete",
        ),
        (
            lambda row: row.update(terminal_status="UNRESOLVED"),
            "unresolved terminal status",
        ),
        (
            lambda row: row["lineage"].update(
                official_reference_score_used_for_selection=True
            ),
            "terminal lineage drift",
        ),
    ],
)
def test_fails_closed_on_terminal_contract_drift(
    tmp_path: Path, mutation, match: str
) -> None:
    _, acquisition, _, kwargs = _fixture(tmp_path)
    path = acquisition / "terminal/train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutation(rows[1])
    _jsonl(path, rows)
    _refresh_acquisition(acquisition)

    with pytest.raises(PublicationError, match=match):
        publish_release(**kwargs)


def test_diagnostic_verdict_and_score_do_not_filter_training_pass(tmp_path: Path) -> None:
    _, acquisition, release, kwargs = _fixture(tmp_path)
    path = acquisition / "terminal/train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["reference_diagnostic"].update(
        diagnostic_pass=False,
        overall_score=0,
        warning="REFERENCE_DIAGNOSTIC_WARNING",
    )
    _jsonl(path, rows)
    candidate = [{"prompt": rows[1]["student_prompt"], "response": rows[1]["sft_response"]}]
    _jsonl(acquisition / "sft_candidate/train.jsonl", candidate)
    _refresh_acquisition(acquisition)

    publish_release(**kwargs)

    assert len((release / "minutes_alignment/train.jsonl").read_text().splitlines()) == 1


def test_rejects_empty_pass_split_without_a_coverage_floor(tmp_path: Path) -> None:
    _, acquisition, _, kwargs = _fixture(tmp_path)
    path = acquisition / "terminal/test.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1] = _terminal_row(
        {
            "candidate": {
                "prompt": rows[1]["student_prompt"],
                "response": "unused",
            },
            "manifest": {
                "sample_id": rows[1]["sample_id"],
                "split": "test",
                "analysis": rows[1]["source_analysis"],
                "official_minutes_sha256": rows[1]["official_minutes_sha256"],
                "meeting_date": rows[1]["meeting_date"],
            },
            "source_index": 1,
        },
        passed=False,
    )
    _jsonl(path, rows)
    _jsonl(acquisition / "sft_candidate/test.jsonl", [])
    _jsonl(acquisition / "manifests/test.jsonl", [])
    _refresh_acquisition(acquisition)

    with pytest.raises(PublicationError, match="split is empty: test"):
        publish_release(**kwargs)


def test_rejects_secret_material_even_when_artifacts_are_hash_bound(tmp_path: Path) -> None:
    _, acquisition, _, kwargs = _fixture(tmp_path)
    path = acquisition / "terminal/train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["generation"]["provider"]["api_key"] = "sk-secret-secret-secret"
    _jsonl(path, rows)
    _refresh_acquisition(acquisition)

    with pytest.raises(PublicationError, match="credential material"):
        publish_release(**kwargs)


def test_verifier_detects_published_byte_drift(tmp_path: Path) -> None:
    _, _, release, kwargs = _fixture(tmp_path)
    publish_release(**kwargs)
    manifest_sha = sha256_file(release / "release_manifest.json")
    assert (
        verify_release(release, expected_manifest_sha256=manifest_sha)["quality_status"]
        == "passed"
    )
    path = release / "minutes_alignment/train.jsonl"
    path.write_text(path.read_text() + "{}\n", encoding="utf-8")

    with pytest.raises(PublicationError, match="SHA-256 mismatch"):
        verify_release(release, expected_manifest_sha256=manifest_sha)


def test_response_over_4096_tokens_is_retained_without_truncation(tmp_path: Path) -> None:
    _, acquisition, release, kwargs = _fixture(tmp_path)
    path = acquisition / "terminal/train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = rows[1]
    reasoning = "R" * 5000
    row["teacher_response_analysis"] = reasoning
    row["teacher_response_analysis_sha256"] = sha256_text(reasoning)
    row["sft_response"] = reasoning + BOUNDARY + row["rewritten_minutes"]
    row["response_sha256"] = sha256_text(row["sft_response"])
    _jsonl(path, rows)
    _jsonl(
        acquisition / "sft_candidate/train.jsonl",
        [{"prompt": row["student_prompt"], "response": row["sft_response"]}],
    )
    _refresh_acquisition(acquisition)

    publish_release(**kwargs)
    replay = json.loads((release / "audits/tokenizer_replay.json").read_text())
    assert replay["contract"]["total_length_gate"] is False
    assert replay["contract"]["total_max"] is None
    assert replay["token_stats"]["total"]["max"] > 4096


def test_release_identity_never_claims_canonical_chk2(tmp_path: Path) -> None:
    _, _, release, kwargs = _fixture(tmp_path)
    publish_release(**kwargs)
    manifest = json.loads((release / "release_manifest.json").read_text())
    assert manifest["schema_version"] == RELEASE_SCHEMA_VERSION
    assert manifest["dataset_role"] == DATASET_ROLE
    assert manifest["canonical_stage_id"] is None
    assert manifest["dag_bindable"] is False
    assert manifest["promotable_as_canonical_chk2"] is False


def test_live_generator_and_publisher_contracts_are_exactly_compatible() -> None:
    analysis = "Economic activity remained steady."
    official = "Economic activity was unchanged."
    prompt = generator.render_user_prompt(analysis)
    prepared = generator.PreparedRow(
        sample_id="contract-sample",
        split="train",
        source_index=0,
        meeting_date="2026-01-01",
        source_analysis=analysis,
        official_minutes=official,
        student_prompt=prompt,
        source_analysis_sha256=sha256_text(analysis),
        official_minutes_sha256=sha256_text(official),
        prompt_sha256=sha256_text(prompt),
        source_manifest_sha256="a" * 64,
        source_response_sha256="b" * 64,
    )
    terminal = generator._base_terminal_record(prepared)

    assert terminal["schema_version"] == TERMINAL_SCHEMA_VERSION
    assert set(terminal) == publisher.TERMINAL_FIELDS
    assert "official_minutes" not in terminal
    assert terminal["lineage"] == LINEAGE

    contract = generator._prompt_contract(
        code_sha256="c" * 64,
        config=generator.ProviderConfig(),
    )
    assert publisher._student_system_prompt(contract) == generator.STUDENT_SYSTEM_PROMPT
    assert contract["student_user_prompt_contract"] == {
        "prefix": "Rewrite the following analysis as formal FOMC Minutes prose:\n\n",
        "json_keys": ["analysis"],
        "source_field": "source_analysis",
    }
    assert contract["selection_policy"]["validator_b_diagnostic_only"] is True
    assert contract["selection_policy"]["validator_b_never_repairs"] is True
    assert contract["selection_policy"]["validator_b_never_changes_training_pass"] is True
