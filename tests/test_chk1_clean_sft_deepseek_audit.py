from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jobs.retrain_v2.audit_chk1_clean_sft_deepseek import (
    BOUNDARY,
    INPUT_SCHEMA_VERSION,
    AuditItem,
    DeepSeekAuditError,
    _canonical_json,
    _plainly_affirmative_explanation,
    _project_task_prompt,
    assert_cache_has_no_raw_text,
    finalize_row,
    load_audit_input,
    parse_stage_a_visible,
    parse_stage_b_visible,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_candidate(root: Path, *, rows: int = 2) -> tuple[Path, str]:
    audits = root / "audits"
    audits.mkdir(parents=True)
    input_rows = []
    repair_rows = []
    ids = []
    for index in range(rows):
        sample_id = f"sample-{index}"
        split = "train" if index == 0 else "eval"
        evidence = json.dumps(
            {
                "schema_version": "chk1-point-in-time-fact-card-v1",
                "atomic_topic": "inflation",
                "evidence": [
                    {
                        "evidence_id": f"ev-{index}",
                        "metric": "CPI",
                        "value": 3.0,
                        "units": "percent",
                        "observation_date": "2020-01-01",
                    }
                ],
            },
            sort_keys=True,
        )
        prompt = f"Analyze inflation. Data: {evidence}. Use neutral prose."
        response = f"The evidence reports 3 percent.{BOUNDARY}Inflation was 3 percent."
        input_rows.append(
            {
                "schema_version": INPUT_SCHEMA_VERSION,
                "sample_id": sample_id,
                "split": split,
                "prompt": prompt,
                "provided_data": evidence,
                "candidate_response": response,
                "prompt_sha256": _sha(prompt),
                "provided_data_sha256": _sha(evidence),
                "candidate_response_sha256": _sha(response),
            }
        )
        repair_rows.append({"sample_id": sample_id})
        ids.append(sample_id)
    input_text = "".join(_canonical_json(row) + "\n" for row in input_rows)
    repair_text = "".join(_canonical_json(row) + "\n" for row in repair_rows)
    input_path = audits / "semantic_audit_input.jsonl"
    repair_path = audits / "repair_manifest.jsonl"
    input_path.write_text(input_text, encoding="utf-8")
    repair_path.write_text(repair_text, encoding="utf-8")
    manifest = {
        "schema_version": "chk1-clean-sft-candidate-v2",
        "quality_status": "pending_semantic_audit",
        "immutable_candidate": True,
        "changed_rows": rows,
        "changed_sample_ids_sha256": _sha(_canonical_json(sorted(ids))),
        "semantic_audit_input": {
            "path": "audits/semantic_audit_input.jsonl",
            "rows": rows,
            "sha256": hashlib.sha256(input_text.encode()).hexdigest(),
        },
        "repair_manifest": {
            "path": "audits/repair_manifest.jsonl",
            "rows": rows,
            "sha256": hashlib.sha256(repair_text.encode()).hexdigest(),
        },
    }
    manifest_path = root / "candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest_path, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def _item() -> AuditItem:
    evidence = '{"value":3.0}'
    prompt = f"Topic inflation. Data {evidence}."
    response = f"The reading is 3 percent.{BOUNDARY}Inflation is 3 percent."
    return AuditItem(
        sample_id="sample-1",
        split="train",
        prompt=prompt,
        provided_data=evidence,
        candidate_response=response,
        think="The reading is 3 percent.",
        answer="Inflation is 3 percent.",
        prompt_sha256=_sha(prompt),
        provided_data_sha256=_sha(evidence),
        candidate_response_sha256=_sha(response),
    )


def _stage_a_payload(violations: list[dict[str, str]]) -> str:
    return json.dumps(
        {
            "data_fidelity": 3,
            "trend_reasoning": 3,
            "policy_relevance": 4,
            "uncertainty_calibration": 3,
            "fomc_style": 4,
            "violations": violations,
        }
    )


def test_load_audit_input_binds_manifest_file_and_every_row(tmp_path: Path) -> None:
    manifest, manifest_sha = _write_candidate(tmp_path / "candidate")
    loaded = load_audit_input(
        manifest, expected_manifest_sha256=manifest_sha, expected_rows=2
    )
    assert len(loaded.items) == 2
    assert [row.sample_id for row in loaded.items] == ["sample-0", "sample-1"]
    assert loaded.input_sha256

    rows_path = manifest.parent / "audits/semantic_audit_input.jsonl"
    rows = rows_path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(rows[0])
    payload["candidate_response"] += " drift"
    rows[0] = _canonical_json(payload)
    rows_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(DeepSeekAuditError, match="semantic audit input SHA-256 mismatch"):
        load_audit_input(
            manifest, expected_manifest_sha256=manifest_sha, expected_rows=2
        )


def test_stage_a_quote_validation_is_section_aware_and_nonfactual_nonblocking() -> None:
    parsed = parse_stage_a_visible(
        _stage_a_payload(
            [
                {
                    "section": "answer",
                    "kind": "numerical",
                    "severity": "major",
                    "candidate_quote": "Inflation  is 3 percent.",
                    "explanation": "The value is contradicted by the point-in-time evidence.",
                },
                {
                    "section": "think",
                    "kind": "format",
                    "severity": "minor",
                    "candidate_quote": "The reading is 3 percent.",
                    "explanation": "This is a formatting concern only.",
                },
                {
                    "section": "think",
                    "kind": "factual",
                    "severity": "minor",
                    "candidate_quote": "text that does not occur",
                    "explanation": "This cannot be found in the evidence.",
                },
            ]
        ),
        think="The reading is 3 percent.",
        answer="Inflation is 3 percent.",
    )
    assert len(parsed["validated_violations"]) == 2
    assert [row["kind"] for row in parsed["proposed_blocking_violations"]] == [
        "numerical"
    ]
    assert parsed["invalid_violations"][0]["invalid_reason"] == (
        "quote_not_in_named_section"
    )


def test_affirmative_blocking_explanation_is_retried_not_counted() -> None:
    assert _plainly_affirmative_explanation(
        "This correctly matches and is supported by the evidence."
    )
    assert not _plainly_affirmative_explanation(
        "This initially matches one value but is contradicted by the later value."
    )
    with pytest.raises(DeepSeekAuditError, match="self-contradictorily affirmative"):
        parse_stage_a_visible(
            _stage_a_payload(
                [
                    {
                        "section": "answer",
                        "kind": "factual",
                        "severity": "major",
                        "candidate_quote": "Inflation is 3 percent.",
                        "explanation": "This correctly matches the supplied evidence.",
                    }
                ]
            ),
            think="The reading is 3 percent.",
            answer="Inflation is 3 percent.",
        )


@pytest.mark.parametrize(
    ("verdict", "basis"),
    [
        ("confirmed", "contradicted"),
        ("false_positive", "supported_or_derivable"),
        ("ambiguous", "ambiguous"),
    ],
)
def test_stage_b_requires_consistent_verdict_and_basis(verdict: str, basis: str) -> None:
    parsed = parse_stage_b_visible(
        json.dumps(
            {
                "verdict": verdict,
                "basis": basis,
                "explanation": "A concise adjudication.",
            }
        )
    )
    assert parsed["verdict"] == verdict
    assert parsed["explanation_truncated"] is False
    with pytest.raises(DeepSeekAuditError, match="verdict/basis mismatch"):
        parse_stage_b_visible(
            json.dumps(
                {
                    "verdict": "false_positive",
                    "basis": "contradicted",
                    "explanation": "Mismatch.",
                }
            )
        )


def test_stage_b_bounds_provider_explanation_without_losing_verdict() -> None:
    parsed = parse_stage_b_visible(
        json.dumps(
            {
                "verdict": "confirmed",
                "basis": "not_derivable",
                "explanation": "x" * 750,
            }
        )
    )
    assert parsed["verdict"] == "confirmed"
    assert len(parsed["explanation"]) == 600
    assert parsed["explanation_original_length"] == 750
    assert parsed["explanation_truncated"] is True


def test_final_status_uses_only_confirmed_and_ambiguity_fails_closed() -> None:
    item = _item()
    proposal = {
        "section": "answer",
        "kind": "numerical",
        "severity": "major",
        "candidate_quote": "Inflation is 3 percent.",
        "explanation": "The value is allegedly contradicted.",
    }
    stage_a = {
        "rubric": {key: 3 for key in (
            "data_fidelity",
            "trend_reasoning",
            "policy_relevance",
            "uncertainty_calibration",
            "fomc_style",
        )},
        "validated_violations": [proposal],
        "invalid_violations": [],
        "proposed_blocking_violations": [proposal],
        "attempt": 1,
        "visible_response_sha256": "a" * 64,
        "provider_reasoning_sha256": "b" * 64,
        "usage": {},
    }
    false_positive = {
        "proposal": proposal,
        "proposal_sha256": "c" * 64,
        "verdict": "false_positive",
        "basis": "supported_or_derivable",
        "explanation": "The value is directly supported.",
        "attempt": 1,
    }
    row = finalize_row(item, stage_a, None, [false_positive], [])
    assert row["status"] == "passed"
    assert row["blocking_violations"] == []
    assert row["false_positive_violations"] == [proposal]

    confirmed = {**false_positive, "verdict": "confirmed", "basis": "contradicted"}
    row = finalize_row(item, stage_a, None, [confirmed], [])
    assert row["status"] == "failed"
    assert row["blocking_violations"] == [proposal]

    ambiguous = {**false_positive, "verdict": "ambiguous", "basis": "ambiguous"}
    row = finalize_row(item, stage_a, None, [ambiguous], [])
    assert row["status"] == "review_required"


def test_cache_security_and_prompt_projection() -> None:
    item = _item()
    projected = _project_task_prompt(item)
    assert item.provided_data not in projected
    assert "POINT_IN_TIME_EVIDENCE_SUPPLIED_SEPARATELY" in projected
    assert_cache_has_no_raw_text(
        {
            "binding": {"candidate_response_sha256": item.candidate_response_sha256},
            "provider_reasoning_sha256": "a" * 64,
            "rubric": {"data_fidelity": 4},
        }
    )
    with pytest.raises(DeepSeekAuditError, match="forbidden raw field"):
        assert_cache_has_no_raw_text({"hidden_reasoning": "must not be stored"})
