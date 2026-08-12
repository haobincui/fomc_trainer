from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jobs.retrain_v2.chk1.source_data import (
    PROMPT_TOKEN_LIMIT,
    STUDENT_PROMPT_REQUIRED_FIELDS,
    CanonicalConflictError,
    EvidenceConflictError,
    InventoryError,
    PromptBudgetError,
    StudentPromptSchemaError,
    build_canonical_samples,
    build_fact_card,
    build_file_inventory,
    evidence_from_loo_ledger,
    make_huggingface_token_counter,
    split_atomic_topics,
    stable_sample_id,
    validate_prompt_token_budget,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _student_row(
    *,
    sample_id: str = "legacy-1",
    meeting_date: str = "2024-01-31",
    split: str = "train",
    topic: str = "GDP Growth",
    section_name: str = "Staff Review of the Economic Situation",
    source_row_index: int = 7,
) -> dict:
    prompt = f"Analyze {topic} for {meeting_date}."
    return {
        "sample_id": sample_id,
        "split": split,
        "meeting_date": meeting_date,
        "section_name": section_name,
        "topic": topic,
        "source_row_index": source_row_index,
        "rate_change": 25,
        "current_rate": 5.5,
        "prompt": prompt,
        "prompt_hash": _sha256_text(prompt),
        "provided_data": "legacy revised table",
        "data_label": "legacy.csv",
        "reference_excerpt": "",
        "reference_source_status": "legacy_prompt",
        "has_nonempty_table": True,
        "table_count": 1,
        "data_table_count": 1,
        "header_only_table_count": 0,
        "prompt_length_chars": len(prompt),
        "prompt_length_words": len(prompt.split()),
        "quality_flags": [],
        "missing_indicators": [],
        "source_files": ["legacy.csv"],
        "archived_response": "privileged old target",
        "response_origin": "archived_response",
    }


def _macro_evidence(
    *,
    evidence_id: str = "ev-gdp-1",
    observation_date: str = "2023-10-01",
    release_ts: str | None = "2024-01-25T13:30:00Z",
    availability_upper_bound_ts: str | None = None,
    vintage_date: str | None = "2024-01-30",
    value: str = "3.2",
    fact_kind: str = "recent_observation",
    formula: str = "",
    operand_evidence_ids: list[str] | None = None,
) -> dict:
    return {
        "evidence_id": evidence_id,
        "meeting_date": "2024-01-31",
        "atomic_topic": "GDP Growth",
        "source_kind": "macro",
        "source_id": "canonical-loo-d1:test:2024-01-31:GDP Growth",
        "source_sha256": SHA_A,
        "series_id": "GDPC1",
        "metric": "Real Gross Domestic Product",
        "fact_kind": fact_kind,
        "value": value,
        "units": "Percent change at annual rate",
        "observation_date": observation_date,
        "release_ts": release_ts,
        "availability_upper_bound_ts": availability_upper_bound_ts,
        "cutoff_ts": "2024-01-31T18:59:59Z",
        "requested_vintage_date": vintage_date,
        "information_as_of_date": vintage_date,
        "availability_as_of_date": vintage_date,
        "availability_evidence_type": "alfred_vintage_snapshot",
        "source_interface": "alfred-graph-csv-v1",
        "formula": formula,
        "operand_evidence_ids": operand_evidence_ids or [],
        "lineage": {
            "raw_sha256": SHA_B,
            "request_id": SHA_C,
            "snapshot_manifest_payload_sha256": SHA_D,
            "registry_sha256": "e" * 64,
        },
    }


def test_inventory_binds_path_bytes_rows_hash_and_privileged_classification(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "dataset/processed_llama/train/analysis_sft/train.jsonl"
    )
    source.parent.mkdir(parents=True)
    body = '{"row":1}\n\n{"row":2}\n'
    source.write_text(body, encoding="utf-8")

    inventory = build_file_inventory([source], repo_root=tmp_path)

    assert inventory["file_count"] == 1
    assert inventory["total_bytes"] == len(body.encode("utf-8"))
    assert len(inventory["payload_sha256"]) == 64
    assert inventory["files"] == [
        {
            "path": "dataset/processed_llama/train/analysis_sft/train.jsonl",
            "bytes": len(body.encode("utf-8")),
            "rows": 2,
            "sha256": _sha256_text(body),
            "classification": "legacy_privileged",
        }
    ]


def test_inventory_rejects_duplicate_alias_and_repo_escape(tmp_path: Path) -> None:
    source = tmp_path / "a.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    with pytest.raises(InventoryError, match="Duplicate inventory path"):
        build_file_inventory([source, "a.jsonl"], repo_root=tmp_path)

    outside = tmp_path.parent / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    try:
        with pytest.raises(InventoryError, match="escapes repository"):
            build_file_inventory([outside], repo_root=tmp_path)
    finally:
        outside.unlink()


def test_declared_student_schema_matches_the_real_processed_row() -> None:
    path = Path(
        "dataset/processed/pipeline/analysis_sft/student_prompts/"
        "after_2009/train.jsonl"
    )
    first = json.loads(path.open("r", encoding="utf-8").readline())
    assert set(first) == STUDENT_PROMPT_REQUIRED_FIELDS
    assert first["reference_excerpt"] == ""
    assert "archived_response" in first


def test_atomic_topics_and_canonical_projection_drop_all_privileged_fields() -> None:
    row = _student_row(
        topic="GDP Growth, Personal Consumption Expenditures (PCE), Other"
    )
    batch = build_canonical_samples(
        [row],
        section_style_by_topic={
            "GDP Growth": "style-real-activity-v1",
            "Personal Consumption Expenditures (PCE)": "style-consumption-v1",
        },
    )

    assert split_atomic_topics(row["topic"]) == (
        "GDP Growth",
        "Personal Consumption Expenditures (PCE)",
        "Other",
    )
    assert len(batch["samples"]) == 2
    assert batch["exclusions"][0]["reason"] == "abnormal_topic"
    for sample in batch["samples"]:
        assert set(sample["canonical_key"]) == {"meeting_date", "atomic_topic"}
        assert sample["sample_id"] == stable_sample_id(
            sample["meeting_date"], sample["atomic_topic"]
        )
        serialized = json.dumps(sample)
        for forbidden in (
            "archived_response",
            "current_rate",
            "rate_change",
            "provided_data",
            "reference_excerpt",
            "prompt",
        ):
            assert forbidden not in serialized


def test_exact_duplicate_canonical_metadata_collapses_without_row_order_effect() -> None:
    first = _student_row(sample_id="legacy-a", source_row_index=1)
    second = _student_row(sample_id="legacy-b", source_row_index=2)
    style = {"GDP Growth": "style-gdp-v1"}

    forward = build_canonical_samples([first, second], section_style_by_topic=style)
    reverse = build_canonical_samples([second, first], section_style_by_topic=style)

    assert forward == reverse
    assert forward["samples"][0]["legacy_sample_ids"] == ["legacy-a", "legacy-b"]
    assert forward["samples"][0]["legacy_source_row_indices"] == [1, 2]


def test_canonical_sample_conflicts_fail_closed() -> None:
    first = _student_row(sample_id="a", split="train", source_row_index=1)
    second = _student_row(sample_id="b", split="eval", source_row_index=2)
    with pytest.raises(CanonicalConflictError, match="appears in both"):
        build_canonical_samples(
            [first, second], section_style_by_topic={"GDP Growth": "style-gdp"}
        )

    other_section = _student_row(
        sample_id="b",
        source_row_index=2,
        section_name="Participants' Views on the Economic Outlook",
    )
    with pytest.raises(CanonicalConflictError, match="conflicting section styles"):
        build_canonical_samples([first, other_section])


def test_student_prompt_schema_and_reference_leakage_fail_closed() -> None:
    row = _student_row()
    del row["source_files"]
    with pytest.raises(StudentPromptSchemaError, match="source_files"):
        build_canonical_samples([row])

    row = _student_row()
    row["reference_excerpt"] = "same-meeting Minutes text"
    with pytest.raises(StudentPromptSchemaError, match="not reference-free"):
        build_canonical_samples([row])


def test_fact_card_accepts_only_bound_d1_vintage_and_keeps_full_lineage() -> None:
    result = build_fact_card(
        meeting_date="2024-01-31",
        atomic_topic="GDP Growth",
        cutoff_ts="2024-01-31T13:59:59-05:00",
        evidence_rows=[_macro_evidence()],
    )

    assert result["status"] == "ready"
    assert result["exclusions"] == []
    card = result["fact_card"]
    assert card["canonical_key"] == {
        "meeting_date": "2024-01-31",
        "atomic_topic": "GDP Growth",
    }
    assert "facts" not in card
    assert card["evidence"][0]["fact_kind"] == "latest"
    assert card["evidence"][0]["release_ts"] == "2024-01-25T13:30:00Z"
    assert card["evidence"][0]["availability_upper_bound_ts"] is None
    assert card["evidence"][0]["availability_basis"] == "actual_release_ts"
    assert card["evidence"][0]["requested_vintage_date"] == "2024-01-30"
    lineage = card["evidence_lineage"][0]
    assert lineage["raw_sha256"] == SHA_B
    assert lineage["request_id"] == SHA_C
    assert lineage["snapshot_manifest_payload_sha256"] == SHA_D
    assert len(card["fact_card_sha256"]) == 64


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ({"requested_vintage_date": None}, "missing_vintage_provenance"),
        ({"requested_vintage_date": "2024-01-29"}, "vintage_cutoff_mismatch"),
        ({"release_ts": None}, "missing_availability_proof"),
        ({"release_ts": "2024-02-01T13:30:00Z"}, "release_after_cutoff"),
        ({"source_interface": "fred-latest-csv"}, "unverified_vintage"),
    ],
)
def test_unverifiable_or_post_cutoff_macro_evidence_is_excluded(
    mutation: dict,
    reason: str,
) -> None:
    row = _macro_evidence()
    row.update(mutation)
    if "requested_vintage_date" in mutation:
        row["information_as_of_date"] = mutation["requested_vintage_date"]
        row["availability_as_of_date"] = mutation["requested_vintage_date"]

    result = build_fact_card(
        meeting_date="2024-01-31",
        atomic_topic="GDP Growth",
        cutoff_ts="2024-01-31T18:59:59Z",
        evidence_rows=[row],
    )

    assert result["status"] == "excluded"
    assert result["fact_card"] is None
    assert [item["reason"] for item in result["exclusions"]] == [reason]


def test_sealed_d1_snapshot_proves_conservative_availability_not_actual_release() -> None:
    row = _macro_evidence(
        release_ts=None,
        availability_upper_bound_ts="2024-01-30T23:59:59Z",
    )
    result = build_fact_card(
        meeting_date="2024-01-31",
        atomic_topic="GDP Growth",
        cutoff_ts="2024-01-31T18:59:59Z",
        evidence_rows=[row],
    )

    assert result["status"] == "ready"
    evidence = result["fact_card"]["evidence"][0]
    assert evidence["release_ts"] is None
    assert evidence["availability_upper_bound_ts"] == "2024-01-30T23:59:59Z"
    assert evidence["availability_basis"] == "sealed_alfred_d1_upper_bound"
    lineage = result["fact_card"]["evidence_lineage"][0]
    assert lineage["availability_basis"] == "sealed_alfred_d1_upper_bound"


def test_upper_bound_without_exact_d1_binding_is_excluded() -> None:
    row = _macro_evidence(
        release_ts=None,
        availability_upper_bound_ts="2024-01-30T12:00:00Z",
    )
    result = build_fact_card(
        meeting_date="2024-01-31",
        atomic_topic="GDP Growth",
        cutoff_ts="2024-01-31T18:59:59Z",
        evidence_rows=[row],
    )
    assert result["status"] == "excluded"
    assert result["exclusions"][0]["reason"] == "availability_upper_bound_mismatch"


def test_derived_facts_require_surviving_formula_operands() -> None:
    base = _macro_evidence(evidence_id="ev-base")
    derived = _macro_evidence(
        evidence_id="ev-yoy",
        value="2.0",
        fact_kind="yoy",
        formula="(latest / lag_4q - 1) * 100",
        operand_evidence_ids=["ev-base", "ev-missing"],
    )
    result = build_fact_card(
        meeting_date="2024-01-31",
        atomic_topic="GDP Growth",
        cutoff_ts="2024-01-31T18:59:59Z",
        evidence_rows=[base, derived],
    )

    assert result["status"] == "ready"
    assert [fact["evidence_id"] for fact in result["fact_card"]["evidence"]] == [
        "ev-base"
    ]
    assert any(
        item["evidence_id"] == "ev-yoy"
        and item["reason"] == "missing_formula_operand"
        for item in result["exclusions"]
    )


def test_conflicting_evidence_id_and_factual_identity_fail_closed() -> None:
    first = _macro_evidence(evidence_id="same", value="3.2")
    second = _macro_evidence(evidence_id="same", value="9.9")
    with pytest.raises(EvidenceConflictError, match="binds different payloads"):
        build_fact_card(
            meeting_date="2024-01-31",
            atomic_topic="GDP Growth",
            cutoff_ts="2024-01-31T18:59:59Z",
            evidence_rows=[first, second],
        )

    second["evidence_id"] = "different-id"
    with pytest.raises(EvidenceConflictError, match="Conflicting values"):
        build_fact_card(
            meeting_date="2024-01-31",
            atomic_topic="GDP Growth",
            cutoff_ts="2024-01-31T18:59:59Z",
            evidence_rows=[first, second],
        )


def _loo_fixture() -> tuple[dict, dict]:
    observations = [{"date": "2023-10-01", "value": "3.2"}]
    source_payload = {
        "sampling_policy": {"version": "d1-frequency-aware-v1"},
        "series": [
            {
                "source_key": "alfred__GDPC1",
                "series_id": "GDPC1",
                "title": "Real Gross Domestic Product",
                "frequency": "quarterly",
                "units": "Percent change at annual rate",
                "seasonal_adjustment": "Seasonally Adjusted Annual Rate",
                "transformation": "identity",
                "availability_as_of_date": "2024-01-30",
                "requested_vintage_date": "2024-01-30",
                "observations": observations,
            }
        ],
    }
    evidence_binding = {
        "schema_version": "loo-indicator-source-evidence-v1",
        "population_id": "test-population",
        "sample_id": "2024-01-31::GDP Growth",
        "source_id": "canonical-loo-d1:test-population:2024-01-31:GDP Growth",
        "information_as_of_date": "2024-01-30",
        "registry_sha256": "e" * 64,
        "snapshot_manifest_payload_sha256": SHA_D,
        "source_payload_sha256": _sha256_text(_canonical_json(source_payload)),
        "requests": [
            {
                "source_key": "alfred__GDPC1",
                "series_id": "GDPC1",
                "license": "public-domain",
                "redistribution_allowed": True,
                "request_id": SHA_C,
                "raw_relative_path": "raw/alfred/GDPC1/response.csv",
                "raw_sha256": SHA_B,
                "byte_count": 100,
                "retrieved_at_utc": "2026-08-03T00:00:00Z",
                "request_batch": {
                    "vintage_dates": ["2024-01-30"],
                    "cosd": ["2022-01-01"],
                    "coed": ["2024-01-30"],
                },
                "selected_batch_index": 0,
                "requested_vintage_date": "2024-01-30",
                "observation_start": "2022-01-01",
                "observation_end": "2024-01-30",
                "candidate_observation_count": 1,
                "selected_observation_count": 1,
                "selected_observation_sha256": _sha256_text(
                    _canonical_json(observations)
                ),
            }
        ],
    }
    source_sha = _sha256_text(_canonical_json(evidence_binding))
    evidence = {**evidence_binding, "source_sha256": source_sha}
    ledger = {
        "schema_version": "canonical-loo-indicator-input-v2",
        "meeting_id": "2024-01-31",
        "sample_id": "2024-01-31::GDP Growth",
        "meeting_timestamp": "2024-01-31T18:59:59Z",
        "meeting_date": "2024-01-31",
        "indicator": "GDP Growth",
        "source_id": evidence["source_id"],
        "source_sha256": source_sha,
        "source_timestamp": "2026-08-03T00:00:00Z",
        "information_as_of_date": "2024-01-30",
        "requested_vintage_date": "2024-01-30",
        "availability_as_of_date": "2024-01-30",
        "availability_evidence_type": "alfred_vintage_snapshot",
        "source_interface": "alfred-graph-csv-v1",
        "observation_date": "2023-10-01",
        "source_payload": source_payload,
    }
    return ledger, evidence


def test_existing_loo_lineage_is_replayed_and_missing_release_still_excludes() -> None:
    ledger, evidence = _loo_fixture()
    candidates = evidence_from_loo_ledger(
        ledger,
        evidence,
        release_ts_by_observation={},
    )
    assert len(candidates) == 1
    assert candidates[0]["lineage"]["raw_sha256"] == SHA_B

    accepted_from_upper_bound = build_fact_card(
        meeting_date="2024-01-31",
        atomic_topic="GDP Growth",
        cutoff_ts="2024-01-31T18:59:59Z",
        evidence_rows=candidates,
    )
    assert accepted_from_upper_bound["status"] == "ready"
    upper_bound_evidence = accepted_from_upper_bound["fact_card"]["evidence"][0]
    assert upper_bound_evidence["release_ts"] is None
    assert (
        upper_bound_evidence["availability_basis"]
        == "sealed_alfred_d1_upper_bound"
    )

    candidates = evidence_from_loo_ledger(
        ledger,
        evidence,
        release_ts_by_observation={
            ("GDPC1", "2023-10-01"): "2024-01-25T13:30:00Z"
        },
    )
    accepted = build_fact_card(
        meeting_date="2024-01-31",
        atomic_topic="GDP Growth",
        cutoff_ts="2024-01-31T18:59:59Z",
        evidence_rows=candidates,
    )
    assert accepted["status"] == "ready"
    assert (
        accepted["fact_card"]["evidence"][0]["availability_basis"]
        == "actual_release_ts"
    )


def test_loo_slug_can_bind_the_canonical_display_topic_only() -> None:
    ledger, evidence = _loo_fixture()
    ledger["indicator"] = "GDP-Growth"
    candidates = evidence_from_loo_ledger(
        ledger,
        evidence,
        atomic_topic="GDP Growth",
    )
    assert {row["atomic_topic"] for row in candidates} == {"GDP Growth"}

    with pytest.raises(EvidenceConflictError, match="does not match"):
        evidence_from_loo_ledger(
            ledger,
            evidence,
            atomic_topic="Unemployment Rate",
        )


def test_loo_payload_or_evidence_tampering_fails_closed() -> None:
    ledger, evidence = _loo_fixture()
    ledger["source_payload"]["series"][0]["observations"][0]["value"] = "99"
    with pytest.raises(EvidenceConflictError, match="source_payload_sha256 mismatch"):
        evidence_from_loo_ledger(
            ledger,
            evidence,
            release_ts_by_observation={},
        )


def test_external_4096_token_gate_accepts_boundary_and_never_truncates() -> None:
    calls: list[str] = []

    def at_limit(text: str) -> int:
        calls.append(text)
        return PROMPT_TOKEN_LIMIT

    audit = validate_prompt_token_budget("final rendered prompt", token_counter=at_limit)
    assert calls == ["final rendered prompt"]
    assert audit == {
        "schema_version": "chk1-prompt-budget-v1",
        "prompt_sha256": _sha256_text("final rendered prompt"),
        "token_count": 4096,
        "max_tokens": 4096,
        "overflow_policy": "error",
        "truncated": False,
    }

    with pytest.raises(PromptBudgetError, match="truncation is forbidden"):
        validate_prompt_token_budget(
            "do not truncate me",
            token_counter=lambda _: PROMPT_TOKEN_LIMIT + 1,
        )


def test_huggingface_tokenizer_adapter_and_invalid_prompt_boundary() -> None:
    class FakeTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is True
            return list(range(len(text.split()) + 2))

    counter = make_huggingface_token_counter(FakeTokenizer())
    assert counter("one two three") == 5
    with pytest.raises(PromptBudgetError, match="encoding anomaly"):
        validate_prompt_token_budget("bad\ufffdtext", token_counter=counter)
    with pytest.raises(PromptBudgetError, match="non-negative integer"):
        validate_prompt_token_budget("text", token_counter=lambda _: True)
