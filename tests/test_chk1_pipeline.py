from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from jobs.retrain_v2.chk1 import pipeline as pipeline_module
from jobs.retrain_v2.chk1.contracts import (
    GENERATOR_SYSTEM_PROMPT,
    canonical_json,
    sha256_text,
)
from jobs.retrain_v2.chk1.pipeline import (
    MINUTES_REFERENCE_HASH_POLICY,
    PRECANONICAL_EXCLUSION_SCHEMA_VERSION,
    PreparationError,
    build_local_token_counters,
    build_minutes_resolver,
    compute_minutes_reference_sha256,
    normalize_minutes_reference_members,
    prepare_chk1_data,
)
from jobs.retrain_v2.chk1.source_data import stable_sample_id
from jobs.retrain_v2.chk1.topic_styles import (
    ATOMIC_TOPICS,
    ECONOMIC_SECTION,
    ledger_indicator_for_topic,
)


_MEETINGS = {
    "train": "2020-01-29",
    "eval": "2020-03-18",
    "test": "2020-04-29",
}
_MINUTES = {
    "train": "Activity appeared to improve while uncertainty remained elevated.",
    "eval": "Demand seemed firmer, although several risks could persist.",
    "test": "Output reportedly edged higher while conditions remained uncertain.",
}
_SECOND_TRAIN_MINUTES = (
    "Conditions remained mixed, although activity could recover gradually."
)
_GENERATOR_TOKENIZER_SHA256 = "1" * 64
_STUDENT_TOKENIZER_SHA256 = "2" * 64


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sealed(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **payload,
        "integrity": {
            "algorithm": "sha256(canonical-json-without-integrity)",
            "payload_sha256": sha256_text(canonical_json(payload)),
        },
    }


def _student_row(split: str, *, leaked_reference: bool = False) -> dict[str, Any]:
    meeting = _MEETINGS[split]
    sample_id = f"legacy-{split}-gdp"
    prompt = f"Reference-free legacy prompt for {meeting}"
    return {
        "sample_id": sample_id,
        "split": split,
        "meeting_date": meeting,
        "section_name": ECONOMIC_SECTION,
        "topic": "GDP Growth",
        "source_row_index": {"train": 1, "eval": 2, "test": 3}[split],
        "rate_change": "",
        "current_rate": "",
        "prompt": prompt,
        "prompt_hash": sha256_text(prompt),
        "provided_data": "",
        "data_label": "",
        "reference_excerpt": _MINUTES[split] if leaked_reference else "",
        "reference_source_status": "available",
        "has_nonempty_table": True,
        "table_count": 1,
        "data_table_count": 1,
        "header_only_table_count": 0,
        "prompt_length_chars": len(prompt),
        "prompt_length_words": len(prompt.split()),
        "quality_flags": [],
        "missing_indicators": [],
        "source_files": [],
        "archived_response": "",
        "response_origin": "",
    }


def _minutes_row(split: str) -> dict[str, Any]:
    return {
        "sample_id": f"legacy-{split}-gdp",
        "split": split,
        "meeting_date": _MEETINGS[split],
        "section_name": ECONOMIC_SECTION,
        "source_row_index": {"train": 1, "eval": 2, "test": 3}[split],
        "reference_excerpt": _MINUTES[split],
    }


def _ledger_rows(
    *,
    population_id: str,
    meeting: str,
    registry_sha256: str,
    snapshot_payload_sha256: str,
    future_gdp: bool,
    observations_per_series: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    meeting_date = date.fromisoformat(meeting)
    as_of = (meeting_date - timedelta(days=1)).isoformat()
    safe_observation = (meeting_date - timedelta(days=7)).isoformat()
    ledger_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for index, topic in enumerate(ATOMIC_TOPICS):
        indicator = ledger_indicator_for_topic(topic)
        sample_id = f"{meeting}::{indicator}"
        source_key = f"alfred__fixture_{index:02d}"
        series_id = f"FIXTURE_SERIES_{index:02d}"
        observation_date = (
            (meeting_date + timedelta(days=1)).isoformat()
            if future_gdp and topic == "GDP Growth"
            else safe_observation
        )
        observations = [
            {
                "date": (
                    observation_date
                    if observation_index == observations_per_series - 1
                    else (
                        meeting_date
                        - timedelta(days=7 * (observations_per_series - observation_index))
                    ).isoformat()
                ),
                "value": str((index + 1) * 100 + observation_index),
            }
            for observation_index in range(observations_per_series)
        ]
        source_payload = {
            "sampling_policy": {"version": "fixture-v1"},
            "series": [
                {
                    "source_key": source_key,
                    "series_id": series_id,
                    "title": f"Fixture metric {index:02d}",
                    "frequency": "Monthly",
                    "units": "Index",
                    "seasonal_adjustment": "Not Seasonally Adjusted",
                    "transformation": "level",
                    "availability_as_of_date": as_of,
                    "requested_vintage_date": as_of,
                    "observations": observations,
                }
            ],
        }
        request = {
            "source_key": source_key,
            "series_id": series_id,
            "request_id": sha256_text(f"request:{population_id}:{sample_id}"),
            "raw_sha256": sha256_text(f"raw:{population_id}:{sample_id}"),
            "selected_observation_sha256": sha256_text(canonical_json(observations)),
        }
        source_id = f"canonical-loo-d1:{population_id}:{meeting}:{indicator}"
        evidence_binding = {
            "schema_version": "loo-indicator-source-evidence-v1",
            "population_id": population_id,
            "sample_id": sample_id,
            "source_id": source_id,
            "information_as_of_date": as_of,
            "registry_sha256": registry_sha256,
            "snapshot_manifest_payload_sha256": snapshot_payload_sha256,
            "source_payload_sha256": sha256_text(canonical_json(source_payload)),
            "requests": [request],
        }
        source_sha256 = sha256_text(canonical_json(evidence_binding))
        evidence_rows.append({**evidence_binding, "source_sha256": source_sha256})
        ledger_rows.append(
            {
                "schema_version": "canonical-loo-indicator-input-v2",
                "meeting_id": meeting,
                "sample_id": sample_id,
                "meeting_timestamp": f"{meeting}T18:59:59Z",
                "meeting_date": meeting,
                "indicator": indicator,
                "source_id": source_id,
                "source_sha256": source_sha256,
                "source_timestamp": f"{as_of}T23:00:00Z",
                "information_as_of_date": as_of,
                "requested_vintage_date": as_of,
                "availability_as_of_date": as_of,
                "availability_evidence_type": "alfred_vintage_snapshot",
                "source_interface": "alfred-graph-csv-v1",
                "observation_date": observation_date,
                "source_payload": source_payload,
            }
        )
        coverage_rows.append(
            {
                "sample_id": sample_id,
                "meeting_date": meeting,
                "information_as_of_date": as_of,
                "indicator": indicator,
                "source_count": 1,
                "observation_count": len(observations),
                "source_keys": [source_key],
                "latest_observation_date": observation_date,
            }
        )
    return ledger_rows, evidence_rows, coverage_rows


def _make_ledger(
    root: Path,
    *,
    population: dict[str, Any],
    population_path: Path,
    split: str,
    registry_path: Path,
    roster_path: Path,
    future_gdp: bool,
    observations_per_series: int = 1,
) -> dict[str, Any]:
    population_id = population["population_id"]
    meeting = population["meeting_dates"][0]
    ledger_dir = root / "source" / "ledgers" / population_id
    ledger_dir.mkdir(parents=True)

    snapshot_payload = {
        "schema_version": "loo-source-snapshot-manifest-v1",
        "status": "complete",
        "population_id": population_id,
        "request_count": len(ATOMIC_TOPICS),
    }
    snapshot = _sealed(snapshot_payload)
    snapshot_path = (
        root / "source" / "snapshots" / population_id / "snapshot_manifest.json"
    )
    _write_json(snapshot_path, snapshot)
    snapshot_payload_sha256 = snapshot["integrity"]["payload_sha256"]

    ledger_rows, evidence_rows, coverage_rows = _ledger_rows(
        population_id=population_id,
        meeting=meeting,
        registry_sha256=_sha256_file(registry_path),
        snapshot_payload_sha256=snapshot_payload_sha256,
        future_gdp=future_gdp,
        observations_per_series=observations_per_series,
    )
    indicator_path = ledger_dir / "indicator_inputs.jsonl"
    evidence_path = ledger_dir / "source_evidence.jsonl"
    excluded_path = ledger_dir / "excluded_records.jsonl"
    coverage_path = ledger_dir / "coverage.json"
    _write_jsonl(indicator_path, ledger_rows)
    _write_jsonl(evidence_path, evidence_rows)
    _write_jsonl(excluded_path, [])
    coverage = {
        "schema_version": "loo-indicator-ledger-coverage-v1",
        "population_id": population_id,
        "meeting_count": 1,
        "indicator_count": len(ATOMIC_TOPICS),
        "row_count": len(ledger_rows),
        "source_series_count": len(ledger_rows),
        "observation_count": len(ledger_rows),
        "excluded_record_count": 0,
        "information_as_of_dates": {
            meeting: (date.fromisoformat(meeting) - timedelta(days=1)).isoformat()
        },
        "rows": coverage_rows,
    }
    _write_json(coverage_path, coverage)

    manifest_payload = {
        "schema_version": "loo-indicator-ledger-manifest-v1",
        "status": "complete",
        "population_id": population_id,
        "information_cutoff_policy": "sealed D-1 fixture",
        "license_summary": {
            "local_research_use_only": False,
            "redistribution_allowed_for_all_sources": True,
            "sources": [],
        },
        "inputs": {
            "registry": {
                "path": str(registry_path),
                "sha256": _sha256_file(registry_path),
            },
            "snapshot_manifest": {
                "path": str(snapshot_path),
                "sha256": _sha256_file(snapshot_path),
                "payload_sha256": snapshot_payload_sha256,
            },
            "population": {
                "path": str(population_path),
                "sha256": _sha256_file(population_path),
            },
            "roster": {
                "path": str(roster_path),
                "sha256": _sha256_file(roster_path),
            },
        },
        "outputs": {
            "indicator_inputs": {
                "path": indicator_path.name,
                "sha256": _sha256_file(indicator_path),
                "row_count": len(ledger_rows),
            },
            "source_evidence": {
                "path": evidence_path.name,
                "sha256": _sha256_file(evidence_path),
                "row_count": len(evidence_rows),
            },
            "excluded_records": {
                "path": excluded_path.name,
                "sha256": _sha256_file(excluded_path),
                "row_count": 0,
            },
            "coverage": {
                "path": coverage_path.name,
                "sha256": _sha256_file(coverage_path),
            },
        },
    }
    manifest = _sealed(manifest_payload)
    manifest_path = ledger_dir / "ledger_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "population_id": population_id,
        "split": split,
        "meeting_count": 1,
        "ledger_dir": str(ledger_dir),
        "ledger_manifest_sha256": _sha256_file(manifest_path),
        "ledger_payload_sha256": manifest["integrity"]["payload_sha256"],
        "row_count": len(ledger_rows),
        "snapshot_manifest": str(snapshot_path),
        "snapshot_manifest_sha256": _sha256_file(snapshot_path),
    }


def _fixture_tree(
    tmp_path: Path,
    *,
    future_split: str | None = None,
    student_leak_split: str | None = None,
    multiple_train_references: bool = False,
    abnormal_train_topic: bool = False,
    observations_per_series: int = 1,
) -> dict[str, Path]:
    student_dir = tmp_path / "student_prompts"
    minutes_dir = tmp_path / "minutes_references"
    for split in _MEETINGS:
        student_rows = [
            _student_row(split, leaked_reference=student_leak_split == split)
        ]
        minutes_rows = [_minutes_row(split)]
        if split == "train" and multiple_train_references:
            alternate_student = _student_row(split)
            alternate_student.update(
                {
                    "sample_id": "legacy-train-gdp-alternate",
                    "source_row_index": 101,
                }
            )
            alternate_minutes = _minutes_row(split)
            alternate_minutes.update(
                {
                    "sample_id": "legacy-train-gdp-alternate",
                    "source_row_index": 101,
                    "reference_excerpt": _SECOND_TRAIN_MINUTES,
                }
            )
            student_rows.append(alternate_student)
            minutes_rows.append(alternate_minutes)
        if split == "train" and abnormal_train_topic:
            abnormal_student = _student_row(split)
            abnormal_student.update(
                {
                    "sample_id": "legacy-train-abnormal-topic",
                    "source_row_index": 102,
                    "topic": "Other",
                }
            )
            student_rows.append(abnormal_student)
        _write_jsonl(
            student_dir / f"{split}.jsonl",
            student_rows,
        )
        _write_jsonl(minutes_dir / f"{split}.jsonl", minutes_rows)

    registry_path = tmp_path / "source" / "registry.json"
    roster_path = tmp_path / "source" / "roster.json"
    _write_json(registry_path, {"schema_version": "fixture-registry-v1"})
    _write_json(
        roster_path,
        {
            "schema_version": "fixture-roster-v1",
            "indicators": [
                ledger_indicator_for_topic(topic) for topic in ATOMIC_TOPICS
            ],
        },
    )

    populations: list[dict[str, Any]] = []
    population_paths: dict[str, Path] = {}
    for split, meeting in _MEETINGS.items():
        population_id = f"fixture_{split}_01"
        population_payload = {
            "schema_version": "loo-population-v1",
            "population_id": population_id,
            "phase": "fixture",
            "split_label": split,
            "meeting_dates": [meeting],
        }
        population_path = tmp_path / "source" / "populations" / f"{population_id}.json"
        _write_json(population_path, population_payload)
        population_paths[population_id] = population_path
        populations.append(
            {
                "population_id": population_id,
                "split": split,
                "meeting_dates": [meeting],
                "path": str(population_path),
                "sha256": _sha256_file(population_path),
                "existing": False,
            }
        )

    plan_payload = {
        "schema_version": "chk1-source-plan-v1",
        "meeting_counts": {split: 1 for split in _MEETINGS},
        "populations": populations,
        "snapshot_batches": [],
    }
    plan = {
        **plan_payload,
        "payload_sha256": sha256_text(canonical_json(plan_payload)),
    }
    plan_path = tmp_path / "source" / "source_plan.json"
    _write_json(plan_path, plan)

    ledgers = [
        _make_ledger(
            tmp_path,
            population=population,
            population_path=population_paths[population["population_id"]],
            split=population["split"],
            registry_path=registry_path,
            roster_path=roster_path,
            future_gdp=population["split"] == future_split,
            observations_per_series=observations_per_series,
        )
        for population in populations
    ]
    handoff_payload = {
        "schema_version": "chk1-source-handoff-v1",
        "source_plan": {
            "path": str(plan_path),
            "sha256": _sha256_file(plan_path),
            "payload_sha256": plan["payload_sha256"],
        },
        "registry_sha256": _sha256_file(registry_path),
        "roster_sha256": _sha256_file(roster_path),
        "ledgers": ledgers,
    }
    handoff = {
        **handoff_payload,
        "payload_sha256": sha256_text(canonical_json(handoff_payload)),
    }
    handoff_path = tmp_path / "source" / "source_handoff.json"
    _write_json(handoff_path, handoff)
    return {
        "repo_root": tmp_path,
        "student_dir": student_dir,
        "minutes_dir": minutes_dir,
        "source_handoff": handoff_path,
        "output_dir": tmp_path / "prepared_bundle",
    }


def _convert_train_ledger_to_sparse(
    paths: dict[str, Path],
    *,
    excluded_topic: str,
) -> tuple[str, str]:
    handoff_path = paths["source_handoff"]
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    record = next(item for item in handoff["ledgers"] if item["split"] == "train")
    population_id = record["population_id"]
    meeting = _MEETINGS["train"]
    indicator = ledger_indicator_for_topic(excluded_topic)
    excluded_ledger_sample_id = f"{meeting}::{indicator}"
    ledger_dir = Path(record["ledger_dir"])
    snapshot_path = Path(record["snapshot_manifest"])

    snapshot_payload = {
        "schema_version": "chk1-sparse-source-snapshot-manifest-v1",
        "status": "complete",
        "population_id": population_id,
        "meeting_count": 1,
    }
    snapshot = _sealed(snapshot_payload)
    _write_json(snapshot_path, snapshot)
    snapshot_payload_sha256 = snapshot["integrity"]["payload_sha256"]

    ledger_rows = _read_jsonl(ledger_dir / "indicator_inputs.jsonl")
    evidence_rows = _read_jsonl(ledger_dir / "source_evidence.jsonl")
    evidence_by_sample: dict[str, dict[str, Any]] = {}
    for evidence in evidence_rows:
        if evidence["sample_id"] == excluded_ledger_sample_id:
            continue
        binding = {
            key: value for key, value in evidence.items() if key != "source_sha256"
        }
        binding["snapshot_manifest_payload_sha256"] = snapshot_payload_sha256
        source_sha256 = sha256_text(canonical_json(binding))
        evidence_by_sample[binding["sample_id"]] = {
            **binding,
            "source_sha256": source_sha256,
        }
    kept_ledger_rows: list[dict[str, Any]] = []
    for row in ledger_rows:
        if row["sample_id"] == excluded_ledger_sample_id:
            continue
        evidence = evidence_by_sample[row["sample_id"]]
        kept_ledger_rows.append(
            {
                **row,
                "source_id": evidence["source_id"],
                "source_sha256": evidence["source_sha256"],
            }
        )
    kept_evidence_rows = [
        evidence_by_sample[row["sample_id"]] for row in kept_ledger_rows
    ]
    indicator_path = ledger_dir / "indicator_inputs.jsonl"
    evidence_path = ledger_dir / "source_evidence.jsonl"
    excluded_path = ledger_dir / "excluded_records.jsonl"
    sample_exclusions_path = ledger_dir / "sample_exclusions.jsonl"
    coverage_path = ledger_dir / "coverage.json"
    _write_jsonl(indicator_path, kept_ledger_rows)
    _write_jsonl(evidence_path, kept_evidence_rows)
    _write_jsonl(excluded_path, [])

    as_of = (date.fromisoformat(meeting) - timedelta(days=1)).isoformat()
    source_exclusion_sample_id = stable_sample_id(meeting, excluded_topic)
    source_exclusion = {
        "schema_version": "chk1-source-exclusion-v1",
        "sample_id": source_exclusion_sample_id,
        "split": "train",
        "meeting_date": meeting,
        "atomic_topic": excluded_topic,
        "ledger_indicator": indicator,
        "reason_code": "no_proven_d1_vintage_evidence",
        "information_as_of_date": as_of,
        "configured_source_keys": ["alfred__unavailable_fixture"],
        "unusable_sources": [
            {
                "source_key": "alfred__unavailable_fixture",
                "reason_code": "exact_vintage_not_echoed",
            }
        ],
        "registry_sha256": handoff["registry_sha256"],
        "roster_sha256": handoff["roster_sha256"],
        "snapshot_manifest_payload_sha256": snapshot_payload_sha256,
    }
    _write_jsonl(sample_exclusions_path, [source_exclusion])
    _write_json(
        coverage_path,
        {
            "schema_version": "chk1-sparse-loo-coverage-v1",
            "population_id": population_id,
            "expected_sample_count": len(ATOMIC_TOPICS),
            "ready_sample_count": len(kept_ledger_rows),
            "excluded_sample_count": 1,
            "sample_coverage_complete": True,
        },
    )

    registry_path = ledger_dir.parents[1] / "registry.json"
    roster_path = ledger_dir.parents[1] / "roster.json"
    manifest_payload = {
        "schema_version": "chk1-sparse-loo-ledger-manifest-v1",
        "status": "complete",
        "population_id": population_id,
        "inputs": {
            "registry": {
                "path": str(registry_path),
                "sha256": _sha256_file(registry_path),
            },
            "roster": {
                "path": str(roster_path),
                "sha256": _sha256_file(roster_path),
            },
            "snapshot_manifest": {
                "path": str(snapshot_path),
                "sha256": _sha256_file(snapshot_path),
                "payload_sha256": snapshot_payload_sha256,
            },
        },
        "outputs": {
            "indicator_inputs": {
                "path": indicator_path.name,
                "sha256": _sha256_file(indicator_path),
                "row_count": len(kept_ledger_rows),
            },
            "source_evidence": {
                "path": evidence_path.name,
                "sha256": _sha256_file(evidence_path),
                "row_count": len(kept_evidence_rows),
            },
            "excluded_records": {
                "path": excluded_path.name,
                "sha256": _sha256_file(excluded_path),
                "row_count": 0,
            },
            "sample_exclusions": {
                "path": sample_exclusions_path.name,
                "sha256": _sha256_file(sample_exclusions_path),
                "row_count": 1,
            },
            "coverage": {
                "path": coverage_path.name,
                "sha256": _sha256_file(coverage_path),
            },
        },
    }
    manifest = _sealed(manifest_payload)
    manifest_path = ledger_dir / "ledger_manifest.json"
    _write_json(manifest_path, manifest)

    record.update(
        {
            "coverage_mode": "sparse",
            "ledger_manifest_sha256": _sha256_file(manifest_path),
            "ledger_payload_sha256": manifest["integrity"]["payload_sha256"],
            "row_count": len(kept_ledger_rows),
            "excluded_sample_count": 1,
            "expected_sample_count": len(ATOMIC_TOPICS),
            "sample_exclusions": {
                "path": str(sample_exclusions_path),
                "sha256": _sha256_file(sample_exclusions_path),
                "row_count": 1,
            },
            "snapshot_manifest_sha256": _sha256_file(snapshot_path),
            "snapshot_payload_sha256": snapshot_payload_sha256,
        }
    )
    handoff_payload = {
        key: value for key, value in handoff.items() if key != "payload_sha256"
    }
    handoff = {
        **handoff_payload,
        "payload_sha256": sha256_text(canonical_json(handoff_payload)),
    }
    _write_json(handoff_path, handoff)
    return population_id, source_exclusion_sample_id


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _prepare(
    paths: dict[str, Path],
    token_counter: Any = None,
    **kwargs: Any,
) -> Path:
    return prepare_chk1_data(
        student_prompt_dir=paths["student_dir"],
        minutes_reference_dir=paths["minutes_dir"],
        source_handoff_path=paths["source_handoff"],
        output_dir=paths["output_dir"],
        repo_root=paths["repo_root"],
        generator_tokenizer_sha256=kwargs.pop(
            "generator_tokenizer_sha256", _GENERATOR_TOKENIZER_SHA256
        ),
        student_tokenizer_sha256=kwargs.pop(
            "student_tokenizer_sha256", _STUDENT_TOKENIZER_SHA256
        ),
        token_counter=token_counter or (lambda text: len(text.split())),
        **kwargs,
    )


def test_preparation_is_reference_free_and_idempotent(tmp_path: Path) -> None:
    paths = _fixture_tree(tmp_path)
    handoff_path = _prepare(paths)
    first_hashes = _tree_hashes(paths["output_dir"])
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))

    assert handoff["counts"]["prepared"] == {"train": 1, "eval": 1, "test": 1}
    assert handoff["counts"]["precanonical_exclusions"] == 0
    assert handoff["minutes_reference_hash_policy"] == MINUTES_REFERENCE_HASH_POLICY
    assert handoff["generator_tokenizer_sha256"] == _GENERATOR_TOKENIZER_SHA256
    assert handoff["student_tokenizer_sha256"] == _STUDENT_TOKENIZER_SHA256
    assert handoff["artifacts"]["audit"]["precanonical_exclusions"]["row_count"] == 0
    assert len(handoff["payload_sha256"]) == 64
    for split in _MEETINGS:
        rows = _read_jsonl(paths["output_dir"] / "prepared" / f"{split}.jsonl")
        assert len(rows) == 1
        row = rows[0]
        assert row["minutes_reference_sha256"] == sha256_text(_MINUTES[split])
        assert row["evidence_lineage"]
        assert not _contains_key(row["fact_card"], "evidence_lineage")
        evidence_by_id = {
            item["evidence_id"]: item for item in row["fact_card"]["evidence"]
        }
        assert set(evidence_by_id) == {
            item["evidence_id"] for item in row["evidence_lineage"]
        }
        assert all(
            evidence_by_id[item["evidence_id"]]["source_id"] == item["source_id"]
            for item in row["evidence_lineage"]
        )
        assert row["fact_card_selection"]["policy"].endswith("no-truncation-v1")
        assert row["fact_card_selection"]["attempted_limits"] == [3]
        assert row["fact_card_selection"][
            "selected_max_recent_observations_per_series"
        ] == 3
        assert row["section_style_guide"]["section_style_id"] == row["section_style_id"]
        assert not _contains_key(row["section_style_guide"], "evidence_lineage")
        assert row["input_truncated"] is False
        assert "evidence_lineage" not in row["generator_prompt"]
        assert "evidence_lineage" not in row["student_prompt"]
        assert "evidence_lineage" not in row["provided_data"]
        assert all(text not in canonical_json(row) for text in _MINUTES.values())
        assert row["prompt_budget"]["student"]["max_tokens"] == 4096
        assert row["prompt_budget"]["student"]["truncated"] is False
        assert (
            row["prompt_budget"]["generator"]["tokenizer_sha256"]
            == _GENERATOR_TOKENIZER_SHA256
        )
        assert (
            row["prompt_budget"]["student"]["tokenizer_sha256"]
            == _STUDENT_TOKENIZER_SHA256
        )
        assert row["row_sha256"] == sha256_text(
            canonical_json(
                {key: value for key, value in row.items() if key != "row_sha256"}
            )
        )

    assert _prepare(paths) == handoff_path
    assert _tree_hashes(paths["output_dir"]) == first_hashes


def test_sparse_source_exclusion_closes_preparation_population(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture_tree(tmp_path)
    population_id, source_exclusion_sample_id = _convert_train_ledger_to_sparse(
        paths,
        excluded_topic="GDP Growth",
    )
    handoff = json.loads(paths["source_handoff"].read_text(encoding="utf-8"))
    sparse_record = next(
        item for item in handoff["ledgers"] if item["population_id"] == population_id
    )

    monkeypatch.setattr(
        pipeline_module,
        "validate_sparse_snapshot_manifest",
        lambda *_args, **_kwargs: {
            "status": "valid",
            "population_id": population_id,
            "manifest_payload_sha256": sparse_record["snapshot_payload_sha256"],
        },
    )
    monkeypatch.setattr(
        pipeline_module,
        "validate_sparse_loo_ledger",
        lambda *_args, **_kwargs: {
            "status": "valid",
            "population_id": population_id,
            "manifest_payload_sha256": sparse_record["ledger_payload_sha256"],
            "ready_sample_count": len(ATOMIC_TOPICS) - 1,
            "excluded_sample_count": 1,
            "expected_sample_count": len(ATOMIC_TOPICS),
        },
    )

    prepare_handoff = json.loads(_prepare(paths).read_text(encoding="utf-8"))
    assert prepare_handoff["counts"]["prepared"] == {
        "train": 0,
        "eval": 1,
        "test": 1,
    }
    assert prepare_handoff["counts"]["sample_exclusions"] == 1
    exclusions = _read_jsonl(
        paths["output_dir"] / "audit" / "sample_exclusions.jsonl"
    )
    assert len(exclusions) == 1
    exclusion = exclusions[0]
    assert exclusion["sample_id"] == source_exclusion_sample_id
    assert exclusion["stage"] == "source_evidence"
    assert exclusion["reason_code"] == "no_proven_d1_vintage_evidence"
    assert exclusion["details"]["coverage_mode"] == "sparse"
    assert exclusion["details"]["population_id"] == population_id
    assert exclusion["details"]["source_exclusion_sample_id"] == (
        source_exclusion_sample_id
    )
    assert exclusion["details"]["unusable_source_reason_codes"] == [
        "exact_vintage_not_echoed"
    ]


def test_minutes_resolver_deduplicates_and_hashes_multiple_references(
    tmp_path: Path,
) -> None:
    paths = _fixture_tree(tmp_path, multiple_train_references=True)
    resolver = build_minutes_resolver(
        student_prompt_dir=paths["student_dir"],
        minutes_reference_dir=paths["minutes_dir"],
    )
    _prepare(paths)
    row = _read_jsonl(paths["output_dir"] / "prepared" / "train.jsonl")[0]
    lookup = {
        field: row[field]
        for field in ("sample_id", "split", "meeting_date", "atomic_topic")
    }
    members = resolver(lookup)

    expected_members = tuple(
        sorted((_MINUTES["train"], _SECOND_TRAIN_MINUTES), key=sha256_text)
    )
    assert members == expected_members
    assert len(members) == 2
    assert row["minutes_reference_sha256"] == compute_minutes_reference_sha256(members)
    assert normalize_minutes_reference_members((*members, members[0])) == members
    assert compute_minutes_reference_sha256([members[0], members[0]]) == sha256_text(
        members[0]
    )

    output_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in paths["output_dir"].rglob("*")
        if path.is_file()
    )
    assert all(reference not in output_text for reference in expected_members)
    with pytest.raises(PreparationError, match="atomic_topic binding mismatch"):
        resolver({**lookup, "atomic_topic": "Unemployment Rate"})


def test_precanonical_exclusions_are_separate_from_release_sample_exclusions(
    tmp_path: Path,
) -> None:
    paths = _fixture_tree(tmp_path, abnormal_train_topic=True)
    handoff_path = _prepare(paths)
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))

    sample_exclusions = _read_jsonl(
        paths["output_dir"] / "audit" / "sample_exclusions.jsonl"
    )
    precanonical_exclusions = _read_jsonl(
        paths["output_dir"] / "audit" / "precanonical_exclusions.jsonl"
    )
    assert sample_exclusions == []
    assert len(precanonical_exclusions) == 1
    exclusion = precanonical_exclusions[0]
    assert exclusion["schema_version"] == PRECANONICAL_EXCLUSION_SCHEMA_VERSION
    assert exclusion["source_sample_id"] == "legacy-train-abnormal-topic"
    assert exclusion["split"] == "train"
    assert exclusion["atomic_topic"] == "Other"
    assert exclusion["stage"] == "canonical_population"
    assert exclusion["reason_code"] == "abnormal_topic"
    assert "sample_id" not in exclusion
    assert exclusion["exclusion_sha256"] == sha256_text(
        canonical_json(
            {
                key: value
                for key, value in exclusion.items()
                if key != "exclusion_sha256"
            }
        )
    )
    assert handoff["counts"]["canonical_samples"] == 3
    assert handoff["counts"]["sample_exclusions"] == 0
    assert handoff["counts"]["precanonical_exclusions"] == 1
    assert handoff["artifacts"]["audit"]["precanonical_exclusions"]["row_count"] == 1


def test_style_entry_is_checked_against_all_minutes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture_tree(tmp_path)
    original_builder = pipeline_module.build_style_guide_from_jsonl

    def contaminated_builder(*args: Any, **kwargs: Any) -> dict[str, Any]:
        artifact = original_builder(*args, **kwargs)
        entry = artifact["styles"][0]
        entry["guide_text"] = _MINUTES["eval"]
        entry["guide_text_sha256"] = sha256_text(entry["guide_text"])
        entry_payload = {
            key: value for key, value in entry.items() if key != "style_entry_sha256"
        }
        entry["style_entry_sha256"] = sha256_text(canonical_json(entry_payload))
        guide_payload = {
            key: value for key, value in artifact.items() if key != "style_guide_sha256"
        }
        artifact["style_guide_sha256"] = sha256_text(canonical_json(guide_payload))
        return artifact

    monkeypatch.setattr(
        pipeline_module,
        "build_style_guide_from_jsonl",
        contaminated_builder,
    )
    with pytest.raises(PreparationError, match="raw Minutes text"):
        _prepare(paths)
    assert not paths["output_dir"].exists()


def test_student_minutes_leak_fails_closed_without_output(tmp_path: Path) -> None:
    paths = _fixture_tree(tmp_path, student_leak_split="train")
    with pytest.raises(PreparationError):
        _prepare(paths)
    assert not paths["output_dir"].exists()


def test_future_evidence_is_audited_and_sample_is_excluded(tmp_path: Path) -> None:
    paths = _fixture_tree(tmp_path, future_split="train")
    _prepare(paths)

    assert _read_jsonl(paths["output_dir"] / "prepared" / "train.jsonl") == []
    evidence_exclusions = _read_jsonl(
        paths["output_dir"] / "audit" / "evidence_exclusions.jsonl"
    )
    assert any(
        row["split"] == "train" and row["reason_code"] == "observation_after_cutoff"
        for row in evidence_exclusions
    )
    sample_exclusions = _read_jsonl(
        paths["output_dir"] / "audit" / "sample_exclusions.jsonl"
    )
    assert any(
        row["split"] == "train"
        and row["stage"] == "fact_card"
        and row["reason_code"] == "fact_card_excluded"
        for row in sample_exclusions
    )


def test_student_prompt_overflow_is_excluded_without_truncation(tmp_path: Path) -> None:
    paths = _fixture_tree(tmp_path)

    def token_counter(text: str) -> int:
        if text.startswith("Analyze the supplied") and _MEETINGS["train"] in text:
            return 4097
        return 20

    _prepare(paths, token_counter=token_counter)
    assert _read_jsonl(paths["output_dir"] / "prepared" / "train.jsonl") == []
    exclusions = _read_jsonl(paths["output_dir"] / "audit" / "sample_exclusions.jsonl")
    overflow = next(row for row in exclusions if row["split"] == "train")
    assert overflow["reason_code"] == "prompt_token_overflow"
    assert overflow["stage"] == "student_prompt"
    assert overflow["details"]["token_count"] == 4097
    assert overflow["details"]["max_tokens"] == 4096
    assert overflow["details"]["truncated"] is False
    assert overflow["details"]["tokenizer_sha256"] == _STUDENT_TOKENIZER_SHA256


def test_fact_card_adapts_recent_observations_without_tokenizer_truncation(
    tmp_path: Path,
) -> None:
    paths = _fixture_tree(tmp_path, observations_per_series=3)

    def generator_counter(text: str) -> int:
        evidence_count = text.count('"evidence_id":')
        return 4097 if evidence_count >= 3 else 100

    _prepare(
        paths,
        generator_token_counter=generator_counter,
        student_token_counter=lambda _text: 90,
    )
    row = _read_jsonl(paths["output_dir"] / "prepared" / "train.jsonl")[0]
    assert len(row["fact_card"]["evidence"]) == 2
    assert row["fact_card_selection"]["attempted_limits"] == [3, 2]
    assert [
        attempt["status"] for attempt in row["fact_card_selection"]["attempts"]
    ] == ["generator_overflow", "selected"]
    assert row["fact_card_selection"][
        "selected_max_recent_observations_per_series"
    ] == 2
    assert row["prompt_budget"]["generator"]["token_count"] == 100
    assert row["input_truncated"] is False


def test_generator_and_student_use_distinct_token_counters(tmp_path: Path) -> None:
    paths = _fixture_tree(tmp_path)
    generator_prompts: list[str] = []
    student_prompts: list[str] = []

    def generator_counter(text: str) -> int:
        generator_prompts.append(text)
        return 19

    def student_counter(text: str) -> int:
        student_prompts.append(text)
        return 4097 if _MEETINGS["train"] in text else 23

    handoff_path = _prepare(
        paths,
        generator_token_counter=generator_counter,
        student_token_counter=student_counter,
    )
    assert generator_prompts and all(text.startswith("{") for text in generator_prompts)
    assert student_prompts and all(
        text.startswith("Analyze the supplied") for text in student_prompts
    )
    assert _read_jsonl(paths["output_dir"] / "prepared" / "train.jsonl") == []

    eval_row = _read_jsonl(paths["output_dir"] / "prepared" / "eval.jsonl")[0]
    assert eval_row["prompt_budget"]["generator"]["token_count"] == 19
    assert eval_row["prompt_budget"]["student"]["token_count"] == 23
    assert (
        eval_row["prompt_budget"]["generator"]["tokenizer_sha256"]
        == _GENERATOR_TOKENIZER_SHA256
    )
    assert (
        eval_row["prompt_budget"]["student"]["tokenizer_sha256"]
        == _STUDENT_TOKENIZER_SHA256
    )

    exclusion = next(
        row
        for row in _read_jsonl(
            paths["output_dir"] / "audit" / "sample_exclusions.jsonl"
        )
        if row["split"] == "train"
    )
    assert exclusion["stage"] == "student_prompt"
    assert exclusion["details"]["tokenizer_sha256"] == _STUDENT_TOKENIZER_SHA256
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    assert handoff["generator_tokenizer_sha256"] == _GENERATOR_TOKENIZER_SHA256
    assert handoff["student_tokenizer_sha256"] == _STUDENT_TOKENIZER_SHA256


def test_tokenizer_digest_provenance_is_required(tmp_path: Path) -> None:
    paths = _fixture_tree(tmp_path)
    with pytest.raises(PreparationError, match="generator_tokenizer_sha256"):
        _prepare(paths, generator_tokenizer_sha256="not-a-digest")
    assert not paths["output_dir"].exists()


def test_local_token_counters_mirror_generator_chat_and_raw_student(
    tmp_path: Path,
) -> None:
    for name in ("DeepSeek-R1-Distill-Llama-8B",):
        model = tmp_path / "models" / name
        model.mkdir(parents=True)
        (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (model / "tokenizer.json").write_text(
            json.dumps({"name": name}), encoding="utf-8"
        )

    class FakeTokenizer:
        def __init__(self, name: str) -> None:
            self.name = name
            self.encoded: list[str] = []
            self.template_calls: list[dict[str, Any]] = []

        def apply_chat_template(self, messages, **kwargs):
            self.template_calls.append({"messages": messages, **kwargs})
            return f"CHAT::{messages[0]['content']}::{messages[1]['content']}"

        def encode(self, text: str, *, add_special_tokens: bool):
            assert add_special_tokens is True
            self.encoded.append(text)
            return list(range(len(text.encode("utf-8"))))

    loaded: dict[str, FakeTokenizer] = {}

    def loader(path: Path) -> FakeTokenizer:
        tokenizer = FakeTokenizer(path.name)
        loaded[path.name] = tokenizer
        return tokenizer

    bundle = build_local_token_counters(
        repo_root=tmp_path,
        tokenizer_loader=loader,
    )
    prompt = "point-in-time prompt"
    generator_count = bundle.generator_token_counter(prompt)
    student_count = bundle.student_token_counter(prompt)

    student = loaded["DeepSeek-R1-Distill-Llama-8B"]
    assert student.template_calls[0]["messages"] == [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    assert "enable_thinking" not in student.template_calls[0]
    assert student.template_calls[0]["add_generation_prompt"] is True
    assert student.template_calls[0]["tokenize"] is False
    assert student.encoded == [
        f"CHAT::{GENERATOR_SYSTEM_PROMPT}::{prompt}",
        prompt,
    ]
    assert generator_count == len(student.encoded[0].encode("utf-8"))
    assert student_count == len(prompt.encode("utf-8"))
    assert bundle.generator_tokenizer_sha256 == bundle.student_tokenizer_sha256
    assert len(bundle.generator_tokenizer_sha256) == 64
    assert len(bundle.student_tokenizer_sha256) == 64


def test_missing_ledger_is_a_hard_failure(tmp_path: Path) -> None:
    paths = _fixture_tree(tmp_path)
    handoff = json.loads(paths["source_handoff"].read_text(encoding="utf-8"))
    handoff["ledgers"] = [row for row in handoff["ledgers"] if row["split"] != "eval"]
    payload = {key: value for key, value in handoff.items() if key != "payload_sha256"}
    handoff["payload_sha256"] = sha256_text(canonical_json(payload))
    _write_json(paths["source_handoff"], handoff)

    with pytest.raises(PreparationError, match="ledger coverage mismatch"):
        _prepare(paths)
    assert not paths["output_dir"].exists()


def test_tampered_ledger_bytes_fail_the_handoff_sha_gate(tmp_path: Path) -> None:
    paths = _fixture_tree(tmp_path)
    handoff = json.loads(paths["source_handoff"].read_text(encoding="utf-8"))
    ledger_dir = Path(handoff["ledgers"][0]["ledger_dir"])
    indicator_path = ledger_dir / "indicator_inputs.jsonl"
    indicator_path.write_text(
        indicator_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PreparationError, match="SHA-256 mismatch"):
        _prepare(paths)
    assert not paths["output_dir"].exists()
