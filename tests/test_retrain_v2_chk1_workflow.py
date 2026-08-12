from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobs.retrain_v2.chk1 import workflow


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _generation_fixture(
    root: Path, *, selected: int = 20, accepted: int = 14
) -> Path:
    ids = [f"sample-{index:03d}" for index in range(selected)]
    topics = {"topic-a": selected // 2, "topic-b": selected - selected // 2}
    _json(
        root / "selection_manifest.json",
        {
            "selected_count": selected,
            "selected_sample_ids": ids,
            "selected_by_topic": topics,
        },
    )
    first_topic = selected // 2
    accepted_first = accepted // 2
    accepted_second = accepted - accepted_first
    accepted_indices = [
        *range(accepted_first),
        *range(first_topic, first_topic + accepted_second),
    ]
    accepted_rows = [
        {
            "sample_id": ids[index],
            "atomic_topic": "topic-a" if index < selected // 2 else "topic-b",
        }
        for index in accepted_indices
    ]
    for split in ("train", "eval", "test"):
        _jsonl(
            root / "manifests" / f"{split}.jsonl",
            accepted_rows if split == "train" else [],
        )
    accepted_ids = {ids[index] for index in accepted_indices}
    exclusions = [
        {
            "sample_id": sample_id,
            "atomic_topic": "topic-a" if index < selected // 2 else "topic-b",
        }
        for index, sample_id in enumerate(ids)
        if sample_id not in accepted_ids
    ]
    _jsonl(root / "audit/exclusions.jsonl", exclusions)
    _json(
        root / "generation_handoff.json",
        {
            "status": "complete",
            "mode": "smoke",
            "selected_count": selected,
            "accepted_count": accepted,
            "excluded_count": selected - accepted,
            "generation_provenance": {"payload_sha256": "a" * 64},
            "payload_sha256": "b" * 64,
        },
    )
    return root


def test_completed_generation_reports_retrieval_without_quality_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setitem(workflow.EXPECTED_COUNTS, "smoke", 20)
    result = workflow.assess_completed_generation(
        _generation_fixture(tmp_path), mode="smoke"
    )
    assert result["status"] == "complete"
    assert result["retrieval_rate"] == 0.70


def test_completed_generation_does_not_gate_total_or_topic_rates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setitem(workflow.EXPECTED_COUNTS, "smoke", 20)
    low_total = workflow.assess_completed_generation(
        _generation_fixture(tmp_path / "total", accepted=13), mode="smoke"
    )
    assert low_total["status"] == "complete"
    assert low_total["retrieval_rate"] == 0.65

    fixture = _generation_fixture(tmp_path / "topic", accepted=14)
    rows = [
        {"sample_id": f"sample-{index:03d}", "atomic_topic": "topic-a"}
        for index in range(10)
    ] + [
        {"sample_id": f"sample-{index:03d}", "atomic_topic": "topic-b"}
        for index in range(10, 14)
    ]
    _jsonl(fixture / "manifests/train.jsonl", rows)
    _jsonl(
        fixture / "audit/exclusions.jsonl",
        [
            {"sample_id": f"sample-{index:03d}", "atomic_topic": "topic-b"}
            for index in range(14, 20)
        ],
    )
    topic = workflow.assess_completed_generation(fixture, mode="smoke")
    assert topic["status"] == "complete"
    assert topic["retrieval_rate"] == 0.70


def test_completed_generation_rejects_missing_or_duplicate_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setitem(workflow.EXPECTED_COUNTS, "smoke", 20)
    fixture = _generation_fixture(tmp_path)
    selection = json.loads((fixture / "selection_manifest.json").read_text())
    selection["selected_sample_ids"][-1] = selection["selected_sample_ids"][0]
    _json(fixture / "selection_manifest.json", selection)
    with pytest.raises(workflow.Chk1WorkflowError, match="selection IDs"):
        workflow.assess_completed_generation(fixture, mode="smoke")


def test_prior_phase_allows_complete_smoke_regardless_of_retrieval_rate(tmp_path: Path):
    paths = {
        "outputs": {"smoke": _generation_fixture(tmp_path, accepted=13)}
    }
    workflow._validate_prior_phase(paths, "pilot")
