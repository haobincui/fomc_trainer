from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobs.generation.promote_chk2_target_derived_machine_pass import (
    PROMOTED_STATUS,
    PromotionError,
    promote_release,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _source_release(root: Path) -> None:
    split_counts = {
        "train": {"machine_pass": 2},
        "validation": {"machine_pass": 1},
        "test": {"machine_pass": 0},
    }
    _write_json(
        root / "summary.json",
        {
            "schema_version": "source-v1",
            "status": "machine_screen_complete_human_review_pending",
            "quality_status": "machine_screen_complete_human_review_pending",
            "phase": "verify",
            "failure_count": 0,
            "teacher_model": "deepseek-v4-flash",
            "training_ready": False,
            "human_review_status": "pending",
            "total_machine_pass": 3,
            "split_counts": split_counts,
            "lineage": {
                "training_only": True,
                "evaluation_eligible": False,
                "training_ready": False,
            },
        },
    )
    for split, count in (("train", 2), ("validation", 1), ("test", 0)):
        candidates = [
            {"prompt": f"prompt-{split}-{index}", "response": "response"}
            for index in range(count)
        ]
        manifests = [
            {
                "sample_id": f"{split}-{index}",
                "split": split,
                "machine_pass": True,
                "human_review_status": "pending",
                "training_ready": False,
                "training_only": True,
                "evaluation_eligible": False,
            }
            for index in range(count)
        ]
        _write_jsonl(root / "sft_candidate" / f"{split}.jsonl", candidates)
        _write_jsonl(root / "manifests" / f"{split}.jsonl", manifests)


def test_promotes_only_machine_pass_rows_without_review_fields(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "promoted"
    _source_release(source)

    summary = promote_release(source_root=source, output_root=output)

    assert summary["status"] == PROMOTED_STATUS
    assert summary["training_ready"] is True
    assert "human_review_status" not in summary
    assert summary["total_training_rows"] == 3
    handoff = json.loads((output / "handoff.json").read_text(encoding="utf-8"))
    assert handoff["training_ready"] is True
    assert handoff["evaluation_eligible"] is False
    rows = [
        json.loads(line)
        for line in (output / "manifests/train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(row["training_ready"] is True for row in rows)
    assert all("human_review_status" not in row for row in rows)
    source_row = json.loads(
        (source / "manifests/train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert source_row["human_review_status"] == "pending"
    assert source_row["training_ready"] is False


def test_refuses_incomplete_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source_release(source)
    summary_path = source / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["failure_count"] = 1
    _write_json(summary_path, summary)

    with pytest.raises(PromotionError, match="request failures"):
        promote_release(source_root=source, output_root=tmp_path / "promoted")
