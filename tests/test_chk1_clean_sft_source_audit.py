from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jobs.retrain_v2.audit_chk1_clean_sft import (
    CleanSftAuditError,
    _load_changed_rows,
    _reconstruct_judge_candidate,
)


def _write_fixture(root: Path, *, changed: bool = True) -> tuple[Path, Path]:
    dataset = root / "analysis_sft"
    dataset.mkdir(parents=True)
    records = []
    for split in ("train", "eval", "test"):
        response = f"reasoning-{split}\n</think>\nanswer-{split}"
        row = {
            "prompt": f"prompt-{split}",
            "response": response,
            "provided_data": json.dumps({"value": 1}),
        }
        (dataset / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        new_sha = hashlib.sha256(response.encode()).hexdigest()
        records.append(
            {
                "sample_id": f"sample-{split}",
                "split": split,
                "old_response_sha256": hashlib.sha256(
                    (("old-" if changed and split == "train" else "") + response).encode()
                ).hexdigest(),
                "new_response_sha256": new_sha,
            }
        )
    manifest = root / "repair_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    return root, manifest


def test_changed_rows_join_by_split_and_response_hash(tmp_path: Path) -> None:
    release, manifest = _write_fixture(tmp_path / "release")
    rows = _load_changed_rows(
        release_dir=release,
        repair_manifest=manifest,
        expected_changed_rows=1,
    )
    assert [row["sample_id"] for row in rows] == ["sample-train"]


def test_changed_rows_fail_closed_on_unbound_or_wrong_count(tmp_path: Path) -> None:
    release, manifest = _write_fixture(tmp_path / "release")
    with pytest.raises(CleanSftAuditError, match="changed row count mismatch"):
        _load_changed_rows(
            release_dir=release,
            repair_manifest=manifest,
            expected_changed_rows=2,
        )
    payload = json.loads((release / "analysis_sft/train.jsonl").read_text())
    payload["response"] += " drift"
    (release / "analysis_sft/train.jsonl").write_text(json.dumps(payload) + "\n")
    with pytest.raises(CleanSftAuditError, match="no repair manifest hash binding"):
        _load_changed_rows(
            release_dir=release,
            repair_manifest=manifest,
            expected_changed_rows=1,
        )


def test_judge_candidate_reconstructs_both_explicit_sections() -> None:
    candidate = _reconstruct_judge_candidate(
        "reasoning text\n</think>\nplain answer", sample_id="sample-1"
    )
    assert candidate == (
        "<think>\nreasoning text\n</think>\n"
        "<answer>\nplain answer\n</answer>"
    )

    with pytest.raises(CleanSftAuditError, match=r"complete reasoning\+answer"):
        _reconstruct_judge_candidate("reasoning only", sample_id="sample-2")
