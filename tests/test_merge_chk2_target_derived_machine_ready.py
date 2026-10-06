from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jobs.generation.merge_chk2_target_derived_machine_ready import (
    MERGE_SCHEMA_VERSION,
    MergeError,
    merge_releases,
)
from jobs.generation.promote_chk2_target_derived_machine_pass import promote_release


SPLITS = ("train", "validation", "test")
SOURCE_COMPLETE_STATUS = "machine_screen_complete_human_review_pending"
STUDENT_PROMPT_PREFIX = (
    "Rewrite the following analysis as formal FOMC Minutes prose:\n\n"
)
SPLIT_AUTHORITY_SHA256 = "a" * 64


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(_canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _row(
    *,
    label: str,
    split: str,
    meeting_date: str,
    line_id: str,
) -> tuple[dict, dict, dict]:
    official = f"Official minutes paragraph for {label}."
    official_sha256 = _sha256_text(official)
    sample_id = (
        f"official-{meeting_date}-{line_id}-{official_sha256[:8]}"
    )
    analysis = f"Analysis for {label}."
    reasoning = f"Reasoning for {label}."
    prompt = STUDENT_PROMPT_PREFIX + _canonical_json({"analysis": analysis})
    response = f"{reasoning}\n</think>\n{official}"
    prepared = {
        "sample_id": sample_id,
        "split": split,
        "meeting_date": meeting_date,
        "line_id": line_id,
        "official_minutes_paragraph": official,
        "official_minutes_raw": official,
        "official_minutes_sha256": official_sha256,
        "official_minutes_raw_sha256": official_sha256,
        "legacy_sample_ids": [f"legacy-{label}"],
        "official_source_files": ["fixture/minutes.csv"],
        "official_source_match": {
            "line_id": line_id,
            "normalization_repairs": [],
            "path": "fixture/minutes.csv",
            "projected_text_sha256": official_sha256,
            "raw_text_sha256": official_sha256,
            "section_name": "Staff Review",
        },
        "official_normalization_repairs": [],
        "section_category": "core_economic_financial",
        "section_name": "Staff Review",
        "topic": label,
    }
    candidate = {"prompt": prompt, "response": response}
    manifest = {
        **prepared,
        "schema_version": "chk2-target-derived-manifest-v1",
        "split_role": (
            "target_derived_training"
            if split == "train"
            else "target_derived_monitoring_quarantine"
        ),
        "analysis": analysis,
        "reasoning": reasoning,
        "machine_pass": True,
        "human_review_status": "pending",
        "training_ready": False,
        "training_only": True,
        "evaluation_eligible": False,
        "analysis_is_reference_free": False,
        "analysis_source_lineage": "target_derived_from_official_minutes",
        "analysis_teacher_saw_official_target": True,
        "reasoning_teacher_saw_official_target": True,
        "student_prompt_has_direct_target_field": False,
        "target_is_teacher_synthetic_rewrite": False,
        "reported_checkpoint_eligible": False,
        "suitable_for_leakage_safe_evaluation": False,
        "teacher": {
            "generation_response_id": f"generation-{label}",
            "model": "deepseek-v4-flash",
            "provider": "deepseek",
            "seed": None,
            "system_fingerprint": "fixture-fingerprint",
            "temperature": None,
            "verification_response_id": f"verification-{label}",
        },
        "prompt_sha256": _sha256_text(prompt),
        "reasoning_sha256": _sha256_text(reasoning),
        "response_sha256": _sha256_text(response),
    }
    manifest.pop("official_normalization_repairs")
    manifest["official_minutes_normalization_repairs"] = []
    return prepared, candidate, manifest


def _write_source_release(
    root: Path,
    *,
    prepared_rows: list[dict],
    passing: list[tuple[dict, dict]],
    rejected_ids: set[str],
    usage_multiplier: int,
) -> None:
    prepared_by_split = {
        split: [row for row in prepared_rows if row["split"] == split]
        for split in SPLITS
    }
    passing_by_split = {
        split: [
            (candidate, manifest)
            for candidate, manifest in passing
            if manifest["split"] == split
        ]
        for split in SPLITS
    }
    split_counts = {
        split: {
            "prepared": len(prepared_by_split[split]),
            "generation_responses": len(prepared_by_split[split]),
            "generation_gate_pass": len(passing_by_split[split]),
            "verification_responses": len(passing_by_split[split]),
            "machine_pass": len(passing_by_split[split]),
            "machine_rejected_or_pending": (
                len(prepared_by_split[split]) - len(passing_by_split[split])
            ),
        }
        for split in SPLITS
    }
    total_prepared = len(prepared_rows)
    total_machine_pass = len(passing)
    _write_json(
        root / "summary.json",
        {
            "schema_version": "chk2-target-derived-summary-v1",
            "status": SOURCE_COMPLETE_STATUS,
            "quality_status": SOURCE_COMPLETE_STATUS,
            "phase": "verify",
            "failure_count": 0,
            "teacher_model": "deepseek-v4-flash",
            "provider_identity": {
                "returned_model": "deepseek-v4-flash",
                "system_fingerprint": "fixture-fingerprint",
            },
            "training_ready": False,
            "human_review_status": "pending",
            "total_prepared": total_prepared,
            "total_machine_pass": total_machine_pass,
            "total_machine_rejected_or_pending": (
                total_prepared - total_machine_pass
            ),
            "split_counts": split_counts,
            "preparation": {
                "meeting_split_manifest": {
                    "path": "fixture/meeting_split_manifest.json",
                    "sha256": SPLIT_AUTHORITY_SHA256,
                }
            },
            "usage": {
                "actual_request_attempts": 2 * usage_multiplier,
                "failed_request_attempts": 0,
                "successful_requests": 2 * usage_multiplier,
                "token_usage": {
                    "prompt_tokens": 10 * usage_multiplier,
                    "completion_tokens": 5 * usage_multiplier,
                    "total_tokens": 15 * usage_multiplier,
                },
            },
            "lineage": {
                "analysis_source_lineage": "target_derived_from_official_minutes",
                "training_only": True,
                "evaluation_eligible": False,
                "training_ready": False,
            },
        },
    )
    _write_json(
        root / "prompt_contract.json",
        {
            "schema_version": "fixture-prompt-contract-v1",
            "student_prompt_prefix": STUDENT_PROMPT_PREFIX,
        },
    )
    for split in SPLITS:
        _write_jsonl(root / "prepared" / f"{split}.jsonl", prepared_by_split[split])
        _write_jsonl(
            root / "machine_rejections" / f"{split}.jsonl",
            [
                {
                    "sample_id": row["sample_id"],
                    "split": split,
                    "rejection_stage": "generation_gate",
                    "rejection_reasons": ["fixture_rejection"],
                }
                for row in prepared_by_split[split]
                if row["sample_id"] in rejected_ids
            ],
        )
        _write_jsonl(
            root / "sft_candidate" / f"{split}.jsonl",
            [candidate for candidate, _ in passing_by_split[split]],
        )
        _write_jsonl(
            root / "manifests" / f"{split}.jsonl",
            [manifest for _, manifest in passing_by_split[split]],
        )


def _build_ready_parents(
    tmp_path: Path,
    *,
    authorized_addition: bool = True,
) -> tuple[Path, Path, Path, dict[str, dict]]:
    base_train, base_train_candidate, base_train_manifest = _row(
        label="base-train",
        split="train",
        meeting_date="2020-01-29",
        line_id="002",
    )
    base_validation, base_validation_candidate, base_validation_manifest = _row(
        label="base-validation",
        split="validation",
        meeting_date="2020-03-15",
        line_id="010",
    )
    authorized, authorized_candidate, authorized_manifest = _row(
        label="authorized-recovery",
        split="train",
        meeting_date="2020-01-29",
        line_id="001",
    )
    rogue, rogue_candidate, rogue_manifest = _row(
        label="rogue-recovery",
        split="test",
        meeting_date="2020-04-29",
        line_id="999",
    )
    addition = (
        (authorized, authorized_candidate, authorized_manifest)
        if authorized_addition
        else (rogue, rogue_candidate, rogue_manifest)
    )

    base_source = tmp_path / "base-source"
    addition_source = tmp_path / "addition-source"
    base_ready = tmp_path / "base-ready"
    addition_ready = tmp_path / "addition-ready"
    output = tmp_path / "merged-ready"
    canonical_prepared = [base_train, base_validation, authorized]
    _write_source_release(
        base_source,
        prepared_rows=canonical_prepared,
        passing=[
            (base_train_candidate, base_train_manifest),
            (base_validation_candidate, base_validation_manifest),
        ],
        rejected_ids={authorized["sample_id"]},
        usage_multiplier=2,
    )
    _write_source_release(
        addition_source,
        prepared_rows=[addition[0]],
        passing=[(addition[1], addition[2])],
        rejected_ids=set(),
        usage_multiplier=1,
    )
    promote_release(source_root=base_source, output_root=base_ready)
    promote_release(source_root=addition_source, output_root=addition_ready)
    return base_ready, addition_ready, output, {
        "base_train": base_train,
        "base_validation": base_validation,
        "authorized": authorized,
        "addition": addition[0],
    }


def test_merges_ready_parents_and_preserves_bound_pairs(tmp_path: Path) -> None:
    base, addition, output, rows = _build_ready_parents(tmp_path)
    base_before = _tree_digest(base)
    addition_before = _tree_digest(addition)

    summary = merge_releases(
        base_root=base,
        addition_root=addition,
        output_root=output,
    )

    assert summary["schema_version"] == MERGE_SCHEMA_VERSION
    assert summary["total_prepared"] == 3
    assert summary["total_machine_pass"] == 3
    assert summary["total_training_rows"] == 3
    assert summary["recovered_rows"] == 1
    assert summary["remaining_original_rejections"] == 0
    assert summary["split_counts"]["train"]["training_rows"] == 2
    assert summary["split_counts"]["validation"]["training_rows"] == 1
    assert summary["split_counts"]["test"]["training_rows"] == 0

    train_manifests = _read_jsonl(output / "manifests/train.jsonl")
    train_candidates = _read_jsonl(output / "sft_candidate/train.jsonl")
    assert [row["sample_id"] for row in train_manifests] == sorted(
        [rows["base_train"]["sample_id"], rows["authorized"]["sample_id"]]
    )
    for candidate, manifest in zip(train_candidates, train_manifests):
        assert _sha256_text(candidate["prompt"]) == manifest["prompt_sha256"]
        assert _sha256_text(candidate["response"]) == manifest["response_sha256"]
        assert manifest["training_ready"] is True
        assert "human_review_status" not in manifest

    handoff = json.loads((output / "handoff.json").read_text(encoding="utf-8"))
    assert handoff["summary"]["sha256"] == _sha256_file(output / "summary.json")
    assert handoff["training_ready"] is True
    assert handoff["evaluation_eligible"] is False
    assert _tree_digest(base) == base_before
    assert _tree_digest(addition) == addition_before


def test_rejects_parent_artifact_sha_tampering(tmp_path: Path) -> None:
    base, addition, output, _ = _build_ready_parents(tmp_path)
    candidate_path = addition / "sft_candidate/train.jsonl"
    candidate_path.write_text(
        candidate_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(MergeError, match=r"addition/train\.sft_candidate SHA256 mismatch"):
        merge_releases(
            base_root=base,
            addition_root=addition,
            output_root=output,
        )
    assert not output.exists()


def test_rejects_addition_outside_original_rejection_ledger(
    tmp_path: Path,
) -> None:
    base, addition, output, rows = _build_ready_parents(
        tmp_path,
        authorized_addition=False,
    )

    with pytest.raises(
        MergeError,
        match=r"addition contains rows outside canonical rejection ledger",
    ):
        merge_releases(
            base_root=base,
            addition_root=addition,
            output_root=output,
        )
    assert rows["addition"]["sample_id"] != rows["authorized"]["sample_id"]
    assert not output.exists()
