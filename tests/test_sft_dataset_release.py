from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from open_r1.trainer.dataset_release import (
    CLEAN_SFT_AUDIT_SCHEMA,
    CLEAN_SFT_RELEASE_SCHEMA,
    CLEAN_SFT_VALIDATION_SCHEMA,
    DatasetReleaseValidationError,
    sha256_file,
    verify_clean_sft_release,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_release(root: Path) -> tuple[Path, Path]:
    dataset = root / "analysis_sft"
    dataset.mkdir(parents=True)
    split_counts = {"train": 2, "eval": 1, "test": 1}
    split_files = {}
    for split, rows in split_counts.items():
        path = dataset / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps({"split": split, "row": index}) + "\n" for index in range(rows)),
            encoding="utf-8",
        )
        split_files[split] = {
            "path": f"analysis_sft/{split}.jsonl",
            "rows": rows,
            "sha256": sha256_file(path),
        }
    coordinates = [("train", 1), ("train", 2), ("eval", 1), ("test", 1)]
    repair_rows = []
    for index, (split, line_number) in enumerate(coordinates):
        response_sha = _sha(f"response-{index}")
        repair_rows.append(
            {
                "sample_id": f"sample-{index}",
                "split": split,
                "source_line_number": line_number,
                "prompt_sha256": _sha(f"prompt-{index}"),
                "provided_data_sha256": _sha(f"evidence-{index}"),
                "old_response_sha256": (
                    _sha(f"old-response-{index}") if index < 2 else response_sha
                ),
                "new_response_sha256": response_sha,
            }
        )
    repair = root / "repair_manifest.jsonl"
    repair.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in repair_rows),
        encoding="utf-8",
    )
    audits = root / "audits"
    audits.mkdir()
    audit_rows = audits / "semantic_row_audits.jsonl"
    audit_rows.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": CLEAN_SFT_AUDIT_SCHEMA,
                    "sample_id": repair_rows[index]["sample_id"],
                    "split": repair_rows[index]["split"],
                    "candidate_sha256": repair_rows[index]["new_response_sha256"],
                    "evidence_sha256": repair_rows[index]["provided_data_sha256"],
                    "judge_attempts": 1,
                    "judge_raw_sha256": _sha(f"judge-{index}"),
                    "validated_violations": [],
                    "blocking_violations": [],
                    "rubric": {
                        "data_fidelity": 4,
                        "trend_reasoning": 4,
                        "policy_relevance": 4,
                        "uncertainty_calibration": 4,
                        "fomc_style": 4,
                    },
                    "status": "passed",
                },
                sort_keys=True,
            )
            + "\n"
            for index in range(2)
        ),
        encoding="utf-8",
    )
    audit_summary = audits / "semantic_audit_summary.json"
    audit_summary.write_text(
        json.dumps(
            {
                "schema_version": CLEAN_SFT_AUDIT_SCHEMA,
                "status": "passed",
                "repair_manifest_sha256": sha256_file(repair),
                "counts": {
                    "expected": 2,
                    "completed": 2,
                    "passed": 2,
                    "failed": 0,
                    "judge_errors": 0,
                    "blocking_violations": 0,
                },
                "row_audit": {
                    "path": "row_audits.jsonl",
                    "rows": 2,
                    "sha256": sha256_file(audit_rows),
                },
                "errors": [],
                "judge": {
                    "model": "Qwen3.5-9B",
                    "health": {
                        "model": "Qwen3.5-9B",
                        "loaded_model_root": "/models/Qwen3.5-9B",
                        "status": "ready",
                        "tokenizer_parity": True,
                        "weight_attested": True,
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    content_binding = [
        {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "line_number": row["source_line_number"],
            "prompt_sha256": row["prompt_sha256"],
            "provided_data_sha256": row["provided_data_sha256"],
            "response_sha256": row["new_response_sha256"],
        }
        for row in repair_rows
    ]
    candidate_sha = _sha("candidate-manifest")
    tokenizer_files = [
        {"name": "tokenizer.json", "bytes": 3, "sha256": _sha("{}\n")}
    ]
    tokenizer_bundle_sha = _sha(_canonical(tokenizer_files))
    validation = {
        "schema_version": CLEAN_SFT_VALIDATION_SCHEMA,
        "status": "passed",
        "contracts": {
            "data": "chk1-clean-sft-data-contract-v2",
            "tokenization": "chk1-sft-single-bos-completion-mask-v1",
        },
        "configuration": {
            "expected_split_counts": split_counts,
            "max_length": 4608,
            "reasoning_token_range": [512, 2400],
            "answer_token_range": [16, 512],
        },
        "inputs": {
            "clean_release": {
                "release_id": "candidate",
                "manifests": [
                    {
                        "name": "candidate_manifest.json",
                        "sha256": candidate_sha,
                    }
                ],
                "splits": {
                    split: {"sha256": split_files[split]["sha256"]}
                    for split in split_counts
                }
            },
            "repair_manifest": {"sha256": sha256_file(repair), "rows": 4},
            "tokenizer": {
                "files": tokenizer_files,
                "tokenizer_bundle_sha256": tokenizer_bundle_sha,
            },
        },
        "counts": {
            "expected_rows": 4,
            "observed_rows": 4,
            "valid_rows": 4,
            "prompt_hashes_unchanged": 4,
            "provided_data_hashes_unchanged": 4,
            "order_unchanged": 4,
            "truncated_rows": 0,
            "issue_rows_or_groups": 0,
            "split_counts": split_counts,
        },
        "content_binding_sha256": _sha(_canonical(content_binding)),
        "quality_gates": {"all": True},
        "issues": [],
        "token_statistics": {"max_total_tokens": 1024},
    }
    validation["validation_sha256"] = _sha(_canonical(validation))
    validation_path = audits / "deterministic_validation.json"
    validation_path.write_text(
        json.dumps(validation, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = root / "release_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": CLEAN_SFT_RELEASE_SCHEMA,
                "release_id": "fixture",
                "quality_status": "passed",
                "immutable": True,
                "split_counts": split_counts,
                "split_files": split_files,
                "repair_manifest": {
                    "path": "repair_manifest.jsonl",
                    "rows": 4,
                    "sha256": sha256_file(repair),
                },
                "changed_rows": 2,
                "changed_sample_ids_sha256": _sha(
                    _canonical(["sample-0", "sample-1"])
                ),
                "candidate_manifest": {
                    "path": "/tmp/candidate/candidate_manifest.json",
                    "sha256": candidate_sha,
                },
                "semantic_audit": {
                    "schema_version": CLEAN_SFT_AUDIT_SCHEMA,
                    "summary": {
                        "path": "audits/semantic_audit_summary.json",
                        "sha256": sha256_file(audit_summary),
                    },
                    "rows": {
                        "path": "audits/semantic_row_audits.jsonl",
                        "rows": 2,
                        "sha256": sha256_file(audit_rows),
                    },
                },
                "deterministic_validation": {
                    "schema_version": CLEAN_SFT_VALIDATION_SCHEMA,
                    "report": {
                        "path": "audits/deterministic_validation.json",
                        "sha256": sha256_file(validation_path),
                    },
                    "validation_sha256": validation["validation_sha256"],
                    "max_length": 4608,
                    "tokenizer_bundle_sha256": validation["inputs"]["tokenizer"][
                        "tokenizer_bundle_sha256"
                    ],
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return dataset, manifest


def test_clean_release_manifest_binds_every_split_and_repair_row(tmp_path: Path) -> None:
    dataset, manifest = _write_release(tmp_path / "release")
    result = verify_clean_sft_release(
        dataset_dir=dataset.resolve(),
        manifest_path=manifest.resolve(),
        expected_manifest_sha256=sha256_file(manifest),
    )
    assert result["quality_status"] == "passed"


def test_clean_release_rejects_manifest_or_split_drift(tmp_path: Path) -> None:
    dataset, manifest = _write_release(tmp_path / "release")
    original_sha = sha256_file(manifest)
    with pytest.raises(DatasetReleaseValidationError, match="SHA-256 disagrees"):
        verify_clean_sft_release(
            dataset_dir=dataset.resolve(),
            manifest_path=manifest.resolve(),
            expected_manifest_sha256=hashlib.sha256(b"wrong").hexdigest(),
        )

    (dataset / "train.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(DatasetReleaseValidationError, match="train split SHA-256"):
        verify_clean_sft_release(
            dataset_dir=dataset.resolve(),
            manifest_path=manifest.resolve(),
            expected_manifest_sha256=original_sha,
        )


def test_clean_release_rejects_non_passed_or_incomplete_repair_manifest(
    tmp_path: Path,
) -> None:
    dataset, manifest = _write_release(tmp_path / "release")
    payload = json.loads(manifest.read_text())
    payload["quality_status"] = "candidate"
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(DatasetReleaseValidationError, match="quality_status=passed"):
        verify_clean_sft_release(
            dataset_dir=dataset.resolve(),
            manifest_path=manifest.resolve(),
            expected_manifest_sha256=sha256_file(manifest),
        )

    payload["quality_status"] = "passed"
    payload["repair_manifest"]["rows"] = 3
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(DatasetReleaseValidationError, match="row count is incomplete"):
        verify_clean_sft_release(
            dataset_dir=dataset.resolve(),
            manifest_path=manifest.resolve(),
            expected_manifest_sha256=sha256_file(manifest),
        )


def test_clean_release_rejects_semantic_audit_drift(tmp_path: Path) -> None:
    dataset, manifest = _write_release(tmp_path / "release")
    audit_rows = manifest.parent / "audits" / "semantic_row_audits.jsonl"
    audit_rows.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(DatasetReleaseValidationError, match="audit rows SHA-256"):
        verify_clean_sft_release(
            dataset_dir=dataset.resolve(),
            manifest_path=manifest.resolve(),
            expected_manifest_sha256=sha256_file(manifest),
        )


def test_clean_release_rejects_semantic_row_status_or_candidate_drift(
    tmp_path: Path,
) -> None:
    dataset, manifest = _write_release(tmp_path / "release")
    rows_path = manifest.parent / "audits" / "semantic_row_audits.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    rows[0]["status"] = "failed"
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    payload = json.loads(manifest.read_text())
    rows_sha = sha256_file(rows_path)
    payload["semantic_audit"]["rows"]["sha256"] = rows_sha
    summary_path = manifest.parent / "audits" / "semantic_audit_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["row_audit"]["sha256"] = rows_sha
    summary_path.write_text(json.dumps(summary) + "\n")
    payload["semantic_audit"]["summary"]["sha256"] = sha256_file(summary_path)
    manifest.write_text(json.dumps(payload) + "\n")

    with pytest.raises(DatasetReleaseValidationError, match="identity/status"):
        verify_clean_sft_release(
            dataset_dir=dataset.resolve(),
            manifest_path=manifest.resolve(),
            expected_manifest_sha256=sha256_file(manifest),
        )


def test_clean_release_rejects_deterministic_validation_drift(tmp_path: Path) -> None:
    dataset, manifest = _write_release(tmp_path / "release")
    validation_path = manifest.parent / "audits" / "deterministic_validation.json"
    validation = json.loads(validation_path.read_text())
    validation["counts"]["truncated_rows"] = 1
    validation_payload = dict(validation)
    validation_payload.pop("validation_sha256")
    validation["validation_sha256"] = _sha(_canonical(validation_payload))
    validation_path.write_text(json.dumps(validation) + "\n")
    payload = json.loads(manifest.read_text())
    payload["deterministic_validation"]["report"]["sha256"] = sha256_file(
        validation_path
    )
    payload["deterministic_validation"]["validation_sha256"] = validation[
        "validation_sha256"
    ]
    manifest.write_text(json.dumps(payload) + "\n")

    with pytest.raises(DatasetReleaseValidationError, match="truncated_rows"):
        verify_clean_sft_release(
            dataset_dir=dataset.resolve(),
            manifest_path=manifest.resolve(),
            expected_manifest_sha256=sha256_file(manifest),
        )


def test_clean_release_rejects_duplicate_semantic_sample_binding(
    tmp_path: Path,
) -> None:
    dataset, manifest = _write_release(tmp_path / "release")
    rows_path = manifest.parent / "audits" / "semantic_row_audits.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    rows[1]["sample_id"] = rows[0]["sample_id"]
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    rows_sha = sha256_file(rows_path)
    summary_path = manifest.parent / "audits" / "semantic_audit_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["row_audit"]["sha256"] = rows_sha
    summary_path.write_text(json.dumps(summary) + "\n")
    payload = json.loads(manifest.read_text())
    payload["semantic_audit"]["rows"]["sha256"] = rows_sha
    payload["semantic_audit"]["summary"]["sha256"] = sha256_file(summary_path)
    manifest.write_text(json.dumps(payload) + "\n")

    with pytest.raises(DatasetReleaseValidationError, match="identity/status"):
        verify_clean_sft_release(
            dataset_dir=dataset.resolve(),
            manifest_path=manifest.resolve(),
            expected_manifest_sha256=sha256_file(manifest),
        )
