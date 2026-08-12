from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from jobs.retrain_v2.compress_chk1_reasoning import (
    BOUNDARY,
    SPLITS,
    _canonical_json,
    _sha256_file,
    _sha256_text,
)
from jobs.retrain_v2.publish_chk1_compressed_sft import (
    AUDIT_SCHEMA_VERSION,
    PublishError,
    main,
)


def _row(split: str) -> dict[str, str]:
    return {
        "prompt": f"Analyze the point-in-time evidence for {split}.",
        "response": (
            f"The {split} observations support a cautious interpretation."
            + BOUNDARY
            + "Economic activity moderated while labor conditions remained stable."
        ),
        "provided_data": f"point-in-time observations for {split}",
    }


def _write_source(root: Path) -> dict[str, list[dict[str, str]]]:
    analysis_sft = root / "analysis_sft"
    analysis_sft.mkdir(parents=True)
    rows: dict[str, list[dict[str, str]]] = {}
    for split in SPLITS:
        rows[split] = [_row(split)]
        (analysis_sft / f"{split}.jsonl").write_text(
            _canonical_json(rows[split][0]) + "\n", encoding="utf-8"
        )
    (root / "run_manifest.json").write_text(
        json.dumps({"schema_version": "test-compression-run"}) + "\n",
        encoding="utf-8",
    )
    return rows


def _write_audit(
    path: Path,
    source: Path,
    rows: dict[str, list[dict[str, str]]],
) -> dict[str, object]:
    samples: list[dict[str, object]] = []
    for split in SPLITS:
        for row in rows[split]:
            reasoning, answer = row["response"].split(BOUNDARY)
            samples.append(
                {
                    "split": split,
                    "row_sha256": _sha256_text(_canonical_json(row)),
                    "reasoning_sha256": _sha256_text(reasoning.strip()),
                    "answer_sha256": _sha256_text(answer.strip()),
                    "provider_error": False,
                    "section_status": {"reasoning": "pass", "answer": "pass"},
                    "status": "pass",
                }
            )
    payload: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "passed",
        "audited_sections": ["reasoning", "answer"],
        "counts": {
            "total": len(samples),
            "completed": len(samples),
            "provider_failed": 0,
            "passed": len(samples),
            "failed": 0,
        },
        "compressed_files": {
            split: {
                "sha256": _sha256_file(source / "analysis_sft" / f"{split}.jsonl")
            }
            for split in SPLITS
        },
        "audited_samples": samples,
        "failed_samples": [],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, list[dict[str, str]]]]:
    source = tmp_path / "compression"
    rows = _write_source(source)
    audit_path = tmp_path / "audit.json"
    _write_audit(audit_path, source, rows)
    return source, audit_path, tmp_path / "release", rows


def _run(source: Path, audit: Path, release: Path) -> int:
    return main(
        [
            "--compression-output",
            str(source),
            "--audit-summary",
            str(audit),
            "--release-dir",
            str(release),
        ]
    )


def test_publishes_only_full_hash_bound_audit(tmp_path: Path) -> None:
    source, audit, release, _ = _fixture(tmp_path)

    assert _run(source, audit, release) == 0

    manifest = json.loads((release / "release_manifest.json").read_text())
    assert manifest["immutable"] is True
    assert manifest["quality_status"] == "passed"
    assert manifest["semantic_audit"]["audited_sections"] == ["answer", "reasoning"]
    assert manifest["semantic_audit"]["binding"] == (
        "canonical_row_and_reasoning_and_answer_sha256"
    )
    assert manifest["split_counts"] == {split: 1 for split in SPLITS}
    assert stat.S_IMODE(release.stat().st_mode) == 0o555
    assert stat.S_IMODE((release / "release_manifest.json").stat().st_mode) == 0o444
    assert stat.S_IMODE((release / "analysis_sft").stat().st_mode) == 0o555
    assert stat.S_IMODE(
        (release / "analysis_sft" / "train.jsonl").stat().st_mode
    ) == 0o444


def test_rejects_lone_open_brace_answer(tmp_path: Path) -> None:
    source, audit, release, rows = _fixture(tmp_path)
    rows["train"][0]["response"] = "Grounded reasoning." + BOUNDARY + "{"
    (source / "analysis_sft/train.jsonl").write_text(
        _canonical_json(rows["train"][0]) + "\n", encoding="utf-8"
    )

    with pytest.raises(PublishError, match="answer is too short"):
        _run(source, audit, release)
    assert not release.exists()


def test_rejects_inline_evidence_id_in_answer(tmp_path: Path) -> None:
    source, audit, release, rows = _fixture(tmp_path)
    rows["eval"][0]["response"] = (
        "Grounded reasoning."
        + BOUNDARY
        + "Economic activity moderated according to ev-output-17."
    )
    (source / "analysis_sft/eval.jsonl").write_text(
        _canonical_json(rows["eval"][0]) + "\n", encoding="utf-8"
    )

    with pytest.raises(PublishError, match="inline evidence ID"):
        _run(source, audit, release)
    assert not release.exists()


@pytest.mark.parametrize("citation", ["(E1)", "[fact7]", "evidence ID A12"])
def test_rejects_generic_inline_evidence_citation(
    tmp_path: Path, citation: str
) -> None:
    source, audit, release, rows = _fixture(tmp_path)
    rows["eval"][0]["response"] = (
        "Grounded reasoning."
        + BOUNDARY
        + f"Economic activity moderated according to {citation}."
    )
    (source / "analysis_sft/eval.jsonl").write_text(
        _canonical_json(rows["eval"][0]) + "\n", encoding="utf-8"
    )

    with pytest.raises(PublishError, match="inline evidence ID"):
        _run(source, audit, release)
    assert not release.exists()


def test_rejects_semantic_failures_without_excluding_rows(tmp_path: Path) -> None:
    source, audit_path, release, _ = _fixture(tmp_path)
    audit = json.loads(audit_path.read_text())
    failed = audit["audited_samples"][0]
    failed["section_status"]["answer"] = "fail"
    failed["status"] = "fail"
    audit["status"] = "semantic_failures"
    audit["counts"]["passed"] -= 1
    audit["counts"]["failed"] = 1
    audit["failed_samples"] = [
        {"split": failed["split"], "row_sha256": failed["row_sha256"]}
    ]
    audit_path.write_text(json.dumps(audit) + "\n", encoding="utf-8")

    with pytest.raises(PublishError, match="semantic audit did not pass"):
        _run(source, audit_path, release)
    assert not release.exists()


def test_rejects_cross_split_duplicate_canonical_row(tmp_path: Path) -> None:
    source, audit, release, rows = _fixture(tmp_path)
    rows["eval"][0] = dict(rows["train"][0])
    (source / "analysis_sft/eval.jsonl").write_text(
        _canonical_json(rows["eval"][0]) + "\n", encoding="utf-8"
    )

    with pytest.raises(PublishError, match="cross-split duplicate canonical row"):
        _run(source, audit, release)
    assert not release.exists()


def test_rejects_reasoning_only_audit(tmp_path: Path) -> None:
    source, audit_path, release, _ = _fixture(tmp_path)
    audit = json.loads(audit_path.read_text())
    audit["audited_sections"] = ["reasoning"]
    for sample in audit["audited_samples"]:
        sample["section_status"] = {"reasoning": "pass"}
    audit_path.write_text(json.dumps(audit) + "\n", encoding="utf-8")

    with pytest.raises(PublishError, match="explicitly cover reasoning and answer"):
        _run(source, audit_path, release)
    assert not release.exists()


def test_rejects_answer_hash_drift(tmp_path: Path) -> None:
    source, audit_path, release, _ = _fixture(tmp_path)
    audit = json.loads(audit_path.read_text())
    audit["audited_samples"][0]["answer_sha256"] = "0" * 64
    audit_path.write_text(json.dumps(audit) + "\n", encoding="utf-8")

    with pytest.raises(PublishError, match="answer hash drift"):
        _run(source, audit_path, release)
    assert not release.exists()


def test_existing_release_is_never_overwritten(tmp_path: Path) -> None:
    source, audit, release, _ = _fixture(tmp_path)
    release.mkdir()
    marker = release / "owner-data.txt"
    marker.write_text("preserve me", encoding="utf-8")

    with pytest.raises(PublishError, match="release already exists"):
        _run(source, audit, release)

    assert marker.read_text(encoding="utf-8") == "preserve me"
