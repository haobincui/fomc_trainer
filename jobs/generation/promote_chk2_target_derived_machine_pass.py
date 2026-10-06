"""Promote a completed CHK2 machine screen to a compact training release.

The provider caches and audit artifacts remain immutable in the source release.
Only machine-PASS SFT rows and their manifests are copied to the promoted release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = REPO_ROOT / (
    "output/data/retrain_v2/chk2/"
    "target_derived_minutes_deepseek_v4_flash_v1_20260828"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    "output/data/retrain_v2/chk2/"
    "target_derived_minutes_deepseek_v4_flash_machine_ready_v1_20260829"
)
SPLITS = ("train", "validation", "test")
SOURCE_COMPLETE_STATUS = "machine_screen_complete_human_review_pending"
PROMOTED_STATUS = "machine_screen_complete_training_ready"
PROMOTION_SCHEMA_VERSION = "chk2-target-derived-machine-ready-v1"


class PromotionError(RuntimeError):
    """The source release is not safe to promote."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionError(f"cannot read JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PromotionError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PromotionError(f"cannot read JSONL: {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise PromotionError(f"blank JSONL row: {path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PromotionError(f"invalid JSONL row: {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise PromotionError(f"non-object JSONL row: {path}:{line_number}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _promoted_lineage(value: object) -> dict[str, Any]:
    lineage = dict(value) if isinstance(value, dict) else {}
    lineage["training_ready"] = True
    return lineage


def promote_release(*, source_root: Path, output_root: Path) -> dict[str, Any]:
    source = source_root.resolve()
    output = output_root.resolve()
    if source == output:
        raise PromotionError("source and output roots must differ")
    if output.exists():
        raise PromotionError(f"output already exists: {output}")

    source_summary_path = source / "summary.json"
    source_summary = _read_json(source_summary_path)
    if source_summary.get("status") != SOURCE_COMPLETE_STATUS:
        raise PromotionError(
            "source status must be "
            f"{SOURCE_COMPLETE_STATUS!r}, got {source_summary.get('status')!r}"
        )
    if source_summary.get("phase") != "verify":
        raise PromotionError("source verification phase is not complete")
    if source_summary.get("failure_count") != 0:
        raise PromotionError("source release contains request failures")
    if source_summary.get("training_ready") is not False:
        raise PromotionError("source release has an unexpected readiness state")
    expected_total = source_summary.get("total_machine_pass")
    if not isinstance(expected_total, int) or expected_total <= 0:
        raise PromotionError("source release has no machine-PASS rows")
    split_counts = source_summary.get("split_counts")
    if not isinstance(split_counts, dict):
        raise PromotionError("source split counts are missing")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        promoted_counts: dict[str, int] = {}
        artifact_records: dict[str, dict[str, Any]] = {}
        for split in SPLITS:
            candidate_source = source / "sft_candidate" / f"{split}.jsonl"
            manifest_source = source / "manifests" / f"{split}.jsonl"
            candidate_rows = _read_jsonl(candidate_source)
            manifest_rows = _read_jsonl(manifest_source)
            expected_split = split_counts.get(split)
            expected_count = (
                expected_split.get("machine_pass")
                if isinstance(expected_split, dict)
                else None
            )
            if len(candidate_rows) != len(manifest_rows) or len(candidate_rows) != expected_count:
                raise PromotionError(
                    f"machine-PASS count mismatch for {split}: "
                    f"candidate={len(candidate_rows)} manifest={len(manifest_rows)} "
                    f"summary={expected_count}"
                )

            promoted_manifests: list[dict[str, Any]] = []
            for row in manifest_rows:
                if row.get("split") != split or row.get("machine_pass") is not True:
                    raise PromotionError(
                        f"non-passing or wrong-split manifest row: {row.get('sample_id')}"
                    )
                promoted = dict(row)
                promoted.pop("human_review_status", None)
                promoted["training_ready"] = True
                promoted_manifests.append(promoted)

            candidate_output = staging / "sft_candidate" / f"{split}.jsonl"
            manifest_output = staging / "manifests" / f"{split}.jsonl"
            _write_jsonl(candidate_output, candidate_rows)
            _write_jsonl(manifest_output, promoted_manifests)
            promoted_counts[split] = len(candidate_rows)
            artifact_records[split] = {
                "rows": len(candidate_rows),
                "sft_candidate": {
                    "path": f"sft_candidate/{split}.jsonl",
                    "sha256": _sha256_file(candidate_output),
                },
                "manifest": {
                    "path": f"manifests/{split}.jsonl",
                    "sha256": _sha256_file(manifest_output),
                },
            }

        promoted_total = sum(promoted_counts.values())
        if promoted_total != expected_total:
            raise PromotionError(
                f"promoted total mismatch: rows={promoted_total} summary={expected_total}"
            )

        promoted_summary = dict(source_summary)
        promoted_summary.pop("human_review_status", None)
        promoted_summary.update(
            {
                "schema_version": PROMOTION_SCHEMA_VERSION,
                "status": PROMOTED_STATUS,
                "quality_status": PROMOTED_STATUS,
                "training_ready": True,
                "total_training_rows": promoted_total,
                "lineage": _promoted_lineage(source_summary.get("lineage")),
                "promotion_policy": {
                    "eligibility": "machine_pass",
                    "machine_pass_required": True,
                    "source_failure_count_must_be_zero": True,
                    "evaluation_eligible": False,
                },
                "source_release": {
                    "path": _display_path(source),
                    "summary_sha256": _sha256_file(source_summary_path),
                },
                "artifacts": artifact_records,
            }
        )
        _write_json(staging / "summary.json", promoted_summary)

        handoff = {
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "status": PROMOTED_STATUS,
            "dataset_path": _display_path(output / "sft_candidate"),
            "manifest_path": _display_path(output / "manifests"),
            "teacher_model": source_summary.get("teacher_model"),
            "split_counts": promoted_counts,
            "total_training_rows": promoted_total,
            "training_ready": True,
            "evaluation_eligible": False,
            "lineage": _promoted_lineage(source_summary.get("lineage")),
            "source_release": promoted_summary["source_release"],
            "summary": {
                "path": "summary.json",
                "sha256": _sha256_file(staging / "summary.json"),
            },
        }
        _write_json(staging / "handoff.json", handoff)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return promoted_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote machine-PASS CHK2 rows to a training-ready release."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = promote_release(
        source_root=args.source_root,
        output_root=args.output_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
