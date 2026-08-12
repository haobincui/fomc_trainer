"""Publish an immutable chk1 SFT dataset after a full semantic audit.

The historical verifier audited only ``reasoning`` and identified rows by line
number.  That is not a sufficient release authority: an invalid final answer
or a post-audit row reorder could still be published.  This publisher accepts
only the v2 audit contract, which explicitly covers both ``reasoning`` and
``answer`` and binds every decision to canonical content hashes.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.retrain_v2.compress_chk1_reasoning import (
    BOUNDARY,
    SPLITS,
    _canonical_json,
    _sha256_file,
    _sha256_text,
    _utc_now,
)


AUDIT_SCHEMA_VERSION = "chk1-sft-semantic-audit-v2"
RELEASE_SCHEMA_VERSION = "chk1-compressed-sft-release-v2"
AUDITED_SECTIONS = frozenset({"reasoning", "answer"})
MIN_ANSWER_CHARACTERS = 16

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_EVIDENCE_ID_RE = re.compile(r"\bev-[0-9a-z-]+\b", flags=re.IGNORECASE)
_GENERIC_EVIDENCE_CITATION_RE = re.compile(
    r"(?:\(|\[)\s*[a-z]+\d+(?:\s*[,;]\s*[a-z]+\d+)*\s*(?:\)|\])|"
    r"\bevidence(?:\s+id)?\s*[:#]?\s*[a-z]+\d+\b",
    flags=re.IGNORECASE,
)
_CONTROL_TAG_RE = re.compile(
    r"<\s*/?\s*(?:think|answer)\s*>|<\|[^>]+\|>", flags=re.IGNORECASE
)
_JSON_LIKE_RE = re.compile(
    r'"(?:analysis|answer|content|evidence_ids?|final_analysis|reasoning(?:_content)?)"\s*:',
    flags=re.IGNORECASE,
)
_SCHEMA_META_RE = re.compile(
    r"\b(?:json|jsonl|schema(?:\s+(?:field|key|object))?|evidence[_ ]ids?|"
    r"reasoning_content|final_analysis|compressed_reasoning|fixed_final_answer|"
    r"source_prompt|output[_ ]field)\b",
    flags=re.IGNORECASE,
)
_MARKDOWN_RE = re.compile(
    r"(?:```|~~~|(?:^|\n)\s{0,3}(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|>\s)|"
    r"\[[^\]]+\]\([^)]+\)|\*\*[^*]+\*\*|__[^_]+__)",
    flags=re.MULTILINE,
)


class PublishError(RuntimeError):
    """The dataset or semantic audit cannot authorize publication."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublishError(message)


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value or "")
    _require(_SHA256_RE.fullmatch(text) is not None, f"{label} is not SHA-256")
    return text


def _read_json(path: Path, *, label: str) -> tuple[Mapping[str, Any], str]:
    _require(path.is_file() and not path.is_symlink(), f"missing regular {label}: {path}")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublishError(f"invalid {label} {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value, hashlib.sha256(raw).hexdigest()


def _validate_response(response: Any, *, label: str) -> tuple[str, str]:
    _require(isinstance(response, str), f"{label}: response must be text")
    _require(response.count(BOUNDARY) == 1, f"{label}: invalid reasoning boundary")
    reasoning, answer = response.split(BOUNDARY)
    _require(reasoning.strip() != "", f"{label}: empty reasoning")
    _require(answer.strip() != "", f"{label}: empty answer")

    # The one literal boundary is the only model-control token allowed in the
    # raw target.  Removing it before this search catches opening <think>,
    # answer wrappers, duplicate closing tags, and tokenizer control tokens.
    without_boundary = response.replace(BOUNDARY, "", 1)
    _require(
        _CONTROL_TAG_RE.search(without_boundary) is None,
        f"{label}: response contains a forbidden control tag",
    )

    answer = answer.strip()
    compact_answer = re.sub(r"\s+", " ", answer)
    _require(
        len(compact_answer) >= MIN_ANSWER_CHARACTERS,
        f"{label}: answer is too short",
    )
    _require(
        answer[0] not in "[{" and answer[-1] not in "]}",
        f"{label}: answer is JSON-like or structured data",
    )
    _require(
        _JSON_LIKE_RE.search(answer) is None,
        f"{label}: answer contains JSON-like fields",
    )
    _require(
        _EVIDENCE_ID_RE.search(answer) is None
        and _GENERIC_EVIDENCE_CITATION_RE.search(answer) is None,
        f"{label}: answer contains an inline evidence ID",
    )
    _require(
        _SCHEMA_META_RE.search(answer) is None,
        f"{label}: answer contains schema/meta text",
    )
    _require(
        _MARKDOWN_RE.search(answer) is None,
        f"{label}: answer contains Markdown",
    )
    _require(
        all(ord(character) >= 32 or character in "\n\r\t" for character in answer),
        f"{label}: answer contains a control character",
    )
    return reasoning.strip(), answer


def _read_source_rows(source: Path) -> tuple[
    dict[str, list[dict[str, str]]],
    dict[tuple[str, str], tuple[str, str]],
    dict[str, dict[str, str]],
]:
    split_rows: dict[str, list[dict[str, str]]] = {}
    row_parts: dict[tuple[str, str], tuple[str, str]] = {}
    split_files: dict[str, dict[str, str]] = {}
    canonical_rows: dict[str, str] = {}
    for split in SPLITS:
        path = source / "analysis_sft" / f"{split}.jsonl"
        _require(path.is_file() and not path.is_symlink(), f"missing regular split: {path}")
        rows: list[dict[str, str]] = []
        try:
            raw_split = path.read_bytes()
            lines = raw_split.decode("utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise PublishError(f"cannot read split {path}: {exc}") from exc
        for line_number, line in enumerate(lines, 1):
            _require(line != "", f"{split}:{line_number}: blank JSONL row")
            try:
                raw_row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PublishError(f"{split}:{line_number}: invalid JSON: {exc}") from exc
            _require(
                isinstance(raw_row, dict)
                and set(raw_row) == {"prompt", "response", "provided_data"},
                f"{split}:{line_number}: invalid row schema",
            )
            _require(
                all(isinstance(raw_row[key], str) for key in raw_row),
                f"{split}:{line_number}: row fields must be text",
            )
            row = {key: raw_row[key] for key in ("prompt", "response", "provided_data")}
            _require(row["prompt"].strip() != "", f"{split}:{line_number}: empty prompt")
            _require(
                row["provided_data"].strip() != "",
                f"{split}:{line_number}: empty provided_data",
            )
            reasoning, answer = _validate_response(
                row["response"], label=f"{split}:{line_number}"
            )
            row_sha = _sha256_text(_canonical_json(row))
            key = (split, row_sha)
            _require(key not in row_parts, f"{split}:{line_number}: duplicate canonical row")
            previous_split = canonical_rows.get(row_sha)
            _require(
                previous_split is None,
                f"{split}:{line_number}: cross-split duplicate canonical row "
                f"already present in {previous_split}",
            )
            canonical_rows[row_sha] = split
            row_parts[key] = (reasoning, answer)
            rows.append(row)
        split_rows[split] = rows
        split_files[split] = {"sha256": hashlib.sha256(raw_split).hexdigest()}
    return split_rows, row_parts, split_files


def _validate_audit(
    audit: Mapping[str, Any],
    *,
    row_parts: Mapping[tuple[str, str], tuple[str, str]],
    split_files: Mapping[str, Mapping[str, str]],
) -> None:
    _require(
        audit.get("schema_version") == AUDIT_SCHEMA_VERSION,
        "semantic audit must use the full reasoning+answer v2 contract",
    )
    sections = audit.get("audited_sections")
    _require(
        isinstance(sections, list) and set(sections) == AUDITED_SECTIONS and len(sections) == 2,
        "semantic audit must explicitly cover reasoning and answer",
    )
    _require(
        audit.get("status") == "passed",
        "semantic audit did not pass",
    )
    counts = audit.get("counts")
    _require(isinstance(counts, dict), "semantic audit counts are missing")
    expected_total = len(row_parts)
    for key in ("total", "completed", "provider_failed", "passed", "failed"):
        _require(type(counts.get(key)) is int, f"semantic audit count {key} is invalid")
    _require(counts["total"] == expected_total, "semantic audit total disagrees with data")
    _require(counts["completed"] == expected_total, "semantic audit is incomplete")
    _require(counts["provider_failed"] == 0, "semantic audit has provider errors")
    _require(
        counts["passed"] + counts["failed"] == expected_total,
        "semantic audit pass/fail counts are inconsistent",
    )
    _require(
        (audit.get("status") == "passed") == (counts["failed"] == 0),
        "semantic audit status disagrees with failed count",
    )
    _require(counts["failed"] == 0, "semantic audit contains failed samples")

    audited_files = audit.get("compressed_files")
    _require(isinstance(audited_files, dict), "semantic audit lacks split file hashes")
    _require(set(audited_files) == set(SPLITS), "semantic audit split files are incomplete")
    for split in SPLITS:
        record = audited_files.get(split)
        _require(isinstance(record, dict), f"semantic audit {split} file record is invalid")
        actual = split_files[split]["sha256"]
        declared = _require_sha256(record.get("sha256"), label=f"audit {split} file")
        _require(declared == actual, f"semantic audit {split} file hash drift")

    samples = audit.get("audited_samples")
    _require(isinstance(samples, list), "semantic audit lacks content-bound audited_samples")
    _require(len(samples) == expected_total, "semantic audit sample population is incomplete")
    seen: set[tuple[str, str]] = set()
    failed_keys: set[tuple[str, str]] = set()
    passed = 0
    for index, sample in enumerate(samples, 1):
        _require(isinstance(sample, dict), f"audited_samples:{index}: invalid record")
        split = sample.get("split")
        _require(split in SPLITS, f"audited_samples:{index}: invalid split")
        row_sha = _require_sha256(
            sample.get("row_sha256"), label=f"audited_samples:{index}: row"
        )
        key = (str(split), row_sha)
        _require(key in row_parts, f"audited_samples:{index}: row hash drift")
        _require(key not in seen, f"audited_samples:{index}: duplicate audited row")
        seen.add(key)
        reasoning, answer = row_parts[key]
        _require(
            _require_sha256(
                sample.get("reasoning_sha256"),
                label=f"audited_samples:{index}: reasoning",
            )
            == _sha256_text(reasoning),
            f"audited_samples:{index}: reasoning hash drift",
        )
        _require(
            _require_sha256(
                sample.get("answer_sha256"), label=f"audited_samples:{index}: answer"
            )
            == _sha256_text(answer),
            f"audited_samples:{index}: answer hash drift",
        )
        _require(
            sample.get("provider_error") is False,
            f"audited_samples:{index}: provider error or missing provider status",
        )
        section_status = sample.get("section_status")
        _require(
            isinstance(section_status, dict)
            and set(section_status) == AUDITED_SECTIONS
            and all(value in {"pass", "fail"} for value in section_status.values()),
            f"audited_samples:{index}: reasoning/answer status is incomplete",
        )
        combined_status = sample.get("status")
        expected_status = (
            "pass" if all(section_status[section] == "pass" for section in AUDITED_SECTIONS) else "fail"
        )
        _require(
            combined_status == expected_status,
            f"audited_samples:{index}: combined status is inconsistent",
        )
        if combined_status == "fail":
            failed_keys.add(key)
        else:
            passed += 1
    _require(seen == set(row_parts), "semantic audit does not cover the exact data population")
    _require(passed == counts["passed"], "semantic audit passed count is inconsistent")
    _require(len(failed_keys) == counts["failed"], "semantic audit failed count is inconsistent")

    failed_samples = audit.get("failed_samples")
    _require(isinstance(failed_samples, list), "semantic audit failed_samples is invalid")
    declared_failed: set[tuple[str, str]] = set()
    for index, record in enumerate(failed_samples, 1):
        _require(isinstance(record, dict), f"failed_samples:{index}: invalid record")
        split = record.get("split")
        _require(split in SPLITS, f"failed_samples:{index}: invalid split")
        row_sha = _require_sha256(
            record.get("row_sha256"), label=f"failed_samples:{index}: row"
        )
        key = (str(split), row_sha)
        _require(key not in declared_failed, f"failed_samples:{index}: duplicate record")
        declared_failed.add(key)
    _require(
        declared_failed == failed_keys,
        "semantic audit failed_samples are not content-bound to failed decisions",
    )
    _require(not failed_keys, "semantic audit contains failed samples")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    directories = sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for path in directories:
        _fsync_directory(path)
    _fsync_directory(root)


def _seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        _require(not path.is_symlink(), f"refusing to seal symlink: {path}")
        path.chmod(0o444 if path.is_file() else 0o555)
    root.chmod(0o555)


def _remove_tree(root: Path) -> None:
    if not root.exists() or root.is_symlink():
        return
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts)):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o755)
    root.chmod(0o755)
    shutil.rmtree(root)


def _rename_noreplace(staging: Path, release: Path) -> None:
    """Atomically publish ``staging`` without any overwrite branch."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    _require(renameat2 is not None, "atomic renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(staging),
        -100,
        os.fsencode(release),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise PublishError(f"release already exists: {release}")
    raise PublishError(f"atomic immutable publish failed: {os.strerror(error)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compression-output", type=Path, required=True)
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.compression_output.resolve()
    audit_path = args.audit_summary.resolve()
    # Do not resolve the final path through a pre-existing symlink.  The exact
    # user-selected name is the immutable publication target.
    release = args.release_dir.expanduser().absolute()
    if release.exists() or release.is_symlink():
        raise PublishError(f"release already exists: {release}")

    _require(source.is_dir() and not source.is_symlink(), f"invalid compression output: {source}")
    run_manifest = source / "run_manifest.json"
    _, run_manifest_sha256 = _read_json(run_manifest, label="compression run manifest")
    audit, audit_sha256 = _read_json(audit_path, label="semantic audit summary")
    split_rows, row_parts, source_split_files = _read_source_rows(source)
    _validate_audit(audit, row_parts=row_parts, split_files=source_split_files)

    release.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{release.name}.", dir=release.parent))
    published = False
    try:
        dataset = staging / "analysis_sft"
        dataset.mkdir()
        split_counts: dict[str, int] = {}
        split_files: dict[str, dict[str, object]] = {}
        for split in SPLITS:
            output_path = dataset / f"{split}.jsonl"
            output_path.write_text(
                "".join(_canonical_json(row) + "\n" for row in split_rows[split]),
                encoding="utf-8",
            )
            split_counts[split] = len(split_rows[split])
            split_files[split] = {
                "path": f"analysis_sft/{split}.jsonl",
                "rows": len(split_rows[split]),
                "sha256": _sha256_file(output_path),
            }
        counts = audit["counts"]
        manifest = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "release_id": release.name,
            "created_at_utc": _utc_now(),
            "quality_status": "passed",
            "status": "passed",
            "immutable": True,
            "source_compression_output": str(source),
            "source_run_manifest_sha256": run_manifest_sha256,
            "source_split_files": {
                split: dict(source_split_files[split]) for split in SPLITS
            },
            "semantic_audit": {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "path": str(audit_path),
                "sha256": audit_sha256,
                "audited_sections": sorted(AUDITED_SECTIONS),
                "total": counts["total"],
                "passed": counts["passed"],
                "excluded": 0,
                "provider_failed": counts["provider_failed"],
                "binding": "canonical_row_and_reasoning_and_answer_sha256",
            },
            "excluded_samples": [],
            "split_counts": split_counts,
            "split_files": split_files,
        }
        (staging / "release_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _fsync_tree(staging)
        _seal_tree(staging)
        _fsync_tree(staging)
        _fsync_directory(staging.parent)
        _rename_noreplace(staging, release)
        _fsync_directory(release.parent)
        published = True
    finally:
        if not published:
            _remove_tree(staging)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
