"""Publish the completed paper chk-2 v6 recovery as an immutable SFT release.

The command is deliberately API-free.  It independently replays the sealed
source handoff and :func:`recover.load_and_verify_recovery`, projects PASS rows
in the original source-handoff order, and publishes a strict ``prompt`` /
``response`` dataset with complete audit and parent-checkpoint provenance.

An existing destination is never overwritten.  It is accepted only when the
entire release verifies byte-for-byte and semantically.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jobs.generation import generate_paper_chk2_chk1_analysis_rewrite as v5
from jobs.generation import publish_paper_chk2_chk1_analysis_rewrite as v5_publisher
from jobs.generation import publish_paper_chk2_downstream_v6 as legacy_publisher
from jobs.generation import recover_paper_chk2_downstream_v6 as recover
from jobs.generation import seal_paper_chk2_source_handoff_v1 as source_sealer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ACQUISITION_ROOT = source_sealer.DEFAULT_ACQUISITION_ROOT
DEFAULT_SOURCE_HANDOFF_ROOT = source_sealer.DEFAULT_HANDOFF_ROOT
DEFAULT_ORIGINAL_ROOT = recover.DEFAULT_ORIGINAL_ROOT
DEFAULT_RECOVERY_ROOT = recover.DEFAULT_RECOVERY_ROOT
DEFAULT_TOKENIZER_PATH = recover.DEFAULT_TOKENIZER_PATH
DEFAULT_RELEASE_ROOT = REPO_ROOT / (
    "dataset/processed/retrain_v2/"
    "chk2_chk1_final_analysis_synthetic_minutes_flash_official_reference_"
    "v6_downstream128_recovery_v1_20260831"
)
DEFAULT_PARENT_MODEL_ROOT = REPO_ROOT / (
    "output/training/retrain_v2/"
    "chk1_clean_v2_lr1e6_selected_cp200_for_chk2_20260810/merged/chk1"
)
DEFAULT_PARENT_AUTHORIZATION = REPO_ROOT / (
    "docs/summary/20260810T124500Z/chk1_cp200_merge_for_chk2/"
    "chk1_cp200_to_chk2_authorization.json"
)
DEFAULT_PARENT_CHECKPOINT_MANIFEST = REPO_ROOT / (
    "docs/summary/20260810T124500Z/chk1_cp200_merge_for_chk2/checkpoint_manifest.json"
)

SPLITS = ("train", "validation", "test")
EXPECTED_SOURCE_ROWS = 1_743
EXPECTED_PASS_COUNTS = {"train": 305, "validation": 42, "test": 44}
EXPECTED_REJECT_ROWS = 1_352
EXPECTED_COMPATIBILITY_COUNT = 22
EXPECTED_COMPATIBILITY_ID_DIGEST = (
    "e62415eecfdb37df240ab7259aba9e244d8fcdfbc8b3fecf907aaf70c5fd5ca7"
)

RELEASE_SCHEMA_VERSION = "paper-chk2-downstream-recovery-release-v1"
RELEASE_HANDOFF_SCHEMA_VERSION = "paper-chk2-downstream-recovery-release-handoff-v1"
PROMPT_CONTRACT_SCHEMA_VERSION = "paper-chk2-student-prompt-contract-v1"
DATASET_ROLE = "paper_chk2_chk1_final_analysis_to_synthetic_minutes_sft_v6_recovery"
TRAINING_SCOPE = "paper-chk2-chk1-cp200-minutes-sft-v6-downstream128"

EXPECTED_SYSTEM_PROMPT_SHA256 = (
    "4730a4ed585238547447ab850836db5a9fc67e5c3b1328b485c88d701ae78c4e"
)
EXPECTED_USER_PROMPT_TEMPLATE_SHA256 = (
    "423e79849cb66d6361c986f705ea4d4b16a03e28fec5826113eb9c8030a976d0"
)
EXPECTED_PARENT_MODEL_SHA256 = (
    "0989b94792f8ab6377e2979aaedeb37010b9a2c5e05c77a459b4e5cda5b806e3"
)
EXPECTED_PARENT_AUTHORIZATION_SHA256 = (
    "c8b400ec48be0fd31929f07dea3d30422a2c2638306ab55f4e2e264cb161cb3c"
)
EXPECTED_PARENT_AUTHORIZATION_FILE_SHA256 = (
    "e4e80fbfdd805b3691b3b05fbf3e3f26353eeef399fb7a08f156260e7f0ca554"
)
EXPECTED_PARENT_CHECKPOINT_MANIFEST_FILE_SHA256 = (
    "59758ab4d2b54592f632776c14d2b06c21c56416e4eb83e09d896f6a06d0a687"
)

PublicationError = legacy_publisher.PublicationError
VerifiedSourceHandoff = legacy_publisher.VerifiedSourceHandoff


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return v5.sha256_text(value)


def sha256_file(path: Path) -> str:
    return v5.sha256_file(path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicationError(message)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(
        path.is_file() and not path.is_symlink(), f"missing or unsafe {label}: {path}"
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationError(f"invalid {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} must be an object: {path}")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _require(
        path.is_file() and not path.is_symlink(), f"missing or unsafe {label}: {path}"
    )
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PublicationError(f"invalid {label} line {number}: {path}") from exc
        _require(isinstance(value, dict), f"non-object {label} line {number}: {path}")
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
    path.write_text(
        "".join(canonical_json(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _descriptor(path: Path, *, root: Path) -> dict[str, Any]:
    _require(
        path.is_file() and not path.is_symlink(), f"unsafe release artifact: {path}"
    )
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise PublicationError(
            f"release artifact escapes staging root: {path}"
        ) from exc
    result: dict[str, Any] = {
        "path": relative.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".jsonl":
        result["rows"] = len(path.read_text(encoding="utf-8").splitlines())
    return result


def _publisher_implementation_contract() -> dict[str, Any]:
    paths = {
        "recovery_publisher": Path(__file__).resolve(),
        "recovery_verifier": Path(recover.__file__).resolve(),
        "token_replay_publisher": Path(v5_publisher.__file__).resolve(),
    }
    artifacts: dict[str, dict[str, str]] = {}
    for label, path in sorted(paths.items()):
        _require(
            path.is_file() and not path.is_symlink(), f"unsafe implementation: {label}"
        )
        artifacts[label] = {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(path),
        }
    return {
        "artifacts": artifacts,
        "composite_sha256": sha256_text(canonical_json(artifacts)),
    }


def _student_prompt_contract() -> dict[str, Any]:
    system_sha = sha256_text(v5.STUDENT_SYSTEM_PROMPT)
    user_sha = sha256_text(v5.STUDENT_USER_PROMPT_TEMPLATE)
    _require(system_sha == EXPECTED_SYSTEM_PROMPT_SHA256, "student system prompt drift")
    _require(
        user_sha == EXPECTED_USER_PROMPT_TEMPLATE_SHA256,
        "student user-prompt template drift",
    )
    return {
        "schema_version": PROMPT_CONTRACT_SCHEMA_VERSION,
        "system_prompt": v5.STUDENT_SYSTEM_PROMPT,
        "system_prompt_sha256": system_sha,
        "user_prompt_template": v5.STUDENT_USER_PROMPT_TEMPLATE,
        "user_prompt_template_sha256": user_sha,
        "response_boundary": "</think>",
        "opening_think_supplied_by_chat_template": True,
    }


def _directory_fingerprint(root: Path) -> dict[str, Any]:
    """Recompute the historical sorted-file directory digest."""

    directory = root.resolve()
    _require(
        directory.is_dir() and not directory.is_symlink(), "unsafe parent model root"
    )
    files: list[Path] = []
    for current, dirnames, filenames in os.walk(directory, followlinks=False):
        current_path = Path(current)
        for dirname in dirnames:
            _require(
                not (current_path / dirname).is_symlink(),
                "symlink forbidden in parent model",
            )
        for filename in filenames:
            path = current_path / filename
            _require(
                path.is_file() and not path.is_symlink(), "unsafe parent model file"
            )
            files.append(path)
    records: list[str] = []
    total_bytes = 0
    for path in sorted(files, key=lambda item: item.relative_to(directory).as_posix()):
        relative = path.relative_to(directory).as_posix()
        size = path.stat().st_size
        total_bytes += size
        records.append(f"{relative}\0{size}\0{sha256_file(path)}\n")
    return {
        "algorithm": (
            "sha256(sorted UTF-8 records '<relative_path>\\0<size>\\0<file_sha256>\\n')"
        ),
        "kind": "directory",
        "file_count": len(files),
        "total_bytes": total_bytes,
        "sha256": hashlib.sha256("".join(records).encode("utf-8")).hexdigest(),
    }


def _validate_parent_binding(
    *,
    model_root: Path,
    authorization_path: Path,
    checkpoint_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    authorization = _read_json(authorization_path, label="chk1-to-chk2 authorization")
    unsigned_authorization = dict(authorization)
    authorization_sha = unsigned_authorization.pop("authorization_sha256", None)
    _require(
        authorization_sha
        == sha256_text(canonical_json(unsigned_authorization))
        == EXPECTED_PARENT_AUTHORIZATION_SHA256,
        "parent authorization signature drift",
    )
    _require(
        authorization.get("schema_version")
        == "chk1-cp200-to-chk2-override-authorization-v1"
        and authorization.get("status") == "authorized"
        and authorization.get("scope", {}).get("target_stage") == "chk2"
        and authorization.get("scope", {}).get("downstream_stages_allowed") == ["chk2"]
        and authorization.get("scope", {}).get("further_downstream_stages_allowed")
        == [],
        "parent authorization scope drift",
    )
    _require(
        sha256_file(authorization_path) == EXPECTED_PARENT_AUTHORIZATION_FILE_SHA256,
        "parent authorization file drift",
    )

    checkpoint = _read_json(
        checkpoint_manifest_path, label="parent checkpoint manifest"
    )
    unsigned_checkpoint = dict(checkpoint)
    integrity = unsigned_checkpoint.pop("integrity", None)
    _require(
        isinstance(integrity, Mapping)
        and integrity.get("payload_sha256")
        == sha256_text(canonical_json(unsigned_checkpoint)),
        "parent checkpoint manifest integrity drift",
    )
    _require(
        sha256_file(checkpoint_manifest_path)
        == EXPECTED_PARENT_CHECKPOINT_MANIFEST_FILE_SHA256,
        "parent checkpoint manifest file drift",
    )
    declared = checkpoint.get("model_fingerprint")
    _require(isinstance(declared, Mapping), "parent model fingerprint missing")
    observed = _directory_fingerprint(model_root)
    _require(
        declared.get("sha256") == observed["sha256"] == EXPECTED_PARENT_MODEL_SHA256
        and declared.get("file_count") == observed["file_count"]
        and declared.get("total_bytes") == observed["total_bytes"],
        "parent model directory digest drift",
    )
    checkpoint_authorization = checkpoint.get("authorization")
    _require(
        checkpoint.get("schema_version") == "chk1-cp200-merged-checkpoint-manifest-v1"
        and checkpoint.get("status") == "ready_for_chk2_parent_under_explicit_override"
        and checkpoint.get("scope", {}).get("allowed_stage") == "chk2"
        and checkpoint.get("scope", {}).get("further_downstream_stages_allowed") == []
        and isinstance(checkpoint_authorization, Mapping)
        and checkpoint_authorization.get("authorization_sha256") == authorization_sha
        and checkpoint_authorization.get("file_sha256")
        == sha256_file(authorization_path),
        "parent checkpoint authorization binding drift",
    )
    binding = {
        "schema_version": checkpoint["schema_version"],
        "model_path": str(model_root.resolve().relative_to(REPO_ROOT)),
        "model_sha256": observed["sha256"],
        "checkpoint_manifest_file_sha256": sha256_file(checkpoint_manifest_path),
        "authorization_schema_version": authorization["schema_version"],
        "authorization_sha256": authorization_sha,
        "authorization_file_sha256": sha256_file(authorization_path),
        "allowed_stage": "chk2",
        "further_downstream_stages_allowed": [],
    }
    return binding, {
        "parent_checkpoint_manifest.json": checkpoint_manifest_path,
        "parent_authorization.json": authorization_path,
    }


def _load_tokenizer(tokenizer_path: Path, tokenizer: Any | None) -> Any:
    if tokenizer is None:
        _require(
            tokenizer_path.resolve() == DEFAULT_TOKENIZER_PATH.resolve(),
            "tokenizer path is not the authorized chk1 cp200 tokenizer",
        )
        try:
            v5_publisher._validate_exact_tokenizer_files(
                tokenizer_path, expected_tokenizer_path=DEFAULT_TOKENIZER_PATH
            )
        except Exception as exc:
            raise PublicationError(f"tokenizer file contract drift: {exc}") from exc
        tokenizer = v5._load_tokenizer(tokenizer_path)
    try:
        v5._verify_tokenizer_runtime_contract(tokenizer)
    except Exception as exc:
        raise PublicationError(f"tokenizer runtime contract drift: {exc}") from exc
    return tokenizer


def _expected_pass_manifest(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "split": record["split"],
        "source_index": record["source_index"],
        "meeting_date": record["meeting_date"],
        "atomic_topic": record["atomic_topic"],
        "section_style_id": record["section_style_id"],
        "terminal_status": record["terminal_status"],
        "source_analysis_sha256": record["source_analysis_sha256"],
        "prompt_sha256": record["prompt_sha256"],
        "response_sha256": record["response_sha256"],
        "lineage": record["lineage"],
    }


def _verify_and_project_recovery(
    *,
    recovery_root: Path,
    original_root: Path,
    source: VerifiedSourceHandoff,
    tokenizer: Any,
) -> dict[str, Any]:
    try:
        verified = recover.load_and_verify_recovery(
            recovery_root,
            original_root=original_root,
            handoff=source.handoff,
            tokenizer=tokenizer,
        )
    except Exception as exc:
        raise PublicationError(f"recovery replay failed: {exc}") from exc
    receipt = verified.receipt
    state = verified.state
    _require(
        receipt.get("status") == "complete"
        and receipt.get("schema_version") == recover.RECEIPT_SCHEMA
        and receipt.get("final_terminal_count") == EXPECTED_SOURCE_ROWS
        and receipt.get("final_missing_ids") == []
        and len(state.terminals) == EXPECTED_SOURCE_ROWS,
        "recovery completion/population drift",
    )
    _require(
        receipt.get("source_handoff_manifest_sha256") == source.signed_manifest_sha256,
        "recovery/source handoff binding drift",
    )

    compatibility_ids = sorted(
        str(item) for item in receipt.get("compatibility_ids", [])
    )
    _require(
        len(compatibility_ids) == EXPECTED_COMPATIBILITY_COUNT
        and sha256_text(canonical_json(compatibility_ids))
        == EXPECTED_COMPATIBILITY_ID_DIGEST,
        "recovery compatibility scope drift",
    )
    _require(
        all(
            state.terminals.get(sample_id, {}).get("terminal_status")
            != v5.TERMINAL_PASS
            for sample_id in compatibility_ids
        ),
        "compatibility replay admitted a training row",
    )

    rows: dict[str, list[dict[str, str]]] = {split: [] for split in SPLITS}
    manifests: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    terminals: dict[str, list[Mapping[str, Any]]] = {split: [] for split in SPLITS}
    tokenizer_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for split in SPLITS:
        for source_row in source.handoff.prepared[split]:
            terminal = state.terminals.get(source_row.sample_id)
            _require(
                terminal is not None,
                f"missing recovery terminal: {source_row.sample_id}",
            )
            _require(
                terminal.get("sample_id") == source_row.sample_id
                and terminal.get("split") == split
                and terminal.get("source_index") == source_row.source_line_number,
                f"recovery terminal source-order binding drift: {source_row.sample_id}",
            )
            _require(
                source_row.sample_id not in seen_ids, "duplicate recovery sample ID"
            )
            seen_ids.add(source_row.sample_id)
            terminals[split].append(terminal)
            if terminal.get("terminal_status") != v5.TERMINAL_PASS:
                continue
            training_row = {
                "prompt": terminal.get("student_prompt"),
                "response": terminal.get("sft_response"),
            }
            _require(
                all(
                    isinstance(value, str) and value for value in training_row.values()
                ),
                f"invalid PASS training text: {source_row.sample_id}",
            )
            _require(
                terminal.get("prompt_sha256") == sha256_text(training_row["prompt"])
                and terminal.get("response_sha256")
                == sha256_text(training_row["response"]),
                f"PASS prompt/response digest drift: {source_row.sample_id}",
            )
            manifest_row = _expected_pass_manifest(terminal)
            rows[split].append(training_row)
            manifests[split].append(manifest_row)
            replay = v5._tokenizer_replay(
                row=source_row,
                response=training_row["response"],
                tokenizer=tokenizer,
            )
            tokenizer_rows.append(
                {"sample_id": source_row.sample_id, "split": split, **replay}
            )

    _require(len(seen_ids) == EXPECTED_SOURCE_ROWS, "recovery source closure drift")
    split_counts = {split: len(rows[split]) for split in SPLITS}
    _require(split_counts == EXPECTED_PASS_COUNTS, "recovery PASS split-count drift")
    producer_tokenizer_rows = _read_jsonl(
        verified.tokenizer_audit_path, label="recovery tokenizer replay"
    )
    _require(
        producer_tokenizer_rows == tokenizer_rows,
        "recovery tokenizer replay content/order drift",
    )
    return {
        "verified": verified,
        "terminals": terminals,
        "pass_rows": rows,
        "pass_manifests": manifests,
        "tokenizer_replay": tokenizer_rows,
        "compatibility_ids": compatibility_ids,
    }


def _copy_provenance(
    *, staging: Path, sources: Mapping[str, Path], artifacts: dict[str, Any]
) -> None:
    for filename, source_path in sources.items():
        _require(
            source_path.is_file() and not source_path.is_symlink(),
            f"missing or unsafe provenance source: {source_path}",
        )
        target = staging / "provenance" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)
        key = Path(filename).stem.replace("-", "_")
        artifacts[key] = _descriptor(target, root=staging)


def _build_release_tree(
    *,
    staging: Path,
    source: VerifiedSourceHandoff,
    projected: Mapping[str, Any],
    parent_binding: Mapping[str, Any],
    parent_sources: Mapping[str, Path],
) -> dict[str, Any]:
    verified: recover.VerifiedRecovery = projected["verified"]
    terminals = projected["terminals"]
    artifacts: dict[str, Any] = {
        "minutes_alignment": {},
        "audits": {},
        "provenance": {},
    }
    prompt_contract_path = staging / "prompt_contract.json"
    _write_json(prompt_contract_path, _student_prompt_contract())
    artifacts["provenance"]["student_prompt_contract"] = _descriptor(
        prompt_contract_path, root=staging
    )

    split_counts: dict[str, int] = {}
    status_counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        rows = [dict(row) for row in projected["pass_rows"][split]]
        sidecars = []
        for release_index, (row, source_sidecar) in enumerate(
            zip(rows, projected["pass_manifests"][split])
        ):
            _require(set(row) == {"prompt", "response"}, "training row schema drift")
            sidecars.append(
                {
                    **dict(source_sidecar),
                    "release_index": release_index,
                    "prompt_sha256": sha256_text(row["prompt"]),
                    "response_sha256": sha256_text(row["response"]),
                }
            )
        data_path = staging / "minutes_alignment" / f"{split}.jsonl"
        manifest_path = staging / "minutes_alignment/manifests" / f"{split}.jsonl"
        _write_jsonl(data_path, rows)
        _write_jsonl(manifest_path, sidecars)
        artifacts["minutes_alignment"][split] = {
            "data": _descriptor(data_path, root=staging),
            "manifest": _descriptor(manifest_path, root=staging),
        }
        split_counts[split] = len(rows)
        status_counts[split] = dict(
            sorted(
                Counter(str(row["terminal_status"]) for row in terminals[split]).items()
            )
        )

    all_terminals = [record for split in SPLITS for record in terminals[split]]
    rejections = [
        {
            "sample_id": record["sample_id"],
            "split": record["split"],
            "source_index": record["source_index"],
            "terminal_status": record["terminal_status"],
            "rejection_stage": record["rejection_stage"],
            "rejection_reasons": record["rejection_reasons"],
        }
        for record in all_terminals
        if record["terminal_status"] != v5.TERMINAL_PASS
    ]
    source_admission = [
        {
            "sample_id": record["sample_id"],
            "split": record["split"],
            "source_index": record["source_index"],
            "terminal_status": record["terminal_status"],
            "source_analysis_sha256": record["source_analysis_sha256"],
            "provided_data_sha256": record["provided_data_sha256"],
            "source_audit": record["source_audit"],
        }
        for record in all_terminals
    ]
    validator_a = [
        {
            "sample_id": record["sample_id"],
            "split": record["split"],
            "source_index": record["source_index"],
            "terminal_status": record["terminal_status"],
            "validator_a": record["validator_a"],
        }
        for record in all_terminals
        if record.get("validator_a")
    ]
    validator_b = [
        {
            "sample_id": record["sample_id"],
            "split": record["split"],
            "source_index": record["source_index"],
            "terminal_status": record["terminal_status"],
            "validator_b": record["validator_b"],
        }
        for record in all_terminals
        if record.get("validator_b")
    ]
    repair_history = [
        {
            "sample_id": record["sample_id"],
            "split": record["split"],
            "source_index": record["source_index"],
            "terminal_status": record["terminal_status"],
            "events": record["repair_history"],
        }
        for record in all_terminals
        if record.get("repair_history")
    ]
    evidence_ledger = [
        {
            "sample_id": record["sample_id"],
            "split": record["split"],
            "source_index": record["source_index"],
            "terminal_status": record["terminal_status"],
            "terminal_record_sha256": sha256_text(canonical_json(record)),
            "source_audit_sha256": sha256_text(canonical_json(record["source_audit"])),
            "validator_a_sha256": (
                sha256_text(canonical_json(record["validator_a"]))
                if record.get("validator_a")
                else None
            ),
            "validator_b_sha256": (
                sha256_text(canonical_json(record["validator_b"]))
                if record.get("validator_b")
                else None
            ),
            "repair_history_sha256": (
                sha256_text(canonical_json(record["repair_history"]))
                if record.get("repair_history")
                else None
            ),
        }
        for record in all_terminals
    ]
    compatibility = [
        {
            "sample_id": sample_id,
            "split": verified.state.terminals[sample_id]["split"],
            "terminal_status": verified.state.terminals[sample_id]["terminal_status"],
            "rejection_stage": verified.state.terminals[sample_id]["rejection_stage"],
            "terminal_record_sha256": sha256_text(
                canonical_json(verified.state.terminals[sample_id])
            ),
        }
        for sample_id in projected["compatibility_ids"]
    ]
    repair_distribution = Counter(
        f"{row['split']}:{event['repair_type']}"
        for row in repair_history
        for event in row["events"]
    )
    audit_rows = {
        "rejections": rejections,
        "source_admission": source_admission,
        "validator_a": validator_a,
        "validator_b": validator_b,
        "repair_history": repair_history,
        "evidence_ledger": evidence_ledger,
        "compatibility_replay": compatibility,
        "tokenizer_replay": projected["tokenizer_replay"],
    }
    for name, rows in audit_rows.items():
        path = staging / "audits" / f"{name}.jsonl"
        _write_jsonl(path, rows)
        artifacts["audits"][name] = _descriptor(path, root=staging)
    quality = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "source_rows": EXPECTED_SOURCE_ROWS,
        "split_pass_counts": split_counts,
        "terminal_status_counts": status_counts,
        "rejected_rows": len(rejections),
        "pass_rows": sum(split_counts.values()),
        "three_pass_splits_nonempty": all(split_counts.values()),
        "validator_a_rows": len(validator_a),
        "validator_b_rows": len(validator_b),
        "repair_event_distribution": dict(sorted(repair_distribution.items())),
        "evidence_ledger_rows": len(evidence_ledger),
        "compatibility_count": len(compatibility),
        "compatibility_all_rejected": all(
            row["terminal_status"] != v5.TERMINAL_PASS for row in compatibility
        ),
        "external_cache_counts": dict(verified.receipt["cache_counts"]),
    }
    quality_path = staging / "audits/data_quality.json"
    _write_json(quality_path, quality)
    artifacts["audits"]["data_quality"] = _descriptor(quality_path, root=staging)

    provenance_sources: dict[str, Path] = {
        "source_handoff_manifest.json": source.handoff.root / "handoff_manifest.json",
        "source_admission_receipt.json": (
            source.handoff.root / "sealed_source/source_admission_receipt.json"
        ),
        "source_prompt_contract.json": (
            source.handoff.root / "sealed_source/prompt_contract.json"
        ),
        "official_pre_action_reference_bank.jsonl": (
            source.handoff.root
            / "sealed_source/official_pre_action_reference_bank.jsonl"
        ),
        "recovery_partial_handoff_manifest.json": (
            verified.root / "partial_handoff_manifest.json"
        ),
        "recovery_receipt.json": verified.root / "recovery_receipt.json",
        "recovery_prompt_contract.json": (
            verified.root / "acquisition/prompt_contract.json"
        ),
        **dict(parent_sources),
    }
    _copy_provenance(
        staging=staging, sources=provenance_sources, artifacts=artifacts["provenance"]
    )
    attempts: dict[str, Any] = {}
    for descriptor in verified.receipt["attempt_artifacts"]:
        source_path = verified.root / descriptor["path"]
        _require(
            source_path.stat().st_size == descriptor["bytes"]
            and sha256_file(source_path) == descriptor["sha256"],
            "recovery attempt artifact drift",
        )
        target = staging / "provenance/recovery_attempts" / source_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)
        attempts[source_path.stem.replace("-", "_")] = _descriptor(target, root=staging)
    artifacts["provenance"]["recovery_attempts"] = attempts

    recovery_receipt_copy = staging / "provenance/recovery_receipt.json"
    partial_copy = staging / "provenance/recovery_partial_handoff_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "status": "complete",
        "quality_status": "passed",
        "dataset_role": DATASET_ROLE,
        "training_scope": TRAINING_SCOPE,
        "immutable": True,
        "training_ready": True,
        "training_only": True,
        "evaluation_eligible": False,
        "dag_bindable": False,
        "promotable_as_canonical_chk2": False,
        "source_rows": EXPECTED_SOURCE_ROWS,
        "split_pass_counts": split_counts,
        "terminal_status_counts": status_counts,
        "student_prompt_contract": {
            "system_prompt_sha256": EXPECTED_SYSTEM_PROMPT_SHA256,
            "user_prompt_template_sha256": EXPECTED_USER_PROMPT_TEMPLATE_SHA256,
            "response_boundary": "</think>",
        },
        "parent_checkpoint": dict(parent_binding),
        "source_handoff": {
            "schema_version": source_sealer.SCHEMA_VERSION,
            "manifest_sha256": source.signed_manifest_sha256,
            "manifest_file_sha256": sha256_file(
                staging / "provenance/source_handoff_manifest.json"
            ),
        },
        "recovery": {
            "schema_version": recover.RECEIPT_SCHEMA,
            "partial_handoff_manifest_sha256": verified.manifest["manifest_sha256"],
            "partial_handoff_manifest_file_sha256": sha256_file(partial_copy),
            "receipt_sha256": verified.receipt["receipt_sha256"],
            "receipt_file_sha256": sha256_file(recovery_receipt_copy),
            "legacy_v6_run_binding_sha256": verified.receipt[
                "legacy_v6_run_binding_sha256"
            ],
            "compatibility_ids": list(projected["compatibility_ids"]),
            "compatibility_id_digest": EXPECTED_COMPATIBILITY_ID_DIGEST,
        },
        "provider_identities": dict(verified.receipt["provider_identities"]),
        "publisher_implementation": _publisher_implementation_contract(),
        "tokenizer_runtime_contract": v5._expected_tokenizer_runtime_contract(),
        "artifacts": artifacts,
    }
    manifest["manifest_sha256"] = sha256_text(canonical_json(manifest))
    manifest_path = staging / "release_manifest.json"
    _write_json(manifest_path, manifest)
    handoff = {
        "schema_version": RELEASE_HANDOFF_SCHEMA_VERSION,
        "status": "complete",
        "created_at": _utc_now(),
        "release_manifest_sha256": manifest["manifest_sha256"],
        "release_manifest_file_sha256": sha256_file(manifest_path),
        "source_handoff_manifest_sha256": source.signed_manifest_sha256,
        "source_handoff_manifest_file_sha256": source.manifest_file_sha256,
        "recovery_receipt_sha256": verified.receipt["receipt_sha256"],
        "recovery_receipt_file_sha256": sha256_file(recovery_receipt_copy),
        "parent_authorization_sha256": parent_binding["authorization_sha256"],
        "parent_authorization_file_sha256": parent_binding["authorization_file_sha256"],
        "parent_checkpoint_model_sha256": parent_binding["model_sha256"],
        "publisher_implementation": manifest["publisher_implementation"],
        "split_data_sha256": {
            split: artifacts["minutes_alignment"][split]["data"]["sha256"]
            for split in SPLITS
        },
    }
    handoff["handoff_sha256"] = sha256_text(canonical_json(handoff))
    _write_json(staging / "handoff.json", handoff)
    return manifest


def _validate_signed_payload(
    payload: Mapping[str, Any], *, signature_field: str, label: str
) -> str:
    unsigned = dict(payload)
    stored = unsigned.pop(signature_field, None)
    _require(
        isinstance(stored, str) and stored == sha256_text(canonical_json(unsigned)),
        f"{label} signature drift",
    )
    return stored


def verify_release(
    release_root: Path = DEFAULT_RELEASE_ROOT,
    *,
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    release = release_root.absolute()
    _require(
        release.is_dir() and not release.is_symlink(), f"invalid release: {release}"
    )
    actual_files = legacy_publisher._strict_tree_files(release, label="release")
    manifest = _read_json(release / "release_manifest.json", label="release manifest")
    manifest_sha = _validate_signed_payload(
        manifest, signature_field="manifest_sha256", label="release manifest"
    )
    handoff = _read_json(release / "handoff.json", label="release handoff")
    handoff_sha = _validate_signed_payload(
        handoff, signature_field="handoff_sha256", label="release handoff"
    )
    _require(manifest_sha and handoff_sha, "release signatures missing")
    _require(
        manifest.get("schema_version") == RELEASE_SCHEMA_VERSION
        and manifest.get("status") == "complete"
        and manifest.get("quality_status") == "passed"
        and manifest.get("dataset_role") == DATASET_ROLE
        and manifest.get("training_scope") == TRAINING_SCOPE
        and all(
            manifest.get(field) is expected
            for field, expected in {
                "immutable": True,
                "training_ready": True,
                "training_only": True,
                "evaluation_eligible": False,
                "dag_bindable": False,
                "promotable_as_canonical_chk2": False,
            }.items()
        ),
        "release role/status/lineage drift",
    )
    _require(
        handoff.get("schema_version") == RELEASE_HANDOFF_SCHEMA_VERSION
        and handoff.get("status") == "complete"
        and handoff.get("release_manifest_sha256") == manifest_sha
        and handoff.get("release_manifest_file_sha256")
        == sha256_file(release / "release_manifest.json"),
        "release handoff/manifest binding drift",
    )
    implementation = _publisher_implementation_contract()
    _require(
        manifest.get("publisher_implementation") == implementation
        and handoff.get("publisher_implementation") == implementation,
        "release publisher implementation drift",
    )
    described: dict[Path, Mapping[str, Any]] = {}
    legacy_publisher._verify_descriptor_tree(
        release,
        manifest.get("artifacts"),
        label="release_manifest.artifacts",
        observed=described,
    )
    expected_files = set(described) | {
        (release / "release_manifest.json").absolute(),
        (release / "handoff.json").absolute(),
    }
    _require(
        {path.absolute() for path in actual_files} == expected_files,
        "release contains missing/orphan files",
    )
    prompt_path = legacy_publisher._resolve_downstream_artifact(
        release,
        manifest["artifacts"]["provenance"]["student_prompt_contract"],
        label="student prompt contract",
    )
    _require(
        _read_json(prompt_path, label="student prompt contract")
        == _student_prompt_contract()
        and manifest.get("student_prompt_contract")
        == {
            "system_prompt_sha256": EXPECTED_SYSTEM_PROMPT_SHA256,
            "user_prompt_template_sha256": EXPECTED_USER_PROMPT_TEMPLATE_SHA256,
            "response_boundary": "</think>",
        },
        "student prompt contract binding drift",
    )
    tokenizer = _load_tokenizer(tokenizer_path, tokenizer)

    split_counts: dict[str, int] = {}
    pass_ids: set[str] = set()
    replay_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        descriptors = manifest["artifacts"]["minutes_alignment"][split]
        data_path = legacy_publisher._resolve_downstream_artifact(
            release, descriptors["data"], label=f"{split} training data"
        )
        sidecar_path = legacy_publisher._resolve_downstream_artifact(
            release, descriptors["manifest"], label=f"{split} training manifest"
        )
        rows = _read_jsonl(data_path, label=f"{split} training data")
        sidecars = _read_jsonl(sidecar_path, label=f"{split} training manifest")
        _require(len(rows) == len(sidecars), f"{split} data/manifest row drift")
        for index, (row, sidecar) in enumerate(zip(rows, sidecars)):
            _require(
                set(row) == {"prompt", "response"}, f"{split}:{index} schema drift"
            )
            sample_id = sidecar.get("sample_id")
            _require(
                isinstance(sample_id, str)
                and sample_id not in pass_ids
                and sidecar.get("release_index") == index
                and sidecar.get("split") == split
                and sidecar.get("terminal_status") == v5.TERMINAL_PASS
                and sidecar.get("prompt_sha256") == sha256_text(row["prompt"])
                and sidecar.get("response_sha256") == sha256_text(row["response"]),
                f"{split}:{index} PASS sidecar drift",
            )
            pass_ids.add(sample_id)
            strict_replay = v5_publisher._token_replay(
                tokenizer=tokenizer,
                system_prompt=v5.STUDENT_SYSTEM_PROMPT,
                prompt=row["prompt"],
                response=row["response"],
                sample_id=sample_id,
            )
            replay_rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "prompt_tokens": strict_replay["prompt_tokens"],
                    "completion_tokens": strict_replay["completion_tokens"],
                    "total_tokens": strict_replay["total_tokens"],
                    "single_bos": True,
                    "single_eos": True,
                    "completion_only_prompt_masked": True,
                    "completion_mask_covers_reasoning_boundary_answer_eos": True,
                    "no_truncation": True,
                }
            )
        split_counts[split] = len(rows)
    _require(
        split_counts == EXPECTED_PASS_COUNTS
        and manifest.get("split_pass_counts") == EXPECTED_PASS_COUNTS
        and manifest.get("source_rows") == EXPECTED_SOURCE_ROWS,
        "release PASS split-count drift",
    )

    audit_descriptors = manifest["artifacts"]["audits"]
    audit_rows = {
        name: _read_jsonl(
            legacy_publisher._resolve_downstream_artifact(
                release, audit_descriptors[name], label=f"{name} audit"
            ),
            label=f"{name} audit",
        )
        for name in (
            "rejections",
            "source_admission",
            "validator_a",
            "validator_b",
            "repair_history",
            "evidence_ledger",
            "compatibility_replay",
            "tokenizer_replay",
        )
    }
    _require(audit_rows["tokenizer_replay"] == replay_rows, "tokenizer audit drift")
    ledger = audit_rows["evidence_ledger"]
    ledger_by_id = {str(row.get("sample_id")): row for row in ledger}
    rejection_ids = {str(row.get("sample_id")) for row in audit_rows["rejections"]}
    _require(
        len(ledger) == len(ledger_by_id) == EXPECTED_SOURCE_ROWS
        and len(rejection_ids) == EXPECTED_REJECT_ROWS
        and not pass_ids & rejection_ids
        and pass_ids | rejection_ids == set(ledger_by_id),
        "release PASS/REJECT partition drift",
    )
    _require(
        {str(row.get("sample_id")) for row in audit_rows["source_admission"]}
        == set(ledger_by_id),
        "source-admission audit population drift",
    )
    for name, hash_field, payload_field in (
        ("source_admission", "source_audit_sha256", "source_audit"),
        ("validator_a", "validator_a_sha256", "validator_a"),
        ("validator_b", "validator_b_sha256", "validator_b"),
        ("repair_history", "repair_history_sha256", "events"),
    ):
        expected_ids = {
            sample_id
            for sample_id, ledger_row in ledger_by_id.items()
            if ledger_row[hash_field] is not None
        }
        rows = audit_rows[name]
        _require(
            {str(row.get("sample_id")) for row in rows} == expected_ids,
            f"{name} audit population drift",
        )
        for row in rows:
            _require(
                sha256_text(canonical_json(row[payload_field]))
                == ledger_by_id[str(row["sample_id"])][hash_field],
                f"{name} audit evidence hash drift",
            )
    compatibility = audit_rows["compatibility_replay"]
    compatibility_ids = [str(row.get("sample_id")) for row in compatibility]
    _require(
        len(compatibility_ids) == EXPECTED_COMPATIBILITY_COUNT
        and compatibility_ids == sorted(compatibility_ids)
        and sha256_text(canonical_json(compatibility_ids))
        == EXPECTED_COMPATIBILITY_ID_DIGEST
        and all(row.get("terminal_status") != v5.TERMINAL_PASS for row in compatibility)
        and set(compatibility_ids) <= rejection_ids
        and manifest.get("recovery", {}).get("compatibility_ids") == compatibility_ids,
        "release compatibility rejection binding drift",
    )
    quality_path = legacy_publisher._resolve_downstream_artifact(
        release, audit_descriptors["data_quality"], label="data quality audit"
    )
    quality = _read_json(quality_path, label="data quality audit")
    _require(
        quality.get("schema_version") == RELEASE_SCHEMA_VERSION
        and quality.get("source_rows") == EXPECTED_SOURCE_ROWS
        and quality.get("pass_rows") == sum(EXPECTED_PASS_COUNTS.values())
        and quality.get("rejected_rows") == EXPECTED_REJECT_ROWS
        and quality.get("split_pass_counts") == EXPECTED_PASS_COUNTS
        and quality.get("evidence_ledger_rows") == EXPECTED_SOURCE_ROWS
        and quality.get("compatibility_count") == EXPECTED_COMPATIBILITY_COUNT
        and quality.get("compatibility_all_rejected") is True,
        "data quality audit drift",
    )

    provenance = manifest["artifacts"]["provenance"]
    provenance_paths = {
        name: legacy_publisher._resolve_downstream_artifact(
            release, provenance[name], label=f"{name} provenance"
        )
        for name in (
            "source_handoff_manifest",
            "recovery_partial_handoff_manifest",
            "recovery_receipt",
            "parent_checkpoint_manifest",
            "parent_authorization",
        )
    }
    source_manifest = _read_json(
        provenance_paths["source_handoff_manifest"], label="source handoff provenance"
    )
    partial_manifest = _read_json(
        provenance_paths["recovery_partial_handoff_manifest"],
        label="recovery partial handoff provenance",
    )
    receipt = _read_json(
        provenance_paths["recovery_receipt"], label="recovery receipt provenance"
    )
    _require(
        _validate_signed_payload(
            source_manifest,
            signature_field="manifest_sha256",
            label="source handoff provenance",
        )
        == manifest["source_handoff"]["manifest_sha256"],
        "source handoff provenance binding drift",
    )
    _require(
        _validate_signed_payload(
            partial_manifest,
            signature_field="manifest_sha256",
            label="recovery partial handoff provenance",
        )
        == manifest["recovery"]["partial_handoff_manifest_sha256"],
        "recovery partial handoff binding drift",
    )
    _require(
        _validate_signed_payload(
            receipt,
            signature_field="receipt_sha256",
            label="recovery receipt provenance",
        )
        == manifest["recovery"]["receipt_sha256"],
        "recovery receipt binding drift",
    )
    attempt_descriptors = provenance.get("recovery_attempts")
    _require(isinstance(attempt_descriptors, Mapping), "recovery attempts missing")
    copied_attempts = {
        Path(
            legacy_publisher._resolve_downstream_artifact(
                release, descriptor, label=f"{key} recovery attempt"
            )
        ).name: descriptor
        for key, descriptor in attempt_descriptors.items()
    }
    expected_attempts = {
        Path(item["path"]).name: item for item in receipt["attempt_artifacts"]
    }
    _require(
        set(copied_attempts) == set(expected_attempts)
        and all(
            copied_attempts[name]["bytes"] == expected_attempts[name]["bytes"]
            and copied_attempts[name]["sha256"] == expected_attempts[name]["sha256"]
            for name in copied_attempts
        ),
        "recovery attempt provenance binding drift",
    )
    authorization = _read_json(
        provenance_paths["parent_authorization"],
        label="parent authorization provenance",
    )
    checkpoint = _read_json(
        provenance_paths["parent_checkpoint_manifest"],
        label="parent checkpoint manifest provenance",
    )
    parent = manifest.get("parent_checkpoint", {})
    _require(
        authorization.get("authorization_sha256")
        == parent.get("authorization_sha256")
        == EXPECTED_PARENT_AUTHORIZATION_SHA256
        and sha256_file(provenance_paths["parent_authorization"])
        == parent.get("authorization_file_sha256")
        == EXPECTED_PARENT_AUTHORIZATION_FILE_SHA256
        and checkpoint.get("model_fingerprint", {}).get("sha256")
        == parent.get("model_sha256")
        == EXPECTED_PARENT_MODEL_SHA256
        and sha256_file(provenance_paths["parent_checkpoint_manifest"])
        == parent.get("checkpoint_manifest_file_sha256")
        == EXPECTED_PARENT_CHECKPOINT_MANIFEST_FILE_SHA256
        and parent.get("allowed_stage") == "chk2"
        and parent.get("further_downstream_stages_allowed") == [],
        "parent checkpoint release binding drift",
    )
    _require(
        handoff.get("source_handoff_manifest_sha256")
        == manifest["source_handoff"]["manifest_sha256"]
        and handoff.get("recovery_receipt_sha256")
        == manifest["recovery"]["receipt_sha256"]
        and handoff.get("parent_authorization_sha256") == parent["authorization_sha256"]
        and handoff.get("parent_checkpoint_model_sha256") == parent["model_sha256"]
        and handoff.get("split_data_sha256")
        == {
            split: manifest["artifacts"]["minutes_alignment"][split]["data"]["sha256"]
            for split in SPLITS
        },
        "release handoff provenance drift",
    )
    return manifest


def _atomic_publish(staging: Path, release: Path) -> None:
    _require(
        not release.exists() and not release.is_symlink(), "release already exists"
    )
    os.rename(staging, release)


def publish_release(
    *,
    source_acquisition_root: Path = DEFAULT_SOURCE_ACQUISITION_ROOT,
    source_handoff_root: Path = DEFAULT_SOURCE_HANDOFF_ROOT,
    recovery_root: Path = DEFAULT_RECOVERY_ROOT,
    original_root: Path = DEFAULT_ORIGINAL_ROOT,
    release_root: Path = DEFAULT_RELEASE_ROOT,
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
    parent_model_root: Path = DEFAULT_PARENT_MODEL_ROOT,
    parent_authorization_path: Path = DEFAULT_PARENT_AUTHORIZATION,
    parent_checkpoint_manifest_path: Path = DEFAULT_PARENT_CHECKPOINT_MANIFEST,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Replay the complete recovery and atomically publish PASS rows only."""

    release = release_root.absolute()
    if release.exists() or release.is_symlink():
        return verify_release(
            release, tokenizer_path=tokenizer_path, tokenizer=tokenizer
        )
    source = legacy_publisher._verify_source_handoff(
        source_acquisition_root=source_acquisition_root,
        source_handoff_root=source_handoff_root,
    )
    tokenizer = _load_tokenizer(tokenizer_path, tokenizer)
    projected = _verify_and_project_recovery(
        recovery_root=recovery_root,
        original_root=original_root,
        source=source,
        tokenizer=tokenizer,
    )
    parent_binding, parent_sources = _validate_parent_binding(
        model_root=parent_model_root,
        authorization_path=parent_authorization_path,
        checkpoint_manifest_path=parent_checkpoint_manifest_path,
    )
    release.parent.mkdir(parents=True, exist_ok=True)
    _require(not release.parent.is_symlink(), "unsafe release parent")
    lock_path = release.parent / f".{release.name}.publish.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if release.exists() or release.is_symlink():
            return verify_release(
                release, tokenizer_path=tokenizer_path, tokenizer=tokenizer
            )
        staging = Path(
            tempfile.mkdtemp(prefix=f".{release.name}.staging.", dir=release.parent)
        )
        try:
            _build_release_tree(
                staging=staging,
                source=source,
                projected=projected,
                parent_binding=parent_binding,
                parent_sources=parent_sources,
            )
            verify_release(staging, tokenizer_path=tokenizer_path, tokenizer=tokenizer)
            _atomic_publish(staging, release)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return verify_release(release, tokenizer_path=tokenizer_path, tokenizer=tokenizer)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-acquisition-root", type=Path, default=DEFAULT_SOURCE_ACQUISITION_ROOT
    )
    parser.add_argument(
        "--source-handoff-root", type=Path, default=DEFAULT_SOURCE_HANDOFF_ROOT
    )
    parser.add_argument("--recovery-root", type=Path, default=DEFAULT_RECOVERY_ROOT)
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument(
        "--parent-model-root", type=Path, default=DEFAULT_PARENT_MODEL_ROOT
    )
    parser.add_argument(
        "--parent-authorization", type=Path, default=DEFAULT_PARENT_AUTHORIZATION
    )
    parser.add_argument(
        "--parent-checkpoint-manifest",
        type=Path,
        default=DEFAULT_PARENT_CHECKPOINT_MANIFEST,
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.verify_only:
        result = verify_release(args.release_root, tokenizer_path=args.tokenizer_path)
    else:
        result = publish_release(
            source_acquisition_root=args.source_acquisition_root,
            source_handoff_root=args.source_handoff_root,
            recovery_root=args.recovery_root,
            original_root=args.original_root,
            release_root=args.release_root,
            tokenizer_path=args.tokenizer_path,
            parent_model_root=args.parent_model_root,
            parent_authorization_path=args.parent_authorization,
            parent_checkpoint_manifest_path=args.parent_checkpoint_manifest,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
