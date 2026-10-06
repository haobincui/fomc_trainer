"""Seal and verify the completed v5 source-admission stage for paper chk-2.

This module is deliberately outside the v5 implementation dependency set.  It
never calls a provider: it replays the immutable source terminals and their
provider caches, then emits a small downstream handoff with content hashes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from jobs.generation import generate_paper_chk2_chk1_analysis_rewrite as v5
from jobs.generation.paper_chk2_official_reference_v2 import (
    deserialize_official_reference_bank,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACQUISITION_ROOT = REPO_ROOT / (
    "output/data/retrain_v2/chk2/"
    "chk1_final_analysis_to_minutes_flash_official_reference_v2_20260831"
)
DEFAULT_HANDOFF_ROOT = REPO_ROOT / (
    "output/data/retrain_v2/chk2/chk1_final_analysis_source_audit_handoff_v1_20260831"
)
SCHEMA_VERSION = "paper-chk2-source-handoff-v1"
SPLITS = ("train", "validation", "test")


class SourceHandoffError(RuntimeError):
    """Raised when the source handoff cannot be sealed or verified."""


@dataclass(frozen=True)
class SourceHandoff:
    root: Path
    manifest: Mapping[str, Any]
    records: Mapping[str, tuple[Mapping[str, Any], ...]]
    prepared: Mapping[str, tuple[v5.PreparedRow, ...]]
    source_results: Mapping[str, Mapping[str, Any]]

    @property
    def rows(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(row for split in SPLITS for row in self.records[split])


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SourceHandoffError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceHandoffError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise SourceHandoffError(f"{label} must be an object: {path}")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise SourceHandoffError(f"missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SourceHandoffError(f"invalid {label} line {number}: {path}") from exc
        if not isinstance(value, dict):
            raise SourceHandoffError(f"non-object {label} line {number}: {path}")
        rows.append(value)
    return rows


def _artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    result = {
        "path": str(path.name if path.parent.name == "records" else path),
        "bytes": path.stat().st_size,
        "sha256": v5.sha256_file(path),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _store_immutable_text(path: Path, text: str) -> None:
    if path.is_symlink():
        raise SourceHandoffError(f"unsafe handoff symlink: {path}")
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise SourceHandoffError(f"immutable handoff conflict: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = path.open("x", encoding="utf-8")
    with descriptor:
        descriptor.write(text)
        descriptor.flush()


def _verify_artifact(root: Path, descriptor: Mapping[str, Any], label: str) -> Path:
    relative = descriptor.get("path")
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise SourceHandoffError(f"unsafe {label} artifact path")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise SourceHandoffError(f"missing {label} artifact: {path}")
    if descriptor.get("bytes") != path.stat().st_size or descriptor.get(
        "sha256"
    ) != v5.sha256_file(path):
        raise SourceHandoffError(f"{label} artifact drift: {path}")
    return path


def _prepared_rows(acquisition: Path) -> dict[str, list[v5.PreparedRow]]:
    result: dict[str, list[v5.PreparedRow]] = {}
    seen: set[str] = set()
    for split in SPLITS:
        raw = _read_jsonl(
            acquisition / "prepared" / f"{split}.jsonl", f"prepared {split}"
        )
        rows: list[v5.PreparedRow] = []
        for value in raw:
            try:
                row = v5.PreparedRow(**value)
            except (TypeError, ValueError) as exc:
                raise SourceHandoffError(f"prepared row shape drift: {split}") from exc
            if row.split != split or row.sample_id in seen:
                raise SourceHandoffError(f"prepared ID/split drift: {row.sample_id}")
            seen.add(row.sample_id)
            rows.append(row)
        result[split] = rows
    return result


def _source_status(result: Mapping[str, Any]) -> str:
    if result.get("machine_pass") is True:
        return "PASS"
    reports = result.get("result")
    repair = reports.get("contract_repair") if isinstance(reports, Mapping) else None
    if isinstance(repair, Mapping) and repair.get("contract_exhausted") is True:
        return v5.TERMINAL_SOURCE_CONTRACT_REJECT
    return v5.TERMINAL_SOURCE_REJECT


def seal_source_handoff(
    acquisition_root: str | Path = DEFAULT_ACQUISITION_ROOT,
    handoff_root: str | Path = DEFAULT_HANDOFF_ROOT,
    *,
    expected_total: int = v5.EXPECTED_TOTAL,
) -> SourceHandoff:
    acquisition = Path(acquisition_root).resolve()
    output = Path(handoff_root).resolve()
    prompt = _read_json(acquisition / "prompt_contract.json", "prompt contract")
    _read_json(acquisition / "prepare_manifest.json", "prepare manifest")
    _read_json(acquisition / "preparation_summary.json", "preparation summary")
    receipt_path = acquisition / "source_admission_receipt.json"
    receipt = _read_json(receipt_path, "source admission receipt")
    if (
        receipt.get("status") != "source_admission_complete"
        or receipt.get("unresolved_rows") != 0
    ):
        raise SourceHandoffError("source admission is not complete")
    code_sha = prompt.get("code_sha256")
    if not isinstance(code_sha, str) or len(code_sha) != 64:
        raise SourceHandoffError("prompt contract code SHA is invalid")
    current_implementation = v5._implementation_contract()
    if code_sha != current_implementation.get("composite_sha256"):
        raise SourceHandoffError("v5 implementation composite drift")
    config = v5.ProviderConfig()
    if prompt != v5._prompt_contract(code_sha256=code_sha, config=config):
        raise SourceHandoffError("complete prompt contract drift")
    reference_bytes = (
        acquisition / "official_pre_action_reference_bank.jsonl"
    ).read_bytes()
    try:
        reference_bank = deserialize_official_reference_bank(reference_bytes)
    except Exception as exc:
        raise SourceHandoffError("official reference bank replay failed") from exc
    prepared = _prepared_rows(acquisition)
    all_rows = [row for split in SPLITS for row in prepared[split]]
    if len(all_rows) != expected_total:
        raise SourceHandoffError(
            f"prepared total drift: {len(all_rows)} != {expected_total}"
        )
    try:
        v5.verify_official_reference_bank(reference_bank, all_rows)
    except Exception as exc:
        raise SourceHandoffError("official reference bank replay failed") from exc
    terminal_root = acquisition / "cache" / "source_terminal"
    terminal_files = (
        set(terminal_root.glob("*.json")) if terminal_root.is_dir() else set()
    )
    expected_files = {
        terminal_root / f"{v5.sha256_text(row.sample_id)}.json" for row in all_rows
    }
    if terminal_files != expected_files:
        raise SourceHandoffError("missing or orphan source terminal cache")

    identity = v5.ProviderIdentityRegistry()
    records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    status_counts: Counter[str] = Counter()
    expected_acquisition_provider_files: dict[str, set[Path]] = {
        role: set()
        for role in (
            v5.ROLE_SOURCE_AUDIT_PRIMARY,
            v5.ROLE_SOURCE_AUDIT_ADJUDICATION,
            v5.ROLE_SOURCE_AUDIT_CONTRACT_REPAIR,
        )
    }
    for split in SPLITS:
        for row in prepared[split]:
            path = terminal_root / f"{v5.sha256_text(row.sample_id)}.json"
            payload = _read_json(path, "source terminal")
            expected_binding = {
                "schema_version": v5.SOURCE_AUDIT_SCHEMA_VERSION,
                "sample_id": row.sample_id,
                "source_analysis_sha256": row.source_analysis_sha256,
                "provided_data_sha256": row.provided_data_sha256,
                "code_sha256": code_sha,
            }
            result = payload.get("result")
            if payload.get("binding") != expected_binding or not isinstance(
                result, dict
            ):
                raise SourceHandoffError(
                    f"source terminal binding drift: {row.sample_id}"
                )
            if payload.get("result_sha256") != v5.sha256_text(
                v5.canonical_json(result)
            ):
                raise SourceHandoffError(
                    f"source terminal result drift: {row.sample_id}"
                )
            try:
                v5._validate_source_terminal_result(row, result)
                v5._resume_source_terminal_provider_caches(
                    output=acquisition,
                    row=row,
                    result=result,
                    identity=identity,
                    environment={},
                    config=config,
                    code_sha256=code_sha,
                )
            except Exception as exc:
                raise SourceHandoffError(
                    f"source replay failed: {row.sample_id}: {exc}"
                ) from exc
            machine_pass = result.get("machine_pass") is True
            status = _source_status(result)
            status_counts[status] += 1
            providers = result.get("provider", {})
            cache_index: dict[str, Any] = {}
            sample_hash = v5.sha256_text(row.sample_id)
            sealed_terminal = output / "sealed_source/cache/source_terminal" / path.name
            _store_immutable_text(sealed_terminal, path.read_text(encoding="utf-8"))
            for role, provider in sorted(providers.items()):
                if role not in {
                    v5.ROLE_SOURCE_AUDIT_PRIMARY,
                    v5.ROLE_SOURCE_AUDIT_ADJUDICATION,
                    v5.ROLE_SOURCE_AUDIT_CONTRACT_REPAIR,
                }:
                    raise SourceHandoffError(f"unexpected provider role: {role}")
                source_cache = acquisition / "cache" / role / f"{sample_hash}.json"
                expected_acquisition_provider_files[role].add(source_cache)
                sealed_cache = output / "sealed_source/cache" / role / source_cache.name
                _store_immutable_text(
                    sealed_cache, source_cache.read_text(encoding="utf-8")
                )
                cache_index[role] = {
                    "path": f"sealed_source/cache/{role}/{sample_hash}.json",
                    "sha256": v5.sha256_file(sealed_cache),
                    "bytes": sealed_cache.stat().st_size,
                    "raw_content_sha256": provider.get("raw_content_sha256"),
                    "raw_reasoning_sha256": provider.get("raw_reasoning_sha256"),
                }
            records[split].append(
                {
                    "sample_id": row.sample_id,
                    "split": split,
                    "split_index": row.split_index,
                    "source_analysis_sha256": row.source_analysis_sha256,
                    "provided_data_sha256": row.provided_data_sha256,
                    "machine_pass": machine_pass,
                    "status": status,
                    "reasons": list(result.get("reasons") or []),
                    "source_terminal_path": (
                        f"sealed_source/cache/source_terminal/{sample_hash}.json"
                    ),
                    "source_terminal_sha256": v5.sha256_file(sealed_terminal),
                    "source_terminal_bytes": sealed_terminal.stat().st_size,
                    "source_result_sha256": payload["result_sha256"],
                    "provider_cache_index": cache_index,
                    "prepared_row": row.to_dict(),
                    "prepared_row_sha256": v5.sha256_text(
                        v5.canonical_json(row.to_dict())
                    ),
                    "source_audit_snapshot": result,
                }
            )

    for role, expected_paths in expected_acquisition_provider_files.items():
        role_root = acquisition / "cache" / role
        observed_paths = set(role_root.glob("*.json")) if role_root.is_dir() else set()
        if observed_paths != expected_paths:
            raise SourceHandoffError(f"source provider missing/orphan drift: {role}")

    source_identity = identity.as_dict()
    receipt_identity = receipt.get("provider_identities")
    if (
        not isinstance(receipt_identity, Mapping)
        or source_identity.get("returned_model")
        != receipt_identity.get("returned_model")
        or source_identity.get("system_fingerprint")
        != receipt_identity.get("system_fingerprint")
        or not set(source_identity.get("roles_observed", ())).issubset(
            set(receipt_identity.get("roles_observed", ()))
        )
    ):
        raise SourceHandoffError("provider identity receipt drift")
    if (
        status_counts["PASS"] != receipt.get("admitted_rows")
        or sum(status_counts.values()) != expected_total
    ):
        raise SourceHandoffError("PASS/REJECT partition drift")
    prepare_manifest = _read_json(
        acquisition / "prepare_manifest.json", "prepare manifest"
    )
    declared_prepare_sha = prepare_manifest.get("prepare_manifest_sha256")
    unsigned_prepare = dict(prepare_manifest)
    unsigned_prepare.pop("prepare_manifest_sha256", None)
    if declared_prepare_sha != v5.sha256_text(v5.canonical_json(unsigned_prepare)):
        raise SourceHandoffError("prepare manifest SHA drift")
    split_counts = {split: len(prepared[split]) for split in SPLITS}
    meeting_counts = {
        split: len({row.meeting_date for row in prepared[split]}) for split in SPLITS
    }
    ordered_sample_ids = [row.sample_id for row in all_rows]
    sample_id_digest = v5.sha256_text(v5.canonical_json(ordered_sample_ids))
    source_rows_digest = v5.sha256_text(
        v5.canonical_json([row.source_row_sha256 for row in all_rows])
    )
    meeting_sets = {
        split: {row.meeting_date for row in prepared[split]} for split in SPLITS
    }
    if any(
        meeting_sets[left] & meeting_sets[right]
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    ):
        raise SourceHandoffError("prepared meeting split overlap")
    preparation_summary = _read_json(
        acquisition / "preparation_summary.json", "preparation summary"
    )
    preparation_source = preparation_summary.get("source")
    if (
        prepare_manifest.get("split_counts") != split_counts
        or prepare_manifest.get("meeting_counts") != meeting_counts
        or prepare_manifest.get("sample_id_digest") != sample_id_digest
        or prepare_manifest.get("source_rows_digest") != source_rows_digest
        or not isinstance(preparation_source, Mapping)
        or preparation_source.get("split_counts") != split_counts
        or preparation_source.get("meeting_counts") != meeting_counts
        or preparation_source.get("sample_id_sha256") != sample_id_digest
        or preparation_source.get("rows_sha256") != source_rows_digest
        or receipt.get("source_rows") != expected_total
        or receipt.get("admitted_rows") + receipt.get("rejected_rows") != expected_total
        or receipt.get("status_counts")
        != {
            split: dict(Counter(record["status"] for record in records[split]))
            for split in SPLITS
        }
        or receipt.get("lineage") != v5.LINEAGE
    ):
        raise SourceHandoffError("prepare/source receipt lineage or count drift")
    for split in SPLITS:
        descriptor = receipt.get("artifacts", {}).get(split)
        if not isinstance(descriptor, Mapping):
            raise SourceHandoffError(f"source receipt artifact missing: {split}")
        source_rows = _read_jsonl(
            acquisition / "source_audit" / f"{split}.jsonl",
            f"source audit {split}",
        )
        if descriptor.get("rows") != len(source_rows) or descriptor.get(
            "sha256"
        ) != v5.sha256_file(acquisition / "source_audit" / f"{split}.jsonl"):
            raise SourceHandoffError(f"source receipt artifact drift: {split}")

    output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}
    for split in SPLITS:
        path = output / "records" / f"{split}.jsonl"
        _store_immutable_text(
            path,
            "".join(v5.canonical_json(row) + "\n" for row in records[split]),
        )
        descriptor = _artifact(path, len(records[split]))
        descriptor["path"] = f"records/{split}.jsonl"
        artifacts[split] = descriptor
    sealed_metadata: dict[str, Any] = {}
    for filename in (
        "prompt_contract.json",
        "official_pre_action_reference_bank.jsonl",
        "prepare_manifest.json",
        "preparation_summary.json",
        "source_admission_receipt.json",
    ):
        source_path = acquisition / filename
        target_path = output / "sealed_source" / filename
        _store_immutable_text(target_path, source_path.read_text(encoding="utf-8"))
        descriptor = _artifact(target_path)
        descriptor["path"] = f"sealed_source/{filename}"
        sealed_metadata[filename] = descriptor
    for split in SPLITS:
        filename = f"source_audit/{split}.jsonl"
        source_path = acquisition / filename
        target_path = output / "sealed_source" / filename
        _store_immutable_text(target_path, source_path.read_text(encoding="utf-8"))
        descriptor = _artifact(target_path, len(records[split]))
        descriptor["path"] = f"sealed_source/{filename}"
        sealed_metadata[filename] = descriptor
    bindings = {
        "implementation_composite_sha256": code_sha,
        "prompt_contract_sha256": v5.sha256_file(acquisition / "prompt_contract.json"),
        "official_reference_bank_sha256": v5.sha256_file(
            acquisition / "official_pre_action_reference_bank.jsonl"
        ),
        "prepare_manifest_sha256": v5.sha256_file(
            acquisition / "prepare_manifest.json"
        ),
        "preparation_summary_sha256": v5.sha256_file(
            acquisition / "preparation_summary.json"
        ),
        "source_admission_receipt_sha256": v5.sha256_file(receipt_path),
    }
    ordered_ids = [row.sample_id for row in all_rows]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "sealed",
        "source_acquisition_root": str(acquisition),
        "total_rows": expected_total,
        "split_counts": {split: len(records[split]) for split in SPLITS},
        "status_counts": dict(sorted(status_counts.items())),
        "source_provider_identities": source_identity,
        "sample_id_digest": v5.sha256_text(v5.canonical_json(ordered_ids)),
        "bindings": bindings,
        "artifacts": artifacts,
        "sealed_source_artifacts": sealed_metadata,
    }
    manifest["manifest_sha256"] = v5.sha256_text(v5.canonical_json(manifest))
    v5.legacy._store_immutable_json(output / "handoff_manifest.json", manifest)
    return load_and_verify_source_handoff(output, expected_total=expected_total)


def load_and_verify_source_handoff(
    handoff_root: str | Path,
    *,
    expected_total: int = v5.EXPECTED_TOTAL,
) -> SourceHandoff:
    root = Path(handoff_root).resolve()
    manifest = _read_json(root / "handoff_manifest.json", "handoff manifest")
    stored_sha = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "sealed"
        or stored_sha != v5.sha256_text(v5.canonical_json(unsigned))
    ):
        raise SourceHandoffError("handoff manifest drift")
    if manifest.get("total_rows") != expected_total:
        raise SourceHandoffError("handoff total drift")
    records: dict[str, tuple[Mapping[str, Any], ...]] = {}
    seen: set[str] = set()
    statuses: Counter[str] = Counter()
    for split in SPLITS:
        descriptor = manifest.get("artifacts", {}).get(split)
        if not isinstance(descriptor, Mapping):
            raise SourceHandoffError(f"missing {split} descriptor")
        path = _verify_artifact(root, descriptor, split)
        rows = _read_jsonl(path, f"handoff {split}")
        if descriptor.get("rows") != len(rows) or manifest.get("split_counts", {}).get(
            split
        ) != len(rows):
            raise SourceHandoffError(f"handoff {split} row-count drift")
        split_indices: set[int] = set()
        for row in rows:
            sid = row.get("sample_id")
            if (
                not isinstance(sid, str)
                or sid in seen
                or row.get("split") != split
                or not isinstance(row.get("split_index"), int)
            ):
                raise SourceHandoffError(f"handoff ID/split/order drift: {sid}")
            split_index = int(row["split_index"])
            if split_index in split_indices:
                raise SourceHandoffError(
                    f"duplicate split index: {split}:{split_index}"
                )
            split_indices.add(split_index)
            snapshot = row.get("source_audit_snapshot")
            if (
                not isinstance(snapshot, Mapping)
                or row.get("machine_pass") != snapshot.get("machine_pass")
                or row.get("status") != _source_status(snapshot)
            ):
                raise SourceHandoffError(f"handoff verdict drift: {sid}")
            seen.add(sid)
            statuses[str(row.get("status"))] += 1
        records[split] = tuple(rows)
    ordered_ids = [row["sample_id"] for split in SPLITS for row in records[split]]
    if (
        len(seen) != expected_total
        or manifest.get("sample_id_digest")
        != v5.sha256_text(v5.canonical_json(ordered_ids))
        or manifest.get("status_counts") != dict(sorted(statuses.items()))
    ):
        raise SourceHandoffError("handoff partition or ID digest drift")
    bindings = manifest.get("bindings")
    if not isinstance(bindings, Mapping):
        raise SourceHandoffError("handoff bindings are invalid")
    bound_files = {
        "prompt_contract_sha256": "prompt_contract.json",
        "official_reference_bank_sha256": "official_pre_action_reference_bank.jsonl",
        "prepare_manifest_sha256": "prepare_manifest.json",
        "preparation_summary_sha256": "preparation_summary.json",
        "source_admission_receipt_sha256": "source_admission_receipt.json",
    }
    sealed_root = root / "sealed_source"
    for key, filename in bound_files.items():
        sealed_descriptor = manifest.get("sealed_source_artifacts", {}).get(filename)
        if not isinstance(sealed_descriptor, Mapping):
            raise SourceHandoffError(f"sealed source descriptor missing: {filename}")
        sealed_path = _verify_artifact(root, sealed_descriptor, filename)
        if bindings.get(key) != v5.sha256_file(sealed_path):
            raise SourceHandoffError(f"sealed source binding drift: {filename}")
    prompt = _read_json(sealed_root / "prompt_contract.json", "prompt contract")
    if prompt.get("code_sha256") != bindings.get("implementation_composite_sha256"):
        raise SourceHandoffError("source code/prompt binding drift")
    if v5._implementation_contract().get("composite_sha256") != prompt.get(
        "code_sha256"
    ):
        raise SourceHandoffError("current v5 implementation drift")
    prepared: dict[str, tuple[v5.PreparedRow, ...]] = {}
    source_results: dict[str, Mapping[str, Any]] = {}
    replay_identity = v5.ProviderIdentityRegistry()
    replay_config = v5.ProviderConfig()
    if prompt != v5._prompt_contract(
        code_sha256=str(prompt["code_sha256"]), config=replay_config
    ):
        raise SourceHandoffError("complete sealed prompt contract drift")
    try:
        sealed_reference_bank = deserialize_official_reference_bank(
            (sealed_root / "official_pre_action_reference_bank.jsonl").read_bytes()
        )
    except Exception as exc:
        raise SourceHandoffError("sealed official reference replay failed") from exc
    replay_code_sha = str(bindings.get("implementation_composite_sha256"))
    expected_terminal_files: set[Path] = set()
    expected_provider_files: dict[str, set[Path]] = {
        role: set()
        for role in (
            v5.ROLE_SOURCE_AUDIT_PRIMARY,
            v5.ROLE_SOURCE_AUDIT_ADJUDICATION,
            v5.ROLE_SOURCE_AUDIT_CONTRACT_REPAIR,
        )
    }
    for split in SPLITS:
        sealed_rows: list[v5.PreparedRow] = []
        for record in records[split]:
            prepared_value = record.get("prepared_row")
            if not isinstance(prepared_value, dict) or record.get(
                "prepared_row_sha256"
            ) != v5.sha256_text(v5.canonical_json(prepared_value)):
                raise SourceHandoffError("sealed prepared-row snapshot drift")
            try:
                source_row = v5.PreparedRow(**prepared_value)
            except (TypeError, ValueError) as exc:
                raise SourceHandoffError("sealed prepared-row shape drift") from exc
            sealed_rows.append(source_row)
            if (
                source_row.sample_id != record.get("sample_id")
                or source_row.split != split
                or source_row.split_index != record.get("split_index")
                or source_row.source_analysis_sha256
                != record.get("source_analysis_sha256")
                or source_row.provided_data_sha256 != record.get("provided_data_sha256")
                or source_row.source_analysis_sha256
                != v5.sha256_text(source_row.source_analysis)
                or source_row.provided_data_sha256
                != v5.sha256_text(source_row.provided_data)
                or source_row.prompt_sha256 != v5.sha256_text(source_row.prompt)
            ):
                raise SourceHandoffError(
                    f"source prepared/record drift: {source_row.sample_id}"
                )
            sample_hash = v5.sha256_text(source_row.sample_id)
            expected_terminal_path = (
                f"sealed_source/cache/source_terminal/{sample_hash}.json"
            )
            if record.get("source_terminal_path") != expected_terminal_path:
                raise SourceHandoffError("source terminal path drift")
            terminal_path = root / expected_terminal_path
            expected_terminal_files.add(terminal_path)
            if (
                terminal_path.is_symlink()
                or not terminal_path.is_file()
                or record.get("source_terminal_bytes") != terminal_path.stat().st_size
                or record.get("source_terminal_sha256") != v5.sha256_file(terminal_path)
            ):
                raise SourceHandoffError(
                    f"source terminal artifact drift: {source_row.sample_id}"
                )
            terminal = _read_json(terminal_path, "source terminal")
            result = terminal.get("result")
            if (
                not isinstance(result, dict)
                or terminal.get("result_sha256") != record.get("source_result_sha256")
                or terminal.get("result_sha256")
                != v5.sha256_text(v5.canonical_json(result))
                or record.get("source_audit_snapshot") != result
            ):
                raise SourceHandoffError(
                    f"source terminal result drift: {source_row.sample_id}"
                )
            try:
                v5._validate_source_terminal_result(source_row, result)
                v5._resume_source_terminal_provider_caches(
                    output=sealed_root,
                    row=source_row,
                    result=result,
                    identity=replay_identity,
                    environment={},
                    config=replay_config,
                    code_sha256=replay_code_sha,
                )
            except Exception as exc:
                raise SourceHandoffError(
                    f"source terminal replay drift: {source_row.sample_id}: {exc}"
                ) from exc
            if result.get("machine_pass") != record.get("machine_pass") or list(
                result.get("reasons") or []
            ) != record.get("reasons"):
                raise SourceHandoffError(
                    f"source normalized verdict drift: {source_row.sample_id}"
                )
            source_results[source_row.sample_id] = result
            cache_index = record.get("provider_cache_index")
            if not isinstance(cache_index, Mapping):
                raise SourceHandoffError(
                    f"source provider index drift: {source_row.sample_id}"
                )
            providers = result.get("provider")
            if not isinstance(providers, Mapping) or set(cache_index) != set(providers):
                raise SourceHandoffError(
                    f"source provider role-set drift: {source_row.sample_id}"
                )
            for role, descriptor in cache_index.items():
                if role not in {
                    v5.ROLE_SOURCE_AUDIT_PRIMARY,
                    v5.ROLE_SOURCE_AUDIT_ADJUDICATION,
                    v5.ROLE_SOURCE_AUDIT_CONTRACT_REPAIR,
                } or not isinstance(descriptor, Mapping):
                    raise SourceHandoffError("source provider descriptor drift")
                expected_cache_path = f"sealed_source/cache/{role}/{sample_hash}.json"
                if descriptor.get("path") != expected_cache_path:
                    raise SourceHandoffError("source provider path drift")
                cache_path = root / expected_cache_path
                expected_provider_files[role].add(cache_path)
                cache = _read_json(cache_path, "source provider cache")
                if (
                    descriptor.get("bytes") != cache_path.stat().st_size
                    or descriptor.get("sha256") != v5.sha256_file(cache_path)
                    or cache.get("raw_content_sha256")
                    != descriptor.get("raw_content_sha256")
                    or cache.get("raw_reasoning_sha256")
                    != descriptor.get("raw_reasoning_sha256")
                ):
                    raise SourceHandoffError(
                        f"source provider raw hash drift: {source_row.sample_id}:{role}"
                    )
        prepared[split] = tuple(sealed_rows)
    observed_terminal_files = set(
        (sealed_root / "cache/source_terminal").glob("*.json")
    )
    if observed_terminal_files != expected_terminal_files:
        raise SourceHandoffError("sealed source terminal missing/orphan drift")
    for role, expected_paths in expected_provider_files.items():
        role_root = sealed_root / "cache" / role
        observed_paths = set(role_root.glob("*.json")) if role_root.is_dir() else set()
        if observed_paths != expected_paths:
            raise SourceHandoffError(f"sealed provider missing/orphan drift: {role}")
    sealed_all_rows = [row for split in SPLITS for row in prepared[split]]
    sealed_prepare = _read_json(
        sealed_root / "prepare_manifest.json", "prepare manifest"
    )
    sealed_prepare_sha = sealed_prepare.get("prepare_manifest_sha256")
    unsigned_sealed_prepare = dict(sealed_prepare)
    unsigned_sealed_prepare.pop("prepare_manifest_sha256", None)
    sealed_id_digest = v5.sha256_text(
        v5.canonical_json([row.sample_id for row in sealed_all_rows])
    )
    sealed_rows_digest = v5.sha256_text(
        v5.canonical_json([row.source_row_sha256 for row in sealed_all_rows])
    )
    if (
        sealed_prepare_sha != v5.sha256_text(v5.canonical_json(unsigned_sealed_prepare))
        or sealed_prepare.get("sample_id_digest") != sealed_id_digest
        or sealed_prepare.get("source_rows_digest") != sealed_rows_digest
    ):
        raise SourceHandoffError("sealed prepare manifest replay drift")
    del sealed_reference_bank  # deserialization already verifies embedded hashes
    receipt = _read_json(
        sealed_root / "source_admission_receipt.json", "source receipt"
    )
    expected_status_by_split: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        descriptor = manifest.get("sealed_source_artifacts", {}).get(
            f"source_audit/{split}.jsonl"
        )
        if not isinstance(descriptor, Mapping):
            raise SourceHandoffError(f"sealed source-audit descriptor missing: {split}")
        audit_path = _verify_artifact(root, descriptor, f"source audit {split}")
        audit_rows = _read_jsonl(audit_path, f"source audit {split}")
        expected_audit_rows = [
            {
                "sample_id": record["sample_id"],
                "split": split,
                **dict(record["source_audit_snapshot"]),
            }
            for record in records[split]
        ]
        if audit_rows != expected_audit_rows:
            raise SourceHandoffError(f"sealed source-audit snapshot drift: {split}")
        receipt_artifact = receipt.get("artifacts", {}).get(split)
        if (
            not isinstance(receipt_artifact, Mapping)
            or receipt_artifact.get("rows") != len(audit_rows)
            or receipt_artifact.get("bytes") != audit_path.stat().st_size
            or receipt_artifact.get("sha256") != v5.sha256_file(audit_path)
        ):
            raise SourceHandoffError(f"source receipt artifact replay drift: {split}")
        expected_status_by_split[split] = dict(
            Counter(record["status"] for record in records[split])
        )
    if (
        receipt.get("schema_version") != v5.SOURCE_AUDIT_SCHEMA_VERSION
        or receipt.get("status") != "source_admission_complete"
        or receipt.get("quality_status") != "passed"
        or receipt.get("source_rows") != expected_total
        or receipt.get("admitted_rows") != statuses.get("PASS", 0)
        or receipt.get("rejected_rows") != expected_total - statuses.get("PASS", 0)
        or receipt.get("unresolved_rows") != 0
        or receipt.get("status_counts") != expected_status_by_split
        or receipt.get("lineage") != v5.LINEAGE
    ):
        raise SourceHandoffError("source receipt schema/count/lineage drift")
    replay_identity_value = replay_identity.as_dict()
    receipt_identity = receipt.get("provider_identities")
    if (
        not isinstance(receipt_identity, Mapping)
        or replay_identity_value.get("returned_model")
        != receipt_identity.get("returned_model")
        or replay_identity_value.get("system_fingerprint")
        != receipt_identity.get("system_fingerprint")
        or not set(replay_identity_value.get("roles_observed", ())).issubset(
            set(receipt_identity.get("roles_observed", ()))
        )
    ):
        raise SourceHandoffError("source provider identity replay drift")
    if replay_identity_value != manifest.get("source_provider_identities"):
        raise SourceHandoffError("sealed source provider identity drift")
    return SourceHandoff(
        root=root,
        manifest=manifest,
        records=records,
        prepared=prepared,
        source_results=source_results,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--acquisition-root", type=Path, default=DEFAULT_ACQUISITION_ROOT
    )
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    handoff = (
        load_and_verify_source_handoff(args.handoff_root)
        if args.verify_only
        else seal_source_handoff(args.acquisition_root, args.handoff_root)
    )
    print(json.dumps(dict(handoff.manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
