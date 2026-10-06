"""Verify and publish the two-stage paper chk-2 v6 downstream release.

This publisher is deliberately API-free.  It first replays the sealed v5
source-admission handoff against the immutable source acquisition, then replays
the v6 downstream acquisition before atomically publishing PASS rows only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jobs.generation import generate_paper_chk2_chk1_analysis_rewrite as v5
from jobs.generation import publish_paper_chk2_chk1_analysis_rewrite as v5_publisher
from jobs.generation import seal_paper_chk2_source_handoff_v1 as source_sealer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ACQUISITION_ROOT = source_sealer.DEFAULT_ACQUISITION_ROOT
DEFAULT_SOURCE_HANDOFF_ROOT = source_sealer.DEFAULT_HANDOFF_ROOT
DEFAULT_DOWNSTREAM_ROOT = REPO_ROOT / (
    "output/data/retrain_v2/chk2/"
    "chk1_final_analysis_to_minutes_flash_official_reference_"
    "v6_downstream128_20260831"
)
DEFAULT_RELEASE_ROOT = REPO_ROOT / (
    "dataset/processed/retrain_v2/"
    "chk2_chk1_final_analysis_synthetic_minutes_flash_official_reference_"
    "v6_downstream128_20260831"
)
DEFAULT_TOKENIZER_PATH = v5_publisher.DEFAULT_TOKENIZER_PATH

SPLITS = ("train", "validation", "test")
RELEASE_SCHEMA_VERSION = "paper-chk2-downstream-release-v6"
RELEASE_HANDOFF_SCHEMA_VERSION = "paper-chk2-downstream-release-handoff-v6"
EXECUTION_RECEIPT_SCHEMA_VERSION = "paper-chk2-downstream-execution-v1"
DATASET_ROLE = "paper_chk2_chk1_final_analysis_to_synthetic_minutes_sft_v6"
TRAINING_SCOPE = "paper-chk2-chk1-cp200-minutes-sft-v6-downstream128"
PUBLISHER_IMPLEMENTATION_DEPENDENCIES = {
    "downstream_v6_publisher": Path(__file__).resolve(),
    "token_replay_publisher": Path(v5_publisher.__file__).resolve(),
}

EXECUTION_RECEIPT_REQUIRED_FIELDS = {
    "schema_version",
    "status",
    "phase",
    "configured_concurrency",
    "maximum_concurrency",
    "runner_sha256",
    "implementation_composite_sha256",
    "source_handoff_manifest_sha256",
    "source_handoff_manifest_file_sha256",
    "run_binding_sha256",
    "prompt_contract_sha256",
    "official_reference_bank_sha256",
    "tokenizer_contract_sha256",
    "provider_identities",
    "source_audit_provider_calls",
    "cache_counts",
    "terminal_cache_count",
    "artifacts",
    "receipt_sha256",
}


class PublicationError(RuntimeError):
    """Raised when either acquisition stage or release bytes drift."""


@dataclass(frozen=True)
class VerifiedSourceHandoff:
    handoff: source_sealer.SourceHandoff
    manifest_file_sha256: str
    signed_manifest_sha256: str
    source_acquisition_root: Path


@dataclass(frozen=True)
class VerifiedDownstream:
    root: Path
    receipt: Mapping[str, Any]
    receipt_path: Path
    final_summary: Mapping[str, Any]
    terminals: Mapping[str, tuple[Mapping[str, Any], ...]]
    pass_rows: Mapping[str, tuple[Mapping[str, str], ...]]
    pass_manifests: Mapping[str, tuple[Mapping[str, Any], ...]]
    tokenizer_replay: tuple[Mapping[str, Any], ...]
    provider_identities: Mapping[str, Any]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return v5.sha256_text(value)


def sha256_file(path: Path) -> str:
    return v5.sha256_file(path)


def _publisher_implementation_contract() -> dict[str, Any]:
    artifacts: dict[str, dict[str, str]] = {}
    for label, path in sorted(PUBLISHER_IMPLEMENTATION_DEPENDENCIES.items()):
        _require(
            path.is_file() and not path.is_symlink(),
            f"publisher implementation dependency is missing or unsafe: {label}",
        )
        try:
            relative = path.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise PublicationError(
                f"publisher implementation dependency is outside repository: {label}"
            ) from exc
        artifacts[label] = {
            "path": str(relative),
            "sha256": sha256_file(path),
        }
    return {
        "artifacts": artifacts,
        "composite_sha256": sha256_text(canonical_json(artifacts)),
    }


def _validate_publisher_implementation_binding(
    manifest: Mapping[str, Any], handoff: Mapping[str, Any]
) -> None:
    expected = _publisher_implementation_contract()
    _require(
        manifest.get("publisher_implementation") == expected,
        "release publisher implementation drift",
    )
    _require(
        handoff.get("publisher_implementation") == expected,
        "release handoff publisher implementation drift",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicationError(message)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationError(f"invalid {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} must be an object: {path}")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _require(path.is_file(), f"missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PublicationError(
                f"invalid {label} line {line_number}: {path}"
            ) from exc
        _require(
            isinstance(value, dict),
            f"non-object {label} line {line_number}: {path}",
        )
        rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_artifact_path(
    root: Path, descriptor: Mapping[str, Any], *, label: str
) -> Path:
    relative = descriptor.get("path")
    _require(isinstance(relative, str) and relative, f"invalid {label} path")
    relative_path = Path(relative)
    _require(
        not relative_path.is_absolute() and ".." not in relative_path.parts,
        f"unsafe {label} path",
    )
    path = root / relative_path
    _validate_strict_path(root, path, label=label, require_file=True)
    _require(descriptor.get("bytes") == path.stat().st_size, f"{label} byte drift")
    _require(descriptor.get("sha256") == sha256_file(path), f"{label} SHA drift")
    return path


def _validate_strict_path(
    root: Path,
    path: Path,
    *,
    label: str,
    require_file: bool = False,
    require_dir: bool = False,
) -> Path:
    """Reject lexical escape and every symlink in a root-relative path."""

    root_abs = root.absolute()
    path_abs = path.absolute()
    _require(root_abs.exists(), f"unsafe {label} root")
    root_cursor = Path(root_abs.anchor)
    for part in root_abs.parts[1:]:
        root_cursor = root_cursor / part
        _require(
            not root_cursor.is_symlink(),
            f"symlink forbidden in {label} root: {root_cursor}",
        )
    try:
        relative = path_abs.relative_to(root_abs)
    except ValueError as exc:
        raise PublicationError(f"{label} escapes authorized root") from exc
    current = root_abs
    for part in relative.parts:
        _require(part not in {"", ".", ".."}, f"unsafe {label} path component")
        current = current / part
        _require(not current.is_symlink(), f"symlink forbidden for {label}: {current}")
    if require_file:
        _require(path_abs.is_file(), f"missing {label}: {path_abs}")
    if require_dir:
        _require(path_abs.is_dir(), f"missing {label} directory: {path_abs}")
    return path_abs


def _strict_cache_inventory(root: Path, *, allowed_roles: set[str]) -> dict[str, set[Path]]:
    cache_root = root / "cache"
    _validate_strict_path(root, cache_root, label="cache root", require_dir=True)
    inventory: dict[str, set[Path]] = {role: set() for role in allowed_roles}
    for directory, dirnames, filenames in os.walk(cache_root, followlinks=False):
        directory_path = Path(directory)
        _validate_strict_path(root, directory_path, label="cache directory", require_dir=True)
        relative_directory = directory_path.relative_to(cache_root)
        for dirname in dirnames:
            child = directory_path / dirname
            _require(not child.is_symlink(), f"symlink forbidden in cache: {child}")
        for filename in filenames:
            path = directory_path / filename
            _validate_strict_path(root, path, label="cache file", require_file=True)
            _require(
                len(relative_directory.parts) == 1
                and relative_directory.parts[0] in allowed_roles,
                f"nested or unknown cache namespace: {path}",
            )
            _require(path.suffix == ".json", f"non-JSON cache artifact: {path}")
            role = relative_directory.parts[0]
            inventory[role].add(path.absolute())
    return inventory


def _strict_tree_files(root: Path, *, label: str) -> set[Path]:
    _validate_strict_path(root.parent, root, label=label, require_dir=True)
    files: set[Path] = set()
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        _validate_strict_path(root, directory_path, label=label, require_dir=True)
        for dirname in dirnames:
            _require(
                not (directory_path / dirname).is_symlink(),
                f"symlink forbidden in {label}: {directory_path / dirname}",
            )
        for filename in filenames:
            path = _validate_strict_path(
                root,
                directory_path / filename,
                label=label,
                require_file=True,
            )
            files.add(path)
    return files


def _verify_source_handoff(
    *,
    source_acquisition_root: Path,
    source_handoff_root: Path,
) -> VerifiedSourceHandoff:
    _require(".." not in source_acquisition_root.parts, "unsafe source acquisition path")
    _require(".." not in source_handoff_root.parts, "unsafe source handoff path")
    acquisition = source_acquisition_root.absolute()
    handoff_root = source_handoff_root.absolute()
    _validate_strict_path(
        acquisition.parent, acquisition, label="source acquisition", require_dir=True
    )
    _validate_strict_path(
        handoff_root.parent, handoff_root, label="source handoff", require_dir=True
    )
    _strict_tree_files(handoff_root, label="source handoff")
    try:
        supplied = source_sealer.load_and_verify_source_handoff(handoff_root)
    except Exception as exc:
        raise PublicationError(f"source handoff verification failed: {exc}") from exc

    manifest = dict(supplied.manifest)
    _require(
        Path(str(manifest.get("source_acquisition_root", ""))).resolve() == acquisition,
        "source handoff acquisition-root drift",
    )
    prompt_path = acquisition / "prompt_contract.json"
    prompt_contract = _read_json(prompt_path, label="old prompt contract")
    implementation = v5._implementation_contract()
    current_code_sha = implementation.get("composite_sha256")
    _require(
        isinstance(current_code_sha, str)
        and prompt_contract.get("code_sha256") == current_code_sha,
        "old implementation/prompt code SHA drift",
    )
    expected_prompt = v5._prompt_contract(
        code_sha256=current_code_sha,
        config=v5.ProviderConfig(),
    )
    _require(prompt_contract == expected_prompt, "old prompt contract replay drift")

    binding_paths = {
        "prompt_contract_sha256": prompt_path,
        "official_reference_bank_sha256": (
            acquisition / "official_pre_action_reference_bank.jsonl"
        ),
        "prepare_manifest_sha256": acquisition / "prepare_manifest.json",
        "preparation_summary_sha256": acquisition / "preparation_summary.json",
        "source_admission_receipt_sha256": (
            acquisition / "source_admission_receipt.json"
        ),
    }
    bindings = manifest.get("bindings")
    _require(isinstance(bindings, Mapping), "source handoff bindings missing")
    _require(
        bindings.get("implementation_composite_sha256") == current_code_sha,
        "source handoff implementation binding drift",
    )
    for field, path in binding_paths.items():
        _require(path.is_file(), f"missing old source artifact: {path}")
        _require(bindings.get(field) == sha256_file(path), f"{field} drift")

    # Re-sealing into a private temporary directory independently replays every
    # prepared row, source terminal, provider cache, identity and receipt.  The
    # supplied handoff must be byte-semantically identical to that replay.
    with tempfile.TemporaryDirectory(prefix="paper_chk2_source_handoff_replay_") as tmp:
        replay_root = Path(tmp) / "handoff"
        try:
            replayed = source_sealer.seal_source_handoff(
                acquisition,
                replay_root,
                expected_total=v5.EXPECTED_TOTAL,
            )
        except Exception as exc:
            raise PublicationError(f"source acquisition replay failed: {exc}") from exc
        _require(
            canonical_json(dict(replayed.manifest)) == canonical_json(manifest),
            "source handoff manifest replay drift",
        )
        for split in SPLITS:
            _require(
                tuple(replayed.records[split]) == tuple(supplied.records[split]),
                f"source handoff {split} records replay drift",
            )

    manifest_path = handoff_root / "handoff_manifest.json"
    signed_sha = manifest.get("manifest_sha256")
    _require(
        isinstance(signed_sha, str) and len(signed_sha) == 64,
        "source handoff signed manifest SHA invalid",
    )
    return VerifiedSourceHandoff(
        handoff=supplied,
        manifest_file_sha256=sha256_file(manifest_path),
        signed_manifest_sha256=signed_sha,
        source_acquisition_root=acquisition,
    )


def _validate_execution_receipt_shape(
    receipt: Mapping[str, Any], *, expected_phase: str | None = None
) -> None:
    _require(
        set(receipt) == EXECUTION_RECEIPT_REQUIRED_FIELDS,
        "v6 execution receipt field drift",
    )
    _require(receipt.get("status") == "complete", "v6 execution is incomplete")
    _require(
        receipt.get("schema_version") == EXECUTION_RECEIPT_SCHEMA_VERSION,
        "v6 execution receipt schema drift",
    )
    _require(
        receipt.get("phase") in {"all", "verify"},
        "v6 execution phase drift",
    )
    if expected_phase is not None:
        _require(
            receipt.get("phase") == expected_phase,
            "v6 execution receipt filename/phase drift",
        )
    _require(
        receipt.get("configured_concurrency") == 128,
        "v6 configured concurrency drift",
    )
    _require(receipt.get("maximum_concurrency") == 128, "v6 maximum concurrency drift")
    _require(
        receipt.get("source_audit_provider_calls") == 0,
        "v6 source-audit provider isolation drift",
    )
    for field in (
        "runner_sha256",
        "implementation_composite_sha256",
        "source_handoff_manifest_sha256",
        "source_handoff_manifest_file_sha256",
        "run_binding_sha256",
        "prompt_contract_sha256",
        "official_reference_bank_sha256",
        "tokenizer_contract_sha256",
    ):
        value = receipt.get(field)
        _require(
            isinstance(value, str) and len(value) == 64,
            f"v6 execution receipt {field} invalid",
        )
    unsigned = dict(receipt)
    stored_sha = unsigned.pop("receipt_sha256", None)
    _require(
        stored_sha == sha256_text(canonical_json(unsigned)),
        "v6 execution receipt self-SHA drift",
    )
    for field in ("provider_identities", "cache_counts", "artifacts"):
        _require(isinstance(receipt.get(field), Mapping), f"v6 receipt {field} invalid")
    _require(
        isinstance(receipt.get("terminal_cache_count"), int)
        and receipt.get("terminal_cache_count") >= 0,
        "v6 receipt terminal count invalid",
    )


def _downstream_module() -> Any:
    try:
        from jobs.generation import generate_paper_chk2_downstream_v6 as downstream
    except ImportError as exc:  # pragma: no cover - deployment ordering guard
        raise PublicationError("v6 downstream producer module is unavailable") from exc
    return downstream


def _resolve_downstream_artifact(root: Path, descriptor: Any, *, label: str) -> Path:
    _require(isinstance(descriptor, Mapping), f"{label} descriptor must be an object")
    relative_value = descriptor.get("path")
    _require(isinstance(relative_value, str) and relative_value, f"{label} path invalid")
    relative = Path(relative_value)
    _require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and all(part not in {"", "."} for part in relative.parts),
        f"unsafe {label} descriptor path",
    )
    root_abs = root.absolute()
    candidates: list[Path] = []
    for candidate in (root_abs / relative, REPO_ROOT.absolute() / relative):
        try:
            candidate.absolute().relative_to(root_abs)
        except ValueError:
            continue
        if candidate.exists():
            candidates.append(candidate)
    unique = list(dict.fromkeys(path.absolute() for path in candidates))
    _require(len(unique) == 1, f"{label} path must resolve uniquely inside root")
    path = _validate_strict_path(root_abs, unique[0], label=label, require_file=True)
    _require(descriptor.get("bytes") == path.stat().st_size, f"{label} byte drift")
    _require(descriptor.get("sha256") == sha256_file(path), f"{label} SHA drift")
    if "rows" in descriptor:
        rows = len(path.read_text(encoding="utf-8").splitlines())
        _require(descriptor.get("rows") == rows, f"{label} row-count drift")
    return path


def _verify_descriptor_tree(
    root: Path,
    value: Any,
    *,
    label: str,
    observed: dict[Path, Mapping[str, Any]],
) -> None:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    if {"path", "sha256", "bytes"} <= set(value):
        path = _resolve_downstream_artifact(root, value, label=label)
        prior = observed.get(path)
        _require(
            prior is None or dict(prior) == dict(value),
            f"conflicting descriptor for {path}",
        )
        observed[path] = value
        return
    for key, nested in value.items():
        _verify_descriptor_tree(
            root,
            nested,
            label=f"{label}.{key}",
            observed=observed,
        )


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


def _verify_downstream(
    *,
    root: Path,
    source: VerifiedSourceHandoff,
    tokenizer: Any,
) -> VerifiedDownstream:
    downstream = _downstream_module()
    _require(".." not in root.parts, "unsafe v6 downstream path")
    downstream_root = root.absolute()
    _validate_strict_path(
        downstream_root.parent,
        downstream_root,
        label="v6 downstream root",
        require_dir=True,
    )
    _require(
        downstream_root.is_dir() and not downstream_root.is_symlink(),
        f"invalid v6 downstream root: {downstream_root}",
    )
    cache_inventory = _strict_cache_inventory(
        downstream_root,
        allowed_roles={*v5.PROVIDER_ROLES, "terminal"},
    )
    all_receipt_path = downstream_root / "execution_receipts" / "all.json"
    verify_receipt_path = downstream_root / "execution_receipts" / "verify.json"
    receipt_path = (
        all_receipt_path if all_receipt_path.is_file() else verify_receipt_path
    )
    _require(
        receipt_path.is_file(),
        "v6 has no canonical complete all/verify execution receipt",
    )
    phase = receipt_path.stem
    try:
        downstream.load_and_verify_downstream(
            downstream_root,
            source_handoff=source.handoff,
            tokenizer=tokenizer,
            phase=phase,
        )
    except Exception as exc:
        raise PublicationError(f"v6 downstream replay failed: {exc}") from exc

    receipt = _read_json(receipt_path, label=f"v6 {phase}-phase execution receipt")
    _validate_execution_receipt_shape(receipt, expected_phase=phase)
    implementation = downstream._implementation_contract()
    implementation_sha = implementation.get("composite_sha256")
    run_binding_sha = downstream._run_binding_sha(
        str(implementation_sha), source.signed_manifest_sha256
    )
    expected_prompt = downstream._prompt_contract(
        implementation=implementation,
        handoff_sha256=source.signed_manifest_sha256,
        run_binding_sha256=run_binding_sha,
        config=downstream.ProviderConfig(),
    )
    prompt_path = downstream_root / "prompt_contract.json"
    prompt = _read_json(prompt_path, label="v6 prompt contract")
    _require(prompt == expected_prompt, "v6 prompt contract recomputation drift")
    reference_path = downstream_root / "official_pre_action_reference_bank.jsonl"
    sealed_reference = (
        source.handoff.root
        / "sealed_source"
        / "official_pre_action_reference_bank.jsonl"
    )
    _require(
        reference_path.read_bytes() == sealed_reference.read_bytes(),
        "v6 official reference differs from source handoff",
    )
    expected_tokenizer_sha = sha256_text(
        canonical_json(v5._expected_tokenizer_runtime_contract())
    )
    _require(
        receipt.get("runner_sha256")
        == implementation["artifacts"]["downstream_v6"]["sha256"]
        and receipt.get("implementation_composite_sha256") == implementation_sha
        and receipt.get("source_handoff_manifest_sha256")
        == source.signed_manifest_sha256
        and receipt.get("source_handoff_manifest_file_sha256")
        == source.manifest_file_sha256
        and receipt.get("run_binding_sha256") == run_binding_sha
        and receipt.get("prompt_contract_sha256") == sha256_file(prompt_path)
        and receipt.get("official_reference_bank_sha256") == sha256_file(reference_path)
        and receipt.get("tokenizer_contract_sha256") == expected_tokenizer_sha,
        "v6 execution receipt provenance drift",
    )

    final_path = downstream_root / "final_summary.json"
    final = _read_json(final_path, label="v6 final summary")
    _require(
        final.get("schema_version") == downstream.SCHEMA_VERSION
        and final.get("status") == "complete"
        and final.get("quality_status") == "passed"
        and final.get("total_source_rows") == v5.EXPECTED_TOTAL
        and final.get("terminal_classified") == v5.EXPECTED_TOTAL
        and final.get("unresolved_failure_count") == 0
        and final.get("source_handoff_manifest_sha256") == source.signed_manifest_sha256
        and final.get("implementation_composite_sha256") == implementation_sha
        and final.get("run_binding_sha256") == run_binding_sha
        and final.get("prompt_contract_sha256") == sha256_file(prompt_path)
        and final.get("official_reference_bank_sha256") == sha256_file(reference_path),
        "v6 final-summary provenance/status drift",
    )
    described: dict[Path, Mapping[str, Any]] = {}
    _verify_descriptor_tree(
        downstream_root,
        receipt.get("artifacts"),
        label="execution_receipt.artifacts",
        observed=described,
    )
    _require(
        final_path.resolve() in described, "v6 receipt does not bind final summary"
    )

    identity = downstream.ProviderIdentityRegistry()
    reference_bank = downstream.official_v2.deserialize_official_reference_bank(
        reference_path.read_bytes()
    )
    replay_config = downstream.ProviderConfig()
    terminals_by_split: dict[str, tuple[Mapping[str, Any], ...]] = {}
    pass_rows_by_split: dict[str, tuple[Mapping[str, str], ...]] = {}
    pass_manifests_by_split: dict[str, tuple[Mapping[str, Any], ...]] = {}
    expected_provider_files: dict[str, set[Path]] = {
        role: set() for role in v5.PROVIDER_ROLES
    }
    expected_terminal_files: set[Path] = set()
    tokenizer_rows: list[Mapping[str, Any]] = []
    status_counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        split_descriptors = final.get("artifacts", {}).get(split)
        _require(isinstance(split_descriptors, Mapping), f"missing {split} artifacts")
        terminal_path = _resolve_downstream_artifact(
            downstream_root,
            split_descriptors.get("terminal"),
            label=f"final.artifacts.{split}.terminal",
        )
        sft_path = _resolve_downstream_artifact(
            downstream_root,
            split_descriptors.get("sft_candidate"),
            label=f"final.artifacts.{split}.sft_candidate",
        )
        manifest_path = _resolve_downstream_artifact(
            downstream_root,
            split_descriptors.get("manifest"),
            label=f"final.artifacts.{split}.manifest",
        )
        terminal_rows = _read_jsonl(terminal_path, label=f"v6 {split} terminals")
        source_rows = source.handoff.prepared[split]
        _require(
            len(terminal_rows) == len(source_rows),
            f"v6 {split} terminal population drift",
        )
        expected_sft: list[dict[str, str]] = []
        expected_manifests: list[dict[str, Any]] = []
        split_statuses: Counter[str] = Counter()
        for row, terminal in zip(source_rows, terminal_rows):
            label = f"v6 terminal {split}:{row.split_index}"
            _require(terminal.get("sample_id") == row.sample_id, f"{label} order drift")
            cache_path = v5._terminal_path(downstream_root, row)
            expected_terminal_files.add(cache_path.absolute())
            cache = _read_json(cache_path, label=f"{label} cache")
            _require(
                cache.get("record") == terminal
                and cache.get("record_sha256") == sha256_text(canonical_json(terminal)),
                f"{label} materialized/cache drift",
            )
            try:
                replayed = downstream._load_downstream_terminal(
                    downstream_root,
                    row,
                    code_sha256=str(run_binding_sha),
                    official_reference_bank_sha256=sha256_file(reference_path),
                    identity=identity,
                    environment={},
                    source_audit=source.handoff.source_results[row.sample_id],
                    reference_bank=reference_bank,
                    config=replay_config,
                    tokenizer=tokenizer,
                )
            except Exception as exc:
                raise PublicationError(f"{label} cache replay failed: {exc}") from exc
            _require(replayed == terminal, f"{label} normalized replay drift")
            for role in v5._terminal_provider_sidecars(row, terminal):
                if not role.startswith("source_audit"):
                    expected_provider_files[role].add(
                        v5._cache_path(downstream_root, role, row)
                    )
            status = str(terminal.get("terminal_status"))
            split_statuses[status] += 1
            if status != v5.TERMINAL_PASS:
                continue
            training_row = {
                "prompt": terminal["student_prompt"],
                "response": terminal["sft_response"],
            }
            _require(
                set(training_row) == {"prompt", "response"}, f"{label} schema drift"
            )
            expected_sft.append(training_row)
            expected_manifests.append(_expected_pass_manifest(terminal))
            replay = v5_publisher._token_replay(
                tokenizer=tokenizer,
                system_prompt=v5.STUDENT_SYSTEM_PROMPT,
                prompt=training_row["prompt"],
                response=training_row["response"],
                sample_id=row.sample_id,
            )
            tokenizer_rows.append(
                {"sample_id": row.sample_id, "split": split, **replay}
            )
        observed_sft = _read_jsonl(sft_path, label=f"v6 {split} SFT candidates")
        observed_manifests = _read_jsonl(manifest_path, label=f"v6 {split} manifests")
        _require(observed_sft == expected_sft, f"v6 {split} SFT projection drift")
        _require(
            observed_manifests == expected_manifests,
            f"v6 {split} manifest projection drift",
        )
        _require(expected_sft, f"v6 {split} has no PASS training rows")
        terminals_by_split[split] = tuple(terminal_rows)
        pass_rows_by_split[split] = tuple(expected_sft)
        pass_manifests_by_split[split] = tuple(expected_manifests)
        status_counts[split] = dict(split_statuses)

    for source_role in downstream.SOURCE_PROVIDER_ROLES:
        _require(
            not cache_inventory[source_role],
            f"v6 contains forbidden source-role cache: {source_role}",
        )
    _require(
        cache_inventory["terminal"] == expected_terminal_files,
        "v6 orphan/missing terminal cache",
    )
    for role in v5.PROVIDER_ROLES:
        if role in downstream.SOURCE_PROVIDER_ROLES:
            continue
        observed = cache_inventory[role]
        _require(
            observed == expected_provider_files[role],
            f"v6 orphan/missing provider cache: {role}",
        )
    _require(
        final.get("status_counts") == status_counts,
        "v6 final status-count drift",
    )
    _require(
        receipt.get("terminal_cache_count") == v5.EXPECTED_TOTAL,
        "v6 receipt terminal-cache population drift",
    )
    actual_cache_counts = {
        role: len(cache_inventory[role])
        for role in (*v5.PROVIDER_ROLES, "terminal")
        if cache_inventory[role]
        or (downstream_root / "cache" / role).is_dir()
    }
    _require(receipt.get("cache_counts") == actual_cache_counts, "v6 cache-count drift")
    _require(
        receipt.get("provider_identities") == identity.as_dict()
        and final.get("provider_identities") == identity.as_dict(),
        "v6 provider identity drift",
    )

    tokenizer_descriptor = final.get("artifacts", {}).get("tokenizer_replay")
    tokenizer_path = _resolve_downstream_artifact(
        downstream_root,
        tokenizer_descriptor,
        label="final.artifacts.tokenizer_replay",
    )
    producer_tokenizer_rows = _read_jsonl(tokenizer_path, label="v6 tokenizer replay")
    _require(
        len(producer_tokenizer_rows) == len(tokenizer_rows)
        and [row["sample_id"] for row in producer_tokenizer_rows]
        == [row["sample_id"] for row in tokenizer_rows],
        "v6 tokenizer replay population/order drift",
    )
    for producer_row, replayed_row in zip(producer_tokenizer_rows, tokenizer_rows):
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            _require(
                producer_row.get(field) == replayed_row.get(field),
                f"v6 tokenizer replay count drift: {producer_row.get('sample_id')}:{field}",
            )
    _require(
        final.get("tokenizer_replay_pass_rows") == len(tokenizer_rows),
        "v6 tokenizer replay summary drift",
    )
    return VerifiedDownstream(
        root=downstream_root,
        receipt=receipt,
        receipt_path=receipt_path,
        final_summary=final,
        terminals=terminals_by_split,
        pass_rows=pass_rows_by_split,
        pass_manifests=pass_manifests_by_split,
        tokenizer_replay=tuple(tokenizer_rows),
        provider_identities=identity.as_dict(),
    )


def _release_descriptor(path: Path, *, root: Path) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".jsonl":
        descriptor["rows"] = len(path.read_text(encoding="utf-8").splitlines())
    return descriptor


def _atomic_publish(staging: Path, release: Path) -> None:
    _require(not release.exists(), f"release already exists: {release}")
    os_replace = getattr(os, "replace", None)
    _require(callable(os_replace), "atomic replace is unavailable")
    os_replace(staging, release)


def publish_release(
    *,
    source_acquisition_root: Path = DEFAULT_SOURCE_ACQUISITION_ROOT,
    source_handoff_root: Path = DEFAULT_SOURCE_HANDOFF_ROOT,
    downstream_root: Path = DEFAULT_DOWNSTREAM_ROOT,
    release_root: Path = DEFAULT_RELEASE_ROOT,
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Replay both stages and atomically publish the PASS-only release."""

    _require(".." not in release_root.parts, "unsafe release path")
    release = release_root.absolute()
    if release.exists():
        return verify_release(
            release, tokenizer_path=tokenizer_path, tokenizer=tokenizer
        )
    source = _verify_source_handoff(
        source_acquisition_root=source_acquisition_root,
        source_handoff_root=source_handoff_root,
    )
    _require(
        tokenizer_path.resolve() == DEFAULT_TOKENIZER_PATH.resolve(),
        "tokenizer path is not the exact authorized chk1 cp200 tokenizer",
    )
    try:
        v5_publisher._validate_exact_tokenizer_files(
            tokenizer_path,
            expected_tokenizer_path=DEFAULT_TOKENIZER_PATH,
        )
    except Exception as exc:
        raise PublicationError(f"tokenizer file contract drift: {exc}") from exc
    if tokenizer is None:
        tokenizer = v5._load_tokenizer(tokenizer_path)
    try:
        v5._verify_tokenizer_runtime_contract(tokenizer)
    except Exception as exc:
        raise PublicationError(f"tokenizer runtime contract drift: {exc}") from exc
    downstream = _verify_downstream(
        root=downstream_root, source=source, tokenizer=tokenizer
    )
    _require(not release.is_symlink(), f"unsafe release path: {release}")
    release.parent.mkdir(parents=True, exist_ok=True)
    _validate_strict_path(
        release.parent,
        release.parent,
        label="release parent",
        require_dir=True,
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{release.name}.", dir=release.parent))
    try:
        artifacts: dict[str, Any] = {
            "minutes_alignment": {},
            "provenance": {},
            "audits": {},
        }
        split_counts: dict[str, int] = {}
        status_counts: dict[str, dict[str, int]] = {}
        for split in SPLITS:
            rows = [dict(row) for row in downstream.pass_rows[split]]
            manifests: list[dict[str, Any]] = []
            for release_index, (row, sidecar) in enumerate(
                zip(rows, downstream.pass_manifests[split])
            ):
                _require(
                    set(row) == {"prompt", "response"}, f"{split} training schema drift"
                )
                manifests.append(
                    {
                        **dict(sidecar),
                        "release_index": release_index,
                        "prompt_sha256": sha256_text(row["prompt"]),
                        "response_sha256": sha256_text(row["response"]),
                    }
                )
            data_path = staging / "minutes_alignment" / f"{split}.jsonl"
            sidecar_path = (
                staging / "minutes_alignment" / "manifests" / f"{split}.jsonl"
            )
            _write_jsonl(data_path, rows)
            _write_jsonl(sidecar_path, manifests)
            artifacts["minutes_alignment"][split] = {
                "data": _release_descriptor(data_path, root=staging),
                "manifest": _release_descriptor(sidecar_path, root=staging),
            }
            split_counts[split] = len(rows)
            status_counts[split] = dict(
                Counter(
                    str(record["terminal_status"])
                    for record in downstream.terminals[split]
                )
            )

        rejections = [
            {
                "sample_id": record["sample_id"],
                "split": split,
                "source_index": record["source_index"],
                "terminal_status": record["terminal_status"],
                "rejection_stage": record["rejection_stage"],
                "rejection_reasons": record["rejection_reasons"],
            }
            for split in SPLITS
            for record in downstream.terminals[split]
            if record["terminal_status"] != v5.TERMINAL_PASS
        ]
        source_admission_audit = [
            {
                "sample_id": record["sample_id"],
                "split": split,
                "source_index": record["source_index"],
                "terminal_status": record["terminal_status"],
                "source_analysis_sha256": record["source_analysis_sha256"],
                "provided_data_sha256": record["provided_data_sha256"],
                "source_audit": record["source_audit"],
            }
            for split in SPLITS
            for record in downstream.terminals[split]
        ]
        validator_a_audit = [
            {
                "sample_id": record["sample_id"],
                "split": split,
                "source_index": record["source_index"],
                "terminal_status": record["terminal_status"],
                "validator_a": record["validator_a"],
            }
            for split in SPLITS
            for record in downstream.terminals[split]
            if record.get("validator_a")
        ]
        validator_b_audit = [
            {
                "sample_id": record["sample_id"],
                "split": split,
                "source_index": record["source_index"],
                "terminal_status": record["terminal_status"],
                "validator_b": record["validator_b"],
            }
            for split in SPLITS
            for record in downstream.terminals[split]
            if record.get("validator_b")
        ]
        repair_history_audit = [
            {
                "sample_id": record["sample_id"],
                "split": split,
                "source_index": record["source_index"],
                "terminal_status": record["terminal_status"],
                "events": record["repair_history"],
            }
            for split in SPLITS
            for record in downstream.terminals[split]
            if record.get("repair_history")
        ]
        evidence_ledger = [
            {
                "sample_id": record["sample_id"],
                "split": split,
                "source_index": record["source_index"],
                "terminal_status": record["terminal_status"],
                "terminal_record_sha256": sha256_text(canonical_json(record)),
                "source_audit_sha256": sha256_text(
                    canonical_json(record["source_audit"])
                ),
                "validator_a_sha256": (
                    sha256_text(canonical_json(record["validator_a"]))
                    if record.get("validator_a") else None
                ),
                "validator_b_sha256": (
                    sha256_text(canonical_json(record["validator_b"]))
                    if record.get("validator_b") else None
                ),
                "repair_history_sha256": (
                    sha256_text(canonical_json(record["repair_history"]))
                    if record.get("repair_history") else None
                ),
            }
            for split in SPLITS
            for record in downstream.terminals[split]
        ]
        b_mean_values = [
            float(item["validator_b"]["mean_score"])
            for item in validator_b_audit
            if isinstance(item["validator_b"].get("mean_score"), (int, float))
        ]
        b_min_values = [
            int(item["validator_b"]["min_score"])
            for item in validator_b_audit
            if isinstance(item["validator_b"].get("min_score"), int)
        ]
        repair_distribution = Counter(
            f"{event['split']}:{history['repair_type']}"
            for event in repair_history_audit
            for history in event["events"]
        )
        rejection_path = staging / "audits" / "rejections.jsonl"
        source_admission_path = staging / "audits" / "source_admission.jsonl"
        validator_a_path = staging / "audits" / "validator_a.jsonl"
        validator_b_path = staging / "audits" / "validator_b.jsonl"
        repair_history_path = staging / "audits" / "repair_history.jsonl"
        tokenizer_replay_path = staging / "audits" / "tokenizer_replay.jsonl"
        quality_path = staging / "audits" / "data_quality.json"
        evidence_ledger_path = staging / "audits" / "evidence_ledger.jsonl"
        _write_jsonl(rejection_path, rejections)
        _write_jsonl(source_admission_path, source_admission_audit)
        _write_jsonl(validator_a_path, validator_a_audit)
        _write_jsonl(validator_b_path, validator_b_audit)
        _write_jsonl(repair_history_path, repair_history_audit)
        _write_jsonl(tokenizer_replay_path, downstream.tokenizer_replay)
        _write_jsonl(evidence_ledger_path, evidence_ledger)
        _write_json(
            quality_path,
            {
                "schema_version": RELEASE_SCHEMA_VERSION,
                "source_rows": v5.EXPECTED_TOTAL,
                "split_pass_counts": split_counts,
                "terminal_status_counts": status_counts,
                "rejected_rows": len(rejections),
                "pass_rows": sum(split_counts.values()),
                "three_pass_splits_nonempty": all(split_counts.values()),
                "validator_a_rows": len(validator_a_audit),
                "validator_b_rows": len(validator_b_audit),
                "validator_b_mean_score": {
                    "count": len(b_mean_values),
                    "mean": (
                        round(sum(b_mean_values) / len(b_mean_values), 6)
                        if b_mean_values
                        else None
                    ),
                    "minimum": min(b_mean_values) if b_mean_values else None,
                    "maximum": max(b_mean_values) if b_mean_values else None,
                },
                "validator_b_min_score": {
                    "count": len(b_min_values),
                    "mean": (
                        round(sum(b_min_values) / len(b_min_values), 6)
                        if b_min_values
                        else None
                    ),
                    "minimum": min(b_min_values) if b_min_values else None,
                    "maximum": max(b_min_values) if b_min_values else None,
                },
                "repair_event_distribution": dict(sorted(repair_distribution.items())),
                "evidence_ledger_rows": len(evidence_ledger),
                "external_cache_counts": downstream.receipt["cache_counts"],
            },
        )
        artifacts["audits"] = {
            "rejections": _release_descriptor(rejection_path, root=staging),
            "source_admission": _release_descriptor(
                source_admission_path, root=staging
            ),
            "validator_a": _release_descriptor(validator_a_path, root=staging),
            "validator_b": _release_descriptor(validator_b_path, root=staging),
            "repair_history": _release_descriptor(repair_history_path, root=staging),
            "tokenizer_replay": _release_descriptor(
                tokenizer_replay_path, root=staging
            ),
            "data_quality": _release_descriptor(quality_path, root=staging),
            "evidence_ledger": _release_descriptor(
                evidence_ledger_path, root=staging
            ),
        }

        provenance_sources = {
            "source_handoff_manifest.json": (
                source.handoff.root / "handoff_manifest.json"
            ),
            "source_admission_receipt.json": (
                source.handoff.root / "sealed_source" / "source_admission_receipt.json"
            ),
            "source_prompt_contract.json": (
                source.handoff.root / "sealed_source" / "prompt_contract.json"
            ),
            "official_pre_action_reference_bank.jsonl": (
                source.handoff.root
                / "sealed_source"
                / "official_pre_action_reference_bank.jsonl"
            ),
            "downstream_execution_receipt.json": downstream.receipt_path,
            "downstream_prompt_contract.json": (
                downstream.root / "prompt_contract.json"
            ),
            "downstream_final_summary.json": downstream.root / "final_summary.json",
        }
        for filename, source_path in provenance_sources.items():
            target = staging / "provenance" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target)
            artifacts["provenance"][Path(filename).stem] = _release_descriptor(
                target, root=staging
            )

        manifest: dict[str, Any] = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "status": "complete",
            "dataset_role": DATASET_ROLE,
            "training_scope": TRAINING_SCOPE,
            "training_only": True,
            "evaluation_eligible": False,
            "source_rows": v5.EXPECTED_TOTAL,
            "split_pass_counts": split_counts,
            "terminal_status_counts": status_counts,
            "source_handoff": {
                "schema_version": source_sealer.SCHEMA_VERSION,
                "manifest_sha256": source.signed_manifest_sha256,
                "manifest_file_sha256": source.manifest_file_sha256,
            },
            "downstream": {
                "schema_version": _downstream_module().SCHEMA_VERSION,
                "execution_receipt_sha256": downstream.receipt["receipt_sha256"],
                "execution_receipt_file_sha256": sha256_file(downstream.receipt_path),
                "implementation_composite_sha256": downstream.receipt[
                    "implementation_composite_sha256"
                ],
                "run_binding_sha256": downstream.receipt["run_binding_sha256"],
            },
            "provider_identities": dict(downstream.provider_identities),
            "publisher_implementation": _publisher_implementation_contract(),
            "tokenizer_runtime_contract": v5._expected_tokenizer_runtime_contract(),
            "artifacts": artifacts,
        }
        manifest["manifest_sha256"] = sha256_text(canonical_json(manifest))
        manifest_path = staging / "release_manifest.json"
        _write_json(manifest_path, manifest)
        release_handoff = {
            "schema_version": RELEASE_HANDOFF_SCHEMA_VERSION,
            "status": "complete",
            "created_at": _utc_now(),
            "release_manifest_sha256": manifest["manifest_sha256"],
            "release_manifest_file_sha256": sha256_file(manifest_path),
            "source_handoff_manifest_sha256": source.signed_manifest_sha256,
            "source_handoff_manifest_file_sha256": source.manifest_file_sha256,
            "downstream_execution_receipt_sha256": downstream.receipt["receipt_sha256"],
            "publisher_implementation": manifest["publisher_implementation"],
            "split_data_sha256": {
                split: artifacts["minutes_alignment"][split]["data"]["sha256"]
                for split in SPLITS
            },
        }
        release_handoff["handoff_sha256"] = sha256_text(canonical_json(release_handoff))
        _write_json(staging / "handoff.json", release_handoff)
        verify_release(staging, tokenizer_path=tokenizer_path, tokenizer=tokenizer)
        _atomic_publish(staging, release)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_release(release, tokenizer_path=tokenizer_path, tokenizer=tokenizer)


def verify_release(
    release_root: Path = DEFAULT_RELEASE_ROOT,
    *,
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    _require(".." not in release_root.parts, "unsafe release path")
    release = release_root.absolute()
    _require(
        release.is_dir() and not release.is_symlink(), f"invalid release: {release}"
    )
    actual_files = _strict_tree_files(release, label="release")
    manifest = _read_json(release / "release_manifest.json", label="release manifest")
    unsigned_manifest = dict(manifest)
    stored_manifest_sha = unsigned_manifest.pop("manifest_sha256", None)
    _require(
        manifest.get("schema_version") == RELEASE_SCHEMA_VERSION
        and manifest.get("status") == "complete"
        and stored_manifest_sha == sha256_text(canonical_json(unsigned_manifest)),
        "release manifest drift",
    )
    handoff = _read_json(release / "handoff.json", label="release handoff")
    unsigned_handoff = dict(handoff)
    stored_handoff_sha = unsigned_handoff.pop("handoff_sha256", None)
    _require(
        handoff.get("schema_version") == RELEASE_HANDOFF_SCHEMA_VERSION
        and handoff.get("status") == "complete"
        and stored_handoff_sha == sha256_text(canonical_json(unsigned_handoff)),
        "release handoff drift",
    )
    _require(
        handoff.get("release_manifest_sha256") == stored_manifest_sha
        and handoff.get("release_manifest_file_sha256")
        == sha256_file(release / "release_manifest.json"),
        "release handoff/manifest binding drift",
    )
    _validate_publisher_implementation_binding(manifest, handoff)
    described: dict[Path, Mapping[str, Any]] = {}
    _verify_descriptor_tree(
        release,
        manifest.get("artifacts"),
        label="release_manifest.artifacts",
        observed=described,
    )
    actual_files = {path.absolute() for path in actual_files}
    expected_files = set(described) | {
        (release / "release_manifest.json").absolute(),
        (release / "handoff.json").absolute(),
    }
    _require(actual_files == expected_files, "release contains missing/orphan files")
    if tokenizer is None:
        _require(
            tokenizer_path.resolve() == DEFAULT_TOKENIZER_PATH.resolve(),
            "release tokenizer path is unauthorized",
        )
        try:
            v5_publisher._validate_exact_tokenizer_files(
                tokenizer_path,
                expected_tokenizer_path=DEFAULT_TOKENIZER_PATH,
            )
        except Exception as exc:
            raise PublicationError(f"release tokenizer file drift: {exc}") from exc
        tokenizer = v5._load_tokenizer(tokenizer_path)
    try:
        v5._verify_tokenizer_runtime_contract(tokenizer)
    except Exception as exc:
        raise PublicationError(f"release tokenizer runtime drift: {exc}") from exc
    split_counts: dict[str, int] = {}
    replay_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        split_artifacts = manifest["artifacts"]["minutes_alignment"][split]
        data_path = _resolve_downstream_artifact(
            release, split_artifacts["data"], label=f"release {split} data"
        )
        sidecar_path = _resolve_downstream_artifact(
            release,
            split_artifacts["manifest"],
            label=f"release {split} manifest",
        )
        rows = _read_jsonl(data_path, label=f"release {split} data")
        sidecars = _read_jsonl(sidecar_path, label=f"release {split} manifest")
        _require(rows and len(rows) == len(sidecars), f"release {split} row drift")
        for index, (row, sidecar) in enumerate(zip(rows, sidecars)):
            _require(
                set(row) == {"prompt", "response"},
                f"release {split}:{index} schema drift",
            )
            _require(
                sidecar.get("release_index") == index
                and sidecar.get("split") == split
                and sidecar.get("terminal_status") == v5.TERMINAL_PASS
                and sidecar.get("prompt_sha256") == sha256_text(row["prompt"])
                and sidecar.get("response_sha256") == sha256_text(row["response"]),
                f"release {split}:{index} sidecar drift",
            )
            replay_rows.append(
                {
                    "sample_id": sidecar["sample_id"],
                    "split": split,
                    **v5_publisher._token_replay(
                        tokenizer=tokenizer,
                        system_prompt=v5.STUDENT_SYSTEM_PROMPT,
                        prompt=row["prompt"],
                        response=row["response"],
                        sample_id=sidecar["sample_id"],
                    ),
                }
            )
        split_counts[split] = len(rows)
    _require(
        manifest.get("split_pass_counts") == split_counts,
        "release split-count drift",
    )
    tokenizer_path_release = _resolve_downstream_artifact(
        release,
        manifest["artifacts"]["audits"]["tokenizer_replay"],
        label="release tokenizer replay",
    )
    _require(
        _read_jsonl(tokenizer_path_release, label="release tokenizer replay")
        == replay_rows,
        "release tokenizer replay drift",
    )
    audit_descriptors = manifest["artifacts"]["audits"]
    audit_rows = {
        name: _read_jsonl(
            _resolve_downstream_artifact(
                release, audit_descriptors[name], label=f"release {name} audit"
            ),
            label=f"release {name} audit",
        )
        for name in (
            "evidence_ledger", "source_admission", "validator_a", "validator_b",
            "repair_history", "rejections",
        )
    }
    ledger = audit_rows["evidence_ledger"]
    _require(len(ledger) == manifest.get("source_rows"), "release ledger population drift")
    ledger_by_id = {str(row.get("sample_id")): row for row in ledger}
    _require(len(ledger_by_id) == len(ledger), "release ledger duplicate sample ID")
    pass_ids = {
        row["sample_id"]
        for split in SPLITS
        for row in _read_jsonl(
            _resolve_downstream_artifact(
                release,
                manifest["artifacts"]["minutes_alignment"][split]["manifest"],
                label=f"release {split} manifest closure",
            ),
            label=f"release {split} manifest closure",
        )
    }
    reject_ids = {str(row.get("sample_id")) for row in audit_rows["rejections"]}
    _require(
        not (pass_ids & reject_ids) and pass_ids | reject_ids == set(ledger_by_id),
        "release PASS/REJECT partition drift",
    )
    _require(
        {str(row.get("sample_id")) for row in audit_rows["source_admission"]}
        == set(ledger_by_id),
        "release source-admission population drift",
    )
    for name, field, payload in (
        ("source_admission", "source_audit_sha256", "source_audit"),
        ("validator_a", "validator_a_sha256", "validator_a"),
        ("validator_b", "validator_b_sha256", "validator_b"),
        ("repair_history", "repair_history_sha256", "events"),
    ):
        rows = audit_rows[name]
        expected_ids = {sample_id for sample_id, item in ledger_by_id.items() if item[field]}
        _require(
            {str(row.get("sample_id")) for row in rows} == expected_ids,
            f"release {name} population drift",
        )
        for row in rows:
            _require(
                sha256_text(canonical_json(row[payload]))
                == ledger_by_id[str(row["sample_id"])][field],
                f"release {name} evidence hash drift",
            )
    quality = _read_json(
        _resolve_downstream_artifact(
            release, audit_descriptors["data_quality"], label="release data quality"
        ),
        label="release data quality",
    )
    _require(
        quality.get("source_rows") == len(ledger)
        and quality.get("pass_rows") == len(pass_ids)
        and quality.get("rejected_rows") == len(reject_ids)
        and quality.get("evidence_ledger_rows") == len(ledger),
        "release data-quality population drift",
    )
    source_copy = _read_json(
        release / "provenance/source_handoff_manifest.json",
        label="release source handoff copy",
    )
    execution_copy = _read_json(
        release / "provenance/downstream_execution_receipt.json",
        label="release downstream execution receipt copy",
    )
    _validate_execution_receipt_shape(execution_copy)
    _require(
        quality.get("external_cache_counts") == execution_copy.get("cache_counts")
        and execution_copy.get("terminal_cache_count") == len(ledger),
        "release external cache ledger drift",
    )
    _require(
        source_copy.get("manifest_sha256")
        == manifest["source_handoff"]["manifest_sha256"]
        == handoff["source_handoff_manifest_sha256"]
        and sha256_file(release / "provenance/source_handoff_manifest.json")
        == manifest["source_handoff"]["manifest_file_sha256"]
        == handoff["source_handoff_manifest_file_sha256"],
        "release source-handoff provenance drift",
    )
    _require(
        execution_copy.get("receipt_sha256")
        == manifest["downstream"]["execution_receipt_sha256"]
        == handoff["downstream_execution_receipt_sha256"],
        "release downstream receipt provenance drift",
    )
    _require(
        sha256_file(release / "provenance/downstream_execution_receipt.json")
        == manifest["downstream"]["execution_receipt_file_sha256"],
        "release downstream receipt file-SHA drift",
    )
    _require(
        handoff.get("split_data_sha256")
        == {
            split: manifest["artifacts"]["minutes_alignment"][split]["data"]["sha256"]
            for split in SPLITS
        },
        "release handoff dataset SHA drift",
    )
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-acquisition-root", type=Path, default=DEFAULT_SOURCE_ACQUISITION_ROOT
    )
    parser.add_argument(
        "--source-handoff-root", type=Path, default=DEFAULT_SOURCE_HANDOFF_ROOT
    )
    parser.add_argument("--downstream-root", type=Path, default=DEFAULT_DOWNSTREAM_ROOT)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
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
            downstream_root=args.downstream_root,
            release_root=args.release_root,
            tokenizer_path=args.tokenizer_path,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
