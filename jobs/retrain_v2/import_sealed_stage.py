"""Import a sealed chk1 artifact from another immutable retrain-v2 run.

This is deliberately a narrow recovery boundary.  It permits a fresh run to
reuse a completed chk1 after chk2-only source code changes, without claiming
that chk1 was produced by the fresh run's execution contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from jobs.retrain_v2.merge_attestation import (
    MergeAttestationError,
    verify_merge_attestation,
)
from jobs.retrain_v2.stage_lock import StageLockError, require_inherited_stage_lock
from open_r1.provenance import fingerprint_artifact_path, sha256_file


SCHEMA_VERSION = 1
ATTESTATION_TYPE = "retrain_v2_sealed_chk1_import"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class SealedStageImportError(ValueError):
    """Raised when a cross-run chk1 import is incomplete or has drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SealedStageImportError(message)


def _canonical_bytes(payload: Any) -> bytes:
    try:
        rendered = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SealedStageImportError("Import evidence is not canonical JSON") from exc
    return rendered.encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(not path.is_symlink(), f"{label} must not be a symlink: {path}")
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SealedStageImportError(f"Unable to parse {label}: {path}") from exc
    _require(isinstance(payload, dict), f"{label} must contain an object")
    return payload


def _canonical_repo_path(
    repo_root: Path, value: str | Path, *, label: str, require_exists: bool = True
) -> Path:
    candidate = Path(value)
    lexical = Path(os.path.abspath(os.fspath(candidate if candidate.is_absolute() else repo_root / candidate)))
    try:
        relative = lexical.relative_to(repo_root)
    except ValueError as exc:
        raise SealedStageImportError(f"{label} escapes the repository: {value}") from exc
    current = repo_root
    for part in relative.parts:
        current /= part
        _require(not current.is_symlink(), f"{label} contains a symlink: {current}")
    resolved = lexical.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise SealedStageImportError(f"{label} escapes the repository: {value}") from exc
    if require_exists:
        _require(resolved.exists(), f"{label} is missing: {resolved}")
    return resolved


def _relative(repo_root: Path, path: Path, *, label: str) -> str:
    return _canonical_repo_path(repo_root, path, label=label).relative_to(repo_root).as_posix()


def _manifest_path(repo_root: Path, value: str | Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    path = _canonical_repo_path(repo_root, value, label=label)
    manifest = _load_json(path, label=label)
    _require(manifest.get("schema_version") == 2, f"Unsupported {label} schema")
    run_id = manifest.get("run_id")
    _require(
        isinstance(run_id, str) and _SAFE_ID_RE.fullmatch(run_id) is not None,
        f"{label} has an unsafe run_id",
    )
    expected = (repo_root / "output" / "training" / "retrain_v2" / run_id / "run_manifest.json").resolve()
    _require(path == expected, f"{label} is outside its canonical run directory")
    return path, manifest


def _validate_sha(value: Any, *, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{label} is not a SHA-256 digest",
    )
    return value


def _validate_stored_execution_contract(record: Any) -> str:
    _require(isinstance(record, dict), "Source execution contract is missing")
    _require(
        set(record)
        == {
            "schema_version",
            "created_at_utc",
            "source_bundle",
            "environment",
            "stage_topology",
            "contract_sha256",
        },
        "Source execution contract schema mismatch",
    )
    _require(record["schema_version"] == 1, "Unsupported source execution contract")
    for key in ("source_bundle", "environment"):
        layer = record[key]
        _require(
            isinstance(layer, dict)
            and set(layer) == {"algorithm", "payload", "sha256"}
            and layer["algorithm"] == "sha256"
            and isinstance(layer["payload"], dict),
            f"Source execution contract {key} layer is invalid",
        )
        _require(
            _canonical_sha256(layer["payload"])
            == _validate_sha(layer["sha256"], label=f"source {key} SHA"),
            f"Source execution contract {key} layer hash mismatch",
        )
    deterministic = {
        "schema_version": 1,
        "source_bundle_sha256": record["source_bundle"]["sha256"],
        "environment_sha256": record["environment"]["sha256"],
        "stage_topology": record["stage_topology"],
    }
    contract_sha = _validate_sha(record["contract_sha256"], label="source contract SHA")
    _require(_canonical_sha256(deterministic) == contract_sha, "Source execution contract hash mismatch")
    return contract_sha


def _load_receipt(path: Path, *, receipt_type: str) -> dict[str, Any]:
    receipt = _load_json(path, label=f"source chk1 {receipt_type} receipt")
    _require(
        set(receipt)
        == {"schema_version", "receipt_type", "recorded_at_utc", "lineage", "lineage_sha256"},
        f"Source chk1 {receipt_type} receipt schema mismatch",
    )
    _require(receipt["schema_version"] == 1, "Unsupported source receipt schema")
    _require(receipt["receipt_type"] == receipt_type, f"Wrong source {receipt_type} receipt type")
    _require(isinstance(receipt["lineage"], dict), "Source receipt lineage is missing")
    lineage_sha = _validate_sha(receipt["lineage_sha256"], label=f"source {receipt_type} lineage SHA")
    _require(
        _canonical_sha256(receipt["lineage"]) == lineage_sha,
        f"Source chk1 {receipt_type} receipt lineage hash mismatch",
    )
    return receipt


def _validate_source_evidence(
    source_path: Path, source: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    source_run_id = source.get("run_id")
    stages = source.get("stages")
    _require(isinstance(stages, dict), "Source stages are missing")
    foundation = stages.get("chk0")
    stage = stages.get("chk1")
    _require(isinstance(foundation, dict), "Source chk0 is missing")
    _require(isinstance(stage, dict), "Source chk1 is missing")
    _require(foundation.get("status") == "sealed", "Source chk0 is not sealed")
    _require(stage.get("status") == "sealed", "Source chk1 is not sealed")
    _require(stage.get("parent") == "chk0", "Source chk1 parent mismatch")
    contract_sha = _validate_stored_execution_contract(source.get("execution_contract"))
    _require(stage.get("execution_contract_sha256") == contract_sha, "Source chk1 contract binding mismatch")

    foundation_record = foundation.get("artifact")
    adapter_record = stage.get("adapter")
    artifact_record = stage.get("artifact")
    for record, label in (
        (foundation_record, "source chk0 artifact"),
        (adapter_record, "source chk1 adapter"),
        (artifact_record, "source chk1 artifact"),
    ):
        _require(isinstance(record, dict), f"{label} record is missing")
    observed_foundation = fingerprint_artifact_path(foundation_record["path"])
    observed_adapter = fingerprint_artifact_path(adapter_record["path"])
    observed_artifact = fingerprint_artifact_path(artifact_record["path"])
    _require(observed_foundation == foundation_record, "Source chk0 artifact record drift")
    _require(observed_adapter == adapter_record, "Source chk1 adapter record drift")
    _require(observed_artifact == artifact_record, "Source chk1 artifact record drift")
    _require(
        stage.get("parent_artifact_sha256") == observed_foundation["sha256"],
        "Source chk1 parent hash mismatch",
    )

    config_record = stage.get("resolved_config")
    binding = stage.get("data_binding")
    _require(isinstance(config_record, dict), "Source chk1 config record is missing")
    _require(isinstance(binding, dict), "Source chk1 data binding is missing")
    config_path = _canonical_repo_path(repo_root, config_record.get("path", ""), label="source chk1 config")
    config_sha = sha256_file(config_path)
    _require(config_sha == config_record.get("sha256"), "Source chk1 config hash mismatch")
    data_binding_sha = _canonical_sha256(binding)

    run_root = source_path.parent
    training_path = run_root / "receipts" / "chk1.training.json"
    merge_path = run_root / "receipts" / "chk1.merge.json"
    training = _load_receipt(training_path, receipt_type="training")
    merge = _load_receipt(merge_path, receipt_type="merge")
    training_lineage = training["lineage"]
    merge_lineage = merge["lineage"]
    topology = source["execution_contract"].get("stage_topology", {}).get("chk1")
    common = {
        "run_id": source_run_id,
        "stage_id": "chk1",
        "config_sha256": config_sha,
        "parent_sha256": observed_foundation["sha256"],
        "data_binding_sha256": data_binding_sha,
        "execution_contract_sha256": contract_sha,
        "topology": topology,
    }
    for key, expected in common.items():
        _require(training_lineage.get(key) == expected, f"Source training receipt {key} mismatch")
        _require(merge_lineage.get(key) == expected, f"Source merge receipt {key} mismatch")
    _require(
        training_lineage.get("adapter_fingerprint", {}).get("sha256") == observed_adapter["sha256"],
        "Source training receipt adapter hash mismatch",
    )
    _require(merge_lineage.get("adapter_sha256") == observed_adapter["sha256"], "Source merge receipt adapter hash mismatch")
    _require(
        merge_lineage.get("merged_fingerprint", {}).get("sha256") == observed_artifact["sha256"],
        "Source merge receipt artifact hash mismatch",
    )
    training_file_sha = sha256_file(training_path)
    _require(merge_lineage.get("training_receipt_file_sha256") == training_file_sha, "Source merge receipt training file hash mismatch")
    _require(merge_lineage.get("training_receipt_sha256") == training["lineage_sha256"], "Source merge receipt training lineage mismatch")
    _require(stage.get("training_receipt_sha256") == training["lineage_sha256"], "Source stage training receipt mismatch")
    _require(stage.get("merge_receipt_sha256") == merge["lineage_sha256"], "Source stage merge receipt mismatch")

    expected_binding = {
        "schema_version": 1,
        "run_id": source_run_id,
        "stage_id": "chk1",
        "base": {
            "path": _relative(repo_root, Path(foundation_record["path"]), label="source chk0 artifact"),
            "sha256": observed_foundation["sha256"],
        },
        "adapter": {
            "path": _relative(repo_root, Path(adapter_record["path"]), label="source chk1 adapter"),
            "sha256": observed_adapter["sha256"],
        },
        "resolved_config": {
            "path": _relative(repo_root, config_path, label="source chk1 config"),
            "sha256": config_sha,
        },
        "training_receipt": {
            "path": _relative(repo_root, training_path, label="source training receipt"),
            "file_sha256": training_file_sha,
            "lineage_sha256": training["lineage_sha256"],
        },
        "execution_contract_sha256": contract_sha,
        "data_binding_sha256": data_binding_sha,
    }
    try:
        merge_attestation = verify_merge_attestation(
            artifact_record["path"], expected_binding=expected_binding
        )
    except MergeAttestationError as exc:
        raise SealedStageImportError(f"Source merge attestation verification failed: {exc}") from exc
    merge_attestation_record = merge_lineage.get("merge_attestation")
    _require(isinstance(merge_attestation_record, dict), "Source merge receipt attestation is missing")
    attestation_path = _canonical_repo_path(
        repo_root, merge_attestation_record.get("path", ""), label="source merge attestation"
    )
    _require(attestation_path == Path(artifact_record["path"]).resolve() / "merge_attestation.json", "Source merge attestation path mismatch")
    _require(sha256_file(attestation_path) == merge_attestation_record.get("sha256"), "Source merge attestation file hash mismatch")
    _require(
        merge_attestation.get("canonical_payload_sha256")
        == merge_attestation_record.get("canonical_payload_sha256"),
        "Source merge attestation payload hash mismatch",
    )

    return {
        "stage": copy.deepcopy(stage),
        "foundation": observed_foundation,
        "adapter": observed_adapter,
        "artifact": observed_artifact,
        "execution_contract_sha256": contract_sha,
        "training_receipt": {
            "path": _relative(repo_root, training_path, label="source training receipt"),
            "file_sha256": training_file_sha,
            "lineage_sha256": training["lineage_sha256"],
        },
        "merge_receipt": {
            "path": _relative(repo_root, merge_path, label="source merge receipt"),
            "file_sha256": sha256_file(merge_path),
            "lineage_sha256": merge["lineage_sha256"],
        },
        "merge_attestation": {
            "path": _relative(repo_root, attestation_path, label="source merge attestation"),
            "file_sha256": sha256_file(attestation_path),
            "canonical_payload_sha256": merge_attestation["canonical_payload_sha256"],
        },
    }


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists() and not path.is_symlink(), f"Import attestation already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise SealedStageImportError(f"Import attestation already exists: {path}") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _write_atomic(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def import_sealed_chk1(
    target_run_manifest: str | Path,
    source_run_manifest: str | Path,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Bind one source chk1 into a fresh target run under the target chk1 lock."""

    root = Path(repo_root).resolve()
    target_path, target = _manifest_path(root, target_run_manifest, label="target run manifest")
    source_path, source = _manifest_path(root, source_run_manifest, label="source run manifest")
    _require(target_path != source_path, "Source and target runs must differ")
    try:
        require_inherited_stage_lock(target_path, "chk1", root)
    except StageLockError as exc:
        raise SealedStageImportError(str(exc)) from exc

    _require(target.get("phase") == "upstream_ready", "Target run is not upstream-ready")
    target_stages = target.get("stages")
    _require(isinstance(target_stages, dict), "Target stages are missing")
    target_chk1 = target_stages.get("chk1")
    target_chk2 = target_stages.get("chk2")
    _require(isinstance(target_chk1, dict) and target_chk1.get("status") == "pending", "Target chk1 is not fresh")
    _require(isinstance(target_chk2, dict) and target_chk2.get("status") == "pending", "Target chk2 is not fresh")
    _require("imported_from" not in target_chk1, "Target chk1 is already imported")
    target_root = target_path.parent
    for path in (
        target_root / "adapters" / "chk1",
        target_root / "merged" / "chk1",
        target_root / "receipts" / "chk1.training.json",
        target_root / "receipts" / "chk1.merge.json",
        target_root / "imports" / "chk1.json",
    ):
        _require(not path.exists() and not path.is_symlink(), f"Target chk1 is not fresh: {path}")

    target_contract = target.get("execution_contract")
    _require(isinstance(target_contract, dict), "Target execution contract is missing")
    target_contract_sha = _validate_sha(target_contract.get("contract_sha256"), label="target contract SHA")
    evidence = _validate_source_evidence(source_path, source, repo_root=root)
    _require(
        target_stages.get("chk0", {}).get("artifact") == evidence["foundation"],
        "Source and target chk0 artifacts differ",
    )
    source_base = source.get("data_releases", {}).get("base")
    target_base = target.get("data_releases", {}).get("base")
    _require(isinstance(source_base, dict) and isinstance(target_base, dict), "Base release binding is missing")
    _require(source_base.get("sha256") == target_base.get("sha256"), "Source and target base releases differ")
    _require(source.get("base_release_id") == target.get("base_release_id"), "Source and target base release IDs differ")

    source_manifest_file_sha = sha256_file(source_path)
    source_stage_sha = _canonical_sha256(evidence["stage"])
    attestation = {
        "schema_version": SCHEMA_VERSION,
        "attestation_type": ATTESTATION_TYPE,
        "recorded_at_utc": _utc_now(),
        "target_run_id": target["run_id"],
        "target_execution_contract_sha256": target_contract_sha,
        "source_run_id": source["run_id"],
        "source_manifest": {
            "path": _relative(root, source_path, label="source run manifest"),
            "file_sha256": source_manifest_file_sha,
        },
        "source_stage_record_sha256": source_stage_sha,
        "source_execution_contract_sha256": evidence["execution_contract_sha256"],
        "source_foundation": evidence["foundation"],
        "source_adapter": evidence["adapter"],
        "source_artifact": evidence["artifact"],
        "source_training_receipt": evidence["training_receipt"],
        "source_merge_receipt": evidence["merge_receipt"],
        "source_merge_attestation": evidence["merge_attestation"],
        "source_base_release_sha256": source_base["sha256"],
    }
    attestation_path = target_root / "imports" / "chk1.json"
    _write_json_exclusive(attestation_path, attestation)

    config_path = _canonical_repo_path(
        root, target_chk2.get("resolved_config", {}).get("path", ""), label="target chk2 config"
    )
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SealedStageImportError("Unable to parse target chk2 config") from exc
    _require(isinstance(config, dict), "Target chk2 config must be a mapping")
    config["model_name_or_path"] = evidence["artifact"]["path"]
    rendered_config = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    _write_atomic(config_path, rendered_config)

    imported_stage = copy.deepcopy(evidence["stage"])
    imported_stage["imported_from"] = {
        "schema_version": SCHEMA_VERSION,
        "source_run_id": source["run_id"],
        "attestation": {
            "path": _relative(root, attestation_path, label="chk1 import attestation"),
            "sha256": sha256_file(attestation_path),
        },
    }
    target["stages"]["chk1"] = imported_stage
    target["stages"]["chk2"]["resolved_config"]["sha256"] = sha256_file(config_path)
    target.setdefault("binding_events", []).append(
        {
            "event": "import_sealed_chk1",
            "bound_at_utc": attestation["recorded_at_utc"],
            "source_run_id": source["run_id"],
            "source_chk1_artifact_sha256": evidence["artifact"]["sha256"],
            "source_execution_contract_sha256": evidence["execution_contract_sha256"],
            "target_execution_contract_sha256": target_contract_sha,
            "attestation_sha256": imported_stage["imported_from"]["attestation"]["sha256"],
        }
    )
    rendered_manifest = json.dumps(target, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_atomic(target_path, rendered_manifest)
    return {
        "status": "imported",
        "target_run_id": target["run_id"],
        "source_run_id": source["run_id"],
        "artifact_sha256": evidence["artifact"]["sha256"],
        "attestation": str(attestation_path),
    }


def verify_imported_chk1(
    target_run_manifest: str | Path,
    imported_stage: Mapping[str, Any],
    repo_root: str | Path,
) -> dict[str, Any]:
    """Revalidate a cross-run chk1 import without requiring old live source code."""

    root = Path(repo_root).resolve()
    target_path, target = _manifest_path(root, target_run_manifest, label="target run manifest")
    recorded_import = imported_stage.get("imported_from")
    _require(isinstance(recorded_import, dict), "Imported chk1 provenance is missing")
    _require(recorded_import.get("schema_version") == SCHEMA_VERSION, "Unsupported chk1 import schema")
    attestation_record = recorded_import.get("attestation")
    _require(isinstance(attestation_record, dict), "Imported chk1 attestation record is missing")
    attestation_path = _canonical_repo_path(root, attestation_record.get("path", ""), label="chk1 import attestation")
    _require(sha256_file(attestation_path) == attestation_record.get("sha256"), "chk1 import attestation file drift")
    attestation = _load_json(attestation_path, label="chk1 import attestation")
    _require(attestation.get("schema_version") == SCHEMA_VERSION, "Unsupported chk1 import attestation")
    _require(attestation.get("attestation_type") == ATTESTATION_TYPE, "Wrong chk1 import attestation type")
    _require(attestation.get("target_run_id") == target.get("run_id"), "chk1 import target run mismatch")
    _require(attestation.get("source_run_id") == recorded_import.get("source_run_id"), "chk1 import source run mismatch")
    _require(
        attestation.get("target_execution_contract_sha256")
        == target.get("execution_contract", {}).get("contract_sha256"),
        "chk1 import target execution contract mismatch",
    )
    source_record = attestation.get("source_manifest")
    _require(isinstance(source_record, dict), "chk1 import source manifest record is missing")
    source_path, source = _manifest_path(root, source_record.get("path", ""), label="source run manifest")
    _require(sha256_file(source_path) == source_record.get("file_sha256"), "Source run manifest drift after import")
    evidence = _validate_source_evidence(source_path, source, repo_root=root)
    source_stage = evidence["stage"]
    stage_without_import = {key: copy.deepcopy(value) for key, value in imported_stage.items() if key != "imported_from"}
    _require(stage_without_import == source_stage, "Imported chk1 stage record differs from source")
    _require(_canonical_sha256(source_stage) == attestation.get("source_stage_record_sha256"), "Imported chk1 source stage hash mismatch")
    comparisons = {
        "source_execution_contract_sha256": evidence["execution_contract_sha256"],
        "source_foundation": evidence["foundation"],
        "source_adapter": evidence["adapter"],
        "source_artifact": evidence["artifact"],
        "source_training_receipt": evidence["training_receipt"],
        "source_merge_receipt": evidence["merge_receipt"],
        "source_merge_attestation": evidence["merge_attestation"],
        "source_base_release_sha256": source.get("data_releases", {}).get("base", {}).get("sha256"),
    }
    for key, expected in comparisons.items():
        _require(attestation.get(key) == expected, f"chk1 import {key} mismatch")
    _require(target.get("stages", {}).get("chk0", {}).get("artifact") == evidence["foundation"], "Imported chk1 foundation differs from target")
    _require(target.get("data_releases", {}).get("base", {}).get("sha256") == attestation["source_base_release_sha256"], "Imported chk1 base release differs from target")
    return {
        "adapter": evidence["adapter"],
        "artifact": evidence["artifact"],
        "adapter_sha256": evidence["adapter"]["sha256"],
        "merged_sha256": evidence["artifact"]["sha256"],
        "training_receipt_sha256": evidence["training_receipt"]["lineage_sha256"],
        "merge_receipt_sha256": evidence["merge_receipt"]["lineage_sha256"],
        "source_execution_contract_sha256": evidence["execution_contract_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--target-run-manifest", type=Path, required=True)
    parser.add_argument("--source-run-manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = import_sealed_chk1(
            args.target_run_manifest,
            args.source_run_manifest,
            repo_root=args.repo_root,
        )
    except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
