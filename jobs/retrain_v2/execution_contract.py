"""Deterministic execution provenance primitives for retrain-v2.

This module is intentionally side-effect free: it reads source files and the
current Python interpreter inventory, but it never writes a contract, invokes a
package manager, starts a process, or reads model/dataset artifacts.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
HASH_ALGORITHM = "sha256"

SOURCE_DIRECTORY_ROOTS = (
    "jobs/train",
    "src/open_r1",
    "jobs/retrain_v2",
    "run/retrain_v2",
)
SOURCE_FILE_ROOTS = (
    "run/check_retrain_v2_envs.sh",
    "run/retrain_v2_env_smoke.py",
    "requirements/retrain_v2_train.lock",
    "requirements/retrain_v2_train.freeze.txt",
    "requirements/retrain_v2_judge.lock",
    "requirements/retrain_v2_judge.freeze.txt",
)
REQUIREMENT_FILES = SOURCE_FILE_ROOTS[-4:]
IGNORED_DIRECTORY_NAMES = frozenset({"__pycache__", ".pytest_cache"})

STAGE_TOPOLOGY = {
    "chk1": {
        "launcher": "ddp",
        "policy_gpus": [0, 1],
        "world_size": 2,
    },
    "chk2": {
        "judge_gpu": 0,
        "launcher": "single_policy_with_judge",
        "policy_gpus": [1],
        "world_size": 1,
    },
    "chk3": {
        "launcher": "ddp",
        "policy_gpus": [0, 1],
        "world_size": 2,
    },
    "chk4": {
        "launcher": "ddp",
        "policy_gpus": [0, 1],
        "world_size": 2,
    },
}

_CANONICAL_DISTRIBUTION_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_EXACT_FREEZE_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)"
    r"==(?P<version>[^\s;@]+)$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ExecutionContractError(ValueError):
    """Raised when execution provenance is incomplete, unsafe, or has drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExecutionContractError(message)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


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
        raise ExecutionContractError("Payload is not canonical JSON") from exc
    return rendered.encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ExecutionContractError(f"Unable to hash source artifact: {path}") from exc
    return digest.hexdigest()


def _repo_relative(repo_root: Path, path: Path, *, label: str) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ExecutionContractError(f"{label} escapes the repository: {path}") from exc
    return relative.as_posix()


def _file_record(repo_root: Path, path: Path, *, label: str) -> dict[str, Any]:
    _require(not path.is_symlink(), f"{label} must not be a symlink: {path}")
    _require(path.is_file(), f"{label} is not a regular file: {path}")
    return {
        "path": _repo_relative(repo_root, path, label=label),
        "sha256": _sha256_file(path),
    }


def _walk_source_directory(repo_root: Path, root: Path) -> list[dict[str, Any]]:
    label = f"source root {root.relative_to(repo_root).as_posix()}"
    _require(not root.is_symlink(), f"{label} must not be a symlink")
    _require(root.is_dir(), f"{label} does not exist or is not a directory")
    _repo_relative(repo_root, root, label=label)

    records: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise ExecutionContractError(
                f"Unable to scan source directory: {directory}"
            ) from exc
        for entry in entries:
            candidate = Path(entry.path)
            if entry.is_symlink():
                raise ExecutionContractError(
                    f"Source bundle must not contain symlinks: {candidate}"
                )
            if entry.is_dir(follow_symlinks=False):
                if entry.name in IGNORED_DIRECTORY_NAMES:
                    continue
                _repo_relative(repo_root, candidate, label="source directory")
                visit(candidate)
                continue
            if entry.is_file(follow_symlinks=False):
                if entry.name.endswith(".pyc"):
                    continue
                records.append(
                    _file_record(repo_root, candidate, label="source artifact")
                )
                continue
            raise ExecutionContractError(
                f"Source bundle contains a non-regular filesystem entry: {candidate}"
            )

    visit(root)
    return records


def _build_source_bundle(repo_root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for relative in SOURCE_DIRECTORY_ROOTS:
        lexical_root = repo_root / relative
        records.extend(_walk_source_directory(repo_root, lexical_root))
    for relative in SOURCE_FILE_ROOTS:
        records.append(
            _file_record(
                repo_root,
                repo_root / relative,
                label=f"source root {relative}",
            )
        )

    records.sort(key=lambda record: record["path"])
    paths = [record["path"] for record in records]
    _require(
        len(paths) == len(set(paths)), "Source roots produced duplicate file paths"
    )
    return {
        "schema_version": 1,
        "directory_roots": list(SOURCE_DIRECTORY_ROOTS),
        "file_roots": list(SOURCE_FILE_ROOTS),
        "ignored": {
            "directory_names": sorted(IGNORED_DIRECTORY_NAMES),
            "file_suffixes": [".pyc"],
        },
        "files": records,
    }


def _canonical_distribution_name(value: Any) -> str:
    _require(
        isinstance(value, str) and value.strip() != "", "Distribution name is missing"
    )
    canonical = re.sub(r"[-_.]+", "-", value.strip()).lower()
    _require(
        _CANONICAL_DISTRIBUTION_NAME_RE.fullmatch(canonical) is not None,
        f"Invalid distribution name: {value!r}",
    )
    return canonical


def _parse_exact_train_freeze(repo_root: Path) -> list[dict[str, str]]:
    """Parse the train freeze as an exact, option-free distribution inventory."""

    path = repo_root / "requirements/retrain_v2_train.freeze.txt"
    _require(not path.is_symlink(), f"Train freeze must not be a symlink: {path}")
    _require(path.is_file(), f"Train freeze is missing: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ExecutionContractError(f"Unable to read train freeze: {path}") from exc

    expected: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _EXACT_FREEZE_RE.fullmatch(line)
        _require(
            match is not None,
            f"Invalid exact train freeze entry at line {line_number}: {line!r}",
        )
        name = _canonical_distribution_name(match.group("name"))
        version = match.group("version")
        _require(
            name not in expected,
            f"Duplicate canonical distribution in train freeze: {name}",
        )
        expected[name] = version
    _require(expected, "Train freeze contains no package pins")
    return [{"name": name, "version": expected[name]} for name in sorted(expected)]


def _require_exact_train_environment(
    expected: list[dict[str, str]], observed: list[dict[str, str]]
) -> None:
    expected_by_name = {item["name"]: item["version"] for item in expected}
    observed_by_name = {item["name"]: item["version"] for item in observed}
    if expected_by_name == observed_by_name:
        return

    missing = sorted(expected_by_name.keys() - observed_by_name.keys())
    extra = sorted(observed_by_name.keys() - expected_by_name.keys())
    mismatched = [
        f"{name}: expected {expected_by_name[name]}, found {observed_by_name[name]}"
        for name in sorted(expected_by_name.keys() & observed_by_name.keys())
        if expected_by_name[name] != observed_by_name[name]
    ]
    details: list[str] = []
    if missing:
        details.append(f"missing={missing!r}")
    if extra:
        details.append(f"extra={extra!r}")
    if mismatched:
        details.append(f"version_mismatch={mismatched!r}")
    raise ExecutionContractError(
        "Interpreter environment drift detected: current interpreter does not exactly match "
        "requirements/retrain_v2_train.freeze.txt: " + "; ".join(details)
    )


def _collect_runtime_observation() -> dict[str, Any]:
    """Collect raw interpreter facts; isolated for deterministic monkeypatching."""

    distributions: list[dict[str, str]] = []
    for distribution in importlib.metadata.distributions():
        distributions.append(
            {
                "name": distribution.metadata.get("Name"),
                "version": distribution.version,
            }
        )

    try:
        import torch
    except ImportError as exc:
        raise ExecutionContractError(
            "Torch is required in the retrain-v2 environment"
        ) from exc

    return {
        "python_version": platform.python_version(),
        "executable": sys.executable,
        "distributions": distributions,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def _normalize_environment(
    repo_root: Path, observation: Mapping[str, Any]
) -> dict[str, Any]:
    expected_keys = {
        "python_version",
        "executable",
        "distributions",
        "torch_version",
        "torch_cuda_version",
    }
    _require(
        set(observation) == expected_keys, "Runtime observation has unexpected fields"
    )

    python_version = observation["python_version"]
    _require(
        isinstance(python_version, str) and python_version.strip() != "",
        "Python version is missing",
    )
    executable = observation["executable"]
    _require(
        isinstance(executable, str) and Path(executable).is_absolute(),
        "Python executable must be an absolute path",
    )
    canonical_executable = str(Path(executable).resolve())

    raw_distributions = observation["distributions"]
    _require(
        isinstance(raw_distributions, Iterable)
        and not isinstance(raw_distributions, (str, bytes, Mapping)),
        "Runtime distributions must be an iterable of records",
    )
    normalized_distributions: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for raw in raw_distributions:
        _require(isinstance(raw, Mapping), "Distribution record must be an object")
        _require(
            set(raw) == {"name", "version"}, "Distribution record has unexpected fields"
        )
        name = _canonical_distribution_name(raw["name"])
        version = raw["version"]
        _require(
            isinstance(version, str) and version.strip() != "",
            f"Distribution {name!r} has no version",
        )
        _require(
            name not in seen_names, f"Duplicate canonical distribution name: {name}"
        )
        seen_names.add(name)
        normalized_distributions.append({"name": name, "version": version.strip()})
    normalized_distributions.sort(key=lambda item: item["name"])
    expected_distributions = _parse_exact_train_freeze(repo_root)
    _require_exact_train_environment(expected_distributions, normalized_distributions)

    torch_version = observation["torch_version"]
    torch_cuda_version = observation["torch_cuda_version"]
    _require(
        isinstance(torch_version, str) and torch_version.strip() != "",
        "Torch version is missing",
    )
    _require(
        isinstance(torch_cuda_version, str) and torch_cuda_version.strip() != "",
        "Torch CUDA version is missing",
    )

    requirements = [
        _file_record(
            repo_root,
            repo_root / relative,
            label=f"requirement artifact {relative}",
        )
        for relative in REQUIREMENT_FILES
    ]
    requirements.sort(key=lambda record: record["path"])
    return {
        "schema_version": 1,
        "python": {
            "version": python_version.strip(),
            "executable": canonical_executable,
        },
        "distributions": normalized_distributions,
        "torch": {
            "version": torch_version.strip(),
            "cuda_version": torch_cuda_version.strip(),
        },
        "requirements": requirements,
    }


def _layer(payload: Mapping[str, Any]) -> dict[str, Any]:
    canonical_payload = copy.deepcopy(dict(payload))
    return {
        "algorithm": HASH_ALGORITHM,
        "payload": canonical_payload,
        "sha256": _canonical_sha256(canonical_payload),
    }


def canonical_stage_topology() -> dict[str, Any]:
    """Return the immutable two-A30 stage topology as a fresh object."""

    return copy.deepcopy(STAGE_TOPOLOGY)


def validate_stage_topology(
    stage_id: str, topology: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a future launch/receipt topology without performing I/O."""

    _require(stage_id in STAGE_TOPOLOGY, f"Unknown retrain-v2 stage: {stage_id}")
    expected = STAGE_TOPOLOGY[stage_id]
    _require(dict(topology) == expected, f"Stage topology mismatch for {stage_id}")
    return copy.deepcopy(expected)


def build_execution_contract(
    repo_root: str | Path, *, created_at_utc: str | None = None
) -> dict[str, Any]:
    """Build an in-memory execution contract for the current source and interpreter."""

    root = Path(repo_root).resolve()
    _require(root.is_dir(), f"Repository root does not exist: {root}")
    source_bundle = _layer(_build_source_bundle(root))
    environment = _layer(_normalize_environment(root, _collect_runtime_observation()))
    topology = canonical_stage_topology()
    deterministic = {
        "schema_version": SCHEMA_VERSION,
        "source_bundle_sha256": source_bundle["sha256"],
        "environment_sha256": environment["sha256"],
        "stage_topology": topology,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": created_at_utc or _utc_now(),
        "source_bundle": source_bundle,
        "environment": environment,
        "stage_topology": topology,
        "contract_sha256": _canonical_sha256(deterministic),
    }


def _validate_layer(layer: Any, *, label: str) -> dict[str, Any]:
    _require(isinstance(layer, Mapping), f"{label} layer must be an object")
    _require(
        set(layer) == {"algorithm", "payload", "sha256"},
        f"{label} layer has unexpected fields",
    )
    _require(
        layer["algorithm"] == HASH_ALGORITHM, f"Unsupported {label} hash algorithm"
    )
    payload = layer["payload"]
    _require(isinstance(payload, Mapping), f"{label} payload must be an object")
    sha = layer["sha256"]
    _require(
        isinstance(sha, str) and _SHA256_RE.fullmatch(sha) is not None,
        f"Invalid {label} SHA-256",
    )
    _require(_canonical_sha256(payload) == sha, f"{label} layer hash mismatch")
    return copy.deepcopy(dict(payload))


def _validate_created_at(value: Any) -> None:
    _require(
        isinstance(value, str) and value.endswith("Z"), "created_at_utc must be UTC"
    )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ExecutionContractError(
            "created_at_utc is not an ISO-8601 timestamp"
        ) from exc
    _require(
        parsed.utcoffset() == timezone.utc.utcoffset(parsed),
        "created_at_utc must be UTC",
    )


def verify_execution_contract(
    record: Mapping[str, Any], repo_root: str | Path
) -> dict[str, Any]:
    """Fail closed if source, requirements, interpreter, or packages have drifted."""

    _require(isinstance(record, Mapping), "Execution contract must be an object")
    expected_keys = {
        "schema_version",
        "created_at_utc",
        "source_bundle",
        "environment",
        "stage_topology",
        "contract_sha256",
    }
    _require(set(record) == expected_keys, "Execution contract has unexpected fields")
    _require(
        record["schema_version"] == SCHEMA_VERSION,
        "Unsupported execution contract schema",
    )
    _validate_created_at(record["created_at_utc"])
    stored_source = _validate_layer(record["source_bundle"], label="source bundle")
    stored_environment = _validate_layer(record["environment"], label="environment")
    stored_topology = record["stage_topology"]
    _require(stored_topology == STAGE_TOPOLOGY, "Execution stage topology drift")

    deterministic = {
        "schema_version": SCHEMA_VERSION,
        "source_bundle_sha256": record["source_bundle"]["sha256"],
        "environment_sha256": record["environment"]["sha256"],
        "stage_topology": stored_topology,
    }
    contract_sha = record["contract_sha256"]
    _require(
        isinstance(contract_sha, str)
        and _SHA256_RE.fullmatch(contract_sha) is not None,
        "Invalid execution contract SHA-256",
    )
    _require(
        _canonical_sha256(deterministic) == contract_sha,
        "Execution contract hash mismatch",
    )

    root = Path(repo_root).resolve()
    _require(root.is_dir(), f"Repository root does not exist: {root}")
    current_source = _build_source_bundle(root)
    current_requirements = [
        _file_record(
            root,
            root / relative,
            label=f"requirement artifact {relative}",
        )
        for relative in REQUIREMENT_FILES
    ]
    current_requirements.sort(key=lambda record: record["path"])

    _require(
        stored_environment.get("requirements") == current_requirements,
        "Requirement artifact drift detected",
    )
    _require(stored_source == current_source, "Source bundle drift detected")
    current_environment = _normalize_environment(root, _collect_runtime_observation())
    stored_runtime = dict(stored_environment)
    current_runtime = dict(current_environment)
    stored_runtime.pop("requirements", None)
    current_runtime.pop("requirements", None)
    _require(
        stored_runtime == current_runtime, "Interpreter environment drift detected"
    )

    return {
        "status": "verified",
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": contract_sha,
        "source_bundle_sha256": record["source_bundle"]["sha256"],
        "environment_sha256": record["environment"]["sha256"],
    }


__all__ = [
    "ExecutionContractError",
    "build_execution_contract",
    "canonical_stage_topology",
    "validate_stage_topology",
    "verify_execution_contract",
]
