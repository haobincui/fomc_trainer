"""Metadata-keyed cache for expensive retrain-v2 preflight checks.

The immutable DAG still performs its full cryptographic verification after any
relevant input changes.  Between changes, launch retries can reuse the last
successful preflight receipt.  The cache key intentionally uses filesystem
metadata (path, type, size, and nanosecond mtime), so probing it never re-reads
multi-gigabyte model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
SCOPES = ("immutable", "launch")
STAGES = ("chk1", "chk2", "chk3", "chk4")
IGNORED_DIRECTORY_NAMES = frozenset({"__pycache__", ".pytest_cache"})
IGNORED_FILE_SUFFIXES = (".pyc",)


class PreflightCacheError(ValueError):
    """Raised when a preflight cache request is incomplete or unsafe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightCacheError(message)


def _canonical_bytes(payload: Any) -> bytes:
    try:
        value = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PreflightCacheError("preflight payload is not canonical JSON") from exc
    return value.encode("utf-8")


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightCacheError(f"unable to read {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} must contain a JSON object")
    return value


def _inside_repo(repo_root: Path, path: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise PreflightCacheError(f"{label} escapes the repository: {path}") from exc
    return resolved


def _path_from_value(
    repo_root: Path, value: Any, *, label: str, allow_outside_repo: bool = False
) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve(strict=False) if allow_outside_repo else _inside_repo(
        repo_root, path, label=label
    )


def _add_path(paths: set[Path], path: Path | None) -> None:
    if path is not None:
        paths.add(path)


def dependency_paths(
    repo_root: str | Path,
    run_manifest: str | Path,
    stage: str,
    scope: str,
) -> list[Path]:
    """Resolve the files/directories whose metadata invalidates one cache scope."""

    _require(stage in STAGES, f"unsupported stage: {stage}")
    _require(scope in SCOPES, f"unsupported cache scope: {scope}")
    root = Path(repo_root).resolve()
    manifest_path = _inside_repo(root, Path(run_manifest), label="run manifest")
    manifest = _load_json(manifest_path, label="run manifest")
    run_root = manifest_path.parent
    paths: set[Path] = {manifest_path}

    source_payload = (
        manifest.get("execution_contract", {})
        .get("source_bundle", {})
        .get("payload", {})
    )
    _require(isinstance(source_payload, dict), "execution source bundle is missing")
    for index, value in enumerate(source_payload.get("directory_roots", [])):
        _add_path(
            paths,
            _path_from_value(root, value, label=f"source directory {index}"),
        )
    for index, value in enumerate(source_payload.get("file_roots", [])):
        _add_path(paths, _path_from_value(root, value, label=f"source file {index}"))

    dag = manifest.get("dag", {})
    if isinstance(dag, dict):
        _add_path(paths, _path_from_value(root, dag.get("path"), label="DAG"))

    stages = manifest.get("stages", {})
    _require(isinstance(stages, dict), "manifest stages are missing")
    stage_entry = stages.get(stage)
    _require(isinstance(stage_entry, dict), f"manifest stage is missing: {stage}")
    for key in ("config_template", "resolved_config"):
        value = stage_entry.get(key)
        if isinstance(value, dict):
            _add_path(
                paths,
                _path_from_value(root, value.get("path"), label=f"{stage} {key}"),
            )

    binding = stage_entry.get("data_binding")
    if isinstance(binding, dict):
        _add_path(
            paths,
            _path_from_value(
                root, binding.get("dataset_path"), label=f"{stage} dataset"
            ),
        )
        release_role = binding.get("release_role")
        releases = manifest.get("data_releases", {})
        release = releases.get(release_role) if isinstance(releases, dict) else None
        if isinstance(release, dict):
            _add_path(
                paths,
                _path_from_value(
                    root, release.get("path"), label=f"{stage} data release"
                ),
            )

    parent_id = stage_entry.get("parent")
    parent = stages.get(parent_id) if isinstance(parent_id, str) else None
    if isinstance(parent, dict):
        artifact = parent.get("artifact")
        if isinstance(artifact, dict):
            _add_path(
                paths,
                _path_from_value(
                    root, artifact.get("path"), label=f"{stage} parent artifact"
                ),
            )

    imported = stage_entry.get("imported_from")
    if isinstance(imported, dict):
        attestation = imported.get("attestation")
        if isinstance(attestation, dict):
            _add_path(
                paths,
                _path_from_value(
                    root,
                    attestation.get("path"),
                    label=f"{stage} import attestation",
                ),
            )

    if stage == "chk2":
        judge = manifest.get("judge")
        if isinstance(judge, dict):
            artifact = judge.get("artifact")
            if isinstance(artifact, dict):
                _add_path(
                    paths,
                    _path_from_value(
                        root, artifact.get("path"), label="judge artifact"
                    ),
                )

    # Package installs mutate conda-meta.  Including both environments makes
    # environment-cache hits invalid immediately after conda/pip maintenance.
    train_prefix = Path(sys.executable).resolve().parents[1]
    _add_path(paths, train_prefix / "conda-meta")
    judge_env = os.environ.get("FOMC_RETRAIN_JUDGE_ENV", "fomc_judge_v2")
    _require("/" not in judge_env and judge_env not in {"", ".", ".."}, "unsafe judge env")
    _add_path(paths, train_prefix.parent / judge_env / "conda-meta")

    if scope == "launch":
        _add_path(paths, run_root / "adapters" / stage)
        _add_path(paths, run_root / "merged" / stage)
        _add_path(paths, run_root / "receipts" / f"{stage}.training.json")
        _add_path(paths, run_root / "receipts" / f"{stage}.merge.json")

    return sorted(paths, key=lambda item: str(item))


def _metadata_rows(path: Path) -> Iterable[dict[str, Any]]:
    lexical_root = path
    if not lexical_root.exists() and not lexical_root.is_symlink():
        yield {"path": str(lexical_root), "type": "missing"}
        return
    _require(not lexical_root.is_symlink(), f"cache dependency is a symlink: {path}")

    def row(candidate: Path, relative: str) -> dict[str, Any]:
        try:
            info = candidate.stat(follow_symlinks=False)
        except OSError as exc:
            raise PreflightCacheError(f"unable to stat dependency: {candidate}") from exc
        kind = "directory" if stat.S_ISDIR(info.st_mode) else "file"
        _require(kind == "directory" or stat.S_ISREG(info.st_mode), f"unsafe dependency: {candidate}")
        return {
            "path": str(lexical_root) if relative == "." else f"{lexical_root}/{relative}",
            "type": kind,
            "size": info.st_size,
            "mtime_ns": info.st_mtime_ns,
        }

    if lexical_root.is_file():
        yield row(lexical_root, ".")
        return
    _require(lexical_root.is_dir(), f"cache dependency is not regular: {path}")
    yield row(lexical_root, ".")
    for directory, dirnames, filenames in os.walk(lexical_root, followlinks=False):
        dirnames[:] = sorted(
            name for name in dirnames if name not in IGNORED_DIRECTORY_NAMES
        )
        filenames = sorted(
            name
            for name in filenames
            if not name.endswith(IGNORED_FILE_SUFFIXES)
        )
        base = Path(directory)
        for name in dirnames:
            candidate = base / name
            _require(not candidate.is_symlink(), f"cache dependency is a symlink: {candidate}")
            yield row(candidate, candidate.relative_to(lexical_root).as_posix())
        for name in filenames:
            candidate = base / name
            _require(not candidate.is_symlink(), f"cache dependency is a symlink: {candidate}")
            yield row(candidate, candidate.relative_to(lexical_root).as_posix())


def build_snapshot(
    repo_root: str | Path,
    run_manifest: str | Path,
    stage: str,
    scope: str,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest_path = _inside_repo(root, Path(run_manifest), label="run manifest")
    rows: list[dict[str, Any]] = []
    dependencies = dependency_paths(root, manifest_path, stage, scope)
    for dependency in dependencies:
        rows.extend(_metadata_rows(dependency))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "repo_root": str(root),
        "run_manifest": str(manifest_path),
        "stage": stage,
        "scope": scope,
        "dependency_count": len(dependencies),
        "rows": rows,
    }
    return {"payload": payload, "sha256": _sha256(payload)}


def cache_path(run_manifest: str | Path, stage: str, scope: str) -> Path:
    return Path(run_manifest).resolve().parent / "preflight_cache" / f"{stage}.{scope}.json"


def probe_cache(
    repo_root: str | Path,
    run_manifest: str | Path,
    stage: str,
    scope: str,
) -> dict[str, Any]:
    snapshot = build_snapshot(repo_root, run_manifest, stage, scope)
    destination = cache_path(run_manifest, stage, scope)
    result: dict[str, Any] = {
        "status": "ready",
        "hit": False,
        "reason": "cache_missing",
        "cache_path": str(destination),
        "snapshot_sha256": snapshot["sha256"],
        "payload": {},
    }
    if not destination.exists() and not destination.is_symlink():
        return result
    record = _load_json(destination, label="preflight cache")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "run_manifest": str(Path(run_manifest).resolve()),
        "stage": stage,
        "scope": scope,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        result["reason"] = "cache_identity_mismatch"
        return result
    if record.get("snapshot_sha256") != snapshot["sha256"]:
        result["reason"] = "directory_state_changed"
        return result
    payload = record.get("payload")
    if not isinstance(payload, dict):
        result["reason"] = "cache_payload_invalid"
        return result
    result.update(hit=True, reason="cache_hit", payload=payload)
    return result


def record_cache(
    repo_root: str | Path,
    run_manifest: str | Path,
    stage: str,
    scope: str,
    expected_snapshot_sha256: str,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = build_snapshot(repo_root, run_manifest, stage, scope)
    _require(
        snapshot["sha256"] == expected_snapshot_sha256,
        "directory state changed while the full preflight was running",
    )
    destination = cache_path(run_manifest, stage, scope)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _require(not destination.is_symlink(), f"unsafe preflight cache path: {destination}")
    record = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "run_manifest": str(Path(run_manifest).resolve()),
        "stage": stage,
        "scope": scope,
        "snapshot_sha256": snapshot["sha256"],
        "payload": dict(payload or {}),
    }
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(record, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "status": "recorded",
        "cache_path": str(destination),
        "snapshot_sha256": snapshot["sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "record"):
        command = commands.add_parser(name)
        command.add_argument("--run-manifest", type=Path, required=True)
        command.add_argument("--stage", choices=STAGES, required=True)
        command.add_argument("--scope", choices=SCOPES, required=True)
        if name == "record":
            command.add_argument("--snapshot-sha256", required=True)
            command.add_argument("--recovery-state")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "probe":
            result = probe_cache(
                args.repo_root, args.run_manifest, args.stage, args.scope
            )
        else:
            payload = {}
            if args.recovery_state is not None:
                payload["recovery_state"] = args.recovery_state
            result = record_cache(
                args.repo_root,
                args.run_manifest,
                args.stage,
                args.scope,
                args.snapshot_sha256,
                payload,
            )
    except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
        print(
            json.dumps(
                {"status": "blocked", "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PreflightCacheError",
    "build_snapshot",
    "cache_path",
    "dependency_paths",
    "probe_cache",
    "record_cache",
]
