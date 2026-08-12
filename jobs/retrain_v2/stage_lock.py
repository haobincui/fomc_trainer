"""Canonical cross-process execution locks for retrain-v2 stages."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import stat
from pathlib import Path


STAGE_IDS = ("chk1", "chk2", "chk3", "chk4")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class StageLockError(RuntimeError):
    """Raised when a launcher does not hold the canonical stage lock."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StageLockError(message)


def canonical_stage_lock_path(
    run_manifest: str | Path, stage_id: str, repo_root: str | Path
) -> Path:
    """Return the only valid lock path after a read-only manifest-location check."""

    _require(stage_id in STAGE_IDS, f"invalid retrain-v2 stage: {stage_id}")
    root = Path(repo_root).resolve()
    manifest = Path(run_manifest)
    _require(manifest.is_absolute(), "run manifest path must be absolute")
    _require(not manifest.is_symlink(), "run manifest must not be a symlink")
    _require(manifest.is_file(), f"run manifest is missing: {manifest}")
    resolved_manifest = manifest.resolve(strict=True)
    try:
        payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageLockError(f"unable to parse run manifest: {resolved_manifest}") from exc
    _require(isinstance(payload, dict), "run manifest must contain an object")
    run_id = payload.get("run_id")
    _require(
        isinstance(run_id, str) and _SAFE_ID.fullmatch(run_id) is not None,
        "run manifest has an unsafe run_id",
    )
    expected_manifest = (
        root / "output" / "training" / "retrain_v2" / run_id / "run_manifest.json"
    ).resolve()
    _require(
        resolved_manifest == expected_manifest,
        "run manifest is outside its canonical retrain-v2 run directory",
    )
    return expected_manifest.parent / ".stage_locks" / f"{stage_id}.lock"


def require_inherited_stage_lock(
    run_manifest: str | Path, stage_id: str, repo_root: str | Path
) -> Path:
    """Prove this process inherited the launcher-held canonical exclusive lock."""

    expected = canonical_stage_lock_path(run_manifest, stage_id, repo_root)
    supplied_path = os.environ.get("FOMC_RETRAIN_STAGE_LOCK_PATH")
    supplied_stage = os.environ.get("FOMC_RETRAIN_STAGE_LOCK_STAGE")
    descriptor_text = os.environ.get("FOMC_RETRAIN_STAGE_LOCK_FD")
    _require(supplied_path == str(expected), "canonical stage lock path was not inherited")
    _require(supplied_stage == stage_id, "stage lock identity was not inherited")
    _require(
        isinstance(descriptor_text, str) and descriptor_text.isdecimal(),
        "stage lock file descriptor was not inherited",
    )
    descriptor = int(descriptor_text)
    _require(descriptor >= 3, "stage lock file descriptor is invalid")
    _require(expected.exists() and not expected.is_symlink(), "stage lock file is unsafe")
    try:
        descriptor_stat = os.fstat(descriptor)
        path_stat = expected.stat()
    except OSError as exc:
        raise StageLockError("stage lock file descriptor is not open") from exc
    _require(stat.S_ISREG(descriptor_stat.st_mode), "stage lock descriptor is not regular")
    _require(
        (descriptor_stat.st_dev, descriptor_stat.st_ino)
        == (path_stat.st_dev, path_stat.st_ino),
        "stage lock descriptor does not refer to the canonical lock file",
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise StageLockError("this process does not own the exclusive stage lock") from exc
    return expected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGE_IDS, required=True)
    parser.add_argument(
        "--verify-inherited",
        action="store_true",
        help="Also prove the caller inherited and owns the canonical lock.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path = (
            require_inherited_stage_lock(
                args.run_manifest, args.stage, args.repo_root
            )
            if args.verify_inherited
            else canonical_stage_lock_path(
                args.run_manifest, args.stage, args.repo_root
            )
        )
    except Exception as exc:  # noqa: BLE001 - fail-closed command boundary
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True))
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
