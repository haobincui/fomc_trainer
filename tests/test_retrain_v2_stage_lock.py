from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

import pytest

from jobs.retrain_v2.stage_lock import (
    StageLockError,
    canonical_stage_lock_path,
    require_inherited_stage_lock,
)


def _manifest(root: Path, run_id: str = "run_test") -> Path:
    path = root / "output/training/retrain_v2" / run_id / "run_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"run_id": run_id}) + "\n", encoding="utf-8")
    return path.resolve()


def test_lock_path_is_canonical_and_read_only(tmp_path: Path):
    manifest = _manifest(tmp_path)
    expected = manifest.parent / ".stage_locks/chk1.lock"
    assert canonical_stage_lock_path(manifest, "chk1", tmp_path) == expected
    assert not expected.parent.exists()


def test_lock_path_rejects_noncanonical_manifest(tmp_path: Path):
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text('{"run_id":"run_test"}\n', encoding="utf-8")
    with pytest.raises(StageLockError, match="outside"):
        canonical_stage_lock_path(manifest.resolve(), "chk1", tmp_path)


def test_inherited_lock_requires_exact_fd_inode_and_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest = _manifest(tmp_path)
    lock = canonical_stage_lock_path(manifest, "chk2", tmp_path)
    lock.parent.mkdir(mode=0o700)
    descriptor = os.open(lock, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setenv("FOMC_RETRAIN_STAGE_LOCK_FD", str(descriptor))
        monkeypatch.setenv("FOMC_RETRAIN_STAGE_LOCK_PATH", str(lock))
        monkeypatch.setenv("FOMC_RETRAIN_STAGE_LOCK_STAGE", "chk2")
        assert require_inherited_stage_lock(manifest, "chk2", tmp_path) == lock
        monkeypatch.setenv("FOMC_RETRAIN_STAGE_LOCK_STAGE", "chk1")
        with pytest.raises(StageLockError, match="identity"):
            require_inherited_stage_lock(manifest, "chk2", tmp_path)
    finally:
        os.close(descriptor)


def test_inherited_lock_rejects_wrong_file_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest = _manifest(tmp_path)
    lock = canonical_stage_lock_path(manifest, "chk3", tmp_path)
    lock.parent.mkdir(mode=0o700)
    lock.touch()
    other = tmp_path / "other.lock"
    descriptor = os.open(other, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        monkeypatch.setenv("FOMC_RETRAIN_STAGE_LOCK_FD", str(descriptor))
        monkeypatch.setenv("FOMC_RETRAIN_STAGE_LOCK_PATH", str(lock))
        monkeypatch.setenv("FOMC_RETRAIN_STAGE_LOCK_STAGE", "chk3")
        with pytest.raises(StageLockError, match="canonical lock file"):
            require_inherited_stage_lock(manifest, "chk3", tmp_path)
    finally:
        os.close(descriptor)
