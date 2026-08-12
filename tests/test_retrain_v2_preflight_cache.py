from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobs.retrain_v2.preflight_cache import (
    PreflightCacheError,
    build_snapshot,
    probe_cache,
    record_cache,
)


def _write(path: Path, value: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _tiny_run(tmp_path: Path) -> Path:
    _write(tmp_path / "jobs/retrain_v2/source.py")
    _write(tmp_path / "run/retrain_v2/stage.sh")
    _write(tmp_path / "requirements/train.freeze.txt")
    _write(tmp_path / "configs/dag.yaml")
    _write(tmp_path / "configs/chk2.yaml")
    _write(tmp_path / "models/chk1/model.safetensors", "parent")
    _write(tmp_path / "models/judge/model.safetensors", "judge")
    _write(tmp_path / "dataset/chk2/train.jsonl", "{}\n")
    _write(tmp_path / "dataset/release/audit.json", "{}")
    run_root = tmp_path / "output/training/retrain_v2/run_test"
    run_root.mkdir(parents=True)
    manifest = {
        "dag": {"path": str(tmp_path / "configs/dag.yaml")},
        "execution_contract": {
            "source_bundle": {
                "payload": {
                    "directory_roots": ["jobs/retrain_v2", "run/retrain_v2"],
                    "file_roots": ["requirements/train.freeze.txt"],
                }
            }
        },
        "data_releases": {"base": {"path": str(tmp_path / "dataset/release")}},
        "judge": {"artifact": {"path": str(tmp_path / "models/judge")}},
        "stages": {
            "chk1": {"artifact": {"path": str(tmp_path / "models/chk1")}},
            "chk2": {
                "parent": "chk1",
                "config_template": {"path": str(tmp_path / "configs/chk2.yaml")},
                "resolved_config": {"path": str(tmp_path / "configs/chk2.yaml")},
                "data_binding": {
                    "dataset_path": str(tmp_path / "dataset/chk2"),
                    "release_role": "base",
                },
            },
        },
    }
    manifest_path = run_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_cache_hits_until_a_relevant_directory_changes(tmp_path: Path) -> None:
    manifest = _tiny_run(tmp_path)
    first = probe_cache(tmp_path, manifest, "chk2", "immutable")
    assert first["hit"] is False
    record_cache(
        tmp_path,
        manifest,
        "chk2",
        "immutable",
        first["snapshot_sha256"],
        {"checked": True},
    )
    hit = probe_cache(tmp_path, manifest, "chk2", "immutable")
    assert hit["hit"] is True
    assert hit["payload"] == {"checked": True}

    _write(tmp_path / "dataset/chk2/train.jsonl", "{}\n{}\n")
    changed = probe_cache(tmp_path, manifest, "chk2", "immutable")
    assert changed["hit"] is False
    assert changed["reason"] == "directory_state_changed"


def test_new_source_file_invalidates_the_directory_snapshot(tmp_path: Path) -> None:
    manifest = _tiny_run(tmp_path)
    snapshot = build_snapshot(tmp_path, manifest, "chk2", "immutable")
    record_cache(
        tmp_path,
        manifest,
        "chk2",
        "immutable",
        snapshot["sha256"],
    )
    _write(tmp_path / "jobs/retrain_v2/new_source.py")
    assert probe_cache(tmp_path, manifest, "chk2", "immutable")["hit"] is False


def test_launch_cache_tracks_partial_adapter_output(tmp_path: Path) -> None:
    manifest = _tiny_run(tmp_path)
    first = probe_cache(tmp_path, manifest, "chk2", "launch")
    record_cache(
        tmp_path,
        manifest,
        "chk2",
        "launch",
        first["snapshot_sha256"],
        {"recovery_state": "fresh_train"},
    )
    assert probe_cache(tmp_path, manifest, "chk2", "launch")["hit"] is True
    _write(manifest.parent / "adapters/chk2/reward.jsonl", "{}\n")
    assert probe_cache(tmp_path, manifest, "chk2", "launch")["hit"] is False


def test_record_fails_if_inputs_changed_during_full_preflight(tmp_path: Path) -> None:
    manifest = _tiny_run(tmp_path)
    snapshot = build_snapshot(tmp_path, manifest, "chk2", "immutable")
    _write(tmp_path / "configs/chk2.yaml", "changed")
    with pytest.raises(PreflightCacheError, match="changed while"):
        record_cache(
            tmp_path,
            manifest,
            "chk2",
            "immutable",
            snapshot["sha256"],
        )
