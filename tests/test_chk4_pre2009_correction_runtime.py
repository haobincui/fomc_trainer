from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jobs.retrain_v2 import materialize_chk4_pre2009_correction_release as materializer
from open_r1.trainer import dataset_release


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch):
    root = tmp_path / dataset_release.CHK4_PRE2009_CORRECTION_RELEASE_ID
    dataset = root / "decision_sft"
    _write(dataset / "train.jsonl", "{}\n")
    _write(dataset / "validation.jsonl", "{}\n")
    _write(root / "manifests/sampler_schedule.jsonl", "{}\n")
    parent = tmp_path / "parent"
    parent.mkdir()
    manifest = root / "release_manifest.json"
    payload = {
        "schema_version": dataset_release.CHK4_PRE2009_CORRECTION_RELEASE_SCHEMA,
        "release_id": dataset_release.CHK4_PRE2009_CORRECTION_RELEASE_ID,
        "dataset_role": dataset_release.CHK4_PRE2009_CORRECTION_SFT_ROLE,
    }
    _write(manifest, json.dumps(payload, sort_keys=True) + "\n")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    sampler = {
        "type": dataset_release.CHK4_PRE2009_CORRECTION_SAMPLER_TYPE,
        "schedule_path": "manifests/sampler_schedule.jsonl",
        "schedule_sha256": "1" * 64,
        "train_path": "decision_sft/train.jsonl",
        "train_sha256": "2" * 64,
    }
    verified = {
        **payload,
        "parent_release": {
            "path": str(parent),
            "manifest_sha256": "3" * 64,
        },
        "training_role": {
            "dataset_path": "decision_sft",
            "completion_only_loss": True,
            "max_length": 3072,
            "parent_role": "selected_pre2009_cp38_exact_merged",
            "sampler_contract": "sampler_contract",
        },
        "sampler_contract": sampler,
        "heldout_contract": {
            "holdout_scope": "correction_stage_only",
            "sample_ids": list(materializer.HELDOUT_SAMPLE_IDS),
        },
    }
    monkeypatch.setattr(materializer, "verify_release", lambda *a, **k: verified)
    monkeypatch.setattr(
        dataset_release,
        "verify_chk4_pre2009_augmented_release",
        lambda **kwargs: {"tokenizer_binding": {"bundle_sha256": "4" * 64}},
    )
    return root, dataset, manifest, digest


def test_correction_runtime_dispatch_requires_external_pin(tmp_path, monkeypatch):
    root, dataset, manifest, digest = _fixture(tmp_path, monkeypatch)
    result = dataset_release.verify_chk4_release_for_role(
        dataset_dir=dataset,
        manifest_path=manifest,
        expected_manifest_sha256=digest,
        dataset_role=dataset_release.CHK4_PRE2009_CORRECTION_SFT_ROLE,
        system_prompt=dataset_release.CHK4_STUDENT_SYSTEM_PROMPT,
        model_path=tmp_path / "model",
    )
    assert result["schema_version"] == "chk4-pre2009-correction-runtime-binding-v1"
    assert result["split_files"]["train"] == (dataset / "train.jsonl").resolve()
    assert result["test_verified_but_not_loaded"] is True
    assert result["sampler_contract"]["type"] == "manifest_fixed_schedule_v3"
    assert result["heldout_contract"]["holdout_scope"] == "correction_stage_only"

    with pytest.raises(
        dataset_release.DatasetReleaseValidationError, match="disagrees"
    ):
        dataset_release.verify_chk4_release_for_role(
            dataset_dir=dataset,
            manifest_path=manifest,
            expected_manifest_sha256="0" * 64,
            dataset_role=dataset_release.CHK4_PRE2009_CORRECTION_SFT_ROLE,
            system_prompt=dataset_release.CHK4_STUDENT_SYSTEM_PROMPT,
            model_path=tmp_path / "model",
        )


def test_correction_runtime_rejects_role_or_prompt_drift(tmp_path, monkeypatch):
    _root, dataset, manifest, digest = _fixture(tmp_path, monkeypatch)
    with pytest.raises(dataset_release.DatasetReleaseValidationError, match="requires"):
        dataset_release.verify_chk4_pre2009_correction_release(
            dataset_dir=dataset,
            manifest_path=manifest,
            expected_manifest_sha256=digest,
            dataset_role="decision_sft_pre2009_balanced",
            system_prompt=dataset_release.CHK4_STUDENT_SYSTEM_PROMPT,
            model_path=tmp_path / "model",
        )
    with pytest.raises(dataset_release.DatasetReleaseValidationError, match="system_prompt"):
        dataset_release.verify_chk4_pre2009_correction_release(
            dataset_dir=dataset,
            manifest_path=manifest,
            expected_manifest_sha256=digest,
            dataset_role=dataset_release.CHK4_PRE2009_CORRECTION_SFT_ROLE,
            system_prompt="drift",
            model_path=tmp_path / "model",
        )
