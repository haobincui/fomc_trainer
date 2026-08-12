from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from jobs.retrain_v2 import judge_attestation
from jobs.retrain_v2.judge_attestation import (
    JudgeAttestationError,
    fetch_service_identity,
    record_judge_attestation,
    verify_judge_attestations,
)


def _contract(root: Path) -> dict:
    model = root / "models/Qwen3.5-9B"
    return {
        "served_model_name": "Qwen3.5-9B",
        "url": "http://127.0.0.1:8000/v1/chat/completions",
        "timeout": 180,
        "max_retries": 3,
        "backoff_seconds": 1.0,
        "tokenizer_path": "models/Qwen3.5-9B",
        "max_model_len": 8192,
        "max_completion_tokens": 2048,
        "candidate_reserve_tokens": 1536,
        "boundary_margin_tokens": 32,
        "artifact": {
            "path": str(model),
            "kind": "directory",
            "sha256": "a" * 64,
            "file_count": 1,
            "total_bytes": 2,
            "algorithm": "test-tree-sha256",
        },
    }


@pytest.fixture
def attestation_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model = tmp_path / "models/Qwen3.5-9B"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    run_root = tmp_path / "output/training/retrain_v2/run-a"
    run_root.mkdir(parents=True)
    manifest = run_root / "run_manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 2, "run_id": "run-a", "phase": "upstream_ready"}),
        encoding="utf-8",
    )
    verified = _contract(tmp_path)
    monkeypatch.setattr(
        judge_attestation,
        "verify_judge_artifact",
        lambda *_args, **_kwargs: verified,
    )
    lock_path = run_root / ".stage_locks/chk2.lock"
    lock_path.parent.mkdir(mode=0o700)
    descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    monkeypatch.setenv("FOMC_RETRAIN_STAGE_LOCK_PATH", str(lock_path))
    monkeypatch.setenv("FOMC_RETRAIN_STAGE_LOCK_STAGE", "chk2")
    monkeypatch.setenv("FOMC_RETRAIN_STAGE_LOCK_FD", str(descriptor))
    try:
        yield tmp_path, manifest, model, verified
    finally:
        os.close(descriptor)


def _identity(model: Path, *, owned_by: str = "vllm") -> dict:
    stable = {
        "endpoint": "http://127.0.0.1:8000",
        "id": "Qwen3.5-9B",
        "object": "model",
        "owned_by": owned_by,
        "root": str(model),
        "parent": None,
        "max_model_len": 8192,
    }
    return {
        "stable_fields": stable,
        "stable_fields_sha256": judge_attestation._canonical_sha256(stable),
        "server_start_time": judge_attestation._SERVER_START_TIME_NOTE,
        "excluded_unstable_fields": ["created"],
    }


def _health(model: Path, *, parity: bool = True) -> dict:
    return {
        "status": "ready",
        "url": "http://127.0.0.1:8000/v1/chat/completions",
        "model": "Qwen3.5-9B",
        "attempts": 1,
        "rubric_keys": ["data_fidelity"],
        "weight_attested": True,
        "loaded_model_root": str(model),
        "max_model_len": 8192,
        "max_completion_tokens": 2048,
        "golden_prompt_tokens": 97,
        "tokenizer_parity": parity,
    }


def _record(
    manifest: Path,
    root: Path,
    model: Path,
    *,
    phase: str,
    timestamp: str,
    owned_by: str = "vllm",
    parity: bool = True,
) -> dict:
    return record_judge_attestation(
        manifest,
        phase=phase,
        repo_root=root,
        recorded_at_utc=timestamp,
        identity_fetcher=lambda **_kwargs: _identity(model, owned_by=owned_by),
        health_checker=lambda **_kwargs: _health(model, parity=parity),
    )


def test_records_exclusive_pre_post_and_verifies_binding(attestation_repo) -> None:
    root, manifest, model, _verified = attestation_repo
    pre = _record(
        manifest,
        root,
        model,
        phase="pre",
        timestamp="2026-08-03T10:00:00Z",
    )
    post = _record(
        manifest,
        root,
        model,
        phase="post",
        timestamp="2026-08-03T11:00:00Z",
    )

    pre_path = Path(pre["path"])
    post_path = Path(post["path"])
    assert pre_path.name == "judge.pre.json"
    assert post_path.name == "judge.post.json"
    assert post["pre_binding"] == {
        "path": pre_path.relative_to(root).as_posix(),
        "file_sha256": judge_attestation._sha256_file(pre_path),
        "canonical_payload_sha256": pre["canonical_payload_sha256"],
    }
    assert pre["token_id_parity"] == {
        "verified": True,
        "method": "exact local/server token-ID sequence via judge_health",
        "golden_prompt_tokens": 97,
        "max_model_len_verified": True,
    }
    assert pre["service_identity"]["excluded_unstable_fields"] == ["created"]
    assert "unavailable" in pre["service_identity"]["server_start_time"]
    verified = verify_judge_attestations(manifest, repo_root=root)
    assert verified["status"] == "verified"
    assert verified["pre_attestation_sha256"] == pre["canonical_payload_sha256"]
    assert verified["post_attestation_sha256"] == post["canonical_payload_sha256"]


def test_pre_is_exclusive_and_never_overwritten(attestation_repo) -> None:
    root, manifest, model, _verified = attestation_repo
    first = _record(
        manifest,
        root,
        model,
        phase="pre",
        timestamp="2026-08-03T10:00:00Z",
    )
    original = Path(first["path"]).read_bytes()
    with pytest.raises(JudgeAttestationError, match="already exists"):
        _record(
            manifest,
            root,
            model,
            phase="pre",
            timestamp="2026-08-03T10:01:00Z",
        )
    assert Path(first["path"]).read_bytes() == original


def test_record_requires_the_inherited_chk2_stage_lock(
    attestation_repo, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest, model, _verified = attestation_repo
    monkeypatch.delenv("FOMC_RETRAIN_STAGE_LOCK_FD")

    with pytest.raises(JudgeAttestationError, match="chk2 stage lock"):
        _record(
            manifest,
            root,
            model,
            phase="pre",
            timestamp="2026-08-03T10:00:00Z",
        )
    assert not (manifest.parent / "attestations/judge.pre.json").exists()


def test_post_requires_pre_and_rejects_service_identity_drift(
    attestation_repo,
) -> None:
    root, manifest, model, _verified = attestation_repo
    with pytest.raises(JudgeAttestationError, match="pre attestation does not exist"):
        _record(
            manifest,
            root,
            model,
            phase="post",
            timestamp="2026-08-03T11:00:00Z",
        )

    _record(
        manifest,
        root,
        model,
        phase="pre",
        timestamp="2026-08-03T10:00:00Z",
    )
    with pytest.raises(JudgeAttestationError, match="identity differs from pre"):
        _record(
            manifest,
            root,
            model,
            phase="post",
            timestamp="2026-08-03T11:00:00Z",
            owned_by="different-service",
        )
    assert not (manifest.parent / "attestations/judge.post.json").exists()


def test_post_rejects_timestamp_before_pre(attestation_repo) -> None:
    root, manifest, model, _verified = attestation_repo
    _record(
        manifest,
        root,
        model,
        phase="pre",
        timestamp="2026-08-03T10:00:00Z",
    )
    with pytest.raises(JudgeAttestationError, match="predates pre"):
        _record(
            manifest,
            root,
            model,
            phase="post",
            timestamp="2026-08-03T09:59:59Z",
        )


def test_refuses_symlinked_attestation_directory(attestation_repo) -> None:
    root, manifest, model, _verified = attestation_repo
    outside = root / "outside"
    outside.mkdir()
    (manifest.parent / "attestations").symlink_to(outside, target_is_directory=True)

    with pytest.raises(JudgeAttestationError, match="symlink"):
        _record(
            manifest,
            root,
            model,
            phase="pre",
            timestamp="2026-08-03T10:00:00Z",
        )
    assert not (outside / "judge.pre.json").exists()


def test_health_must_prove_exact_token_id_parity(attestation_repo) -> None:
    root, manifest, model, _verified = attestation_repo
    with pytest.raises(JudgeAttestationError, match="Token-ID parity"):
        _record(
            manifest,
            root,
            model,
            phase="pre",
            timestamp="2026-08-03T10:00:00Z",
            parity=False,
        )
    assert not (manifest.parent / "attestations/judge.pre.json").exists()


def test_verify_detects_tampering_even_with_recomputed_payload_hash(
    attestation_repo,
) -> None:
    root, manifest, model, _verified = attestation_repo
    pre = _record(
        manifest,
        root,
        model,
        phase="pre",
        timestamp="2026-08-03T10:00:00Z",
    )
    path = Path(pre["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["token_id_parity"]["verified"] = False
    without_hash = {
        key: value
        for key, value in payload.items()
        if key != "canonical_payload_sha256"
    }
    payload["canonical_payload_sha256"] = judge_attestation._canonical_sha256(
        without_hash
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(JudgeAttestationError, match="Token-ID parity"):
        verify_judge_attestations(manifest, repo_root=root, require_post=False)


def test_post_revalidates_pre_contract_before_observing_service(
    attestation_repo,
) -> None:
    root, manifest, model, _verified = attestation_repo
    pre = _record(
        manifest,
        root,
        model,
        phase="pre",
        timestamp="2026-08-03T10:00:00Z",
    )
    path = Path(pre["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["immutable_judge_contract"]["boundary_margin_tokens"] = 31
    without_hash = {
        key: value
        for key, value in payload.items()
        if key != "canonical_payload_sha256"
    }
    payload["canonical_payload_sha256"] = judge_attestation._canonical_sha256(
        without_hash
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(JudgeAttestationError, match="judge contract drifted"):
        _record(
            manifest,
            root,
            model,
            phase="post",
            timestamp="2026-08-03T11:00:00Z",
        )
    assert not (manifest.parent / "attestations/judge.post.json").exists()


def test_verify_rejects_attestation_directory_replaced_by_symlink(
    attestation_repo,
) -> None:
    root, manifest, model, _verified = attestation_repo
    _record(
        manifest,
        root,
        model,
        phase="pre",
        timestamp="2026-08-03T10:00:00Z",
    )
    attestations = manifest.parent / "attestations"
    moved = root / "moved-attestations"
    attestations.rename(moved)
    attestations.symlink_to(moved, target_is_directory=True)

    with pytest.raises(JudgeAttestationError, match="symlink"):
        verify_judge_attestations(manifest, repo_root=root, require_post=False)


def test_verify_survives_unrelated_mutable_manifest_fields(attestation_repo) -> None:
    root, manifest, model, _verified = attestation_repo
    _record(
        manifest,
        root,
        model,
        phase="pre",
        timestamp="2026-08-03T10:00:00Z",
    )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["phase"] = "complete"
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")

    assert (
        verify_judge_attestations(manifest, repo_root=root, require_post=False)[
            "status"
        ]
        == "verified"
    )


def test_fetch_service_identity_uses_stable_model_card_fields_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "Qwen3.5-9B"
    model.mkdir()
    responses = []
    for created in (100, 999):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "data": [
                {
                    "id": "Qwen3.5-9B",
                    "object": "model",
                    "created": created,
                    "owned_by": "vllm",
                    "root": str(model),
                    "parent": None,
                    "max_model_len": 8192,
                }
            ]
        }
        responses.append(response)
    request = Mock(side_effect=responses)
    monkeypatch.setattr(judge_attestation.requests, "get", request)

    kwargs = {
        "url": "http://127.0.0.1:8000/v1/chat/completions",
        "model": "Qwen3.5-9B",
        "timeout": 5,
        "expected_model_root": model,
    }
    first = fetch_service_identity(**kwargs)
    second = fetch_service_identity(**kwargs)
    assert first == second
    assert "created" not in first["stable_fields"]
    assert first["excluded_unstable_fields"] == ["created"]
    assert request.call_count == 2
