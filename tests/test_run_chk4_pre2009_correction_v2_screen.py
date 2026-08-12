from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobs.retrain_v2 import audit_chk4_pre2009_correction_v2 as audit
from jobs.retrain_v2 import probe_chk4_pre2009_correction_v2 as probe
from jobs.retrain_v2 import run_chk4_pre2009_correction_v2_screen as runner
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import seal_manifest


def _write_sealed(path: Path, value: dict[str, object]) -> dict[str, object]:
    sealed = seal_manifest(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sealed, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o400)
    return sealed


def test_probe_cli_has_no_direct_generation_entrypoint() -> None:
    with pytest.raises(SystemExit):
        probe._parser().parse_args(["run"])


def test_paths_are_fixed_under_versioned_run_root() -> None:
    assert probe.SCREEN_ROOT.name == "static_screening_attempt1_auth_binding_fix"
    assert audit.RECEIPT_ROOT == probe.SCREEN_ROOT / "receipts"
    assert runner.output_path(probe.SELECTION_STAGE, "baseline", None) == (
        runner.SCREEN_ROOT / "selection/baseline-exact-cp38"
    )
    assert runner.output_path(probe.BLIND_STAGE, "candidate", 4) == (
        runner.SCREEN_ROOT / "blind/checkpoint-4"
    )
    assert runner.authorization_path(probe.RETENTION_STAGE, "candidate", 6) == (
        runner.SCREEN_ROOT / "authorizations/retention-checkpoint-6.json"
    )


def test_retention_requires_blind_to_bind_exact_selection_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_sha = "1" * 64
    selection_path = tmp_path / "selection.json"
    _write_sealed(
        selection_path,
        {
            "schema_version": audit.SELECTION_RECEIPT_SCHEMA,
            "status": "selection_passed",
            "selected_checkpoint": 4,
            "manifest": {"sha256": manifest_sha},
            "implementation": {},
        },
    )
    blind_path = tmp_path / "blind.json"
    _write_sealed(
        blind_path,
        {
            "schema_version": audit.CONFIRMATION_RECEIPT_SCHEMA,
            "status": "confirmation_passed",
            "stage": probe.BLIND_STAGE,
            "locked_checkpoint": 4,
            "manifest": {"sha256": manifest_sha},
            "selection_receipt": {
                "path": str(selection_path.resolve()),
                "sha256": sha256_file(selection_path),
            },
            "implementation": {},
        },
    )
    monkeypatch.setattr(runner, "_verify_bound_implementation", lambda _value: None)
    monkeypatch.setattr(
        audit, "verify_confirmation_receipt", lambda **_kwargs: blind_path
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    value, binding = runner._blind_predecessor(
        stage=probe.RETENTION_STAGE,
        path=blind_path,
        selection_path=selection_path,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        locked_step=4,
    )
    assert value is not None
    assert binding is not None and binding["sha256"] == sha256_file(blind_path)

    other_selection = tmp_path / "other-selection.json"
    other_selection.write_text(
        selection_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(
        runner.CorrectionV2ScreenLaunchError,
        match="exact locked selection",
    ):
        runner._blind_predecessor(
            stage=probe.RETENTION_STAGE,
            path=blind_path,
            selection_path=other_selection,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha,
            locked_step=4,
        )


def test_self_sealed_fake_selection_passed_is_not_authority(tmp_path: Path) -> None:
    manifest_sha = "2" * 64
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    forged_path = tmp_path / "forged-selection.json"
    _write_sealed(
        forged_path,
        {
            "schema_version": audit.SELECTION_RECEIPT_SCHEMA,
            "status": "selection_passed",
            "selected_checkpoint": 2,
            "manifest": {"sha256": manifest_sha},
            "implementation": {},
        },
    )
    with pytest.raises(
        runner.CorrectionV2ScreenLaunchError,
        match="implementation inventory",
    ):
        runner._selection_predecessor(
            stage=probe.BLIND_STAGE,
            role="baseline",
            checkpoint_step=None,
            path=forged_path,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha,
        )


def test_execute_launch_passes_minimal_authorization_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "selection-baseline.json"
    observed = {
        "integrity": {"payload_sha256": "a" * 64},
        "status": "authorized",
    }
    auth_path.write_text(json.dumps(observed) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(runner, "authorization_path", lambda *_args: auth_path)
    monkeypatch.setattr(runner, "validate_manifest_integrity", lambda _value: None)
    monkeypatch.setattr(runner, "authorization_payload", lambda **_kwargs: observed)
    monkeypatch.setattr(
        runner.probe,
        "run_panel",
        lambda **kwargs: captured.update(kwargs) or {"status": "complete"},
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    result = runner.launch(
        repo_root=tmp_path,
        manifest_path=manifest_path,
        manifest_sha256="b" * 64,
        stage=probe.SELECTION_STAGE,
        role="baseline",
        checkpoint_step=None,
        selection_receipt_path=None,
        blind_receipt_path=None,
        execute=True,
    )

    assert result == {"status": "complete"}
    binding = captured["authorization_binding"]
    assert isinstance(binding, dict)
    assert set(binding) == {"path", "sha256", "payload_sha256"}
    assert "bytes" not in binding
