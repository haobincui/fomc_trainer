from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from trl import TrlParser

import jobs.retrain_v2.start_chk2_deepseek_low600_resume as launcher
from jobs.retrain_v2.start_chk2_deepseek_low600_resume import (
    BOUND_ARTIFACT_LABELS,
    EXPECTED_CONFIG,
    EXPECTED_OUTPUT_DIR,
    EXPECTED_RESUME_CHECKPOINT,
    EXPECTED_RESUME_GLOBAL_STEP,
    EXPECTED_RUN_ROOT,
    EXPECTED_SESSION,
    EXPECTED_SOURCE_LOW_CONFIG,
    MIN_DISK_FREE_BYTES,
    PreflightError,
    _validate_artifact,
    _validate_config,
    _validate_resume_checkpoint,
    _validate_secret_environment,
    preflight,
)
from open_r1.configs import GRPOConfig, GRPOScriptArguments, LoraArguments, ModelConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
LOW_CONFIG = REPO_ROOT / EXPECTED_SOURCE_LOW_CONFIG
LOW600_CONFIG = REPO_ROOT / EXPECTED_CONFIG
LAUNCHER = (
    REPO_ROOT / "run/retrain_v2/start_chk2_cp200_deepseek_low600_resume_cp4.sh"
)
LAUNCH_MODULE = (
    REPO_ROOT / "jobs/retrain_v2/start_chk2_deepseek_low600_resume.py"
)


def _yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_low600_resume_config_parses_with_current_trl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Config parsing does not require a GPU and must remain testable while an
    # unrelated process occupies physical GPU0.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    script, training, model, peft = TrlParser(
        (GRPOScriptArguments, GRPOConfig, ModelConfig, LoraArguments)
    ).parse_args_and_config(
        args=[
            "--config",
            str(LOW600_CONFIG),
            "--use_cpu",
            "true",
            "--tf32",
            "false",
        ]
    )

    assert script.reward_funcs == ["grounded_analysis_v3_deepseek_low_timeout600"]
    assert script.judge_model == "deepseek-v4-flash"
    assert script.judge_timeout == 600
    assert script.judge_api_key_env == "DEEPSEEK_API_KEY"
    assert training.resume_from_checkpoint == EXPECTED_RESUME_CHECKPOINT.as_posix()
    assert training.output_dir == EXPECTED_OUTPUT_DIR.as_posix()
    assert training.max_completion_length == 4096
    assert training.generation_batch_size == 4
    assert model.load_in_4bit is True
    assert peft.peft_r == 32


def test_low600_changes_exactly_the_five_approved_fields() -> None:
    low = _yaml(LOW_CONFIG)
    low600 = _yaml(LOW600_CONFIG)
    changed = {key for key in low if low[key] != low600[key]}

    assert set(low600) == set(low)
    assert changed == {
        "reward_funcs",
        "judge_timeout",
        "peft_merged_model_path",
        "output_dir",
        "resume_from_checkpoint",
    }
    assert low600["reward_funcs"] == [
        "grounded_analysis_v3_deepseek_low_timeout600"
    ]
    assert low600["judge_timeout"] == 600
    assert low600["resume_from_checkpoint"] == EXPECTED_RESUME_CHECKPOINT.as_posix()
    assert low600["output_dir"] == EXPECTED_OUTPUT_DIR.as_posix()
    assert low600["peft_merged_model_path"] == (
        EXPECTED_RUN_ROOT / "merged/chk2"
    ).as_posix()


def test_low600_config_and_checkpoint_contracts_are_valid() -> None:
    payload = _validate_config(LOW600_CONFIG, LOW_CONFIG, REPO_ROOT)
    checkpoint = (REPO_ROOT / EXPECTED_RESUME_CHECKPOINT).resolve()
    state = _validate_resume_checkpoint(checkpoint)

    assert payload["judge_timeout"] == 600
    assert state["global_step"] == EXPECTED_RESUME_GLOBAL_STEP


def test_config_validation_rejects_timeout_drift(tmp_path: Path) -> None:
    payload = _yaml(LOW600_CONFIG)
    payload["judge_timeout"] = 599
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(PreflightError, match="contract mismatch"):
        _validate_config(changed, LOW_CONFIG, REPO_ROOT)


def test_checkpoint_validation_rejects_wrong_global_step(tmp_path: Path) -> None:
    checkpoint = (tmp_path / "checkpoint-4").resolve()
    checkpoint.mkdir()
    required = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
        "training_args.bin",
    }
    for name in required:
        (checkpoint / name).write_bytes(b"present")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 3}), encoding="utf-8"
    )

    with pytest.raises(PreflightError, match="global_step=4"):
        _validate_resume_checkpoint(checkpoint)


def test_bound_artifact_hash_is_required_and_verified(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("pinned", encoding="utf-8")
    digest = hashlib.sha256(b"pinned").hexdigest()

    path, fingerprint = _validate_artifact(
        tmp_path,
        {"path": "artifact.txt", "sha256": digest},
        label="artifact",
    )
    assert path == artifact.resolve()
    assert fingerprint["sha256"] == digest

    artifact.write_text("tampered", encoding="utf-8")
    with pytest.raises(PreflightError, match="fingerprint mismatch"):
        _validate_artifact(
            tmp_path,
            {"path": "artifact.txt", "sha256": digest},
            label="artifact",
        )


@pytest.mark.parametrize(
    "environment,match",
    [
        ({}, "DEEPSEEK_API_KEY"),
        ({"DEEPSEEK_API_KEY": "key", "OPENAI_LOG": "debug"}, "OPENAI_LOG"),
        (
            {"DEEPSEEK_API_KEY": "key", "HTTPX_LOG_LEVEL": "TRACE"},
            "HTTPX_LOG_LEVEL",
        ),
        (
            {"DEEPSEEK_API_KEY": "key", "SSLKEYLOGFILE": "/tmp/keys"},
            "SSLKEYLOGFILE",
        ),
    ],
)
def test_secret_environment_fails_closed(
    environment: dict[str, str], match: str
) -> None:
    with pytest.raises(PreflightError, match=match):
        _validate_secret_environment(environment)


def test_secret_environment_accepts_key_without_unsafe_logging() -> None:
    _validate_secret_environment(
        {
            "DEEPSEEK_API_KEY": "configured-but-never-persisted",
            "OPENAI_LOG": "info",
            "HTTPX_LOG_LEVEL": "warning",
        }
    )


def test_launcher_is_single_process_gpu1_and_versioned_low600() -> None:
    wrapper = LAUNCHER.read_text(encoding="utf-8")
    module = LAUNCH_MODULE.read_text(encoding="utf-8")

    assert "conda run --no-capture-output -n fomc_trainer" in wrapper
    assert "--execute" in wrapper
    assert "CUDA_VISIBLE_DEVICES=1" in module
    assert "unset RANK LOCAL_RANK WORLD_SIZE" in module
    assert '"jobs.train.train_grpo"' in module
    assert "torchrun" not in wrapper + module
    assert EXPECTED_SESSION in module
    assert "judge_timeout_seconds\": 600" in module
    assert "api_key_recorded\": False" in module
    assert "start_chk2_background.sh" not in wrapper + module
    assert "stage.sh" not in wrapper + module
    assert "judge.sh" not in wrapper + module
    assert "chk3" not in wrapper + module
    assert "chk4" not in wrapper + module


def test_launcher_binding_contract_hashes_every_required_artifact() -> None:
    module = LAUNCH_MODULE.read_text(encoding="utf-8")

    assert set(BOUND_ARTIFACT_LABELS) == {
        "config",
        "source_low_config",
        "resume_checkpoint",
        "merged_chk1",
        "dataset_overlay",
        "overlay_manifest",
        "reward_module",
        "reward_registry",
        "trainer",
        "benchmark_summary",
        "authorization",
        "launch_module",
        "launcher_wrapper",
    }
    assert "fingerprint_artifact_path(path)" in module
    assert "actual[\"sha256\"] != expected_sha" in module
    assert MIN_DISK_FREE_BYTES == 250 * 1024**3


def _mock_preflight_artifacts(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> Path:
    artifacts = {
        label: {"path": f"artifacts/{label}", "sha256": "a" * 64}
        for label in BOUND_ARTIFACT_LABELS
    }
    artifacts["config"]["path"] = EXPECTED_CONFIG.as_posix()
    artifacts["source_low_config"]["path"] = EXPECTED_SOURCE_LOW_CONFIG.as_posix()
    artifacts["resume_checkpoint"]["path"] = (
        EXPECTED_RESUME_CHECKPOINT.as_posix()
    )
    binding = {
        "schema_version": launcher.SCHEMA_VERSION,
        "session_name": EXPECTED_SESSION,
        "run_id": EXPECTED_RUN_ROOT.name,
        "artifacts": artifacts,
    }
    monkeypatch.setattr(launcher, "_load_object", lambda *_args, **_kwargs: binding)

    def fake_validate_artifact(
        root: Path, raw: object, *, label: str
    ) -> tuple[Path, dict[str, object]]:
        assert isinstance(raw, dict)
        return (root / str(raw["path"])).resolve(), {"sha256": "a" * 64}

    monkeypatch.setattr(launcher, "_validate_artifact", fake_validate_artifact)
    monkeypatch.setattr(
        launcher,
        "_validate_config",
        lambda *_args: {
            "resume_from_checkpoint": EXPECTED_RESUME_CHECKPOINT.as_posix()
        },
    )
    monkeypatch.setattr(launcher, "_validate_benchmark", lambda *_args: {})
    monkeypatch.setattr(launcher, "_validate_resume_checkpoint", lambda *_args: {})
    binding_path = repo_root / "binding.json"
    return binding_path


def test_preflight_rejects_existing_versioned_run_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding_path = _mock_preflight_artifacts(monkeypatch, tmp_path)
    (tmp_path / EXPECTED_RUN_ROOT).mkdir(parents=True)
    monkeypatch.setattr(launcher, "_session_exists", lambda _session: False)

    with pytest.raises(PreflightError, match="Fresh run directory already exists"):
        preflight(repo_root=tmp_path, binding_path=binding_path)


def test_preflight_rejects_existing_versioned_tmux_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding_path = _mock_preflight_artifacts(monkeypatch, tmp_path)
    monkeypatch.setattr(launcher, "_session_exists", lambda _session: True)

    with pytest.raises(PreflightError, match="tmux session already exists"):
        preflight(repo_root=tmp_path, binding_path=binding_path)
