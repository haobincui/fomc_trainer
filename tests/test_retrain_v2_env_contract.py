from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from jobs.retrain_v2.grpo_smoke import (
    DEFAULT_COMPLETION_LENGTH,
    DEFAULT_GENERATION_BATCH_SIZE,
    DEFAULT_PROMPT_LENGTH,
)
from jobs.retrain_v2.model_smoke import (
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_USE_LIGER,
)
from run import retrain_v2_env_smoke


ROOT = Path(__file__).resolve().parents[1]
TRAIN_LOCK = ROOT / "requirements" / "retrain_v2_train.lock"
JUDGE_LOCK = ROOT / "requirements" / "retrain_v2_judge.lock"
TRAIN_FREEZE = ROOT / "requirements" / "retrain_v2_train.freeze.txt"
JUDGE_FREEZE = ROOT / "requirements" / "retrain_v2_judge.freeze.txt"


def _locked_packages(path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        assert "==" in line, f"unpinned requirement in {path}: {line}"
        name, package_version = line.split("==", 1)
        normalized_name = re.sub(r"[-_.]+", "-", name).lower()
        packages[normalized_name] = package_version
    return packages


def test_retrain_v2_lock_separates_training_and_judge_dependencies() -> None:
    train = _locked_packages(TRAIN_LOCK)
    judge = _locked_packages(JUDGE_LOCK)

    assert train["torch"] == judge["torch"] == "2.10.0"
    assert train["transformers"] == "4.57.6"
    assert judge["transformers"] == "5.5.4"
    assert train["bitsandbytes"] == judge["bitsandbytes"] == "0.48.2"
    assert train["trl"] == "1.2.0"
    assert train["peft"] == "0.15.2"
    assert train["liger-kernel"] == "0.8.1"
    assert judge["vllm"] == "0.19.1"

    forbidden_in_train = {
        "deepspeed",
        "e2b",
        "e2b-code-interpreter",
        "flash-attn",
        "lighteval",
        "vllm",
        "xformers",
    }
    assert forbidden_in_train.isdisjoint(train)


def test_retrain_v2_shell_entrypoints_parse() -> None:
    scripts = [
        ROOT / "run" / "setup_retrain_v2_envs.sh",
        ROOT / "run" / "check_retrain_v2_envs.sh",
        ROOT / "run" / "retrain_v2" / "gpu_gate.sh",
        ROOT / "run" / "retrain_v2" / "resource_gate.sh",
        ROOT / "run" / "retrain_v2" / "stage.sh",
    ]
    subprocess.run(["bash", "-n", *(str(path) for path in scripts)], check=True)


def test_retrain_v2_python_smoke_entrypoint_parses() -> None:
    smoke = ROOT / "run" / "retrain_v2_env_smoke.py"
    result = subprocess.run(
        [sys.executable, str(smoke), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--role {train,judge}" in result.stdout
    assert "--skip-cuda" in result.stdout


def _distribution(name: str, package_version: str) -> SimpleNamespace:
    return SimpleNamespace(metadata={"Name": name}, version=package_version)


@pytest.mark.parametrize(
    ("installed", "expected_difference"),
    [
        ([('alpha', '1.0')], "bravo: expected 2.0, found missing"),
        (
            [('alpha', '1.0'), ('bravo', '2.0'), ('charlie', '3.0')],
            "charlie: expected absent, found 3.0",
        ),
        (
            [('alpha', '9.0'), ('bravo', '2.0')],
            "alpha: expected 1.0, found 9.0",
        ),
    ],
)
def test_freeze_contract_rejects_missing_extra_and_version_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installed: list[tuple[str, str]],
    expected_difference: str,
) -> None:
    freeze = tmp_path / "train.freeze.txt"
    freeze.write_text("alpha==1.0\nbravo==2.0\n", encoding="utf-8")
    monkeypatch.setitem(retrain_v2_env_smoke.ROLE_FREEZE_PATHS, "train", freeze)
    monkeypatch.setattr(
        retrain_v2_env_smoke,
        "distributions",
        lambda: [_distribution(name, value) for name, value in installed],
    )

    with pytest.raises(RuntimeError, match=re.escape(expected_difference)) as error:
        retrain_v2_env_smoke.check_freeze_contract("train")
    assert str(tmp_path) not in str(error.value)


def test_freeze_parser_rejects_duplicate_canonical_package(
    tmp_path: Path,
) -> None:
    freeze = tmp_path / "duplicate.freeze.txt"
    freeze.write_text("Demo_Package==1.0\ndemo-package==1.0\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Duplicate package in freeze: demo-package"):
        retrain_v2_env_smoke.parse_exact_freeze(freeze)


def test_freeze_contract_rejects_duplicate_installed_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze = tmp_path / "train.freeze.txt"
    freeze.write_text("demo-package==1.0\n", encoding="utf-8")
    monkeypatch.setitem(retrain_v2_env_smoke.ROLE_FREEZE_PATHS, "train", freeze)
    monkeypatch.setattr(
        retrain_v2_env_smoke,
        "distributions",
        lambda: [
            _distribution("Demo_Package", "1.0"),
            _distribution("demo-package", "2.0"),
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="demo-package: expected one installed version, found 1.0, 2.0",
    ) as error:
        retrain_v2_env_smoke.check_freeze_contract("train")
    assert str(tmp_path) not in str(error.value)


@pytest.mark.parametrize(
    "entry",
    [
        "demo @ https://example.invalid/demo.whl",
        "-e ./demo",
        "demo==1.0; python_version > '3'",
        "demo>=1.0",
    ],
)
def test_freeze_parser_rejects_non_exact_entries(
    tmp_path: Path, entry: str
) -> None:
    freeze = tmp_path / "invalid.freeze.txt"
    freeze.write_text(f"{entry}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Invalid freeze entry at line 1") as error:
        retrain_v2_env_smoke.parse_exact_freeze(freeze)
    assert entry not in str(error.value)


def test_repository_freezes_are_strict_exact_pin_sets() -> None:
    assert retrain_v2_env_smoke.parse_exact_freeze(TRAIN_FREEZE)
    assert retrain_v2_env_smoke.parse_exact_freeze(JUDGE_FREEZE)


def test_model_smoke_defaults_match_validated_sft_contract() -> None:
    assert DEFAULT_SEQUENCE_LENGTH == 4096
    assert DEFAULT_USE_LIGER is True
    smoke = ROOT / "jobs" / "retrain_v2" / "model_smoke.py"
    result = subprocess.run(
        [sys.executable, str(smoke), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--sequence-length" in result.stdout
    assert "--use-liger | --no-use-liger" in result.stdout
    assert "--distributed" in result.stdout


def test_grpo_smoke_defaults_match_validated_chk2_contract() -> None:
    assert DEFAULT_PROMPT_LENGTH == 2560
    assert DEFAULT_COMPLETION_LENGTH == 1024
    assert DEFAULT_GENERATION_BATCH_SIZE == 4
    smoke = ROOT / "jobs" / "retrain_v2" / "grpo_smoke.py"
    result = subprocess.run(
        [sys.executable, str(smoke), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--prompt-length" in result.stdout
    assert "--generation-batch-size" in result.stdout
