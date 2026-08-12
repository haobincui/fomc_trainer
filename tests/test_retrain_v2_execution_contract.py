from __future__ import annotations

import copy
from pathlib import Path

import pytest

from jobs.retrain_v2 import execution_contract
from jobs.retrain_v2.execution_contract import (
    ExecutionContractError,
    build_execution_contract,
    canonical_stage_topology,
    validate_stage_topology,
    verify_execution_contract,
)


DIRECTORY_ROOTS = (
    "jobs/train",
    "src/open_r1",
    "jobs/retrain_v2",
    "run/retrain_v2",
)
FILE_ROOTS = (
    "run/check_retrain_v2_envs.sh",
    "run/retrain_v2_env_smoke.py",
    "requirements/retrain_v2_train.lock",
    "requirements/retrain_v2_train.freeze.txt",
    "requirements/retrain_v2_judge.lock",
    "requirements/retrain_v2_judge.freeze.txt",
)
TRAIN_FREEZE = "\n".join(
    (
        "bitsandbytes==0.48.2",
        "torch==2.10.0+cu128",
        "trl==1.2.0",
        "",
    )
)


def _build_tiny_repo(root: Path) -> None:
    for relative in DIRECTORY_ROOTS:
        directory = root / relative
        directory.mkdir(parents=True)
        (directory / "tracked.py").write_text(
            f"# source: {relative}\n", encoding="utf-8"
        )
    for relative in FILE_ROOTS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            TRAIN_FREEZE
            if relative == "requirements/retrain_v2_train.freeze.txt"
            else f"artifact={relative}\n"
        )
        path.write_text(content, encoding="utf-8")


def _observation(*, torch_version: str = "2.10.0+cu128") -> dict:
    return {
        "python_version": "3.10.9",
        "executable": "/opt/fomc_train_v2/bin/python",
        "distributions": [
            {"name": "Torch", "version": torch_version},
            {"name": "trl", "version": "1.2.0"},
            {"name": "bitsandbytes", "version": "0.48.2"},
        ],
        "torch_version": torch_version,
        "torch_cuda_version": "12.8",
    }


@pytest.fixture
def tiny_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _build_tiny_repo(tmp_path)
    monkeypatch.setattr(
        execution_contract,
        "_collect_runtime_observation",
        lambda: _observation(),
    )
    return tmp_path


def test_contract_layers_are_canonical_and_created_time_is_excluded(
    tiny_repo: Path,
) -> None:
    first = build_execution_contract(tiny_repo, created_at_utc="2026-08-03T00:00:00Z")
    second = build_execution_contract(tiny_repo, created_at_utc="2026-08-03T01:00:00Z")

    assert first["created_at_utc"] != second["created_at_utc"]
    assert first["source_bundle"] == second["source_bundle"]
    assert first["environment"] == second["environment"]
    assert first["contract_sha256"] == second["contract_sha256"]
    assert verify_execution_contract(first, tiny_repo)["status"] == "verified"

    source_paths = {item["path"] for item in first["source_bundle"]["payload"]["files"]}
    assert source_paths == {
        *(f"{root}/tracked.py" for root in DIRECTORY_ROOTS),
        *FILE_ROOTS,
    }
    distributions = first["environment"]["payload"]["distributions"]
    assert [item["name"] for item in distributions] == [
        "bitsandbytes",
        "torch",
        "trl",
    ]
    assert first["environment"]["payload"]["python"] == {
        "version": "3.10.9",
        "executable": "/opt/fomc_train_v2/bin/python",
    }


@pytest.mark.parametrize("mutation", ["add", "modify", "delete"])
def test_source_add_delete_or_modification_is_detected(
    tiny_repo: Path, mutation: str
) -> None:
    record = build_execution_contract(tiny_repo)
    tracked = tiny_repo / "jobs/train/tracked.py"
    if mutation == "add":
        (tracked.parent / "new_module.py").write_text("new = True\n", encoding="utf-8")
    elif mutation == "modify":
        tracked.write_text("changed = True\n", encoding="utf-8")
    else:
        tracked.unlink()

    with pytest.raises(ExecutionContractError, match="Source bundle drift"):
        verify_execution_contract(record, tiny_repo)


def test_only_declared_cache_artifacts_are_ignored(tiny_repo: Path) -> None:
    record = build_execution_contract(tiny_repo)
    source = tiny_repo / "src/open_r1"
    (source / "generated.pyc").write_bytes(b"ignored")
    pycache = source / "__pycache__"
    pycache.mkdir()
    (pycache / "anything.py").write_text("ignored = True\n", encoding="utf-8")
    pytest_cache = source / ".pytest_cache"
    pytest_cache.mkdir()
    (pytest_cache / "state").write_text("ignored\n", encoding="utf-8")

    assert verify_execution_contract(record, tiny_repo)["status"] == "verified"

    (source / ".mypy_cache").mkdir()
    (source / ".mypy_cache/state").write_text("not ignored\n", encoding="utf-8")
    with pytest.raises(ExecutionContractError, match="Source bundle drift"):
        verify_execution_contract(record, tiny_repo)


def test_nested_and_root_symlinks_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        execution_contract,
        "_collect_runtime_observation",
        lambda: _observation(),
    )
    nested_repo = tmp_path / "nested"
    _build_tiny_repo(nested_repo)
    outside = tmp_path / "outside.py"
    outside.write_text("outside = True\n", encoding="utf-8")
    (nested_repo / "jobs/train/link.py").symlink_to(outside)
    with pytest.raises(ExecutionContractError, match="must not contain symlinks"):
        build_execution_contract(nested_repo)

    root_repo = tmp_path / "root_link"
    _build_tiny_repo(root_repo)
    real_train = tmp_path / "real_train"
    (root_repo / "jobs/train").rename(real_train)
    (root_repo / "jobs/train").symlink_to(real_train, target_is_directory=True)
    with pytest.raises(ExecutionContractError, match="root jobs/train must not"):
        build_execution_contract(root_repo)


def test_fixed_file_symlink_path_escape_is_rejected(
    tiny_repo: Path, tmp_path: Path
) -> None:
    target = tmp_path / "outside-lock.txt"
    target.write_text("outside\n", encoding="utf-8")
    lock = tiny_repo / "requirements/retrain_v2_train.lock"
    lock.unlink()
    lock.symlink_to(target)

    with pytest.raises(ExecutionContractError, match="must not be a symlink"):
        build_execution_contract(tiny_repo)


def test_requirement_and_environment_drift_are_separate_failures(
    tiny_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = build_execution_contract(tiny_repo)
    requirement = tiny_repo / "requirements/retrain_v2_train.freeze.txt"
    requirement.write_text("changed==1\n", encoding="utf-8")
    with pytest.raises(ExecutionContractError, match="Requirement artifact drift"):
        verify_execution_contract(record, tiny_repo)

    requirement.write_text(TRAIN_FREEZE, encoding="utf-8")
    monkeypatch.setattr(
        execution_contract,
        "_collect_runtime_observation",
        lambda: _observation(torch_version="2.10.1+cu128"),
    )
    with pytest.raises(ExecutionContractError, match="environment drift"):
        verify_execution_contract(record, tiny_repo)


@pytest.mark.parametrize(
    ("freeze", "message"),
    [
        (
            "bitsandbytes==0.48.2\ntorch==2.10.0+cu128\n",
            "extra=.*trl",
        ),
        (
            TRAIN_FREEZE + "unexpected-package==1.0\n",
            "missing=.*unexpected-package",
        ),
        (
            TRAIN_FREEZE.replace("trl==1.2.0", "trl==1.2.1"),
            "version_mismatch=.*trl",
        ),
    ],
)
def test_contract_build_requires_exact_train_freeze_environment(
    tiny_repo: Path, freeze: str, message: str
) -> None:
    (tiny_repo / "requirements/retrain_v2_train.freeze.txt").write_text(
        freeze, encoding="utf-8"
    )

    with pytest.raises(ExecutionContractError, match=message):
        build_execution_contract(tiny_repo)


@pytest.mark.parametrize(
    "invalid_line",
    [
        "torch>=2.10.0",
        "--extra-index-url https://example.invalid",
        "trl==1.2.0; python_version >= '3.10'",
    ],
)
def test_contract_build_rejects_non_exact_train_freeze_entries(
    tiny_repo: Path, invalid_line: str
) -> None:
    (tiny_repo / "requirements/retrain_v2_train.freeze.txt").write_text(
        TRAIN_FREEZE + invalid_line + "\n", encoding="utf-8"
    )

    with pytest.raises(ExecutionContractError, match="Invalid exact train freeze"):
        build_execution_contract(tiny_repo)


def test_duplicate_canonical_distribution_names_fail_closed(
    tiny_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observation = _observation()
    observation["distributions"].append({"name": "bitsandbytes", "version": "0.48.2"})
    monkeypatch.setattr(
        execution_contract,
        "_collect_runtime_observation",
        lambda: observation,
    )
    with pytest.raises(ExecutionContractError, match="Duplicate canonical"):
        build_execution_contract(tiny_repo)


def test_record_layer_and_contract_hash_tampering_is_rejected(tiny_repo: Path) -> None:
    record = build_execution_contract(tiny_repo)
    tampered_layer = copy.deepcopy(record)
    tampered_layer["source_bundle"]["payload"]["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ExecutionContractError, match="layer hash mismatch"):
        verify_execution_contract(tampered_layer, tiny_repo)

    tampered_contract = copy.deepcopy(record)
    tampered_contract["contract_sha256"] = "0" * 64
    with pytest.raises(ExecutionContractError, match="contract hash mismatch"):
        verify_execution_contract(tampered_contract, tiny_repo)


def test_stage_topology_is_exact_and_return_values_are_independent() -> None:
    topology = canonical_stage_topology()
    assert topology["chk2"] == {
        "judge_gpu": 0,
        "launcher": "single_policy_with_judge",
        "policy_gpus": [1],
        "world_size": 1,
    }
    for stage_id in ("chk1", "chk3", "chk4"):
        assert topology[stage_id] == {
            "launcher": "ddp",
            "policy_gpus": [0, 1],
            "world_size": 2,
        }
        assert (
            validate_stage_topology(stage_id, topology[stage_id]) == topology[stage_id]
        )

    topology["chk1"]["world_size"] = 1
    assert canonical_stage_topology()["chk1"]["world_size"] == 2
    with pytest.raises(ExecutionContractError, match="topology mismatch"):
        validate_stage_topology("chk1", topology["chk1"])
