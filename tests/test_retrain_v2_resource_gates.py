from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GPU_GATE = ROOT / "run/retrain_v2/gpu_gate.sh"
RESOURCE_GATE = ROOT / "run/retrain_v2/resource_gate.sh"
ENV_CHECK = ROOT / "run/check_retrain_v2_envs.sh"
STAGE = ROOT / "run/retrain_v2/stage.sh"
TRAIN_FREEZE = ROOT / "requirements/retrain_v2_train.freeze.txt"
JUDGE_FREEZE = ROOT / "requirements/retrain_v2_judge.freeze.txt"


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _fake_path(tmp_path: Path, commands: dict[str, str]) -> str:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, source in commands.items():
        _write_executable(bin_dir / name, source)
    return f"{bin_dir}:{os.environ['PATH']}"


def _run(script: Path, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _gpu_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = _fake_path(
        tmp_path,
        {
            "nvidia-smi": """#!/usr/bin/env bash
if [[ "$*" == *"--query-compute-apps=pid"* ]]; then
  [[ -n "${FAKE_GPU_PID:-}" ]] && echo "${FAKE_GPU_PID}"
  exit 0
fi
if [[ "$*" == *"--query-gpu=memory.used,memory.free,utilization.gpu"* ]]; then
  echo "${FAKE_GPU_USED_MIB:-5729}, ${FAKE_GPU_FREE_MIB:-18321}, ${FAKE_GPU_UTILIZATION:-61}"
  exit 0
fi
exit 2
""",
        },
    )
    return env


def test_gpu_gate_accepts_measured_shared_baseline_and_stricter_thresholds(
    tmp_path: Path,
) -> None:
    env = _gpu_env(tmp_path)
    assert _run(GPU_GATE, "0", env=env).returncode == 0

    env.update(
        {
            "FOMC_RETRAIN_GPU0_MAX_PREEXISTING_USED_MIB": "6000",
            "FOMC_RETRAIN_GPU0_MIN_FREE_MIB": "18100",
            "FAKE_GPU_USED_MIB": "5900",
            "FAKE_GPU_FREE_MIB": "18200",
            "FAKE_GPU_UTILIZATION": "100",
        }
    )
    assert _run(GPU_GATE, "0", env=env).returncode == 0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("FOMC_RETRAIN_GPU0_MAX_PREEXISTING_USED_MIB", "6145"),
        ("FOMC_RETRAIN_GPU0_MAX_PREEXISTING_USED_MIB", "not-an-integer"),
        ("FOMC_RETRAIN_GPU0_MAX_PREEXISTING_USED_MIB", "6000.5"),
        ("FOMC_RETRAIN_GPU0_MAX_PREEXISTING_USED_MIB", "1000000"),
        ("FOMC_RETRAIN_GPU0_MIN_FREE_MIB", "17999"),
        ("FOMC_RETRAIN_GPU0_MIN_FREE_MIB", "-1"),
        ("FOMC_RETRAIN_GPU0_MIN_FREE_MIB", "not-an-integer"),
        ("FOMC_RETRAIN_GPU0_MIN_FREE_MIB", "18000.5"),
    ],
)
def test_gpu_gate_rejects_weaker_or_invalid_overrides(
    tmp_path: Path, name: str, value: str
) -> None:
    env = _gpu_env(tmp_path)
    env[name] = value
    result = _run(GPU_GATE, "0", env=env)
    assert result.returncode == 2
    assert "ERROR:" in result.stderr


def test_gpu_gate_rejects_duplicates_but_allows_existing_processes_within_baseline(
    tmp_path: Path,
) -> None:
    env = _gpu_env(tmp_path)
    duplicate = _run(GPU_GATE, "0", "0", env=env)
    assert duplicate.returncode == 2
    assert "duplicate GPU ID" in duplicate.stderr

    env["FAKE_GPU_PID"] = "4242"
    occupied = _run(GPU_GATE, "0", env=env)
    assert occupied.returncode == 0
    assert "existing_pids=4242" in occupied.stdout


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("FAKE_GPU_USED_MIB", "6145", "already uses"),
        ("FAKE_GPU_FREE_MIB", "17999", "shared baseline requires"),
        ("FAKE_GPU_UTILIZATION", "101", "invalid utilization"),
    ],
)
def test_gpu_gate_blocks_growth_beyond_shared_baseline(
    tmp_path: Path, name: str, value: str, message: str
) -> None:
    env = _gpu_env(tmp_path)
    env[name] = value
    result = _run(GPU_GATE, "0", env=env)
    assert result.returncode == 2
    assert message in result.stderr


def _disk_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = _fake_path(
        tmp_path,
        {
            "df": """#!/usr/bin/env bash
echo "Filesystem 1024-blocks Used Available Capacity Mounted on"
echo "/dev/fake 500000000 1 ${FAKE_AVAILABLE_KIB:-400000000} 1% /workspace"
""",
        },
    )
    return env


def test_resource_gate_accepts_default_and_higher_minimum(tmp_path: Path) -> None:
    env = _disk_env(tmp_path)
    assert _run(RESOURCE_GATE, env=env).returncode == 0
    env["FOMC_RETRAIN_MIN_DISK_GIB"] = "300"
    assert _run(RESOURCE_GATE, env=env).returncode == 0


def test_merge_resource_gate_has_independent_150_gib_floor(tmp_path: Path) -> None:
    env = _disk_env(tmp_path)
    assert _run(RESOURCE_GATE, "--merge", env=env).returncode == 0
    env["FOMC_RETRAIN_MIN_MERGE_DISK_GIB"] = "200"
    assert _run(RESOURCE_GATE, "--merge", env=env).returncode == 0
    env["FOMC_RETRAIN_MIN_MERGE_DISK_GIB"] = "149"
    result = _run(RESOURCE_GATE, "--merge", env=env)
    assert result.returncode == 2
    assert "cannot weaken the 150 GiB baseline" in result.stderr


def test_resource_gate_rejects_unknown_arguments(tmp_path: Path) -> None:
    result = _run(RESOURCE_GATE, "--unknown", env=_disk_env(tmp_path))
    assert result.returncode == 2
    assert "Usage:" in result.stderr


@pytest.mark.parametrize("value", ["249", "0", "-1", "abc", "250.5", "1000000"])
def test_resource_gate_rejects_lower_or_invalid_override(
    tmp_path: Path, value: str
) -> None:
    env = _disk_env(tmp_path)
    env["FOMC_RETRAIN_MIN_DISK_GIB"] = value
    result = _run(RESOURCE_GATE, env=env)
    assert result.returncode == 2
    assert "ERROR:" in result.stderr


def test_single_policy_gpu_env_check_uses_real_cuda_smoke_contract(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "conda.log"
    env = os.environ.copy()
    env.update(
        {
            "FAKE_CONDA_LOG": str(log_path),
            "FOMC_RETRAIN_TRAIN_GPUS": "1",
            "PATH": _fake_path(
                tmp_path,
                {
                    "conda": """#!/usr/bin/env bash
printf 'CUDA=%s ARGS=' "${CUDA_VISIBLE_DEVICES:-}" >> "${FAKE_CONDA_LOG}"
printf '%q ' "$@" >> "${FAKE_CONDA_LOG}"
printf '\n' >> "${FAKE_CONDA_LOG}"
exit 0
""",
                },
            ),
        }
    )

    result = _run(ENV_CHECK, "--train", "--skip-nccl", env=env)
    assert result.returncode == 0, result.stderr
    calls = log_path.read_text(encoding="utf-8")
    assert "CUDA=1" in calls
    assert "minimum_gpu_count=1" in calls
    assert "check_cuda_and_bitsandbytes" in calls
    assert "check_supplied_freeze_path" in calls
    assert "check_versions" in calls
    assert str(TRAIN_FREEZE) in calls
    assert "torch.distributed.run" not in calls
    assert "--skip-cuda" not in calls


def test_skip_gpu_environment_checks_pass_fixed_role_freezes(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "conda.log"
    env = os.environ.copy()
    env.update(
        {
            "FAKE_CONDA_LOG": str(log_path),
            "PATH": _fake_path(
                tmp_path,
                {
                    "conda": """#!/usr/bin/env bash
printf 'ARGS=' >> "${FAKE_CONDA_LOG}"
printf '%q ' "$@" >> "${FAKE_CONDA_LOG}"
printf '\n' >> "${FAKE_CONDA_LOG}"
exit 0
""",
                },
            ),
        }
    )

    result = _run(ENV_CHECK, "--skip-gpu", env=env)
    assert result.returncode == 0, result.stderr
    calls = log_path.read_text(encoding="utf-8")
    assert f"--role train --skip-cuda {TRAIN_FREEZE}" in calls
    assert f"--role judge --skip-cuda {JUDGE_FREEZE}" in calls


def test_environment_check_rejects_duplicate_visible_gpu_ids(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "FAKE_CONDA_LOG": str(tmp_path / "conda.log"),
            "FOMC_RETRAIN_TRAIN_GPUS": "1,1",
            "PATH": _fake_path(
                tmp_path,
                {"conda": "#!/usr/bin/env bash\nexit 0\n"},
            ),
        }
    )
    result = _run(ENV_CHECK, "--train", "--skip-gpu", env=env)
    assert result.returncode == 2
    assert "duplicate GPU ID" in result.stderr


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("FOMC_RETRAIN_DDP_GPUS", "1,0", "DDP topology is pinned"),
        ("FOMC_RETRAIN_DDP_GPUS", "0", "DDP topology is pinned"),
        ("FOMC_RETRAIN_POLICY_GPU_ID", "0", "chk2 policy is pinned"),
    ],
)
def test_stage_rejects_physical_gpu_role_remapping(
    tmp_path: Path, name: str, value: str, message: str
) -> None:
    env = os.environ.copy()
    env[name] = value
    result = _run(
        STAGE,
        "--run-manifest",
        str(tmp_path / "not-read.json"),
        "--stage",
        "chk2",
        env=env,
    )
    assert result.returncode == 2
    assert message in result.stderr


def test_chk2_stage_caches_cuda_smoke_and_keeps_dynamic_gpu_gate() -> None:
    source = STAGE.read_text(encoding="utf-8")
    preflight = source[source.index("run_training_preflight()") : source.index("record_launch_preflight_if_needed()")]
    branch = source[source.index("run_stage_training()") : source.index("record_training_if_needed()")]
    assert 'FOMC_RETRAIN_TRAIN_GPUS="${POLICY_GPU_ID}"' in preflight
    assert 'check_retrain_v2_envs.sh" --train --skip-nccl' in preflight
    assert "preflight_cache" in source
    assert branch.index("gpu_gate.sh") < branch.index("jobs.train.train_grpo")
    assert 'check_retrain_v2_envs.sh" --train --skip-nccl' not in branch


def test_cross_numa_ddp_pins_the_validated_nccl_transport() -> None:
    environment_check = ENV_CHECK.read_text(encoding="utf-8")
    stage = STAGE.read_text(encoding="utf-8")
    for source in (environment_check, stage):
        assert "export NCCL_P2P_DISABLE=1" in source
        assert "export NCCL_IB_DISABLE=1" in source
    assert "--kill-after=15s 60s" in environment_check
    assert "--kill-after=15s 600s" in stage
