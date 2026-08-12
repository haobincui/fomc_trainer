import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "run/retrain_v2/judge.sh"


@pytest.mark.parametrize(
    ("env_name", "value", "message"),
    [
        ("FOMC_RETRAIN_JUDGE_GPU_ID", "1", "pinned to physical GPU 0"),
        ("FOMC_RETRAIN_JUDGE_HOST", "0.0.0.0", "pinned to 127.0.0.1:8000"),
        ("FOMC_RETRAIN_JUDGE_PORT", "18080", "pinned to 127.0.0.1:8000"),
        ("FOMC_RETRAIN_JUDGE_MAX_MODEL_LEN", "4096", "pinned to 8192 or 12288"),
        ("FOMC_RETRAIN_JUDGE_MAX_NUM_SEQS", "8", "pinned to 4"),
        ("FOMC_RETRAIN_JUDGE_GPU_MEMORY_UTILIZATION", "0.8", "pinned to 0.72"),
    ],
)
def test_judge_launcher_rejects_hardware_or_endpoint_override(env_name, value, message):
    env = os.environ.copy()
    env[env_name] = value
    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode == 2
    assert message in completed.stderr


def test_judge_launcher_uses_the_shared_gpu_capacity_gate():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'gpu_gate.sh" "${JUDGE_GPU_ID}"' in text
    assert "free_mib < 22000" not in text
    assert "already has compute process" not in text
    assert "--max-model-len \"${JUDGE_MAX_MODEL_LEN}\"" in text
    assert "--max-num-seqs \"${JUDGE_MAX_NUM_SEQS}\"" in text
    assert "--gpu-memory-utilization \"${JUDGE_GPU_MEMORY_UTILIZATION}\"" in text
