from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run/retrain_v2/start_chk2_background.sh"
SMOKE_LAUNCHER = ROOT / "run/retrain_v2/run_chk2_v3_smoke12.sh"


def test_chk2_background_launcher_persists_and_captures_logs() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "tmux new-session -d" in source
    assert "run/retrain_v2/judge.sh" in source
    assert "run/retrain_v2/stage.sh" in source
    assert '>>"${JUDGE_LOG}" 2>&1' in source
    assert '>>"${TRAIN_LOG}" 2>&1' in source
    assert "trap cleanup_judge EXIT INT TERM HUP" in source
    assert "MAX_STAGE_ATTEMPTS=6" in source
    assert "retrying from the latest verified checkpoint" in source


def test_chk2_background_launcher_waits_for_pinned_model() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'JUDGE_URL="http://127.0.0.1:8000/v1/models"' in source
    assert "grep -q 'Qwen3.5-9B'" in source


def test_chk2_v3_smoke_launcher_is_twelve_step_and_fail_closed() -> None:
    source = SMOKE_LAUNCHER.read_text(encoding="utf-8")
    assert '"${ROOT_DIR}/run/retrain_v2/gpu_gate.sh" 0 1' in source
    assert '"${ROOT_DIR}/run/retrain_v2/judge.sh"' in source
    assert "--max_steps 12" in source
    assert "check_chk2_v3_smoke" in source
    assert "--enforce" in source
    assert "trap cleanup_judge EXIT INT TERM HUP" in source
