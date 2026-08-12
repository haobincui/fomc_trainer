import json

from jobs.retrain_v2.check_chk2_v3_smoke import check_smoke


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _reward(value):
    return {
        "type": "grounded_analysis_v3",
        "reward": value,
        "judge_attempts": 1,
        "contract": {"well_formed": True, "answer_nonempty": True},
    }


def test_smoke_gate_accepts_healthy_twelve_steps(tmp_path):
    _write_jsonl(
        tmp_path / "runtime_safety.jsonl",
        [
            {
                "step": step,
                "metrics": {"grad_norm": 0.1},
                "completion_clipped_ratio": 0.2,
                "peak_allocated_gib": 17.0,
                "peak_reserved_gib": 20.0,
            }
            for step in range(1, 13)
        ],
    )
    _write_jsonl(tmp_path / "reward.jsonl", [_reward(0.2 + index / 100) for index in range(48)])

    result = check_smoke(tmp_path)

    assert result["passed"] is True
    assert result["reward_records"] == 48
    assert (tmp_path / "smoke_gate_summary.json").is_file()


def test_smoke_gate_rejects_clipped_or_retrying_run(tmp_path):
    _write_jsonl(
        tmp_path / "runtime_safety.jsonl",
        [
            {
                "step": step,
                "metrics": {"grad_norm": 0.0 if step <= 2 else 0.1},
                "completion_clipped_ratio": 0.5,
                "peak_allocated_gib": 17.0,
                "peak_reserved_gib": 20.0,
            }
            for step in range(1, 13)
        ],
    )
    rewards = [_reward(0.25) for _ in range(48)]
    rewards[0]["judge_attempts"] = 2
    _write_jsonl(tmp_path / "reward.jsonl", rewards)

    result = check_smoke(tmp_path)

    assert result["passed"] is False
    assert result["gates"]["completion_clipped_ratio_le_0_25"] is False
    assert result["gates"]["zero_or_fully_clipped_steps_le_0_15"] is False
    assert result["gates"]["judge_requests_error_free"] is False
