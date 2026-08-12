from __future__ import annotations

import json
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import pytest
from trl import TrlParser
from trl.trainer.utils import RepeatSampler

from jobs.retrain_v2 import chk4_v5_cp24_grpo_continuation as continuation
from jobs.retrain_v2 import merge_chk4_selected_sft_checkpoint as selected
from open_r1.configs import GRPOConfig, GRPOScriptArguments, LoraArguments, ModelConfig
from open_r1.provenance import sha256_file
from open_r1.utils.callbacks import Chk4DecisionSmokeGateCallback


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def _smoke_config(tmp_path: Path, **overrides) -> SimpleNamespace:
    values = {
        "output_dir": str(tmp_path),
        "max_steps": 2,
        "generation_batch_size": 8,
        "num_generations": 4,
        "steps_per_generation": 8,
        "gradient_accumulation_steps": 8,
        "per_device_train_batch_size": 1,
        "world_size": 1,
        "num_iterations": 1,
        "seed": 5,
        "data_seed": 5,
        "do_eval": False,
        "eval_strategy": "no",
        "logging_strategy": "steps",
        "logging_steps": 1,
        "logging_first_step": True,
        "shuffle_dataset": True,
        "resume_from_checkpoint": None,
        "overwrite_output_dir": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _smoke_callback(tmp_path: Path, **overrides) -> Chk4DecisionSmokeGateCallback:
    return Chk4DecisionSmokeGateCallback(
        _smoke_config(tmp_path, **overrides),
        SimpleNamespace(),
    )


def _write_reward_batch(
    path: Path,
    *,
    step: int,
    rewards: list[float],
    format_only: bool = False,
) -> None:
    directions = continuation.SMOKE_STEP_TARGET_DIRECTIONS[step]
    layout = [direction for direction in directions for _ in range(4)]
    with path.open("a", encoding="utf-8") as handle:
        for direction, reward in zip(layout, rewards, strict=True):
            target_magnitude = 0 if direction == "hold" else 25
            if format_only or reward == 0.0:
                prediction = None
                direction_correct = False
                semantic = 0.0
            else:
                prediction = {
                    "direction": direction,
                    "magnitude_bp": target_magnitude,
                }
                direction_correct = True
                semantic = 0.95
            handle.write(
                json.dumps(
                    {
                        "type": "decision_dense_v3",
                        "target": {
                            "direction": direction,
                            "magnitude_bp": target_magnitude,
                        },
                        "prediction": prediction,
                        "direction_correct": direction_correct,
                        "semantic_score_before_discount": semantic,
                        "reward": reward,
                    }
                )
                + "\n"
            )


def test_smoke_step_callback_accepts_two_diverse_learning_steps(tmp_path: Path) -> None:
    callback = _smoke_callback(tmp_path)
    state = SimpleNamespace(global_step=0, is_world_process_zero=True)
    callback.on_train_begin(None, state, None)
    for step in (1, 2):
        _write_reward_batch(
            tmp_path / "reward.jsonl",
            step=step,
            rewards=[1.0, 0.0, 0.2, 0.2] * 2,
        )
        state.global_step = step
        callback.on_log(
            None,
            state,
            None,
            logs={
                "loss": 0.1,
                "grad_norm": 0.2,
                "completions/clipped_ratio": 0.25,
            },
        )
    callback.on_train_end(None, state, None)
    rows = [
        json.loads(line)
        for line in (tmp_path / "chk4_smoke_step_gate.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in rows] == [1, 2]
    assert all(row["status"] == "passed" for row in rows)


@pytest.mark.parametrize(
    ("rewards", "logs", "reason"),
    [
        (
            [0.0] * 8,
            {"loss": 0.1, "grad_norm": 0.2, "completions/clipped_ratio": 0.0},
            "direction_correct_nonzero",
        ),
        (
            [1.0] * 8,
            {"loss": 0.1, "grad_norm": 0.2, "completions/clipped_ratio": 0.0},
            "reward_std_gt_zero",
        ),
        (
            [1.0, 0.0, 0.2, 0.2] * 2,
            {
                "loss": float("nan"),
                "grad_norm": 0.2,
                "completions/clipped_ratio": 0.0,
            },
            "loss_finite",
        ),
        (
            [1.0, 0.0, 0.2, 0.2] * 2,
            {"loss": 0.1, "grad_norm": 0.0, "completions/clipped_ratio": 0.0},
            "grad_norm_gt_1e_12",
        ),
        (
            [1.0, 0.0, 0.2, 0.2] * 2,
            {"loss": 0.1, "grad_norm": 0.2, "completions/clipped_ratio": 0.26},
            "clipped_ratio_le_0_25",
        ),
    ],
)
def test_smoke_step_callback_fails_immediately(
    tmp_path: Path, rewards: list[float], logs: dict[str, float], reason: str
) -> None:
    callback = _smoke_callback(tmp_path)
    _write_reward_batch(tmp_path / "reward.jsonl", step=1, rewards=rewards)
    state = SimpleNamespace(global_step=1, is_world_process_zero=True)
    with pytest.raises(RuntimeError, match=reason):
        callback.on_log(None, state, None, logs=logs)
    row = json.loads(
        (tmp_path / "chk4_smoke_step_gate.jsonl").read_text().splitlines()[0]
    )
    assert row["status"] == "failed"


def test_smoke_step_callback_rejects_format_only_rewards(tmp_path: Path) -> None:
    callback = _smoke_callback(tmp_path)
    _write_reward_batch(
        tmp_path / "reward.jsonl",
        step=1,
        rewards=[0.05, 0.0, 0.05, 0.0] * 2,
        format_only=True,
    )
    state = SimpleNamespace(global_step=1, is_world_process_zero=True)
    with pytest.raises(RuntimeError, match="each_target_direction_correct_nonzero"):
        callback.on_log(
            None,
            state,
            None,
            logs={
                "loss": 0.0,
                "grad_norm": 0.2,
                "completions/clipped_ratio": 0.0,
            },
        )


def test_smoke_step_callback_rejects_wrong_target_layout(tmp_path: Path) -> None:
    callback = _smoke_callback(tmp_path)
    _write_reward_batch(
        tmp_path / "reward.jsonl",
        step=2,
        rewards=[1.0, 0.0, 0.2, 0.2] * 2,
    )
    state = SimpleNamespace(global_step=1, is_world_process_zero=True)
    with pytest.raises(RuntimeError, match="target layout drift"):
        callback.on_log(
            None,
            state,
            None,
            logs={
                "loss": 0.0,
                "grad_norm": 0.2,
                "completions/clipped_ratio": 0.0,
            },
        )


def test_smoke_step_callback_requires_contiguous_logs_and_complete_end(
    tmp_path: Path,
) -> None:
    callback = _smoke_callback(tmp_path)
    state = SimpleNamespace(global_step=1, is_world_process_zero=True)
    with pytest.raises(RuntimeError, match="has no loss"):
        callback.on_log(None, state, None, logs={"reward": 0.1})
    state.global_step = 2
    with pytest.raises(RuntimeError, match="without both immediate step gates"):
        callback.on_train_end(None, state, None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gradient_accumulation_steps", 4),
        ("per_device_train_batch_size", 2),
        ("world_size", 2),
        ("num_iterations", 2),
        ("steps_per_generation", 4),
        ("logging_steps", 2),
        ("shuffle_dataset", False),
        ("seed", 42),
        ("data_seed", 42),
        ("resume_from_checkpoint", "checkpoint-1"),
    ],
)
def test_smoke_callback_constructor_rejects_topology_drift(
    tmp_path: Path, field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        _smoke_callback(tmp_path, **{field: value})


def test_smoke_reward_replay_rejects_string_boolean_and_numeric_types(
    tmp_path: Path,
) -> None:
    callback = _smoke_callback(tmp_path)
    _write_reward_batch(
        tmp_path / "reward.jsonl",
        step=1,
        rewards=[1.0, 0.0, 0.2, 0.2] * 2,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "reward.jsonl").read_text().splitlines()
    ]
    rows[0]["direction_correct"] = "false"
    with (tmp_path / "reward.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    state = SimpleNamespace(global_step=1, is_world_process_zero=True)
    with pytest.raises(RuntimeError, match="direction_correct must be boolean"):
        callback.on_log(
            None,
            state,
            None,
            logs={
                "loss": 0.0,
                "grad_norm": 0.2,
                "completions/clipped_ratio": 0.0,
            },
        )


def test_versioned_configs_are_hash_bound_and_isolated() -> None:
    smoke = continuation._validate_config(SOURCE_ROOT, continuation.SMOKE)
    full = continuation._validate_config(SOURCE_ROOT, continuation.FULL)
    assert (
        sha256_file(SOURCE_ROOT / continuation.SMOKE.config)
        == continuation.SMOKE.config_sha256
    )
    assert (
        sha256_file(SOURCE_ROOT / continuation.FULL.config)
        == continuation.FULL.config_sha256
    )
    assert smoke["model_name_or_path"] == full["model_name_or_path"]
    assert (
        "selected_sft_checkpoints/checkpoint-24/merged/chk4_sft"
        in smoke["model_name_or_path"]
    )
    assert smoke["output_dir"] != full["output_dir"]
    assert smoke["max_steps"] == 2
    assert "max_steps" not in full
    assert smoke["reward_funcs"] == full["reward_funcs"] == ["decision_dense_v3"]


def test_sealed_smoke_config_constructs_strict_callback_with_current_trl() -> None:
    _, training, model, _ = TrlParser(
        (GRPOScriptArguments, GRPOConfig, ModelConfig, LoraArguments)
    ).parse_args_and_config(
        args=["--config", str(SOURCE_ROOT / continuation.SMOKE.config)]
    )
    callback = Chk4DecisionSmokeGateCallback(training, model)
    assert training.steps_per_generation == 8
    assert training.world_size == 1
    assert callback._EXPECTED_TARGET_DIRECTIONS == {
        1: ("hold", "hike"),
        2: ("hold", "cut"),
    }


def test_sealed_smoke_repeat_sampler_has_exact_two_step_direction_prefix() -> None:
    config = continuation._validate_config(SOURCE_ROOT, continuation.SMOKE)
    contract = continuation._smoke_sampling_contract(SOURCE_ROOT, config)
    sampler = RepeatSampler(
        range(contract["source_train"]["rows"]),
        mini_repeat_count=4,
        batch_size=2,
        repeat_count=8,
        shuffle=True,
        seed=5,
    )
    observed = list(islice(sampler, 72))
    first_chunk = [131] * 4 + [55] * 4
    second_chunk = [125] * 4 + [138] * 4
    assert observed == first_chunk * 8 + second_chunk
    assert contract["source_indices"] == [131, 55, 125, 138]
    assert contract["optimizer_step_target_groups"] == {
        "1": ["hold", "hike"],
        "2": ["hold", "cut"],
    }
    assert contract["verified_sampler_prefix"]["length"] == 72
    assert Path(contract["sampler_source"]["path"]).name == "utils.py"
    assert (
        Path(contract["trainer_sampler_integration_source"]["path"]).name
        == "grpo_trainer.py"
    )


def test_runtime_implementation_contract_binds_reward_and_callback_injection() -> None:
    contract = continuation._runtime_implementation_contract(SOURCE_ROOT)
    assert set(contract) == {
        "reward",
        "reward_registry",
        "callbacks",
        "trainer_runtime",
        "grpo_runtime",
        "entrypoint",
    }
    for descriptor in contract.values():
        path = Path(descriptor["path"])
        assert path.is_file() and not path.is_symlink()
        assert descriptor["sha256"] == sha256_file(path)


def test_v5_selection_profile_is_cp24_only_and_pins_pilot() -> None:
    try:
        profile = selected._activate_selection_profile(selected.V5_PROFILE_NAME)
        assert profile.allowed_checkpoint_steps == frozenset({24})
        assert profile.sft_release_sha256 != profile.grpo_release_sha256
        assert (
            profile.required_pilot_manifest_sha256 == continuation.PILOT_MANIFEST_SHA256
        )
        assert (
            profile.required_pilot_summary_sha256 == continuation.PILOT_SUMMARY_SHA256
        )
    finally:
        selected._activate_selection_profile(selected.LEGACY_PROFILE_NAME)


def test_full_authorization_fails_before_smoke_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked(_: Path):
        raise continuation.ContinuationError("smoke gate receipt is missing")

    monkeypatch.setattr(continuation, "_verify_smoke_gate", blocked)
    with pytest.raises(continuation.ContinuationError, match="smoke gate"):
        continuation._authorization_payload(SOURCE_ROOT, continuation.FULL)


def test_launcher_is_gpu1_only_and_has_cpu_merge_entrypoint() -> None:
    path = SOURCE_ROOT / "run/retrain_v2/chk4_v5_cp24_grpo_continuation.sh"
    text = path.read_text(encoding="utf-8")
    assert '"${ROOT_DIR}/run/retrain_v2/gpu_gate.sh" 1' in text
    assert "--num_processes" in continuation.training_command.__code__.co_consts
    assert 'export CUDA_VISIBLE_DEVICES=""' in text
    assert "merge-selected requires --execute" in text
