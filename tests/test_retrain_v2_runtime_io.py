import json
from types import SimpleNamespace
from unittest.mock import Mock

import yaml

from open_r1.trainer.trainer import Trainer
from open_r1.utils.callbacks import LossLoggingCallback, RewardLoggingCallback


class _RuntimeConfigTrainer(Trainer):
    def load_trainer(self):
        raise NotImplementedError

    def plot_customized_curve(self):
        raise NotImplementedError


def _callback_state(*, is_world_process_zero: bool, step: int = 1):
    return SimpleNamespace(
        is_world_process_zero=is_world_process_zero,
        global_step=step,
    )


def test_loss_callback_only_writes_on_world_process_zero(tmp_path):
    callback = LossLoggingCallback(SimpleNamespace(output_dir=str(tmp_path)), None)
    callback.on_log(
        None,
        _callback_state(is_world_process_zero=False),
        None,
        logs={"loss": 2.0},
    )
    assert not (tmp_path / "loss_history.jsonl").exists()

    callback.on_log(
        None,
        _callback_state(is_world_process_zero=True),
        None,
        logs={"loss": 1.25},
    )
    record = json.loads((tmp_path / "loss_history.jsonl").read_text())
    assert record == {"step": 1, "train_loss": 1.25, "eval_loss": None}


def test_reward_callback_only_writes_on_world_process_zero(tmp_path):
    callback = RewardLoggingCallback(SimpleNamespace(output_dir=str(tmp_path)), None)
    callback.on_log(
        None,
        _callback_state(is_world_process_zero=False),
        None,
        logs={"reward": 0.5},
    )
    assert not (tmp_path / "reward_history.jsonl").exists()

    callback.on_log(
        None,
        _callback_state(is_world_process_zero=True),
        None,
        logs={
            "reward": 0.75,
            "rewards/decision_dense_reward_v2/mean": 0.625,
        },
    )
    record = json.loads((tmp_path / "reward_history.jsonl").read_text())
    assert record["step"] == 1
    assert record["reward"] == 0.75
    assert record["decision_dense_reward"] == 0.625
    assert record["reward_components"] == {
        "rewards/decision_dense_reward_v2/mean": 0.625
    }


def test_reward_callback_prefers_v3_dense_reward_and_keeps_components(tmp_path):
    callback = RewardLoggingCallback(SimpleNamespace(output_dir=str(tmp_path)), None)
    callback.on_log(
        None,
        _callback_state(is_world_process_zero=True),
        None,
        logs={
            "reward": 0.75,
            "rewards/decision_dense_reward_v2/mean": 0.25,
            "rewards/decision_dense_reward_v3/mean": 0.625,
        },
    )

    record = json.loads((tmp_path / "reward_history.jsonl").read_text())
    assert record["decision_dense_reward"] == 0.625
    assert record["reward_components"] == {
        "rewards/decision_dense_reward_v2/mean": 0.25,
        "rewards/decision_dense_reward_v3/mean": 0.625,
    }


def _runtime_config_trainer(tmp_path, *, process_index: int, payload: dict):
    trainer = object.__new__(_RuntimeConfigTrainer)
    trainer.training_args = SimpleNamespace(
        output_dir=str(tmp_path),
        process_index=process_index,
    )
    trainer._logger = Mock()
    trainer._build_runtime_config = Mock(return_value=payload)
    return trainer


def test_runtime_config_only_world_process_zero_writes_atomically(tmp_path):
    worker = _runtime_config_trainer(
        tmp_path,
        process_index=1,
        payload={"writer": "worker"},
    )
    worker.save_runtime_config()
    destination = tmp_path / "resolved_runtime_config.json"
    assert not destination.exists()
    worker._build_runtime_config.assert_not_called()

    main = _runtime_config_trainer(
        tmp_path,
        process_index=0,
        payload={"writer": "main"},
    )
    main.save_runtime_config()
    assert json.loads(destination.read_text()) == {"writer": "main"}
    assert list(tmp_path.glob(".resolved_runtime_config.json.*.tmp")) == []


def test_grpo_semantics_are_explicitly_pinned():
    expected = {
        "loss_type": "dapo",
        "scale_rewards": "group",
        "epsilon": 0.2,
        "num_iterations": 1,
        "importance_sampling_level": "token",
    }
    for filename in ("chk2_analysis_grpo.yaml", "chk4_decision_grpo.yaml"):
        with open(f"configs/retrain_v2/{filename}", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        assert {key: config[key] for key in expected} == expected
