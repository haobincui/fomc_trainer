from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, SFTConfig
from open_r1.trainer.trainer import Trainer
from open_r1.utils import callbacks as callbacks_module
from open_r1.utils.callbacks import (
    CALLBACKS,
    CheckpointRetentionCallback,
)


STAGE_CONFIGS = (
    "chk1_analysis_sft.yaml",
    "chk1_analysis_sft_compressed_flash_max_v1_20260805.yaml",
    "chk2_analysis_grpo.yaml",
    "chk2_analysis_grpo_v3.yaml",
    "chk2_analysis_grpo_compressed_chk1_v1_20260805.yaml",
    "chk3_minutes_sft.yaml",
    "chk4_decision_grpo.yaml",
)

SFT_STAGE_CONFIGS = (
    "chk1_analysis_sft.yaml",
    "chk1_analysis_sft_compressed_flash_max_v1_20260805.yaml",
    "chk3_minutes_sft.yaml",
)


class _CallbackLoadingTrainer(Trainer):
    def load_trainer(self):
        raise NotImplementedError

    def plot_customized_curve(self):
        raise NotImplementedError


def _retention_config(tmp_path, **overrides):
    values = {
        "output_dir": str(tmp_path),
        "checkpoint_keep_last": 3,
        "checkpoint_keep_every_n_steps": 10,
        "save_total_limit": None,
        "save_strategy": "steps",
        "save_steps": 1,
        "save_only_model": False,
        "callbacks": ["checkpoint_retention"],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _state(*, step: int, is_world_process_zero: bool = True, best=None):
    return SimpleNamespace(
        global_step=step,
        is_world_process_zero=is_world_process_zero,
        best_model_checkpoint=best,
    )


def _make_checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        f'{{"global_step": {step}}}\n', encoding="utf-8"
    )
    return checkpoint


def _checkpoint_steps(root: Path) -> set[int]:
    return {
        int(path.name.removeprefix("checkpoint-"))
        for path in root.glob("checkpoint-*")
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
    }


def test_retention_keeps_latest_three_and_every_tenth_checkpoint(tmp_path):
    callback = CheckpointRetentionCallback(_retention_config(tmp_path), None)
    args = SimpleNamespace(output_dir=str(tmp_path))
    control = object()

    for step in range(1, 14):
        _make_checkpoint(tmp_path, step)
        assert callback.on_save(args, _state(step=step), control) is control
    assert _checkpoint_steps(tmp_path) == {10, 11, 12, 13}

    for step in range(14, 24):
        _make_checkpoint(tmp_path, step)
        callback.on_save(args, _state(step=step), control)
    assert _checkpoint_steps(tmp_path) == {10, 20, 21, 22, 23}
    assert get_last_checkpoint(str(tmp_path)) == str(tmp_path / "checkpoint-23")


def test_milestone_that_is_also_recent_is_kept_once(tmp_path):
    callback = CheckpointRetentionCallback(_retention_config(tmp_path), None)
    for step in range(1, 21):
        _make_checkpoint(tmp_path, step)
    callback.on_save(
        SimpleNamespace(output_dir=str(tmp_path)),
        _state(step=20),
        object(),
    )
    assert _checkpoint_steps(tmp_path) == {10, 18, 19, 20}


def test_exact_milestones_are_retained_with_latest_three(tmp_path):
    callback = CheckpointRetentionCallback(
        _retention_config(
            tmp_path,
            checkpoint_keep_every_n_steps=0,
            checkpoint_keep_steps=[12, 16, 20, 24],
        ),
        None,
    )
    for step in range(1, 25):
        _make_checkpoint(tmp_path, step)
        callback.on_save(
            SimpleNamespace(output_dir=str(tmp_path)),
            _state(step=step),
            object(),
        )
    assert _checkpoint_steps(tmp_path) == {12, 16, 20, 22, 23, 24}


def test_best_checkpoint_is_preserved(tmp_path):
    callback = CheckpointRetentionCallback(_retention_config(tmp_path), None)
    for step in range(1, 7):
        _make_checkpoint(tmp_path, step)
    best = tmp_path / "checkpoint-1"
    callback.on_save(
        SimpleNamespace(output_dir=str(tmp_path)),
        _state(step=6, best=best.name),
        object(),
    )
    assert _checkpoint_steps(tmp_path) == {1, 4, 5, 6}


def test_non_checkpoint_entries_and_symlinks_are_not_touched(tmp_path):
    callback = CheckpointRetentionCallback(_retention_config(tmp_path), None)
    for step in range(1, 5):
        _make_checkpoint(tmp_path, step)
    unusual_dir = tmp_path / "checkpoint-not-a-step"
    unusual_dir.mkdir()
    unrelated_file = tmp_path / "checkpoint-2-copy"
    unrelated_file.write_text("keep", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    checkpoint_symlink = tmp_path / "checkpoint-99"
    checkpoint_symlink.symlink_to(outside, target_is_directory=True)

    callback.on_save(
        SimpleNamespace(output_dir=str(tmp_path)),
        _state(step=4),
        object(),
    )

    assert _checkpoint_steps(tmp_path) == {2, 3, 4, 99}
    assert unusual_dir.is_dir()
    assert unrelated_file.read_text(encoding="utf-8") == "keep"
    assert checkpoint_symlink.is_symlink()
    assert outside.is_dir()


def test_non_main_process_does_not_delete(tmp_path):
    callback = CheckpointRetentionCallback(_retention_config(tmp_path), None)
    for step in range(1, 6):
        _make_checkpoint(tmp_path, step)
    control = object()
    assert (
        callback.on_save(
            SimpleNamespace(output_dir=str(tmp_path)),
            _state(step=5, is_world_process_zero=False),
            control,
        )
        is control
    )
    assert _checkpoint_steps(tmp_path) == {1, 2, 3, 4, 5}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"checkpoint_keep_last": 0}, "checkpoint_keep_last"),
        ({"checkpoint_keep_every_n_steps": -1}, "checkpoint_keep_every_n_steps"),
        ({"checkpoint_keep_last": True}, "checkpoint_keep_last"),
        ({"save_total_limit": 3}, "save_total_limit"),
        ({"save_total_limit": -1}, "save_total_limit"),
        ({"save_strategy": "epoch"}, "save_strategy"),
        ({"save_steps": 10}, "save_steps"),
        ({"save_only_model": True}, "save_only_model"),
    ],
)
def test_invalid_retention_configuration_fails_closed(
    tmp_path, overrides, message
):
    with pytest.raises(ValueError, match=message):
        CheckpointRetentionCallback(
            _retention_config(tmp_path, **overrides), None
        )


def test_checkpoint_deletion_failure_fails_closed(tmp_path, monkeypatch):
    callback = CheckpointRetentionCallback(_retention_config(tmp_path), None)
    for step in range(1, 5):
        _make_checkpoint(tmp_path, step)

    def fail_delete(path):
        raise PermissionError(f"cannot remove {path}")

    monkeypatch.setattr(callbacks_module.shutil, "rmtree", fail_delete)
    with pytest.raises(PermissionError, match="cannot remove"):
        callback.on_save(
            SimpleNamespace(output_dir=str(tmp_path)),
            _state(step=4),
            object(),
        )


def test_retention_callback_load_failure_is_not_suppressed(tmp_path):
    trainer = object.__new__(_CallbackLoadingTrainer)
    trainer.training_args = _retention_config(
        tmp_path, checkpoint_keep_last=0
    )
    trainer.model_args = SimpleNamespace()
    trainer._logger = Mock()

    with pytest.raises(ValueError, match="checkpoint_keep_last"):
        trainer.load_callbacks()
    trainer._logger.error.assert_called_once()


def test_retention_api_defaults_are_disabled_and_registered():
    assert GRPOConfig.__dataclass_fields__["checkpoint_keep_last"].default == 0
    assert (
        GRPOConfig.__dataclass_fields__["checkpoint_keep_every_n_steps"].default
        == 0
    )
    assert SFTConfig.__dataclass_fields__["checkpoint_keep_last"].default == 0
    assert (
        SFTConfig.__dataclass_fields__["checkpoint_keep_every_n_steps"].default
        == 0
    )
    assert CALLBACKS["checkpoint_retention"] is CheckpointRetentionCallback


@pytest.mark.parametrize("filename", STAGE_CONFIGS)
def test_all_retrain_stage_configs_enable_checkpoint_retention(filename):
    path = Path("configs/retrain_v2") / filename
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["save_strategy"] == "steps"
    assert config["save_steps"] == 1
    assert config["save_total_limit"] is None
    assert config["save_only_model"] is False
    assert config["checkpoint_keep_last"] == 3
    assert config["checkpoint_keep_every_n_steps"] == 10
    assert "checkpoint_retention" in config["callbacks"]


@pytest.mark.parametrize("filename", SFT_STAGE_CONFIGS)
def test_all_retrain_sft_configs_evaluate_every_ten_steps(filename):
    path = Path("configs/retrain_v2") / filename
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["do_eval"] is True
    assert config["eval_strategy"] == "steps"
    assert config["eval_steps"] == 10
    assert config["logging_strategy"] == "steps"
    assert config["logging_steps"] <= config["eval_steps"]
