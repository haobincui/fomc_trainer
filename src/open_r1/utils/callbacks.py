#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import math
import re
import shutil
import statistics
import subprocess
from collections import deque
from pathlib import Path
from typing import List

from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from .evaluation import run_benchmark_jobs
from .hub import push_to_hub_revision


LOGGER = logging.getLogger(__name__)


def is_slurm_available() -> bool:
    # returns true if a slurm queueing system is available
    try:
        subprocess.run(
            ["sinfo"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return True
    except FileNotFoundError:
        return False


class DummyConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class PushToHubRevisionCallback(TrainerCallback):
    def __init__(
        self,
        train_config,
        model_config,
    ) -> None:
        self.model_config = model_config

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            global_step = state.global_step

            # WARNING: if you use dataclasses.replace(args, ...) the accelerator dist state will be broken, so I do this workaround
            # Also if you instantiate a new SFTConfig, the accelerator dist state will be broken
            dummy_config = DummyConfig(
                hub_model_id=args.hub_model_id,
                hub_model_revision=f"{args.hub_model_revision}-step-{global_step:09d}",
                output_dir=f"{args.output_dir}/checkpoint-{global_step}",
                system_prompt=args.system_prompt,
            )

            future = push_to_hub_revision(
                dummy_config, extra_ignore_patterns=["*.pt"]
            )  # don't push the optimizer states

            if is_slurm_available():
                dummy_config.benchmarks = args.benchmarks

                def run_benchmark_callback(_):
                    print(f"Checkpoint {global_step} pushed to hub.")
                    run_benchmark_jobs(dummy_config, self.model_config)

                future.add_done_callback(run_benchmark_callback)


class LossLoggingCallback(TrainerCallback):
    def __init__(self, train_config, model_config, output_file="loss_history.jsonl"):
        self.output_file = train_config.output_dir + "/" + output_file

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and state.is_world_process_zero:
            with open(self.output_file, "a") as f:
                json.dump(
                    {
                        "step": state.global_step,
                        "train_loss": logs.get("loss"),
                        "eval_loss": logs.get("eval_loss"),
                    },
                    f,
                )
                f.write("\n")


class RewardLoggingCallback(TrainerCallback):
    """
    日志仅输出 reward 和 eval_reward，兼容 plot_reward_curve。
    """

    def __init__(self, train_config, model_config, output_file="reward_history.jsonl"):
        self.output_file = train_config.output_dir + "/" + output_file

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and state.is_world_process_zero:
            reward_components = {
                key: value
                for key, value in sorted(logs.items())
                if key.startswith("rewards/") and key.endswith("/mean")
            }
            with open(self.output_file, "a") as f:
                json.dump(
                    {
                        "step": state.global_step,
                        "reward": logs.get("reward"),
                        "decision_dense_reward": logs.get(
                            "rewards/decision_dense_reward_v3/mean",
                            logs.get("rewards/decision_dense_reward_v2/mean"),
                        ),
                        "reward_components": reward_components,
                        "vote_accuracy_reward": logs.get(
                            "rewards/rate_accuracy_reward/mean"
                        ),
                        "vote_format_reward": logs.get(
                            "rewards/rate_format_reward/mean"
                        ),
                        "reasoning_reward": logs.get("rewards/reasoning_reward/mean"),
                        "answer_reward": logs.get("rewards/answer_reward/mean"),
                        "format_reward": logs.get("rewards/format_reward/mean"),
                        "eval_reward": logs.get("eval_reward"),
                    },
                    f,
                )
                f.write("\n")


class CheckpointRetentionCallback(TrainerCallback):
    """Keep recent and periodic full Trainer checkpoints."""

    _CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")

    def __init__(self, train_config, model_config) -> None:
        del model_config
        self.keep_last = self._positive_int(
            getattr(train_config, "checkpoint_keep_last", 0),
            name="checkpoint_keep_last",
        )
        self.keep_every_n_steps = self._positive_int(
            getattr(train_config, "checkpoint_keep_every_n_steps", 0),
            name="checkpoint_keep_every_n_steps",
            allow_zero=True,
        )
        raw_keep_steps = getattr(train_config, "checkpoint_keep_steps", [])
        if not isinstance(raw_keep_steps, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in raw_keep_steps
        ):
            raise ValueError(
                "checkpoint_keep_steps must be a list of positive integers"
            )
        if len(set(raw_keep_steps)) != len(raw_keep_steps):
            raise ValueError("checkpoint_keep_steps must not contain duplicates")
        self.keep_steps = frozenset(raw_keep_steps)
        save_total_limit = getattr(train_config, "save_total_limit", None)
        if save_total_limit not in (None, 0):
            raise ValueError(
                "checkpoint_retention requires save_total_limit to be null or zero "
                "so native rotation cannot delete milestone checkpoints"
            )
        save_strategy = getattr(train_config, "save_strategy", None)
        save_strategy = getattr(save_strategy, "value", save_strategy)
        if save_strategy != "steps":
            raise ValueError("checkpoint_retention requires save_strategy=steps")
        if getattr(train_config, "save_steps", None) != 1:
            raise ValueError("checkpoint_retention requires save_steps=1")
        if getattr(train_config, "save_only_model", False):
            raise ValueError(
                "checkpoint_retention requires save_only_model=false for "
                "fully resumable checkpoints"
            )

    @staticmethod
    def _positive_int(value, *, name: str, allow_zero: bool = False) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or (value == 0 and not allow_zero)
        ):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be a {qualifier} integer")
        return value

    def _checkpoint_directories(self, output_dir: Path) -> list[tuple[int, Path]]:
        checkpoints: list[tuple[int, Path]] = []
        if not output_dir.is_dir():
            return checkpoints
        for candidate in output_dir.iterdir():
            if candidate.is_symlink() or not candidate.is_dir():
                continue
            match = self._CHECKPOINT_PATTERN.fullmatch(candidate.name)
            if match is not None:
                checkpoints.append((int(match.group(1)), candidate))
        return sorted(checkpoints)

    def on_save(self, args, state, control, **kwargs):
        del kwargs
        if not state.is_world_process_zero:
            return control

        output_dir = Path(args.output_dir)
        checkpoints = self._checkpoint_directories(output_dir)
        recent_steps = {step for step, _ in checkpoints[-self.keep_last :]}
        milestone_steps = set(self.keep_steps)
        if self.keep_every_n_steps:
            milestone_steps.update(
                step for step, _ in checkpoints if step % self.keep_every_n_steps == 0
            )
        protected_steps = recent_steps | milestone_steps

        best_checkpoint = getattr(state, "best_model_checkpoint", None)
        if best_checkpoint:
            best_path = Path(best_checkpoint)
            if not best_path.is_absolute():
                best_path = (
                    output_dir / best_path
                    if best_path.parent == Path(".")
                    else Path.cwd() / best_path
                )
            match = self._CHECKPOINT_PATTERN.fullmatch(best_path.name)
            if (
                match is not None
                and not best_path.is_symlink()
                and best_path.parent.resolve() == output_dir.resolve()
            ):
                protected_steps.add(int(match.group(1)))

        deleted_steps: list[int] = []
        for step, checkpoint in checkpoints:
            if step in protected_steps:
                continue
            # checkpoint came from a direct, non-symlink child scan above. Do
            # not suppress errors: retention failures must stop the run before
            # unbounded checkpoint growth can fill the filesystem.
            shutil.rmtree(checkpoint)
            deleted_steps.append(step)

        LOGGER.info(
            "Checkpoint retention kept steps=%s and deleted steps=%s in %s",
            sorted(protected_steps),
            deleted_steps,
            output_dir,
        )
        return control


class RuntimeSafetyLoggingCallback(TrainerCallback):
    """Persist chk2 safety metrics and fail closed on sustained zero gradients."""

    def __init__(self, train_config, model_config, output_file="runtime_safety.jsonl"):
        del model_config
        self.output_file = train_config.output_dir + "/" + output_file
        self.zero_gradient_window: deque[bool] = deque(maxlen=20)

    def on_train_begin(self, args, state, control, **kwargs):
        del args, state, control, kwargs
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_log(self, args, state, control, logs=None, **kwargs):
        del args, control, kwargs
        if logs is None or not state.is_world_process_zero:
            return
        import torch

        checked_metrics: dict[str, float] = {}
        for name in ("loss", "reward", "grad_norm"):
            value = logs.get(name)
            if value is None:
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                raise RuntimeError(f"non-finite chk2 runtime metric: {name}={numeric}")
            checked_metrics[name] = numeric

        grad_norm = checked_metrics.get("grad_norm")
        if grad_norm is not None:
            self.zero_gradient_window.append(abs(grad_norm) <= 1e-12)

        gib = 1024**3
        peak_allocated = (
            torch.cuda.max_memory_allocated() / gib
            if torch.cuda.is_available()
            else 0.0
        )
        peak_reserved = (
            torch.cuda.max_memory_reserved() / gib if torch.cuda.is_available() else 0.0
        )
        record = {
            "step": state.global_step,
            "metrics": checked_metrics,
            "completion_clipped_ratio": logs.get("completions/clipped_ratio"),
            "frac_reward_zero_std": logs.get("frac_reward_zero_std"),
            "peak_allocated_gib": peak_allocated,
            "peak_reserved_gib": peak_reserved,
            "zero_gradient_window_size": len(self.zero_gradient_window),
            "zero_gradient_window_fraction": (
                sum(self.zero_gradient_window) / len(self.zero_gradient_window)
                if self.zero_gradient_window
                else None
            ),
        }
        with open(self.output_file, "a", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, allow_nan=False)
            handle.write("\n")

        if (
            len(self.zero_gradient_window) == self.zero_gradient_window.maxlen
            and sum(self.zero_gradient_window) / len(self.zero_gradient_window) > 0.25
        ):
            raise RuntimeError("chk2 rolling-20 zero-gradient rate exceeded 25 percent")


class Chk4DecisionSmokeGateCallback(TrainerCallback):
    """Fail the isolated two-step chk4 smoke at the first unsafe step."""

    _EXPECTED_STEPS = 2
    _EXPECTED_REWARDS_PER_STEP = 8
    _EXPECTED_GENERATIONS_PER_PROMPT = 4
    _EXPECTED_TARGET_DIRECTIONS = {
        1: ("hold", "hike"),
        2: ("hold", "cut"),
    }
    _MIN_GRAD_NORM = 1e-12
    _AUDIT_SCHEMA = "chk4-decision-grpo-smoke-step-gate-v2"
    _EXPECTED_SEED = 5
    _EXPECTED_DATA_SEED = 5

    def __init__(self, train_config, model_config) -> None:
        del model_config
        self.output_dir = Path(train_config.output_dir)
        self.reward_path = self.output_dir / "reward.jsonl"
        self.audit_path = self.output_dir / "chk4_smoke_step_gate.jsonl"
        self.seen_steps: set[int] = set()
        expected_ints = {
            "max_steps": self._EXPECTED_STEPS,
            "generation_batch_size": self._EXPECTED_REWARDS_PER_STEP,
            "num_generations": self._EXPECTED_GENERATIONS_PER_PROMPT,
            "steps_per_generation": 8,
            "gradient_accumulation_steps": 8,
            "per_device_train_batch_size": 1,
            "world_size": 1,
            "num_iterations": 1,
            "seed": self._EXPECTED_SEED,
            "data_seed": self._EXPECTED_DATA_SEED,
        }
        for name, expected in expected_ints.items():
            value = getattr(train_config, name, None)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value != expected
            ):
                raise ValueError(f"chk4 smoke gate requires {name}={expected}")
        if bool(getattr(train_config, "do_eval", False)):
            raise ValueError("chk4 smoke gate forbids evaluation during the pilot")
        eval_strategy = getattr(train_config, "eval_strategy", None)
        if getattr(eval_strategy, "value", eval_strategy) != "no":
            raise ValueError("chk4 smoke gate requires eval_strategy=no")
        logging_strategy = getattr(train_config, "logging_strategy", None)
        if getattr(logging_strategy, "value", logging_strategy) != "steps":
            raise ValueError("chk4 smoke gate requires logging_strategy=steps")
        logging_steps = getattr(train_config, "logging_steps", None)
        if (
            isinstance(logging_steps, bool)
            or not isinstance(logging_steps, (int, float))
            or float(logging_steps) != 1.0
        ):
            raise ValueError("chk4 smoke gate requires logging_steps=1")
        if getattr(train_config, "logging_first_step", None) is not True:
            raise ValueError("chk4 smoke gate requires logging_first_step=true")
        if getattr(train_config, "shuffle_dataset", None) is not True:
            raise ValueError("chk4 smoke gate requires shuffle_dataset=true")
        if getattr(train_config, "resume_from_checkpoint", None) is not None:
            raise ValueError("chk4 smoke gate forbids resume_from_checkpoint")
        if getattr(train_config, "overwrite_output_dir", None) is not False:
            raise ValueError("chk4 smoke gate requires overwrite_output_dir=false")

    @staticmethod
    def _number(value, *, label: str, require_finite: bool = True) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"{label} must be numeric")
        result = float(value)
        if require_finite and not math.isfinite(result):
            raise RuntimeError(f"{label} must be finite")
        return result

    @staticmethod
    def _prediction_direction(row: dict, *, label: str) -> str | None:
        prediction = row.get("prediction")
        if prediction is None:
            return None
        if not isinstance(prediction, dict) or set(prediction) != {
            "direction",
            "magnitude_bp",
        }:
            raise RuntimeError(f"{label} prediction schema is invalid")
        direction = prediction.get("direction")
        magnitude = prediction.get("magnitude_bp")
        if direction not in {"hold", "hike", "cut"}:
            raise RuntimeError(f"{label} prediction direction is invalid")
        if isinstance(magnitude, bool) or not isinstance(magnitude, int):
            raise RuntimeError(f"{label} prediction magnitude is invalid")
        allowed = {0} if direction == "hold" else {25, 50, 75, 100}
        if magnitude not in allowed:
            raise RuntimeError(f"{label} prediction magnitude is invalid")
        return direction

    def _validate_batch(self, step: int, batch: list[dict]) -> tuple[list[float], dict]:
        expected_directions = self._EXPECTED_TARGET_DIRECTIONS[step]
        expected_layout = [
            direction
            for direction in expected_directions
            for _ in range(self._EXPECTED_GENERATIONS_PER_PROMPT)
        ]
        observed_layout: list[str] = []
        rewards: list[float] = []
        correct_flags: list[bool] = []
        semantic_scores: list[float] = []
        for offset, row in enumerate(batch, start=1):
            label = f"chk4 smoke step {step} reward row {offset}"
            if row.get("type") != "decision_dense_v3":
                raise RuntimeError(f"{label} is not decision_dense_v3")
            target = row.get("target")
            if not isinstance(target, dict) or set(target) != {
                "direction",
                "magnitude_bp",
            }:
                raise RuntimeError(f"{label} target schema is invalid")
            direction = target.get("direction")
            magnitude = target.get("magnitude_bp")
            if direction not in {"hold", "hike", "cut"}:
                raise RuntimeError(f"{label} target direction is invalid")
            if isinstance(magnitude, bool) or not isinstance(magnitude, int):
                raise RuntimeError(f"{label} target magnitude is invalid")
            allowed = {0} if direction == "hold" else {25, 50, 75, 100}
            if magnitude not in allowed:
                raise RuntimeError(f"{label} target magnitude is invalid")
            observed_layout.append(direction)

            reward = self._number(row.get("reward"), label=f"{label} reward")
            if not 0.0 <= reward <= 1.0:
                raise RuntimeError(f"{label} reward is outside [0, 1]")
            semantic = self._number(
                row.get("semantic_score_before_discount"),
                label=f"{label} semantic score",
            )
            if not 0.0 <= semantic <= 0.95:
                raise RuntimeError(f"{label} semantic score is outside [0, 0.95]")
            rewards.append(reward)
            semantic_scores.append(semantic)
            direction_correct = row.get("direction_correct")
            if not isinstance(direction_correct, bool):
                raise RuntimeError(f"{label} direction_correct must be boolean")
            predicted_direction = self._prediction_direction(row, label=label)
            if direction_correct is not (predicted_direction == direction):
                raise RuntimeError(f"{label} direction_correct is inconsistent")
            correct_flags.append(direction_correct)
        if observed_layout != expected_layout:
            raise RuntimeError(
                f"chk4 smoke step {step} target layout drift: "
                f"expected={expected_layout}, observed={observed_layout}"
            )

        target_groups: dict[str, dict[str, float | int]] = {}
        group_size = self._EXPECTED_GENERATIONS_PER_PROMPT
        for group_index, direction in enumerate(expected_directions):
            start = group_index * group_size
            stop = start + group_size
            group_rewards = rewards[start:stop]
            correct_nonzero = sum(
                correct and reward > 0.0 and semantic > 0.0
                for correct, reward, semantic in zip(
                    correct_flags[start:stop],
                    group_rewards,
                    semantic_scores[start:stop],
                    strict=True,
                )
            )
            target_groups[direction] = {
                "reward_rows": group_size,
                "direction_correct_nonzero_count": correct_nonzero,
                "reward_mean": statistics.fmean(group_rewards),
                "reward_std": statistics.pstdev(group_rewards),
            }
        return rewards, target_groups

    def on_train_begin(self, args, state, control, **kwargs):
        del args, control, kwargs
        if not state.is_world_process_zero:
            return
        if int(state.global_step) != 0 or self.seen_steps:
            raise RuntimeError("chk4 smoke must begin fresh at global_step=0")
        if self.output_dir.is_symlink():
            raise RuntimeError("chk4 smoke output directory must not be a symlink")
        unsafe = [
            self.reward_path,
            self.audit_path,
            self.output_dir / "trainer_state.json",
            self.output_dir / "train_results.json",
            self.output_dir / "adapter_model.safetensors",
        ]
        unsafe.extend(self.output_dir.glob("checkpoint-*"))
        existing = [str(path) for path in unsafe if path.exists() or path.is_symlink()]
        if existing:
            raise RuntimeError(
                "chk4 smoke fresh output contains prior training artifacts: "
                + ", ".join(sorted(existing))
            )

    def _reward_rows(self) -> list[dict]:
        if not self.reward_path.is_file() or self.reward_path.is_symlink():
            raise RuntimeError("chk4 smoke reward audit is missing or unsafe")
        rows: list[dict] = []
        try:
            with self.reward_path.open("r", encoding="utf-8") as handle:
                for line_number, raw in enumerate(handle, start=1):
                    if not raw.strip():
                        raise RuntimeError(
                            f"chk4 smoke reward audit has blank row {line_number}"
                        )
                    value = json.loads(raw)
                    if not isinstance(value, dict):
                        raise RuntimeError(
                            f"chk4 smoke reward row {line_number} is not an object"
                        )
                    rows.append(value)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("chk4 smoke reward audit is unreadable") from exc
        return rows

    def on_log(self, args, state, control, logs=None, **kwargs):
        del args, control, kwargs
        if not state.is_world_process_zero:
            return
        step = int(state.global_step)
        if logs is None or logs.get("loss") is None:
            if 1 <= step <= self._EXPECTED_STEPS and step not in self.seen_steps:
                raise RuntimeError(f"chk4 smoke training log step {step} has no loss")
            return
        expected_step = len(self.seen_steps) + 1
        if (
            step < 1
            or step > self._EXPECTED_STEPS
            or step in self.seen_steps
            or step != expected_step
        ):
            raise RuntimeError(f"unexpected chk4 smoke training log step: {step}")
        expected_rows = step * self._EXPECTED_REWARDS_PER_STEP
        rows = self._reward_rows()
        if len(rows) != expected_rows:
            raise RuntimeError(
                "chk4 smoke reward rows do not align one batch per optimizer step: "
                f"expected={expected_rows}, observed={len(rows)}"
            )
        batch = rows[-self._EXPECTED_REWARDS_PER_STEP :]
        rewards, target_groups = self._validate_batch(step, batch)
        direction_correct_nonzero = sum(
            int(group["direction_correct_nonzero_count"])
            for group in target_groups.values()
        )
        reward_std = statistics.pstdev(rewards)
        loss = self._number(
            logs.get("loss"), label="chk4 smoke loss", require_finite=False
        )
        grad_norm = self._number(
            logs.get("grad_norm"),
            label="chk4 smoke grad_norm",
            require_finite=False,
        )
        clipped_ratio = logs.get("completions/clipped_ratio")
        if clipped_ratio is None:
            raise RuntimeError("chk4 smoke completion clipped ratio is missing")
        clipped_ratio = self._number(
            clipped_ratio,
            label="chk4 smoke completion clipped ratio",
            require_finite=False,
        )
        checks = {
            "each_target_direction_correct_nonzero": all(
                int(group["direction_correct_nonzero_count"]) >= 1
                for group in target_groups.values()
            ),
            "each_target_reward_std_gt_zero": all(
                float(group["reward_std"]) > 0.0 for group in target_groups.values()
            ),
            "loss_finite": math.isfinite(loss),
            "grad_norm_gt_1e_12": (
                math.isfinite(grad_norm) and grad_norm > self._MIN_GRAD_NORM
            ),
            "clipped_ratio_le_0_25": (
                math.isfinite(clipped_ratio) and 0.0 <= clipped_ratio <= 0.25
            ),
        }
        record = {
            "schema_version": self._AUDIT_SCHEMA,
            "step": step,
            "reward_rows": self._EXPECTED_REWARDS_PER_STEP,
            "target_directions": list(self._EXPECTED_TARGET_DIRECTIONS[step]),
            "target_groups": target_groups,
            "direction_correct_nonzero_count": direction_correct_nonzero,
            "reward_mean": statistics.fmean(rewards),
            "reward_std": reward_std,
            "loss": loss if math.isfinite(loss) else None,
            "grad_norm": grad_norm if math.isfinite(grad_norm) else None,
            "completion_clipped_ratio": (
                clipped_ratio if math.isfinite(clipped_ratio) else None
            ),
            "checks": checks,
            "status": "passed" if all(checks.values()) else "failed",
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        self.seen_steps.add(step)
        if record["status"] != "passed":
            failed = sorted(key for key, passed in checks.items() if not passed)
            raise RuntimeError("chk4 smoke step gate failed: " + ", ".join(failed))

    def on_train_end(self, args, state, control, **kwargs):
        del args, control, kwargs
        if not state.is_world_process_zero:
            return
        expected_steps = set(range(1, self._EXPECTED_STEPS + 1))
        if (
            int(state.global_step) != self._EXPECTED_STEPS
            or self.seen_steps != expected_steps
        ):
            raise RuntimeError(
                "chk4 smoke ended without both immediate step gates: "
                f"global_step={state.global_step}, seen={sorted(self.seen_steps)}"
            )
        rows = self._reward_rows()
        if len(rows) != self._EXPECTED_STEPS * self._EXPECTED_REWARDS_PER_STEP:
            raise RuntimeError("chk4 smoke ended with an incomplete reward audit")
        if not self.audit_path.is_file() or self.audit_path.is_symlink():
            raise RuntimeError("chk4 smoke immediate gate audit is missing or unsafe")
        try:
            raw_rows = self.audit_path.read_text(encoding="utf-8").splitlines()
            if any(not raw.strip() for raw in raw_rows):
                raise RuntimeError(
                    "chk4 smoke immediate gate audit contains a blank row"
                )
            audit_rows = [json.loads(raw) for raw in raw_rows]
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("chk4 smoke immediate gate audit is unreadable") from exc
        if (
            len(audit_rows) != self._EXPECTED_STEPS
            or any(not isinstance(row, dict) for row in audit_rows)
            or [row.get("step") for row in audit_rows] != [1, 2]
            or any(
                row.get("schema_version") != self._AUDIT_SCHEMA
                or row.get("status") != "passed"
                for row in audit_rows
            )
        ):
            raise RuntimeError("chk4 smoke immediate gate audit is incomplete")


class Chk4Pre2009Cp38SmokeGateCallback(Chk4DecisionSmokeGateCallback):
    """Immediate gate for the hash-bound pre-2009 cp38 sampler prefix."""

    _EXPECTED_SEED = 31416
    _EXPECTED_DATA_SEED = 31416
    _AUDIT_SCHEMA = "chk4-pre2009-cp38-grpo-smoke-step-gate-v1"


CALLBACKS = {
    "push_to_hub_revision": PushToHubRevisionCallback,
    "loss_log": LossLoggingCallback,
    "reward_log": RewardLoggingCallback,
    "runtime_safety": RuntimeSafetyLoggingCallback,
    "chk4_decision_smoke_gate": Chk4DecisionSmokeGateCallback,
    "chk4_pre2009_cp38_smoke_gate": Chk4Pre2009Cp38SmokeGateCallback,
    "checkpoint_retention": CheckpointRetentionCallback,
}


def get_callbacks(train_config, model_config) -> List[TrainerCallback]:
    callbacks = []
    for callback_name in train_config.callbacks:
        if callback_name not in CALLBACKS:
            raise ValueError(f"Callback {callback_name} not found in CALLBACKS.")
        callbacks.append(CALLBACKS[callback_name](train_config, model_config))

    return callbacks
