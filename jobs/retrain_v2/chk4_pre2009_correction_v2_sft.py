"""Preflight, authorize, and launch the one-shot chk4 correction-v2 SFT.

This workflow is additive and create-only.  It is deliberately unusable while
the config contains the draft release-manifest placeholder; formal publication
and an external SHA-256 pin must happen before authorization or GPU execution.
"""

from __future__ import annotations

import argparse
import fcntl
import inspect
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, IO

import accelerate.commands.launch as accelerate_launch
import yaml
from peft import LoraConfig
from torch.utils.data import SequentialSampler
from transformers import Trainer as TransformersTrainer
from trl import SFTTrainer, TrlParser

from jobs.retrain_v2 import chk4_pre2009_cp38_grpo_continuation as parent_workflow
from jobs.retrain_v2 import (
    materialize_chk4_pre2009_correction_v2_release as materializer,
)
from jobs.retrain_v2.train_chk4_pre2009_correction_v2 import CorrectionV2SFTConfig
from open_r1.configs import LoraArguments, ModelConfig, SFTScriptArguments
from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.trainer.dataset_release import CHK4_STUDENT_SYSTEM_PROMPT
from open_r1.trainer.dataset_release_correction_v2 import (
    verify_correction_v2_runtime_binding,
)
from open_r1.trainer.fixed_schedule_sampler_correction_v2 import (
    SAMPLER_TYPE,
    validate_runtime_contract,
)
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


BRANCH_ID = "chk4_from_pre2009_cp38_selected_correction_sft_v2_lr2e6_steps6_20260811"
PROFILE = "pre2009_cp38_correction_sft_v2_lr2e6_steps6"
AUTHORIZATION_SCHEMA = "chk4-pre2009-cp38-correction-sft-authorization-v2"
RUN_ROOT = Path(f"output/training/retrain_v2/{BRANCH_ID}")
CONFIG = Path(
    "configs/retrain_v2/"
    "chk4_decision_sft_from_pre2009_cp38_correction_v2_lr2e6_steps6_20260811.yaml"
)
RELEASE_MANIFEST = Path(
    f"dataset/processed/retrain_v2/{materializer.RELEASE_ID}/release_manifest.json"
)
MANIFEST_PLACEHOLDER = "__CORRECTION_V2_MANIFEST_SHA256_PENDING_FORMAL_PUBLISH__"
EXPECTED_PYTHON = materializer.EXPECTED_PYTHON
PARENT_MODEL_SHA256 = materializer.TRAINING_PARENT_SHA256
CANDIDATE_STEPS = (2, 4, 6)
_EXEC_LOCK: IO[str] | None = None

IMPLEMENTATION_RELATIVES = (
    "configs/retrain_v2/chk4_decision_sft_from_pre2009_cp38_correction_v2_lr2e6_steps6_20260811.yaml",
    "jobs/retrain_v2/chk4_pre2009_correction_v2_sft.py",
    "jobs/retrain_v2/materialize_chk4_pre2009_correction_v2_release.py",
    "jobs/retrain_v2/train_chk4_pre2009_correction_v2.py",
    "jobs/train/train_sft.py",
    "run/retrain_v2/chk4_pre2009_correction_v2_sft.sh",
    "run/retrain_v2/gpu_gate.sh",
    "src/open_r1/configs.py",
    "src/open_r1/data_loader.py",
    "src/open_r1/provenance.py",
    "src/open_r1/structured_response.py",
    "src/open_r1/trainer/dataset_release.py",
    "src/open_r1/trainer/dataset_release_correction_v2.py",
    "src/open_r1/trainer/fixed_schedule_sampler.py",
    "src/open_r1/trainer/fixed_schedule_sampler_v3.py",
    "src/open_r1/trainer/fixed_schedule_sampler_correction_v2.py",
    "src/open_r1/trainer/prompt_contract.py",
    "src/open_r1/trainer/sft_prompt_renderer.py",
    "src/open_r1/trainer/sft_trainer.py",
    "src/open_r1/trainer/trainer.py",
    "src/open_r1/trainer/rewards/reward_funcs/online_reward.py",
    "src/open_r1/trainer/rewards/reward_funcs/structured_response.py",
    "src/open_r1/utils/__init__.py",
    "src/open_r1/utils/callbacks.py",
    "src/open_r1/utils/evaluation.py",
    "src/open_r1/utils/hub.py",
    "src/open_r1/utils/import_utils.py",
    "src/open_r1/utils/model_utils.py",
    "src/open_r1/utils/plot_loss.py",
    "src/open_r1/utils/wandb_logging.py",
)


class CorrectionV2SFTWorkflowError(RuntimeError):
    """The correction-v2 SFT is not safely launchable."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorrectionV2SFTWorkflowError(message)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorrectionV2SFTWorkflowError(f"invalid {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing config: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "correction-v2 config must be a mapping")
    return value


def _resolve(repo_root: Path, value: object, *, label: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{label} path missing")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _descriptor(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    _require(
        resolved.is_file() and not path.is_symlink(), f"implementation missing: {path}"
    )
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _environment_contract() -> dict[str, Any]:
    executable = Path(sys.executable).absolute()
    _require(
        executable == EXPECTED_PYTHON, "workflow must use the exact train-env Python"
    )
    _require(
        sys.version_info[:3] == materializer.EXPECTED_PYTHON_VERSION,
        "workflow Python version drift",
    )
    import accelerate
    import datasets
    import peft
    import tokenizers
    import torch
    import transformers
    import trl

    modules = {
        "accelerate": accelerate,
        "datasets": datasets,
        "peft": peft,
        "tokenizers": tokenizers,
        "torch": torch,
        "transformers": transformers,
        "trl": trl,
    }
    versions = {name: str(module.__version__) for name, module in modules.items()}
    _require(
        versions == materializer.EXPECTED_PACKAGE_VERSIONS,
        "workflow package version drift",
    )
    sources = {
        "accelerate_launch": Path(str(accelerate_launch.__file__)),
        "trl_sft_trainer": Path(inspect.getfile(SFTTrainer)),
        "trl_parser": Path(inspect.getfile(TrlParser)),
        "transformers_trainer": Path(inspect.getfile(TransformersTrainer)),
        "torch_sequential_sampler": Path(inspect.getfile(SequentialSampler)),
        "peft_lora_config": Path(inspect.getfile(LoraConfig)),
        "python_executable": executable.resolve(),
    }
    return {
        "schema_version": "chk4-correction-v2-training-environment-v1",
        "python": {
            "executable": str(executable),
            "version": ".".join(
                str(value) for value in materializer.EXPECTED_PYTHON_VERSION
            ),
        },
        "package_versions": versions,
        "runtime_sources": {name: _descriptor(path) for name, path in sources.items()},
    }


def _implementation(repo_root: Path) -> dict[str, Any]:
    return {
        relative: _descriptor(repo_root / relative)
        for relative in IMPLEMENTATION_RELATIVES
    }


def _expected_config(manifest_sha: str) -> dict[str, Any]:
    return {
        "model_name_or_path": materializer.TRAINING_PARENT_MODEL.relative_to(
            materializer.REPO_ROOT
        ).as_posix(),
        "model_revision": None,
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "use_bnb_nested_quant": True,
        "bnb_4bit_quant_storage": "bfloat16",
        "dataset_name": (RELEASE_MANIFEST.parent / "decision_sft").as_posix(),
        "dataset_chk4_role": materializer.DATASET_ROLE,
        "dataset_chk4_release_manifest": RELEASE_MANIFEST.as_posix(),
        "dataset_chk4_release_manifest_sha256": manifest_sha,
        "dataset_prompt_column": "prompt",
        "dataset_train_split": "train",
        "dataset_test_split": "validation",
        "system_prompt": CHK4_STUDENT_SYSTEM_PROMPT,
        "bf16": True,
        "tf32": True,
        "optim": "paged_adamw_8bit",
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "learning_rate": 2.0e-6,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.0,
        "warmup_steps": 1,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "packing": False,
        "completion_only_loss": True,
        "max_length": 3072,
        "train_sampler": SAMPLER_TYPE,
        "shuffle_dataset": False,
        "group_by_length": False,
        "dataloader_drop_last": False,
        "do_eval": True,
        "eval_strategy": "steps",
        "eval_steps": 1,
        "max_steps": 6,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "ddp_find_unused_parameters": False,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": True,
        "peft_merged_model_path": (
            RUN_ROOT / "merged/unselected_root_adapter_prohibited"
        ).as_posix(),
        "peft_r": 32,
        "peft_lora_alpha": 64,
        "peft_lora_dropout": 0.05,
        "peft_bias": "none",
        "peft_target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "output_dir": (RUN_ROOT / "adapters/chk4_correction_sft").as_posix(),
        "overwrite_output_dir": False,
        "resume_from_checkpoint": None,
        "save_strategy": "steps",
        "save_steps": 1,
        "save_total_limit": None,
        "save_only_model": False,
        "checkpoint_keep_last": 3,
        "checkpoint_keep_every_n_steps": 0,
        "checkpoint_keep_steps": list(CANDIDATE_STEPS),
        "callbacks": ["loss_log", "checkpoint_retention"],
        "log_level": "info",
        "logging_strategy": "steps",
        "logging_steps": 1,
        "logging_first_step": True,
        "push_to_hub": False,
        "report_to": [],
        "seed": 1710190225,
        "data_seed": 1710190225,
        "use_cache": False,
        "use_liger_kernel": False,
        "activation_offloading": False,
    }


def _validate_parser_schema(config: dict[str, Any]) -> None:
    dataclass_types = (
        SFTScriptArguments,
        CorrectionV2SFTConfig,
        ModelConfig,
        LoraArguments,
    )
    recognized = {
        field.name
        for dataclass_type in dataclass_types
        for field in fields(dataclass_type)
    }
    _require(
        set(config) <= recognized,
        f"custom TrlParser schema lost config keys: {sorted(set(config) - recognized)}",
    )


def _config_contract(repo_root: Path) -> tuple[Path, dict[str, Any], str]:
    path = (repo_root / CONFIG).resolve()
    config = _read_yaml(path)
    manifest_sha = config.get("dataset_chk4_release_manifest_sha256")
    _require(
        manifest_sha != MANIFEST_PLACEHOLDER
        and isinstance(manifest_sha, str)
        and len(manifest_sha) == 64
        and all(character in "0123456789abcdef" for character in manifest_sha),
        "formal correction-v2 release SHA-256 has not been pinned in the config",
    )
    expected_config = _expected_config(str(manifest_sha))
    _require(
        set(config) == set(expected_config),
        "correction-v2 config key set drift",
    )
    for key, expected in expected_config.items():
        _require(config[key] == expected, f"correction-v2 config {key} drift")
    parent = _resolve(repo_root, config.get("model_name_or_path"), label="parent")
    _require(
        parent == materializer.TRAINING_PARENT_MODEL.resolve(),
        "exact cp38 parent path drift",
    )
    release = _resolve(
        repo_root, config.get("dataset_chk4_release_manifest"), label="release manifest"
    )
    _require(
        release == (repo_root / RELEASE_MANIFEST).resolve(), "formal release path drift"
    )
    dataset = _resolve(repo_root, config.get("dataset_name"), label="dataset")
    _require(dataset == release.parent / "decision_sft", "dataset path drift")
    output = _resolve(repo_root, config.get("output_dir"), label="training output")
    _require(
        output == (repo_root / RUN_ROOT / "adapters/chk4_correction_sft").resolve(),
        "training output path drift",
    )
    prohibited_merge = _resolve(
        repo_root, config.get("peft_merged_model_path"), label="prohibited root merge"
    )
    _require(
        prohibited_merge
        == (
            repo_root / RUN_ROOT / "merged/unselected_root_adapter_prohibited"
        ).resolve(),
        "prohibited root merge path drift",
    )
    # CPU preflight deliberately hides GPUs, while TrainingArguments rejects
    # bf16 construction without one. Validate the exact parser schema here;
    # the real GPU entrypoint performs the full TrlParser construction.
    _validate_parser_schema(config)
    return path, config, str(manifest_sha)


def preflight(repo_root: Path) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    environment = _environment_contract()
    config_path, config, manifest_sha = _config_contract(root)
    release_path = (root / RELEASE_MANIFEST).resolve()
    _require(
        release_path.is_file()
        and not release_path.is_symlink()
        and sha256_file(release_path) == manifest_sha,
        "formal correction-v2 manifest external pin drift",
    )
    parent_fingerprint = fingerprint_artifact_path(materializer.TRAINING_PARENT_MODEL)
    _require(
        parent_fingerprint["sha256"] == PARENT_MODEL_SHA256, "exact cp38 parent drift"
    )
    plan, selection = parent_workflow._published_selection(root)
    _require(
        plan.destination.resolve() == materializer.TRAINING_PARENT_MODEL.resolve(),
        "cp38 selection path drift",
    )
    _require(
        selection.get("merged_artifact") == parent_fingerprint,
        "cp38 selection receipt drift",
    )
    runtime_release = verify_correction_v2_runtime_binding(
        dataset_dir=release_path.parent / "decision_sft",
        manifest_path=release_path,
        expected_manifest_sha256=manifest_sha,
        dataset_role=materializer.DATASET_ROLE,
        system_prompt=CHK4_STUDENT_SYSTEM_PROMPT,
        model_path=materializer.TRAINING_PARENT_MODEL,
    )
    sampler = validate_runtime_contract(
        training_args=SimpleNamespace(
            train_sampler=SAMPLER_TYPE,
            world_size=1,
            n_gpu=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            max_steps=6,
            shuffle_dataset=False,
            group_by_length=False,
            dataloader_drop_last=False,
            use_liger_kernel=False,
        ),
        release_binding=runtime_release,
    )
    output = _resolve(root, config["output_dir"], label="training output")
    prohibited_merge = _resolve(
        root, config["peft_merged_model_path"], label="prohibited root merge"
    )
    _require(
        not output.exists() and not output.is_symlink(), "fresh-only output exists"
    )
    _require(
        not prohibited_merge.exists() and not prohibited_merge.is_symlink(),
        "unselected root adapter merge exists",
    )
    return {
        "schema_version": "chk4-pre2009-correction-v2-sft-preflight-v1",
        "status": "ready",
        "branch_id": BRANCH_ID,
        "profile": PROFILE,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "run_root": str((root / RUN_ROOT).resolve()),
        "parent": {
            "merged_artifact": parent_fingerprint,
            "selection_receipt": {
                "path": str(plan.receipt_path),
                "sha256": sha256_file(plan.receipt_path),
                "payload_sha256": selection["integrity"]["payload_sha256"],
            },
        },
        "release": {
            "path": str(release_path),
            "sha256": manifest_sha,
            "release_id": materializer.RELEASE_ID,
            "train_sha256": sampler["train_file"]["sha256"],
            "schedule_sha256": sampler["schedule"]["sha256"],
            "runtime_binding_schema": runtime_release["schema_version"],
            "sampler": sampler,
        },
        "training": {
            "fresh_lora": True,
            "max_steps": 6,
            "candidate_checkpoints": list(CANDIDATE_STEPS),
            "learning_rate": 2.0e-6,
            "warmup_steps": 1,
            "seed": 1710190225,
            "data_seed": 1710190225,
            "gpu": 1,
            "resume": "forbidden",
            "root_adapter_merge_authorized": False,
        },
        "environment": environment,
        "implementation": _implementation(root),
    }


def _authorization_path(repo_root: Path) -> Path:
    return (repo_root / RUN_ROOT / "receipts/authorization.json").resolve()


def _authorization_payload(repo_root: Path) -> dict[str, Any]:
    return {
        "schema_version": AUTHORIZATION_SCHEMA,
        "status": "authorized",
        "authorization_basis": "explicit_user_instruction_trial_correction_v2_sft",
        "scope": {
            "operation": "fresh_six_step_correction_v2_sft_on_gpu1",
            "not_authorized": [
                "gpu0",
                "different_seed",
                "different_data",
                "different_parent",
                "resume_or_second_run",
                "root_adapter_merge",
                "historical_output_overwrite",
                "grpo_before_selected_checkpoint_confirmation",
            ],
        },
        "preflight": preflight(repo_root),
    }


def authorize(repo_root: Path, *, execute: bool) -> dict[str, Any]:
    payload = seal_manifest(_authorization_payload(repo_root))
    path = _authorization_path(repo_root)
    if not execute:
        return {**payload, "status": "ready_to_authorize", "path": str(path)}
    _require(
        not path.exists() and not path.is_symlink(),
        f"authorization is create-only: {path}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def _verify_authorization(repo_root: Path) -> dict[str, Any]:
    path = _authorization_path(repo_root)
    authorization = _read_json(path, label="correction-v2 authorization")
    _require(
        path.stat().st_mode & 0o777 == 0o400,
        "correction-v2 authorization mode must be 0400",
    )
    validate_manifest_integrity(authorization)
    _require(
        authorization == seal_manifest(_authorization_payload(repo_root)),
        "correction-v2 authorization no longer matches preflight",
    )
    return authorization


def training_command(repo_root: Path) -> dict[str, Any]:
    _verify_authorization(repo_root)
    config_path, _config, _manifest_sha = _config_contract(repo_root)
    return {
        "status": "ready",
        "branch_id": BRANCH_ID,
        "gpu": 1,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "argv": [
            str(EXPECTED_PYTHON),
            "-m",
            "accelerate.commands.launch",
            "--num_processes",
            "1",
            "--mixed_precision",
            "bf16",
            "--dynamo_backend",
            "no",
            "-m",
            "jobs.retrain_v2.train_chk4_pre2009_correction_v2",
            "--config",
            str(config_path),
        ],
        "resume": "forbidden_fresh_only_v2",
    }


def execute_training(repo_root: Path) -> None:
    global _EXEC_LOCK
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "launch requires GPU1")
    command = training_command(repo_root)
    lock_path = (repo_root / RUN_ROOT / ".training.lock").resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    _EXEC_LOCK = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(_EXEC_LOCK.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise CorrectionV2SFTWorkflowError(
            "correction-v2 training lock is already held"
        ) from exc
    os.set_inheritable(_EXEC_LOCK.fileno(), True)
    _require(
        os.get_inheritable(_EXEC_LOCK.fileno()),
        "correction-v2 training lock is not exec-inheritable",
    )
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "1",
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "PYTHONPATH": f"{repo_root / 'src'}:{repo_root}",
            "FOMC_CHK4_BRANCH_ID": BRANCH_ID,
            "FOMC_CHK4_BRANCH_PROFILE": PROFILE,
            "FOMC_CHK4_BRANCH_STAGE": "decision_sft_correction_v2",
            "FOMC_CHK4_BRANCH_CONFIG_PATH": command["config"]["path"],
            "FOMC_CHK4_BRANCH_CONFIG_SHA256": command["config"]["sha256"],
        }
    )
    os.chdir(repo_root)
    os.execvpe(command["argv"][0], command["argv"], environment)


def status(repo_root: Path) -> dict[str, Any]:
    output = (repo_root / RUN_ROOT / "adapters/chk4_correction_sft").resolve()
    checkpoints = (
        sorted(
            int(path.name.split("-", 1)[1])
            for path in output.glob("checkpoint-[0-9]*")
            if path.is_dir() and path.name.split("-", 1)[1].isdigit()
        )
        if output.is_dir()
        else []
    )
    return {
        "branch_id": BRANCH_ID,
        "authorization": (
            "present" if _authorization_path(repo_root).is_file() else "absent"
        ),
        "training_output": "absent" if not output.exists() else "present",
        "checkpoints": checkpoints,
        "candidate_checkpoints_present": [
            step for step in CANDIDATE_STEPS if step in checkpoints
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("status")
    authorize_parser = subparsers.add_parser("authorize")
    authorize_parser.add_argument("--execute", action="store_true")
    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = args.repo_root.expanduser().resolve()
    try:
        if args.command == "preflight":
            result = preflight(root)
        elif args.command == "status":
            result = status(root)
        elif args.command == "authorize":
            result = authorize(root, execute=args.execute)
        elif args.command == "launch":
            if args.execute:
                execute_training(root)
                raise AssertionError("exec unexpectedly returned")
            result = training_command(root)
        else:  # pragma: no cover
            raise CorrectionV2SFTWorkflowError(f"unsupported command: {args.command}")
    except (CorrectionV2SFTWorkflowError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
