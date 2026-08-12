"""Exclusive execution receipts for retrain-v2 training and adapter merges."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from jobs.retrain_v2.dag import verify_parent
from jobs.retrain_v2.execution_contract import (
    ExecutionContractError,
    validate_stage_topology,
    verify_execution_contract,
)
from jobs.retrain_v2.merge_attestation import (
    MergeAttestationError,
    verify_merge_attestation,
)
from jobs.retrain_v2.stage_lock import StageLockError, require_inherited_stage_lock


RECEIPT_SCHEMA_VERSION = 1
STAGE_IDS = ("chk1", "chk2", "chk3", "chk4")
ADAPTER_REQUIRED_JSON = (
    "adapter_config.json",
    "trainer_state.json",
    "train_results.json",
    "resolved_runtime_config.json",
)
ADAPTER_REQUIRED_BINARY = "training_args.bin"
ADAPTER_WEIGHT = "adapter_model.safetensors"
MERGED_REQUIRED_JSON = ("config.json", "tokenizer.json", "tokenizer_config.json")
RUNTIME_TOP_LEVEL_KEYS = {
    "schema_version",
    "model",
    "dataset",
    "training",
    "generation",
    "peft",
    "rewards",
    "environment",
}
RUNTIME_MODEL_KEYS = {
    "model_name_or_path",
    "model_revision",
    "torch_dtype",
    "dtype",
    "attn_implementation",
    "quantization",
}
RUNTIME_QUANTIZATION_KEYS = {
    "enabled",
    "load_in_4bit",
    "load_in_8bit",
    "bnb_4bit_quant_type",
    "bnb_4bit_compute_dtype",
    "bnb_4bit_quant_storage",
    "bnb_4bit_use_double_quant",
    "prepared_for_kbit_training",
}
RUNTIME_DATASET_KEYS = {"name", "prompt_column", "train_split", "eval_split"}
RUNTIME_TRAINING_KEYS = {
    "output_dir",
    "learning_rate",
    "num_train_epochs",
    "max_steps",
    "optimizer",
    "lr_scheduler_type",
    "warmup_ratio",
    "gradient_accumulation_steps",
    "gradient_checkpointing",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "seed",
    "bf16",
}
RUNTIME_GENERATION_KEYS = {
    "max_prompt_length",
    "max_completion_length",
    "num_generations",
    "temperature",
    "top_p",
}
RUNTIME_PEFT_KEYS = {
    "merged_model_path",
    "r",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
}
RUNTIME_REWARD_KEYS = {"reward_funcs", "reward_weights"}
RUNTIME_ENVIRONMENT_KEYS = {
    "device",
    "n_gpu",
    "world_size",
    "process_index",
    "local_process_index",
    "cuda_visible_devices",
    "cuda_available",
    "cuda_device_count",
    "cuda_device_names",
}
RUNTIME_JUDGE_KEYS = {
    "url",
    "model",
    "timeout",
    "verbose",
    "sleep_seconds",
    "api_key_env",
    "max_retries",
    "backoff_seconds",
    "tokenizer_path",
    "max_model_len",
    "max_completion_tokens",
    "candidate_reserve_tokens",
    "boundary_margin_tokens",
}
PEFT_COMPATIBILITY_DEFAULTS = {
    "corda_config": None,
    "eva_config": None,
    "exclude_modules": None,
    "lora_bias": False,
    "trainable_token_indices": None,
}
EXPECTED_STAGE_REWARDS = {
    "chk1": ([], None),
    "chk3": ([], None),
    "chk4": (["decision_dense_v2"], [1.0]),
}
SUPPORTED_CHK2_REWARD_FUNCS = {
    ("grounded_analysis_v2",),
    ("grounded_analysis_v3",),
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LORA_A_KEY_RE = re.compile(r"(?:^|\.)lora_A(?:\.|$)")
_LORA_B_KEY_RE = re.compile(r"(?:^|\.)lora_B(?:\.|$)")
_CHECKPOINT_DIRECTORY_RE = re.compile(r"^checkpoint-([1-9][0-9]*)$")
_COMPLETION_LOG_RE = re.compile(r"^completions_[0-9]{5}\.parquet$")


class ExecutionReceiptError(ValueError):
    """Raised when a receipt or the artifacts behind it are unsafe or drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExecutionReceiptError(message)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical_bytes(payload: Any) -> bytes:
    try:
        rendered = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ExecutionReceiptError("Receipt payload is not canonical JSON") from exc
    return rendered.encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ExecutionReceiptError(f"Unable to hash artifact: {path}") from exc
    return digest.hexdigest()


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_no_symlink_components(path: Path, *, repo_root: Path, label: str) -> None:
    lexical = _lexical_absolute(path)
    try:
        relative = lexical.relative_to(repo_root)
    except ValueError as exc:
        raise ExecutionReceiptError(f"{label} escapes the repository: {path}") from exc
    current = repo_root
    for part in relative.parts:
        current /= part
        _require(not current.is_symlink(), f"{label} must not contain symlinks: {current}")


def _canonical_repo_path(
    repo_root: Path,
    value: str | Path,
    *,
    label: str,
    require_exists: bool = True,
) -> Path:
    candidate = Path(value)
    lexical = (
        _lexical_absolute(candidate)
        if candidate.is_absolute()
        else _lexical_absolute(repo_root / candidate)
    )
    _require_no_symlink_components(lexical, repo_root=repo_root, label=label)
    resolved = lexical.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ExecutionReceiptError(f"{label} escapes the repository: {value}") from exc
    if require_exists:
        _require(resolved.exists(), f"{label} does not exist: {resolved}")
    return resolved


def _relative_path(repo_root: Path, path: Path, *, label: str) -> str:
    canonical = _canonical_repo_path(repo_root, path, label=label)
    return canonical.relative_to(repo_root).as_posix()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(not path.is_symlink(), f"{label} must not be a symlink: {path}")
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutionReceiptError(f"Unable to parse {label}: {path}") from exc
    _require(isinstance(payload, dict), f"{label} must contain a JSON object")
    return payload


def _load_yaml(path: Path, *, label: str) -> dict[str, Any]:
    _require(not path.is_symlink(), f"{label} must not be a symlink: {path}")
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ExecutionReceiptError(f"Unable to parse {label}: {path}") from exc
    _require(isinstance(payload, dict), f"{label} must contain a mapping")
    return payload


def _file_record(repo_root: Path, path: Path, *, label: str) -> dict[str, Any]:
    canonical = _canonical_repo_path(repo_root, path, label=label)
    _require(canonical.is_file(), f"{label} is not a regular file: {canonical}")
    _require(canonical.stat().st_size > 0, f"{label} is empty: {canonical}")
    return {
        "path": canonical.relative_to(repo_root).as_posix(),
        "sha256": _sha256_file(canonical),
    }


def _tree_fingerprint(repo_root: Path, root: Path, *, label: str) -> dict[str, Any]:
    canonical_root = _canonical_repo_path(repo_root, root, label=label)
    _require(canonical_root.is_dir(), f"{label} is not a directory: {canonical_root}")
    files: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise ExecutionReceiptError(f"Unable to scan {label}: {directory}") from exc
        for entry in entries:
            candidate = Path(entry.path)
            if entry.is_symlink():
                raise ExecutionReceiptError(f"{label} must not contain symlinks: {candidate}")
            if entry.is_dir(follow_symlinks=False):
                visit(candidate)
                continue
            if not entry.is_file(follow_symlinks=False):
                raise ExecutionReceiptError(
                    f"{label} contains a non-regular filesystem entry: {candidate}"
                )
            files.append(
                {
                    "path": candidate.relative_to(canonical_root).as_posix(),
                    "size": candidate.stat().st_size,
                    "sha256": _sha256_file(candidate),
                }
            )

    visit(canonical_root)
    _require(files, f"{label} contains no files")
    files.sort(key=lambda record: record["path"])
    digest = hashlib.sha256()
    for record in files:
        digest.update(
            (
                f"{record['path']}\0{record['size']}\0"
                f"{record['sha256']}\n"
            ).encode("utf-8")
        )
    return {
        "algorithm": (
            "sha256(sorted UTF-8 records "
            "'<relative_path>\\0<size>\\0<file_sha256>\\n')"
        ),
        "files": files,
        "sha256": digest.hexdigest(),
    }


def _require_positive_number(value: Any, *, label: str) -> None:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0,
        f"{label} must be positive",
    )


def _require_exact_keys(
    payload: Any, expected: set[str], *, label: str
) -> Mapping[str, Any]:
    _require(isinstance(payload, dict), f"{label} must be an object")
    _require(set(payload) == expected, f"{label} schema mismatch")
    return payload


def _require_equal(actual: Any, expected: Any, *, label: str) -> None:
    if isinstance(expected, float):
        _require(
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and float(actual) == expected,
            f"{label} mismatch",
        )
        return
    _require(actual == expected, f"{label} mismatch")


def _validate_runtime_config(context: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _load_json(
        context["adapter_path"] / "resolved_runtime_config.json",
        label="resolved_runtime_config.json",
    )
    config = context["config"]
    expected_top = set(RUNTIME_TOP_LEVEL_KEYS)
    reward_funcs = config.get("reward_funcs", [])
    if context["stage_id"] == "chk2":
        _require(
            isinstance(reward_funcs, list)
            and tuple(reward_funcs) in SUPPORTED_CHK2_REWARD_FUNCS,
            "Resolved chk2 reward functions mismatch",
        )
        expected_reward_weights = [1.0]
    else:
        expected_reward_funcs, expected_reward_weights = EXPECTED_STAGE_REWARDS[
            context["stage_id"]
        ]
        _require(
            reward_funcs == expected_reward_funcs,
            "Resolved reward functions mismatch",
        )
    _require(
        config.get("reward_weights") == expected_reward_weights,
        "Resolved reward weights mismatch",
    )
    if tuple(reward_funcs) in SUPPORTED_CHK2_REWARD_FUNCS:
        expected_top.add("judge")
    _require_exact_keys(runtime, expected_top, label="runtime config")
    _require(runtime.get("schema_version") == 1, "Unsupported runtime config schema")

    model = _require_exact_keys(
        runtime["model"], RUNTIME_MODEL_KEYS, label="runtime model"
    )
    for field, config_field in (
        ("model_revision", "model_revision"),
        ("torch_dtype", "dtype"),
        ("dtype", "dtype"),
        ("attn_implementation", "attn_implementation"),
    ):
        _require_equal(
            model[field], config.get(config_field), label=f"runtime model.{field}"
        )
    _require_equal(
        model["model_name_or_path"],
        config.get("model_name_or_path"),
        label="runtime model.model_name_or_path",
    )
    runtime_parent = _canonical_repo_path(
        context["repo_root"],
        str(model["model_name_or_path"]),
        label="runtime parent model",
    )
    _require(runtime_parent == context["parent_path"], "Runtime parent model mismatch")
    quantization = _require_exact_keys(
        model["quantization"],
        RUNTIME_QUANTIZATION_KEYS,
        label="runtime model.quantization",
    )
    expected_quantization = {
        "enabled": bool(config.get("load_in_4bit") or config.get("load_in_8bit")),
        "load_in_4bit": bool(config.get("load_in_4bit", False)),
        "load_in_8bit": bool(config.get("load_in_8bit", False)),
        "bnb_4bit_quant_type": config.get("bnb_4bit_quant_type"),
        "bnb_4bit_compute_dtype": str(
            config.get("bnb_4bit_compute_dtype") or config.get("dtype")
        ),
        "bnb_4bit_quant_storage": str(config.get("bnb_4bit_quant_storage")),
        "bnb_4bit_use_double_quant": bool(
            config.get("use_bnb_nested_quant", False)
        ),
        "prepared_for_kbit_training": bool(
            config.get("load_in_4bit") or config.get("load_in_8bit")
        ),
    }
    _require(
        dict(quantization) == expected_quantization,
        "Runtime model.quantization mismatch",
    )

    dataset = _require_exact_keys(
        runtime["dataset"], RUNTIME_DATASET_KEYS, label="runtime dataset"
    )
    expected_dataset = {
        "name": config.get("dataset_name"),
        "prompt_column": config.get("dataset_prompt_column"),
        "train_split": config.get("dataset_train_split"),
        "eval_split": config.get("dataset_test_split"),
    }
    _require(dict(dataset) == expected_dataset, "Runtime dataset mismatch")
    runtime_dataset = _canonical_repo_path(
        context["repo_root"], str(dataset["name"]), label="runtime dataset"
    )
    bound_dataset = _canonical_repo_path(
        context["repo_root"],
        str(context["data_binding"].get("dataset_path", "")),
        label="manifest-bound dataset",
    )
    _require(runtime_dataset == bound_dataset, "Runtime dataset binding mismatch")

    training = _require_exact_keys(
        runtime["training"], RUNTIME_TRAINING_KEYS, label="runtime training"
    )
    expected_training = {
        "output_dir": config.get("output_dir"),
        "learning_rate": config.get("learning_rate"),
        "num_train_epochs": config.get("num_train_epochs"),
        "max_steps": config.get("max_steps", -1),
        "optimizer": config.get("optim"),
        "lr_scheduler_type": config.get("lr_scheduler_type"),
        "warmup_ratio": config.get("warmup_ratio"),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps"),
        "gradient_checkpointing": config.get("gradient_checkpointing"),
        "per_device_train_batch_size": config.get("per_device_train_batch_size"),
        "per_device_eval_batch_size": config.get("per_device_eval_batch_size"),
        "seed": config.get("seed"),
        "bf16": config.get("bf16"),
    }
    for field, expected in expected_training.items():
        _require_equal(training[field], expected, label=f"runtime training.{field}")
    runtime_output = _canonical_repo_path(
        context["repo_root"], str(training["output_dir"]), label="runtime output"
    )
    _require(runtime_output == context["adapter_path"], "Runtime adapter output mismatch")

    generation = _require_exact_keys(
        runtime["generation"], RUNTIME_GENERATION_KEYS, label="runtime generation"
    )
    for field in RUNTIME_GENERATION_KEYS:
        _require_equal(
            generation[field], config.get(field), label=f"runtime generation.{field}"
        )

    peft = _require_exact_keys(runtime["peft"], RUNTIME_PEFT_KEYS, label="runtime peft")
    expected_peft = {
        "merged_model_path": config.get("peft_merged_model_path"),
        "r": config.get("peft_r"),
        "lora_alpha": config.get("peft_lora_alpha"),
        "lora_dropout": config.get("peft_lora_dropout"),
        "target_modules": config.get("peft_target_modules"),
    }
    _require(dict(peft) == expected_peft, "Runtime PEFT config mismatch")
    runtime_merged = _canonical_repo_path(
        context["repo_root"],
        str(peft["merged_model_path"]),
        label="runtime merged output",
        require_exists=False,
    )
    _require(runtime_merged == context["merged_path"], "Runtime merged output mismatch")

    rewards = _require_exact_keys(
        runtime["rewards"], RUNTIME_REWARD_KEYS, label="runtime rewards"
    )
    expected_weights = config.get("reward_weights")
    _require(
        rewards["reward_funcs"] == reward_funcs,
        "Runtime reward functions mismatch",
    )
    _require(rewards["reward_weights"] == expected_weights, "Runtime reward weights mismatch")
    if reward_funcs:
        _require(
            isinstance(expected_weights, list)
            and len(expected_weights) == len(reward_funcs),
            "Resolved reward functions/weights mismatch",
        )

    environment = _require_exact_keys(
        runtime["environment"],
        RUNTIME_ENVIRONMENT_KEYS,
        label="runtime environment",
    )
    policy_gpus = context["topology"]["policy_gpus"]
    expected_visible = ",".join(str(item) for item in policy_gpus)
    _require(environment["world_size"] == context["topology"]["world_size"], "Runtime world_size mismatch")
    _require(environment["process_index"] == 0, "Runtime config was not written by global rank 0")
    _require(environment["local_process_index"] == 0, "Runtime config was not written by local rank 0")
    _require(environment["device"] == "cuda:0", "Runtime rank-0 device must be cuda:0")
    _require(environment["n_gpu"] == 1, "Runtime rank must expose exactly one training GPU")
    _require(environment["cuda_available"] is True, "Runtime CUDA must be available")
    _require(
        environment["cuda_visible_devices"] == expected_visible,
        "Runtime CUDA_VISIBLE_DEVICES topology mismatch",
    )
    _require(
        environment["cuda_device_count"] == len(policy_gpus),
        "Runtime visible CUDA device count mismatch",
    )
    names = environment["cuda_device_names"]
    _require(
        isinstance(names, list)
        and len(names) == len(policy_gpus)
        and all(isinstance(name, str) and "A30" in name for name in names),
        "Runtime CUDA device names do not match the A30 topology",
    )

    if "judge" in expected_top:
        judge = _require_exact_keys(
            runtime["judge"], RUNTIME_JUDGE_KEYS, label="runtime judge"
        )
        expected_judge = {
            "url": config.get("judge_url"),
            "model": config.get("judge_model"),
            "timeout": config.get("judge_timeout"),
            "verbose": config.get("judge_verbose"),
            "sleep_seconds": config.get("judge_sleep_seconds", 0.0),
            "api_key_env": config.get("judge_api_key_env"),
            "max_retries": config.get("judge_max_retries"),
            "backoff_seconds": config.get("judge_backoff_seconds"),
            "tokenizer_path": config.get("judge_tokenizer_path"),
            "max_model_len": config.get("judge_max_model_len"),
            "max_completion_tokens": config.get("judge_max_completion_tokens"),
            "candidate_reserve_tokens": config.get("judge_candidate_reserve_tokens"),
            "boundary_margin_tokens": config.get("judge_boundary_margin_tokens"),
        }
        _require(dict(judge) == expected_judge, "Runtime judge config mismatch")
    return runtime


def _safetensors_tensor_keys(path: Path, *, label: str) -> list[str]:
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            _require(keys, f"{label} contains no tensors")
            for key in keys:
                shape = handle.get_slice(key).get_shape()
                _require(
                    shape
                    and all(isinstance(item, int) and item > 0 for item in shape),
                    f"{label} tensor {key!r} is empty",
                )
    except ExecutionReceiptError:
        raise
    except Exception as exc:  # noqa: BLE001 - malformed safetensors must be blocked
        raise ExecutionReceiptError(f"Unable to validate {label}") from exc
    return keys


def _validate_lora_config(
    adapter_config: Mapping[str, Any], context: Mapping[str, Any]
) -> None:
    repo_root = context["repo_root"]
    parent_path = context["parent_path"]
    config = context["config"]
    _require(adapter_config.get("peft_type") == "LORA", "adapter peft_type must be LORA")
    _require(
        adapter_config.get("task_type") == "CAUSAL_LM",
        "adapter task_type must be CAUSAL_LM",
    )
    base_value = adapter_config.get("base_model_name_or_path")
    _require(
        isinstance(base_value, str) and base_value != "",
        "adapter base_model_name_or_path is missing",
    )
    configured_parent = _canonical_repo_path(
        repo_root, base_value, label="adapter base model"
    )
    _require(configured_parent == parent_path, "adapter base model does not match sealed parent")
    expected_targets = config.get("peft_target_modules")
    observed_targets = adapter_config.get("target_modules")
    _require(
        isinstance(expected_targets, list)
        and len(expected_targets) == len(set(expected_targets)),
        "Resolved PEFT target modules are invalid",
    )
    _require(
        isinstance(observed_targets, list)
        and len(observed_targets) == len(set(observed_targets))
        and set(observed_targets) == set(expected_targets),
        "adapter target_modules mismatch",
    )
    for adapter_field, config_field in (
        ("r", "peft_r"),
        ("lora_alpha", "peft_lora_alpha"),
        ("lora_dropout", "peft_lora_dropout"),
    ):
        _require_equal(
            adapter_config.get(adapter_field),
            config.get(config_field),
            label=f"adapter {adapter_field}",
        )
    for field, expected in (
        ("bias", "none"),
        ("fan_in_fan_out", False),
        ("use_dora", False),
        ("use_rslora", False),
        ("inference_mode", True),
    ):
        _require(adapter_config.get(field) == expected, f"adapter {field} mismatch")
    for field, expected in PEFT_COMPATIBILITY_DEFAULTS.items():
        _require(
            adapter_config.get(field, expected) == expected,
            f"adapter {field} uses unsupported non-default semantics",
        )
    for field, expected in (
        ("alpha_pattern", {}),
        ("rank_pattern", {}),
        ("modules_to_save", None),
        ("layers_to_transform", None),
        ("layer_replication", None),
    ):
        _require(
            adapter_config.get(field, expected) == expected,
            f"adapter {field} uses unsupported non-default semantics",
        )


def _validate_adapter(
    context: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    repo_root = context["repo_root"]
    adapter_root = context["adapter_path"]
    fingerprint = _tree_fingerprint(repo_root, adapter_root, label="adapter root")
    stable: dict[str, dict[str, Any]] = {}
    for filename in ADAPTER_REQUIRED_JSON:
        path = adapter_root / filename
        _load_json(path, label=f"adapter {filename}")
        stable[filename] = _file_record(
            repo_root, path, label=f"adapter {filename}"
        )
    training_args = adapter_root / ADAPTER_REQUIRED_BINARY
    stable[ADAPTER_REQUIRED_BINARY] = _file_record(
        repo_root, training_args, label="adapter training_args.bin"
    )

    weight = adapter_root / ADAPTER_WEIGHT
    alternate_root_weights = {
        path.name
        for path in adapter_root.glob("adapter_model*")
        if path.name != ADAPTER_WEIGHT
    }
    _require(
        not alternate_root_weights,
        "Only the root adapter_model.safetensors weight artifact is allowed; "
        f"found {sorted(alternate_root_weights)!r}",
    )
    stable[ADAPTER_WEIGHT] = _file_record(
        repo_root, weight, label="adapter safetensors"
    )
    keys = _safetensors_tensor_keys(weight, label="Adapter safetensors")
    _require(
        any(_LORA_A_KEY_RE.search(key) for key in keys),
        "Adapter has no LoRA A tensor",
    )
    _require(
        any(_LORA_B_KEY_RE.search(key) for key in keys),
        "Adapter has no LoRA B tensor",
    )

    adapter_config = _load_json(
        adapter_root / "adapter_config.json", label="adapter_config.json"
    )
    _validate_lora_config(adapter_config, context)

    _validate_runtime_config(context)

    trainer_state = _load_json(
        adapter_root / "trainer_state.json", label="trainer_state.json"
    )
    global_step = trainer_state.get("global_step")
    _require(
        isinstance(global_step, int) and not isinstance(global_step, bool) and global_step > 0,
        "trainer_state.global_step must be a positive integer",
    )
    train_results = _load_json(
        adapter_root / "train_results.json", label="train_results.json"
    )
    _require_positive_number(
        train_results.get("train_samples"), label="train_results.train_samples"
    )
    _require_positive_number(
        train_results.get("train_runtime"), label="train_results.train_runtime"
    )
    return fingerprint, stable


def _validate_merged(
    repo_root: Path, merged_root: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    fingerprint = _tree_fingerprint(repo_root, merged_root, label="merged root")
    stable: dict[str, dict[str, Any]] = {}
    for filename in MERGED_REQUIRED_JSON:
        path = merged_root / filename
        _load_json(path, label=f"merged {filename}")
        stable[filename] = _file_record(repo_root, path, label=f"merged {filename}")

    single = merged_root / "model.safetensors"
    index = merged_root / "model.safetensors.index.json"
    _require(
        single.is_file() != index.is_file(),
        "Merged model must contain exactly one of model.safetensors or its index",
    )
    weight_records: list[dict[str, Any]] = []
    if single.is_file():
        _require(
            not list(merged_root.glob("model-*.safetensors")),
            "Single-file merged model must not contain unindexed shards",
        )
        weight_records.append(
            _file_record(repo_root, single, label="merged model.safetensors")
        )
        _safetensors_tensor_keys(single, label="Merged model.safetensors")
    else:
        index_payload = _load_json(index, label="merged safetensors index")
        stable[index.name] = _file_record(
            repo_root, index, label="merged safetensors index"
        )
        weight_map = index_payload.get("weight_map")
        _require(
            isinstance(weight_map, dict) and weight_map,
            "Merged safetensors index must contain a non-empty weight_map",
        )
        shard_names: set[str] = set()
        for tensor_name, shard_name in weight_map.items():
            _require(
                isinstance(tensor_name, str) and tensor_name,
                "Merged weight_map contains an invalid tensor name",
            )
            _require(
                isinstance(shard_name, str)
                and Path(shard_name).name == shard_name
                and shard_name.startswith("model-")
                and shard_name.endswith(".safetensors"),
                "Merged weight_map shard path must be a canonical root filename",
            )
            shard_names.add(shard_name)
        actual_shards = {
            path.name for path in merged_root.glob("model-*.safetensors") if path.is_file()
        }
        _require(actual_shards == shard_names, "Merged safetensors shard set mismatch")
        actual_tensor_locations: dict[str, str] = {}
        for shard_name in sorted(shard_names):
            for tensor_name in _safetensors_tensor_keys(
                merged_root / shard_name,
                label=f"Merged shard {shard_name}",
            ):
                _require(
                    tensor_name not in actual_tensor_locations,
                    f"Merged tensor {tensor_name!r} is duplicated across shards",
                )
                actual_tensor_locations[tensor_name] = shard_name
            weight_records.append(
                _file_record(
                    repo_root,
                    merged_root / shard_name,
                    label=f"merged shard {shard_name}",
                )
            )
        _require(
            set(actual_tensor_locations) == set(weight_map),
            "Merged safetensors tensor key set does not match weight_map",
        )
        for tensor_name, shard_name in actual_tensor_locations.items():
            _require(
                weight_map[tensor_name] == shard_name,
                f"Merged tensor {tensor_name!r} is stored in the wrong shard",
            )
    return fingerprint, stable, weight_records


def _manifest_context(
    run_manifest: str | Path,
    stage_id: str,
    repo_root: str | Path,
    *,
    require_pending: bool,
    require_adapter: bool = True,
) -> dict[str, Any]:
    _require(stage_id in STAGE_IDS, f"Unsupported receipt stage: {stage_id}")
    root = Path(repo_root).resolve()
    _require(root.is_dir(), f"Repository root does not exist: {root}")
    manifest_path = _canonical_repo_path(
        root, run_manifest, label="run manifest"
    )
    manifest = _load_json(manifest_path, label="run manifest")
    _require(manifest.get("schema_version") == 2, "Unsupported run manifest schema")
    run_id = manifest.get("run_id")
    _require(isinstance(run_id, str) and run_id, "Run manifest has no run_id")
    expected_run_root = _canonical_repo_path(
        root,
        Path("output/training/retrain_v2") / run_id,
        label="unique run root",
    )
    _require(
        manifest_path == expected_run_root / "run_manifest.json",
        "Run manifest is not at its unique canonical run path",
    )
    run_root = manifest_path.parent

    stages = manifest.get("stages")
    _require(isinstance(stages, dict), "Run manifest has no stages")
    stage = stages.get(stage_id)
    _require(isinstance(stage, dict), f"Run manifest has no {stage_id} stage")
    if require_pending:
        _require(stage.get("status") == "pending", f"{stage_id} is not pending")
    else:
        _require(stage.get("status") in {"pending", "sealed"}, f"{stage_id} has invalid status")

    config_record = stage.get("resolved_config")
    _require(isinstance(config_record, dict), f"{stage_id} has no resolved config")
    config_path = _canonical_repo_path(
        root, str(config_record.get("path", "")), label="resolved config"
    )
    try:
        config_path.relative_to(run_root)
    except ValueError as exc:
        raise ExecutionReceiptError("Resolved config is outside the unique run root") from exc
    config_sha = _sha256_file(config_path)
    _require(config_sha == config_record.get("sha256"), "Resolved config hash mismatch")
    config = _load_yaml(config_path, label="resolved config")

    verification = verify_parent(
        manifest_path,
        stage_id=stage_id,
        repo_root=root,
        allow_stage_outputs=True,
    )
    parent = verification.get("parent")
    _require(isinstance(parent, dict), "Parent verification returned no parent")
    parent_sha = parent.get("sha256")
    _require(
        isinstance(parent_sha, str) and _SHA256_RE.fullmatch(parent_sha) is not None,
        "Parent verification returned an invalid SHA",
    )
    parent_path = _canonical_repo_path(
        root, str(parent.get("path", "")), label="sealed parent"
    )
    configured_parent = _canonical_repo_path(
        root, str(config.get("model_name_or_path", "")), label="configured parent"
    )
    _require(configured_parent == parent_path, "Resolved config parent mismatch")

    binding = stage.get("data_binding")
    _require(isinstance(binding, dict) and binding, f"{stage_id} has no data binding")
    data_binding_sha = _canonical_sha256(binding)

    execution_record = manifest.get("execution_contract")
    _require(isinstance(execution_record, dict), "Run manifest has no execution contract")
    try:
        execution_result = verify_execution_contract(execution_record, root)
    except ExecutionContractError as exc:
        raise ExecutionReceiptError(str(exc)) from exc
    execution_sha = execution_result.get("contract_sha256")
    _require(
        execution_sha == execution_record.get("contract_sha256"),
        "Execution contract verification mismatch",
    )
    topology_record = execution_record.get("stage_topology")
    _require(isinstance(topology_record, dict), "Execution contract has no stage topology")
    stage_topology = topology_record.get(stage_id)
    _require(isinstance(stage_topology, dict), f"Execution contract has no {stage_id} topology")
    try:
        topology = validate_stage_topology(stage_id, stage_topology)
    except ExecutionContractError as exc:
        raise ExecutionReceiptError(str(exc)) from exc

    adapter_path = _canonical_repo_path(
        root,
        str(config.get("output_dir", "")),
        label="adapter root",
        require_exists=require_adapter,
    )
    merged_path = _canonical_repo_path(
        root,
        str(config.get("peft_merged_model_path", "")),
        label="merged root",
        require_exists=False,
    )
    _require(adapter_path == run_root / "adapters" / stage_id, "Unexpected adapter root")
    _require(merged_path == run_root / "merged" / stage_id, "Unexpected merged root")
    manifest_artifact = _canonical_repo_path(
        root,
        str(stage.get("artifact_path", "")),
        label="manifest stage artifact",
        require_exists=False,
    )
    _require(manifest_artifact == merged_path, "Manifest stage artifact path mismatch")
    return {
        "repo_root": root,
        "manifest_path": manifest_path,
        "run_root": run_root,
        "run_id": run_id,
        "stage_id": stage_id,
        "config_path": config_path,
        "config_sha256": config_sha,
        "config": config,
        "parent_sha256": parent_sha,
        "parent_path": parent_path,
        "data_binding_sha256": data_binding_sha,
        "data_binding": binding,
        "execution_contract_sha256": execution_sha,
        "topology": topology,
        "adapter_path": adapter_path,
        "merged_path": merged_path,
    }


def _receipt_path(context: Mapping[str, Any], receipt_type: str) -> Path:
    _require(receipt_type in {"training", "merge"}, "Invalid receipt type")
    return context["run_root"] / "receipts" / (
        f"{context['stage_id']}.{receipt_type}.json"
    )


def _require_stage_lock(
    run_manifest: str | Path, stage_id: str, repo_root: str | Path
) -> None:
    try:
        require_inherited_stage_lock(run_manifest, stage_id, repo_root)
    except StageLockError as exc:
        raise ExecutionReceiptError(str(exc)) from exc


def _merge_attestation_binding(
    context: Mapping[str, Any], training_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": context["run_id"],
        "stage_id": context["stage_id"],
        "base": {
            "path": _relative_path(
                context["repo_root"], context["parent_path"], label="sealed parent"
            ),
            "sha256": context["parent_sha256"],
        },
        "adapter": {
            "path": _relative_path(
                context["repo_root"], context["adapter_path"], label="adapter root"
            ),
            "sha256": training_receipt["lineage"]["adapter_fingerprint"]["sha256"],
        },
        "resolved_config": {
            "path": _relative_path(
                context["repo_root"], context["config_path"], label="resolved config"
            ),
            "sha256": context["config_sha256"],
        },
        "training_receipt": {
            "path": _relative_path(
                context["repo_root"],
                _receipt_path(context, "training"),
                label="training receipt",
            ),
            "file_sha256": _sha256_file(_receipt_path(context, "training")),
            "lineage_sha256": training_receipt["lineage_sha256"],
        },
        "execution_contract_sha256": context["execution_contract_sha256"],
        "data_binding_sha256": context["data_binding_sha256"],
    }


def _verify_expected_merge_attestation(
    context: Mapping[str, Any], training_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        return verify_merge_attestation(
            context["merged_path"],
            expected_binding=_merge_attestation_binding(context, training_receipt),
        )
    except MergeAttestationError as exc:
        raise ExecutionReceiptError(str(exc)) from exc


def _write_exclusive_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.parent.is_symlink(), f"Receipt directory must not be a symlink: {path.parent}")
    _require(not path.exists() and not path.is_symlink(), f"Receipt already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ExecutionReceiptError(f"Receipt already exists: {path}") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _receipt_payload(receipt_type: str, lineage: Mapping[str, Any]) -> dict[str, Any]:
    canonical_lineage = dict(lineage)
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_type": receipt_type,
        "recorded_at_utc": _utc_now(),
        "lineage": canonical_lineage,
        "lineage_sha256": _canonical_sha256(canonical_lineage),
    }


def _load_receipt(path: Path, *, expected_type: str) -> dict[str, Any]:
    receipt = _load_json(path, label=f"{expected_type} receipt")
    _require(
        set(receipt)
        == {
            "schema_version",
            "receipt_type",
            "recorded_at_utc",
            "lineage",
            "lineage_sha256",
        },
        "Receipt has unexpected fields",
    )
    _require(receipt.get("schema_version") == RECEIPT_SCHEMA_VERSION, "Unsupported receipt schema")
    _require(receipt.get("receipt_type") == expected_type, "Wrong receipt type")
    recorded_at = receipt.get("recorded_at_utc")
    _require(
        isinstance(recorded_at, str) and recorded_at.endswith("Z"),
        "Receipt timestamp must be UTC",
    )
    try:
        datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ExecutionReceiptError("Receipt timestamp is not ISO-8601") from exc
    lineage = receipt.get("lineage")
    _require(isinstance(lineage, dict), "Receipt lineage must be an object")
    sha = receipt.get("lineage_sha256")
    _require(
        isinstance(sha, str) and _SHA256_RE.fullmatch(sha) is not None,
        "Receipt lineage SHA is invalid",
    )
    _require(_canonical_sha256(lineage) == sha, "Receipt lineage hash mismatch")
    return receipt


def _training_lineage(context: Mapping[str, Any]) -> dict[str, Any]:
    adapter_fingerprint, stable_files = _validate_adapter(context)
    lineage = {
        "run_id": context["run_id"],
        "stage_id": context["stage_id"],
        "config_sha256": context["config_sha256"],
        "parent_sha256": context["parent_sha256"],
        "data_binding_sha256": context["data_binding_sha256"],
        "execution_contract_sha256": context["execution_contract_sha256"],
        "topology": context["topology"],
        "adapter_path": _relative_path(
            context["repo_root"], context["adapter_path"], label="adapter root"
        ),
        "adapter_fingerprint": adapter_fingerprint,
        "stable_files": stable_files,
    }
    if context["stage_id"] == "chk2":
        lineage["judge_attestations"] = _judge_attestation_lineage(context)
    return lineage


def _judge_attestation_lineage(context: Mapping[str, Any]) -> dict[str, Any]:
    """Offline-verify and bind both chk2 judge observations into lineage."""

    from jobs.retrain_v2.judge_attestation import (
        JudgeAttestationError,
        verify_judge_attestations,
    )

    try:
        verified = verify_judge_attestations(
            context["manifest_path"],
            repo_root=context["repo_root"],
            require_post=True,
        )
    except JudgeAttestationError as exc:
        raise ExecutionReceiptError(str(exc)) from exc
    result: dict[str, Any] = {"schema_version": 1}
    for phase in ("pre", "post"):
        binding = verified.get(f"{phase}_attestation")
        _require(
            isinstance(binding, dict)
            and set(binding)
            == {"path", "file_sha256", "canonical_payload_sha256"},
            f"Judge {phase} attestation binding is incomplete",
        )
        expected_path = context["run_root"] / "attestations" / f"judge.{phase}.json"
        actual_path = _canonical_repo_path(
            context["repo_root"],
            str(binding["path"]),
            label=f"judge {phase} attestation",
        )
        _require(
            actual_path == expected_path,
            f"Judge {phase} attestation path is not canonical",
        )
        file_record = _file_record(
            context["repo_root"],
            actual_path,
            label=f"judge {phase} attestation",
        )
        _require(
            binding["file_sha256"] == file_record["sha256"],
            f"Judge {phase} attestation file hash mismatch",
        )
        canonical_sha = binding["canonical_payload_sha256"]
        _require(
            isinstance(canonical_sha, str)
            and _SHA256_RE.fullmatch(canonical_sha) is not None,
            f"Judge {phase} attestation content hash is invalid",
        )
        result[phase] = {
            "path": file_record["path"],
            "file_sha256": file_record["sha256"],
            "canonical_payload_sha256": canonical_sha,
        }
    service_sha = verified.get("service_identity_sha256")
    _require(
        isinstance(service_sha, str) and _SHA256_RE.fullmatch(service_sha) is not None,
        "Judge attestation service identity hash is invalid",
    )
    result["service_identity_sha256"] = service_sha
    return result


def _verify_training_receipt(context: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _load_receipt(
        _receipt_path(context, "training"), expected_type="training"
    )
    expected_lineage = _training_lineage(context)
    _require(receipt["lineage"] == expected_lineage, "Training receipt lineage drift")
    return receipt


def _require_regular_nonempty_file(path: Path, *, label: str) -> None:
    _require(not path.is_symlink(), f"{label} must not be a symlink: {path}")
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ExecutionReceiptError(f"Unable to inspect {label}: {path}") from exc
    _require(size > 0, f"{label} is empty: {path}")


def _validate_partial_completion_logs(
    context: Mapping[str, Any], completion_root: Path
) -> None:
    """Allow only the deterministic GRPO completion logs beside checkpoints."""

    _require(
        context["stage_id"] in {"chk2", "chk4"}
        and context["config"].get("log_completions") is True,
        f"Unexpected partial directory in adapter root: {completion_root}",
    )
    _require(
        not completion_root.is_symlink() and completion_root.is_dir(),
        f"Unsafe partial completion log directory: {completion_root}",
    )
    try:
        entries = list(os.scandir(completion_root))
    except OSError as exc:
        raise ExecutionReceiptError(
            f"Unable to inspect partial completion logs: {completion_root}"
        ) from exc
    _require(entries, f"Partial completion log directory is empty: {completion_root}")
    for entry in entries:
        candidate = Path(entry.path)
        _require(
            not entry.is_symlink()
            and entry.is_file(follow_symlinks=False)
            and _COMPLETION_LOG_RE.fullmatch(entry.name) is not None,
            f"Unsafe partial completion log entry: {candidate}",
        )
        _require_regular_nonempty_file(
            candidate, label="partial completion parquet"
        )


def _validate_resume_checkpoint(context: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the newest Trainer checkpoint as a deterministic resume point.

    Optimizer, scheduler, RNG and TrainingArguments artifacts are pickle-based in
    the pinned Transformers stack.  This boundary proves their presence and byte
    identity, while the Trainer remains responsible for deserializing them.
    """

    adapter_root = context["adapter_path"]
    _require(not adapter_root.is_symlink(), "Adapter root must not be a symlink")
    _require(adapter_root.is_dir(), f"Adapter root is missing: {adapter_root}")
    checkpoint_entries: list[tuple[int, Path]] = []
    try:
        entries = list(os.scandir(adapter_root))
    except OSError as exc:
        raise ExecutionReceiptError(
            f"Unable to inspect adapter root: {adapter_root}"
        ) from exc
    for entry in entries:
        candidate = Path(entry.path)
        _require(
            not entry.is_symlink(),
            f"Adapter root must not contain symlinks: {candidate}",
        )
        if entry.is_dir(follow_symlinks=False):
            match = _CHECKPOINT_DIRECTORY_RE.fullmatch(entry.name)
            if match is None and entry.name == "completions":
                _validate_partial_completion_logs(context, candidate)
                continue
            _require(
                match is not None,
                f"Unexpected partial directory in adapter root: {candidate}",
            )
            checkpoint_entries.append((int(match.group(1)), candidate))
        else:
            _require(
                entry.is_file(follow_symlinks=False),
                f"Adapter root contains a non-regular entry: {candidate}",
            )
    _require(checkpoint_entries, "Adapter root has no verifiable resume checkpoint")
    checkpoint_entries.sort(key=lambda item: item[0])
    step, checkpoint = checkpoint_entries[-1]

    _tree_fingerprint(
        context["repo_root"], checkpoint, label="resume checkpoint"
    )
    required_files = (
        "adapter_config.json",
        ADAPTER_WEIGHT,
        "trainer_state.json",
        "training_args.bin",
        "optimizer.pt",
        "scheduler.pt",
    )
    for filename in required_files:
        _require_regular_nonempty_file(
            checkpoint / filename, label=f"checkpoint {filename}"
        )

    world_size = context["topology"]["world_size"]
    expected_rng = (
        {"rng_state.pth"}
        if world_size == 1
        else {f"rng_state_{index}.pth" for index in range(world_size)}
    )
    observed_rng = {
        path.name
        for path in checkpoint.glob("rng_state*.pth")
        if path.is_file() and not path.is_symlink()
    }
    _require(
        observed_rng == expected_rng,
        "Checkpoint RNG state set does not match the stage topology",
    )
    for filename in sorted(expected_rng):
        _require_regular_nonempty_file(
            checkpoint / filename, label=f"checkpoint {filename}"
        )

    adapter_config = _load_json(
        checkpoint / "adapter_config.json", label="checkpoint adapter config"
    )
    _validate_lora_config(adapter_config, context)
    tensor_keys = _safetensors_tensor_keys(
        checkpoint / ADAPTER_WEIGHT, label="Checkpoint adapter safetensors"
    )
    _require(
        any(_LORA_A_KEY_RE.search(key) for key in tensor_keys),
        "Checkpoint adapter has no LoRA A tensor",
    )
    _require(
        any(_LORA_B_KEY_RE.search(key) for key in tensor_keys),
        "Checkpoint adapter has no LoRA B tensor",
    )
    trainer_state = _load_json(
        checkpoint / "trainer_state.json", label="checkpoint trainer state"
    )
    _require(
        trainer_state.get("global_step") == step,
        "Checkpoint directory step does not match trainer_state.global_step",
    )
    _validate_runtime_config(context)
    fingerprint = _tree_fingerprint(
        context["repo_root"], checkpoint, label="resume checkpoint"
    )
    return {
        "path": _relative_path(
            context["repo_root"], checkpoint, label="resume checkpoint"
        ),
        "global_step": step,
        "fingerprint": fingerprint,
        "deserialization_boundary": (
            "Optimizer, scheduler, RNG, and training_args payload semantics are "
            "validated by the pinned Trainer when resume begins."
        ),
    }


def _merge_lineage(
    context: Mapping[str, Any],
    training_receipt: Mapping[str, Any],
    *,
    attestation: Mapping[str, Any],
) -> dict[str, Any]:
    merged_fingerprint, stable_files, weights = _validate_merged(
        context["repo_root"], context["merged_path"]
    )
    return {
        "run_id": context["run_id"],
        "stage_id": context["stage_id"],
        "config_sha256": context["config_sha256"],
        "parent_sha256": context["parent_sha256"],
        "data_binding_sha256": context["data_binding_sha256"],
        "execution_contract_sha256": context["execution_contract_sha256"],
        "topology": context["topology"],
        "adapter_sha256": training_receipt["lineage"]["adapter_fingerprint"][
            "sha256"
        ],
        "training_receipt_sha256": training_receipt["lineage_sha256"],
        "training_receipt_file_sha256": _sha256_file(
            _receipt_path(context, "training")
        ),
        "merge_attestation": {
            **_file_record(
                context["repo_root"],
                context["merged_path"] / "merge_attestation.json",
                label="merge attestation",
            ),
            "canonical_payload_sha256": attestation[
                "canonical_payload_sha256"
            ],
        },
        "merged_path": _relative_path(
            context["repo_root"], context["merged_path"], label="merged root"
        ),
        "merged_fingerprint": merged_fingerprint,
        "stable_files": stable_files,
        "weights": weights,
    }


def record_training_receipt(
    run_manifest: str | Path, stage_id: str, repo_root: str | Path
) -> dict[str, Any]:
    """Record immutable evidence that a stable adapter was produced by a stage."""

    _require_stage_lock(run_manifest, stage_id, repo_root)
    context = _manifest_context(
        run_manifest, stage_id, repo_root, require_pending=True
    )
    path = _receipt_path(context, "training")
    _require(not path.exists() and not path.is_symlink(), f"Receipt already exists: {path}")
    receipt = _receipt_payload("training", _training_lineage(context))
    _write_exclusive_atomic(path, receipt)
    return {"status": "recorded", "path": str(path), **receipt}


def record_merge_receipt(
    run_manifest: str | Path, stage_id: str, repo_root: str | Path
) -> dict[str, Any]:
    """Record immutable merge lineage after revalidating the training receipt."""

    _require_stage_lock(run_manifest, stage_id, repo_root)
    context = _manifest_context(
        run_manifest, stage_id, repo_root, require_pending=True
    )
    path = _receipt_path(context, "merge")
    _require(not path.exists() and not path.is_symlink(), f"Receipt already exists: {path}")
    training_receipt = _verify_training_receipt(context)
    attestation = _verify_expected_merge_attestation(context, training_receipt)
    receipt = _receipt_payload(
        "merge",
        _merge_lineage(
            context, training_receipt, attestation=attestation
        ),
    )
    _write_exclusive_atomic(path, receipt)
    return {"status": "recorded", "path": str(path), **receipt}


def merge_and_record_receipt(
    run_manifest: str | Path,
    stage_id: str,
    repo_root: str | Path,
    *,
    merge_executor=None,
) -> dict[str, Any]:
    """Merge, attest, publish, and receipt one stage under its inherited lock.

    If a process was killed after the atomic publish but before receipt
    publication, the only recovery path is a complete in-directory attestation
    revalidation against the current immutable inputs.
    """

    _require_stage_lock(run_manifest, stage_id, repo_root)
    context = _manifest_context(
        run_manifest, stage_id, repo_root, require_pending=True
    )
    training_receipt = _verify_training_receipt(context)
    merge_path = _receipt_path(context, "merge")
    if merge_path.exists() or merge_path.is_symlink():
        verified = verify_stage_receipts(run_manifest, stage_id, repo_root)
        return {"status": "already_recorded", **verified}
    binding = _merge_attestation_binding(context, training_receipt)
    recovered_after_publish = context["merged_path"].exists()
    if not recovered_after_publish:
        from jobs.retrain_v2.merge_adapter import _merge_model, merge_from_config

        executor = merge_executor or _merge_model
        merge_from_config(
            context["config_path"],
            merge_executor=executor,
            merge_attestation_binding=binding,
            repo_root=context["repo_root"],
        )
    receipt = record_merge_receipt(run_manifest, stage_id, repo_root)
    return {
        "status": (
            "recovered_after_attested_publish"
            if recovered_after_publish
            else "merged_and_recorded"
        ),
        "merge_receipt_sha256": receipt["lineage_sha256"],
        "merge_attestation_sha256": _sha256_file(
            context["merged_path"] / "merge_attestation.json"
        ),
        "merged_sha256": receipt["lineage"]["merged_fingerprint"]["sha256"],
    }


def stage_recovery_state(
    run_manifest: str | Path, stage_id: str, repo_root: str | Path
) -> dict[str, Any]:
    """Return the only safe next action for one stage under its inherited lock."""

    _require_stage_lock(run_manifest, stage_id, repo_root)
    context = _manifest_context(
        run_manifest,
        stage_id,
        repo_root,
        require_pending=False,
        require_adapter=False,
    )
    manifest = _load_json(context["manifest_path"], label="run manifest")
    stage = manifest["stages"][stage_id]
    if stage["status"] == "sealed":
        verified = verify_stage_receipts(run_manifest, stage_id, repo_root)
        return {
            "state": "sealed_complete",
            **{key: value for key, value in verified.items() if key != "status"},
        }

    training_path = _receipt_path(context, "training")
    merge_path = _receipt_path(context, "merge")
    adapter_present = (
        context["adapter_path"].exists() or context["adapter_path"].is_symlink()
    )
    merged_present = (
        context["merged_path"].exists() or context["merged_path"].is_symlink()
    )
    training_present = training_path.exists() or training_path.is_symlink()
    merge_present = merge_path.exists() or merge_path.is_symlink()

    _require(
        not merge_present or training_present,
        "Merge receipt exists without its training receipt",
    )
    if training_present:
        _require(adapter_present, "Training receipt exists without its adapter")
        training_receipt = _verify_training_receipt(context)
        if merge_present:
            _require(merged_present, "Merge receipt exists without its merged model")
            verified = verify_stage_receipts(run_manifest, stage_id, repo_root)
            return {
                "state": "merge_receipt_ready_to_seal",
                **{key: value for key, value in verified.items() if key != "status"},
            }
        if merged_present:
            attestation = _verify_expected_merge_attestation(
                context, training_receipt
            )
            _merge_lineage(
                context, training_receipt, attestation=attestation
            )
            return {
                "state": "attested_merge_ready_to_record",
                "run_id": context["run_id"],
                "stage_id": stage_id,
                "merge_attestation_sha256": _sha256_file(
                    context["merged_path"] / "merge_attestation.json"
                ),
                "output_payload_sha256": attestation[
                    "output_payload_fingerprint"
                ]["sha256"],
            }
        return {
            "state": "training_receipt_ready_to_merge",
            "run_id": context["run_id"],
            "stage_id": stage_id,
            "training_receipt_sha256": training_receipt["lineage_sha256"],
        }

    _require(not merged_present, "Merged output exists without a training receipt")
    if not adapter_present:
        return {
            "state": "fresh_train",
            "run_id": context["run_id"],
            "stage_id": stage_id,
        }

    final_markers = {
        "adapter_config.json",
        "trainer_state.json",
        "train_results.json",
        ADAPTER_REQUIRED_BINARY,
        ADAPTER_WEIGHT,
    }
    if any((context["adapter_path"] / name).exists() for name in final_markers):
        # Recovery must be able to identify a fully written chk2 adapter when
        # training finished but the live post-judge attestation was interrupted.
        # The subsequent record-training boundary still builds the complete
        # lineage and therefore still requires and binds both attestations.
        adapter_fingerprint, _stable_files = _validate_adapter(context)
        return {
            "state": "training_complete_ready_to_record",
            "run_id": context["run_id"],
            "stage_id": stage_id,
            "adapter_sha256": adapter_fingerprint["sha256"],
        }
    checkpoint = _validate_resume_checkpoint(context)
    return {
        "state": "checkpoint_resume_ready",
        "run_id": context["run_id"],
        "stage_id": stage_id,
        "checkpoint": checkpoint,
    }


def verify_stage_receipts(
    run_manifest: str | Path, stage_id: str, repo_root: str | Path
) -> dict[str, Any]:
    """Recompute and verify training and merge receipt lineage for one stage."""

    context = _manifest_context(
        run_manifest, stage_id, repo_root, require_pending=False
    )
    training_receipt = _verify_training_receipt(context)
    merge_receipt = _load_receipt(
        _receipt_path(context, "merge"), expected_type="merge"
    )
    attestation = _verify_expected_merge_attestation(context, training_receipt)
    expected_merge = _merge_lineage(
        context, training_receipt, attestation=attestation
    )
    _require(merge_receipt["lineage"] == expected_merge, "Merge receipt lineage drift")
    return {
        "status": "verified",
        "run_id": context["run_id"],
        "stage_id": stage_id,
        "training_receipt_sha256": training_receipt["lineage_sha256"],
        "merge_receipt_sha256": merge_receipt["lineage_sha256"],
        "adapter_sha256": training_receipt["lineage"]["adapter_fingerprint"][
            "sha256"
        ],
        "merged_sha256": merge_receipt["lineage"]["merged_fingerprint"][
            "sha256"
        ],
    }


class _BlockedArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print(
            json.dumps(
                {"status": "blocked", "error": message},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        self.exit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = _BlockedArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("record-training", "record-merge", "merge-and-record", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--run-manifest", type=Path, required=True)
        command.add_argument("--stage", choices=STAGE_IDS, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    functions = {
        "record-training": record_training_receipt,
        "record-merge": record_merge_receipt,
        "merge-and-record": merge_and_record_receipt,
        "verify": verify_stage_receipts,
    }
    try:
        result = functions[args.command](
            args.run_manifest, args.stage, args.repo_root
        )
    except Exception as exc:  # noqa: BLE001 - CLI is a strict fail-closed boundary
        print(
            json.dumps(
                {"status": "blocked", "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ExecutionReceiptError",
    "merge_and_record_receipt",
    "record_merge_receipt",
    "record_training_receipt",
    "stage_recovery_state",
    "verify_stage_receipts",
]
