"""Fail-closed preflight and launcher for the paper Model chk-2 Minutes SFT.

This is a fresh-only, two-GPU training branch.  It verifies the immutable
recovery release, exact chk-1 checkpoint-200 parent authorization, runtime
environment, GPU availability, and the fixed three-epoch schedule before
execing the repository's standard SFT trainer.  It never publishes data,
resumes a checkpoint, selects an adapter, or merges a model.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO

import yaml
from packaging.utils import canonicalize_name

from open_r1.configs import LoraArguments, ModelConfig, SFTConfig, SFTScriptArguments
from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.validator.loo_generation_spec import validate_manifest_integrity


REPO_ROOT = Path(__file__).resolve().parents[2]
BRANCH_ID = "paper_chk2_chk1_cp200_minutes_v6_recovery_full3ep_lr1e6_v1_20260901"
TRAINING_SCOPE = "paper-chk2-chk1-cp200-minutes-sft-v6-downstream128"
CONFIG = Path(
    "configs/retrain_v2/"
    "paper_chk2_minutes_sft_chk1_cp200_v6_recovery_full3ep_lr1e6_20260901.yaml"
)
RELEASE_ROOT = Path(
    "dataset/processed/retrain_v2/"
    "chk2_chk1_final_analysis_synthetic_minutes_flash_official_reference_"
    "v6_downstream128_recovery_v1_20260831"
)
RELEASE_MANIFEST = RELEASE_ROOT / "release_manifest.json"
MANIFEST_PLACEHOLDER = "__PAPER_CHK2_RELEASE_MANIFEST_SHA256__"
RUN_ROOT = Path("output/training/retrain_v2") / BRANCH_ID
OUTPUT_DIR = RUN_ROOT / "adapters/chk2"
PROHIBITED_MERGE = RUN_ROOT / "merged/unselected_root_adapter_prohibited"
PREFLIGHT_RECEIPT = RUN_ROOT / "preflight_receipt.json"
RUNTIME_LAUNCH_RECEIPT = RUN_ROOT / "runtime_launch_receipt.json"
PARENT_MODEL = Path(
    "output/training/retrain_v2/"
    "chk1_clean_v2_lr1e6_selected_cp200_for_chk2_20260810/merged/chk1"
)
PARENT_SHA256 = "0989b94792f8ab6377e2979aaedeb37010b9a2c5e05c77a459b4e5cda5b806e3"
PARENT_CHECKPOINT_MANIFEST = Path(
    "docs/summary/20260810T124500Z/chk1_cp200_merge_for_chk2/checkpoint_manifest.json"
)
PARENT_CHECKPOINT_MANIFEST_SHA256 = (
    "59758ab4d2b54592f632776c14d2b06c21c56416e4eb83e09d896f6a06d0a687"
)
PARENT_AUTHORIZATION = Path(
    "docs/summary/20260810T124500Z/chk1_cp200_merge_for_chk2/"
    "chk1_cp200_to_chk2_authorization.json"
)
PARENT_AUTHORIZATION_SHA256 = (
    "e4e80fbfdd805b3691b3b05fbf3e3f26353eeef399fb7a08f156260e7f0ca554"
)

EXPECTED_PYTHON = Path("/home/haobin_cui/.conda/envs/fomc_trainer/bin/python")
EXPECTED_PYTHON_VERSION = (3, 10, 9)
EXPECTED_CORE_VERSIONS = {
    "accelerate": "1.4.0",
    "bitsandbytes": "0.48.2",
    "datasets": "4.8.4",
    "liger-kernel": "0.8.1",
    "peft": "0.15.2",
    "tokenizers": "0.22.2",
    "torch": "2.10.0+cu128",
    "transformers": "4.57.6",
    "trl": "1.2.0",
}
TRAIN_FREEZE = Path("requirements/retrain_v2_train.freeze.txt")
FORBIDDEN_TRAIN_DISTRIBUTIONS = {
    "deepspeed",
    "e2b",
    "e2b-code-interpreter",
    "flash-attn",
    "lighteval",
    "vllm",
    "xformers",
}
EXPECTED_SPLIT_COUNTS = {"train": 305, "validation": 42, "test": 44}
EXPECTED_WORLD_SIZE = 2
EXPECTED_TRAIN_EPOCHS = 3
EXPECTED_GRADIENT_ACCUMULATION = 8
EXPECTED_OPTIMIZER_STEPS = 60
EXPECTED_GPU_IDS = (0, 1)
EXPECTED_GPU_NAME = "NVIDIA A30"
MIN_GPU_TOTAL_MIB = 24_000
# The same 8B NF4 QLoRA/r32 training profile was measured at 14.619 GiB
# reserved memory with a materially longer 7,168-token sequence.  Requiring
# 17 GiB free leaves about 2.4 GiB above that measured peak for CUDA/NCCL and
# allocator variation while allowing the user's existing small WGAN workers
# to remain resident.  Utilization is intentionally not a gate: co-tenancy is
# explicitly authorized, but memory below this threshold still fails closed.
REFERENCE_QLORA_PEAK_RESERVED_MIB = 14_970
MIN_GPU_FREE_MIB = 17 * 1024
GPU_MEMORY_HEADROOM_MIB = MIN_GPU_FREE_MIB - REFERENCE_QLORA_PEAK_RESERVED_MIB
GPU_MEMORY_REFERENCE = "docs/summary/20260803T150258Z/fomc_retraining_v2_plan.md"
MIN_DISK_FREE_GIB = 100

STUDENT_SYSTEM_PROMPT = """\
You are a Federal Reserve Minutes editor. The user supplies a complete
economic or financial analysis. Use the native reasoning section for the
complete reasoning process, including any useful deliberation about the task,
prompt, JSON transport, answer contract, length, or drafting. Within that full
reasoning trace, identify every substantive claim, quantity, date, direction,
comparison, attribution, causal relation, and expression of uncertainty that
the formal rewrite must preserve. Then express the same information as exactly
one formal FOMC Minutes paragraph.

Do not add, remove, broaden, narrow, or contradict any substantive claim.
Preserve the numeric-quantity multiset and every explicit calendar reference.
The final paragraph may reuse phrases, sentences, or extensive wording from
the analysis when that wording is already suitable; lexical overlap is not an
error. The final paragraph as a whole must not be a verbatim copy of the whole
analysis. Do not emit headings, lists, JSON, citations, answer tags, or
model-control tags in the final paragraph.
"""
STUDENT_SYSTEM_PROMPT_SHA256 = (
    "4730a4ed585238547447ab850836db5a9fc67e5c3b1328b485c88d701ae78c4e"
)
STUDENT_USER_PROMPT_TEMPLATE = (
    "Rewrite the following analysis as formal FOMC Minutes prose:\n\n"
    '{"analysis":"[SOURCE_ANALYSIS]"}'
)
STUDENT_USER_PROMPT_TEMPLATE_SHA256 = (
    "423e79849cb66d6361c986f705ea4d4b16a03e28fec5826113eb9c8030a976d0"
)

IMPLEMENTATION_RELATIVES = (
    CONFIG.as_posix(),
    "jobs/retrain_v2/paper_chk2_sft_launch.py",
    "jobs/train/train_sft.py",
    "run/retrain_v2/start_paper_chk2_minutes_sft_v6_recovery.sh",
    "src/open_r1/configs.py",
    "src/open_r1/data_loader.py",
    "src/open_r1/trainer/dataset_release.py",
    "src/open_r1/trainer/sft_prompt_renderer.py",
    "src/open_r1/trainer/sft_trainer.py",
    "src/open_r1/trainer/trainer.py",
)

_EXEC_LOCK: IO[str] | None = None


class PaperChk2LaunchError(RuntimeError):
    """The paper chk-2 full SFT is not safely launchable."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PaperChk2LaunchError(message)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    """Create a private receipt without following or replacing any path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise PaperChk2LaunchError(
            f"create-only receipt already exists: {path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # The exclusive create prevents overwriting evidence.  A partial file
        # remains fail-closed and is never silently replaced.
        raise


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(
        path.is_file() and not path.is_symlink(), f"missing or unsafe {label}: {path}"
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PaperChk2LaunchError(f"invalid {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    _require(
        path.is_file() and not path.is_symlink(), f"missing or unsafe config: {path}"
    )
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PaperChk2LaunchError(f"invalid config: {path}") from exc
    _require(isinstance(value, dict), "paper chk-2 config must be a mapping")
    return value


def _resolve(repo_root: Path, value: object, *, label: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{label} path missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _expected_config(manifest_sha256: str) -> dict[str, Any]:
    return {
        "model_name_or_path": PARENT_MODEL.as_posix(),
        "model_revision": None,
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "use_bnb_nested_quant": True,
        "bnb_4bit_quant_storage": "bfloat16",
        "dataset_name": (RELEASE_ROOT / "minutes_alignment").as_posix(),
        "dataset_paper_chk2_scope": TRAINING_SCOPE,
        "dataset_paper_chk2_release_manifest": RELEASE_MANIFEST.as_posix(),
        "dataset_paper_chk2_release_manifest_sha256": manifest_sha256,
        "dataset_prompt_column": "prompt",
        "dataset_train_split": "train",
        "dataset_test_split": "validation",
        "system_prompt": STUDENT_SYSTEM_PROMPT,
        "bf16": True,
        "tf32": True,
        "optim": "paged_adamw_8bit",
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "learning_rate": 1.0e-6,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "packing": False,
        "completion_only_loss": True,
        "max_length": 4096,
        "do_eval": True,
        "eval_strategy": "steps",
        "eval_steps": 10,
        "max_steps": -1,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "ddp_find_unused_parameters": False,
        "dataloader_num_workers": 2,
        "dataloader_pin_memory": True,
        "peft_merged_model_path": PROHIBITED_MERGE.as_posix(),
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
        "output_dir": OUTPUT_DIR.as_posix(),
        "overwrite_output_dir": False,
        "resume_from_checkpoint": None,
        "save_strategy": "steps",
        "save_steps": 1,
        "save_total_limit": None,
        "save_only_model": False,
        "checkpoint_keep_last": 3,
        "checkpoint_keep_every_n_steps": 10,
        "callbacks": ["loss_log", "checkpoint_retention"],
        "log_level": "info",
        "logging_strategy": "steps",
        "logging_steps": 1,
        "logging_first_step": True,
        "push_to_hub": False,
        "report_to": [],
        "seed": 42,
        "data_seed": 42,
        "use_cache": False,
        "use_liger_kernel": True,
        "activation_offloading": False,
    }


def _validate_parser_schema(config: Mapping[str, Any]) -> None:
    recognized = {
        field.name
        for dataclass_type in (
            SFTScriptArguments,
            SFTConfig,
            ModelConfig,
            LoraArguments,
        )
        for field in fields(dataclass_type)
    }
    unknown = sorted(set(config) - recognized)
    _require(not unknown, f"TrlParser schema does not recognize config keys: {unknown}")


def _config_contract(repo_root: Path) -> tuple[Path, dict[str, Any], str]:
    path = (repo_root / CONFIG).resolve()
    config = _read_yaml(path)
    manifest_sha = config.get("dataset_paper_chk2_release_manifest_sha256")
    _require(
        isinstance(manifest_sha, str)
        and manifest_sha != MANIFEST_PLACEHOLDER
        and len(manifest_sha) == 64
        and all(character in "0123456789abcdef" for character in manifest_sha),
        "formal paper chk-2 release manifest SHA-256 has not been pinned",
    )
    expected = _expected_config(manifest_sha)
    _require(set(config) == set(expected), "paper chk-2 config key set drift")
    for key, expected_value in expected.items():
        _require(config[key] == expected_value, f"paper chk-2 config {key} drift")
    _require(
        _sha256_text(config["system_prompt"]) == STUDENT_SYSTEM_PROMPT_SHA256,
        "paper chk-2 student system prompt hash drift",
    )
    _require(
        _resolve(repo_root, config["model_name_or_path"], label="parent")
        == (repo_root / PARENT_MODEL).resolve(),
        "paper chk-2 parent path drift",
    )
    _require(
        _resolve(repo_root, config["dataset_name"], label="dataset")
        == (repo_root / RELEASE_ROOT / "minutes_alignment").resolve(),
        "paper chk-2 dataset path drift",
    )
    _require(
        _resolve(repo_root, config["output_dir"], label="output")
        == (repo_root / OUTPUT_DIR).resolve(),
        "paper chk-2 output path drift",
    )
    _validate_parser_schema(config)
    return path, config, manifest_sha


def _verify_environment() -> dict[str, Any]:
    executable = Path(sys.executable).absolute()
    _require(
        executable == EXPECTED_PYTHON, "launcher must use exact fomc_trainer Python"
    )
    _require(
        sys.version_info[:3] == EXPECTED_PYTHON_VERSION,
        "fomc_trainer Python version drift",
    )
    observed: dict[str, str] = {}
    for package, expected in EXPECTED_CORE_VERSIONS.items():
        try:
            observed[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise PaperChk2LaunchError(f"missing core package: {package}") from exc
        _require(
            observed[package] == expected,
            f"core package version drift for {package}: expected {expected}, "
            f"observed {observed[package]}",
        )
    inventory: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if isinstance(name, str) and name:
            canonical = canonicalize_name(name)
            _require(
                canonical not in inventory,
                f"duplicate installed distribution: {canonical}",
            )
            inventory[canonical] = distribution.version
    freeze_path = REPO_ROOT / TRAIN_FREEZE
    _require(
        freeze_path.is_file() and not freeze_path.is_symlink(),
        "training environment freeze is missing or unsafe",
    )
    required: dict[str, str] = {}
    for line_number, raw in enumerate(
        freeze_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        _require(
            line.count("==") == 1 and ";" not in line and " " not in line,
            f"invalid exact train freeze row {line_number}",
        )
        name, version = line.split("==", 1)
        canonical = canonicalize_name(name)
        _require(
            canonical not in required and bool(version),
            f"invalid train freeze row {line_number}",
        )
        required[canonical] = version
    missing = sorted(set(required) - set(inventory))
    mismatched = {
        name: {"expected": required[name], "observed": inventory[name]}
        for name in sorted(set(required) & set(inventory))
        if required[name] != inventory[name]
    }
    _require(not missing, f"required training distributions are missing: {missing}")
    _require(
        not mismatched, f"required training distribution versions drifted: {mismatched}"
    )
    forbidden = sorted(FORBIDDEN_TRAIN_DISTRIBUTIONS & set(inventory))
    _require(
        not forbidden,
        f"forbidden judge/legacy distributions are installed: {forbidden}",
    )
    extras = sorted(set(inventory) - set(required))
    return {
        "python": {
            "executable": str(executable),
            "version": ".".join(map(str, EXPECTED_PYTHON_VERSION)),
        },
        "core_versions": observed,
        "required_freeze": {
            "path": str(freeze_path.resolve()),
            "sha256": sha256_file(freeze_path),
            "distribution_count": len(required),
        },
        "installed_distributions": dict(sorted(inventory.items())),
        "unrelated_extra_distributions": extras,
        "unrelated_extra_distribution_count": len(extras),
        "forbidden_distributions_present": [],
    }


def _parse_gpu_row(raw: str) -> dict[str, Any]:
    fields_raw = [part.strip() for part in raw.split(",")]
    _require(len(fields_raw) == 6, f"invalid nvidia-smi GPU row: {raw!r}")
    try:
        index, total, used, free, utilization = (
            int(fields_raw[0]),
            int(fields_raw[2]),
            int(fields_raw[3]),
            int(fields_raw[4]),
            int(fields_raw[5]),
        )
    except ValueError as exc:
        raise PaperChk2LaunchError(f"invalid numeric GPU telemetry: {raw!r}") from exc
    return {
        "index": index,
        "name": fields_raw[1],
        "memory_total_mib": total,
        "memory_used_mib": used,
        "memory_free_mib": free,
        "utilization_percent": utilization,
    }


def _parse_compute_process_row(raw: str) -> dict[str, Any]:
    try:
        fields_raw = next(csv.reader([raw], skipinitialspace=True))
    except (csv.Error, StopIteration) as exc:
        raise PaperChk2LaunchError(
            f"invalid nvidia-smi compute-process row: {raw!r}"
        ) from exc
    _require(
        len(fields_raw) == 3,
        f"invalid nvidia-smi compute-process row: {raw!r}",
    )
    try:
        pid = int(fields_raw[0].strip())
        used_memory_mib = int(fields_raw[2].strip())
    except ValueError as exc:
        raise PaperChk2LaunchError(
            f"invalid numeric compute-process telemetry: {raw!r}"
        ) from exc
    _require(pid > 0 and used_memory_mib >= 0, "invalid compute-process telemetry")
    process_name = fields_raw[1].strip()
    _require(bool(process_name), "compute-process name is empty")
    return {
        "pid": pid,
        "process_name": process_name,
        "used_memory_mib": used_memory_mib,
    }


def _run_nvidia_smi(arguments: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PaperChk2LaunchError("unable to query GPU inventory") from exc
    return result.stdout


def _verify_gpus() -> list[dict[str, Any]]:
    output = _run_nvidia_smi(
        (
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        )
    )
    rows = [_parse_gpu_row(line) for line in output.splitlines() if line.strip()]
    by_index = {row["index"]: row for row in rows}
    _require(len(by_index) == len(rows), "duplicate GPU indices returned by nvidia-smi")
    selected: list[dict[str, Any]] = []
    for gpu_id in EXPECTED_GPU_IDS:
        _require(gpu_id in by_index, f"required GPU {gpu_id} is missing")
        row = dict(by_index[gpu_id])
        _require(row["name"] == EXPECTED_GPU_NAME, f"GPU {gpu_id} is not an A30")
        _require(
            row["memory_total_mib"] >= MIN_GPU_TOTAL_MIB,
            f"GPU {gpu_id} memory size drift",
        )
        _require(
            row["memory_free_mib"] >= MIN_GPU_FREE_MIB,
            f"GPU {gpu_id} has insufficient free memory for shared QLoRA: "
            f"required {MIN_GPU_FREE_MIB} MiB, observed "
            f"{row['memory_free_mib']} MiB",
        )
        processes = _run_nvidia_smi(
            (
                f"--id={gpu_id}",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            )
        )
        active = [
            _parse_compute_process_row(line)
            for line in processes.splitlines()
            if line.strip()
        ]
        _require(
            len({process["pid"] for process in active}) == len(active),
            f"GPU {gpu_id} returned duplicate compute-process PIDs",
        )
        row["active_compute_processes"] = active
        row["active_compute_pids"] = [process["pid"] for process in active]
        row["active_compute_process_memory_mib"] = sum(
            process["used_memory_mib"] for process in active
        )
        row["co_tenancy_observed"] = bool(active)
        row["shared_memory_headroom_mib"] = row["memory_free_mib"] - MIN_GPU_FREE_MIB
        selected.append(row)
    return selected


def _gpu_allocation_policy() -> dict[str, Any]:
    return {
        "mode": "shared-remaining-memory-v1",
        "selected_gpu_ids": list(EXPECTED_GPU_IDS),
        "existing_compute_processes_allowed": True,
        "high_utilization_allowed": True,
        "minimum_free_memory_per_gpu_mib": MIN_GPU_FREE_MIB,
        "reference_qlora_peak_reserved_mib": REFERENCE_QLORA_PEAK_RESERVED_MIB,
        "safety_headroom_mib": GPU_MEMORY_HEADROOM_MIB,
        "reference": GPU_MEMORY_REFERENCE,
        "external_processes_must_not_be_signaled_or_stopped": True,
    }


def _verify_parent(repo_root: Path) -> dict[str, Any]:
    auth_path = (repo_root / PARENT_AUTHORIZATION).resolve()
    checkpoint_path = (repo_root / PARENT_CHECKPOINT_MANIFEST).resolve()
    _require(
        sha256_file(auth_path) == PARENT_AUTHORIZATION_SHA256,
        "chk1-cp200-to-chk2 authorization file drift",
    )
    _require(
        sha256_file(checkpoint_path) == PARENT_CHECKPOINT_MANIFEST_SHA256,
        "chk1 cp200 checkpoint manifest file drift",
    )
    authorization = _read_json(auth_path, label="chk1-cp200-to-chk2 authorization")
    checkpoint = _read_json(checkpoint_path, label="chk1 cp200 checkpoint manifest")
    validate_manifest_integrity(checkpoint)
    _require(
        authorization.get("schema_version")
        == "chk1-cp200-to-chk2-override-authorization-v1"
        and authorization.get("status") == "authorized",
        "chk1 cp200 authorization status drift",
    )
    scope = authorization.get("scope")
    _require(
        isinstance(scope, dict)
        and scope.get("target_stage") == "chk2"
        and scope.get("downstream_stages_allowed") == ["chk2"]
        and scope.get("further_downstream_stages_allowed") == [],
        "chk1 cp200 authorization is not restricted to chk2",
    )
    _require(
        checkpoint.get("scope")
        == {"allowed_stage": "chk2", "further_downstream_stages_allowed": []},
        "chk1 cp200 checkpoint scope drift",
    )
    declared = checkpoint.get("model_fingerprint")
    _require(
        isinstance(declared, dict)
        and declared.get("sha256") == PARENT_SHA256
        and checkpoint.get("model_path") == PARENT_MODEL.as_posix(),
        "chk1 cp200 checkpoint parent binding drift",
    )
    parent_path = (repo_root / PARENT_MODEL).resolve()
    observed = fingerprint_artifact_path(parent_path)
    _require(
        observed.get("sha256") == PARENT_SHA256, "exact chk1 cp200 parent digest drift"
    )
    _require(observed == declared, "chk1 cp200 parent fingerprint/manifest drift")
    return {
        "model": observed,
        "checkpoint_manifest": {
            "path": str(checkpoint_path),
            "sha256": PARENT_CHECKPOINT_MANIFEST_SHA256,
        },
        "authorization": {
            "path": str(auth_path),
            "sha256": PARENT_AUTHORIZATION_SHA256,
        },
    }


def _verify_release(
    repo_root: Path, *, config: Mapping[str, Any], manifest_sha: str
) -> dict[str, Any]:
    from open_r1.trainer.dataset_release import verify_paper_chk2_sft_release

    manifest_path = (repo_root / RELEASE_MANIFEST).resolve()
    _require(
        manifest_path.is_file()
        and not manifest_path.is_symlink()
        and sha256_file(manifest_path) == manifest_sha,
        "paper chk-2 release manifest external pin drift",
    )
    verified = verify_paper_chk2_sft_release(
        dataset_dir=(repo_root / RELEASE_ROOT / "minutes_alignment").resolve(),
        manifest_path=manifest_path,
        expected_manifest_sha256=manifest_sha,
        training_scope=TRAINING_SCOPE,
        system_prompt=str(config["system_prompt"]),
        model_path=(repo_root / PARENT_MODEL).resolve(),
    )
    _require(
        isinstance(verified, dict),
        "paper chk-2 runtime verifier returned invalid receipt",
    )
    _require(
        verified.get("split_counts") == EXPECTED_SPLIT_COUNTS,
        "paper chk-2 release split counts drift",
    )
    split_files = verified.get("split_files")
    _require(
        isinstance(split_files, Mapping)
        and set(split_files) == {"train", "validation"},
        "paper chk-2 runtime may expose only train and validation split files",
    )
    _require(
        verified.get("test_verified_but_not_loaded") is True,
        "paper chk-2 test split is not sealed from the trainer",
    )
    return verified


def _optimizer_steps(train_rows: int) -> dict[str, int]:
    per_rank_batches = math.ceil(train_rows / EXPECTED_WORLD_SIZE)
    updates_per_epoch = math.ceil(per_rank_batches / EXPECTED_GRADIENT_ACCUMULATION)
    return {
        "train_rows": train_rows,
        "world_size": EXPECTED_WORLD_SIZE,
        "per_rank_batches_per_epoch": per_rank_batches,
        "gradient_accumulation_steps": EXPECTED_GRADIENT_ACCUMULATION,
        "updates_per_epoch": updates_per_epoch,
        "epochs": EXPECTED_TRAIN_EPOCHS,
        "optimizer_steps": updates_per_epoch * EXPECTED_TRAIN_EPOCHS,
        "effective_global_batch": EXPECTED_WORLD_SIZE * EXPECTED_GRADIENT_ACCUMULATION,
    }


def _implementation(repo_root: Path) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for relative in IMPLEMENTATION_RELATIVES:
        path = repo_root / relative
        _require(
            path.is_file() and not path.is_symlink(),
            f"implementation dependency missing: {relative}",
        )
        records[relative] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    return records


def preflight(repo_root: Path, *, verify_gpus: bool = True) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    _require(root == REPO_ROOT.resolve(), "paper chk-2 launcher repository root drift")
    _require(
        _sha256_text(STUDENT_SYSTEM_PROMPT) == STUDENT_SYSTEM_PROMPT_SHA256,
        "embedded system prompt drift",
    )
    _require(
        _sha256_text(STUDENT_USER_PROMPT_TEMPLATE)
        == STUDENT_USER_PROMPT_TEMPLATE_SHA256,
        "embedded user prompt template drift",
    )
    environment = _verify_environment()
    config_path, config, manifest_sha = _config_contract(root)
    parent = _verify_parent(root)
    release = _verify_release(root, config=config, manifest_sha=manifest_sha)
    schedule = _optimizer_steps(EXPECTED_SPLIT_COUNTS["train"])
    _require(
        schedule["optimizer_steps"] == EXPECTED_OPTIMIZER_STEPS,
        "paper chk-2 optimizer-step calculation drift",
    )
    output = (root / OUTPUT_DIR).resolve()
    prohibited_merge = (root / PROHIBITED_MERGE).resolve()
    _require(
        not output.exists() and not output.is_symlink(),
        "fresh-only adapter output exists",
    )
    _require(
        not prohibited_merge.exists() and not prohibited_merge.is_symlink(),
        "prohibited merge output exists",
    )
    disk = shutil.disk_usage(root)
    free_gib = disk.free // (1024**3)
    _require(
        free_gib >= MIN_DISK_FREE_GIB, f"insufficient disk space: {free_gib} GiB free"
    )
    gpu_inventory = _verify_gpus() if verify_gpus else []
    return {
        "schema_version": "paper-chk2-minutes-sft-preflight-v2",
        "status": "ready",
        "generated_at": _utc_now(),
        "branch_id": BRANCH_ID,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "release": {
            "manifest_path": str((root / RELEASE_MANIFEST).resolve()),
            "manifest_sha256": manifest_sha,
            "split_counts": EXPECTED_SPLIT_COUNTS,
            "runtime_binding_schema": release.get("schema_version"),
            "test_verified_but_not_loaded": True,
        },
        "parent": parent,
        "prompt_contract": {
            "system_prompt_sha256": STUDENT_SYSTEM_PROMPT_SHA256,
            "user_prompt_template_sha256": STUDENT_USER_PROMPT_TEMPLATE_SHA256,
        },
        "training": {
            **schedule,
            "fresh_only": True,
            "completion_only_loss": True,
            "packing": False,
            "max_length": 4096,
            "evaluation_split": "validation",
            "test_loaded": False,
            "resume": "forbidden",
            "merge": "forbidden",
            "smoke": "explicitly_skipped_by_user",
        },
        "environment": environment,
        "resources": {
            "disk_free_gib": free_gib,
            "gpu_allocation_policy": _gpu_allocation_policy(),
            "gpu_snapshot": {
                "captured_at": _utc_now(),
                "gpus": gpu_inventory,
                "co_tenancy_observed": any(
                    bool(row.get("co_tenancy_observed")) for row in gpu_inventory
                ),
            },
        },
        "implementation": _implementation(root),
    }


def training_command(repo_root: Path) -> dict[str, Any]:
    config_path, _config, _manifest_sha = _config_contract(repo_root.resolve())
    return {
        "status": "ready",
        "branch_id": BRANCH_ID,
        "gpus": list(EXPECTED_GPU_IDS),
        "num_processes": EXPECTED_WORLD_SIZE,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "argv": [
            str(EXPECTED_PYTHON),
            "-m",
            "accelerate.commands.launch",
            "--num_processes",
            "2",
            "--mixed_precision",
            "bf16",
            "--dynamo_backend",
            "no",
            "-m",
            "jobs.train.train_sft",
            "--config",
            str(config_path),
        ],
        "resume": "forbidden_fresh_only_v1",
    }


def execute_training(repo_root: Path) -> None:
    global _EXEC_LOCK
    root = repo_root.expanduser().resolve()
    _require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1", "launch requires GPUs 0,1"
    )
    preflight_path = (root / PREFLIGHT_RECEIPT).resolve()
    runtime_receipt_path = (root / RUNTIME_LAUNCH_RECEIPT).resolve()
    _require(
        not preflight_path.exists() and not preflight_path.is_symlink(),
        "fresh-only preflight receipt already exists",
    )
    _require(
        not runtime_receipt_path.exists() and not runtime_receipt_path.is_symlink(),
        "fresh-only runtime launch receipt already exists",
    )
    preflight_receipt = preflight(root)
    command = training_command(root)
    lock_path = (root / RUN_ROOT / ".training.lock").resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    _EXEC_LOCK = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(_EXEC_LOCK.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise PaperChk2LaunchError("paper chk-2 training lock is already held") from exc
    os.set_inheritable(_EXEC_LOCK.fileno(), True)
    _require(
        os.get_inheritable(_EXEC_LOCK.fileno()), "training lock is not exec-inheritable"
    )
    # Close the small race between the first resource check and lock ownership.
    _require(
        not (root / OUTPUT_DIR).exists(),
        "fresh-only adapter output appeared during launch",
    )
    runtime_gpu_inventory = _verify_gpus()
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{root / 'src'}:{root}",
            "FOMC_PAPER_CHK2_BRANCH_ID": BRANCH_ID,
            "FOMC_PAPER_CHK2_CONFIG_PATH": command["config"]["path"],
            "FOMC_PAPER_CHK2_CONFIG_SHA256": command["config"]["sha256"],
        }
    )
    _write_json_create_only(preflight_path, preflight_receipt)
    runtime_receipt = {
        "schema_version": "paper-chk2-minutes-sft-runtime-launch-v1",
        "status": "ready_for_exec",
        "generated_at": _utc_now(),
        "branch_id": BRANCH_ID,
        "preflight_receipt": {
            "path": str(preflight_path),
            "sha256": sha256_file(preflight_path),
        },
        "gpu_allocation_policy": _gpu_allocation_policy(),
        "gpu_snapshot_immediately_before_exec": {
            "captured_at": _utc_now(),
            "gpus": runtime_gpu_inventory,
            "co_tenancy_observed": any(
                bool(row["co_tenancy_observed"]) for row in runtime_gpu_inventory
            ),
        },
        "external_process_actions": {
            "signals_sent": [],
            "processes_stopped": [],
            "policy": "observe-only; never signal or stop co-tenant processes",
        },
        "distributed_training": {
            "cuda_visible_devices": "0,1",
            "world_size": EXPECTED_WORLD_SIZE,
            "num_processes": command["num_processes"],
        },
        "environment_overrides": {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "command": command,
    }
    _write_json_create_only(runtime_receipt_path, runtime_receipt)
    os.chdir(root)
    os.execvpe(command["argv"][0], command["argv"], environment)


def status(repo_root: Path) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    output = (root / OUTPUT_DIR).resolve()
    checkpoints = (
        sorted(
            int(path.name.split("-", 1)[1])
            for path in output.glob("checkpoint-[0-9]*")
            if path.is_dir() and path.name.split("-", 1)[1].isdigit()
        )
        if output.is_dir()
        else []
    )
    pid_path = (root / RUN_ROOT / "training.pid").resolve()
    pid: int | None = None
    running = False
    if pid_path.is_file() and not pid_path.is_symlink():
        raw = pid_path.read_text(encoding="utf-8").strip()
        if raw.isdigit():
            pid = int(raw)
            try:
                os.kill(pid, 0)
                running = True
            except ProcessLookupError:
                pass
            except PermissionError:
                running = True
    return {
        "branch_id": BRANCH_ID,
        "status": "running" if running else "not_running",
        "pid": pid,
        "training_output": "present" if output.exists() else "absent",
        "checkpoints": checkpoints,
        "expected_final_step": EXPECTED_OPTIMIZER_STEPS,
        "log": str((root / RUN_ROOT / "logs/train.log").resolve()),
        "preflight_receipt": str((root / PREFLIGHT_RECEIPT).resolve()),
        "runtime_launch_receipt": str((root / RUNTIME_LAUNCH_RECEIPT).resolve()),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("status")
    launch = subparsers.add_parser("launch")
    launch.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = args.repo_root.expanduser().resolve()
    try:
        if args.command == "preflight":
            result = preflight(root)
        elif args.command == "status":
            result = status(root)
        elif args.command == "launch":
            if args.execute:
                execute_training(root)
                raise AssertionError("exec unexpectedly returned")
            preflight(root)
            result = training_command(root)
        else:  # pragma: no cover
            raise PaperChk2LaunchError(f"unsupported command: {args.command}")
    except (PaperChk2LaunchError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
