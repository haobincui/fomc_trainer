"""Authorize and launch the one-shot chk4 pre-2009 correction SFT.

The branch is independent and create-only.  Its parent is the exact selected
checkpoint-38 SFT merge; it trains one fresh LoRA for exactly six optimizer
steps on the immutable 48-row correction release.  Checkpoints 2/4/6 are
pre-registered for later static screening.  The root adapter is never merged.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, IO

import yaml

from jobs.retrain_v2 import chk4_pre2009_cp38_grpo_continuation as source_continuation
from jobs.retrain_v2.materialize_chk4_pre2009_correction_release import (
    DATASET_ROLE,
    RELEASE_ID,
    verify_runtime_release,
)
from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.trainer.dataset_release import CHK4_STUDENT_SYSTEM_PROMPT
from open_r1.trainer.fixed_schedule_sampler_v3 import (
    SAMPLER_TYPE_V3,
    validate_runtime_contract,
)
from open_r1.validator.loo_generation_spec import seal_manifest, validate_manifest_integrity


BRANCH_ID = (
    "chk4_from_pre2009_cp38_selected_correction_sft_lr2e6_steps6_v1_20260811"
)
PROFILE = "pre2009_cp38_correction_sft_lr2e6_steps6_v1"
AUTHORIZATION_SCHEMA = "chk4-pre2009-cp38-correction-sft-authorization-v1"
RUN_ROOT = Path(f"output/training/retrain_v2/{BRANCH_ID}")
CONFIG = Path(
    "configs/retrain_v2/"
    "chk4_decision_sft_from_pre2009_cp38_correction_lr2e6_steps6_v1_20260811.yaml"
)
CONFIG_SHA256 = "34613046b8030136a56e9afda17cddfbfa216834b143352a1b8b3fff310a23e9"
RELEASE_MANIFEST = Path(
    "dataset/processed/retrain_v2/"
    "chk4_decision_pre2009_correction_sft_v1_20260811/release_manifest.json"
)
RELEASE_MANIFEST_SHA256 = (
    "1930991a24ce4cd615b0e96ef233991e2885b3ee9a45423686e1005181be0713"
)
TRAIN_SHA256 = "b8285febc620637eabef29876747b92bbea3d7fca933f0752781b2686eabc06f"
SCHEDULE_SHA256 = "84649a794dd556567f8fba547cdede33c4e428ee106e970938321a213ef524aa"
PARENT_MODEL_SHA256 = (
    "715bdb763043e7fda49e4a189b601d5e070fefbcecd14388bde41dbf7c165dd0"
)
CANDIDATE_STEPS = (2, 4, 6)
_EXEC_LOCK: IO[str] | None = None

FAILED_SMOKE_ROOT = Path(
    "output/training/retrain_v2/"
    "chk4_from_pre2009_cp38_selected_grpo_smoke_cap1536_v2_20260811"
)
FAILED_SMOKE_ARTIFACTS = {
    "authorization": (
        FAILED_SMOKE_ROOT / "receipts/authorization.json",
        "0fbb9e4cc7a8704a7b5a8536e571dfd585b3c95e301474c4e8c5e7c871332886",
    ),
    "resolved_runtime": (
        FAILED_SMOKE_ROOT / "adapters/chk4_grpo/resolved_runtime_config.json",
        "6d072860fa2d6941d57d5a2915daba117154ba8ae96d8f812fe74105486c2bd5",
    ),
    "reward": (
        FAILED_SMOKE_ROOT / "adapters/chk4_grpo/reward.jsonl",
        "5b80741e2f9d222ea803bf683882869f85ecf6a4a3e0524bf41d82a106f1ce3f",
    ),
    "runtime_safety": (
        FAILED_SMOKE_ROOT / "adapters/chk4_grpo/runtime_safety.jsonl",
        "e7ee7d025c4b7fb8ea1496377e3a1da53d22f9dca1c518c31ea7ce737ee232c3",
    ),
    "step_gate": (
        FAILED_SMOKE_ROOT / "adapters/chk4_grpo/chk4_smoke_step_gate.jsonl",
        "b2b421b416788968a76c8e92025b0f9e83b74e51e1ca2fc6cbd2de3d53216160",
    ),
    "log": (
        FAILED_SMOKE_ROOT / "logs/grpo.log",
        "d083978f59c8c8d2c26fb27b3b6ee7f4091bca04f2ef6e204c19c1525271a870",
    ),
}


class CorrectionSFTError(RuntimeError):
    """The one-shot correction SFT is not safely launchable."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorrectionSFTError(message)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorrectionSFTError(f"invalid {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        _require(bool(raw), f"{label}:{line_number}: blank row")
        value = json.loads(raw)
        _require(isinstance(value, dict), f"{label}:{line_number}: not an object")
        rows.append(value)
    return rows


def _read_yaml(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing config: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "correction config must be a mapping")
    return value


def _resolve(repo_root: Path, value: object, *, label: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{label} path missing")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _implementation(repo_root: Path) -> dict[str, Any]:
    relatives = (
        "jobs/retrain_v2/chk4_pre2009_correction_sft.py",
        "jobs/train/train_sft.py",
        "run/retrain_v2/chk4_pre2009_correction_sft.sh",
        "src/open_r1/configs.py",
        "src/open_r1/trainer/dataset_release.py",
        "src/open_r1/trainer/fixed_schedule_sampler_v3.py",
        "src/open_r1/trainer/sft_trainer.py",
        "src/open_r1/trainer/trainer.py",
        "jobs/retrain_v2/materialize_chk4_pre2009_correction_release.py",
    )
    return {
        relative: {
            "path": str((repo_root / relative).resolve()),
            "sha256": sha256_file(repo_root / relative),
        }
        for relative in relatives
    }


def _failed_smoke(repo_root: Path) -> dict[str, Any]:
    descriptors: dict[str, Any] = {}
    for name, (relative, expected_sha) in FAILED_SMOKE_ARTIFACTS.items():
        path = (repo_root / relative).resolve()
        _require(path.is_file() and not path.is_symlink(), f"failed smoke {name} missing")
        observed = sha256_file(path)
        _require(observed == expected_sha, f"failed smoke {name} hash drift")
        descriptor: dict[str, Any] = {"path": str(path), "sha256": observed}
        if path.suffix == ".jsonl":
            descriptor["rows"] = len(_read_jsonl(path, label=f"failed smoke {name}"))
        descriptors[name] = descriptor
    gate = _read_jsonl(
        Path(descriptors["step_gate"]["path"]), label="failed smoke step gate"
    )
    _require(len(gate) == 1, "failed smoke must have exactly one gate row")
    row = gate[0]
    expected_checks = {
        "each_target_direction_correct_nonzero": False,
        "each_target_reward_std_gt_zero": False,
        "loss_finite": True,
        "grad_norm_gt_1e_12": True,
        "clipped_ratio_le_0_25": True,
    }
    _require(
        row.get("status") == "failed"
        and row.get("step") == 1
        and row.get("checks") == expected_checks
        and row.get("completion_clipped_ratio") == 0.125
        and row.get("target_groups", {}).get("hold", {}).get(
            "direction_correct_nonzero_count"
        )
        == 2
        and row.get("target_groups", {}).get("hike", {}).get(
            "direction_correct_nonzero_count"
        )
        == 0,
        "failed cap-1536 smoke diagnosis drift",
    )
    rewards = _read_jsonl(
        Path(descriptors["reward"]["path"]), label="failed smoke rewards"
    )
    _require(
        len(rewards) == 8
        and [item.get("target", {}).get("direction") for item in rewards]
        == ["hold"] * 4 + ["hike"] * 4
        and [item.get("reward") for item in rewards[4:]] == [0, 0, 0, 0],
        "failed cap-1536 reward evidence drift",
    )
    return {
        "schema_version": "chk4-pre2009-cap1536-failure-evidence-v1",
        "outcome": "failed_closed_at_step_1_hike_zero_signal",
        "artifacts": descriptors,
        "observed": {
            "completion_clipped_ratio": 0.125,
            "hold_correct_nonzero": 2,
            "hike_correct_nonzero": 0,
        },
    }


def _config_contract(repo_root: Path) -> tuple[Path, dict[str, Any]]:
    path = (repo_root / CONFIG).resolve()
    _require(sha256_file(path) == CONFIG_SHA256, "correction config hash drift")
    config = _read_yaml(path)
    run_root = (repo_root / RUN_ROOT).resolve()
    parent = _resolve(repo_root, config.get("model_name_or_path"), label="parent")
    output = _resolve(repo_root, config.get("output_dir"), label="output")
    prohibited_merge = _resolve(
        repo_root, config.get("peft_merged_model_path"), label="root merge"
    )
    exact = {
        "dataset_chk4_role": DATASET_ROLE,
        "dataset_chk4_release_manifest_sha256": RELEASE_MANIFEST_SHA256,
        "learning_rate": 2.0e-6,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.0,
        "warmup_steps": 1,
        "gradient_accumulation_steps": 8,
        "max_steps": 6,
        "num_train_epochs": 1,
        "eval_steps": 1,
        "save_steps": 1,
        "save_total_limit": None,
        "checkpoint_keep_last": 3,
        "checkpoint_keep_every_n_steps": 0,
        "checkpoint_keep_steps": list(CANDIDATE_STEPS),
        "train_sampler": SAMPLER_TYPE_V3,
        "shuffle_dataset": False,
        "group_by_length": False,
        "dataloader_drop_last": False,
        "per_device_train_batch_size": 1,
        "seed": 42,
        "data_seed": 42,
        "overwrite_output_dir": False,
        "resume_from_checkpoint": None,
        "use_liger_kernel": False,
    }
    for key, expected in exact.items():
        _require(config.get(key) == expected, f"correction config {key} drift")
    _require(config.get("system_prompt") == CHK4_STUDENT_SYSTEM_PROMPT, "system prompt drift")
    _require(
        parent == (repo_root / source_continuation.SELECTED_MODEL).resolve(),
        "correction parent is not selected cp38 exact merge",
    )
    _require(output == run_root / "adapters/chk4_correction_sft", "output path drift")
    _require(
        prohibited_merge == run_root / "merged/unselected_root_adapter_prohibited",
        "prohibited root merge path drift",
    )
    release_path = _resolve(
        repo_root, config.get("dataset_chk4_release_manifest"), label="release"
    )
    dataset = _resolve(repo_root, config.get("dataset_name"), label="dataset")
    _require(release_path == (repo_root / RELEASE_MANIFEST).resolve(), "release path drift")
    _require(dataset == release_path.parent / "decision_sft", "dataset path drift")
    return path, config


def preflight(repo_root: Path) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    config_path, config = _config_contract(root)
    plan, selection = source_continuation._published_selection(root)
    parent_fingerprint = fingerprint_artifact_path(plan.destination)
    _require(parent_fingerprint.get("sha256") == PARENT_MODEL_SHA256, "cp38 parent drift")
    _require(selection.get("merged_artifact") == parent_fingerprint, "cp38 receipt drift")
    release_path = (root / RELEASE_MANIFEST).resolve()
    _require(sha256_file(release_path) == RELEASE_MANIFEST_SHA256, "release hash drift")
    runtime_release = verify_runtime_release(
        dataset_dir=release_path.parent / "decision_sft",
        manifest_path=release_path,
        expected_manifest_sha256=RELEASE_MANIFEST_SHA256,
        dataset_role=DATASET_ROLE,
        system_prompt=CHK4_STUDENT_SYSTEM_PROMPT,
        model_path=plan.destination,
    )
    sampler = validate_runtime_contract(
        training_args=SimpleNamespace(
            train_sampler=SAMPLER_TYPE_V3,
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
    _require(sampler["schedule"]["sha256"] == SCHEDULE_SHA256, "schedule hash drift")
    _require(sampler["train_file"]["sha256"] == TRAIN_SHA256, "train hash drift")
    output = _resolve(root, config["output_dir"], label="output")
    prohibited_merge = _resolve(
        root, config["peft_merged_model_path"], label="prohibited root merge"
    )
    _require(not output.exists() and not output.is_symlink(), "fresh-only output exists")
    _require(
        not prohibited_merge.exists() and not prohibited_merge.is_symlink(),
        "prohibited root merge exists",
    )
    return {
        "schema_version": "chk4-pre2009-correction-sft-preflight-v1",
        "status": "ready",
        "branch_id": BRANCH_ID,
        "profile": PROFILE,
        "config": {"path": str(config_path), "sha256": CONFIG_SHA256},
        "run_root": str((root / RUN_ROOT).resolve()),
        "parent": {
            "merged_artifact": parent_fingerprint,
            "selection_receipt": {
                "path": str(plan.receipt_path),
                "sha256": sha256_file(plan.receipt_path),
                "payload_sha256": selection["integrity"]["payload_sha256"],
            },
            "exact_merge": {
                "path": str(plan.exact_evidence_path),
                "sha256": sha256_file(plan.exact_evidence_path),
            },
        },
        "release": {
            "path": str(release_path),
            "sha256": RELEASE_MANIFEST_SHA256,
            "release_id": RELEASE_ID,
            "train_sha256": TRAIN_SHA256,
            "schedule_sha256": SCHEDULE_SHA256,
            "runtime_binding_schema": runtime_release["schema_version"],
            "sampler": sampler,
        },
        "motivation": _failed_smoke(root),
        "training": {
            "fresh_lora": True,
            "max_steps": 6,
            "candidate_checkpoints": list(CANDIDATE_STEPS),
            "learning_rate": 2.0e-6,
            "seed": 42,
            "data_seed": 42,
            "gpu": 1,
            "root_adapter_merge_authorized": False,
        },
        "implementation": _implementation(root),
    }


def _authorization_path(repo_root: Path) -> Path:
    return (repo_root / RUN_ROOT / "receipts/authorization.json").resolve()


def _authorization_payload(repo_root: Path) -> dict[str, Any]:
    return {
        "schema_version": AUTHORIZATION_SCHEMA,
        "status": "authorized",
        "authorization_basis": "explicit_user_instruction_one_correction_sft",
        "scope": {
            "operation": "fresh_six_step_correction_sft_on_gpu1",
            "not_authorized": [
                "gpu0",
                "different_seed",
                "different_data",
                "different_parent",
                "resume_or_second_run",
                "root_adapter_merge",
                "checkpoint39",
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
    if path.exists() or path.is_symlink():
        raise CorrectionSFTError(f"authorization is create-only: {path}")
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
    authorization = _read_json(path, label="correction authorization")
    validate_manifest_integrity(authorization)
    _require(
        authorization == seal_manifest(_authorization_payload(repo_root)),
        "correction authorization no longer matches preflight",
    )
    return authorization


def training_command(repo_root: Path) -> dict[str, Any]:
    _verify_authorization(repo_root)
    config_path, _ = _config_contract(repo_root)
    return {
        "status": "ready",
        "branch_id": BRANCH_ID,
        "gpu": 1,
        "config": {"path": str(config_path), "sha256": CONFIG_SHA256},
        "argv": [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--num_processes",
            "1",
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
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "launch requires GPU1")
    command = training_command(repo_root)
    lock_path = (repo_root / RUN_ROOT / ".training.lock").resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    _EXEC_LOCK = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(_EXEC_LOCK.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise CorrectionSFTError("correction training lock is already held") from exc
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "1",
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "FOMC_CHK4_BRANCH_ID": BRANCH_ID,
            "FOMC_CHK4_BRANCH_PROFILE": PROFILE,
            "FOMC_CHK4_BRANCH_STAGE": "decision_sft",
            "FOMC_CHK4_BRANCH_CONFIG_PATH": command["config"]["path"],
            "FOMC_CHK4_BRANCH_CONFIG_SHA256": CONFIG_SHA256,
        }
    )
    os.chdir(repo_root)
    os.execvpe(command["argv"][0], command["argv"], environment)


def status(repo_root: Path) -> dict[str, Any]:
    output = (repo_root / RUN_ROOT / "adapters/chk4_correction_sft").resolve()
    checkpoints = sorted(
        int(path.name.split("-", 1)[1])
        for path in output.glob("checkpoint-[0-9]*")
        if path.is_dir() and path.name.split("-", 1)[1].isdigit()
    ) if output.is_dir() else []
    return {
        "branch_id": BRANCH_ID,
        "authorization": "present" if _authorization_path(repo_root).is_file() else "absent",
        "training_output": "absent" if not output.exists() else "present",
        "checkpoints": checkpoints,
        "candidate_checkpoints_present": [step for step in CANDIDATE_STEPS if step in checkpoints],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("status")
    authorize_parser = sub.add_parser("authorize")
    authorize_parser.add_argument("--execute", action="store_true")
    launch = sub.add_parser("launch")
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
        elif args.command == "authorize":
            result = authorize(root, execute=args.execute)
        elif args.command == "launch":
            if args.execute:
                execute_training(root)
                raise AssertionError("exec unexpectedly returned")
            result = training_command(root)
        else:  # pragma: no cover
            raise CorrectionSFTError(f"unsupported command: {args.command}")
    except (CorrectionSFTError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
