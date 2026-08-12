"""GPU1-only full GRPO from exact cp38 with an explicit no-smoke override.

This workflow is intentionally separate from the gated smoke/full continuation.
It preserves all prior smoke artifacts, binds the same full-run hyperparameters,
and records that the smoke prerequisite was removed by explicit user direction.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from jobs.retrain_v2 import chk4_pre2009_cp38_grpo_continuation as cp38
from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.validator.loo_generation_spec import seal_manifest, validate_manifest_integrity


BRANCH_ID = "chk4_from_pre2009_cp38_direct_grpo_full_no_smoke_v1_20260812"
RUN_ROOT = Path("output/training/retrain_v2") / BRANCH_ID
CONFIG = Path(
    "configs/retrain_v2/"
    "chk4_decision_grpo_from_pre2009_cp38_direct_full_no_smoke_v1_20260812.yaml"
)
CONFIG_SHA256 = "222c70d76823681b7ff07971704c38fd153abcac4620a4e0fab4fce00d77f06f"
REFERENCE_CONFIG = Path(
    "configs/retrain_v2/"
    "chk4_decision_grpo_from_pre2009_cp38_selected_full_cap1536_v2_20260811.yaml"
)
REFERENCE_CONFIG_SHA256 = (
    "a483e09a44f4e164aca3932c25bd940038e73281c1b5171df50262a5ede7bd3a"
)
AUTH_SCHEMA = "chk4-pre2009-cp38-direct-full-no-smoke-authorization-v1"
SOURCE_MODEL_SHA256 = "715bdb763043e7fda49e4a189b601d5e070fefbcecd14388bde41dbf7c165dd0"
RELEASE_MANIFEST = Path(
    "dataset/processed/retrain_v2/"
    "chk4_decision_pre2009_train_balanced_v1_20260811/release_manifest.json"
)
RELEASE_MANIFEST_SHA256 = (
    "5141e00569b94e4af96f030f46176e9cd409199a6f26ac1c63a2ce335fe1e2d6"
)
TRAIN_FILE = RELEASE_MANIFEST.parent / "decision_grpo/train.jsonl"
TRAIN_SHA256 = "e8c9e17892fb31a02f811c6aee222b1bd23486e60164c9426228444743eab046"
EXPECTED_TRAIN_ROWS = 312
AUTHORIZATION = RUN_ROOT / "receipts/authorization.json"
OUTPUT = RUN_ROOT / "adapters/chk4_grpo"
MERGED = RUN_ROOT / "merged/chk4_grpo"
IMPLEMENTATION_FILES = (
    "jobs/retrain_v2/chk4_pre2009_cp38_grpo_direct_full.py",
    "jobs/train/train_grpo.py",
    "src/open_r1/trainer/trainer.py",
    "src/open_r1/trainer/grpo_trainer.py",
    "src/open_r1/trainer/rewards/reward_funcs/decision_reward_v3.py",
    "src/open_r1/trainer/rewards/reward_register.py",
    "src/open_r1/utils/callbacks.py",
)


class DirectFullError(RuntimeError):
    """The direct-full run does not satisfy its explicit authorization."""


_LOCK_HANDLE = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DirectFullError(message)


def _resolve(repo_root: Path, path: Path) -> Path:
    return (repo_root / path).resolve()


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"YAML is not a mapping: {path}")
    return value


def _descriptor(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"unsafe or missing file: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.rstrip("\n"))


def _runtime_implementation(repo_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for relative in IMPLEMENTATION_FILES:
        source = _resolve(repo_root, Path(relative))
        descriptor = _descriptor(source)
        if relative.startswith("src/open_r1/"):
            installed = (
                Path(sys.executable).resolve().parent.parent
                / "lib/python3.10/site-packages"
                / relative.removeprefix("src/")
            )
            installed_descriptor = _descriptor(installed)
            _require(
                installed_descriptor["sha256"] == descriptor["sha256"],
                f"installed runtime differs from source: {relative}",
            )
            descriptor["installed"] = installed_descriptor
        result[relative] = descriptor
    return result


def _config_contract(repo_root: Path) -> dict[str, Any]:
    path = _resolve(repo_root, CONFIG)
    reference_path = _resolve(repo_root, REFERENCE_CONFIG)
    _require(sha256_file(path) == CONFIG_SHA256, "direct-full config hash drift")
    _require(
        sha256_file(reference_path) == REFERENCE_CONFIG_SHA256,
        "reference full config hash drift",
    )
    config = _read_yaml(path)
    reference = _read_yaml(reference_path)
    differences = {
        key: {"from": reference.get(key), "to": config.get(key)}
        for key in sorted(set(reference) | set(config))
        if reference.get(key) != config.get(key)
    }
    _require(
        set(differences) == {"output_dir", "peft_merged_model_path"},
        f"direct-full config changed training settings: {sorted(differences)}",
    )
    _require("max_steps" not in config, "direct-full must use the three-epoch contract")
    callbacks = config.get("callbacks")
    _require(
        isinstance(callbacks, list)
        and "chk4_pre2009_cp38_smoke_gate" not in callbacks,
        "direct-full config retained a smoke callback",
    )
    _require(config.get("num_train_epochs") == 3, "epoch contract drift")
    _require(config.get("seed") == config.get("data_seed") == 42, "seed contract drift")
    _require(config.get("learning_rate") == 1e-6, "learning-rate contract drift")
    _require(config.get("max_completion_length") == 1536, "completion cap drift")
    _require(config.get("reward_funcs") == ["decision_dense_v3"], "reward contract drift")
    _require(config.get("output_dir") == str(OUTPUT), "output path drift")
    _require(config.get("peft_merged_model_path") == str(MERGED), "merge path drift")
    return {"config": config, "differences": differences}


def preflight(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    contract = _config_contract(repo_root)
    plan, selection = cp38._published_selection(repo_root)
    model = fingerprint_artifact_path(plan.destination)
    _require(model.get("sha256") == SOURCE_MODEL_SHA256, "exact cp38 model drift")
    release = _resolve(repo_root, RELEASE_MANIFEST)
    train = _resolve(repo_root, TRAIN_FILE)
    _require(sha256_file(release) == RELEASE_MANIFEST_SHA256, "GRPO release drift")
    _require(sha256_file(train) == TRAIN_SHA256, "GRPO train data drift")
    _require(_jsonl_rows(train) == EXPECTED_TRAIN_ROWS, "GRPO train row count drift")
    output = _resolve(repo_root, OUTPUT)
    merged = _resolve(repo_root, MERGED)
    _require(not output.exists() and not output.is_symlink(), "direct-full output exists")
    _require(not merged.exists() and not merged.is_symlink(), "direct-full merge output exists")
    integrity = selection.get("integrity")
    _require(isinstance(integrity, Mapping), "cp38 selection integrity missing")
    return {
        "schema_version": "chk4-pre2009-cp38-direct-full-no-smoke-preflight-v1",
        "status": "ready",
        "branch_id": BRANCH_ID,
        "source": {
            "checkpoint_step": 38,
            "merged_model": model,
            "selection_receipt": {
                **_descriptor(plan.receipt_path),
                "payload_sha256": integrity["payload_sha256"],
            },
        },
        "data": {
            "release_manifest": _descriptor(release),
            "train": {**_descriptor(train), "rows": EXPECTED_TRAIN_ROWS},
        },
        "config": {
            **_descriptor(_resolve(repo_root, CONFIG)),
            "delta_from_gated_full": contract["differences"],
        },
        "runtime_implementation": _runtime_implementation(repo_root),
        "gpu_contract": {"physical_gpu": 1, "visible_device_count": 1, "gpu0": False},
        "smoke": {
            "required": False,
            "removed_by": "explicit_user_instruction_2026-08-12",
            "prior_artifacts_preserved": True,
        },
    }


def _authorization_payload(repo_root: Path) -> dict[str, Any]:
    ready = preflight(repo_root)
    return {
        "schema_version": AUTH_SCHEMA,
        "status": "authorized",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_basis": "explicit_user_instruction_remove_smoke_and_start_full_grpo",
        "branch_id": BRANCH_ID,
        "scope": {
            "allowed": ["cp38_full_grpo_on_gpu1"],
            "not_authorized": [
                "gpu0",
                "overwrite_prior_smoke_artifacts",
                "resume_failed_smoke_adapter",
                "different_parent_checkpoint",
                "canonical_dag_claim",
            ],
        },
        "preflight": ready,
        "output": {
            "run_root": str(_resolve(repo_root, RUN_ROOT)),
            "adapter": str(_resolve(repo_root, OUTPUT)),
            "merged": str(_resolve(repo_root, MERGED)),
        },
    }


def authorize(repo_root: Path, *, execute: bool) -> dict[str, Any]:
    payload = seal_manifest(_authorization_payload(repo_root))
    path = _resolve(repo_root, AUTHORIZATION)
    if not execute:
        return {**payload, "authorization_path": str(path), "create_only": True}
    _require(not path.exists() and not path.is_symlink(), "authorization already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "status": "authorized",
        "path": str(path),
        "sha256": sha256_file(path),
        "payload_sha256": payload["integrity"]["payload_sha256"],
    }


def _verify_authorization(repo_root: Path) -> dict[str, Any]:
    path = _resolve(repo_root, AUTHORIZATION)
    _require(path.stat().st_mode & 0o777 == 0o400, "authorization mode drift")
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest_integrity(value)
    _require(
        value.get("schema_version") == AUTH_SCHEMA
        and value.get("status") == "authorized"
        and value.get("branch_id") == BRANCH_ID
        and value.get("authorization_basis")
        == "explicit_user_instruction_remove_smoke_and_start_full_grpo",
        "authorization scope drift",
    )
    recorded = value.get("preflight")
    current = preflight(repo_root)
    _require(recorded == current, "authorization binding drift")
    return value


def launch(repo_root: Path, *, execute: bool) -> dict[str, Any]:
    authorization = _verify_authorization(repo_root)
    command = [
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
        "jobs.train.train_grpo",
        "--config",
        str(_resolve(repo_root, CONFIG)),
    ]
    result = {
        "schema_version": "chk4-pre2009-cp38-direct-full-no-smoke-launch-v1",
        "status": "ready",
        "branch_id": BRANCH_ID,
        "gpu": 1,
        "authorization": {
            **_descriptor(_resolve(repo_root, AUTHORIZATION)),
            "payload_sha256": authorization["integrity"]["payload_sha256"],
        },
        "argv": command,
    }
    if not execute:
        return result
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "launch requires GPU1")
    global _LOCK_HANDLE
    lock = _resolve(repo_root, RUN_ROOT / ".training.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    _LOCK_HANDLE = lock.open("a+", encoding="utf-8")
    try:
        fcntl.flock(_LOCK_HANDLE.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise DirectFullError("another direct-full process owns the training lock") from exc
    os.set_inheritable(_LOCK_HANDLE.fileno(), True)
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
            "FOMC_CHK4_BRANCH_PROFILE": "pre2009_cp38_direct_full_no_smoke_v1",
            "FOMC_CHK4_BRANCH_STAGE": "decision_grpo",
            "FOMC_CHK4_BRANCH_CONFIG_PATH": str(_resolve(repo_root, CONFIG)),
            "FOMC_CHK4_BRANCH_CONFIG_SHA256": CONFIG_SHA256,
        }
    )
    os.chdir(repo_root)
    os.execvpe(command[0], command, environment)
    raise AssertionError("exec unexpectedly returned")


def status(repo_root: Path) -> dict[str, Any]:
    root = _resolve(repo_root, RUN_ROOT)
    adapter = _resolve(repo_root, OUTPUT)
    checkpoints = []
    if adapter.is_dir():
        checkpoints = sorted(
            int(path.name.removeprefix("checkpoint-"))
            for path in adapter.glob("checkpoint-*")
            if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
        )
    return {
        "schema_version": "chk4-pre2009-cp38-direct-full-no-smoke-status-v1",
        "branch_id": BRANCH_ID,
        "authorization": "present" if _resolve(repo_root, AUTHORIZATION).is_file() else "missing",
        "output": "present" if adapter.exists() else "missing",
        "checkpoints": checkpoints,
        "trainer_state": "present" if (adapter / "trainer_state.json").is_file() else "missing",
        "run_root": str(root),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "status", "authorize", "launch"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    try:
        if args.command == "preflight":
            _require(not args.execute, "preflight does not accept --execute")
            result = preflight(repo_root)
        elif args.command == "status":
            _require(not args.execute, "status does not accept --execute")
            result = status(repo_root)
        elif args.command == "authorize":
            result = authorize(repo_root, execute=args.execute)
        else:
            result = launch(repo_root, execute=args.execute)
    except (DirectFullError, FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
