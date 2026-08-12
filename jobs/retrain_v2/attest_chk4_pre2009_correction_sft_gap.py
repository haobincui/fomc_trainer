"""Seal the correction-SFT post-run provenance gap without rewriting history.

The original authorization omitted several runtime source files.  This receipt
is deliberately labelled supplemental and post-run: it records the observable
source hashes/timestamps and completed artifacts, but it does not pretend those
files were covered by the pre-run authorization.  Downstream selection, merge,
and GRPO authorization must bind both receipts and retain this limitation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


RUN_ROOT = Path(
    "output/training/retrain_v2/"
    "chk4_from_pre2009_cp38_selected_correction_sft_lr2e6_steps6_v1_20260811"
)
TRAINING_OUTPUT = RUN_ROOT / "adapters/chk4_correction_sft"
AUTHORIZATION = RUN_ROOT / "receipts/authorization.json"
ATTESTATION = RUN_ROOT / (
    "receipts/supplemental_post_run_provenance_gap_attestation_v1.json"
)
SFT_LOG = RUN_ROOT / "logs/sft.log"
ABORTED_PROBE = RUN_ROOT / "static_screening/checkpoint-2-selection-cap1536-v1"

SCHEMA = "chk4-pre2009-correction-sft-post-run-provenance-gap-v1"
STATUS = "supplemental_not_pre_authorization"
TRAINING_START_UTC = "2026-08-11T20:41:10Z"
AUTHORIZATION_SHA256 = (
    "428a5972f862ab18db650fd0f8e2a17f367b2787e791e46500dd8d7316c05154"
)
AUTHORIZATION_PAYLOAD_SHA256 = (
    "b445e2bd19c6059101f7ac8bcdb684d12c086d03850c7185a9d40da34b961d6e"
)

ARTIFACT_SHA256 = {
    "resolved_runtime_config.json": (
        "22173e6f36efc48c3b4c528879f3368f200abb06496530950c66509fec28f7f3"
    ),
    "trainer_state.json": (
        "6bb9b0859bfc97e4ee9d6be204cefd7ad6164e37e447192622816468e8a91b90"
    ),
    "train_results.json": (
        "b13a52dc001e04f061be7141511bb72d9c8b975a3cf18e21ee06e043ca85938f"
    ),
    "eval_results.json": (
        "e4e9a04c646fc51123db091642eaaa1a54f4603ec23ce44829adb40b6be8ac62"
    ),
    "loss_history.jsonl": (
        "6379dfafb4a0b0790405a0c5e368ce57d74d951c219d4303590826ad1353f144"
    ),
    "all_results.json": (
        "2c865f72e01f132726319663aead8ee59e41bcd146a3e1b6f9011108a6a9a5b6"
    ),
}
SFT_LOG_SHA256 = "1a408ed1997b196f670af2dcedebdd3371b35b2c21dbe8c123035acb7bb0d96e"
ABORTED_PROBE_LAUNCH_SHA256 = (
    "09e00acf497dfe6cac3253e4c730003a39d6c823e62e4cbf22f8c71fb72f6c0d"
)

GAP_RUNTIME_SOURCES = (
    "src/open_r1/utils/callbacks.py",
    "src/open_r1/utils/evaluation.py",
    "src/open_r1/utils/hub.py",
    "src/open_r1/data_loader.py",
    "src/open_r1/trainer/sft_prompt_renderer.py",
    "src/open_r1/trainer/prompt_contract.py",
    "src/open_r1/utils/model_utils.py",
    "src/open_r1/utils/__init__.py",
    "src/open_r1/utils/import_utils.py",
    "run/retrain_v2/gpu_gate.sh",
    "src/open_r1/trainer/fixed_schedule_sampler.py",
    "src/open_r1/provenance.py",
)


class GapAttestationError(RuntimeError):
    """The supplemental receipt cannot be established safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GapAttestationError(message)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GapAttestationError(f"invalid {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        _require(bool(raw), f"{label}:{line_number}: blank row")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GapAttestationError(
                f"{label}:{line_number}: invalid JSON"
            ) from exc
        _require(isinstance(value, dict), f"{label}:{line_number}: not an object")
        rows.append(value)
    return rows


def _descriptor(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing artifact: {path}")
    observed = sha256_file(path)
    if expected_sha256 is not None:
        _require(observed == expected_sha256, f"artifact hash drift: {path}")
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": observed,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def _finite(value: object, *, label: str) -> float:
    _require(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value)),
        f"{label} must be finite",
    )
    return float(value)


def _training_start_ns() -> int:
    parsed = datetime.strptime(TRAINING_START_UTC, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    return int(parsed.timestamp() * 1_000_000_000)


def _source_gap(repo_root: Path, *, authorization_mtime_ns: int) -> dict[str, Any]:
    training_start_ns = _training_start_ns()
    sources: dict[str, Any] = {}
    for relative in GAP_RUNTIME_SOURCES:
        descriptor = _descriptor(repo_root / relative)
        descriptor["mtime_before_authorization"] = (
            descriptor["mtime_ns"] < authorization_mtime_ns
        )
        descriptor["mtime_before_training_start"] = (
            descriptor["mtime_ns"] < training_start_ns
        )
        descriptor["ctime_before_authorization"] = (
            descriptor["ctime_ns"] < authorization_mtime_ns
        )
        descriptor["ctime_before_training_start"] = (
            descriptor["ctime_ns"] < training_start_ns
        )
        _require(
            all(
                descriptor[key]
                for key in (
                    "mtime_before_authorization",
                    "mtime_before_training_start",
                    "ctime_before_authorization",
                    "ctime_before_training_start",
                )
            ),
            f"gap source timestamp is not pre-run: {relative}",
        )
        sources[relative] = descriptor
    return {
        "observed_post_run": True,
        "source_count": len(sources),
        "authorization_mtime_ns": authorization_mtime_ns,
        "training_start_utc": TRAINING_START_UTC,
        "training_start_ns": training_start_ns,
        "all_mtime_and_ctime_pre_authorization_and_training": True,
        "sources": sources,
    }


def _runtime_evidence(repo_root: Path) -> dict[str, Any]:
    output = repo_root / TRAINING_OUTPUT
    artifacts = {
        name: _descriptor(output / name, expected_sha256=expected)
        for name, expected in ARTIFACT_SHA256.items()
    }
    runtime = _read_json(output / "resolved_runtime_config.json", label="runtime")
    sampler = runtime.get("training_sampler")
    sampler_runtime = sampler.get("runtime") if isinstance(sampler, Mapping) else None
    expected_sampler_runtime = {
        "world_size": 1,
        "visible_gpus": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "effective_batch_size": 8,
        "max_steps": 6,
        "shuffle_dataset": False,
        "trl_shuffle_dataset": False,
        "secondary_shuffle": False,
        "use_liger_kernel": False,
    }
    _require(sampler_runtime == expected_sampler_runtime, "runtime sampler drift")
    _require(
        isinstance(sampler, Mapping)
        and sampler.get("type") == "manifest_fixed_schedule_v3"
        and sampler.get("schedule", {}).get("sha256")
        == "84649a794dd556567f8fba547cdede33c4e428ee106e970938321a213ef524aa"
        and sampler.get("train_file", {}).get("sha256")
        == "b8285febc620637eabef29876747b92bbea3d7fca933f0752781b2686eabc06f",
        "runtime v3 release binding drift",
    )
    state = _read_json(output / "trainer_state.json", label="trainer state")
    _require(
        state.get("global_step") == 6
        and state.get("max_steps") == 6
        and _finite(state.get("epoch"), label="trainer epoch") == 1.0,
        "completed trainer state drift",
    )
    train = _read_json(output / "train_results.json", label="train results")
    evaluation = _read_json(output / "eval_results.json", label="eval results")
    _finite(train.get("train_loss"), label="train loss")
    _finite(evaluation.get("eval_loss"), label="eval loss")
    loss_rows = _read_jsonl(output / "loss_history.jsonl", label="loss history")
    _require(len(loss_rows) == 14, "loss history row count drift")
    _require(
        {int(row.get("step", -1)) for row in loss_rows} == set(range(1, 7)),
        "loss history optimizer steps drift",
    )
    candidates = {}
    for step in (2, 4, 6):
        checkpoint = output / f"checkpoint-{step}"
        fingerprint = fingerprint_artifact_path(checkpoint)
        checkpoint_state = _read_json(
            checkpoint / "trainer_state.json", label=f"checkpoint-{step} state"
        )
        _require(
            checkpoint_state.get("global_step") == step,
            f"checkpoint-{step} global step drift",
        )
        candidates[str(step)] = fingerprint
    return {
        "artifacts": artifacts,
        "training_completed": {
            "global_step": 6,
            "max_steps": 6,
            "epoch": 1.0,
            "train_loss": train["train_loss"],
            "eval_loss": evaluation["eval_loss"],
            "loss_history_rows": len(loss_rows),
        },
        "candidate_checkpoint_fingerprints": candidates,
        "sampler_runtime": expected_sampler_runtime,
    }


def _gpu1_evidence(repo_root: Path) -> dict[str, Any]:
    log_path = repo_root / SFT_LOG
    log_descriptor = _descriptor(log_path, expected_sha256=SFT_LOG_SHA256)
    log = log_path.read_text(encoding="utf-8", errors="strict")
    launch_line = f"[{TRAINING_START_UTC}] launching one-shot chk4 correction SFT on GPU1"
    device_pattern = re.compile(
        r"Process rank: 0, device: cuda:0, n_gpu: 1 distributed training: True"
    )
    _require(log.splitlines()[0] == launch_line, "GPU1 launch line drift")
    matches = device_pattern.findall(log)
    _require(len(matches) == 1, "trainer device evidence drift")
    launcher_path = repo_root / "run/retrain_v2/chk4_pre2009_correction_sft.sh"
    launcher = launcher_path.read_text(encoding="utf-8")
    _require(
        '"${ROOT_DIR}/run/retrain_v2/gpu_gate.sh" 1' in launcher
        and "export CUDA_VISIBLE_DEVICES=1" in launcher,
        "physical GPU1 launcher binding drift",
    )
    return {
        "physical_device": 1,
        "visible_cuda_device_inside_process": 0,
        "world_size": 1,
        "evidence_chain": [
            "hash-bound launcher calls gpu_gate.sh for physical GPU 1",
            "hash-bound launcher exports CUDA_VISIBLE_DEVICES=1",
            "hash-bound log records GPU1 launch",
            "hash-bound trainer runtime records cuda:0 with n_gpu=1",
            "hash-bound sampler runtime records visible_gpus=1 and world_size=1",
        ],
        "launch_line": launch_line,
        "trainer_device_line": matches[0],
        "log": log_descriptor,
        "launcher": _descriptor(launcher_path),
        "gpu_gate": _descriptor(repo_root / "run/retrain_v2/gpu_gate.sh"),
    }


def _aborted_probe_evidence(repo_root: Path) -> dict[str, Any]:
    root = repo_root / ABORTED_PROBE
    _require(root.is_dir() and not root.is_symlink(), "aborted probe directory missing")
    inventory = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    )
    _require(inventory == ["launch.json"], "aborted probe inventory drift")
    launch_path = root / "launch.json"
    launch = _read_json(launch_path, label="aborted probe launch")
    _require(
        launch.get("status") == "initializing"
        and launch.get("model_label") == "correction-sft-checkpoint-2"
        and launch.get("purpose") == "checkpoint_selection",
        "aborted probe launch payload drift",
    )
    return {
        "status": "aborted_before_generation",
        "version": "checkpoint-2-selection-cap1536-v1",
        "launch": _descriptor(
            launch_path, expected_sha256=ABORTED_PROBE_LAUNCH_SHA256
        ),
        "file_inventory": inventory,
        "results_file_present": False,
        "summary_file_present": False,
        "completion_rows_landed": 0,
        "eligible_as_selection_evidence": False,
        "retry_requirement": "fresh_create_only_checkpoint-2-selection-cap1536-v2",
    }


def build_attestation(repo_root: Path) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    authorization_path = root / AUTHORIZATION
    authorization = _read_json(authorization_path, label="original authorization")
    validate_manifest_integrity(authorization)
    _require(
        sha256_file(authorization_path) == AUTHORIZATION_SHA256,
        "original authorization file hash drift",
    )
    _require(
        authorization.get("integrity", {}).get("payload_sha256")
        == AUTHORIZATION_PAYLOAD_SHA256,
        "original authorization payload hash drift",
    )
    authorization_descriptor = _descriptor(
        authorization_path, expected_sha256=AUTHORIZATION_SHA256
    )
    return seal_manifest(
        {
            "schema_version": SCHEMA,
            "status": STATUS,
            "scope": {
                "run": str((root / RUN_ROOT).resolve()),
                "purpose": "record_post_run_runtime_source_inventory_gap",
                "does_not_rewrite_original_authorization": True,
                "is_not_pre_authorization": True,
                "does_not_authorize_rerun": True,
                "does_not_authorize_merge_or_grpo": True,
                "downstream_must_bind_original_and_supplemental_receipts": True,
            },
            "original_authorization": {
                **authorization_descriptor,
                "payload_sha256": AUTHORIZATION_PAYLOAD_SHA256,
            },
            "runtime_source_gap": _source_gap(
                root,
                authorization_mtime_ns=authorization_descriptor["mtime_ns"],
            ),
            "completed_run": _runtime_evidence(root),
            "gpu1_evidence": _gpu1_evidence(root),
            "aborted_probe_attempt": _aborted_probe_evidence(root),
            "generator": _descriptor(Path(__file__).resolve()),
            "limitations": [
                "These runtime source files were not bound by the pre-run authorization.",
                "This post-run receipt cannot retroactively create pre-authorization.",
                "Pre-run source mtimes/ctimes are observable metadata, not a clean-worktree or commit attestation.",
                "The shared worktree was already dirty; no clean commit describes the exact launch tree.",
                "Downstream use must preserve these limitations and bind both receipts explicitly.",
            ],
        }
    )


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _require(not path.exists() and not path.is_symlink(), f"receipt exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def create(repo_root: Path) -> dict[str, Any]:
    payload = build_attestation(repo_root)
    path = repo_root.expanduser().resolve() / ATTESTATION
    _write_exclusive(path, payload)
    return {
        "status": STATUS,
        "path": str(path),
        "sha256": sha256_file(path),
        "payload_sha256": payload["integrity"]["payload_sha256"],
    }


def verify(repo_root: Path) -> dict[str, Any]:
    path = repo_root.expanduser().resolve() / ATTESTATION
    observed = _read_json(path, label="supplemental attestation")
    validate_manifest_integrity(observed)
    expected = build_attestation(repo_root)
    _require(observed == expected, "supplemental attestation no longer replays")
    return {
        "status": STATUS,
        "path": str(path),
        "sha256": sha256_file(path),
        "payload_sha256": observed["integrity"]["payload_sha256"],
        "verified": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("command", choices=("create", "verify"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = create(args.repo_root) if args.command == "create" else verify(args.repo_root)
    except (GapAttestationError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
