"""Cap-1536 retry for the selected pre-2009 cp38 GRPO continuation.

The failed cap-1024 smoke remains an immutable predecessor.  This retry uses
fresh smoke/full branches and changes only ``max_completion_length`` (plus the
required versioned output paths).  Seed 31416, the sealed sampler prefix, all
reward gates, and all other optimization/generation settings are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from jobs.retrain_v2 import chk4_pre2009_cp38_grpo_continuation as v1
from jobs.retrain_v2 import chk4_v5_cp24_grpo_continuation as engine
from jobs.retrain_v2 import merge_chk4_selected_sft_checkpoint as selected
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import validate_manifest_integrity


MAX_COMPLETION_LENGTH = 1536
PREDECESSOR_MAX_COMPLETION_LENGTH = 1024
SMOKE_SCHEMA = "chk4-pre2009-cp38-grpo-cap1536-smoke-authorization-v2"
FULL_SCHEMA = "chk4-pre2009-cp38-grpo-cap1536-full-authorization-v2"
GATE_SCHEMA = "chk4-pre2009-cp38-grpo-cap1536-smoke-gate-v2"
STEP_GATE_SCHEMA = v1.STEP_GATE_SCHEMA

SMOKE = engine.Phase(
    name="smoke",
    branch_id=("chk4_from_pre2009_cp38_selected_grpo_smoke_cap1536_v2_20260811"),
    config=Path(
        "configs/retrain_v2/"
        "chk4_decision_grpo_from_pre2009_cp38_selected_"
        "smoke_cap1536_v2_20260811.yaml"
    ),
    config_sha256="289cd52194e3d9dbd1f615fdfad53c584ec6dfdab7abc13e6d23d94280df162a",
    authorization_schema=SMOKE_SCHEMA,
    max_steps=2,
)
FULL = engine.Phase(
    name="full",
    branch_id=("chk4_from_pre2009_cp38_selected_grpo_full_cap1536_v2_20260811"),
    config=Path(
        "configs/retrain_v2/"
        "chk4_decision_grpo_from_pre2009_cp38_selected_"
        "full_cap1536_v2_20260811.yaml"
    ),
    config_sha256="a483e09a44f4e164aca3932c25bd940038e73281c1b5171df50262a5ede7bd3a",
    authorization_schema=FULL_SCHEMA,
    max_steps=-1,
)

PREDECESSOR_BRANCH_ID = v1.SMOKE.branch_id
PREDECESSOR_RUN_ROOT = v1.SMOKE.run_root
PREDECESSOR_CONFIG = v1.SMOKE.config
PREDECESSOR_CONFIG_SHA256 = v1.SMOKE.config_sha256
PREDECESSOR_ARTIFACTS = {
    "authorization": (
        PREDECESSOR_RUN_ROOT / "receipts/authorization.json",
        "e44b63b54795f57c1facc64d585c3ad27ddbbe959b6cb82c6ae83ac3d4422df4",
    ),
    "resolved_runtime": (
        PREDECESSOR_RUN_ROOT / "adapters/chk4_grpo/resolved_runtime_config.json",
        "212175060ab8a837db99321927436dffbace970153e32874067dda7a2748ba32",
    ),
    "reward": (
        PREDECESSOR_RUN_ROOT / "adapters/chk4_grpo/reward.jsonl",
        "627c34ff9029a46a5979e2ac0efb0f6e1939438a95b49827a1de639b041cdab4",
    ),
    "runtime_safety": (
        PREDECESSOR_RUN_ROOT / "adapters/chk4_grpo/runtime_safety.jsonl",
        "0855e1a7bffffbbdaf3cb6d01a60c3e5ac420ec836b00e587ba4211cd168b2ec",
    ),
    "step_gate": (
        PREDECESSOR_RUN_ROOT / "adapters/chk4_grpo/chk4_smoke_step_gate.jsonl",
        "c71c47850e12a5760d5191c5aafb7f9d31f1241c08f4818dcf103e522bbdbf11",
    ),
    "log": (
        PREDECESSOR_RUN_ROOT / "logs/grpo.log",
        "baef13804c66588366409163bde7b939a3e0b8e3f87c389482f577f43c97b782",
    ),
}

ContinuationError = engine.ContinuationError


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ContinuationError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContinuationError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ContinuationError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise ContinuationError(f"{label} is missing or unsafe: {path}")
    rows: list[dict[str, Any]] = []
    try:
        for line_number, raw in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw.strip():
                raise ContinuationError(f"{label} has blank row {line_number}")
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ContinuationError(f"{label} row {line_number} is not an object")
            rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContinuationError(f"{label} is invalid JSONL: {path}") from exc
    return rows


def _read_yaml(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ContinuationError(f"{label} is missing or unsafe: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ContinuationError(f"{label} is invalid YAML: {path}") from exc
    if not isinstance(value, dict):
        raise ContinuationError(f"{label} must be a YAML mapping")
    return value


def _artifact_descriptor(repo_root: Path, name: str) -> dict[str, Any]:
    relative, expected_sha = PREDECESSOR_ARTIFACTS[name]
    path = (repo_root / relative).resolve()
    if not path.is_file() or path.is_symlink():
        raise ContinuationError(f"predecessor {name} is missing or unsafe: {path}")
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha:
        raise ContinuationError(
            f"predecessor {name} hash drift: expected={expected_sha}, "
            f"observed={observed_sha}"
        )
    descriptor: dict[str, Any] = {
        "path": str(path),
        "sha256": observed_sha,
    }
    if name in {"reward", "runtime_safety", "step_gate"}:
        descriptor["rows"] = len(_read_jsonl(path, label=f"predecessor {name}"))
    return descriptor


def _validate_predecessor_failure(repo_root: Path) -> dict[str, Any]:
    descriptors = {
        name: _artifact_descriptor(repo_root, name) for name in PREDECESSOR_ARTIFACTS
    }
    authorization = _read_json(
        Path(descriptors["authorization"]["path"]),
        label="predecessor authorization",
    )
    validate_manifest_integrity(authorization)
    auth_runtime = authorization.get("runtime")
    sampling = (
        auth_runtime.get("sampling_contract")
        if isinstance(auth_runtime, Mapping)
        else None
    )
    if (
        authorization.get("schema_version") != v1.SMOKE_SCHEMA
        or authorization.get("status") != "authorized"
        or authorization.get("branch_id") != PREDECESSOR_BRANCH_ID
        or authorization.get("config")
        != {
            "path": str((repo_root / PREDECESSOR_CONFIG).resolve()),
            "sha256": PREDECESSOR_CONFIG_SHA256,
        }
        or not isinstance(auth_runtime, Mapping)
        or auth_runtime.get("max_completion_length")
        != PREDECESSOR_MAX_COMPLETION_LENGTH
        or auth_runtime.get("max_steps") != 2
        or auth_runtime.get("gpu") != 1
        or not isinstance(sampling, Mapping)
        or sampling.get("seed") != v1.SMOKE_SEED
        or sampling.get("data_seed") != v1.SMOKE_SEED
        or sampling.get("source_indices") != list(v1.SMOKE_SOURCE_INDICES)
        or sampling.get("verified_sampler_prefix", {}).get("sha256")
        != "ee14d73632b79cd7e0610ca0464b19523cd076f6ceb98d96408c9b3d48ebabf2"
    ):
        raise ContinuationError("predecessor authorization binding drift")

    runtime = _read_json(
        Path(descriptors["resolved_runtime"]["path"]),
        label="predecessor resolved runtime",
    )
    branch = runtime.get("chk4_standalone_branch")
    training = runtime.get("training")
    generation = runtime.get("generation")
    environment = runtime.get("environment")
    if (
        not isinstance(branch, Mapping)
        or branch.get("branch_id") != PREDECESSOR_BRANCH_ID
        or branch.get("profile") != "pre2009_cp38_selected_smoke_v1"
        or not isinstance(training, Mapping)
        or training.get("seed") != v1.SMOKE_SEED
        or training.get("max_steps") != 2
        or not isinstance(generation, Mapping)
        or generation.get("max_completion_length") != PREDECESSOR_MAX_COMPLETION_LENGTH
        or generation.get("temperature") != 0.7
        or generation.get("top_p") != 0.9
        or not isinstance(environment, Mapping)
        or environment.get("cuda_visible_devices") != "1"
        or environment.get("world_size") != 1
    ):
        raise ContinuationError("predecessor resolved-runtime binding drift")

    rewards = _read_jsonl(
        Path(descriptors["reward"]["path"]), label="predecessor rewards"
    )
    expected_layout = ["hold"] * 4 + ["hike"] * 4
    observed_layout = [
        row.get("target", {}).get("direction")
        if isinstance(row.get("target"), Mapping)
        else None
        for row in rewards
    ]
    if (
        len(rewards) != 8
        or observed_layout != expected_layout
        or [row.get("reward") for row in rewards[4:]] != [0, 0, 0, 0]
    ):
        raise ContinuationError("predecessor reward failure evidence drift")

    safety_rows = _read_jsonl(
        Path(descriptors["runtime_safety"]["path"]),
        label="predecessor runtime safety",
    )
    if len(safety_rows) != 1:
        raise ContinuationError("predecessor runtime safety must have one row")
    safety = safety_rows[0]
    metrics = safety.get("metrics")
    if (
        safety.get("step") != 1
        or safety.get("completion_clipped_ratio") != 0.625
        or not isinstance(metrics, Mapping)
        or not math.isfinite(float(metrics.get("loss", math.nan)))
        or float(metrics.get("grad_norm", 0.0)) <= 1e-12
    ):
        raise ContinuationError("predecessor runtime-safety evidence drift")

    gate_rows = _read_jsonl(
        Path(descriptors["step_gate"]["path"]), label="predecessor step gate"
    )
    expected_checks = {
        "each_target_direction_correct_nonzero": False,
        "each_target_reward_std_gt_zero": False,
        "loss_finite": True,
        "grad_norm_gt_1e_12": True,
        "clipped_ratio_le_0_25": False,
    }
    if (
        len(gate_rows) != 1
        or gate_rows[0].get("schema_version") != STEP_GATE_SCHEMA
        or gate_rows[0].get("step") != 1
        or gate_rows[0].get("status") != "failed"
        or gate_rows[0].get("checks") != expected_checks
        or gate_rows[0].get("completion_clipped_ratio") != 0.625
    ):
        raise ContinuationError("predecessor immediate-gate evidence drift")

    log_path = Path(descriptors["log"]["path"])
    log_text = log_path.read_text(encoding="utf-8")
    expected_error = (
        "chk4 smoke step gate failed: clipped_ratio_le_0_25, "
        "each_target_direction_correct_nonzero, each_target_reward_std_gt_zero"
    )
    if expected_error not in log_text:
        raise ContinuationError("predecessor terminal error is missing")
    predecessor_output = (
        repo_root / PREDECESSOR_RUN_ROOT / "adapters/chk4_grpo"
    ).resolve()
    unexpected_terminal = [
        predecessor_output / "trainer_state.json",
        predecessor_output / "train_results.json",
        predecessor_output / "adapter_model.safetensors",
    ]
    if any(path.exists() or path.is_symlink() for path in unexpected_terminal):
        raise ContinuationError("predecessor failure acquired terminal artifacts")
    if list(predecessor_output.glob("checkpoint-*")):
        raise ContinuationError("predecessor failure acquired a checkpoint")
    return {
        "schema_version": "chk4-pre2009-cp38-smoke-failure-evidence-v1",
        "branch_id": PREDECESSOR_BRANCH_ID,
        "outcome": "failed_closed_at_step_1",
        "max_completion_length": PREDECESSOR_MAX_COMPLETION_LENGTH,
        "completion_clipped_ratio": 0.625,
        "failed_checks": [
            "clipped_ratio_le_0_25",
            "each_target_direction_correct_nonzero",
            "each_target_reward_std_gt_zero",
        ],
        "passed_checks": ["loss_finite", "grad_norm_gt_1e_12"],
        "artifacts": descriptors,
        "terminal_artifacts_absent": [str(path) for path in unexpected_terminal],
    }


def _config_delta(repo_root: Path, phase: engine.Phase) -> dict[str, Any]:
    predecessor_phase = v1.SMOKE if phase.name == "smoke" else v1.FULL
    predecessor_path = (repo_root / predecessor_phase.config).resolve()
    retry_path = (repo_root / phase.config).resolve()
    predecessor = _read_yaml(predecessor_path, label="predecessor config")
    retry = _read_yaml(retry_path, label="retry config")
    differences = {
        key: {"from": predecessor.get(key), "to": retry.get(key)}
        for key in sorted(set(predecessor) | set(retry))
        if predecessor.get(key) != retry.get(key)
    }
    expected_difference_keys = {
        "max_completion_length",
        "output_dir",
        "peft_merged_model_path",
    }
    if set(differences) != expected_difference_keys:
        raise ContinuationError(
            f"retry config has unauthorized differences: {sorted(differences)}"
        )
    if differences["max_completion_length"] != {
        "from": PREDECESSOR_MAX_COMPLETION_LENGTH,
        "to": MAX_COMPLETION_LENGTH,
    }:
        raise ContinuationError("retry completion cap delta drift")
    predecessor_unchanged = {
        key: value
        for key, value in predecessor.items()
        if key not in expected_difference_keys
    }
    retry_unchanged = {
        key: value
        for key, value in retry.items()
        if key not in expected_difference_keys
    }
    if predecessor_unchanged != retry_unchanged:
        raise ContinuationError("retry unchanged config payload drift")
    unchanged_sha = _sha256_json(predecessor_unchanged)
    return {
        "schema_version": "chk4-grpo-single-hyperparameter-retry-v1",
        "claim": "max_completion_length_is_the_only_hyperparameter_change",
        "phase": phase.name,
        "predecessor_config": {
            "path": str(predecessor_path),
            "sha256": sha256_file(predecessor_path),
        },
        "retry_config": {
            "path": str(retry_path),
            "sha256": sha256_file(retry_path),
        },
        "hyperparameter_change": differences["max_completion_length"],
        "required_versioned_path_changes": {
            key: differences[key] for key in ("output_dir", "peft_merged_model_path")
        },
        "unchanged_config_payload_sha256": unchanged_sha,
        "smoke_seed_contract": (
            {
                "seed": v1.SMOKE_SEED,
                "data_seed": v1.SMOKE_SEED,
                "sampler_prefix_sha256": (
                    "ee14d73632b79cd7e0610ca0464b19523cd076f6ceb98d96408c9b3d48ebabf2"
                ),
                "target_groups": {"1": ["hold", "hike"], "2": ["hold", "cut"]},
            }
            if phase.name == "smoke"
            else None
        ),
    }


def _authorization_context(repo_root: Path, phase: engine.Phase) -> Mapping[str, Any]:
    return {
        "schema_version": "chk4-pre2009-cp38-cap1536-retry-context-v2",
        "predecessor_failure": _validate_predecessor_failure(repo_root),
        "authorized_delta": _config_delta(repo_root, phase),
    }


def _engine_updates() -> dict[str, Any]:
    updates = v1._engine_updates()
    updates.update(
        {
            "SMOKE_SCHEMA": SMOKE_SCHEMA,
            "FULL_SCHEMA": FULL_SCHEMA,
            "GATE_SCHEMA": GATE_SCHEMA,
            "STEP_GATE_SCHEMA": STEP_GATE_SCHEMA,
            "SMOKE": SMOKE,
            "FULL": FULL,
            "MAX_COMPLETION_LENGTH": MAX_COMPLETION_LENGTH,
            "AUTHORIZATION_CONTEXT_PROVIDER": _authorization_context,
            "RUNTIME_PROFILE_PREFIX": "pre2009_cp38_selected_cap1536_v2",
            "CONTINUATION_SCHEMA_PREFIX": "chk4-pre2009-cp38-grpo-cap1536-v2",
            "FORBIDDEN_CANONICAL_SCOPE": ("historical_pre2009_canonical_grpo_output"),
            "ADDITIONAL_FORBIDDEN_SCOPES": (
                "source_checkpoint_39",
                "source_root_adapter_checkpoint_39",
                "failed_cap1024_smoke_output_reuse",
                "cap_other_than_1536",
            ),
            "CONTINUATION_IMPLEMENTATION_FILES": (
                "jobs/retrain_v2/chk4_pre2009_cp38_grpo_cap1536_v2_continuation.py",
                "jobs/retrain_v2/chk4_v5_cp24_grpo_continuation.py",
                "jobs/retrain_v2/chk4_pre2009_cp38_grpo_continuation.py",
                "jobs/retrain_v2/merge_chk4_selected_sft_checkpoint.py",
            ),
        }
    )
    return updates


@contextmanager
def _engine_contract() -> Iterator[None]:
    updates = _engine_updates()
    previous = {name: getattr(engine, name) for name in updates}
    previous_profile = selected.ACTIVE_SELECTION_PROFILE.name
    try:
        for name, value in updates.items():
            setattr(engine, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(engine, name, value)
        selected._activate_selection_profile(previous_profile)


def _call(name: str, *args: Any, **kwargs: Any) -> Any:
    with _engine_contract():
        return getattr(engine, name)(*args, **kwargs)


def _validate_config(repo_root: Path, phase: engine.Phase) -> dict[str, Any]:
    return _call("_validate_config", repo_root, phase)


def _smoke_sampling_contract(
    repo_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    return _call("_smoke_sampling_contract", repo_root, config)


def preflight(repo_root: Path) -> dict[str, Any]:
    result = _call("preflight", repo_root)
    result["retry"] = {
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "predecessor_failure": _validate_predecessor_failure(repo_root),
        "smoke_delta": _config_delta(repo_root, SMOKE),
        "full_delta": _config_delta(repo_root, FULL),
    }
    return result


def status(repo_root: Path) -> dict[str, Any]:
    return _call("status", repo_root)


def authorize(repo_root: Path, phase: engine.Phase, *, execute: bool) -> dict[str, Any]:
    return _call("authorize", repo_root, phase, execute=execute)


def training_command(repo_root: Path, phase: engine.Phase) -> dict[str, Any]:
    return _call("training_command", repo_root, phase)


def execute_training(repo_root: Path, phase: engine.Phase) -> None:
    with _engine_contract():
        engine.execute_training(repo_root, phase)


def gate_smoke(repo_root: Path, *, execute: bool) -> dict[str, Any]:
    return _call("gate_smoke", repo_root, execute=execute)


def _authorization_payload(repo_root: Path, phase: engine.Phase) -> dict[str, Any]:
    return _call("_authorization_payload", repo_root, phase)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "status"):
        subparsers.add_parser(command)
    for command in (
        "authorize-smoke",
        "launch-smoke",
        "gate-smoke",
        "authorize-full",
        "launch-full",
    ):
        child = subparsers.add_parser(command)
        child.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    try:
        if args.command == "preflight":
            result = preflight(repo_root)
        elif args.command == "status":
            result = status(repo_root)
        elif args.command == "authorize-smoke":
            result = authorize(repo_root, SMOKE, execute=args.execute)
        elif args.command == "authorize-full":
            result = authorize(repo_root, FULL, execute=args.execute)
        elif args.command == "gate-smoke":
            result = gate_smoke(repo_root, execute=args.execute)
        elif args.command in {"launch-smoke", "launch-full"}:
            phase = SMOKE if args.command == "launch-smoke" else FULL
            if args.execute:
                execute_training(repo_root, phase)
                raise AssertionError("os.execvpe unexpectedly returned")
            result = training_command(repo_root, phase)
        else:  # pragma: no cover
            raise ContinuationError(f"unsupported command: {args.command}")
    except (
        ContinuationError,
        selected.SelectionMergeError,
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
