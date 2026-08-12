"""Isolated Decision-GRPO v3 continuation from an exact Decision-SFT merge.

This module deliberately cannot train or merge the source SFT stage.  It accepts
only the immutable, exact-merge lineage published by the existing
``warm_fix_lr1e5_v2`` branch and launches a fresh, separately authorized GRPO
branch.  All commands are read-only unless ``authorize --execute`` or
``launch --execute`` is explicitly requested.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Iterator, Mapping, Sequence

import yaml

from jobs.main.checkpoint_provenance import write_immutable_json
from jobs.retrain_v2.merge_attestation import verify_merge_attestation
from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.validator.loo_generation_spec import validate_manifest_integrity


SCHEMA_VERSION = "chk4-relaxed-grpo-continuation-v1"
AUTHORIZATION_SCHEMA = "chk4-relaxed-grpo-authorization-v1"
RECEIPT_SCHEMA = "chk4-relaxed-grpo-source-reuse-receipt-v1"
BRANCH_ID = "chk4_grpo_relaxed_v3_from_warm_fix_lr1e5_v2_20260810"
PROFILE_NAME = "relaxed_v3_continuation"
SOURCE_BRANCH_ID = "chk4_from_chk1_cp200_sft_grpo_warm_fix_lr1e5_v2_20260810"
SOURCE_PROFILE = "warm_fix_lr1e5_v2"
SOURCE_ROOT = Path(f"output/training/retrain_v2/{SOURCE_BRANCH_ID}")
RUN_ROOT = Path(f"output/training/retrain_v2/{BRANCH_ID}")
SOURCE_SFT_CONFIG = Path(
    "configs/retrain_v2/"
    "chk4_decision_sft_from_chk1_cp200_warm_fix_lr1e5_v2_20260810.yaml"
)
GRPO_CONFIG = Path(
    "configs/retrain_v2/"
    "chk4_decision_grpo_relaxed_v3_from_sft_warm_fix_lr1e5_v2_20260810.yaml"
)
RELEASE_MANIFEST = Path(
    "dataset/processed/retrain_v2/"
    "chk4_decision_warmstart_grpo_core_v3_20260810/release_manifest.json"
)
REWARD_SOURCE = Path(
    "src/open_r1/trainer/rewards/reward_funcs/decision_reward_v3.py"
)
REWARD_REGISTRY = Path("src/open_r1/trainer/rewards/reward_register.py")
EXPECTED_RELEASE_SHA256 = (
    "8b05dce09bcb0a0b27ee240da1e5d730be93921ce19e650a3b3e8b2689adb893"
)
EXPECTED_SYSTEM_PROMPT_SHA256 = (
    "426b64532dfb955a3941db2cc2bcfc9a785c0540ffba9ee94afabc189fe4ce18"
)
USER_RELAXATION_INSTRUCTION = (
    "帮我把这个地方放宽一下，有</think>标签就可以，然后帮我确认一下"
    "是不是</think>标签之后的都是answer"
)
APPLIED_RELAXATION = {
    "required_boundary": "exactly one literal </think>",
    "reasoning": "non-empty text before the boundary",
    "answer": "all text after the boundary",
    "strict_answer": "one schema-exact plain JSON decision object keeps v2 scoring",
    "narrow_recovery": (
        "one complete lowercase json Markdown fence containing one schema-exact "
        "decision object receives semantic credit at a fixed 0.25 discount"
    ),
    "still_zero": (
        "missing/multiple boundary, empty reasoning/answer, malformed or partial "
        "fence, extra text/JSON, or schema/domain violation"
    ),
}
CHECKPOINT_RE = re.compile(r"\Acheckpoint-(0|[1-9][0-9]*)\Z")
_EXEC_LOCK_HANDLE: IO[str] | None = None


class RelaxedBranchError(RuntimeError):
    """Raised when a continuation lineage or launch invariant fails closed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _signed(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    unsigned = dict(payload)
    return {**unsigned, field: _sha256_text(_canonical_json(unsigned))}


def _validate_signature(
    payload: Mapping[str, Any], *, field: str, label: str
) -> None:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    if observed != _sha256_text(_canonical_json(unsigned)):
        raise RelaxedBranchError(f"{label} canonical payload hash drift")


def _repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _safe_file(repo_root: Path, value: str | Path, *, label: str) -> Path:
    path = _repo_path(repo_root, value)
    if not path.is_file() or path.is_symlink():
        raise RelaxedBranchError(f"{label} is missing or unsafe: {path}")
    return path


def _safe_dir(repo_root: Path, value: str | Path, *, label: str) -> Path:
    path = _repo_path(repo_root, value)
    if not path.is_dir() or path.is_symlink():
        raise RelaxedBranchError(f"{label} is missing or unsafe: {path}")
    return path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RelaxedBranchError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RelaxedBranchError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RelaxedBranchError(f"{label} must contain a JSON object: {path}")
    return value


def _read_yaml(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RelaxedBranchError(f"{label} is missing or unsafe: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RelaxedBranchError(f"{label} is invalid YAML: {path}") from exc
    if not isinstance(value, dict):
        raise RelaxedBranchError(f"{label} must contain a YAML mapping: {path}")
    return value


def _path_from_config(
    repo_root: Path, config: Mapping[str, Any], key: str
) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise RelaxedBranchError(f"training config is missing {key}")
    return _repo_path(repo_root, value)


def _file_binding(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RelaxedBranchError(f"bound file is missing or unsafe: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _require_equal(observed: Any, expected: Any, *, label: str) -> None:
    if observed != expected:
        raise RelaxedBranchError(
            f"{label} drift: expected {expected!r}, observed {observed!r}"
        )


def _paths(repo_root: Path) -> dict[str, Path]:
    root = _repo_path(repo_root, RUN_ROOT)
    receipts = root / "receipts"
    return {
        "root": root,
        "authorization": receipts / "relaxed_grpo_authorization.json",
        "source_receipt": receipts / "source_sft_reuse_receipt.json",
        "lock": root / ".branch.lock",
    }


def _source_paths(repo_root: Path) -> dict[str, Path]:
    root = _repo_path(repo_root, SOURCE_ROOT)
    return {
        "root": root,
        "config": _safe_file(repo_root, SOURCE_SFT_CONFIG, label="source SFT config"),
        "adapter": root / "adapters/chk4_sft",
        "merged": root / "merged/chk4_sft",
        "stage": root / "receipts/sft_merge_stage.json",
        "exact": root / "receipts/sft_exact_merge.json",
    }


def _config_and_reward_bindings(repo_root: Path) -> dict[str, Any]:
    config_path = _safe_file(repo_root, GRPO_CONFIG, label="GRPO v3 config")
    source_config_path = _safe_file(
        repo_root, SOURCE_SFT_CONFIG, label="source SFT config"
    )
    release_path = _safe_file(
        repo_root, RELEASE_MANIFEST, label="chk4 release manifest"
    )
    reward_source = _safe_file(repo_root, REWARD_SOURCE, label="reward v3 source")
    reward_registry = _safe_file(repo_root, REWARD_REGISTRY, label="reward registry")
    config = _read_yaml(config_path, label="GRPO v3 config")
    source_config = _read_yaml(source_config_path, label="source SFT config")

    expected_source_merged = _path_from_config(
        repo_root, source_config, "peft_merged_model_path"
    )
    expected_output = _repo_path(repo_root, RUN_ROOT / "adapters/chk4_grpo")
    expected_merged = _repo_path(repo_root, RUN_ROOT / "merged/chk4_grpo")
    observed_parent = _path_from_config(repo_root, config, "model_name_or_path")
    observed_output = _path_from_config(repo_root, config, "output_dir")
    observed_merged = _path_from_config(repo_root, config, "peft_merged_model_path")
    _require_equal(observed_parent, expected_source_merged, label="GRPO source SFT merge")
    _require_equal(observed_output, expected_output, label="fresh GRPO adapter path")
    _require_equal(observed_merged, expected_merged, label="fresh GRPO merged path")
    if len({observed_parent, observed_output, observed_merged}) != 3:
        raise RelaxedBranchError("source, adapter, and merged paths must be distinct")
    if str(observed_output).startswith(str(_repo_path(repo_root, SOURCE_ROOT))) or str(
        observed_merged
    ).startswith(str(_repo_path(repo_root, SOURCE_ROOT))):
        raise RelaxedBranchError("GRPO outputs must not overlap the source SFT branch")

    expected_fields = {
        "dataset_chk4_role": "decision_grpo",
        "dataset_train_split": "train",
        "dataset_test_split": "validation",
        "dataset_chk4_release_manifest_sha256": EXPECTED_RELEASE_SHA256,
        "reward_funcs": ["decision_dense_v3"],
        "reward_weights": [1.0],
        "max_prompt_length": 2560,
        "max_completion_length": 1024,
        "num_generations": 4,
        "generation_batch_size": 8,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "load_in_4bit": True,
        "dtype": "bfloat16",
        "use_vllm": False,
        "overwrite_output_dir": False,
        "resume_from_checkpoint": None,
        "save_steps": 1,
        "save_total_limit": None,
        "checkpoint_keep_last": 3,
        "checkpoint_keep_every_n_steps": 10,
    }
    for key, expected in expected_fields.items():
        _require_equal(config.get(key), expected, label=f"GRPO config {key}")
    _require_equal(
        _repo_path(repo_root, str(config.get("dataset_chk4_release_manifest"))),
        release_path,
        label="GRPO release path",
    )
    system_prompt = config.get("system_prompt")
    if not isinstance(system_prompt, str):
        raise RelaxedBranchError("GRPO system prompt is missing")
    _require_equal(
        _sha256_text(system_prompt),
        EXPECTED_SYSTEM_PROMPT_SHA256,
        label="GRPO system prompt SHA",
    )
    _require_equal(
        system_prompt, source_config.get("system_prompt"), label="SFT/GRPO prompt"
    )
    if sha256_file(release_path) != EXPECTED_RELEASE_SHA256:
        raise RelaxedBranchError("chk4 release manifest hash drift")
    release = _read_json(release_path, label="chk4 release manifest")
    if (
        release.get("immutable") is not True
        or release.get("training_ready") is not True
        or release.get("quality_status") != "passed"
    ):
        raise RelaxedBranchError("chk4 release is not immutable and training-ready")

    reward_text = reward_source.read_text(encoding="utf-8")
    registry_text = reward_registry.read_text(encoding="utf-8")
    for literal, label in (
        ("def decision_dense_reward_v3(", "reward implementation"),
        ('"decision_dense_v3":', "reward registration"),
    ):
        text = reward_text if "implementation" in label else registry_text
        if literal not in text:
            raise RelaxedBranchError(f"decision_dense_v3 {label} is missing")
    reward_contract = {
        "name": "decision_dense_v3",
        "max_completion_length": 1024,
        "reward_weights": [1.0],
        "applied_relaxation": APPLIED_RELAXATION,
    }
    return {
        "grpo_config": _file_binding(config_path),
        "source_sft_config": _file_binding(source_config_path),
        "release_manifest": _file_binding(release_path),
        "reward": {
            "source": _file_binding(reward_source),
            "registry": _file_binding(reward_registry),
            "config": _file_binding(config_path),
            "contract": reward_contract,
            "contract_sha256": _sha256_text(_canonical_json(reward_contract)),
        },
        "paths": {
            "source_merged_sft": str(observed_parent),
            "fresh_grpo_adapter": str(observed_output),
            "fresh_grpo_merged": str(observed_merged),
        },
    }


def _source_training_binding(repo_root: Path) -> dict[str, Any]:
    paths = _source_paths(repo_root)
    config = _read_yaml(paths["config"], label="source SFT config")
    adapter = _safe_dir(repo_root, paths["adapter"], label="source SFT adapter")
    required = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "resolved_runtime_config.json",
        "train_results.json",
        "trainer_state.json",
    )
    for name in required:
        _safe_file(repo_root, adapter / name, label=f"source SFT {name}")
    state_path = adapter / "trainer_state.json"
    runtime_path = adapter / "resolved_runtime_config.json"
    results_path = adapter / "train_results.json"
    state = _read_json(state_path, label="source SFT trainer state")
    runtime = _read_json(runtime_path, label="source SFT runtime")
    results = _read_json(results_path, label="source SFT results")
    step = state.get("global_step")
    epoch = state.get("epoch")
    max_steps = config.get("max_steps")
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or step != max_steps
        or step != 8
        or isinstance(epoch, bool)
        or not isinstance(epoch, (int, float))
        or not math.isfinite(float(epoch))
    ):
        raise RelaxedBranchError("source SFT did not complete the authorized 8 steps")
    train_loss = results.get("train_loss")
    if (
        isinstance(train_loss, bool)
        or not isinstance(train_loss, (int, float))
        or not math.isfinite(float(train_loss))
    ):
        raise RelaxedBranchError("source SFT train loss is not finite")
    branch = runtime.get("chk4_standalone_branch")
    dataset = runtime.get("dataset")
    decision = dataset.get("chk4_decision") if isinstance(dataset, Mapping) else None
    release = decision.get("release_manifest") if isinstance(decision, Mapping) else None
    scope = decision.get("scope") if isinstance(decision, Mapping) else None
    runtime_config = branch.get("config") if isinstance(branch, Mapping) else None
    if (
        not isinstance(branch, Mapping)
        or branch.get("branch_id") != SOURCE_BRANCH_ID
        or branch.get("profile") != SOURCE_PROFILE
        or branch.get("stage") != "decision_sft"
        or not isinstance(runtime_config, Mapping)
        or Path(str(runtime_config.get("path"))).resolve() != paths["config"]
        or runtime_config.get("sha256") != sha256_file(paths["config"])
        or not isinstance(scope, Mapping)
        or scope.get("role") != "decision_sft"
        or not isinstance(release, Mapping)
        or release.get("sha256") != EXPECTED_RELEASE_SHA256
    ):
        raise RelaxedBranchError("source SFT runtime lineage drift")
    model = runtime.get("model")
    training = runtime.get("training")
    peft = runtime.get("peft")
    if not all(isinstance(item, Mapping) for item in (model, training, peft)):
        raise RelaxedBranchError("source SFT runtime model/training/PEFT binding missing")
    checks = {
        "model_name_or_path": (
            _repo_path(repo_root, str(model.get("model_name_or_path"))),
            _path_from_config(repo_root, config, "model_name_or_path"),
        ),
        "output_dir": (
            _repo_path(repo_root, str(training.get("output_dir"))), adapter
        ),
        "merged_model_path": (
            _repo_path(repo_root, str(peft.get("merged_model_path"))),
            _path_from_config(repo_root, config, "peft_merged_model_path"),
        ),
    }
    for label, (observed, expected) in checks.items():
        _require_equal(observed, expected, label=f"source runtime {label}")
    for key in (
        "learning_rate",
        "num_train_epochs",
        "max_steps",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "seed",
        "bf16",
    ):
        observed = training.get(key)
        expected = config.get(key)
        if isinstance(expected, float) and isinstance(observed, (int, float)):
            if isinstance(observed, bool) or not math.isclose(
                float(observed), expected, rel_tol=0.0, abs_tol=1e-15
            ):
                raise RelaxedBranchError(f"source runtime training.{key} drift")
        elif observed != expected:
            raise RelaxedBranchError(f"source runtime training.{key} drift")
    adapter_fingerprint = fingerprint_artifact_path(adapter)
    return {
        "adapter": adapter_fingerprint,
        "runtime_config": _file_binding(runtime_path),
        "trainer_state": _file_binding(state_path),
        "train_results": _file_binding(results_path),
        "global_step": step,
        "epoch": float(epoch),
        "train_loss": float(train_loss),
    }


def _source_lineage(repo_root: Path) -> dict[str, Any]:
    paths = _source_paths(repo_root)
    stage_path = _safe_file(repo_root, paths["stage"], label="source merge stage")
    exact_path = _safe_file(repo_root, paths["exact"], label="source exact evidence")
    merged = _safe_dir(repo_root, paths["merged"], label="source merged SFT")
    training = _source_training_binding(repo_root)
    stage = _read_json(stage_path, label="source merge stage")
    _validate_signature(stage, field="receipt_sha256", label="source merge stage")
    attestation_path = _safe_file(
        repo_root, merged / "merge_attestation.json", label="source merge attestation"
    )
    exact = _read_json(exact_path, label="source exact evidence")
    validate_manifest_integrity(exact)
    attestation = _read_json(attestation_path, label="source merge attestation")
    binding = attestation.get("binding")
    sources = exact.get("sources")
    completion = binding.get("training_completion") if isinstance(binding, Mapping) else None
    observed_merged = fingerprint_artifact_path(merged)
    stage_exact = stage.get("exact_merge_evidence")
    stage_attestation = stage.get("merge_attestation")
    if (
        stage.get("schema_version") != "chk4-standalone-stage-receipt-v1"
        or stage.get("status") != "passed"
        or
        stage.get("branch_id") != SOURCE_BRANCH_ID
        or stage.get("stage") != "sft_merge"
        or Path(str(stage.get("config", {}).get("path"))).resolve() != paths["config"]
        or stage.get("config", {}).get("sha256") != sha256_file(paths["config"])
        or stage.get("merged_artifact") != observed_merged
        or not isinstance(stage_exact, Mapping)
        or Path(str(stage_exact.get("path"))).resolve() != exact_path
        or stage_exact.get("file_sha256") != sha256_file(exact_path)
        or stage_exact.get("payload_sha256")
        != exact.get("integrity", {}).get("payload_sha256")
        or not isinstance(stage_attestation, Mapping)
        or Path(str(stage_attestation.get("path"))).resolve() != attestation_path
        or stage_attestation.get("file_sha256") != sha256_file(attestation_path)
        or exact.get("schema_version") != "lora-merge-lineage-evidence-v1"
        or exact.get("subject_artifact_id") != "chk4-decision-sft-warm-start"
        or exact.get("conclusion") != "exact_base_plus_adapter_merge_verified"
        or not isinstance(binding, Mapping)
        or attestation.get("schema_version") != 1
        or attestation.get("attestation_type") != "retrain_v2_merge"
        or stage_attestation.get("binding_sha256")
        != _sha256_text(_canonical_json(binding))
        or binding.get("branch_id") != SOURCE_BRANCH_ID
        or binding.get("schema_version") != "chk4-standalone-merge-binding-v1"
        or binding.get("merge_kind") != "sft_warm_start"
        or binding.get("release_manifest", {}).get("sha256")
        != EXPECTED_RELEASE_SHA256
        or binding.get("source_adapter") != training["adapter"]
        or not isinstance(completion, Mapping)
        or completion.get("adapter") != training["adapter"]
        or completion.get("config_sha256") != sha256_file(paths["config"])
        or completion.get("runtime_config_sha256")
        != training["runtime_config"]["sha256"]
        or completion.get("trainer_state_sha256")
        != training["trainer_state"]["sha256"]
        or completion.get("train_results_sha256")
        != training["train_results"]["sha256"]
        or completion.get("global_step") != training["global_step"]
        or not isinstance(sources, Mapping)
        or sources.get("adapter") != training["adapter"]
        or sources.get("merged_model") != stage.get("merged_artifact")
    ):
        raise RelaxedBranchError("source SFT merge/training binding drift")
    try:
        verify_merge_attestation(merged, expected_binding=binding)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RelaxedBranchError(f"source merge attestation drift: {exc}") from exc
    return {
        "branch_id": SOURCE_BRANCH_ID,
        "profile": SOURCE_PROFILE,
        "training": training,
        "merge_stage": {
            **_file_binding(paths["stage"]),
            "receipt_sha256": stage.get("receipt_sha256"),
        },
        "merge_attestation": {
            **_file_binding(attestation_path),
            "binding_sha256": stage_attestation.get("binding_sha256"),
            "canonical_payload_sha256": attestation.get(
                "canonical_payload_sha256"
            ),
        },
        "exact_merge_evidence": {
            **_file_binding(exact_path),
            "payload_sha256": exact.get("integrity", {}).get("payload_sha256"),
            "conclusion": exact.get("conclusion"),
        },
        "merged_artifact": dict(stage["merged_artifact"]),
    }


def _binding_snapshot(repo_root: Path) -> dict[str, Any]:
    config_reward = _config_and_reward_bindings(repo_root)
    source = _source_lineage(repo_root)
    return {
        "source_sft": source,
        "grpo_config": config_reward["grpo_config"],
        "source_sft_config": config_reward["source_sft_config"],
        "release_manifest": config_reward["release_manifest"],
        "reward": config_reward["reward"],
        "paths": config_reward["paths"],
    }


def _authorization_payload(bindings: Mapping[str, Any]) -> dict[str, Any]:
    instruction = {
        "language": "zh-CN",
        "verbatim": USER_RELAXATION_INSTRUCTION,
        "sha256": _sha256_text(USER_RELAXATION_INSTRUCTION),
        "applied_interpretation": APPLIED_RELAXATION,
    }
    payload = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "status": "authorized",
        "authorization_basis": "explicit_user_instruction",
        "branch_id": BRANCH_ID,
        "source_branch_id": SOURCE_BRANCH_ID,
        "explicit_relaxation_instruction": instruction,
        "scope": {
            "allowed": ["chk4_decision_grpo_training", "chk4_decision_grpo_merge"],
            "not_authorized": [
                "source_sft_retraining",
                "source_sft_copy",
                "source_sft_mutation",
                "canonical_retrain_v2_dag",
                "chk1",
                "chk2",
                "chk3",
            ],
        },
        "launch_contract": {
            "physical_gpu": 1,
            "cuda_visible_devices": "1",
            "accelerate_num_processes": 1,
            "effective_train_batch": 8,
            "fresh_or_resume": "fresh output or numerically latest complete checkpoint only",
        },
        "bindings": dict(bindings),
    }
    return _signed(payload, "authorization_sha256")


def _source_receipt_payload(
    bindings: Mapping[str, Any], authorization_path: Path, authorization: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "passed",
        "stage": "source_sft_reuse_authorized",
        "branch_id": BRANCH_ID,
        "training_performed": False,
        "source_copied": False,
        "authorization": {
            **_file_binding(authorization_path),
            "payload_sha256": authorization["authorization_sha256"],
        },
        "bindings": dict(bindings),
    }
    return _signed(payload, "receipt_sha256")


def _load_authorized(
    repo_root: Path, *, bindings: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    if bindings is None:
        bindings = _binding_snapshot(repo_root)
    expected_authorization = _authorization_payload(bindings)
    paths = _paths(repo_root)
    authorization = _read_json(paths["authorization"], label="GRPO authorization")
    _validate_signature(
        authorization, field="authorization_sha256", label="GRPO authorization"
    )
    if authorization != expected_authorization:
        raise RelaxedBranchError("GRPO authorization bindings drift")
    expected_receipt = _source_receipt_payload(
        bindings, paths["authorization"], authorization
    )
    receipt = _read_json(paths["source_receipt"], label="source SFT reuse receipt")
    _validate_signature(receipt, field="receipt_sha256", label="source SFT reuse receipt")
    if receipt != expected_receipt:
        raise RelaxedBranchError("source SFT reuse receipt bindings drift")
    return authorization, receipt


@contextmanager
def _branch_lock(repo_root: Path) -> Iterator[None]:
    lock = _paths(repo_root)["lock"]
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.is_symlink():
        raise RelaxedBranchError("branch lock must not be a symlink")
    with lock.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RelaxedBranchError("another relaxed GRPO operation owns the lock") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _acquire_exec_lock(repo_root: Path) -> None:
    global _EXEC_LOCK_HANDLE
    if _EXEC_LOCK_HANDLE is not None:
        raise RelaxedBranchError("current process already owns the branch lock")
    lock = _paths(repo_root)["lock"]
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.is_symlink():
        raise RelaxedBranchError("branch lock must not be a symlink")
    handle = lock.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RelaxedBranchError("another relaxed GRPO operation owns the lock") from exc
    os.set_inheritable(handle.fileno(), True)
    _EXEC_LOCK_HANDLE = handle


def preflight(repo_root: Path) -> dict[str, Any]:
    bindings = _binding_snapshot(repo_root)
    paths = _paths(repo_root)
    authorization = None
    if paths["authorization"].exists() or paths["source_receipt"].exists():
        auth, receipt = _load_authorized(repo_root, bindings=bindings)
        authorization = {
            "status": "verified",
            "authorization": {
                **_file_binding(paths["authorization"]),
                "payload_sha256": auth["authorization_sha256"],
            },
            "source_reuse_receipt": {
                **_file_binding(paths["source_receipt"]),
                "payload_sha256": receipt["receipt_sha256"],
            },
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "branch_id": BRANCH_ID,
        "source_branch_id": SOURCE_BRANCH_ID,
        "canonical_dag_bindable": False,
        "gpu_training_started": False,
        "bindings": bindings,
        "authorization": authorization,
    }


def authorize(repo_root: Path, *, execute: bool) -> dict[str, Any]:
    checked = preflight(repo_root)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "status": "planned" if not execute else "executing",
        "operation": "authorize-relaxed-grpo-continuation",
        "execute": execute,
        "branch_id": BRANCH_ID,
        "source_branch_id": SOURCE_BRANCH_ID,
        "training_performed": False,
        "source_copied": False,
    }
    if not execute:
        return plan
    with _branch_lock(repo_root):
        # Recheck all source, merge, reward, release, and config hashes while the
        # new branch lock is held.  Source artifacts are never written here.
        bindings = _binding_snapshot(repo_root)
        if bindings != checked["bindings"]:
            raise RelaxedBranchError("preflight bindings changed before authorization")
        paths = _paths(repo_root)
        authorization = _authorization_payload(bindings)
        write_immutable_json(paths["authorization"], authorization)
        paths["authorization"].chmod(0o400)
        stored_authorization = _read_json(
            paths["authorization"], label="GRPO authorization"
        )
        if stored_authorization != authorization:
            raise RelaxedBranchError("published GRPO authorization differs")
        receipt = _source_receipt_payload(
            bindings, paths["authorization"], stored_authorization
        )
        write_immutable_json(paths["source_receipt"], receipt)
        paths["source_receipt"].chmod(0o400)
        _load_authorized(repo_root, bindings=bindings)
    return {
        **plan,
        "status": "passed",
        "authorization": {
            **_file_binding(paths["authorization"]),
            "payload_sha256": authorization["authorization_sha256"],
        },
        "source_reuse_receipt": {
            **_file_binding(paths["source_receipt"]),
            "payload_sha256": receipt["receipt_sha256"],
        },
    }


def _verify_runtime(
    repo_root: Path, config_path: Path, config: Mapping[str, Any], output: Path
) -> None:
    runtime = _read_json(output / "resolved_runtime_config.json", label="GRPO runtime")
    branch = runtime.get("chk4_standalone_branch")
    runtime_config = branch.get("config") if isinstance(branch, Mapping) else None
    if (
        not isinstance(branch, Mapping)
        or branch.get("branch_id") != BRANCH_ID
        or branch.get("profile") != PROFILE_NAME
        or branch.get("stage") != "decision_grpo"
        or not isinstance(runtime_config, Mapping)
        or Path(str(runtime_config.get("path"))).resolve() != config_path
        or runtime_config.get("sha256") != sha256_file(config_path)
    ):
        raise RelaxedBranchError("GRPO runtime branch/config binding drift")
    model = runtime.get("model")
    dataset = runtime.get("dataset")
    decision = dataset.get("chk4_decision") if isinstance(dataset, Mapping) else None
    release = decision.get("release_manifest") if isinstance(decision, Mapping) else None
    scope = decision.get("scope") if isinstance(decision, Mapping) else None
    generation = runtime.get("generation")
    rewards = runtime.get("rewards")
    training = runtime.get("training")
    peft = runtime.get("peft")
    environment = runtime.get("environment")
    if not all(
        isinstance(item, Mapping)
        for item in (model, generation, rewards, training, peft, environment)
    ):
        raise RelaxedBranchError("GRPO runtime binding is incomplete")
    if (
        _repo_path(repo_root, str(model.get("model_name_or_path")))
        != _path_from_config(repo_root, config, "model_name_or_path")
        or not isinstance(scope, Mapping)
        or scope.get("role") != "decision_grpo"
        or not isinstance(release, Mapping)
        or release.get("sha256") != EXPECTED_RELEASE_SHA256
        or rewards.get("reward_funcs") != ["decision_dense_v3"]
        or list(rewards.get("reward_weights") or []) != [1.0]
        or environment.get("cuda_visible_devices") != "1"
        or environment.get("n_gpu") != 1
        or environment.get("world_size") != 1
    ):
        raise RelaxedBranchError("GRPO runtime source/reward/GPU binding drift")
    for key in (
        "max_prompt_length",
        "max_completion_length",
        "num_generations",
        "temperature",
        "top_p",
    ):
        if generation.get(key) != config.get(key):
            raise RelaxedBranchError(f"GRPO runtime generation.{key} drift")
    expected_training = {
        "output_dir": config["output_dir"],
        "learning_rate": config["learning_rate"],
        "num_train_epochs": config["num_train_epochs"],
        "max_steps": config.get("max_steps", -1),
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "gradient_checkpointing": config["gradient_checkpointing"],
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "per_device_eval_batch_size": config["per_device_eval_batch_size"],
        "seed": config["seed"],
        "bf16": config["bf16"],
    }
    for key, expected in expected_training.items():
        observed = training.get(key)
        if key == "output_dir":
            if _repo_path(repo_root, str(observed)) != output:
                raise RelaxedBranchError("GRPO runtime output directory drift")
        elif isinstance(expected, float) and isinstance(observed, (int, float)):
            if isinstance(observed, bool) or not math.isclose(
                float(observed), expected, rel_tol=0.0, abs_tol=1e-15
            ):
                raise RelaxedBranchError(f"GRPO runtime training.{key} drift")
        elif observed != expected:
            raise RelaxedBranchError(f"GRPO runtime training.{key} drift")
    if _repo_path(repo_root, str(peft.get("merged_model_path"))) != _path_from_config(
        repo_root, config, "peft_merged_model_path"
    ):
        raise RelaxedBranchError("GRPO runtime merged output drift")


def _training_reached_end(
    config: Mapping[str, Any], *, epoch: float, global_step: int
) -> bool:
    max_steps = int(config.get("max_steps", -1))
    if max_steps > 0:
        return global_step >= max_steps
    return epoch + 1e-6 >= float(config.get("num_train_epochs", 0))


def _resume_state(repo_root: Path) -> dict[str, Any]:
    config_path = _safe_file(repo_root, GRPO_CONFIG, label="GRPO v3 config")
    config = _read_yaml(config_path, label="GRPO v3 config")
    output = _path_from_config(repo_root, config, "output_dir")
    if not output.exists():
        return {"mode": "fresh", "output_dir": str(output)}
    if not output.is_dir() or output.is_symlink():
        raise RelaxedBranchError("GRPO output is not a safe directory")
    _verify_runtime(repo_root, config_path, config, output)
    root_state = output / "trainer_state.json"
    if root_state.is_file() and not root_state.is_symlink():
        state = _read_json(root_state, label="GRPO root trainer state")
        epoch = state.get("epoch")
        step = state.get("global_step")
        if (
            not isinstance(epoch, bool)
            and isinstance(epoch, (int, float))
            and math.isfinite(float(epoch))
            and not isinstance(step, bool)
            and isinstance(step, int)
            and _training_reached_end(config, epoch=float(epoch), global_step=step)
        ):
            raise RelaxedBranchError("GRPO training is complete; refusing to relaunch")
    checkpoints: dict[int, Path] = {}
    for candidate in output.iterdir():
        if not candidate.name.startswith("checkpoint-"):
            continue
        match = CHECKPOINT_RE.fullmatch(candidate.name)
        if match is None or candidate.is_symlink() or not candidate.is_dir():
            raise RelaxedBranchError(f"unsafe checkpoint-like entry: {candidate}")
        step = int(match.group(1))
        if step <= 0:
            raise RelaxedBranchError(f"checkpoint step must be positive: {candidate}")
        checkpoints[step] = candidate
    if not checkpoints:
        raise RelaxedBranchError(
            "GRPO output exists but has no resumable numeric checkpoint"
        )
    step = max(checkpoints)
    latest = checkpoints[step]
    required = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    )
    missing = [
        name
        for name in required
        if not (latest / name).is_file() or (latest / name).is_symlink()
    ]
    if missing:
        raise RelaxedBranchError(
            f"latest checkpoint-{step} is incomplete; missing {sorted(missing)}"
        )
    rng = (latest / "rng_state.pth", latest / "rng_state_0.pth")
    if not any(path.is_file() and not path.is_symlink() for path in rng):
        raise RelaxedBranchError(f"latest checkpoint-{step} is missing RNG state")
    state = _read_json(latest / "trainer_state.json", label="latest checkpoint state")
    if state.get("global_step") != step:
        raise RelaxedBranchError("latest checkpoint global_step drift")
    return {
        "mode": "resume",
        "output_dir": str(output),
        "checkpoint": str(latest),
        "global_step": step,
    }


def _verify_environment() -> dict[str, str]:
    expected = {
        "accelerate": "1.4.0",
        "bitsandbytes": "0.48.2",
        "peft": "0.15.2",
        "torch": "2.10.0+cu128",
        "transformers": "4.57.6",
        "trl": "1.2.0",
    }
    observed: dict[str, str] = {}
    for package, required in expected.items():
        try:
            observed[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RelaxedBranchError(f"missing fomc_trainer package: {package}") from exc
        if observed[package] != required:
            raise RelaxedBranchError(
                f"{package} version drift: expected {required}, observed {observed[package]}"
            )
    if Path(sys.prefix).name != "fomc_trainer":
        raise RelaxedBranchError(
            f"relaxed GRPO must run in fomc_trainer; sys.prefix={sys.prefix}"
        )
    return observed


def training_command(repo_root: Path) -> dict[str, Any]:
    authorization, receipt = _load_authorized(repo_root)
    versions = _verify_environment()
    resume = _resume_state(repo_root)
    config_path = _safe_file(repo_root, GRPO_CONFIG, label="GRPO v3 config")
    argv = [
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
        str(config_path),
    ]
    if resume["mode"] == "resume":
        argv.extend(["--resume_from_checkpoint", resume["checkpoint"]])
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "branch_id": BRANCH_ID,
        "stage": "decision_grpo",
        "gpus": [1],
        "cuda_visible_devices": "1",
        "accelerate_num_processes": 1,
        "effective_train_batch": 8,
        "environment": "fomc_trainer",
        "environment_versions": versions,
        "config": _file_binding(config_path),
        "authorization_sha256": authorization["authorization_sha256"],
        "source_reuse_receipt_sha256": receipt["receipt_sha256"],
        "resume": resume,
        "automatic_resume": "explicit_numerically_latest_complete_checkpoint",
        "argv": argv,
    }


def execute_training(repo_root: Path) -> None:
    _acquire_exec_lock(repo_root)
    command = training_command(repo_root)
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
            "FOMC_CHK4_BRANCH_PROFILE": PROFILE_NAME,
            "FOMC_CHK4_BRANCH_STAGE": "decision_grpo",
            "FOMC_CHK4_BRANCH_CONFIG_PATH": command["config"]["path"],
            "FOMC_CHK4_BRANCH_CONFIG_SHA256": command["config"]["sha256"],
        }
    )
    os.chdir(repo_root)
    os.execvpe(command["argv"][0], command["argv"], environment)


def status(repo_root: Path) -> dict[str, Any]:
    checked = preflight(repo_root)
    authorized = checked["authorization"] is not None
    resume = _resume_state(repo_root) if authorized else None
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready" if authorized else "awaiting_authorization",
        "branch_id": BRANCH_ID,
        "source_sft": "exact_merge_verified",
        "authorization": checked["authorization"],
        "training": resume,
    }


def build_parser() -> argparse.ArgumentParser:
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
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    try:
        if args.command == "preflight":
            result = preflight(repo_root)
        elif args.command == "status":
            result = status(repo_root)
        elif args.command == "authorize":
            result = authorize(repo_root, execute=args.execute)
        elif args.command == "launch" and args.execute:
            execute_training(repo_root)
            raise AssertionError("os.execvpe unexpectedly returned")
        elif args.command == "launch":
            result = training_command(repo_root)
        else:  # pragma: no cover
            raise RelaxedBranchError(f"unsupported command: {args.command}")
    except (
        FileNotFoundError,
        OSError,
        RelaxedBranchError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {"schema_version": SCHEMA_VERSION, "status": "blocked", "error": str(exc)},
                ensure_ascii=False,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
