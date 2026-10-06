"""Promote chk4 GRPO checkpoint-450 and run its one-shot sealed test.

The workflow is deliberately split into irreversible boundaries:

* ``prepare`` binds every pre-test input without parsing test rows;
* ``merge`` creates a CPU-only, no-overwrite merge from an adapter copy;
* ``run`` validates the merge receipt and only then opens the sealed test.

The test run is resumable only at pre-registered batch boundaries.  Existing
batch artifacts are replay-validated and never overwritten.  Test evidence is
never used to select another checkpoint or change generation settings.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from jobs.retrain_v2 import probe_chk4_decision_grpo_stratified as reward_probe
from jobs.retrain_v2 import probe_chk4_decision_sft_generation as generation_probe
from jobs.retrain_v2 import probe_chk4_pre2009_correction_v2 as eos_probe
from jobs.retrain_v2.merge_adapter import merge_from_config
from jobs.retrain_v2.merge_attestation import verify_merge_attestation
from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.trainer.dataset_release import (
    CHK4_PRE2009_GRPO_ROLE,
    verify_chk4_pre2009_augmented_release,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = REPO_ROOT / (
    "output/training/retrain_v2/"
    "chk4_from_pre2009_cp38_direct_grpo_full_no_smoke_v1_20260812"
)
CHECKPOINT_STEP = 450
CHECKPOINT = TRAIN_ROOT / f"adapters/chk4_grpo/checkpoint-{CHECKPOINT_STEP}"
BASE_MODEL = REPO_ROOT / (
    "output/training/retrain_v2/"
    "chk4_from_chk1_cp200_sft_grpo_pre2009_balanced_lr1e5_steps39_v1_20260811/"
    "selected_sft_checkpoints/checkpoint-38/merged/chk4_sft"
)
MERGED_MODEL = TRAIN_ROOT / (
    "selected_grpo_checkpoints/checkpoint-450/merged/chk4_grpo"
)
MERGE_CONFIG = REPO_ROOT / (
    "configs/retrain_v2/chk4_cp450_final_merge_v1_20260814.yaml"
)
TRAINING_CONFIG = REPO_ROOT / (
    "configs/retrain_v2/"
    "chk4_decision_grpo_from_pre2009_cp38_direct_full_no_smoke_v1_20260812.yaml"
)
RELEASE_ROOT = REPO_ROOT / (
    "dataset/processed/retrain_v2/chk4_decision_pre2009_train_balanced_v1_20260811"
)
RELEASE_MANIFEST = RELEASE_ROOT / "release_manifest.json"
SELECTION_ROOT = REPO_ROOT / (
    "output/evaluation/retrain_v2/"
    "chk4_cp38_direct_grpo_checkpoint_probe_v2_batch8_20260814"
)
SELECTION = SELECTION_ROOT / "selection.json"
PROBE_MANIFEST = SELECTION_ROOT / "probe_manifest.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    "output/evaluation/retrain_v2/chk4_cp450_final_test_v1_20260814"
)

BASE_MODEL_SHA256 = "715bdb763043e7fda49e4a189b601d5e070fefbcecd14388bde41dbf7c165dd0"
CHECKPOINT_SHA256 = "ee123017fbb28f6dbfbcd32330b2f2bb8f4828cc7a38a8628e9fc045abd94a2c"
ADAPTER_MODEL_SHA256 = (
    "e79d66c9abf44c603b8b9d860b16704864ac4aa38a086e8e12164a1124f380e8"
)
TRAINER_STATE_SHA256 = (
    "0b18cc89a5776cce5eea3e095dd2fcdc893af6f12fb9705e381348f37ef21329"
)
TRAINING_CONFIG_SHA256 = (
    "222c70d76823681b7ff07971704c38fd153abcac4620a4e0fab4fce00d77f06f"
)
RELEASE_MANIFEST_SHA256 = (
    "5141e00569b94e4af96f030f46176e9cd409199a6f26ac1c63a2ce335fe1e2d6"
)
SELECTION_SHA256 = "bc2a0cfee9866e9eb9597339dd47ff34249f20074addb89c8fe3aa9a26f65c5c"
PROBE_MANIFEST_SHA256 = (
    "534d9ff01b81bd5981eb158a4e307d7e8a03189e41ffec953c5349a74128922f"
)
SYSTEM_PROMPT_SHA256 = (
    "426b64532dfb955a3941db2cc2bcfc9a785c0540ffba9ee94afabc189fe4ce18"
)

MANIFEST_SCHEMA = "chk4-cp450-final-test-manifest-v1"
AUTH_SCHEMA = "chk4-cp450-final-test-authorization-v1"
MERGE_RECEIPT_SCHEMA = "chk4-cp450-final-merge-receipt-v1"
OPEN_RECEIPT_SCHEMA = "chk4-cp450-sealed-test-open-receipt-v1"
RESULT_SCHEMA = "chk4-cp450-final-test-result-v1"
SUMMARY_SCHEMA = "chk4-cp450-final-test-summary-v1"
AUTHORIZATION_BASIS = "explicit_user_instruction_2026-08-14_non_destructive_merge_then_one_shot_sealed_test"

MAX_NEW_TOKENS = 1536
TAIL_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
SAMPLED_PER_PROMPT = 4
MAX_SEQUENCES_PER_BATCH = 8
GREEDY_BATCH_SIZE = 8
SAMPLED_PROMPT_BATCH_SIZE = 2
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260814
DIRECTIONS = ("cut", "hold", "hike")
PREDICTION_LABELS = (*DIRECTIONS, "invalid")


class FinalEvaluationError(RuntimeError):
    """A final-evaluation input or sealed artifact failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalEvaluationError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_payload_sha(value: Mapping[str, Any]) -> str:
    return _sha256_text(canonical_json(value))


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label} missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalEvaluationError(f"cannot read {label}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must contain an object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"{label} missing or unsafe")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                value = json.loads(line)
                _require(isinstance(value, dict), f"{label} row {line_number} invalid")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalEvaluationError(f"cannot read {label}: {exc}") from exc
    return rows


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _write_exclusive_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"unsafe file: {path}")
    rendered = str(path.resolve())
    if relative_to is not None:
        rendered = path.resolve().relative_to(relative_to.resolve()).as_posix()
    return {"path": rendered, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _jsonl_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    record = _file_record(path, relative_to=relative_to)
    with path.open("rb") as handle:
        record["rows"] = sum(bool(line.strip()) for line in handle)
    return record


def _chunks(values: Sequence[str], size: int) -> list[list[str]]:
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


@contextmanager
def _exclusive_lock(output_root: Path) -> Iterable[None]:
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".final_eval.lock"
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FinalEvaluationError("final evaluation lock is held") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _training_config() -> dict[str, Any]:
    value = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "training config is not a mapping")
    return value


def _manifest_test_records(release: Mapping[str, Any]) -> dict[str, Any]:
    files = release.get("files")
    _require(isinstance(files, Mapping), "release files missing")
    records: dict[str, Any] = {}
    for relative in ("decision_grpo/test.jsonl", "manifests/unique/test.jsonl"):
        record = files.get(relative)
        _require(isinstance(record, Mapping), f"release does not bind {relative}")
        path = RELEASE_ROOT / relative
        observed = _jsonl_record(path, relative_to=RELEASE_ROOT)
        _require(observed == dict(record), f"sealed test file drift: {relative}")
        records[relative] = observed
    return records


def _checkpoint_binding() -> dict[str, Any]:
    observed = fingerprint_artifact_path(CHECKPOINT)
    _require(observed.get("sha256") == CHECKPOINT_SHA256, "cp450 directory drift")
    _require(
        sha256_file(CHECKPOINT / "adapter_model.safetensors") == ADAPTER_MODEL_SHA256,
        "cp450 adapter weights drift",
    )
    _require(
        sha256_file(CHECKPOINT / "trainer_state.json") == TRAINER_STATE_SHA256,
        "cp450 trainer state drift",
    )
    state = _read_json(CHECKPOINT / "trainer_state.json", label="cp450 trainer state")
    _require(state.get("global_step") == CHECKPOINT_STEP, "cp450 step drift")
    adapter_config = _read_json(
        CHECKPOINT / "adapter_config.json", label="cp450 adapter config"
    )
    declared = Path(str(adapter_config.get("base_model_name_or_path")))
    if not declared.is_absolute():
        declared = REPO_ROOT / declared
    _require(
        declared.resolve() == BASE_MODEL.resolve(), "cp450 parent declaration drift"
    )
    return {
        "step": CHECKPOINT_STEP,
        "epoch": state.get("epoch"),
        "directory": observed,
        "adapter_model": _file_record(CHECKPOINT / "adapter_model.safetensors"),
        "adapter_config": _file_record(CHECKPOINT / "adapter_config.json"),
        "trainer_state": _file_record(CHECKPOINT / "trainer_state.json"),
    }


def build_manifest(output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    _require(
        sha256_file(TRAINING_CONFIG) == TRAINING_CONFIG_SHA256, "training config drift"
    )
    _require(sha256_file(RELEASE_MANIFEST) == RELEASE_MANIFEST_SHA256, "release drift")
    _require(sha256_file(SELECTION) == SELECTION_SHA256, "selection artifact drift")
    _require(
        sha256_file(PROBE_MANIFEST) == PROBE_MANIFEST_SHA256, "probe manifest drift"
    )
    base = fingerprint_artifact_path(BASE_MODEL)
    _require(base.get("sha256") == BASE_MODEL_SHA256, "cp38 parent drift")
    selection = _read_json(SELECTION, label="checkpoint selection")
    _require(
        selection.get("status") == "provisional_no_candidate_passed_all_gates"
        and selection.get("selected_checkpoint_step") is None
        and selection.get("best_observed_checkpoint_step") == CHECKPOINT_STEP,
        "selection state no longer matches the explicit override",
    )
    release = _read_json(RELEASE_MANIFEST, label="pre-2009 release manifest")
    _require(release.get("quality_status") == "passed", "release quality is not passed")
    _require(release.get("immutable") is True, "release is not immutable")
    _require(
        release.get("test_is_sealed_evaluation_only") is True,
        "release test is not sealed evaluation-only",
    )
    _require(
        release.get("canonical_dag_bindable") is False,
        "unexpected canonical DAG status drift",
    )
    config = _training_config()
    _require(
        _sha256_text(str(config.get("system_prompt"))) == SYSTEM_PROMPT_SHA256,
        "system prompt drift",
    )
    return {
        "schema_version": MANIFEST_SCHEMA,
        "purpose": "one_shot_final_evaluation_of_user_locked_chk4_cp450",
        "authorization_basis": AUTHORIZATION_BASIS,
        "checkpoint_policy": {
            "checkpoint_step": CHECKPOINT_STEP,
            "user_locked": True,
            "automatically_selected": False,
            "selection_status": selection["status"],
            "test_must_not_change_checkpoint_or_hyperparameters": True,
            "canonical_dag_claim_allowed": False,
        },
        "source": {
            "base_model": base,
            "checkpoint": _checkpoint_binding(),
            "training_config": _file_record(TRAINING_CONFIG),
            "selection": _file_record(SELECTION),
            "probe_manifest": _file_record(PROBE_MANIFEST),
        },
        "merge": {
            "config": _file_record(MERGE_CONFIG),
            "destination": str(MERGED_MODEL.resolve()),
            "cpu_only": True,
            "adapter_copy_required": True,
            "overwrite_forbidden": True,
        },
        "release": {
            "root": str(RELEASE_ROOT.resolve()),
            "manifest": _file_record(RELEASE_MANIFEST),
            "test_files": _manifest_test_records(release),
            "expected_rows": 13,
            "pre2009_evaluation_rows": 0,
            "test_open_before_merge_receipt": False,
        },
        "prompt_contract": {
            "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
            "tokenizer_behavior": "training_compatible_without_fix_mistral_regex_override",
        },
        "generation": {
            "primary": "one_greedy_generation_per_meeting",
            "robustness": "four_fixed_sampled_generations_per_meeting",
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "greedy_per_prompt": 1,
            "sampled_per_prompt": SAMPLED_PER_PROMPT,
            "max_sequences_per_batch": MAX_SEQUENCES_PER_BATCH,
            "greedy_prompt_batch_size": GREEDY_BATCH_SIZE,
            "sampled_prompt_batch_size": SAMPLED_PROMPT_BATCH_SIZE,
            "batch_seed_rule": "sha256('chk4-cp450-final-test-v1:<mode>:<batch-index>') mod 2^31",
            "physical_gpu": 1,
            "visible_device_count": 1,
            "load_in_4bit": True,
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
        },
        "metrics": {
            "primary_population": "13 greedy meeting-level predictions",
            "sampled_outputs_are_independent_meetings": False,
            "direction": [
                "balanced_accuracy",
                "macro_f1",
                "confusion_matrix",
                "per_direction_recall",
            ],
            "action": [
                "exact_direction_magnitude_accuracy",
                "signed_bp_mae_on_parseable",
            ],
            "delivery": [
                "delivery_valid",
                "strict_json",
                "eos",
                "cap",
                "periodic_tail",
            ],
            "uncertainty": {
                "method": "meeting_level_stratified_bootstrap",
                "draws": BOOTSTRAP_DRAWS,
                "seed": BOOTSTRAP_SEED,
            },
            "baselines": [
                "train_majority",
                "lag1_action",
                "date_only_logistic",
                "tfidf_logistic",
            ],
            "sample_frequency_brier": "four_class_direction_plus_invalid_secondary_only",
        },
        "gates": {
            "greedy_delivery_valid_rate_min": 0.75,
            "greedy_cap_rate_max": 0.25,
            "all_outputs_periodic_tail_count_max": 0,
            "greedy_balanced_accuracy_gt": 1.0 / 3.0,
            "greedy_macro_f1_must_exceed": "train_majority",
            "each_test_direction_recall_gt": 0.0,
            "passing_label": "promotion_candidate_experimental_noncanonical",
        },
        "runtime_source": _file_record(Path(__file__).resolve()),
        "output_root": str(output_root),
    }


def _authorization(manifest_path: Path, manifest_sha256: str) -> dict[str, Any]:
    payload = {
        "schema_version": AUTH_SCHEMA,
        "status": "authorized",
        "created_at_utc": _utc_now(),
        "authorization_basis": AUTHORIZATION_BASIS,
        "scope": {
            "allowed": [
                "cpu_non_destructive_cp450_merge",
                "gpu1_one_shot_sealed_test",
                "exact_manifest_resume",
            ],
            "forbidden": [
                "gpu0",
                "checkpoint_reselection",
                "test_driven_retuning",
                "overwrite",
                "canonical_dag_claim",
            ],
        },
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest_sha256,
        },
    }
    return {
        **payload,
        "integrity": {
            "algorithm": "sha256(canonical-json-without-integrity)",
            "payload_sha256": _canonical_payload_sha(payload),
        },
    }


def prepare(output_root: Path) -> dict[str, Any]:
    root = output_root.resolve()
    _require(
        not root.exists() and not root.is_symlink(),
        f"output root already exists: {root}",
    )
    _require(
        not MERGED_MODEL.exists() and not MERGED_MODEL.is_symlink(),
        "merged destination already exists",
    )
    root.mkdir(parents=True, exist_ok=False)
    try:
        manifest = build_manifest(root)
        manifest_path = root / "evaluation_manifest.json"
        _write_exclusive_json(manifest_path, manifest)
        manifest_sha = sha256_file(manifest_path)
        auth = _authorization(manifest_path, manifest_sha)
        _write_exclusive_json(root / "authorization.json", auth)
        return {
            "status": "prepared_and_authorized",
            "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
            "authorization": _file_record(root / "authorization.json"),
        }
    except Exception:
        # Only remove an empty/partially prepared root created by this invocation.
        for child in sorted(root.glob("*")):
            if child.is_file() and not child.is_symlink():
                child.unlink()
        try:
            root.rmdir()
        except OSError:
            pass
        raise


def validate_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    _require(sha256_file(path) == expected_sha256, "evaluation manifest SHA drift")
    value = _read_json(path, label="evaluation manifest")
    _require(value.get("schema_version") == MANIFEST_SCHEMA, "manifest schema drift")
    rebuilt = build_manifest(path.parent)
    _require(
        canonical_json(value) == canonical_json(rebuilt),
        "evaluation manifest replay drift",
    )
    auth_path = path.parent / "authorization.json"
    auth = _read_json(auth_path, label="evaluation authorization")
    _require(
        auth.get("schema_version") == AUTH_SCHEMA
        and auth.get("status") == "authorized",
        "authorization state drift",
    )
    integrity = auth.get("integrity")
    _require(isinstance(integrity, Mapping), "authorization integrity missing")
    unsigned = dict(auth)
    unsigned.pop("integrity", None)
    _require(
        integrity.get("payload_sha256") == _canonical_payload_sha(unsigned),
        "authorization payload drift",
    )
    _require(
        auth.get("manifest")
        == {"path": str(path.resolve()), "sha256": expected_sha256},
        "authorization manifest binding drift",
    )
    return value


def _merge_binding(manifest: Mapping[str, Any], manifest_sha: str) -> dict[str, Any]:
    return {
        "purpose": "chk4_cp450_final_evaluation",
        "checkpoint_step": CHECKPOINT_STEP,
        "evaluation_manifest_sha256": manifest_sha,
        "authorization_basis": AUTHORIZATION_BASIS,
        "base_model_sha256": manifest["source"]["base_model"]["sha256"],
        "source_adapter_sha256": manifest["source"]["checkpoint"]["directory"][
            "sha256"
        ],
        "adapter_weights_sha256": manifest["source"]["checkpoint"]["adapter_model"][
            "sha256"
        ],
        "training_config_sha256": manifest["source"]["training_config"]["sha256"],
        "release_manifest_sha256": manifest["release"]["manifest"]["sha256"],
        "selection_sha256": manifest["source"]["selection"]["sha256"],
        "noncanonical": True,
    }


def merge(manifest_path: Path, manifest_sha256: str) -> dict[str, Any]:
    _require(
        os.environ.get("CUDA_VISIBLE_DEVICES", "") in {"", "-1"},
        "merge must hide all GPUs",
    )
    manifest = validate_manifest(manifest_path, manifest_sha256)
    receipt_path = manifest_path.parent / "merge_receipt.json"
    _require(not receipt_path.exists(), "merge receipt already exists")
    _require(
        not MERGED_MODEL.exists() and not MERGED_MODEL.is_symlink(),
        "merged destination already exists",
    )
    binding = _merge_binding(manifest, manifest_sha256)
    source_before = fingerprint_artifact_path(CHECKPOINT)
    result = merge_from_config(
        MERGE_CONFIG,
        merge_attestation_binding=binding,
        repo_root=REPO_ROOT,
    )
    source_after = fingerprint_artifact_path(CHECKPOINT)
    _require(source_before == source_after, "cp450 changed across merge")
    attestation = verify_merge_attestation(MERGED_MODEL, expected_binding=binding)
    payload = {
        "schema_version": MERGE_RECEIPT_SCHEMA,
        "status": "merged_and_attested",
        "created_at_utc": _utc_now(),
        "evaluation_manifest_sha256": manifest_sha256,
        "source_checkpoint_before": source_before,
        "source_checkpoint_after": source_after,
        "source_checkpoint_unchanged": True,
        "merged_artifact": result["merged_artifact"],
        "merge_attestation": {
            "path": str((MERGED_MODEL / "merge_attestation.json").resolve()),
            "sha256": sha256_file(MERGED_MODEL / "merge_attestation.json"),
            "canonical_payload_sha256": attestation["canonical_payload_sha256"],
        },
        "semantic_evidence": attestation["semantic_evidence"],
    }
    unsigned = dict(payload)
    payload["integrity"] = {
        "algorithm": "sha256(canonical-json-without-integrity)",
        "payload_sha256": _canonical_payload_sha(unsigned),
    }
    _write_exclusive_json(receipt_path, payload)
    return payload


def _validate_merge_receipt(
    manifest: Mapping[str, Any], manifest_sha: str
) -> dict[str, Any]:
    path = Path(str(manifest["output_root"])) / "merge_receipt.json"
    receipt = _read_json(path, label="merge receipt")
    _require(
        receipt.get("schema_version") == MERGE_RECEIPT_SCHEMA
        and receipt.get("status") == "merged_and_attested",
        "merge receipt state drift",
    )
    integrity = receipt.get("integrity")
    _require(isinstance(integrity, Mapping), "merge receipt integrity missing")
    unsigned = dict(receipt)
    unsigned.pop("integrity", None)
    _require(
        integrity.get("payload_sha256") == _canonical_payload_sha(unsigned),
        "merge receipt payload drift",
    )
    _require(
        receipt.get("evaluation_manifest_sha256") == manifest_sha,
        "merge receipt manifest drift",
    )
    _require(
        fingerprint_artifact_path(CHECKPOINT) == receipt.get("source_checkpoint_after"),
        "cp450 changed after merge",
    )
    _require(
        fingerprint_artifact_path(MERGED_MODEL) == receipt.get("merged_artifact"),
        "merged model drift",
    )
    verify_merge_attestation(
        MERGED_MODEL, expected_binding=_merge_binding(manifest, manifest_sha)
    )
    return receipt


def _load_test_after_authorization(
    manifest: Mapping[str, Any], tokenizer: Any
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    physical_path = RELEASE_ROOT / "decision_grpo/test.jsonl"
    unique_path = RELEASE_ROOT / "manifests/unique/test.jsonl"
    expected = manifest["release"]["test_files"]
    _require(
        _jsonl_record(physical_path, relative_to=RELEASE_ROOT)
        == expected["decision_grpo/test.jsonl"],
        "physical sealed test drift",
    )
    _require(
        _jsonl_record(unique_path, relative_to=RELEASE_ROOT)
        == expected["manifests/unique/test.jsonl"],
        "unique sealed test drift",
    )
    physical = _read_jsonl(physical_path, label="sealed physical test")
    unique = _read_jsonl(unique_path, label="sealed unique test")
    _require(
        len(physical) == len(unique) == int(manifest["release"]["expected_rows"]),
        "sealed test row-count drift",
    )
    physical_by_id: dict[str, dict[str, Any]] = {}
    prompts: dict[str, str] = {}
    for row in physical:
        _require(
            set(row) == {"sample_id", "prompt", "direction", "magnitude_bp"},
            "physical test schema drift",
        )
        sample_id = str(row["sample_id"])
        _require(sample_id not in physical_by_id, "duplicate physical test sample")
        physical_by_id[sample_id] = row
        prompts[sample_id] = str(row["prompt"])
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    config = _training_config()
    system_prompt = str(config["system_prompt"])
    for row in unique:
        sample_id = str(row.get("sample_id"))
        _require(
            sample_id in physical_by_id and sample_id not in seen,
            "test ID closure drift",
        )
        seen.add(sample_id)
        source = physical_by_id[sample_id]
        _require(
            row.get("split") == "test" and row.get("repeat_factor") == 1,
            "unique test role drift",
        )
        _require(
            row.get("direction") == source.get("direction")
            and row.get("magnitude_bp") == source.get("magnitude_bp"),
            "test target drift",
        )
        _require(
            _sha256_text(prompts[sample_id]) == row.get("prompt_sha256"),
            "test prompt hash drift",
        )
        prompt_ids = generation_probe._prompt_ids(
            tokenizer,
            generation_probe._messages(system_prompt, prompts[sample_id]),
        )
        _require(len(prompt_ids) <= 2560, "test prompt exceeds bound")
        samples.append(
            {
                "sample_id": sample_id,
                "meeting_date": row.get("meeting_date"),
                "direction": row.get("direction"),
                "magnitude_bp": row.get("magnitude_bp"),
                "prompt_sha256": row.get("prompt_sha256"),
                "prompt_token_count": len(prompt_ids),
            }
        )
    _require(seen == set(physical_by_id), "test sample coverage drift")
    return samples, prompts


def _batch_seed(mode: str, batch_index: int) -> int:
    text = f"chk4-cp450-final-test-v1:{mode}:{batch_index}"
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**31)


def _batch_contract(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ids = [str(sample["sample_id"]) for sample in samples]
    result: list[dict[str, Any]] = []
    for mode, size, returns in (
        ("greedy", GREEDY_BATCH_SIZE, 1),
        ("sampled", SAMPLED_PROMPT_BATCH_SIZE, SAMPLED_PER_PROMPT),
    ):
        for batch_index, batch_ids in enumerate(_chunks(ids, size)):
            _require(
                len(batch_ids) * returns <= MAX_SEQUENCES_PER_BATCH,
                "batch exceeds sequence limit",
            )
            result.append(
                {
                    "mode": mode,
                    "batch_index": batch_index,
                    "sample_ids": batch_ids,
                    "num_return_sequences": returns,
                    "seed": _batch_seed(mode, batch_index),
                }
            )
    return result


def _open_receipt(
    manifest_sha: str, samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    payload = {
        "schema_version": OPEN_RECEIPT_SCHEMA,
        "status": "sealed_test_opened_after_merge",
        "created_at_utc": _utc_now(),
        "evaluation_manifest_sha256": manifest_sha,
        "sample_count": len(samples),
        "samples": list(samples),
        "batches": _batch_contract(samples),
    }
    return {
        **payload,
        "integrity": {
            "algorithm": "sha256(canonical-json-without-integrity)",
            "payload_sha256": _canonical_payload_sha(payload),
        },
    }


def _validate_open_receipt(
    path: Path, manifest_sha: str, samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    value = _read_json(path, label="sealed-test open receipt")
    integrity = value.get("integrity")
    _require(isinstance(integrity, Mapping), "open receipt integrity missing")
    unsigned = dict(value)
    unsigned.pop("integrity", None)
    _require(
        integrity.get("payload_sha256") == _canonical_payload_sha(unsigned),
        "open receipt payload drift",
    )
    expected = _open_receipt(manifest_sha, samples)
    for key in ("created_at_utc", "integrity"):
        expected.pop(key, None)
        value.pop(key, None)
    _require(
        canonical_json(value) == canonical_json(expected), "open receipt replay drift"
    )
    return _read_json(path, label="sealed-test open receipt")


def _load_merged_model() -> Any:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(MERGED_MODEL),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        quantization_config=quantization,
        device_map={"": 0},
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    _require(
        not any(tensor.is_meta for tensor in model.parameters()),
        "merged model has meta parameters",
    )
    model.eval()
    model.config.use_cache = True
    return model


def _result_row(
    *,
    manifest_sha: str,
    merged_sha: str,
    sample: Mapping[str, Any],
    mode: str,
    generation_index: int,
    batch_index: int,
    seed: int,
    raw_ids: Sequence[int],
    eos_ids: set[int],
    tokenizer: Any,
) -> dict[str, Any]:
    normalized_ids, normalization = eos_probe.normalize_at_first_bound_eos(
        raw_ids, eos_ids
    )
    completion = generation_probe.decode_completion_preserving_boundary(
        tokenizer, normalized_ids, eos_ids
    )
    delivery = generation_probe.analyze_completion(
        text=completion,
        generated_token_ids=normalized_ids,
        eos_token_ids=eos_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        tail_tokens=TAIL_TOKENS,
    )
    replay = reward_probe.replay_decision_dense_v3(
        completion,
        {"direction": sample["direction"], "magnitude_bp": sample["magnitude_bp"]},
        hit_eos=bool(delivery["hit_eos"]),
        cap_reached=bool(delivery["cap_reached"]),
    )
    return {
        "schema_version": RESULT_SCHEMA,
        "model_label": "chk4_cp450_merged",
        "checkpoint_step": CHECKPOINT_STEP,
        "merged_model_sha256": merged_sha,
        "sample_id": sample["sample_id"],
        "split": "test",
        "target_direction": sample["direction"],
        "target_magnitude_bp": sample["magnitude_bp"],
        "prompt_sha256": sample["prompt_sha256"],
        "prompt_token_count": sample["prompt_token_count"],
        "generation_mode": mode,
        "generation_index": generation_index,
        "batch_index": batch_index,
        "batch_seed": seed,
        "generation_parameters": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": mode == "sampled",
            "temperature": TEMPERATURE if mode == "sampled" else 0.0,
            "top_p": TOP_P if mode == "sampled" else 1.0,
        },
        "generated_token_ids_raw_padded": list(raw_ids),
        "generated_token_ids_first_eos_inclusive": normalized_ids,
        "batched_padding_normalization": normalization,
        "completion": completion,
        "completion_sha256": _sha256_text(completion),
        **delivery,
        "decision_dense_v3_reward": replay["reward"],
        "decision_prediction": replay["prediction"],
        "decision_direction_correct": replay["direction_correct"],
        "decision_exact": replay["exact"],
        "decision_rejection_reason": replay["rejection_reason"],
        "decision_forced_zero_reason": replay["forced_zero_reason"],
        "response_format": replay["response_format"],
        "strict_json": replay["strict_json"],
        "fenced_json": replay["fenced_json"],
        "evaluation_manifest_sha256": manifest_sha,
    }


def _validate_batch_rows(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    manifest_sha: str,
    merged_sha: str,
) -> None:
    expected_count = len(contract["sample_ids"]) * int(contract["num_return_sequences"])
    _require(len(rows) == expected_count, "batch row count drift")
    expected_pairs = []
    for sample_id in contract["sample_ids"]:
        for index in range(int(contract["num_return_sequences"])):
            expected_pairs.append(
                (sample_id, 0 if contract["mode"] == "greedy" else index + 1)
            )
    observed_pairs = [
        (row.get("sample_id"), row.get("generation_index")) for row in rows
    ]
    _require(observed_pairs == expected_pairs, "batch sample/generation order drift")
    for row in rows:
        _require(row.get("schema_version") == RESULT_SCHEMA, "batch schema drift")
        _require(
            row.get("evaluation_manifest_sha256") == manifest_sha,
            "batch manifest drift",
        )
        _require(row.get("merged_model_sha256") == merged_sha, "batch model drift")
        _require(row.get("generation_mode") == contract["mode"], "batch mode drift")
        _require(
            row.get("batch_index") == contract["batch_index"]
            and row.get("batch_seed") == contract["seed"],
            "batch index/seed drift",
        )


def _signed_bp(direction: str, magnitude: int) -> int:
    return -magnitude if direction == "cut" else magnitude if direction == "hike" else 0


def _prediction_label(row: Mapping[str, Any]) -> str:
    prediction = row.get("decision_prediction")
    if isinstance(prediction, Mapping) and prediction.get("direction") in DIRECTIONS:
        return str(prediction["direction"])
    return "invalid"


def metric_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(rows), "cannot score empty rows")
    total = len(rows)
    matrix = {
        actual: {guess: 0 for guess in PREDICTION_LABELS} for actual in DIRECTIONS
    }
    exact = 0
    signed_errors: list[int] = []
    rewards: list[float] = []
    for row in rows:
        actual = str(row["target_direction"])
        guess = _prediction_label(row)
        _require(actual in DIRECTIONS, "invalid target direction")
        matrix[actual][guess] += 1
        exact += int(bool(row.get("decision_exact")))
        rewards.append(float(row["decision_dense_v3_reward"]))
        prediction = row.get("decision_prediction")
        if isinstance(prediction, Mapping) and guess in DIRECTIONS:
            signed_errors.append(
                abs(
                    _signed_bp(actual, int(row["target_magnitude_bp"]))
                    - _signed_bp(guess, int(prediction["magnitude_bp"]))
                )
            )
    per_direction: dict[str, Any] = {}
    f1s: list[float] = []
    recalls: list[float] = []
    for label in DIRECTIONS:
        support = sum(matrix[label].values())
        tp = matrix[label][label]
        fp = sum(matrix[actual][label] for actual in DIRECTIONS if actual != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_direction[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        recalls.append(recall)
        f1s.append(f1)

    def rate(field: str) -> float:
        return sum(bool(row.get(field)) for row in rows) / total

    return {
        "cases": total,
        "reward_mean": statistics.fmean(rewards),
        "reward_std": statistics.pstdev(rewards),
        "direction_accuracy": sum(matrix[label][label] for label in DIRECTIONS) / total,
        "balanced_accuracy": statistics.fmean(recalls),
        "macro_f1": statistics.fmean(f1s),
        "per_direction": per_direction,
        "confusion_matrix": matrix,
        "exact_direction_magnitude_accuracy": exact / total,
        "signed_bp_mae_parseable": statistics.fmean(signed_errors)
        if signed_errors
        else None,
        "parseable_count": len(signed_errors),
        "delivery_valid_rate": rate("delivery_valid"),
        "strict_json_rate": rate("strict_json"),
        "eos_rate": rate("hit_eos"),
        "cap_rate": rate("cap_reached"),
        "periodic_tail_count": sum(
            bool(row.get("strict_periodic_tail")) for row in rows
        ),
    }


def _prediction_row(
    sample: Mapping[str, Any], direction: str, magnitude: int
) -> dict[str, Any]:
    return {
        "target_direction": sample["direction"],
        "target_magnitude_bp": sample["magnitude_bp"],
        "decision_prediction": {"direction": direction, "magnitude_bp": magnitude},
        "decision_exact": direction == sample["direction"]
        and magnitude == sample["magnitude_bp"],
        "decision_dense_v3_reward": 0.0,
        "delivery_valid": True,
        "strict_json": True,
        "hit_eos": True,
        "cap_reached": False,
        "strict_periodic_tail": False,
    }


def _load_unique_split(split: str) -> list[dict[str, Any]]:
    return _read_jsonl(
        RELEASE_ROOT / f"manifests/unique/{split}.jsonl", label=f"unique {split}"
    )


def _load_physical_prompts(split: str) -> dict[str, str]:
    rows = _read_jsonl(
        RELEASE_ROOT / f"decision_grpo/{split}.jsonl", label=f"physical {split}"
    )
    result: dict[str, str] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        prompt = str(row["prompt"])
        if sample_id in result:
            _require(result[sample_id] == prompt, f"{split} prompt variants")
        result[sample_id] = prompt
    return result


def baseline_metrics(
    samples: Sequence[Mapping[str, Any]], test_prompts: Mapping[str, str]
) -> dict[str, Any]:
    train = _load_unique_split("train")
    validation = _load_unique_split("validation")
    train_prompts = _load_physical_prompts("train")
    action_counts = Counter(
        (str(row["direction"]), int(row["magnitude_bp"])) for row in train
    )
    majority_action = sorted(
        action_counts.items(), key=lambda item: (-item[1], item[0])
    )[0][0]
    majority_rows = [_prediction_row(sample, *majority_action) for sample in samples]

    all_rows = sorted(
        [*train, *validation, *samples], key=lambda row: str(row["meeting_date"])
    )
    lag_by_id: dict[str, tuple[str, int]] = {}
    previous: tuple[str, int] | None = None
    for row in all_rows:
        sample_id = str(row["sample_id"])
        if previous is not None:
            lag_by_id[sample_id] = previous
        previous = (str(row["direction"]), int(row["magnitude_bp"]))
    lag_rows = [
        _prediction_row(sample, *lag_by_id[str(sample["sample_id"])])
        for sample in samples
    ]

    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise FinalEvaluationError(f"baseline dependencies unavailable: {exc}") from exc

    action_labels = [f"{row['direction']}:{int(row['magnitude_bp'])}" for row in train]
    train_dates = [str(row["meeting_date"]) for row in train]
    test_dates = [str(sample["meeting_date"]) for sample in samples]

    def date_features(values: Sequence[str]) -> Any:
        rows = []
        for value in values:
            year, month, day = (int(part) for part in value.split("-")[:3])
            ordinal = __import__("datetime").date(year, month, day).toordinal()
            rows.append([ordinal, year, month, day])
        return np.asarray(rows, dtype=float)

    date_model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=5000, random_state=BOOTSTRAP_SEED)
    )
    date_model.fit(date_features(train_dates), action_labels)
    date_predictions = list(date_model.predict(date_features(test_dates)))

    tfidf_model = make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=20_000, sublinear_tf=True
        ),
        LogisticRegression(max_iter=5000, random_state=BOOTSTRAP_SEED),
    )
    train_texts = [train_prompts[str(row["sample_id"])] for row in train]
    test_texts = [test_prompts[str(sample["sample_id"])] for sample in samples]
    tfidf_model.fit(train_texts, action_labels)
    tfidf_predictions = list(tfidf_model.predict(test_texts))

    def from_labels(labels: Sequence[str]) -> list[dict[str, Any]]:
        rows = []
        for sample, label in zip(samples, labels, strict=True):
            direction, magnitude = label.split(":", 1)
            rows.append(_prediction_row(sample, direction, int(magnitude)))
        return rows

    return {
        "train_majority": {
            "action": {
                "direction": majority_action[0],
                "magnitude_bp": majority_action[1],
            },
            "metrics": metric_block(majority_rows),
        },
        "lag1_action": {"metrics": metric_block(lag_rows)},
        "date_only_logistic": {
            "features": ["ordinal_day", "year", "month", "day"],
            "train_split_only": True,
            "metrics": metric_block(from_labels(date_predictions)),
        },
        "tfidf_logistic": {
            "features": "prompt unigram+bigram TF-IDF",
            "train_split_only": True,
            "metrics": metric_block(from_labels(tfidf_predictions)),
        },
    }


def _stratified_bootstrap(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups = {
        direction: [row for row in rows if row["target_direction"] == direction]
        for direction in DIRECTIONS
    }
    _require(all(groups.values()), "bootstrap requires every direction")
    rng = random.Random(BOOTSTRAP_SEED)
    values = {
        key: []
        for key in (
            "direction_accuracy",
            "balanced_accuracy",
            "macro_f1",
            "exact_direction_magnitude_accuracy",
        )
    }
    for _ in range(BOOTSTRAP_DRAWS):
        draw: list[Mapping[str, Any]] = []
        for direction in DIRECTIONS:
            group = groups[direction]
            draw.extend(group[rng.randrange(len(group))] for _ in range(len(group)))
        metrics = metric_block(draw)
        for key in values:
            values[key].append(float(metrics[key]))

    def percentile(items: Sequence[float], probability: float) -> float:
        ordered = sorted(items)
        position = (len(ordered) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "method": "meeting_level_stratified_by_target_direction_percentile",
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "interval": 0.95,
        "metrics": {
            key: {"lower": percentile(items, 0.025), "upper": percentile(items, 0.975)}
            for key, items in values.items()
        },
    }


def _sample_frequency_metrics(
    rows: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    sampled = [row for row in rows if row["generation_mode"] == "sampled"]
    by_id: dict[str, list[Mapping[str, Any]]] = {}
    for row in sampled:
        by_id.setdefault(str(row["sample_id"]), []).append(row)
    brier_values = []
    unanimous = 0
    for sample in samples:
        group = by_id[str(sample["sample_id"])]
        _require(len(group) == SAMPLED_PER_PROMPT, "sample-frequency group drift")
        counts = Counter(_prediction_label(row) for row in group)
        probabilities = {
            label: counts[label] / SAMPLED_PER_PROMPT for label in PREDICTION_LABELS
        }
        brier_values.append(
            sum(
                (probabilities[label] - float(label == sample["direction"])) ** 2
                for label in PREDICTION_LABELS
            )
        )
        unanimous += int(len(counts) == 1)
    return {
        "definition": "mean multiclass squared error over cut/hold/hike/invalid empirical frequencies",
        "draws_per_meeting": SAMPLED_PER_PROMPT,
        "brier_four_class": statistics.fmean(brier_values),
        "unanimous_prediction_rate": unanimous / len(samples),
    }


def summarize(
    rows: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    prompts: Mapping[str, str],
    manifest_sha: str,
    merged_sha: str,
) -> dict[str, Any]:
    greedy = [row for row in rows if row["generation_mode"] == "greedy"]
    sampled = [row for row in rows if row["generation_mode"] == "sampled"]
    _require(
        len(greedy) == len(samples)
        and len(sampled) == len(samples) * SAMPLED_PER_PROMPT,
        "final result coverage drift",
    )
    primary = metric_block(greedy)
    sampled_by_draw = {
        str(index): metric_block(
            [row for row in sampled if row["generation_index"] == index]
        )
        for index in range(1, SAMPLED_PER_PROMPT + 1)
    }
    baselines = baseline_metrics(samples, prompts)
    majority = baselines["train_majority"]["metrics"]
    all_periodic = sum(bool(row.get("strict_periodic_tail")) for row in rows)
    gate_checks = {
        "greedy_delivery_valid_rate": primary["delivery_valid_rate"] >= 0.75,
        "greedy_cap_rate": primary["cap_rate"] <= 0.25,
        "all_outputs_periodic_tail_count": all_periodic == 0,
        "greedy_balanced_accuracy": primary["balanced_accuracy"] > 1.0 / 3.0,
        "greedy_macro_f1_vs_train_majority": primary["macro_f1"] > majority["macro_f1"],
        "each_test_direction_recall": all(
            primary["per_direction"][direction]["recall"] > 0
            for direction in DIRECTIONS
        ),
    }
    passed = all(gate_checks.values())
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "verdict": "promotion_candidate_experimental_noncanonical"
        if passed
        else "quality_not_demonstrated",
        "checkpoint_step": CHECKPOINT_STEP,
        "evaluation_manifest_sha256": manifest_sha,
        "merged_model_sha256": merged_sha,
        "test_meetings": len(samples),
        "generated_completions": len(rows),
        "primary_greedy": primary,
        "sampled_aggregate": metric_block(sampled),
        "sampled_by_draw": sampled_by_draw,
        "sample_frequency": _sample_frequency_metrics(rows, samples),
        "bootstrap": _stratified_bootstrap(greedy),
        "baselines": baselines,
        "gate_checks": gate_checks,
        "limitations": [
            "test has only 13 meetings and one hike case",
            "pre-2009 augmentation is train-only; this test has zero new pre-2009 rows",
            "sample-frequency Brier is an empirical four-draw diagnostic, not a calibrated model probability",
            "the release is not canonical-DAG-bindable",
        ],
        "created_at_utc": _utc_now(),
    }


def run(manifest_path: Path, manifest_sha256: str) -> dict[str, Any]:
    _require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "1",
        "sealed test requires physical GPU1 only",
    )
    manifest = validate_manifest(manifest_path, manifest_sha256)
    _validate_merge_receipt(manifest, manifest_sha256)
    output_root = manifest_path.parent
    run_root = output_root / "run"
    run_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "summary.json"
    if summary_path.is_file():
        return _read_json(summary_path, label="final summary")

    config = _training_config()
    runtime_binding = verify_chk4_pre2009_augmented_release(
        dataset_dir=RELEASE_ROOT / "decision_grpo",
        manifest_path=RELEASE_MANIFEST,
        expected_manifest_sha256=RELEASE_MANIFEST_SHA256,
        dataset_role=CHK4_PRE2009_GRPO_ROLE,
        system_prompt=str(config["system_prompt"]),
        model_path=MERGED_MODEL,
    )
    _require(
        runtime_binding.get("test_verified_but_not_loaded") is True,
        "release verifier leaked test",
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(MERGED_MODEL), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # This is the intentional sealed boundary: authorization and merge have
    # already been validated before these rows are parsed.
    samples, prompts = _load_test_after_authorization(manifest, tokenizer)
    open_path = run_root / "test_open_receipt.json"
    if open_path.exists():
        open_receipt = _validate_open_receipt(open_path, manifest_sha256, samples)
    else:
        open_receipt = _open_receipt(manifest_sha256, samples)
        _write_exclusive_json(open_path, open_receipt)

    merged_sha = str(manifest_path.parent.joinpath("merge_receipt.json"))
    merged_record = _read_json(Path(merged_sha), label="merge receipt")[
        "merged_artifact"
    ]
    merged_fingerprint_sha = str(merged_record["sha256"])
    launch_path = run_root / "launch.json"
    launch = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "running",
        "created_at_utc": _utc_now(),
        "evaluation_manifest_sha256": manifest_sha256,
        "merged_model_sha256": merged_fingerprint_sha,
        "runtime": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "cuda_visible_devices": "1",
        },
        "test_open_receipt_sha256": sha256_file(open_path),
    }
    if launch_path.exists():
        observed_launch = _read_json(launch_path, label="launch receipt")
        for key in (
            "schema_version",
            "evaluation_manifest_sha256",
            "merged_model_sha256",
            "test_open_receipt_sha256",
        ):
            _require(
                observed_launch.get(key) == launch.get(key), "launch receipt drift"
            )
    else:
        _write_exclusive_json(launch_path, launch)

    contracts = list(open_receipt["batches"])
    model: Any = None
    try:
        import torch
        import transformers

        _require(
            torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "runtime must expose exactly one GPU",
        )
        model = _load_merged_model()
        eos_value = getattr(model.generation_config, "eos_token_id", None)
        if eos_value is None:
            eos_value = tokenizer.eos_token_id
        eos_ids = set(generation_probe._normalize_eos_ids(eos_value))
        _require(bool(eos_ids), "model/tokenizer has no EOS")
        sample_by_id = {str(sample["sample_id"]): sample for sample in samples}
        batch_root = run_root / "batches"
        batch_root.mkdir(parents=True, exist_ok=True)
        torch.cuda.reset_peak_memory_stats()
        for contract in contracts:
            mode = str(contract["mode"])
            batch_index = int(contract["batch_index"])
            batch_path = batch_root / f"{mode}_{batch_index:02d}.jsonl"
            if batch_path.exists():
                existing = _read_jsonl(
                    batch_path, label=f"completed batch {mode}/{batch_index}"
                )
                _validate_batch_rows(
                    existing, contract, manifest_sha256, merged_fingerprint_sha
                )
                continue
            transformers.set_seed(int(contract["seed"]))
            batch_ids = [str(value) for value in contract["sample_ids"]]
            rendered = [
                tokenizer.apply_chat_template(
                    generation_probe._messages(
                        str(config["system_prompt"]), prompts[sample_id]
                    ),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for sample_id in batch_ids
            ]
            encoded = tokenizer(
                rendered, return_tensors="pt", padding=True, add_special_tokens=False
            )
            input_ids = encoded["input_ids"].to("cuda:0")
            attention_mask = encoded["attention_mask"].to("cuda:0")
            for prompt_index, sample_id in enumerate(batch_ids):
                _require(
                    int(attention_mask[prompt_index].sum().item())
                    == int(sample_by_id[sample_id]["prompt_token_count"]),
                    f"prompt token count drift: {sample_id}",
                )
            sampled = mode == "sampled"
            kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": sampled,
                "num_return_sequences": int(contract["num_return_sequences"]),
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": sorted(eos_ids),
                "use_cache": True,
            }
            if sampled:
                kwargs.update(temperature=TEMPERATURE, top_p=TOP_P)
            with torch.inference_mode():
                sequences = model.generate(**kwargs)
            expected_count = len(batch_ids) * int(contract["num_return_sequences"])
            _require(len(sequences) == expected_count, "generation count drift")
            prompt_width = int(input_ids.shape[1])
            batch_rows: list[dict[str, Any]] = []
            for prompt_index, sample_id in enumerate(batch_ids):
                for return_index in range(int(contract["num_return_sequences"])):
                    sequence_index = (
                        prompt_index * int(contract["num_return_sequences"])
                        + return_index
                    )
                    raw_ids = sequences[sequence_index, prompt_width:].tolist()
                    batch_rows.append(
                        _result_row(
                            manifest_sha=manifest_sha256,
                            merged_sha=merged_fingerprint_sha,
                            sample=sample_by_id[sample_id],
                            mode=mode,
                            generation_index=0
                            if mode == "greedy"
                            else return_index + 1,
                            batch_index=batch_index,
                            seed=int(contract["seed"]),
                            raw_ids=raw_ids,
                            eos_ids=eos_ids,
                            tokenizer=tokenizer,
                        )
                    )
            _validate_batch_rows(
                batch_rows, contract, manifest_sha256, merged_fingerprint_sha
            )
            _write_exclusive_jsonl(batch_path, batch_rows)
            print(
                canonical_json(
                    {
                        "status": "batch_complete",
                        "mode": mode,
                        "batch_index": batch_index,
                        "rows": len(batch_rows),
                    }
                ),
                flush=True,
            )
            del input_ids, attention_mask, sequences

        rows: list[dict[str, Any]] = []
        for contract in contracts:
            path = (
                run_root
                / "batches"
                / f"{contract['mode']}_{int(contract['batch_index']):02d}.jsonl"
            )
            batch_rows = _read_jsonl(path, label="final batch")
            _validate_batch_rows(
                batch_rows, contract, manifest_sha256, merged_fingerprint_sha
            )
            rows.extend(batch_rows)
        result_path = run_root / "results.jsonl"
        if result_path.exists():
            existing = _read_jsonl(result_path, label="final results")
            _require(
                canonical_json(existing) == canonical_json(rows),
                "final result assembly drift",
            )
        else:
            _write_exclusive_jsonl(result_path, rows)
        summary = summarize(
            rows, samples, prompts, manifest_sha256, merged_fingerprint_sha
        )
        summary["runtime"] = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        }
        summary["results"] = _file_record(result_path)
        _write_exclusive_json(summary_path, summary)
        return summary
    finally:
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def status(output_root: Path) -> dict[str, Any]:
    root = output_root.resolve()
    result: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA,
        "output_root": str(root),
        "manifest": "present"
        if (root / "evaluation_manifest.json").is_file()
        else "missing",
        "authorization": "present"
        if (root / "authorization.json").is_file()
        else "missing",
        "merge_receipt": "present"
        if (root / "merge_receipt.json").is_file()
        else "missing",
        "test_open_receipt": "present"
        if (root / "run/test_open_receipt.json").is_file()
        else "missing",
        "completed_batches": [],
        "summary": None,
    }
    for path in sorted((root / "run/batches").glob("*.jsonl")):
        result["completed_batches"].append(
            {
                "path": path.name,
                "rows": len(_read_jsonl(path, label="status batch")),
                "sha256": sha256_file(path),
            }
        )
    if (root / "run/summary.json").is_file():
        result["summary"] = _read_json(root / "run/summary.json", label="final summary")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    for name in ("merge", "run"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--manifest-sha256", required=True)
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.output_root)
        elif args.command == "merge":
            with _exclusive_lock(args.manifest.parent):
                result = merge(args.manifest.resolve(), args.manifest_sha256)
        elif args.command == "run":
            with _exclusive_lock(args.manifest.parent):
                result = run(args.manifest.resolve(), args.manifest_sha256)
        else:
            result = status(args.output_root)
    except (
        FinalEvaluationError,
        FileNotFoundError,
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(
            canonical_json(
                {
                    "status": "blocked",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
