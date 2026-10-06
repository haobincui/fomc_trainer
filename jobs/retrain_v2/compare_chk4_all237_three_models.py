"""Greedy comparison of chk1, chk4 SFT-only, and chk4 GRPO on 237 meetings.

The comparison uses every unique meeting from the current chk4 release exactly
once per model.  Results are reported for the full 237-meeting population and
separately for train, validation, and test.  The full-population score is a
diagnostic training-population score, not a held-out generalization estimate.

Checkpoint-450's 13 test rows are imported from its completed one-shot sealed
evaluation.  They are not regenerated.  Every other row is generated greedily
on physical GPU1 with the same chk4 prompt and completion contract.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import itertools
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

from jobs.retrain_v2 import chk4_cp450_final_evaluation as final_protocol
from open_r1.provenance import fingerprint_artifact_path, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = REPO_ROOT / (
    "dataset/processed/retrain_v2/chk4_decision_pre2009_train_balanced_v1_20260811"
)
RELEASE_MANIFEST = RELEASE_ROOT / "release_manifest.json"
TRAINING_CONFIG = REPO_ROOT / (
    "configs/retrain_v2/"
    "chk4_decision_grpo_from_pre2009_cp38_direct_full_no_smoke_v1_20260812.yaml"
)
CP450_FINAL_ROOT = REPO_ROOT / (
    "output/evaluation/retrain_v2/chk4_cp450_final_test_v1_20260814"
)
CP450_FINAL_RESULTS = CP450_FINAL_ROOT / "run/results.jsonl"
CP450_FINAL_SUMMARY = CP450_FINAL_ROOT / "run/summary.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    "output/evaluation/retrain_v2/chk4_three_model_all237_greedy_v1_20260814"
)

RELEASE_MANIFEST_SHA256 = (
    "5141e00569b94e4af96f030f46176e9cd409199a6f26ac1c63a2ce335fe1e2d6"
)
TRAINING_CONFIG_SHA256 = (
    "222c70d76823681b7ff07971704c38fd153abcac4620a4e0fab4fce00d77f06f"
)
CP450_FINAL_RESULTS_SHA256 = (
    "8823d20974ccae60c3fb8d259494105960f49e197f98639cc031130945a3638e"
)
CP450_FINAL_SUMMARY_SHA256 = (
    "0ff44f8f962dac488543f5d6c44645950ea7f14cbbba95018f3fc9f5e7a07dae"
)
SYSTEM_PROMPT_SHA256 = (
    "426b64532dfb955a3941db2cc2bcfc9a785c0540ffba9ee94afabc189fe4ce18"
)

MODELS: tuple[dict[str, Any], ...] = (
    {
        "label": "chk1_cp200",
        "display_name": "chk1 cp200",
        "stage": "chk1",
        "checkpoint_step": 200,
        "path": REPO_ROOT
        / (
            "output/training/retrain_v2/"
            "chk1_clean_v2_lr1e6_selected_cp200_for_chk2_20260810/merged/chk1"
        ),
        "sha256": "0989b94792f8ab6377e2979aaedeb37010b9a2c5e05c77a459b4e5cda5b806e3",
        "test_origin": "fresh_generation_after_seal_opened",
    },
    {
        "label": "chk4_sft_cp38",
        "display_name": "chk4 SFT-only cp38",
        "stage": "chk4_sft_only",
        "checkpoint_step": 38,
        "path": REPO_ROOT
        / (
            "output/training/retrain_v2/"
            "chk4_from_chk1_cp200_sft_grpo_pre2009_balanced_lr1e5_steps39_v1_20260811/"
            "selected_sft_checkpoints/checkpoint-38/merged/chk4_sft"
        ),
        "sha256": "715bdb763043e7fda49e4a189b601d5e070fefbcecd14388bde41dbf7c165dd0",
        "test_origin": "fresh_generation_after_seal_opened",
    },
    {
        "label": "chk4_grpo_cp450",
        "display_name": "chk4 GRPO cp450",
        "stage": "chk4_grpo",
        "checkpoint_step": 450,
        "path": REPO_ROOT
        / (
            "output/training/retrain_v2/"
            "chk4_from_pre2009_cp38_direct_grpo_full_no_smoke_v1_20260812/"
            "selected_grpo_checkpoints/checkpoint-450/merged/chk4_grpo"
        ),
        "sha256": "92cb337a4d36486185074a1a99574860ae011663f6679a18d53b822543dee0a5",
        "test_origin": "reused_completed_one_shot_sealed_test",
    },
)

SPLITS = ("train", "validation", "test")
DIRECTIONS = ("cut", "hold", "hike")
EXPECTED_UNIQUE_COUNTS = {"train": 211, "validation": 13, "test": 13}
EXPECTED_TOTAL = 237
EXPECTED_RESULT_ROWS = EXPECTED_TOTAL * len(MODELS)
EXPECTED_FRESH_GENERATIONS = EXPECTED_RESULT_ROWS - EXPECTED_UNIQUE_COUNTS["test"]
MAX_NEW_TOKENS = 1536
TAIL_TOKENS = 256
BATCH_SIZE = 8
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260814

MANIFEST_SCHEMA = "chk4-three-model-all237-comparison-manifest-v1"
RESULT_SCHEMA = "chk4-three-model-all237-comparison-result-v1"
SUMMARY_SCHEMA = "chk4-three-model-all237-comparison-summary-v1"


class ComparisonError(RuntimeError):
    """An input, model, batch, or metric failed a comparison invariant."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonError(message)


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


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label} missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"cannot read {label}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must contain an object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"{label} missing or unsafe")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                _require(isinstance(value, dict), f"{label} row {line_number} invalid")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"cannot read {label}: {exc}") from exc
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


def _write_exclusive_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            if not value.endswith("\n"):
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _file_record(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"unsafe file: {path}")
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".jsonl":
        with path.open("rb") as handle:
            record["rows"] = sum(bool(line.strip()) for line in handle)
    return record


@contextmanager
def _exclusive_lock(output_root: Path) -> Iterable[None]:
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".comparison.lock"
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ComparisonError("comparison lock is held") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _training_config() -> dict[str, Any]:
    _require(
        sha256_file(TRAINING_CONFIG) == TRAINING_CONFIG_SHA256,
        "training config drift",
    )
    value = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "training config must be a mapping")
    _require(
        _sha256_text(str(value.get("system_prompt"))) == SYSTEM_PROMPT_SHA256,
        "system prompt drift",
    )
    return value


def _model_by_label(label: str) -> dict[str, Any]:
    matches = [model for model in MODELS if model["label"] == label]
    _require(len(matches) == 1, f"unknown model label: {label}")
    return matches[0]


def verify_model(model: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(model["path"])
    observed = fingerprint_artifact_path(path)
    _require(observed.get("sha256") == model["sha256"], f"{model['label']} drift")
    return observed


def _load_tokenizer(model: Mapping[str, Any]) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model["path"]), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_population(tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, str]]:
    release = _read_json(RELEASE_MANIFEST, label="release manifest")
    _require(
        sha256_file(RELEASE_MANIFEST) == RELEASE_MANIFEST_SHA256,
        "release manifest drift",
    )
    _require(
        release.get("quality_status") == "passed"
        and release.get("immutable") is True
        and release.get("training_ready") is True,
        "release is not passed, immutable, and training-ready",
    )
    _require(
        release.get("unique_split_counts") == EXPECTED_UNIQUE_COUNTS,
        "unique split counts drift",
    )
    config = _training_config()
    system_prompt = str(config["system_prompt"])
    samples: list[dict[str, Any]] = []
    prompts: dict[str, str] = {}
    seen: set[str] = set()
    for split in SPLITS:
        unique_path = RELEASE_ROOT / f"manifests/unique/{split}.jsonl"
        physical_path = RELEASE_ROOT / f"decision_grpo/{split}.jsonl"
        expected_unique = release["files"][f"manifests/unique/{split}.jsonl"]
        expected_physical = release["files"][f"decision_grpo/{split}.jsonl"]
        for path, expected, label in (
            (unique_path, expected_unique, f"unique {split}"),
            (physical_path, expected_physical, f"physical {split}"),
        ):
            observed = _file_record(path)
            _require(
                observed["sha256"] == expected["sha256"]
                and observed["bytes"] == expected["bytes"]
                and observed["rows"] == expected["rows"],
                f"{label} release binding drift",
            )
        physical = _read_jsonl(physical_path, label=f"physical {split}")
        prompt_map: dict[str, str] = {}
        target_map: dict[str, tuple[str, int]] = {}
        for row in physical:
            sample_id = str(row["sample_id"])
            prompt = str(row["prompt"])
            target = (str(row["direction"]), int(row["magnitude_bp"]))
            if sample_id in prompt_map:
                _require(prompt_map[sample_id] == prompt, f"{split} prompt variants")
                _require(target_map[sample_id] == target, f"{split} target variants")
            prompt_map[sample_id] = prompt
            target_map[sample_id] = target
        unique = _read_jsonl(unique_path, label=f"unique {split}")
        _require(len(unique) == EXPECTED_UNIQUE_COUNTS[split], f"{split} count drift")
        _require(
            set(prompt_map) == {str(row["sample_id"]) for row in unique},
            f"{split} ID closure drift",
        )
        for row in unique:
            sample_id = str(row["sample_id"])
            _require(sample_id not in seen, f"cross-split duplicate: {sample_id}")
            seen.add(sample_id)
            prompt = prompt_map[sample_id]
            _require(
                _sha256_text(prompt) == row["prompt_sha256"],
                f"{sample_id} prompt hash drift",
            )
            _require(
                target_map[sample_id]
                == (str(row["direction"]), int(row["magnitude_bp"])),
                f"{sample_id} target drift",
            )
            prompt_ids = final_protocol.generation_probe._prompt_ids(
                tokenizer,
                final_protocol.generation_probe._messages(system_prompt, prompt),
            )
            _require(len(prompt_ids) <= 2560, f"{sample_id} prompt exceeds bound")
            samples.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "meeting_date": row["meeting_date"],
                    "direction": row["direction"],
                    "magnitude_bp": int(row["magnitude_bp"]),
                    "prompt_sha256": row["prompt_sha256"],
                    "prompt_token_count": len(prompt_ids),
                }
            )
            prompts[sample_id] = prompt
    _require(len(samples) == len(prompts) == EXPECTED_TOTAL, "population count drift")
    return samples, prompts


def _population_binding(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_split = Counter(str(sample["split"]) for sample in samples)
    by_direction = Counter(str(sample["direction"]) for sample in samples)
    fields = [
        {
            key: sample[key]
            for key in (
                "sample_id",
                "split",
                "meeting_date",
                "direction",
                "magnitude_bp",
                "prompt_sha256",
                "prompt_token_count",
            )
        }
        for sample in samples
    ]
    return {
        "unique_meetings": len(samples),
        "by_split": dict(sorted(by_split.items())),
        "by_direction": dict(sorted(by_direction.items())),
        "ordered_population_sha256": _sha256_text(canonical_json(fields)),
        "sample_ids_sha256": _sha256_text(
            canonical_json([sample["sample_id"] for sample in samples])
        ),
    }


def build_manifest(output_root: Path) -> dict[str, Any]:
    tokenizer = _load_tokenizer(MODELS[-1])
    samples, _ = load_population(tokenizer)
    tokenizer_hashes = {
        sha256_file(Path(model["path"]) / "tokenizer.json") for model in MODELS
    }
    _require(len(tokenizer_hashes) == 1, "model tokenizers differ")
    return {
        "schema_version": MANIFEST_SCHEMA,
        "purpose": "three_model_greedy_comparison_on_all_237_unique_chk4_meetings",
        "created_at_utc": _utc_now(),
        "authorization_basis": "explicit_user_request_2026-08-14_all237_three_model_comparison",
        "output_root": str(output_root.resolve()),
        "models": [
            {
                "label": model["label"],
                "display_name": model["display_name"],
                "stage": model["stage"],
                "checkpoint_step": model["checkpoint_step"],
                "artifact": verify_model(model),
                "test_origin": model["test_origin"],
            }
            for model in MODELS
        ],
        "release": {
            "manifest": _file_record(RELEASE_MANIFEST),
            "population": _population_binding(samples),
            "files": {
                f"{kind}/{split}": _file_record(
                    RELEASE_ROOT
                    / (
                        f"manifests/unique/{split}.jsonl"
                        if kind == "unique"
                        else f"decision_grpo/{split}.jsonl"
                    )
                )
                for kind in ("unique", "physical")
                for split in SPLITS
            },
            "test_was_previously_opened": True,
            "full_population_is_not_held_out": True,
        },
        "prompt_contract": {
            "training_config": _file_record(TRAINING_CONFIG),
            "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
            "shared_tokenizer_sha256": next(iter(tokenizer_hashes)),
            "max_prompt_tokens": 2560,
        },
        "generation": {
            "mode": "greedy_one_completion_per_unique_meeting_per_model",
            "max_new_tokens": MAX_NEW_TOKENS,
            "batch_size": BATCH_SIZE,
            "physical_gpu": 1,
            "visible_device_count": 1,
            "load_in_4bit": True,
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "expected_result_rows": EXPECTED_RESULT_ROWS,
            "expected_fresh_gpu_generations": EXPECTED_FRESH_GENERATIONS,
            "reused_cp450_test_rows": EXPECTED_UNIQUE_COUNTS["test"],
        },
        "metrics": {
            "scopes": ["all_237", *SPLITS],
            "primary": [
                "direction_accuracy",
                "balanced_accuracy_fixed_three_class",
                "macro_f1_fixed_three_class",
                "exact_direction_magnitude_accuracy",
            ],
            "delivery": [
                "delivery_valid_rate",
                "strict_json_rate",
                "eos_rate",
                "cap_rate",
                "periodic_tail_count",
            ],
            "paired_bootstrap": {
                "scope": "all_237",
                "draws": BOOTSTRAP_DRAWS,
                "seed": BOOTSTRAP_SEED,
                "stratified_by": "target_direction",
            },
        },
        "reuse_source": {
            "cp450_final_results": _file_record(CP450_FINAL_RESULTS),
            "cp450_final_summary": _file_record(CP450_FINAL_SUMMARY),
        },
        "limitations": [
            "211 of 237 meetings are the chk4 training split",
            "the all-237 result is diagnostic and not a held-out generalization estimate",
            "validation has no cut meetings",
            "test has 13 meetings and only one hike",
            "the release is not canonical-DAG-bindable",
        ],
    }


def prepare(output_root: Path) -> dict[str, Any]:
    root = output_root.resolve()
    _require(not root.exists() and not root.is_symlink(), f"output root exists: {root}")
    _require(
        sha256_file(CP450_FINAL_RESULTS) == CP450_FINAL_RESULTS_SHA256,
        "cp450 final results drift",
    )
    _require(
        sha256_file(CP450_FINAL_SUMMARY) == CP450_FINAL_SUMMARY_SHA256,
        "cp450 final summary drift",
    )
    root.mkdir(parents=True, exist_ok=False)
    manifest = build_manifest(root)
    path = root / "evaluation_manifest.json"
    _write_exclusive_json(path, manifest)
    return {
        "status": "prepared",
        "manifest": _file_record(path),
        "unique_meetings": EXPECTED_TOTAL,
        "expected_result_rows": EXPECTED_RESULT_ROWS,
        "expected_fresh_gpu_generations": EXPECTED_FRESH_GENERATIONS,
    }


def validate_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    _require(sha256_file(path) == expected_sha256, "evaluation manifest SHA drift")
    manifest = _read_json(path, label="evaluation manifest")
    _require(manifest.get("schema_version") == MANIFEST_SCHEMA, "manifest schema drift")
    _require(
        manifest.get("output_root") == str(path.parent.resolve()), "output root drift"
    )
    _require(
        sha256_file(RELEASE_MANIFEST) == RELEASE_MANIFEST_SHA256,
        "release manifest drift",
    )
    _require(
        sha256_file(CP450_FINAL_RESULTS) == CP450_FINAL_RESULTS_SHA256,
        "cp450 final results drift",
    )
    _require(
        sha256_file(CP450_FINAL_SUMMARY) == CP450_FINAL_SUMMARY_SHA256,
        "cp450 final summary drift",
    )
    for model, bound in zip(MODELS, manifest["models"], strict=True):
        observed = verify_model(model)
        _require(observed == bound["artifact"], f"{model['label']} manifest drift")
    tokenizer = _load_tokenizer(MODELS[-1])
    samples, _ = load_population(tokenizer)
    _require(
        _population_binding(samples) == manifest["release"]["population"],
        "population binding drift",
    )
    return manifest


def _chunks(
    values: Sequence[Mapping[str, Any]], size: int
) -> list[list[dict[str, Any]]]:
    return [
        [dict(value) for value in values[index : index + size]]
        for index in range(0, len(values), size)
    ]


def _batch_seed(model_label: str, split: str, batch_index: int) -> int:
    value = f"chk4-three-model-all237-v1:{model_label}:{split}:{batch_index}"
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16) % (2**31)


def batch_contracts(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for model in MODELS:
        for split in SPLITS:
            split_samples = [sample for sample in samples if sample["split"] == split]
            for batch_index, batch in enumerate(_chunks(split_samples, BATCH_SIZE)):
                contracts.append(
                    {
                        "model_label": model["label"],
                        "split": split,
                        "batch_index": batch_index,
                        "sample_ids": [sample["sample_id"] for sample in batch],
                        "seed": _batch_seed(model["label"], split, batch_index),
                        "origin": model["test_origin"]
                        if split == "test"
                        else "fresh_generation",
                    }
                )
    return contracts


def _batch_path(run_root: Path, contract: Mapping[str, Any]) -> Path:
    return (
        run_root
        / "batches"
        / str(contract["model_label"])
        / f"{contract['split']}_{int(contract['batch_index']):02d}.jsonl"
    )


def _validate_batch_rows(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    manifest_sha: str,
) -> None:
    _require(len(rows) == len(contract["sample_ids"]), "batch row count drift")
    _require(
        [row.get("sample_id") for row in rows] == list(contract["sample_ids"]),
        "batch sample order drift",
    )
    model = _model_by_label(str(contract["model_label"]))
    for row in rows:
        _require(row.get("schema_version") == RESULT_SCHEMA, "batch schema drift")
        _require(row.get("model_label") == model["label"], "batch model drift")
        _require(
            row.get("merged_model_sha256") == model["sha256"], "batch model SHA drift"
        )
        _require(row.get("split") == contract["split"], "batch split drift")
        _require(row.get("generation_mode") == "greedy", "batch mode drift")
        _require(row.get("generation_index") == 0, "batch generation index drift")
        _require(
            row.get("batch_index") == contract["batch_index"]
            and row.get("batch_seed") == contract["seed"],
            "batch seed/index drift",
        )
        _require(
            row.get("evaluation_manifest_sha256") == manifest_sha,
            "batch manifest drift",
        )


def _load_model(path: Path) -> Any:
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
        str(path),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        quantization_config=quantization,
        device_map={"": 0},
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    _require(
        not any(value.is_meta for value in model.parameters()),
        "model has meta parameters",
    )
    model.eval()
    model.config.use_cache = True
    return model


def _fresh_result_row(
    *,
    manifest_sha: str,
    model_spec: Mapping[str, Any],
    sample: Mapping[str, Any],
    batch_index: int,
    seed: int,
    raw_ids: Sequence[int],
    eos_ids: set[int],
    tokenizer: Any,
) -> dict[str, Any]:
    row = final_protocol._result_row(
        manifest_sha=manifest_sha,
        merged_sha=str(model_spec["sha256"]),
        sample=sample,
        mode="greedy",
        generation_index=0,
        batch_index=batch_index,
        seed=seed,
        raw_ids=raw_ids,
        eos_ids=eos_ids,
        tokenizer=tokenizer,
    )
    row.update(
        {
            "schema_version": RESULT_SCHEMA,
            "model_label": model_spec["label"],
            "model_display_name": model_spec["display_name"],
            "model_stage": model_spec["stage"],
            "checkpoint_step": model_spec["checkpoint_step"],
            "split": sample["split"],
            "meeting_date": sample["meeting_date"],
            "result_origin": "fresh_generation",
        }
    )
    return row


def _cp450_reused_rows(
    samples: Sequence[Mapping[str, Any]],
    contracts: Sequence[Mapping[str, Any]],
    manifest_sha: str,
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    _require(
        sha256_file(CP450_FINAL_RESULTS) == CP450_FINAL_RESULTS_SHA256,
        "cp450 source results drift",
    )
    source = [
        row
        for row in _read_jsonl(CP450_FINAL_RESULTS, label="cp450 final results")
        if row.get("generation_mode") == "greedy"
    ]
    test_samples = [sample for sample in samples if sample["split"] == "test"]
    _require(len(source) == len(test_samples) == 13, "cp450 reused test count drift")
    _require(
        [row["sample_id"] for row in source]
        == [sample["sample_id"] for sample in test_samples],
        "cp450 reused test order drift",
    )
    source_by_id = {str(row["sample_id"]): row for row in source}
    sample_by_id = {str(sample["sample_id"]): sample for sample in test_samples}
    result: dict[tuple[str, int], list[dict[str, Any]]] = {}
    relevant = [
        contract
        for contract in contracts
        if contract["model_label"] == "chk4_grpo_cp450" and contract["split"] == "test"
    ]
    for contract in relevant:
        rows: list[dict[str, Any]] = []
        for sample_id in contract["sample_ids"]:
            old = source_by_id[str(sample_id)]
            sample = sample_by_id[str(sample_id)]
            _require(
                old.get("merged_model_sha256") == MODELS[-1]["sha256"],
                "cp450 source model drift",
            )
            _require(
                old.get("prompt_sha256") == sample["prompt_sha256"],
                "cp450 source prompt drift",
            )
            _require(
                old.get("target_direction") == sample["direction"],
                "cp450 source target drift",
            )
            row = dict(old)
            row.update(
                {
                    "schema_version": RESULT_SCHEMA,
                    "model_label": MODELS[-1]["label"],
                    "model_display_name": MODELS[-1]["display_name"],
                    "model_stage": MODELS[-1]["stage"],
                    "checkpoint_step": MODELS[-1]["checkpoint_step"],
                    "split": "test",
                    "meeting_date": sample["meeting_date"],
                    "batch_index": contract["batch_index"],
                    "batch_seed": contract["seed"],
                    "evaluation_manifest_sha256": manifest_sha,
                    "result_origin": "reused_completed_one_shot_sealed_test",
                    "source_final_evaluation_manifest_sha256": old.get(
                        "evaluation_manifest_sha256"
                    ),
                    "source_final_result_schema_version": old.get("schema_version"),
                    "source_final_batch_index": old.get("batch_index"),
                    "source_final_batch_seed": old.get("batch_seed"),
                    "source_final_results_sha256": CP450_FINAL_RESULTS_SHA256,
                }
            )
            rows.append(row)
        result[("test", int(contract["batch_index"]))] = rows
    return result


def _extended_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = final_protocol.metric_block(rows)
    supported = [
        direction
        for direction in DIRECTIONS
        if metrics["per_direction"][direction]["support"] > 0
    ]
    metrics["balanced_accuracy_supported_classes"] = statistics.fmean(
        metrics["per_direction"][direction]["recall"] for direction in supported
    )
    metrics["supported_directions"] = supported
    metrics["response_format_counts"] = dict(
        sorted(Counter(str(row.get("response_format")) for row in rows).items())
    )
    metrics["decision_rejection_reason_counts"] = dict(
        sorted(
            Counter(
                str(row.get("decision_rejection_reason"))
                for row in rows
                if row.get("decision_rejection_reason") is not None
            ).items()
        )
    )
    token_counts = [int(row.get("completion_token_count", 0)) for row in rows]
    metrics["completion_tokens"] = {
        "mean": statistics.fmean(token_counts),
        "median": statistics.median(token_counts),
        "max": max(token_counts),
    }
    return metrics


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_comparison(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    _require(len(rows_a) == len(rows_b) and bool(rows_a), "paired row count drift")
    ids_a = [str(row["sample_id"]) for row in rows_a]
    ids_b = [str(row["sample_id"]) for row in rows_b]
    _require(ids_a == ids_b, "paired sample order drift")
    metrics_a = _extended_metrics(rows_a)
    metrics_b = _extended_metrics(rows_b)
    keys = (
        "direction_accuracy",
        "balanced_accuracy",
        "macro_f1",
        "exact_direction_magnitude_accuracy",
        "delivery_valid_rate",
        "strict_json_rate",
    )
    deltas = {key: float(metrics_b[key]) - float(metrics_a[key]) for key in keys}
    a_correct = [bool(row.get("decision_direction_correct")) for row in rows_a]
    b_correct = [bool(row.get("decision_direction_correct")) for row in rows_b]
    discordance = {
        "a_only_direction_correct": sum(
            a and not b for a, b in zip(a_correct, b_correct, strict=True)
        ),
        "b_only_direction_correct": sum(
            b and not a for a, b in zip(a_correct, b_correct, strict=True)
        ),
        "both_direction_correct": sum(
            a and b for a, b in zip(a_correct, b_correct, strict=True)
        ),
        "neither_direction_correct": sum(
            not a and not b for a, b in zip(a_correct, b_correct, strict=True)
        ),
    }
    groups = {
        direction: [
            index
            for index, row in enumerate(rows_a)
            if row["target_direction"] == direction
        ]
        for direction in DIRECTIONS
    }
    _require(all(groups.values()), "paired bootstrap requires every direction")
    values = {key: [] for key in keys}
    rng = random.Random(seed)
    for _ in range(draws):
        indices: list[int] = []
        for direction in DIRECTIONS:
            group = groups[direction]
            indices.extend(group[rng.randrange(len(group))] for _ in range(len(group)))
        draw_a = [rows_a[index] for index in indices]
        draw_b = [rows_b[index] for index in indices]
        block_a = final_protocol.metric_block(draw_a)
        block_b = final_protocol.metric_block(draw_b)
        for key in keys:
            values[key].append(float(block_b[key]) - float(block_a[key]))
    return {
        "delta_definition": "model_b_minus_model_a",
        "deltas": deltas,
        "paired_direction_correctness": discordance,
        "bootstrap": {
            "method": "paired_meeting_level_stratified_by_target_direction_percentile",
            "draws": draws,
            "seed": seed,
            "interval": 0.95,
            "deltas": {
                key: {
                    "lower": _percentile(value, 0.025),
                    "upper": _percentile(value, 0.975),
                }
                for key, value in values.items()
            },
        },
    }


def summarize(rows: Sequence[Mapping[str, Any]], manifest_sha: str) -> dict[str, Any]:
    _require(len(rows) == EXPECTED_RESULT_ROWS, "result row count drift")
    expected_keys = {
        (model["label"], sample_id)
        for model in MODELS
        for sample_id in {
            str(row["sample_id"])
            for row in rows
            if row["model_label"] == MODELS[0]["label"]
        }
    }
    observed_keys = {(str(row["model_label"]), str(row["sample_id"])) for row in rows}
    _require(len(observed_keys) == EXPECTED_RESULT_ROWS, "duplicate result key")
    _require(observed_keys == expected_keys, "three-model result closure drift")
    model_rows = {
        model["label"]: [row for row in rows if row["model_label"] == model["label"]]
        for model in MODELS
    }
    models: dict[str, Any] = {}
    for model in MODELS:
        label = str(model["label"])
        current = model_rows[label]
        models[label] = {
            "display_name": model["display_name"],
            "stage": model["stage"],
            "checkpoint_step": model["checkpoint_step"],
            "model_sha256": model["sha256"],
            "overall": _extended_metrics(current),
            "by_split": {
                split: _extended_metrics(
                    [row for row in current if row["split"] == split]
                )
                for split in SPLITS
            },
        }
    pairwise: dict[str, Any] = {}
    for model_a, model_b in itertools.combinations(MODELS, 2):
        label_a = str(model_a["label"])
        label_b = str(model_b["label"])
        key = f"{label_a}__vs__{label_b}"
        pairwise[key] = {
            "model_a": label_a,
            "model_b": label_b,
            "all_237": paired_comparison(model_rows[label_a], model_rows[label_b]),
            "by_split_deltas": {
                split: {
                    metric: models[label_b]["by_split"][split][metric]
                    - models[label_a]["by_split"][split][metric]
                    for metric in (
                        "direction_accuracy",
                        "balanced_accuracy",
                        "macro_f1",
                        "exact_direction_magnitude_accuracy",
                        "delivery_valid_rate",
                        "strict_json_rate",
                    )
                }
                for split in SPLITS
            },
        }
    leaderboards = {
        scope: sorted(
            (
                {
                    "model_label": model["label"],
                    "balanced_accuracy": models[model["label"]][
                        "overall" if scope == "all_237" else "by_split"
                    ]["balanced_accuracy"]
                    if scope == "all_237"
                    else models[model["label"]]["by_split"][scope]["balanced_accuracy"],
                    "direction_accuracy": models[model["label"]][
                        "overall" if scope == "all_237" else "by_split"
                    ]["direction_accuracy"]
                    if scope == "all_237"
                    else models[model["label"]]["by_split"][scope][
                        "direction_accuracy"
                    ],
                }
                for model in MODELS
            ),
            key=lambda item: (
                -item["balanced_accuracy"],
                -item["direction_accuracy"],
                item["model_label"],
            ),
        )
        for scope in ("all_237", *SPLITS)
    }
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "evaluation_manifest_sha256": manifest_sha,
        "unique_meetings": EXPECTED_TOTAL,
        "models_compared": len(MODELS),
        "result_rows": len(rows),
        "fresh_gpu_generations": sum(
            row["result_origin"] == "fresh_generation" for row in rows
        ),
        "reused_cp450_sealed_test_rows": sum(
            row["result_origin"] == "reused_completed_one_shot_sealed_test"
            for row in rows
        ),
        "population": {
            "by_split": dict(
                sorted(
                    Counter(
                        str(row["split"]) for row in model_rows[MODELS[0]["label"]]
                    ).items()
                )
            ),
            "by_direction": dict(
                sorted(
                    Counter(
                        str(row["target_direction"])
                        for row in model_rows[MODELS[0]["label"]]
                    ).items()
                )
            ),
        },
        "models": models,
        "pairwise": pairwise,
        "leaderboards": leaderboards,
        "interpretation_rules": [
            "all_237 is a training-population diagnostic, not a generalization estimate",
            "test is the held-out comparison but has only 13 meetings",
            "validation balanced accuracy includes a zero-recall absent cut class under the fixed-three-class definition",
            "paired confidence intervals describe resampling uncertainty over these 237 meetings, not new time periods",
            "no canonical promotion claim is authorized by this comparison",
        ],
    }


def _summary_text(summary: Mapping[str, Any]) -> str:
    lines = [
        "chk4 three-model comparison on 237 unique meetings",
        "",
        "All-237 scores are diagnostic because 211 meetings are from the chk4 training split.",
        "The 13-meeting test split is the held-out comparison.",
        "",
        "model | scope | direction_accuracy | balanced_accuracy | macro_f1 | exact_action | delivery_valid | strict_json",
    ]
    for model in MODELS:
        block = summary["models"][model["label"]]
        for scope in ("all_237", *SPLITS):
            metrics = (
                block["overall"] if scope == "all_237" else block["by_split"][scope]
            )
            lines.append(
                " | ".join(
                    [
                        str(model["display_name"]),
                        scope,
                        f"{metrics['direction_accuracy']:.6f}",
                        f"{metrics['balanced_accuracy']:.6f}",
                        f"{metrics['macro_f1']:.6f}",
                        f"{metrics['exact_direction_magnitude_accuracy']:.6f}",
                        f"{metrics['delivery_valid_rate']:.6f}",
                        f"{metrics['strict_json_rate']:.6f}",
                    ]
                )
            )
    return "\n".join(lines) + "\n"


def run(manifest_path: Path, manifest_sha: str) -> dict[str, Any]:
    _require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "1",
        "comparison requires physical GPU1 only",
    )
    validate_manifest(manifest_path, manifest_sha)
    output_root = manifest_path.parent
    run_root = output_root / "run"
    run_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "summary.json"
    if summary_path.is_file():
        return _read_json(summary_path, label="comparison summary")

    config = _training_config()
    reference_tokenizer = _load_tokenizer(MODELS[-1])
    samples, prompts = load_population(reference_tokenizer)
    sample_by_id = {str(sample["sample_id"]): sample for sample in samples}
    contracts = batch_contracts(samples)
    _require(len(contracts) == 93, "batch contract count drift")
    contract_path = run_root / "batch_contract.json"
    if contract_path.exists():
        _require(
            _read_json(contract_path, label="batch contract").get("batches")
            == contracts,
            "batch contract drift",
        )
    else:
        _write_exclusive_json(
            contract_path,
            {"schema_version": MANIFEST_SCHEMA, "batches": contracts},
        )
    launch_path = run_root / "launch.json"
    if not launch_path.exists():
        _write_exclusive_json(
            launch_path,
            {
                "schema_version": SUMMARY_SCHEMA,
                "status": "running",
                "created_at_utc": _utc_now(),
                "evaluation_manifest_sha256": manifest_sha,
                "runtime": {
                    "python": sys.executable,
                    "python_version": platform.python_version(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                },
            },
        )

    reused = _cp450_reused_rows(samples, contracts, manifest_sha)
    for contract in contracts:
        if contract["origin"] != "reused_completed_one_shot_sealed_test":
            continue
        path = _batch_path(run_root, contract)
        rows = reused[(str(contract["split"]), int(contract["batch_index"]))]
        _validate_batch_rows(rows, contract, manifest_sha)
        if path.exists():
            existing = _read_jsonl(path, label="reused cp450 batch")
            _validate_batch_rows(existing, contract, manifest_sha)
            _require(
                canonical_json(existing) == canonical_json(rows), "reused batch drift"
            )
        else:
            _write_exclusive_jsonl(path, rows)

    import torch
    import transformers

    _require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "runtime must expose exactly one GPU",
    )
    peak_by_model: dict[str, Any] = {}
    for model_spec in MODELS:
        pending = [
            contract
            for contract in contracts
            if contract["model_label"] == model_spec["label"]
            and contract["origin"] != "reused_completed_one_shot_sealed_test"
            and not _batch_path(run_root, contract).exists()
        ]
        if not pending:
            print(
                canonical_json(
                    {"status": "model_already_complete", "model": model_spec["label"]}
                ),
                flush=True,
            )
            continue
        tokenizer = _load_tokenizer(model_spec)
        model = None
        try:
            print(
                canonical_json(
                    {
                        "status": "loading_model",
                        "model": model_spec["label"],
                        "pending_batches": len(pending),
                    }
                ),
                flush=True,
            )
            model = _load_model(Path(model_spec["path"]))
            eos_value = getattr(model.generation_config, "eos_token_id", None)
            if eos_value is None:
                eos_value = tokenizer.eos_token_id
            eos_ids = set(final_protocol.generation_probe._normalize_eos_ids(eos_value))
            _require(bool(eos_ids), "model/tokenizer has no EOS")
            torch.cuda.reset_peak_memory_stats()
            for contract in [
                item
                for item in contracts
                if item["model_label"] == model_spec["label"]
                and item["origin"] != "reused_completed_one_shot_sealed_test"
            ]:
                path = _batch_path(run_root, contract)
                if path.exists():
                    existing = _read_jsonl(path, label="completed comparison batch")
                    _validate_batch_rows(existing, contract, manifest_sha)
                    continue
                transformers.set_seed(int(contract["seed"]))
                batch_samples = [
                    sample_by_id[str(sample_id)] for sample_id in contract["sample_ids"]
                ]
                rendered = [
                    tokenizer.apply_chat_template(
                        final_protocol.generation_probe._messages(
                            str(config["system_prompt"]),
                            prompts[str(sample["sample_id"])],
                        ),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for sample in batch_samples
                ]
                encoded = tokenizer(
                    rendered,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                input_ids = encoded["input_ids"].to("cuda:0")
                attention_mask = encoded["attention_mask"].to("cuda:0")
                for index, sample in enumerate(batch_samples):
                    _require(
                        int(attention_mask[index].sum().item())
                        == sample["prompt_token_count"],
                        f"prompt token count drift: {sample['sample_id']}",
                    )
                with torch.inference_mode():
                    sequences = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=sorted(eos_ids),
                        use_cache=True,
                    )
                _require(len(sequences) == len(batch_samples), "generation count drift")
                prompt_width = int(input_ids.shape[1])
                rows = [
                    _fresh_result_row(
                        manifest_sha=manifest_sha,
                        model_spec=model_spec,
                        sample=sample,
                        batch_index=int(contract["batch_index"]),
                        seed=int(contract["seed"]),
                        raw_ids=sequences[index, prompt_width:].tolist(),
                        eos_ids=eos_ids,
                        tokenizer=tokenizer,
                    )
                    for index, sample in enumerate(batch_samples)
                ]
                _validate_batch_rows(rows, contract, manifest_sha)
                _write_exclusive_jsonl(path, rows)
                completed = sum(
                    _batch_path(run_root, item).exists()
                    for item in contracts
                    if item["model_label"] == model_spec["label"]
                )
                total = sum(
                    item["model_label"] == model_spec["label"] for item in contracts
                )
                print(
                    canonical_json(
                        {
                            "status": "batch_complete",
                            "model": model_spec["label"],
                            "split": contract["split"],
                            "batch_index": contract["batch_index"],
                            "rows": len(rows),
                            "model_batches_complete": completed,
                            "model_batches_total": total,
                        }
                    ),
                    flush=True,
                )
                del input_ids, attention_mask, sequences
            peak_by_model[str(model_spec["label"])] = {
                "peak_allocated_gib": round(
                    torch.cuda.max_memory_allocated() / 1024**3, 3
                ),
                "peak_reserved_gib": round(
                    torch.cuda.max_memory_reserved() / 1024**3, 3
                ),
            }
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    for contract in contracts:
        path = _batch_path(run_root, contract)
        batch_rows = _read_jsonl(path, label="final comparison batch")
        _validate_batch_rows(batch_rows, contract, manifest_sha)
        rows.extend(batch_rows)
    result_path = run_root / "results.jsonl"
    if result_path.exists():
        existing = _read_jsonl(result_path, label="comparison results")
        _require(
            canonical_json(existing) == canonical_json(rows), "result assembly drift"
        )
    else:
        _write_exclusive_jsonl(result_path, rows)
    summary = summarize(rows, manifest_sha)
    summary["runtime"] = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peak_by_model": peak_by_model,
    }
    summary["results"] = _file_record(result_path)
    _write_exclusive_json(summary_path, summary)
    _write_exclusive_text(run_root / "comparison.txt", _summary_text(summary))
    return summary


def status(output_root: Path) -> dict[str, Any]:
    root = output_root.resolve()
    batch_root = root / "run/batches"
    completed = {
        model["label"]: len(list((batch_root / model["label"]).glob("*.jsonl")))
        if (batch_root / model["label"]).is_dir()
        else 0
        for model in MODELS
    }
    result: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA,
        "output_root": str(root),
        "manifest": _file_record(root / "evaluation_manifest.json")
        if (root / "evaluation_manifest.json").is_file()
        else None,
        "completed_batches": completed,
        "expected_batches_per_model": 31,
        "summary": None,
    }
    if (root / "run/summary.json").is_file():
        result["summary"] = _read_json(root / "run/summary.json", label="summary")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--manifest-sha256", required=True)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.output_root)
        elif args.command == "run":
            with _exclusive_lock(args.manifest.parent):
                result = run(args.manifest, args.manifest_sha256)
        else:
            result = status(args.output_root)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except ComparisonError as exc:
        print(
            canonical_json({"status": "failed_closed", "error": str(exc)}),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
