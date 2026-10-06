"""Select a chk4 GRPO checkpoint without opening the sealed test split.

The probe has three pre-registered panels:

* ``selection``: nine validation prompts, greedy plus one sampled completion,
  for the exact cp38 baseline and five GRPO checkpoints;
* ``blind``: the four disjoint validation prompts, greedy plus four sampled
  completions, for the two best selection checkpoints; and
* ``retention``: six deterministic train prompts (two per direction), greedy
  plus one sampled completion, used only as a cut/hold/hike forgetting guard.

The validation split is intentionally reported as skewed (10 hike, 3 hold,
zero cut).  The train retention panel must never be described as independent
evaluation evidence, and ``decision_grpo/test.jsonl`` is never read here.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

from jobs.retrain_v2 import probe_chk4_decision_grpo_stratified as reward_probe
from jobs.retrain_v2 import probe_chk4_decision_sft_generation as generation_probe
from jobs.retrain_v2 import probe_chk4_pre2009_correction_v2 as eos_probe
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
ADAPTER_ROOT = TRAIN_ROOT / "adapters/chk4_grpo"
BASE_MODEL = REPO_ROOT / (
    "output/training/retrain_v2/"
    "chk4_from_chk1_cp200_sft_grpo_pre2009_balanced_lr1e5_steps39_v1_20260811/"
    "selected_sft_checkpoints/checkpoint-38/merged/chk4_sft"
)
BASE_MODEL_SHA256 = (
    "715bdb763043e7fda49e4a189b601d5e070fefbcecd14388bde41dbf7c165dd0"
)
TRAINING_CONFIG = REPO_ROOT / (
    "configs/retrain_v2/"
    "chk4_decision_grpo_from_pre2009_cp38_direct_full_no_smoke_v1_20260812.yaml"
)
TRAINING_CONFIG_SHA256 = (
    "222c70d76823681b7ff07971704c38fd153abcac4620a4e0fab4fce00d77f06f"
)
RELEASE_ROOT = REPO_ROOT / (
    "dataset/processed/retrain_v2/"
    "chk4_decision_pre2009_train_balanced_v1_20260811"
)
RELEASE_MANIFEST_SHA256 = (
    "5141e00569b94e4af96f030f46176e9cd409199a6f26ac1c63a2ce335fe1e2d6"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    "output/evaluation/retrain_v2/"
    "chk4_cp38_direct_grpo_checkpoint_probe_v2_batch8_20260814"
)

CANDIDATE_STEPS = (310, 360, 410, 450, 468)
SELECTION_STAGE = "selection"
BLIND_STAGE = "blind"
RETENTION_STAGE = "retention"
STAGES = (SELECTION_STAGE, BLIND_STAGE, RETENTION_STAGE)
MAX_NEW_TOKENS = 1536
TAIL_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
SELECTION_SAMPLED_PER_PROMPT = 1
BLIND_SAMPLED_PER_PROMPT = 4
RETENTION_SAMPLED_PER_PROMPT = 1
MAX_SEQUENCES_PER_BATCH = 8

MANIFEST_SCHEMA = "chk4-cp38-direct-grpo-checkpoint-probe-manifest-v2"
RESULT_SCHEMA = "chk4-cp38-direct-grpo-checkpoint-probe-result-v1"
SUMMARY_SCHEMA = "chk4-cp38-direct-grpo-checkpoint-probe-summary-v1"
SELECTION_SCHEMA = "chk4-cp38-direct-grpo-checkpoint-selection-v1"

VALIDATION_PARTITION_SALT = "chk4-grpo-cp-selection-validation-v1"
RETENTION_PARTITION_SALT = "chk4-grpo-cp-selection-retention-v1"
STAGE_SEED_SALTS = {
    SELECTION_STAGE: "chk4-grpo-cp-selection-generation-v1",
    BLIND_STAGE: "chk4-grpo-cp-blind-generation-v1",
    RETENTION_STAGE: "chk4-grpo-cp-retention-generation-v1",
}


class CheckpointProbeError(RuntimeError):
    """A checkpoint probe input or result failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointProbeError(message)


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
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointProbeError(f"invalid {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        _require(bool(raw), f"blank row in {label}:{line_number}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CheckpointProbeError(
                f"invalid JSON in {label}:{line_number}"
            ) from exc
        _require(isinstance(value, dict), f"non-object row in {label}:{line_number}")
        rows.append(value)
    return rows


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    _require(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _file_record(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"unsafe file: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _stage_seed(stage: str) -> int:
    _require(stage in STAGES, f"unsupported stage: {stage}")
    return int(_sha256_text(STAGE_SEED_SALTS[stage])[:8], 16) & 0x7FFFFFFF


def _panel_contract(
    *, population_role: str, sample_ids: Sequence[str], sampled_per_prompt: int, seed: int
) -> dict[str, Any]:
    _require(sampled_per_prompt > 0, "sampled completions must be positive")
    greedy_batch_size = MAX_SEQUENCES_PER_BATCH
    sampled_batch_size = max(1, MAX_SEQUENCES_PER_BATCH // sampled_per_prompt)
    return {
        "population_role": population_role,
        "sample_ids": list(sample_ids),
        "greedy_per_prompt": 1,
        "sampled_per_prompt": sampled_per_prompt,
        "seed": seed,
        "max_sequences_per_generation_call": MAX_SEQUENCES_PER_BATCH,
        "batches": {
            "greedy": _chunks(list(sample_ids), greedy_batch_size),
            "sampled": _chunks(list(sample_ids), sampled_batch_size),
        },
    }


def _salted_order(sample_ids: Sequence[str], salt: str) -> list[str]:
    return sorted(sample_ids, key=lambda value: (_sha256_text(f"{salt}\0{value}"), value))


def partition_validation(
    samples: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    by_direction: dict[str, list[str]] = {"hold": [], "hike": [], "cut": []}
    for sample_id, sample in samples.items():
        by_direction[str(sample["direction"])].append(sample_id)
    _require(
        {key: len(value) for key, value in by_direction.items()}
        == {"hold": 3, "hike": 10, "cut": 0},
        "validation direction population drift",
    )
    hold = _salted_order(by_direction["hold"], VALIDATION_PARTITION_SALT + ":hold")
    hike = _salted_order(by_direction["hike"], VALIDATION_PARTITION_SALT + ":hike")
    blind = [hold[0], *hike[:3]]
    selection = [*hold[1:], *hike[3:]]
    selection = _salted_order(selection, VALIDATION_PARTITION_SALT + ":selection")
    blind = _salted_order(blind, VALIDATION_PARTITION_SALT + ":blind")
    _require(len(selection) == 9 and len(blind) == 4, "validation panel size drift")
    _require(not set(selection).intersection(blind), "validation panels overlap")
    _require(set(selection).union(blind) == set(samples), "validation coverage drift")
    return selection, blind


def partition_retention(samples: Mapping[str, Mapping[str, Any]]) -> list[str]:
    selected: list[str] = []
    for direction in ("hold", "hike", "cut"):
        candidates = [
            sample_id
            for sample_id, sample in samples.items()
            if sample["direction"] == direction
        ]
        ordered = _salted_order(
            candidates, f"{RETENTION_PARTITION_SALT}:{direction}"
        )
        _require(len(ordered) >= 2, f"too few train samples for {direction}")
        selected.extend(ordered[:2])
    return _salted_order(selected, RETENTION_PARTITION_SALT + ":panel")


def _load_config() -> dict[str, Any]:
    _require(
        sha256_file(TRAINING_CONFIG) == TRAINING_CONFIG_SHA256,
        "training config SHA-256 drift",
    )
    try:
        import yaml

        value = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CheckpointProbeError("cannot load training config") from exc
    _require(isinstance(value, dict), "training config must be a mapping")
    expected = {
        "dataset_chk4_role": CHK4_PRE2009_GRPO_ROLE,
        "dataset_chk4_release_manifest_sha256": RELEASE_MANIFEST_SHA256,
        "dataset_train_split": "train",
        "dataset_test_split": "validation",
        "max_completion_length": MAX_NEW_TOKENS,
        "num_generations": 4,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "reward_funcs": ["decision_dense_v3"],
        "use_vllm": False,
    }
    for key, expected_value in expected.items():
        _require(value.get(key) == expected_value, f"training config {key} drift")
    _require(
        (REPO_ROOT / str(value.get("model_name_or_path"))).resolve()
        == BASE_MODEL.resolve(),
        "training base model drift",
    )
    return value


def _release_file_record(
    release_manifest: Mapping[str, Any], relative: str
) -> tuple[Path, dict[str, Any]]:
    files = release_manifest.get("files")
    record = files.get(relative) if isinstance(files, Mapping) else None
    _require(
        isinstance(record, Mapping) and record.get("path") == relative,
        f"release file record missing: {relative}",
    )
    path = (RELEASE_ROOT / relative).resolve()
    _require(path.is_file() and not path.is_symlink(), f"unsafe release file: {path}")
    _require(sha256_file(path) == record.get("sha256"), f"release hash drift: {relative}")
    return path, dict(record)


def _valid_target(direction: object, magnitude: object) -> bool:
    return bool(
        direction in {"hold", "hike", "cut"}
        and isinstance(magnitude, int)
        and not isinstance(magnitude, bool)
        and (
            (direction == "hold" and magnitude == 0)
            or (direction in {"hike", "cut"} and magnitude in {25, 50, 75, 100})
        )
    )


def _load_split(
    split: str,
    release_manifest: Mapping[str, Any],
    tokenizer: Any,
    system_prompt: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    _require(split in {"train", "validation"}, "probe may load only train/validation")
    physical_path, physical_record = _release_file_record(
        release_manifest, f"decision_grpo/{split}.jsonl"
    )
    unique_path, unique_record = _release_file_record(
        release_manifest, f"manifests/unique/{split}.jsonl"
    )
    physical_rows = _read_jsonl(physical_path, label=f"{split} physical rows")
    unique_rows = _read_jsonl(unique_path, label=f"{split} unique rows")
    _require(physical_record.get("rows") == len(physical_rows), f"{split} row drift")
    _require(unique_record.get("rows") == len(unique_rows), f"{split} unique row drift")
    unique_by_id: dict[str, dict[str, Any]] = {}
    for row in unique_rows:
        sample_id = row.get("sample_id")
        _require(isinstance(sample_id, str) and sample_id, f"{split} missing sample ID")
        _require(sample_id not in unique_by_id, f"{split} duplicate sample ID")
        _require(row.get("split") == split, f"{split} split marker drift")
        _require(
            _valid_target(row.get("direction"), row.get("magnitude_bp")),
            f"{split} target drift: {sample_id}",
        )
        unique_by_id[sample_id] = row
    physical_by_id: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(physical_rows, start=1):
        _require(
            set(row) == {"direction", "magnitude_bp", "prompt", "sample_id"},
            f"{split} physical schema drift",
        )
        sample_id = str(row.get("sample_id"))
        _require(sample_id in unique_by_id, f"{split} physical ID not unique-bound")
        prompt = row.get("prompt")
        _require(isinstance(prompt, str) and prompt, f"{split} empty prompt")
        unique = unique_by_id[sample_id]
        _require(
            _sha256_text(prompt) == unique.get("prompt_sha256"),
            f"{split} prompt hash drift: {sample_id}",
        )
        _require(
            row.get("direction") == unique.get("direction")
            and row.get("magnitude_bp") == unique.get("magnitude_bp"),
            f"{split} physical target drift: {sample_id}",
        )
        current = physical_by_id.setdefault(
            sample_id, {"prompt": prompt, "line_numbers": []}
        )
        _require(current["prompt"] == prompt, f"{split} prompt variants: {sample_id}")
        current["line_numbers"].append(line_number)
    _require(set(physical_by_id) == set(unique_by_id), f"{split} ID coverage drift")
    samples: dict[str, dict[str, Any]] = {}
    for sample_id, unique in unique_by_id.items():
        physical = physical_by_id[sample_id]
        _require(
            len(physical["line_numbers"]) == int(unique.get("repeat_factor", 0)),
            f"{split} repeat factor drift: {sample_id}",
        )
        prompt_ids = generation_probe._prompt_ids(
            tokenizer,
            generation_probe._messages(system_prompt, physical["prompt"]),
        )
        samples[sample_id] = {
            "sample_id": sample_id,
            "split": split,
            "population_role": "independent_validation" if split == "validation" else "train_retention_only",
            "direction": unique["direction"],
            "magnitude_bp": unique["magnitude_bp"],
            "prompt_sha256": unique["prompt_sha256"],
            "prompt_token_count": len(prompt_ids),
            "physical_line_numbers": physical["line_numbers"],
        }
    return samples, {
        "physical": {"path": str(physical_path), **physical_record},
        "unique": {"path": str(unique_path), **unique_record},
    }


def _candidate_binding(step: int) -> dict[str, Any]:
    path = ADAPTER_ROOT / f"checkpoint-{step}"
    _require(path.is_dir() and not path.is_symlink(), f"missing checkpoint-{step}")
    state_path = path / "trainer_state.json"
    state = _read_json(state_path, label=f"checkpoint-{step} trainer state")
    _require(state.get("global_step") == step, f"checkpoint-{step} step drift")
    adapter_binding, adapter_config = reward_probe._fingerprint_adapter(
        path, base_model=BASE_MODEL.resolve()
    )
    composition = {
        "base_model_sha256": BASE_MODEL_SHA256,
        "adapter_config_sha256": adapter_binding["files"]["adapter_config.json"]["sha256"],
        "adapter_weights_sha256": adapter_binding["files"]["adapter_model.safetensors"]["sha256"],
    }
    return {
        "step": step,
        "path": str(path.resolve()),
        "epoch": state.get("epoch"),
        "trainer_state": _file_record(state_path),
        "adapter": adapter_binding,
        "adapter_config": adapter_config,
        "effective_model_sha256": _sha256_text(canonical_json(composition)),
        "composition": composition,
    }


def build_manifest() -> dict[str, Any]:
    config = _load_config()
    manifest_path = RELEASE_ROOT / "release_manifest.json"
    _require(
        sha256_file(manifest_path) == RELEASE_MANIFEST_SHA256,
        "release manifest SHA-256 drift",
    )
    release_manifest = _read_json(manifest_path, label="release manifest")
    base_fingerprint = fingerprint_artifact_path(BASE_MODEL)
    _require(
        base_fingerprint.get("sha256") == BASE_MODEL_SHA256,
        "exact cp38 base fingerprint drift",
    )
    system_prompt = config.get("system_prompt")
    _require(isinstance(system_prompt, str) and system_prompt, "missing system prompt")
    runtime_binding = verify_chk4_pre2009_augmented_release(
        dataset_dir=RELEASE_ROOT / "decision_grpo",
        manifest_path=manifest_path,
        expected_manifest_sha256=RELEASE_MANIFEST_SHA256,
        dataset_role=CHK4_PRE2009_GRPO_ROLE,
        system_prompt=system_prompt,
        model_path=BASE_MODEL,
    )
    _require(
        runtime_binding.get("test_verified_but_not_loaded") is True,
        "release verifier did not preserve sealed test",
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(BASE_MODEL), local_files_only=True, trust_remote_code=False
    )
    validation, validation_files = _load_split(
        "validation", release_manifest, tokenizer, system_prompt
    )
    train, train_files = _load_split("train", release_manifest, tokenizer, system_prompt)
    selection_ids, blind_ids = partition_validation(validation)
    retention_ids = partition_retention(train)
    samples = {
        sample_id: (validation.get(sample_id) or train[sample_id])
        for sample_id in [*selection_ids, *blind_ids, *retention_ids]
    }
    candidates = [_candidate_binding(step) for step in CANDIDATE_STEPS]
    return {
        "schema_version": MANIFEST_SCHEMA,
        "purpose": "select_best_chk4_cp38_direct_grpo_checkpoint",
        "sealed_test_policy": {
            "path": "decision_grpo/test.jsonl",
            "status": "verified_by_release_verifier_but_never_loaded_for_selection",
        },
        "training": {
            "root": str(TRAIN_ROOT.resolve()),
            "config": _file_record(TRAINING_CONFIG),
            "completed_steps": 468,
        },
        "release": {
            "root": str(RELEASE_ROOT.resolve()),
            "manifest": _file_record(manifest_path),
            "validation_files": validation_files,
            "train_files": train_files,
            "validation_population": {"rows": 13, "hold": 3, "hike": 10, "cut": 0},
        },
        "base_model": base_fingerprint,
        "tokenizer_bundle": runtime_binding["tokenizer_binding"],
        "prompt_contract": {
            "system_prompt_sha256": _sha256_text(system_prompt),
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        },
        "candidate_steps": list(CANDIDATE_STEPS),
        "candidates": candidates,
        "samples": samples,
        "panels": {
            SELECTION_STAGE: _panel_contract(
                population_role="independent_validation_selection",
                sample_ids=selection_ids,
                sampled_per_prompt=SELECTION_SAMPLED_PER_PROMPT,
                seed=_stage_seed(SELECTION_STAGE),
            ),
            BLIND_STAGE: _panel_contract(
                population_role="independent_validation_blind",
                sample_ids=blind_ids,
                sampled_per_prompt=BLIND_SAMPLED_PER_PROMPT,
                seed=_stage_seed(BLIND_STAGE),
            ),
            RETENTION_STAGE: _panel_contract(
                population_role="train_retention_only_not_independent_evaluation",
                sample_ids=retention_ids,
                sampled_per_prompt=RETENTION_SAMPLED_PER_PROMPT,
                seed=_stage_seed(RETENTION_STAGE),
            ),
        },
        "ranking_policy": {
            "selection_top_two": (
                "eligible(delivery>=0.75,cap<=0.25,periodic=0), then reward_mean, "
                "exact_rate, direction_correct_rate, delivery_rate, lower cap, earlier step"
            ),
            "final": (
                "blind eligible(delivery>=0.75,cap<=0.25,periodic=0) and retention "
                "has direction-correct output for hold/hike/cut; then blind reward_mean, "
                "combined validation reward_mean, blind exact_rate, combined exact_rate, "
                "retention reward_mean, earlier step"
            ),
            "baseline": "exact cp38 is an anchor and is never selectable",
        },
    }


def validate_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    _require(sha256_file(path) == expected_sha256, "probe manifest SHA-256 drift")
    observed = _read_json(path, label="probe manifest")
    _require(observed.get("schema_version") == MANIFEST_SCHEMA, "manifest schema drift")
    rebuilt = build_manifest()
    _require(canonical_json(observed) == canonical_json(rebuilt), "manifest replay drift")
    return observed


def _load_prompt_texts(manifest: Mapping[str, Any]) -> dict[str, str]:
    wanted = set(manifest["samples"])
    result: dict[str, str] = {}
    for split in ("validation", "train"):
        rows = _read_jsonl(
            RELEASE_ROOT / f"decision_grpo/{split}.jsonl", label=f"bound {split} rows"
        )
        for row in rows:
            sample_id = str(row.get("sample_id"))
            if sample_id not in wanted:
                continue
            prompt = str(row.get("prompt"))
            if sample_id in result:
                _require(result[sample_id] == prompt, f"prompt variants: {sample_id}")
            result[sample_id] = prompt
    _require(set(result) == wanted, "bound prompt population incomplete")
    for sample_id, prompt in result.items():
        _require(
            _sha256_text(prompt) == manifest["samples"][sample_id]["prompt_sha256"],
            f"prompt hash drift: {sample_id}",
        )
    return result


def _chunks(values: Sequence[str], size: int) -> list[list[str]]:
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def _metric_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(rows), "cannot summarize empty probe rows")
    rewards = [float(row["decision_dense_v3_reward"]) for row in rows]
    _require(all(math.isfinite(value) for value in rewards), "non-finite reward")
    total = len(rows)
    def rate(field: str) -> float:
        return round(sum(bool(row.get(field)) for row in rows) / total, 8)
    directions: dict[str, Any] = {}
    for direction in ("hold", "hike", "cut"):
        subset = [row for row in rows if row["target_direction"] == direction]
        if subset:
            directions[direction] = {
                "cases": len(subset),
                "reward_mean": round(
                    statistics.fmean(float(row["decision_dense_v3_reward"]) for row in subset), 8
                ),
                "direction_correct_rate": round(
                    sum(bool(row["decision_direction_correct"]) for row in subset)
                    / len(subset),
                    8,
                ),
                "exact_rate": round(
                    sum(bool(row["decision_exact"]) for row in subset) / len(subset), 8
                ),
            }
    return {
        "cases": total,
        "reward_mean": round(statistics.fmean(rewards), 8),
        "reward_std": round(statistics.pstdev(rewards), 8),
        "reward_nonzero_rate": round(sum(value > 1e-12 for value in rewards) / total, 8),
        "direction_correct_rate": rate("decision_direction_correct"),
        "exact_rate": rate("decision_exact"),
        "delivery_valid_rate": rate("delivery_valid"),
        "strict_json_rate": rate("strict_json"),
        "eos_rate": rate("hit_eos"),
        "cap_rate": rate("cap_reached"),
        "periodic_tail_count": sum(bool(row.get("strict_periodic_tail")) for row in rows),
        "completion_tokens_mean": round(
            statistics.fmean(int(row["completion_token_count"]) for row in rows), 3
        ),
        "directions": directions,
    }


def _selection_eligible(metrics: Mapping[str, Any]) -> bool:
    return bool(
        float(metrics["delivery_valid_rate"]) >= 0.75
        and float(metrics["cap_rate"]) <= 0.25
        and int(metrics["periodic_tail_count"]) == 0
    )


def selection_rank_key(summary: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = summary["metrics"]
    step = int(summary["checkpoint_step"])
    return (
        int(_selection_eligible(metrics)),
        float(metrics["reward_mean"]),
        float(metrics["exact_rate"]),
        float(metrics["direction_correct_rate"]),
        float(metrics["delivery_valid_rate"]),
        -float(metrics["cap_rate"]),
        -step,
    )


def _retention_pass(metrics: Mapping[str, Any]) -> bool:
    directions = metrics.get("directions")
    return bool(
        isinstance(directions, Mapping)
        and all(
            isinstance(directions.get(direction), Mapping)
            and float(directions[direction]["direction_correct_rate"]) > 0
            for direction in ("hold", "hike", "cut")
        )
    )


def final_rank_key(
    *, step: int, blind: Mapping[str, Any], combined: Mapping[str, Any], retention: Mapping[str, Any]
) -> tuple[Any, ...]:
    eligible = _selection_eligible(blind) and _retention_pass(retention)
    return (
        int(eligible),
        float(blind["reward_mean"]),
        float(combined["reward_mean"]),
        float(blind["exact_rate"]),
        float(combined["exact_rate"]),
        float(retention["reward_mean"]),
        -step,
    )


def _model_spec(manifest: Mapping[str, Any], label: str) -> tuple[Path | None, Mapping[str, Any] | None, dict[str, Any]]:
    if label == "baseline_cp38":
        return None, None, {
            "model_label": label,
            "checkpoint_step": None,
            "base_model_sha256": BASE_MODEL_SHA256,
            "effective_model_sha256": BASE_MODEL_SHA256,
        }
    _require(label.startswith("checkpoint_"), f"invalid model label: {label}")
    step = int(label.removeprefix("checkpoint_"))
    candidate = next(
        (item for item in manifest["candidates"] if item["step"] == step), None
    )
    _require(candidate is not None, f"unbound checkpoint: {step}")
    return Path(candidate["path"]), candidate["adapter_config"], {
        "model_label": label,
        "checkpoint_step": step,
        "base_model_sha256": BASE_MODEL_SHA256,
        "adapter_sha256": candidate["adapter"]["directory"]["sha256"],
        "effective_model_sha256": candidate["effective_model_sha256"],
    }


def _load_model(adapter: Path | None, adapter_config: Mapping[str, Any] | None) -> Any:
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
        str(BASE_MODEL),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        quantization_config=quantization,
        device_map={"": 0},
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    if adapter is not None:
        _require(adapter_config is not None, "adapter config missing")
        model = reward_probe._attach_local_peft_adapter(
            model, adapter_path=adapter, adapter_config=adapter_config
        )
    model.eval()
    model.config.use_cache = True
    return model


def _result_row(
    *, manifest: Mapping[str, Any], manifest_sha256: str, stage: str,
    model_provenance: Mapping[str, Any], sample_id: str, generation_mode: str,
    generation_index: int, batch_index: int, raw_ids: Sequence[int],
    eos_token_ids: set[int], tokenizer: Any,
) -> dict[str, Any]:
    normalized_ids, normalization = eos_probe.normalize_at_first_bound_eos(
        raw_ids, eos_token_ids
    )
    completion = generation_probe.decode_completion_preserving_boundary(
        tokenizer, normalized_ids, eos_token_ids
    )
    metrics = generation_probe.analyze_completion(
        text=completion,
        generated_token_ids=normalized_ids,
        eos_token_ids=eos_token_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        tail_tokens=TAIL_TOKENS,
    )
    sample = manifest["samples"][sample_id]
    replay = reward_probe.replay_decision_dense_v3(
        completion,
        {"direction": sample["direction"], "magnitude_bp": sample["magnitude_bp"]},
        hit_eos=bool(metrics["hit_eos"]),
        cap_reached=bool(metrics["cap_reached"]),
    )
    return {
        "schema_version": RESULT_SCHEMA,
        "stage": stage,
        **model_provenance,
        "sample_id": sample_id,
        "split": sample["split"],
        "population_role": sample["population_role"],
        "target_direction": sample["direction"],
        "target_magnitude_bp": sample["magnitude_bp"],
        "prompt_token_count": sample["prompt_token_count"],
        "generation_mode": generation_mode,
        "generation_index": generation_index,
        "batch_index": batch_index,
        "stage_seed": manifest["panels"][stage]["seed"],
        "generation_parameters": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": generation_mode == "sampled",
            "temperature": TEMPERATURE if generation_mode == "sampled" else 0.0,
            "top_p": TOP_P if generation_mode == "sampled" else 1.0,
        },
        "generated_token_ids_raw_padded": list(raw_ids),
        "generated_token_ids_first_eos_inclusive": normalized_ids,
        "batched_padding_normalization": normalization,
        "completion": completion,
        "completion_sha256": _sha256_text(completion),
        **metrics,
        "decision_dense_v3_reward": replay["reward"],
        "decision_prediction": replay["prediction"],
        "decision_direction_correct": replay["direction_correct"],
        "decision_exact": replay["exact"],
        "decision_rejection_reason": replay["rejection_reason"],
        "decision_forced_zero_reason": replay["forced_zero_reason"],
        "response_format": replay["response_format"],
        "strict_json": replay["strict_json"],
        "fenced_json": replay["fenced_json"],
        "manifest_sha256": manifest_sha256,
    }


def run_panel(
    *, manifest: Mapping[str, Any], manifest_sha256: str, stage: str,
    model_label: str, tokenizer: Any, prompts: Mapping[str, str], output_dir: Path,
) -> dict[str, Any]:
    _require(stage in STAGES, f"invalid stage: {stage}")
    _require(not output_dir.exists(), f"probe output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    adapter, adapter_config, model_provenance = _model_spec(manifest, model_label)
    launch = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "initializing",
        "created_at_utc": _utc_now(),
        "stage": stage,
        "model": model_provenance,
        "manifest_sha256": manifest_sha256,
        "runtime": {"python": sys.executable, "python_version": platform.python_version(), "cuda_visible_devices": "1"},
    }
    _write_exclusive_json(output_dir / "launch.json", launch)
    result_path = output_dir / "results.jsonl"
    rows: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc)
    model: Any = None
    try:
        import torch
        import transformers

        model = _load_model(adapter, adapter_config)
        eos_value = getattr(model.generation_config, "eos_token_id", None)
        if eos_value is None:
            eos_value = tokenizer.eos_token_id
        eos_ids = set(generation_probe._normalize_eos_ids(eos_value))
        _require(bool(eos_ids), "model/tokenizer has no EOS")
        sample_ids = list(manifest["panels"][stage]["sample_ids"])
        sampled_per = int(manifest["panels"][stage]["sampled_per_prompt"])

        def render(sample_id: str) -> str:
            return tokenizer.apply_chat_template(
                generation_probe._messages(
                    _load_config()["system_prompt"], prompts[sample_id]
                ),
                tokenize=False,
                add_generation_prompt=True,
            )

        def persist(handle: IO[str], row: Mapping[str, Any]) -> None:
            handle.write(canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        torch.cuda.reset_peak_memory_stats()
        with result_path.open("x", encoding="utf-8") as handle:
            for mode in ("greedy", "sampled"):
                transformers.set_seed(int(manifest["panels"][stage]["seed"]))
                num_return = 1 if mode == "greedy" else sampled_per
                batches = manifest["panels"][stage]["batches"][mode]
                flattened = [sample_id for batch in batches for sample_id in batch]
                _require(flattened == sample_ids, f"{stage}/{mode} batch layout drift")
                for batch_index, batch_ids in enumerate(batches):
                    _require(
                        len(batch_ids) * num_return <= MAX_SEQUENCES_PER_BATCH,
                        "generation call exceeds the bound sequence batch",
                    )
                    encoded = tokenizer(
                        [render(sample_id) for sample_id in batch_ids],
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    input_ids = encoded["input_ids"].to("cuda:0")
                    attention_mask = encoded["attention_mask"].to("cuda:0")
                    for prompt_index, sample_id in enumerate(batch_ids):
                        observed_tokens = int(attention_mask[prompt_index].sum().item())
                        _require(
                            observed_tokens == manifest["samples"][sample_id]["prompt_token_count"],
                            f"prompt token count drift: {sample_id}",
                        )
                    kwargs: dict[str, Any] = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "do_sample": mode == "sampled",
                        "num_return_sequences": num_return,
                        "pad_token_id": tokenizer.pad_token_id,
                        "eos_token_id": sorted(eos_ids),
                        "use_cache": True,
                    }
                    if mode == "sampled":
                        kwargs.update(temperature=TEMPERATURE, top_p=TOP_P)
                    with torch.inference_mode():
                        sequences = model.generate(**kwargs)
                    _require(
                        len(sequences) == len(batch_ids) * num_return,
                        "batched generation count drift",
                    )
                    prompt_width = int(input_ids.shape[1])
                    for prompt_index, sample_id in enumerate(batch_ids):
                        for generation_index in range(num_return):
                            sequence_index = prompt_index * num_return + generation_index
                            raw_ids = sequences[sequence_index, prompt_width:].tolist()
                            row = _result_row(
                                manifest=manifest,
                                manifest_sha256=manifest_sha256,
                                stage=stage,
                                model_provenance=model_provenance,
                                sample_id=sample_id,
                                generation_mode=mode,
                                generation_index=(0 if mode == "greedy" else generation_index + 1),
                                batch_index=batch_index,
                                raw_ids=raw_ids,
                                eos_token_ids=eos_ids,
                                tokenizer=tokenizer,
                            )
                            rows.append(row)
                            persist(handle, row)
                    del input_ids, attention_mask, sequences
        expected = len(sample_ids) * (1 + sampled_per)
        _require(len(rows) == expected, "panel result count drift")
        metrics = _metric_block(rows)
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "status": "complete",
            "stage": stage,
            **model_provenance,
            "cases": len(rows),
            "metrics": metrics,
            "eligible_selection_quality": _selection_eligible(metrics),
            "elapsed_seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 3),
            "runtime": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
            },
            "results": {"path": str(result_path.resolve()), "rows": len(rows), "sha256": sha256_file(result_path)},
            "manifest_sha256": manifest_sha256,
        }
        _write_exclusive_json(output_dir / "summary.json", summary)
        return summary
    except Exception as exc:
        if not (output_dir / "failure.json").exists():
            _write_exclusive_json(
                output_dir / "failure.json",
                {**launch, "status": "failed", "error_type": type(exc).__name__, "error": str(exc), "completed_cases": len(rows)},
            )
        raise
    finally:
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _read_result_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = Path(str(summary["results"]["path"]))
    _require(sha256_file(path) == summary["results"]["sha256"], "result hash drift")
    return _read_jsonl(path, label="probe results")


def run_suite(
    *, manifest_path: Path, manifest_sha256: str, output_root: Path
) -> dict[str, Any]:
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "probe requires physical GPU1")
    _require(output_root.resolve() == manifest_path.parent.resolve(), "output/manifest root mismatch")
    _require(not (output_root / "runs").exists(), "probe runs already exist")
    manifest = validate_manifest(manifest_path, manifest_sha256)
    prompts = _load_prompt_texts(manifest)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(BASE_MODEL), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    selection_summaries: list[dict[str, Any]] = []
    labels = ["baseline_cp38", *[f"checkpoint_{step}" for step in CANDIDATE_STEPS]]
    for label in labels:
        summary = run_panel(
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            stage=SELECTION_STAGE,
            model_label=label,
            tokenizer=tokenizer,
            prompts=prompts,
            output_dir=output_root / "runs" / SELECTION_STAGE / label,
        )
        selection_summaries.append(summary)
        print(canonical_json({"status": "panel_complete", "stage": SELECTION_STAGE, "model": label, "metrics": summary["metrics"]}), flush=True)
    candidates = [item for item in selection_summaries if item["checkpoint_step"] is not None]
    ranked = sorted(candidates, key=selection_rank_key, reverse=True)
    top_two = [int(item["checkpoint_step"]) for item in ranked[:2]]
    ranking_artifact = {
        "schema_version": SELECTION_SCHEMA,
        "status": "selection_panel_complete",
        "top_two": top_two,
        "ranking": [
            {"checkpoint_step": item["checkpoint_step"], "rank_key": list(selection_rank_key(item)), "metrics": item["metrics"]}
            for item in ranked
        ],
        "baseline": next(item for item in selection_summaries if item["checkpoint_step"] is None),
        "manifest_sha256": manifest_sha256,
    }
    _write_exclusive_json(output_root / "selection_ranking.json", ranking_artifact)

    followups: dict[int, dict[str, dict[str, Any]]] = {}
    for step in top_two:
        label = f"checkpoint_{step}"
        followups[step] = {}
        for stage in (BLIND_STAGE, RETENTION_STAGE):
            summary = run_panel(
                manifest=manifest,
                manifest_sha256=manifest_sha256,
                stage=stage,
                model_label=label,
                tokenizer=tokenizer,
                prompts=prompts,
                output_dir=output_root / "runs" / stage / label,
            )
            followups[step][stage] = summary
            print(canonical_json({"status": "panel_complete", "stage": stage, "model": label, "metrics": summary["metrics"]}), flush=True)

    final_rows: dict[int, dict[str, Any]] = {}
    for step in top_two:
        selection_summary = next(item for item in ranked if item["checkpoint_step"] == step)
        blind_summary = followups[step][BLIND_STAGE]
        retention_summary = followups[step][RETENTION_STAGE]
        validation_rows = [
            *_read_result_rows(selection_summary),
            *_read_result_rows(blind_summary),
        ]
        combined = _metric_block(validation_rows)
        blind = blind_summary["metrics"]
        retention = retention_summary["metrics"]
        final_rows[step] = {
            "checkpoint_step": step,
            "selection": selection_summary["metrics"],
            "blind": blind,
            "combined_validation": combined,
            "retention_train_only": retention,
            "blind_quality_pass": _selection_eligible(blind),
            "retention_direction_pass": _retention_pass(retention),
            "rank_key": list(final_rank_key(step=step, blind=blind, combined=combined, retention=retention)),
        }
    final_ranked = sorted(
        final_rows.values(), key=lambda item: tuple(item["rank_key"]), reverse=True
    )
    winner = final_ranked[0]
    selected = bool(winner["rank_key"][0])
    final = {
        "schema_version": SELECTION_SCHEMA,
        "status": "selected" if selected else "provisional_no_candidate_passed_all_gates",
        "selected_checkpoint_step": winner["checkpoint_step"] if selected else None,
        "best_observed_checkpoint_step": winner["checkpoint_step"],
        "top_two": top_two,
        "final_ranking": final_ranked,
        "validation_limitations": {
            "rows": 13,
            "direction_distribution": {"hold": 3, "hike": 10, "cut": 0},
            "cut_evidence": "train_retention_only_not_independent",
            "sealed_test_used": False,
        },
        "manifest": {"path": str(manifest_path.resolve()), "sha256": manifest_sha256},
        "created_at_utc": _utc_now(),
    }
    _write_exclusive_json(output_root / "selection.json", final)
    return final


def status(output_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA,
        "output_root": str(output_root.resolve()),
        "manifest": "present" if (output_root / "probe_manifest.json").is_file() else "missing",
        "completed_panels": [],
        "selection": None,
    }
    for path in sorted((output_root / "runs").glob("*/*/summary.json")):
        value = _read_json(path, label="panel summary")
        result["completed_panels"].append(
            {"stage": value["stage"], "model_label": value["model_label"], "cases": value["cases"], "metrics": value["metrics"]}
        )
    if (output_root / "selection.json").is_file():
        result["selection"] = _read_json(output_root / "selection.json", label="selection")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--manifest-sha256", required=True)
    run.add_argument("--output-root", type=Path, required=True)
    check = sub.add_parser("status")
    check.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            root = args.output_root.expanduser().resolve()
            _require(not root.exists(), f"output root already exists: {root}")
            root.mkdir(parents=True, exist_ok=False)
            manifest = build_manifest()
            path = root / "probe_manifest.json"
            _write_exclusive_json(path, manifest)
            print(canonical_json({"status": "prepared", "path": str(path), "sha256": sha256_file(path), "candidate_steps": list(CANDIDATE_STEPS)}))
            return 0
        if args.command == "status":
            print(json.dumps(status(args.output_root), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        final = run_suite(
            manifest_path=args.manifest.expanduser().resolve(),
            manifest_sha256=args.manifest_sha256,
            output_root=args.output_root.expanduser().resolve(),
        )
        print(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if final["status"] == "selected" else 2
    except (CheckpointProbeError, OSError, RuntimeError, ValueError) as exc:
        print(canonical_json({"status": "blocked", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
