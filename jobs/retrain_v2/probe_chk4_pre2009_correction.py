"""Create-only static generation screens for the chk4 correction SFT.

The checkpoint-selection screen is deliberately fixed before training: it uses
the four historical GRPO-smoke prompts in two ordered batches, seeds the
generation process once with 31416, requests four sampled completions per
prompt, and uses the cap-1536 generation contract.  It is a *static generation
screen*, not a bitwise replay of GRPO (Trainer/LoRA initialization can consume
RNG state).  Checkpoints are considered only in the pre-registered order
2 -> 4 -> 6; the first fully passing checkpoint is locked.

After locking, a separate blind confirmation uses (a) the sealed pilot's exact
three prompts and seeds 20260810..20260824 with a versioned 1024 -> 1536
single-delta contract and (b) two pre-2009 hold/hike prompts selected by the
pre-registered minimum salted-ID rule.  Confirmation failure invalidates the
correction run and cannot be used to choose a later checkpoint.

Every raw completion, generated token ID, EOS/cap metric, replayed reward, and
model/release hash is written create-only.  This utility never trains or merges
a model.
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

from jobs.retrain_v2 import probe_chk4_decision_grpo_stratified as legacy_probe
from jobs.retrain_v2 import probe_chk4_decision_sft_generation as base_probe
from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.trainer.dataset_release import (
    CHK4_PRE2009_GRPO_ROLE,
    CHK4_STUDENT_SYSTEM_PROMPT,
    verify_chk4_pre2009_augmented_release,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PARENT_RELEASE = (
    REPO_ROOT
    / "dataset/processed/retrain_v2/"
    "chk4_decision_pre2009_train_balanced_v1_20260811"
)
PARENT_MANIFEST_SHA256 = (
    "5141e00569b94e4af96f030f46176e9cd409199a6f26ac1c63a2ce335fe1e2d6"
)
SOURCE_CONFIG = REPO_ROOT / (
    "configs/retrain_v2/"
    "chk4_decision_grpo_from_pre2009_cp38_selected_"
    "smoke_cap1536_v2_20260811.yaml"
)
SOURCE_CONFIG_SHA256 = (
    "289cd52194e3d9dbd1f615fdfad53c584ec6dfdab7abc13e6d23d94280df162a"
)
SELECTED_CP38_MODEL = REPO_ROOT / (
    "output/training/retrain_v2/"
    "chk4_from_chk1_cp200_sft_grpo_pre2009_balanced_lr1e5_steps39_v1_20260811/"
    "selected_sft_checkpoints/checkpoint-38/merged/chk4_sft"
)
SELECTED_CP38_MODEL_SHA256 = (
    "715bdb763043e7fda49e4a189b601d5e070fefbcecd14388bde41dbf7c165dd0"
)
LEGACY_PILOT_MANIFEST = REPO_ROOT / (
    "output/training/retrain_v2/"
    "chk4_from_chk1_cp200_sft_grpo_pre2009_balanced_lr1e5_steps39_v1_20260811/"
    "selected_sft_checkpoints/pre2009_train_stratified_manifest_v1.json"
)
LEGACY_PILOT_MANIFEST_SHA256 = (
    "99853a20f246684b0e03a90087af5dcb3ca23732bf5db06fd5433b09516611a7"
)

MANIFEST_SCHEMA = "chk4-pre2009-correction-static-screen-manifest-v1"
RESULT_SCHEMA = "chk4-pre2009-correction-static-screen-result-v1"
SUMMARY_SCHEMA = "chk4-pre2009-correction-static-screen-summary-v1"
SELECTION_PURPOSE = "checkpoint_selection"
CONFIRMATION_PURPOSE = "locked_checkpoint_blind_confirmation"
MAX_NEW_TOKENS = 1536
TAIL_TOKENS = 256
SELECTION_SEED = 31416
NUM_RETURN_SEQUENCES = 4
TEMPERATURE = 0.7
TOP_P = 0.9
DERIVED_HOLDOUT_SALT = "corrective-holdout-v1"

HISTORICAL_HELDOUT_IDS = (
    "dec-0ad00e5fe7cb7333beafdab1",
    "dec-bb7d9c61358a339a9a1f4aa5",
    "dec-29de02cb43945c20f838fcf7",
    "dec-4dfab939b0a910c949931475",
    "dec-8b8d55ea065b662a19cefe88",
)
PRE2009_HOLD_ID = "dec-0bc683809351ff9da117027b"
PRE2009_HIKE_ID = "dec-62f0a67d390dc9a866611dac"
ALL_HELDOUT_IDS = (*HISTORICAL_HELDOUT_IDS, PRE2009_HOLD_ID, PRE2009_HIKE_ID)
SELECTION_BATCH_IDS = (
    (
        "dec-bb7d9c61358a339a9a1f4aa5",
        "dec-4dfab939b0a910c949931475",
    ),
    (
        "dec-29de02cb43945c20f838fcf7",
        "dec-8b8d55ea065b662a19cefe88",
    ),
)
PRE2009_CONFIRMATION_BATCH_IDS = ((PRE2009_HOLD_ID, PRE2009_HIKE_ID),)


class CorrectionProbeError(RuntimeError):
    """A static-screen input, output, or gate failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorrectionProbeError(message)


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


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise CorrectionProbeError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorrectionProbeError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CorrectionProbeError(f"{label} must contain an object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise CorrectionProbeError(f"{label} is missing or unsafe: {path}")
    rows: list[dict[str, Any]] = []
    try:
        raw_rows = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CorrectionProbeError(f"cannot read {label}: {path}") from exc
    for line_number, raw in enumerate(raw_rows, 1):
        _require(bool(raw), f"{label}:{line_number}: blank row")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CorrectionProbeError(
                f"{label}:{line_number}: invalid JSON"
            ) from exc
        _require(isinstance(value, dict), f"{label}:{line_number}: not an object")
        rows.append(value)
    return rows


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise CorrectionProbeError(f"refusing to overwrite artifact: {path}")
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


def _validate_source_config() -> dict[str, Any]:
    _require(
        sha256_file(SOURCE_CONFIG) == SOURCE_CONFIG_SHA256,
        "cap-1536 source config hash drift",
    )
    try:
        import yaml

        config = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CorrectionProbeError("cannot load cap-1536 source config") from exc
    _require(isinstance(config, dict), "source config must be a mapping")
    expected = {
        "dataset_chk4_role": CHK4_PRE2009_GRPO_ROLE,
        "dataset_chk4_release_manifest_sha256": PARENT_MANIFEST_SHA256,
        "max_completion_length": MAX_NEW_TOKENS,
        "num_generations": NUM_RETURN_SEQUENCES,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "use_vllm": False,
        "reward_funcs": ["decision_dense_v3"],
    }
    for key, value in expected.items():
        _require(config.get(key) == value, f"source config {key} drift")
    return config


def _verify_parent_release(model_path: Path) -> Mapping[str, Any]:
    config = _validate_source_config()
    try:
        return verify_chk4_pre2009_augmented_release(
            dataset_dir=PARENT_RELEASE / "decision_grpo",
            manifest_path=PARENT_RELEASE / "release_manifest.json",
            expected_manifest_sha256=PARENT_MANIFEST_SHA256,
            dataset_role=CHK4_PRE2009_GRPO_ROLE,
            system_prompt=str(config["system_prompt"]),
            model_path=model_path,
        )
    except Exception as exc:
        raise CorrectionProbeError(f"parent release replay failed: {exc}") from exc


def _verify_correction_release(
    manifest_path: Path, expected_sha256: str
) -> Mapping[str, Any]:
    try:
        from jobs.retrain_v2.materialize_chk4_pre2009_correction_release import (
            ALL_HELDOUT_IDS as RELEASE_HELDOUT_IDS,
        )
    except ImportError:
        # The materializer intentionally exposes HELDOUT_SAMPLE_IDS; retain a
        # clearer failure if a review-time rename has not yet landed.
        RELEASE_HELDOUT_IDS = ()
    try:
        from jobs.retrain_v2.materialize_chk4_pre2009_correction_release import (
            HELDOUT_SAMPLE_IDS,
            verify_release,
        )

        verified = verify_release(
            manifest_path.expanduser().resolve().parent,
            expected_manifest_sha256=expected_sha256,
        )
    except Exception as exc:
        raise CorrectionProbeError(f"correction release replay failed: {exc}") from exc
    release_ids = tuple(HELDOUT_SAMPLE_IDS or RELEASE_HELDOUT_IDS)
    _require(release_ids == ALL_HELDOUT_IDS, "correction heldout ID order drift")
    contract = verified.get("heldout_contract")
    _require(isinstance(contract, Mapping), "correction heldout_contract is missing")
    heldout_path = manifest_path.parent / "manifests/heldout_probe.jsonl"
    heldout_rows = _read_jsonl(heldout_path, label="correction heldout manifest")
    _require(
        tuple(str(row.get("sample_id")) for row in heldout_rows) == ALL_HELDOUT_IDS,
        "correction heldout manifest ID drift",
    )
    return {
        "path": str(manifest_path.resolve()),
        "sha256": expected_sha256,
        "heldout_contract": dict(contract),
        "heldout_contract_sha256": _sha256_text(_canonical_json(contract)),
        "heldout_manifest": {
            "path": str(heldout_path.resolve()),
            "sha256": sha256_file(heldout_path),
            "rows": len(heldout_rows),
        },
    }


def _derive_pre2009_ids(source_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    historical = set(HISTORICAL_HELDOUT_IDS)
    result: dict[str, Any] = {}
    for direction, expected_id in (
        ("hold", PRE2009_HOLD_ID),
        ("hike", PRE2009_HIKE_ID),
    ):
        candidates = [
            row
            for row in source_rows
            if row.get("population_role") == "supplement"
            and row.get("direction") == direction
            and row.get("sample_id") not in historical
        ]
        ranked = sorted(
            candidates,
            key=lambda row: (
                _sha256_text(
                    str(row.get("sample_id")) + "|" + DERIVED_HOLDOUT_SALT
                ),
                str(row.get("sample_id")),
            ),
        )
        _require(bool(ranked), f"no pre-2009 {direction} holdout candidates")
        chosen = ranked[0]
        _require(
            chosen.get("sample_id") == expected_id,
            f"derived pre-2009 {direction} holdout drift",
        )
        result[direction] = {
            "sample_id": expected_id,
            "salt": DERIVED_HOLDOUT_SALT,
            "selection_sha256": _sha256_text(
                expected_id + "|" + DERIVED_HOLDOUT_SALT
            ),
            "algorithm": "minimum-sha256-salted-sample-id-v1",
            "population_rows": len(ranked),
        }
    return result


def _source_records() -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    parent_manifest = _read_json(
        PARENT_RELEASE / "release_manifest.json", label="parent release manifest"
    )
    files = parent_manifest.get("files")
    _require(isinstance(files, Mapping), "parent release file inventory missing")
    source_path = PARENT_RELEASE / "manifests/source_unique_train.jsonl"
    train_path = PARENT_RELEASE / "decision_grpo/train.jsonl"
    for relative, path in (
        ("manifests/source_unique_train.jsonl", source_path),
        ("decision_grpo/train.jsonl", train_path),
    ):
        record = files.get(relative)
        _require(isinstance(record, Mapping), f"parent record missing: {relative}")
        _require(sha256_file(path) == record.get("sha256"), f"parent hash drift: {relative}")

    source_rows = _read_jsonl(source_path, label="parent unique source train")
    by_id = {str(row.get("sample_id")): row for row in source_rows}
    _require(len(by_id) == len(source_rows), "parent unique source IDs are duplicated")
    physical_rows = _read_jsonl(train_path, label="parent GRPO train")
    line_numbers: dict[str, list[int]] = {}
    prompts: dict[str, str] = {}
    for line_number, row in enumerate(physical_rows, 1):
        sample_id = str(row.get("sample_id"))
        line_numbers.setdefault(sample_id, []).append(line_number)
        prompt = str(row.get("prompt"))
        if sample_id in prompts:
            _require(prompts[sample_id] == prompt, f"prompt variants: {sample_id}")
        prompts[sample_id] = prompt
    selected: dict[str, Mapping[str, Any]] = {}
    for sample_id in ALL_HELDOUT_IDS:
        source = by_id.get(sample_id)
        _require(source is not None, f"heldout source missing: {sample_id}")
        _require(sample_id in prompts, f"heldout physical prompt missing: {sample_id}")
        _require(
            _sha256_text(prompts[sample_id]) == source.get("prompt_sha256"),
            f"heldout prompt hash drift: {sample_id}",
        )
        selected[sample_id] = {
            **source,
            "prompt": prompts[sample_id],
            "physical_line_numbers": line_numbers[sample_id],
        }
    records = {
        "unique_source": {
            "path": str(source_path.resolve()),
            "sha256": sha256_file(source_path),
            "rows": len(source_rows),
        },
        "physical_train": {
            "path": str(train_path.resolve()),
            "sha256": sha256_file(train_path),
            "rows": len(physical_rows),
        },
        "derived_pre2009_holdouts": _derive_pre2009_ids(source_rows),
    }
    return records, selected


def _tokenizer_binding() -> Mapping[str, Any]:
    _require(
        sha256_file(LEGACY_PILOT_MANIFEST) == LEGACY_PILOT_MANIFEST_SHA256,
        "legacy pilot manifest hash drift",
    )
    legacy = _read_json(LEGACY_PILOT_MANIFEST, label="legacy pilot manifest")
    tokenizer = legacy.get("tokenizer")
    _require(isinstance(tokenizer, Mapping), "legacy tokenizer binding missing")
    try:
        _, files, bundle_sha = legacy_probe._sample_tokenizer_binding(legacy)
    except Exception as exc:
        raise CorrectionProbeError(f"legacy tokenizer replay failed: {exc}") from exc
    _require(tokenizer.get("bundle_sha256") == bundle_sha, "tokenizer bundle drift")
    _require(tokenizer.get("files") == files, "tokenizer file binding drift")
    return dict(tokenizer)


def _sample_descriptor(
    source: Mapping[str, Any], *, tokenizer: Any
) -> dict[str, Any]:
    prompt_ids = base_probe._prompt_ids(
        tokenizer,
        base_probe._messages(CHK4_STUDENT_SYSTEM_PROMPT, str(source["prompt"])),
    )
    return {
        "sample_id": source["sample_id"],
        "population_role": source["population_role"],
        "direction": source["direction"],
        "magnitude_bp": source["magnitude_bp"],
        "meeting_date": source["meeting_date"],
        "prompt_sha256": source["prompt_sha256"],
        "prompt_token_count": len(prompt_ids),
        "physical_line_numbers": source["physical_line_numbers"],
    }


def build_manifest(
    *,
    purpose: str,
    correction_release_manifest: Path,
    correction_release_manifest_sha256: str,
    tokenizer: Any,
) -> dict[str, Any]:
    """Build one text-free, immutable static-screen manifest."""

    _require(
        purpose in {SELECTION_PURPOSE, CONFIRMATION_PURPOSE},
        "unsupported correction probe purpose",
    )
    _verify_parent_release(SELECTED_CP38_MODEL)
    correction = _verify_correction_release(
        correction_release_manifest, correction_release_manifest_sha256
    )
    source_records, sources = _source_records()
    tokenizer_record = _tokenizer_binding()
    samples = {
        sample_id: _sample_descriptor(source, tokenizer=tokenizer)
        for sample_id, source in sources.items()
    }
    _require(
        max(int(sample["prompt_token_count"]) for sample in samples.values()) <= 2560,
        "heldout prompt exceeds max_prompt_length",
    )

    if purpose == SELECTION_PURPOSE:
        panels: list[dict[str, Any]] = [
            {
                "name": "static_smoke_selection",
                "kind": "batched_sampled",
                "seed": SELECTION_SEED,
                "seed_once_before_all_batches": True,
                "batches": [list(batch) for batch in SELECTION_BATCH_IDS],
                "num_return_sequences": NUM_RETURN_SEQUENCES,
            }
        ]
        sample_ids = [item for batch in SELECTION_BATCH_IDS for item in batch]
        gate = {
            "step1": {
                "hold_correct_nonzero_min": 2,
                "hike_correct_nonzero_min": 1,
                "each_direction_reward_std_gt": 0.0,
                "cap_count_max": 2,
                "boundary_count_min": 6,
                "periodic_tail_max": 0,
            },
            "static_step2_prior": {
                "hold_correct_nonzero_min": 1,
                "cut_correct_nonzero_min": 1,
                "each_direction_reward_std_gt": 0.0,
                "cap_count_max": 2,
                "boundary_count_min": 6,
                "periodic_tail_max": 0,
            },
        }
        predecessor = None
    else:
        legacy = _read_json(LEGACY_PILOT_MANIFEST, label="legacy pilot manifest")
        legacy_samples = legacy.get("samples")
        _require(isinstance(legacy_samples, list), "legacy pilot samples missing")
        expected_legacy_ids = (
            HISTORICAL_HELDOUT_IDS[0],
            HISTORICAL_HELDOUT_IDS[3],
            HISTORICAL_HELDOUT_IDS[4],
        )
        _require(
            tuple(str(row.get("sample_id")) for row in legacy_samples)
            == expected_legacy_ids,
            "legacy pilot sample order drift",
        )
        panels = [
            {
                "name": "versioned_legacy_stratified_cap1536",
                "kind": "legacy_seed_matrix",
                "samples": [
                    {
                        "sample_id": row["sample_id"],
                        "greedy_seed": row["greedy_seed"],
                        "sample_seeds": row["sample_seeds"],
                    }
                    for row in legacy_samples
                ],
            },
            {
                "name": "pre2009_blind",
                "kind": "batched_sampled",
                "seed": SELECTION_SEED,
                "seed_once_before_all_batches": True,
                "batches": [list(batch) for batch in PRE2009_CONFIRMATION_BATCH_IDS],
                "num_return_sequences": NUM_RETURN_SEQUENCES,
            },
        ]
        sample_ids = [*expected_legacy_ids, PRE2009_HOLD_ID, PRE2009_HIKE_ID]
        gate = {
            "legacy_stratified": {
                "hold_sampled_correct_nonzero_min": 3,
                "hike_sampled_correct_nonzero_min": 1,
                "cut_sampled_correct_nonzero_min": 1,
                "each_direction_sampled_reward_std_gt": 0.0,
                "overall_cap_rate_max": 0.25,
                "overall_boundary_rate_min": 0.75,
                "periodic_tail_max": 0,
            },
            "pre2009_blind": {
                "hold_correct_nonzero_min": 2,
                "hike_correct_nonzero_min": 1,
                "each_direction_reward_std_gt": 0.0,
                "cap_count_max": 2,
                "boundary_count_min": 6,
                "periodic_tail_max": 0,
            },
        }
        predecessor = {
            "manifest": {
                "path": str(LEGACY_PILOT_MANIFEST.resolve()),
                "sha256": LEGACY_PILOT_MANIFEST_SHA256,
            },
            "single_delta": {
                "claim": "only_max_new_tokens_changes",
                "max_new_tokens": {"from": 1024, "to": MAX_NEW_TOKENS},
                "sample_ids_order_seeds_temperature_top_p_unchanged": True,
            },
        }

    return {
        "schema_version": MANIFEST_SCHEMA,
        "purpose": purpose,
        "methodology": {
            "name": "static_generation_screening",
            "not_a_bitwise_grpo_replay": True,
            "authoritative_gate": "fresh_two_step_cap1536_grpo_smoke_after_exact_merge",
        },
        "release": {
            "path": str(PARENT_RELEASE.resolve()),
            "manifest_sha256": PARENT_MANIFEST_SHA256,
            "role": CHK4_PRE2009_GRPO_ROLE,
            "split": "train",
            **source_records,
        },
        "correction_release": correction,
        "source_config": {
            "path": str(SOURCE_CONFIG.resolve()),
            "sha256": SOURCE_CONFIG_SHA256,
        },
        "base_model": fingerprint_artifact_path(SELECTED_CP38_MODEL),
        "tokenizer": tokenizer_record,
        "generation": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "num_return_sequences": NUM_RETURN_SEQUENCES,
            "tail_tokens": TAIL_TOKENS,
            "raw_completion_and_token_ids_required": True,
        },
        "sample_order": sample_ids,
        "samples": {sample_id: samples[sample_id] for sample_id in sample_ids},
        "panels": panels,
        "quality_gate": gate,
        "predecessor": predecessor,
    }


def _validate_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    _require(sha256_file(path) == expected_sha256, "probe manifest hash drift")
    manifest = _read_json(path, label="probe manifest")
    _require(manifest.get("schema_version") == MANIFEST_SCHEMA, "probe schema drift")
    purpose = str(manifest.get("purpose"))
    tokenizer_path = Path(str(manifest.get("tokenizer", {}).get("path")))
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True, trust_remote_code=False
        )
    except Exception as exc:
        raise CorrectionProbeError(f"cannot reload bound tokenizer: {exc}") from exc
    rebuilt = build_manifest(
        purpose=purpose,
        correction_release_manifest=Path(
            str(manifest.get("correction_release", {}).get("path"))
        ),
        correction_release_manifest_sha256=str(
            manifest.get("correction_release", {}).get("sha256")
        ),
        tokenizer=tokenizer,
    )
    _require(
        _canonical_json(rebuilt) == _canonical_json(manifest),
        "probe manifest no longer matches sealed inputs",
    )
    return manifest


def _model_source(
    *, model_path: Path | None, base_model_path: Path | None, adapter_path: Path | None
) -> Mapping[str, Any]:
    try:
        source = legacy_probe._prepare_model_source(
            model_path=model_path,
            base_model_path=base_model_path,
            adapter_path=adapter_path,
        )
    except Exception as exc:
        raise CorrectionProbeError(f"model source validation failed: {exc}") from exc
    base = source["model_fingerprint"]
    _require(
        base.get("sha256") == SELECTED_CP38_MODEL_SHA256,
        "static screen base model is not the selected exact cp38 merge",
    )
    return source


def _load_bound_prompts(manifest: Mapping[str, Any]) -> dict[str, str]:
    rows = _read_jsonl(
        Path(str(manifest["release"]["physical_train"]["path"])),
        label="bound physical train",
    )
    wanted = set(manifest["sample_order"])
    prompts: dict[str, str] = {}
    for row in rows:
        sample_id = str(row.get("sample_id"))
        if sample_id in wanted:
            prompt = str(row.get("prompt"))
            if sample_id in prompts:
                _require(prompts[sample_id] == prompt, f"prompt variant: {sample_id}")
            prompts[sample_id] = prompt
    _require(set(prompts) == wanted, "bound prompt population is incomplete")
    for sample_id, prompt in prompts.items():
        _require(
            _sha256_text(prompt) == manifest["samples"][sample_id]["prompt_sha256"],
            f"bound prompt hash drift: {sample_id}",
        )
    return prompts


def _metric_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rewards = [float(row["decision_dense_v3_reward"]) for row in rows]
    return {
        "cases": len(rows),
        "correct_nonzero_count": sum(
            bool(row["decision_direction_correct"])
            and float(row["decision_dense_v3_reward"]) > 0
            for row in rows
        ),
        "nonzero_count": sum(value > 0 for value in rewards),
        "reward_mean": round(statistics.fmean(rewards), 8),
        "reward_std": round(statistics.pstdev(rewards), 8) if len(rewards) > 1 else 0.0,
        "cap_count": sum(bool(row["cap_reached"]) for row in rows),
        "boundary_count": sum(int(row["think_boundary_count"]) == 1 for row in rows),
        "periodic_tail_count": sum(bool(row["strict_periodic_tail"]) for row in rows),
    }


def summarize_results(
    results: Sequence[Mapping[str, Any]], *, manifest: Mapping[str, Any], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    purpose = manifest["purpose"]
    reasons: list[str] = []
    panels: dict[str, Any] = {}
    if purpose == SELECTION_PURPOSE:
        expected_rows = 16
        _require(len(results) == expected_rows, "selection screen requires 16 rows")
        for batch_index, name in enumerate(("step1", "static_step2_prior")):
            rows = [row for row in results if row.get("batch_index") == batch_index]
            _require(len(rows) == 8, f"{name} requires 8 rows")
            directions: dict[str, Any] = {}
            for direction in sorted({str(row["target_direction"]) for row in rows}):
                directions[direction] = _metric_block(
                    [row for row in rows if row["target_direction"] == direction]
                )
            delivery = _metric_block(rows)
            panels[name] = {"directions": directions, "delivery": delivery}
        step1 = panels["step1"]
        if step1["directions"]["hold"]["correct_nonzero_count"] < 2:
            reasons.append("step1:hold_correct_nonzero_lt_2")
        if step1["directions"]["hike"]["correct_nonzero_count"] < 1:
            reasons.append("step1:hike_correct_nonzero_lt_1")
        for direction in ("hold", "hike"):
            if step1["directions"][direction]["reward_std"] <= 0:
                reasons.append(f"step1:{direction}_reward_zero_std")
        step2 = panels["static_step2_prior"]
        for direction in ("hold", "cut"):
            if step2["directions"][direction]["correct_nonzero_count"] < 1:
                reasons.append(f"step2:{direction}_correct_nonzero_lt_1")
            if step2["directions"][direction]["reward_std"] <= 0:
                reasons.append(f"step2:{direction}_reward_zero_std")
        for name in ("step1", "static_step2_prior"):
            delivery = panels[name]["delivery"]
            if delivery["cap_count"] > 2:
                reasons.append(f"{name}:cap_count_gt_2")
            if delivery["boundary_count"] < 6:
                reasons.append(f"{name}:boundary_count_lt_6")
            if delivery["periodic_tail_count"]:
                reasons.append(f"{name}:periodic_tail_present")
    else:
        legacy_rows = [row for row in results if row.get("panel") == "legacy_stratified"]
        pre_rows = [row for row in results if row.get("panel") == "pre2009_blind"]
        _require(len(legacy_rows) == 15, "legacy confirmation requires 15 rows")
        _require(len(pre_rows) == 8, "pre-2009 confirmation requires 8 rows")
        legacy_directions: dict[str, Any] = {}
        for direction in ("hold", "hike", "cut"):
            sampled = [
                row
                for row in legacy_rows
                if row["target_direction"] == direction
                and row["generation_mode"] == "sampled"
            ]
            _require(len(sampled) == 4, f"legacy {direction} sampled rows drift")
            legacy_directions[direction] = _metric_block(sampled)
        legacy_all = _metric_block(legacy_rows)
        panels["legacy_stratified"] = {
            "directions": legacy_directions,
            "overall": legacy_all,
        }
        minima = {"hold": 3, "hike": 1, "cut": 1}
        for direction, minimum in minima.items():
            block = legacy_directions[direction]
            if block["correct_nonzero_count"] < minimum:
                reasons.append(f"legacy:{direction}_correct_nonzero_lt_{minimum}")
            if block["reward_std"] <= 0:
                reasons.append(f"legacy:{direction}_reward_zero_std")
        if legacy_all["cap_count"] / 15 > 0.25:
            reasons.append("legacy:cap_rate_gt_0.25")
        if legacy_all["boundary_count"] / 15 < 0.75:
            reasons.append("legacy:boundary_rate_lt_0.75")
        if legacy_all["periodic_tail_count"]:
            reasons.append("legacy:periodic_tail_present")

        pre_directions = {
            direction: _metric_block(
                [row for row in pre_rows if row["target_direction"] == direction]
            )
            for direction in ("hold", "hike")
        }
        pre_all = _metric_block(pre_rows)
        panels["pre2009_blind"] = {
            "directions": pre_directions,
            "delivery": pre_all,
        }
        for direction, minimum in (("hold", 2), ("hike", 1)):
            block = pre_directions[direction]
            if block["correct_nonzero_count"] < minimum:
                reasons.append(f"pre2009:{direction}_correct_nonzero_lt_{minimum}")
            if block["reward_std"] <= 0:
                reasons.append(f"pre2009:{direction}_reward_zero_std")
        if pre_all["cap_count"] > 2:
            reasons.append("pre2009:cap_count_gt_2")
        if pre_all["boundary_count"] < 6:
            reasons.append("pre2009:boundary_count_lt_6")
        if pre_all["periodic_tail_count"]:
            reasons.append("pre2009:periodic_tail_present")

    if any(
        not math.isfinite(float(row.get("decision_dense_v3_reward", math.nan)))
        for row in results
    ):
        reasons.append("nonfinite_reward")
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "quality_status": "passed" if not reasons else "failed",
        "purpose": purpose,
        "quality_gate": {"contract": manifest["quality_gate"], "reasons": reasons},
        "cases": len(results),
        "panels": panels,
        "provenance": dict(provenance),
    }


def _result_row(
    *,
    manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
    sample_id: str,
    completion: str,
    generated_ids: Sequence[int],
    panel: str,
    generation_mode: str,
    generation_index: int,
    seed: int,
    batch_index: int | None,
) -> dict[str, Any]:
    sample = manifest["samples"][sample_id]
    metrics = base_probe.analyze_completion(
        text=completion,
        generated_token_ids=list(generated_ids),
        eos_token_ids=set(provenance["eos_token_ids"]),
        max_new_tokens=MAX_NEW_TOKENS,
        tail_tokens=TAIL_TOKENS,
    )
    replay = legacy_probe.replay_decision_dense_v3(
        completion,
        {"direction": sample["direction"], "magnitude_bp": sample["magnitude_bp"]},
        hit_eos=bool(metrics["hit_eos"]),
        cap_reached=bool(metrics["cap_reached"]),
    )
    return {
        "schema_version": RESULT_SCHEMA,
        "purpose": manifest["purpose"],
        "panel": panel,
        "batch_index": batch_index,
        "sample_id": sample_id,
        "population_role": sample["population_role"],
        "target_direction": sample["direction"],
        "target_magnitude_bp": sample["magnitude_bp"],
        "generation_mode": generation_mode,
        "generation_index": generation_index,
        "seed": seed,
        "generation_parameters": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE if generation_mode == "sampled" else 0.0,
            "top_p": TOP_P if generation_mode == "sampled" else 1.0,
        },
        "generated_token_ids": list(generated_ids),
        "completion": completion,
        "completion_sha256": _sha256_text(completion),
        **metrics,
        "decision_dense_v3_reward": replay["reward"],
        "decision_dense_v3_nonzero": replay["nonzero"],
        "decision_prediction": replay["prediction"],
        "decision_direction_correct": replay["direction_correct"],
        "decision_exact": replay["exact"],
        "decision_rejection_reason": replay["rejection_reason"],
        "decision_forced_zero_reason": replay["forced_zero_reason"],
        "response_format": replay["response_format"],
        "strict_json": replay["strict_json"],
        "fenced_json": replay["fenced_json"],
        "provenance": {
            "manifest_sha256": provenance["manifest_sha256"],
            "model_sha256": provenance["model_sha256"],
            "effective_model_sha256": provenance["effective_model_sha256"],
        },
    }


def run_screen(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    output_dir: Path,
    model_label: str,
    model_path: Path | None = None,
    base_model_path: Path | None = None,
    adapter_path: Path | None = None,
    load_in_4bit: bool = True,
) -> Mapping[str, Any]:
    """Run one immutable selection or confirmation screen on GPU1."""

    _require(not output_dir.exists() and not output_dir.is_symlink(), "output exists")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "screen requires GPU1 only")
    manifest = _validate_manifest(manifest_path.resolve(), manifest_sha256)
    tokenizer_binding = manifest["tokenizer"]
    tokenizer_path = Path(str(tokenizer_binding["path"])).resolve()
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True, trust_remote_code=False
        )
    except Exception as exc:
        raise CorrectionProbeError(f"cannot load tokenizer: {exc}") from exc
    prompts = _load_bound_prompts(manifest)
    source = _model_source(
        model_path=model_path,
        base_model_path=base_model_path,
        adapter_path=adapter_path,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "results.jsonl"
    launch_path = output_dir / "launch.json"
    source_provenance = source["provenance"]
    provenance: dict[str, Any] = {
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha256},
        "release_manifest_sha256": PARENT_MANIFEST_SHA256,
        "correction_release_manifest_sha256": manifest["correction_release"]["sha256"],
        "model_source": source_provenance,
        "model_sha256": source["model_fingerprint"]["sha256"],
        "effective_model_sha256": source["effective_model_fingerprint"]["sha256"],
        "manifest_sha256": manifest_sha256,
    }
    launch = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "initializing",
        "model_label": model_label,
        "purpose": manifest["purpose"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "cuda_visible_devices": "1",
        },
        "provenance": provenance,
    }
    _write_exclusive_json(launch_path, launch)
    model: Any = None
    results: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc)
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "one visible CUDA device required")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        quantization = None
        if load_in_4bit:
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.bfloat16,
            )
        model = AutoModelForCausalLM.from_pretrained(
            str(source["load_path"]),
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            quantization_config=quantization,
            device_map={"": 0},
            local_files_only=True,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        if source["mode"] == legacy_probe.PEFT_ADAPTER_MODE:
            model = legacy_probe._attach_local_peft_adapter(
                model,
                adapter_path=source["adapter_path"],
                adapter_config=source["adapter_config"],
            )
        model.eval()
        model.config.use_cache = True
        loaded_model = model
        eos_value = getattr(model.generation_config, "eos_token_id", None)
        if eos_value is None:
            eos_value = tokenizer.eos_token_id
        eos_ids = sorted(base_probe._normalize_eos_ids(eos_value))
        _require(bool(eos_ids), "model/tokenizer has no EOS IDs")
        provenance["eos_token_ids"] = eos_ids
        torch.cuda.reset_peak_memory_stats()

        def render(sample_id: str) -> str:
            messages = base_probe._messages(
                CHK4_STUDENT_SYSTEM_PROMPT, prompts[sample_id]
            )
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        def persist_row(handle: IO[str], row: dict[str, Any]) -> None:
            handle.write(_canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        def run_batched_panel(
            panel: Mapping[str, Any], panel_name: str, handle: IO[str]
        ) -> None:
            transformers.set_seed(int(panel["seed"]))
            for batch_index, sample_ids in enumerate(panel["batches"]):
                encoded = tokenizer(
                    [render(str(sample_id)) for sample_id in sample_ids],
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                input_ids = encoded["input_ids"].to("cuda:0")
                attention_mask = encoded["attention_mask"].to("cuda:0")
                with torch.inference_mode():
                    sequences = loaded_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        num_return_sequences=NUM_RETURN_SEQUENCES,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=eos_ids,
                        use_cache=True,
                    )
                prompt_width = input_ids.shape[1]
                _require(
                    len(sequences) == len(sample_ids) * NUM_RETURN_SEQUENCES,
                    "batched generation row count drift",
                )
                for prompt_index, sample_id in enumerate(sample_ids):
                    for generation_index in range(NUM_RETURN_SEQUENCES):
                        sequence_index = (
                            prompt_index * NUM_RETURN_SEQUENCES + generation_index
                        )
                        generated = sequences[sequence_index, prompt_width:].tolist()
                        completion = base_probe.decode_completion_preserving_boundary(
                            tokenizer, generated, set(eos_ids)
                        )
                        row = _result_row(
                            manifest=manifest,
                            provenance=provenance,
                            sample_id=str(sample_id),
                            completion=completion,
                            generated_ids=generated,
                            panel=panel_name,
                            generation_mode="sampled",
                            generation_index=generation_index + 1,
                            seed=int(panel["seed"]),
                            batch_index=batch_index,
                        )
                        results.append(row)
                        persist_row(handle, row)

        def run_legacy_panel(panel: Mapping[str, Any], handle: IO[str]) -> None:
            for sample_contract in panel["samples"]:
                sample_id = str(sample_contract["sample_id"])
                cases = [
                    ("greedy", 0, int(sample_contract["greedy_seed"])),
                    *[
                        ("sampled", index, int(seed))
                        for index, seed in enumerate(
                            sample_contract["sample_seeds"], start=1
                        )
                    ],
                ]
                prompt_ids = base_probe._prompt_ids(
                    tokenizer,
                    base_probe._messages(
                        CHK4_STUDENT_SYSTEM_PROMPT, prompts[sample_id]
                    ),
                )
                for mode, index, seed in cases:
                    transformers.set_seed(seed)
                    input_ids = torch.tensor(
                        [prompt_ids], dtype=torch.long, device="cuda:0"
                    )
                    attention_mask = torch.ones_like(input_ids)
                    kwargs: dict[str, Any] = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "do_sample": mode == "sampled",
                        "pad_token_id": tokenizer.pad_token_id,
                        "eos_token_id": eos_ids,
                        "use_cache": True,
                    }
                    if mode == "sampled":
                        kwargs.update(temperature=TEMPERATURE, top_p=TOP_P)
                    with torch.inference_mode():
                        sequence = loaded_model.generate(**kwargs)[0]
                    generated = sequence[input_ids.shape[1] :].tolist()
                    completion = base_probe.decode_completion_preserving_boundary(
                        tokenizer, generated, set(eos_ids)
                    )
                    row = _result_row(
                        manifest=manifest,
                        provenance=provenance,
                        sample_id=sample_id,
                        completion=completion,
                        generated_ids=generated,
                        panel="legacy_stratified",
                        generation_mode=mode,
                        generation_index=index,
                        seed=seed,
                        batch_index=None,
                    )
                    results.append(row)
                    persist_row(handle, row)

        with result_path.open("x", encoding="utf-8") as handle:
            for panel in manifest["panels"]:
                if panel["kind"] == "batched_sampled":
                    name = (
                        "selection"
                        if manifest["purpose"] == SELECTION_PURPOSE
                        else "pre2009_blind"
                    )
                    run_batched_panel(panel, name, handle)
                else:
                    run_legacy_panel(panel, handle)
        result_path.chmod(0o444)
        clean_results = list(results)
        summary = summarize_results(
            clean_results, manifest=manifest, provenance=provenance
        )
        summary.update(
            {
                "model_label": model_label,
                "elapsed_seconds": round(
                    (datetime.now(timezone.utc) - started).total_seconds(), 3
                ),
                "runtime": {
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "peak_allocated_gib": round(
                        torch.cuda.max_memory_allocated() / 1024**3, 3
                    ),
                    "peak_reserved_gib": round(
                        torch.cuda.max_memory_reserved() / 1024**3, 3
                    ),
                },
                "results": {
                    "path": str(result_path.resolve()),
                    "sha256": sha256_file(result_path),
                    "rows": len(clean_results),
                    "raw_completions_and_token_ids_landed": True,
                },
            }
        )
        _write_exclusive_json(output_dir / "summary.json", summary)
        output_dir.chmod(0o555)
        return summary
    except Exception as exc:
        if result_path.exists():
            result_path.chmod(0o444)
        _write_exclusive_json(
            output_dir / "failure.json",
            {
                **launch,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_cases": len(results),
            },
        )
        output_dir.chmod(0o555)
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--purpose", choices=(SELECTION_PURPOSE, CONFIRMATION_PURPOSE), required=True)
    prepare.add_argument("--correction-release-manifest", type=Path, required=True)
    prepare.add_argument("--correction-release-manifest-sha256", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path)
    source.add_argument("--base-model", type=Path)
    run.add_argument("--adapter", type=Path)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--manifest-sha256", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--model-label", required=True)
    run.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            tokenizer_binding = _tokenizer_binding()
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_binding["path"]),
                local_files_only=True,
                trust_remote_code=False,
            )
            manifest = build_manifest(
                purpose=args.purpose,
                correction_release_manifest=args.correction_release_manifest,
                correction_release_manifest_sha256=args.correction_release_manifest_sha256,
                tokenizer=tokenizer,
            )
            _write_exclusive_json(args.output, manifest)
            result = {
                "status": "prepared",
                "purpose": args.purpose,
                "path": str(args.output.resolve()),
                "sha256": sha256_file(args.output),
            }
        else:
            summary = run_screen(
                manifest_path=args.manifest,
                manifest_sha256=args.manifest_sha256,
                output_dir=args.output_dir,
                model_label=args.model_label,
                model_path=args.model,
                base_model_path=args.base_model,
                adapter_path=args.adapter,
                load_in_4bit=args.load_in_4bit,
            )
            result = summary
        print(_canonical_json(result))
        return 0 if result.get("quality_status", "passed") == "passed" else 2
    except (CorrectionProbeError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
