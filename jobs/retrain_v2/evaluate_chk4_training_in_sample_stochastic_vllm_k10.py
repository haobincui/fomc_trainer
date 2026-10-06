"""Stochastic Decision-release training-meeting diagnostic with four models.

This create-only workflow evaluates the 211 *unique* meetings in the frozen
Decision training split.  It deliberately does not evaluate the 312-row SFT
or GRPO sampling schedule: repeated training exposures are not independent
meetings and are never admitted to the diagnostic population.

Each of four paper-facing artifacts receives ten fresh stochastic completions
per meeting at ``temperature=0.6`` and ``top_p=0.9``.  Request seeds are shared
across models within every meeting--replicate block.  Two independent vLLM
DP=1 workers use the two physical GPUs.  The resulting 8,440 rows support only
an in-sample, post-hoc fit diagnostic; they are not an external, held-out, or
generalization evaluation.

Raw-token persistence, crash-safe resume, two-GPU smoke testing, authorization,
material replay, and parser/reward replay reuse the already audited stochastic
Decision evaluator.  This module supplies a distinct population, four-model
binding, create-only manifest, training-specific result materialization, and a
training-only exact-action report by policy direction and magnitude.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.retrain_v2 import evaluate_chk4_external_core8_panels as source
from jobs.retrain_v2 import (
    evaluate_chk4_external_core8_stochastic_vllm_k10 as base,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = REPO_ROOT / (
    "dataset/processed/retrain_v2/"
    "chk4_decision_pre2009_train_balanced_v1_20260811"
)
RELEASE_MANIFEST = RELEASE_ROOT / "release_manifest.json"
SOURCE_UNIQUE_TRAIN = RELEASE_ROOT / "manifests/source_unique_train.jsonl"
UNIQUE_TRAIN = RELEASE_ROOT / "manifests/unique/train.jsonl"
PHYSICAL_GRPO_TRAIN = RELEASE_ROOT / "decision_grpo/train.jsonl"
PHYSICAL_SFT_TRAIN = RELEASE_ROOT / "decision_sft/train.jsonl"
TRAINING_CONFIG = REPO_ROOT / (
    "configs/retrain_v2/"
    "chk4_decision_grpo_from_pre2009_cp38_direct_full_no_smoke_v1_20260812.yaml"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    "output/evaluation/retrain_v2/"
    "chk4_training_in_sample_stochastic_n211_vllm_t06_p09_k10_v1_20260826"
)

RELEASE_MANIFEST_SHA256 = (
    "5141e00569b94e4af96f030f46176e9cd409199a6f26ac1c63a2ce335fe1e2d6"
)
SOURCE_UNIQUE_TRAIN_SHA256 = (
    "f316f29ccfa1ed957a1c3aad4d1a6c017833b124094cf4b94a2a8c9805b3f204"
)
UNIQUE_TRAIN_SHA256 = (
    "e6f4ac900c21b9afce8dfbfaa0a20916efeff267b9c938db89e0efb8e69d58eb"
)
PHYSICAL_GRPO_TRAIN_SHA256 = (
    "e8c9e17892fb31a02f811c6aee222b1bd23486e60164c9426228444743eab046"
)
PHYSICAL_SFT_TRAIN_SHA256 = (
    "d034bb0c40cf3945f5c5ab984d4e3b1bc7f2b0b4bbe4b4d32bd0b955ed1b89e9"
)
TRAINING_CONFIG_SHA256 = (
    "222c70d76823681b7ff07971704c38fd153abcac4620a4e0fab4fce00d77f06f"
)

PANEL = "decision_training_unique_n211"
MEETING_COUNT = 211
REPLICATES = 10
MODEL_COUNT = 4
EXPECTED_ROWS = MEETING_COUNT * REPLICATES * MODEL_COUNT
EXPECTED_DIRECTION_COUNTS = {"cut": 23, "hold": 156, "hike": 32}
EXPECTED_DIRECTION_MAGNITUDE_COUNTS = {
    ("cut", 25): 12,
    ("cut", 50): 8,
    ("cut", 75): 2,
    ("cut", 100): 1,
    ("hold", 0): 156,
    ("hike", 25): 29,
    ("hike", 50): 3,
}
CHK0_RUNTIME_PAYLOAD_SHA256 = (
    "97480b0fa2614940b2a4277a828620585485495c6742ba463634869adab7cabc"
)
CHK0_RUNTIME_FILES: dict[str, tuple[int, str]] = {
    "config.json": (
        828,
        "c6162f33d194369772137ecdd4bdcac4cf3fed4f40607d694b94bcbb8e5dc39f",
    ),
    "generation_config.json": (
        181,
        "cd5194726d1e8f7361a8c8425fc11d33ade5e69de1fd7615eb23fae5601af68b",
    ),
    "model-00001-of-000002.safetensors": (
        8_667_826_246,
        "7e6b24744354ef4ba547547cc758339090f46ba2da917845cfc69f7d4ded9edb",
    ),
    "model-00002-of-000002.safetensors": (
        7_392_730_108,
        "19fb83b79bd0d06d49b7cf6f86b83f5183cd292aa2c028d633ca4fceac1ae742",
    ),
    "model.safetensors.index.json": (
        24_240,
        "83bdf4be4bb1a054ff315cd804554c48a88036226fdfbc65bee84ff562fea32a",
    ),
    "tokenizer.json": (
        9_084_480,
        "b9c9eb63a8e03059914880f918cd28a880dec8b6e15e4461e1ff677e3743dbb8",
    ),
    "tokenizer_config.json": (
        3_072,
        "5a773d1f7a8716f53f414040cbbf94ddf5f55f4f00bc3b4cc8fd8d6b64369777",
    ),
}
REPLICATE_SEEDS = tuple(202_608_260 + index for index in range(REPLICATES))
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 202_608_261

EVALUATION_ID = "chk4-decision-training-in-sample-n211-vllm-t06-p09-k10-four-model-v1"
MANIFEST_SCHEMA = "chk4-training-in-sample-stochastic-vllm-k10-manifest-v1"
TOKEN_LEDGER_SCHEMA = "chk4-training-in-sample-stochastic-vllm-k10-token-ledger-v1"
AUTHORIZATION_SCHEMA = "chk4-training-in-sample-stochastic-vllm-k10-authorization-v1"
SMOKE_RECEIPT_SCHEMA = "chk4-training-in-sample-stochastic-vllm-k10-smoke-receipt-v1"
RAW_ROW_SCHEMA = "chk4-training-in-sample-stochastic-vllm-k10-raw-row-v1"
RESULT_ROW_SCHEMA = "chk4-training-in-sample-stochastic-vllm-k10-result-row-v1"
CHUNK_RECEIPT_SCHEMA = "chk4-training-in-sample-stochastic-vllm-k10-chunk-receipt-v1"
WORKER_RECEIPT_SCHEMA = "chk4-training-in-sample-stochastic-vllm-k10-worker-receipt-v1"
RUN_RECEIPT_SCHEMA = "chk4-training-in-sample-stochastic-vllm-k10-run-receipt-v1"
SUMMARY_SCHEMA = "chk4-training-in-sample-exact-action-summary-v1"

MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "label": "model_chk0",
        "display_name": "Model chk-0",
        "paper_model": "Model chk-0",
        "checkpoint_step": None,
        "training_state": "pretrained_base_model",
        "path": REPO_ROOT / "models/DeepSeek-R1-Distill-Llama-8B",
        "sha256": CHK0_RUNTIME_PAYLOAD_SHA256,
        "decision_training_exposure": False,
    },
    {
        "label": "model_chk1_cp200",
        "display_name": "Model chk-1 cp200",
        "paper_model": "Model chk-1",
        "checkpoint_step": 200,
        "training_state": "analysis_sft",
        "path": REPO_ROOT
        / (
            "output/training/retrain_v2/"
            "chk1_clean_v2_lr1e6_selected_cp200_for_chk2_20260810/merged/chk1"
        ),
        "sha256": "0989b94792f8ab6377e2979aaedeb37010b9a2c5e05c77a459b4e5cda5b806e3",
        "decision_training_exposure": False,
    },
    {
        "label": "model_chk3_sft_cp38",
        "display_name": "Model chk-3 SFT cp38",
        "paper_model": "Model chk-3",
        "checkpoint_step": 38,
        "training_state": "decision_sft_pre_grpo",
        "path": REPO_ROOT
        / (
            "output/training/retrain_v2/"
            "chk4_from_chk1_cp200_sft_grpo_pre2009_balanced_lr1e5_steps39_v1_20260811/"
            "selected_sft_checkpoints/checkpoint-38/merged/chk4_sft"
        ),
        "sha256": "715bdb763043e7fda49e4a189b601d5e070fefbcecd14388bde41dbf7c165dd0",
        "decision_training_exposure": True,
    },
    {
        "label": "model_chk3_grpo_cp450",
        "display_name": "Model chk-3 GRPO cp450",
        "paper_model": "Model chk-3",
        "checkpoint_step": 450,
        "training_state": "decision_grpo_best_observed_exploratory_candidate",
        "path": REPO_ROOT
        / (
            "output/training/retrain_v2/"
            "chk4_from_pre2009_cp38_direct_grpo_full_no_smoke_v1_20260812/"
            "selected_grpo_checkpoints/checkpoint-450/merged/chk4_grpo"
        ),
        "sha256": "92cb337a4d36486185074a1a99574860ae011663f6679a18d53b822543dee0a5",
        "decision_training_exposure": True,
    },
)

ORIGINAL_BASE_MATERIAL_RESULT = base._material_result


def _source_record(path: Path, *, rows: int) -> dict[str, Any]:
    record = base._file_record(path)
    base._require(record.get("rows") == rows, f"source row count drift: {path}")
    return record


def _row_seed(sample_id: str, replicate_id: int) -> int:
    base._require(0 <= replicate_id < REPLICATES, "invalid replicate ID")
    preimage = f"{EVALUATION_ID}:{REPLICATE_SEEDS[replicate_id]}:{sample_id}"
    return int(hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16], 16) % (
        2**31
    )


def _configure_static() -> None:
    """Configure the frozen generation engine for this distinct profile."""

    source.MODELS = MODEL_SPECS
    base.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    base.EVALUATION_ID = EVALUATION_ID
    base.PANELS = (PANEL,)
    base.PANEL_COUNTS = {PANEL: MEETING_COUNT}
    base.REPLICATES = REPLICATES
    base.REPLICATE_SEEDS = REPLICATE_SEEDS
    base.MODEL_COUNT = MODEL_COUNT
    base.MEETING_COUNT = MEETING_COUNT
    base.PAIRED_BLOCKS = MEETING_COUNT * REPLICATES
    base.EXPECTED_RAW_ROWS = EXPECTED_ROWS
    base.EXPECTED_PANEL_ROWS = {PANEL: EXPECTED_ROWS}
    base.EXPECTED_BLOCKS_PER_SHARD = base.PAIRED_BLOCKS // base.SHARD_COUNT
    base.EXPECTED_ROWS_PER_SHARD = base.EXPECTED_BLOCKS_PER_SHARD * MODEL_COUNT
    base.SMOKE_MODEL_LABEL = "model_chk0"
    # Rank 49 is the frozen longest prompt (1,246 tokens); rank 0 provides a
    # second, ordinary-length request.  Each meeting's ten replicates split
    # five/five across the two shards, yielding ten smoke rows per GPU.
    base.SMOKE_MEETING_RANKS = (0, 49)
    base.SMOKE_ROWS = 20
    base.SMOKE_ROWS_PER_SHARD = 10
    base.BOOTSTRAP_DRAWS = BOOTSTRAP_DRAWS
    base.BOOTSTRAP_SEED = BOOTSTRAP_SEED
    base.GPU_LOCK_TEMPLATE = (
        "/tmp/fomc_trainer_chk4_training_in_sample_k10_gpu{index}.lock"
    )
    base.MANIFEST_SCHEMA = MANIFEST_SCHEMA
    base.TOKEN_LEDGER_SCHEMA = TOKEN_LEDGER_SCHEMA
    base.AUTHORIZATION_SCHEMA = AUTHORIZATION_SCHEMA
    base.SMOKE_RECEIPT_SCHEMA = SMOKE_RECEIPT_SCHEMA
    base.RAW_ROW_SCHEMA = RAW_ROW_SCHEMA
    base.RESULT_ROW_SCHEMA = RESULT_ROW_SCHEMA
    base.CHUNK_RECEIPT_SCHEMA = CHUNK_RECEIPT_SCHEMA
    base.WORKER_RECEIPT_SCHEMA = WORKER_RECEIPT_SCHEMA
    base.RUN_RECEIPT_SCHEMA = RUN_RECEIPT_SCHEMA
    base.SUMMARY_SCHEMA = SUMMARY_SCHEMA
    base.IMPLEMENTATION_FILES = {
        "training_in_sample_evaluator": Path(__file__).resolve(),
        "frozen_stochastic_engine": Path(base.__file__).resolve(),
        "source_population_loader": Path(
            REPO_ROOT / "jobs/retrain_v2/compare_chk4_all237_three_models.py"
        ),
        "final_result_parser_and_scorer": REPO_ROOT
        / "jobs/retrain_v2/chk4_cp450_final_evaluation.py",
        "generation_contract": REPO_ROOT
        / "jobs/retrain_v2/probe_chk4_decision_sft_generation.py",
        "reward_replay": REPO_ROOT
        / "jobs/retrain_v2/probe_chk4_decision_grpo_stratified.py",
        "eos_normalization": REPO_ROOT
        / "jobs/retrain_v2/probe_chk4_pre2009_correction_v2.py",
        "decision_reward": REPO_ROOT
        / "src/open_r1/trainer/rewards/reward_funcs/decision_reward_v3.py",
        "background_orchestrator": REPO_ROOT
        / "run/evaluate_chk4_training_in_sample_vllm_k10.sh",
        "focused_tests": REPO_ROOT
        / "tests/test_evaluate_chk4_training_in_sample_stochastic_vllm_k10.py",
    }
    base._row_seed = _row_seed
    base._launch_worker_pair = _launch_worker_pair
    base._material_result = _material_result
    base.validate_manifest = validate_manifest
    base._validate_authorization = validate_authorization
    base._validate_smoke_receipt = _validate_smoke_receipt
    base._validate_run_receipt = _validate_run_receipt


def _configure_token_contract(manifest: Mapping[str, Any]) -> None:
    prompt = manifest.get("prompt_contract")
    base._require(isinstance(prompt, Mapping), "prompt contract missing")
    observed_range = prompt.get("observed_prompt_token_range")
    base._require(
        isinstance(observed_range, list)
        and len(observed_range) == 2
        and all(isinstance(value, int) for value in observed_range),
        "prompt-token range missing",
    )
    base.GLOBAL_PROMPT_TOKEN_IDS_SHA256 = str(
        prompt["global_prompt_token_ids_sha256"]
    )
    base.MIN_PROMPT_TOKENS = int(observed_range[0])
    base.MAX_OBSERVED_PROMPT_TOKENS = int(observed_range[1])


def _validate_release_sources() -> dict[str, Any]:
    base._require(
        base._sha256_file(RELEASE_MANIFEST) == RELEASE_MANIFEST_SHA256,
        "release manifest drift",
    )
    release = base._read_json(RELEASE_MANIFEST, label="Decision release manifest")
    base._require(
        release.get("quality_status") == "passed"
        and release.get("immutable") is True
        and release.get("training_ready") is True
        and release.get("unique_split_counts", {}).get("train") == MEETING_COUNT,
        "Decision release is not the frozen training-ready N=211 release",
    )
    expected = (
        (SOURCE_UNIQUE_TRAIN, SOURCE_UNIQUE_TRAIN_SHA256, MEETING_COUNT),
        (UNIQUE_TRAIN, UNIQUE_TRAIN_SHA256, MEETING_COUNT),
        (PHYSICAL_GRPO_TRAIN, PHYSICAL_GRPO_TRAIN_SHA256, 312),
        (PHYSICAL_SFT_TRAIN, PHYSICAL_SFT_TRAIN_SHA256, 312),
    )
    for path, sha256, rows in expected:
        base._require(base._sha256_file(path) == sha256, f"source drift: {path}")
        _source_record(path, rows=rows)
    base._require(
        base._sha256_file(TRAINING_CONFIG) == TRAINING_CONFIG_SHA256,
        "training config drift",
    )
    return release


def _training_samples() -> list[dict[str, Any]]:
    """Read the exact teacher-compressed N=211 source, never the 312-row views."""

    _validate_release_sources()
    source_rows = base._read_jsonl(
        SOURCE_UNIQUE_TRAIN, label="source_unique_train N=211"
    )
    unique_rows = base._read_jsonl(UNIQUE_TRAIN, label="unique training manifest")
    unique_by_id = {str(row["sample_id"]): row for row in unique_rows}
    base._require(
        len(source_rows) == len(unique_rows) == len(unique_by_id) == MEETING_COUNT,
        "unique training population closure drift",
    )
    physical_prompt_maps: dict[str, dict[str, str]] = {}
    for role, path, id_field in (
        ("decision_sft", PHYSICAL_SFT_TRAIN, "source_sample_id"),
        ("decision_grpo", PHYSICAL_GRPO_TRAIN, "sample_id"),
    ):
        prompt_map: dict[str, str] = {}
        for physical in base._read_jsonl(path, label=f"physical {role} train"):
            sample_id = str(physical.get(id_field) or "")
            prompt = str(physical.get("prompt") or "")
            base._require(sample_id and prompt, f"invalid physical {role} row")
            if sample_id in prompt_map:
                base._require(
                    prompt_map[sample_id] == prompt,
                    f"multiple prompt bytes for {role}/{sample_id}",
                )
            prompt_map[sample_id] = prompt
        base._require(
            len(prompt_map) == MEETING_COUNT
            and set(prompt_map) == set(unique_by_id),
            f"physical {role} unique-prompt closure drift",
        )
        physical_prompt_maps[role] = prompt_map
    rows = []
    for row in source_rows:
        sample_id = str(row.get("sample_id") or "")
        counterpart = unique_by_id.get(sample_id)
        base._require(counterpart is not None, f"unknown training sample: {sample_id}")
        prompt = str(row.get("prompt") or "")
        direction = str(row.get("direction") or "")
        magnitude = int(row.get("magnitude_bp"))
        meeting_date = str(row.get("meeting_date") or "")
        base._require(
            prompt
            and base._sha256_text(prompt) == row.get("prompt_sha256")
            and row.get("prompt_sha256") == counterpart.get("prompt_sha256")
            and direction == counterpart.get("direction")
            and magnitude == int(counterpart.get("magnitude_bp")),
            f"training prompt/target binding drift: {sample_id}",
        )
        base._require(
            all(prompt_map[sample_id] == prompt for prompt_map in physical_prompt_maps.values()),
            f"source/SFT/GRPO prompt-byte drift: {sample_id}",
        )
        rows.append(
            {
                "sample_id": sample_id,
                "panel": PANEL,
                "split": "train",
                "meeting_id": meeting_date,
                "meeting_start_date": meeting_date,
                "meeting_date": meeting_date,
                "direction": direction,
                "magnitude_bp": magnitude,
                "prompt": prompt,
                "prompt_sha256": row["prompt_sha256"],
                "population": row.get("population"),
                "population_role": row.get("population_role"),
                "source_row_sha256": base._sha256_text(base.canonical_json(row)),
            }
        )
    rows.sort(key=lambda value: (value["meeting_date"], value["sample_id"]))
    base._require(
        len({row["sample_id"] for row in rows}) == MEETING_COUNT
        and len({row["meeting_date"] for row in rows}) == MEETING_COUNT,
        "duplicate training meeting/sample identity",
    )
    base._require(
        Counter(row["direction"] for row in rows) == Counter(EXPECTED_DIRECTION_COUNTS)
        and Counter((row["direction"], row["magnitude_bp"]) for row in rows)
        == Counter(EXPECTED_DIRECTION_MAGNITUDE_COUNTS),
        "training direction/magnitude distribution drift",
    )
    return rows


def _manifest_models() -> list[dict[str, Any]]:
    from open_r1.provenance import fingerprint_artifact_path

    models = []
    for spec in MODEL_SPECS:
        artifact = (
            _fingerprint_chk0_runtime_payload(Path(spec["path"]))
            if spec["label"] == "model_chk0"
            else fingerprint_artifact_path(Path(spec["path"]))
        )
        base._require(
            artifact.get("sha256") == spec["sha256"],
            f"model artifact drift: {spec['label']}",
        )
        models.append(
            {
                key: copy.deepcopy(value)
                for key, value in spec.items()
                if key != "path"
            }
            | {"artifact": artifact}
        )
    return models


def _fingerprint_chk0_runtime_payload(path: Path) -> dict[str, Any]:
    """Bind runtime-bearing chk-0 files while excluding mutable ``.git`` metadata."""

    root = path.resolve()
    base._require(root.is_dir() and not root.is_symlink(), "unsafe chk-0 root")
    records = []
    aggregate = hashlib.sha256()
    total_bytes = 0
    for relative, (expected_bytes, expected_sha256) in sorted(
        CHK0_RUNTIME_FILES.items()
    ):
        candidate = root / relative
        base._require(
            candidate.is_file() and not candidate.is_symlink(),
            f"chk-0 runtime file missing or unsafe: {relative}",
        )
        observed_bytes = candidate.stat().st_size
        observed_sha256 = base._sha256_file(candidate)
        base._require(
            observed_bytes == expected_bytes and observed_sha256 == expected_sha256,
            f"chk-0 runtime payload drift: {relative}",
        )
        records.append(
            {"path": relative, "bytes": observed_bytes, "sha256": observed_sha256}
        )
        aggregate.update(
            f"{relative}\0{observed_bytes}\0{observed_sha256}\n".encode("utf-8")
        )
        total_bytes += observed_bytes
    payload_sha256 = aggregate.hexdigest()
    base._require(
        payload_sha256 == CHK0_RUNTIME_PAYLOAD_SHA256
        and total_bytes == 16_069_669_155,
        "chk-0 runtime payload aggregate drift",
    )
    return {
        "path": str(root),
        "kind": "directory_runtime_payload_allowlist",
        "sha256": payload_sha256,
        "file_count": len(records),
        "total_bytes": total_bytes,
        "algorithm": (
            "sha256(sorted runtime allowlist records "
            "'<relative_path>\\0<size>\\0<file_sha256>\\n')"
        ),
        "files": records,
        "excluded_from_fingerprint": [
            ".git/**",
            ".gitattributes",
            "README.md",
            "LICENSE",
            "figures/**",
        ],
    }


def prepare(output_root: Path) -> dict[str, Any]:
    """Seal the 211-meeting input and 8,440-case schedule create-only."""

    _configure_static()
    release = _validate_release_sources()
    root = output_root.resolve()
    base._require(
        not root.exists() and not root.is_symlink(), f"output root exists: {root}"
    )
    rows = _training_samples()
    system_prompt, prompt_contract = source._training_contract()
    tokenizer = source._load_tokenizer(MODEL_SPECS[-1])
    token_rows = []
    for row in rows:
        prompt_ids = source._prompt_ids(
            tokenizer, source._messages(system_prompt, row["prompt"])
        )
        base._require(
            len(prompt_ids) <= 2560, f"prompt exceeds 2560 tokens: {row['sample_id']}"
        )
        row["prompt_token_count"] = len(prompt_ids)
        token_rows.append(
            {
                "schema_version": TOKEN_LEDGER_SCHEMA,
                "sample_id": row["sample_id"],
                "prompt_sha256": row["prompt_sha256"],
                "prompt_token_count": len(prompt_ids),
                "prompt_token_ids": prompt_ids,
                "prompt_token_ids_sha256": base._sha256_text(
                    base.canonical_json(prompt_ids)
                ),
            }
        )
    # All four artifacts receive the exact same frozen prompt IDs.  This is a
    # semantic parity gate, not merely a comparison of tokenizer file hashes;
    # the pretrained chk-0 tokenizer bundle differs on disk from the merged
    # checkpoints even though these 211 rendered prompts tokenize identically.
    for model in MODEL_SPECS:
        model_tokenizer = source._load_tokenizer(model)
        for sample, expected in zip(rows, token_rows, strict=True):
            observed = source._prompt_ids(
                model_tokenizer,
                source._messages(system_prompt, sample["prompt"]),
            )
            base._require(
                observed == expected["prompt_token_ids"],
                f"cross-model prompt-token drift: {model['label']}/{sample['sample_id']}",
            )
    global_preimage = [[row["sample_id"], row["prompt_token_ids"]] for row in token_rows]
    global_sha = base._sha256_text(
        json.dumps(global_preimage, separators=(",", ":"), ensure_ascii=False)
    )
    token_counts = [int(row["prompt_token_count"]) for row in token_rows]
    base.GLOBAL_PROMPT_TOKEN_IDS_SHA256 = global_sha
    base.MIN_PROMPT_TOKENS = min(token_counts)
    base.MAX_OBSERVED_PROMPT_TOKENS = max(token_counts)
    models = _manifest_models()
    cases = base.canonical_cases(rows, models)

    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        samples_path = staging / "inputs/training_unique_samples.jsonl"
        tokens_path = staging / "inputs/prompt_token_ledger.jsonl"
        cases_path = staging / "inputs/case_schedule.jsonl"
        source_path = staging / "sources/source_unique_train.jsonl"
        base._write_exclusive_jsonl(samples_path, rows)
        base._write_exclusive_jsonl(tokens_path, token_rows)
        base._write_exclusive_jsonl(cases_path, cases)
        base._copy_regular_file(SOURCE_UNIQUE_TRAIN, source_path)
        runtime = base._vllm_runtime_probe()
        gpu_inventory = base._physical_gpu_inventory()
        manifest = base._sealed(
            {
                "schema_version": MANIFEST_SCHEMA,
                "status": "prepared_pending_independent_audit",
                "created_at_utc": base._utc_now(),
                "evaluation_id": EVALUATION_ID,
                "purpose": "post_hoc_in_sample_decision_training_fit_diagnostic",
                "output_root": str(root),
                "implementation": base._implementation_records(),
                "models": models,
                "checkpoint_governance": {
                    "model_chk3_grpo_cp450": {
                        "selected_checkpoint_step": None,
                        "release_verdict": "quality_not_demonstrated",
                        "paper_facing_status": (
                            "best-observed exploratory candidate used for evaluation"
                        ),
                    }
                },
                "release": {
                    "manifest": base._file_record(RELEASE_MANIFEST),
                    "manifest_expected_sha256": RELEASE_MANIFEST_SHA256,
                    "source_unique_train": base._file_record(SOURCE_UNIQUE_TRAIN),
                    "unique_train": base._file_record(UNIQUE_TRAIN),
                    "physical_sft_train": base._file_record(PHYSICAL_SFT_TRAIN),
                    "physical_grpo_train": base._file_record(PHYSICAL_GRPO_TRAIN),
                    "release_unique_train_rows": release["unique_split_counts"]["train"],
                    "source_unique_rows_admitted": MEETING_COUNT,
                    "physical_schedule_rows_admitted": 0,
                    "duplicate_training_exposures_admitted": 0,
                    "canonical_dag_bindable": False,
                    "training_composition_unique_meetings": {
                        "post2008_core": 102,
                        "pre2009_supplement": 109,
                    },
                    "supplement_timing_caveat": (
                        "25 of 109 supplement meetings use a stored meeting-end "
                        "date, so D-1 may coincide with the first day of a two-day meeting"
                    ),
                    "supplement_hold_label_caveat": (
                        "all 67 supplement hold labels use the implemented "
                        "missing-record-to-zero fallback"
                    ),
                },
                "inputs": {
                    "samples": base._file_record(samples_path, relative_to=staging),
                    "prompt_token_ledger": base._file_record(
                        tokens_path, relative_to=staging
                    ),
                    "case_schedule": base._file_record(cases_path, relative_to=staging),
                    "source_unique_train_copy": base._file_record(
                        source_path, relative_to=staging
                    ),
                },
                "prompt_contract": {
                    **copy.deepcopy(prompt_contract),
                    "training_config": base._file_record(TRAINING_CONFIG),
                    "source": "frozen_teacher_compressed_source_unique_train",
                    "source_rows": MEETING_COUNT,
                    "tokenizer_fix_mistral_regex": False,
                    "global_prompt_token_ids_sha256": global_sha,
                    "observed_prompt_token_range": [min(token_counts), max(token_counts)],
                    "physical_sample_order": "meeting_date_then_sample_id",
                    "all_four_model_tokenization_parity": "passed_211_of_211",
                },
                "population": {
                    "panel": PANEL,
                    "independent_meetings": MEETING_COUNT,
                    "direction_counts": EXPECTED_DIRECTION_COUNTS,
                    "direction_magnitude_counts": {
                        f"{direction}_{magnitude}bp": count
                        for (direction, magnitude), count in sorted(
                            EXPECTED_DIRECTION_MAGNITUDE_COUNTS.items()
                        )
                    },
                    "models": MODEL_COUNT,
                    "replicates_per_meeting_model": REPLICATES,
                    "expected_rows": EXPECTED_ROWS,
                    "generation_rows_are_nested_not_independent": True,
                    "training_performed_by_this_workflow": False,
                },
                "generation": {
                    "backend": "vllm-async-engine-v1-two-independent-dp1-workers",
                    "generation_mode": "stochastic_sampling",
                    "do_sample": True,
                    "temperature": base.TEMPERATURE,
                    "top_p": base.TOP_P,
                    "top_k": base.TOP_K,
                    "repetition_penalty": base.REPETITION_PENALTY,
                    "max_tokens": base.MAX_NEW_TOKENS,
                    "max_model_len": base.MAX_MODEL_LEN,
                    "max_num_seqs_per_worker": base.MAX_NUM_SEQS,
                    "max_num_batched_tokens_per_worker": base.MAX_NUM_BATCHED_TOKENS,
                    "gpu_memory_utilization_per_worker": base.GPU_MEMORY_UTILIZATION,
                    "dtype": "bfloat16",
                    "quantization": None,
                    "data_parallel_size_per_worker": 1,
                    "tensor_parallel_size_per_worker": 1,
                    "physical_gpu_indexes": [0, 1],
                    "workers_per_model": 2,
                    "models_processed_sequentially": True,
                    "paired_block": "meeting_x_replicate_all_four_models",
                    "shard_function": "paired_block_id_mod_2",
                    "replicate_seeds": list(REPLICATE_SEEDS),
                    "row_seed": (
                        "sha256(evaluation_id:replicate_seed:sample_id)_first64_mod_2^31"
                    ),
                    "row_seed_shared_across_models": True,
                    "fresh_generations_only": True,
                    "expected_paired_blocks": MEETING_COUNT * REPLICATES,
                    "expected_blocks_per_shard": base.EXPECTED_BLOCKS_PER_SHARD,
                    "expected_rows_per_shard": base.EXPECTED_ROWS_PER_SHARD,
                },
                "runtime_prebinding": {
                    "vllm_python": runtime,
                    "physical_gpu_inventory": gpu_inventory,
                    "required_environment": base.REQUIRED_VLLM_ENV,
                    "gpu_lock_template": base.GPU_LOCK_TEMPLATE,
                },
                "persistence": {
                    "single_writer_completion_order_wal_per_model_shard": True,
                    "append_flush_fsync_per_completed_request": True,
                    "resume_only_missing_tuple_keys": True,
                    "raw_completion_token_ids_retained": True,
                    "full_raw_token_replay_required": True,
                    "post_run_receipt_required": True,
                },
                "statistics": {
                    "independent_unit": "meeting",
                    "replicates_nested_within_meeting": True,
                    "post_hoc_exact_action_cells": True,
                    "invalid_predictions_count_as_errors": True,
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "bootstrap": (
                        "shared-index paired two-stage hierarchical bootstrap: "
                        "meetings within direction-magnitude cell, then K replicate "
                        "indices within selected meeting"
                    ),
                    "pairwise_p_values": None,
                },
                "interpretation": {
                    "status": "in_sample_post_hoc_diagnostic_only",
                    "external_or_held_out": False,
                    "generalization_claim_permitted": False,
                    "asymmetric_exposure": (
                        "Only Model chk-3 SFT cp38 and GRPO cp450 were trained on "
                        "the Decision release; chk-0 and chk-1 are background comparators."
                    ),
                },
                "limitations": [
                    "The 211 meetings are the Decision training split.",
                    "The 312-row SFT and GRPO schedules are repeated exposures and are excluded.",
                    "K=10 measures decoding variability and does not increase N=211.",
                    "Model comparisons have asymmetric Decision-training exposure.",
                    "This diagnostic cannot estimate temporal or out-of-sample generalization.",
                    "The infrastructure smoke uses chk-0 and includes the longest prompt; the three tuned artifacts already passed the sealed external vLLM run under the same engine contract.",
                    "The cut-100, cut-75, and hike-50 cells contain only 1, 2, and 3 meetings and are descriptive only.",
                    "Bootstrap intervals measure frozen-release and decoding-resampling sensitivity; they omit training-run and dataset-construction uncertainty.",
                ],
            }
        )
        manifest_path = staging / "evaluation_manifest.json"
        base._write_exclusive_json(manifest_path, manifest)
        os.rename(staging, root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "prepared_pending_independent_audit",
        "manifest": base._file_record(root / "evaluation_manifest.json"),
        "unique_meetings": MEETING_COUNT,
        "expected_rows": EXPECTED_ROWS,
    }


def validate_manifest(
    path: Path, expected_sha256: str, *, deep_models: bool = False
) -> dict[str, Any]:
    _configure_static()
    base._require(base._sha256_file(path) == expected_sha256, "manifest SHA drift")
    value = base._read_json(path, label="training diagnostic manifest")
    base._require(value.get("schema_version") == MANIFEST_SCHEMA, "manifest schema drift")
    base._validate_integrity(value, label="training diagnostic manifest")
    root = path.parent.resolve()
    base._require(value.get("output_root") == str(root), "manifest root drift")
    base._require(
        value.get("implementation") == base._implementation_records(),
        "implementation drift",
    )
    _validate_release_sources()
    _configure_token_contract(value)
    for name in (
        "samples",
        "prompt_token_ledger",
        "case_schedule",
        "source_unique_train_copy",
    ):
        record = value["inputs"][name]
        prepared = base._safe_prepared_path(
            root, record.get("path"), label=f"prepared input {name}"
        )
        base._require(
            base._file_record(prepared, relative_to=root) == record,
            f"prepared input drift: {name}",
        )
    base._require(
        base._sha256_file(root / value["inputs"]["source_unique_train_copy"]["path"])
        == SOURCE_UNIQUE_TRAIN_SHA256,
        "copied source_unique_train drift",
    )
    samples, _, cases = base._load_prepared(value, root)
    base._require(
        len(samples) == MEETING_COUNT
        and len(cases) == EXPECTED_ROWS
        and len({row["sample_id"] for row in samples}) == MEETING_COUNT
        and len({row["meeting_date"] for row in samples}) == MEETING_COUNT,
        "prepared population closure drift",
    )
    base._require(
        Counter(row["direction"] for row in samples)
        == Counter(EXPECTED_DIRECTION_COUNTS)
        and Counter((row["direction"], row["magnitude_bp"]) for row in samples)
        == Counter(EXPECTED_DIRECTION_MAGNITUDE_COUNTS),
        "prepared class distribution drift",
    )
    base._require(
        value["population"]["expected_rows"] == EXPECTED_ROWS
        and value["release"]["physical_schedule_rows_admitted"] == 0,
        "manifest grain contract drift",
    )
    if deep_models:
        from open_r1.provenance import fingerprint_artifact_path

        for model in value["models"]:
            observed = (
                _fingerprint_chk0_runtime_payload(base._model_path(model))
                if model["label"] == "model_chk0"
                else fingerprint_artifact_path(base._model_path(model))
            )
            base._require(
                observed == model["artifact"],
                f"model drift: {model['label']}",
            )
    return value


def _material_result(**kwargs: Any) -> dict[str, Any]:
    result = ORIGINAL_BASE_MATERIAL_RESULT(**kwargs)
    result.update(
        {
            "split": "train_in_sample_diagnostic",
            "result_origin": "fresh_vllm_stochastic_training_meeting_diagnostic",
            "interpretation_scope": "post_hoc_in_sample_not_generalization",
        }
    )
    return result


def authorize(
    manifest_path: Path,
    manifest_sha256: str,
    *,
    audit_statement: str,
) -> dict[str, Any]:
    """Authorize only the exact 8,440-row formal scope after smoke replay."""

    manifest = validate_manifest(manifest_path, manifest_sha256, deep_models=True)
    base._require(
        len(audit_statement.strip()) >= 20,
        "independent audit statement is too short",
    )
    smoke_path = manifest_path.parent / "smoke/smoke_receipt.json"
    smoke = base._validate_smoke_receipt(
        smoke_path,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )
    path = manifest_path.parent / "formal_generation_authorization.json"
    base._require(
        not path.exists() and not path.is_symlink(), "authorization already exists"
    )
    value = base._sealed(
        {
            "schema_version": AUTHORIZATION_SCHEMA,
            "status": "authorized_after_independent_audit",
            "created_at_utc": base._utc_now(),
            "evaluation_id": EVALUATION_ID,
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": manifest_sha256,
            },
            "audit_statement": audit_statement.strip(),
            "authorized_scope": "formal_8440_row_training_meeting_diagnostic_generation_only",
            "validated_infrastructure_smoke": {
                "receipt": base._file_record(smoke_path),
                "rows": smoke["rows"],
                "shards": smoke["shards"],
                "gates": copy.deepcopy(smoke["gates"]),
            },
            "generation_contract": copy.deepcopy(manifest["generation"]),
            "interpretation_scope": "post_hoc_in_sample_not_generalization",
        }
    )
    base._write_exclusive_json(path, value)
    return {"status": value["status"], "authorization": base._file_record(path)}


def validate_authorization(
    path: Path,
    expected_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    *,
    deep_smoke: bool = True,
) -> dict[str, Any]:
    base._require(base._sha256_file(path) == expected_sha256, "authorization SHA drift")
    value = base._read_json(path, label="training diagnostic authorization")
    base._validate_integrity(value, label="training diagnostic authorization")
    manifest = validate_manifest(manifest_path, manifest_sha256, deep_models=False)
    base._require(
        value.get("schema_version") == AUTHORIZATION_SCHEMA
        and value.get("status") == "authorized_after_independent_audit"
        and value.get("evaluation_id") == EVALUATION_ID
        and value.get("manifest")
        == {"path": str(manifest_path.resolve()), "sha256": manifest_sha256}
        and value.get("authorized_scope")
        == "formal_8440_row_training_meeting_diagnostic_generation_only"
        and value.get("generation_contract") == manifest["generation"]
        and value.get("interpretation_scope")
        == "post_hoc_in_sample_not_generalization",
        "training diagnostic authorization scope drift",
    )
    smoke_binding = value.get("validated_infrastructure_smoke", {}).get("receipt")
    base._require(isinstance(smoke_binding, Mapping), "authorization lacks smoke binding")
    smoke_path = Path(str(smoke_binding.get("path") or ""))
    base._require(
        base._file_record(smoke_path) == smoke_binding,
        "authorized smoke receipt drift",
    )
    if deep_smoke:
        smoke = _validate_smoke_receipt(
            smoke_path,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
        )
    else:
        smoke = base._read_json(smoke_path, label="authorized smoke receipt")
        base._validate_integrity(smoke, label="authorized smoke receipt")
        base._require(
            smoke.get("schema_version") == SMOKE_RECEIPT_SCHEMA
            and smoke.get("status") == "passed"
            and smoke.get("evaluation_id") == EVALUATION_ID
            and smoke.get("manifest")
            == {"path": str(manifest_path.resolve()), "sha256": manifest_sha256}
            and smoke.get("rows") == base.SMOKE_ROWS
            and smoke.get("shards") == {"shard0": 10, "shard1": 10}
            and smoke.get("prior_generation_rows_reused") == 0,
            "authorized smoke shallow scope drift",
        )
    base._require(
        value["validated_infrastructure_smoke"]["rows"] == smoke["rows"]
        and value["validated_infrastructure_smoke"]["shards"] == smoke["shards"]
        and value["validated_infrastructure_smoke"]["gates"] == smoke["gates"],
        "authorized smoke evidence drift",
    )
    return value


def _validate_smoke_receipt(
    path: Path, *, manifest_path: Path, manifest_sha256: str
) -> dict[str, Any]:
    value = base._read_json(path, label="training diagnostic smoke receipt")
    base._validate_integrity(value, label="training diagnostic smoke receipt")
    required_gates = {
        "exact_formal_engine_and_sampling_contract": True,
        "exact_default_unfixed_prompt_token_ids": True,
        "returned_prompt_token_identity": True,
        "two_gpu_receipts": True,
        "exact_twenty_row_union": True,
        "normal_finish_reason_and_token_volume": True,
        "full_raw_token_replay": True,
        "prior_generation_rows_reused_zero": True,
    }
    base._require(
        value.get("schema_version") == SMOKE_RECEIPT_SCHEMA
        and value.get("status") == "passed"
        and value.get("evaluation_id") == EVALUATION_ID
        and value.get("manifest")
        == {"path": str(manifest_path.resolve()), "sha256": manifest_sha256}
        and value.get("rows") == base.SMOKE_ROWS
        and value.get("shards") == {"shard0": 10, "shard1": 10}
        and value.get("prior_generation_rows_reused") == 0
        and value.get("gates") == required_gates,
        "training diagnostic smoke role/gate drift",
    )
    manifest = validate_manifest(manifest_path, manifest_sha256, deep_models=False)
    material, raw, worker_receipts = base._load_smoke_rows(
        root=manifest_path.parent,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )
    root = manifest_path.parent
    base._require(value.get("workers") == worker_receipts, "smoke worker binding drift")
    base._require(
        value.get("raw_results")
        == base._file_record(root / "smoke/raw_results.jsonl", relative_to=root)
        and value.get("material_results")
        == base._file_record(root / "smoke/results.jsonl", relative_to=root)
        and base._read_jsonl(
            root / "smoke/raw_results.jsonl", label="smoke raw assembly"
        )
        == raw
        and base._read_jsonl(
            root / "smoke/results.jsonl", label="smoke material assembly"
        )
        == material,
        "smoke assembled-result replay drift",
    )
    return value


def smoke(
    manifest_path: Path, manifest_sha256: str, *, resume: bool
) -> dict[str, Any]:
    manifest = validate_manifest(manifest_path, manifest_sha256, deep_models=True)
    root = manifest_path.parent
    base._require(
        not (root / "formal_generation_authorization.json").exists(),
        "smoke must precede formal authorization",
    )
    receipt_path = root / "smoke/smoke_receipt.json"
    if receipt_path.exists():
        value = _validate_smoke_receipt(
            receipt_path,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
        )
        return {
            "status": "already_passed",
            "smoke_receipt": base._file_record(receipt_path),
            "rows": value["rows"],
        }
    _launch_worker_pair(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        authorization_path=None,
        authorization_sha256=None,
        model_label=base.SMOKE_MODEL_LABEL,
        resume=resume,
        scope="smoke",
    )
    material, raw, worker_receipts = base._load_smoke_rows(
        root=root, manifest=manifest, manifest_sha256=manifest_sha256
    )
    base._require(
        all(
            row["vllm_finish_reason"] in {"stop", "length"}
            and 1 <= len(row["generated_token_ids_raw"]) <= base.MAX_NEW_TOKENS
            for row in raw
        ),
        "smoke finish/token-volume gate failed",
    )
    raw_path = root / "smoke/raw_results.jsonl"
    material_path = root / "smoke/results.jsonl"
    base._write_or_validate_jsonl(raw_path, raw, label="assembled smoke raw results")
    base._write_or_validate_jsonl(
        material_path, material, label="assembled smoke material results"
    )
    receipt = base._sealed(
        {
            "schema_version": SMOKE_RECEIPT_SCHEMA,
            "status": "passed",
            "created_at_utc": base._utc_now(),
            "evaluation_id": EVALUATION_ID,
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": manifest_sha256,
            },
            "scope": "two_training_meetings_times_ten_replicates_chk0_two_gpus",
            "model_label": base.SMOKE_MODEL_LABEL,
            "meeting_ranks": list(base.SMOKE_MEETING_RANKS),
            "rows": len(raw),
            "shards": {"shard0": 10, "shard1": 10},
            "workers": worker_receipts,
            "raw_results": base._file_record(raw_path, relative_to=root),
            "material_results": base._file_record(material_path, relative_to=root),
            "prior_generation_rows_reused": 0,
            "gates": {
                "exact_formal_engine_and_sampling_contract": True,
                "exact_default_unfixed_prompt_token_ids": True,
                "returned_prompt_token_identity": True,
                "two_gpu_receipts": True,
                "exact_twenty_row_union": True,
                "normal_finish_reason_and_token_volume": True,
                "full_raw_token_replay": True,
                "prior_generation_rows_reused_zero": True,
            },
        }
    )
    base._write_exclusive_json(receipt_path, receipt)
    _validate_smoke_receipt(
        receipt_path,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )
    return {
        "status": "passed",
        "smoke_receipt": base._file_record(receipt_path),
        "rows": base.SMOKE_ROWS,
    }


def _validate_run_receipt(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    authorization_path: Path,
    authorization_sha256: str,
    manifest: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = (
        validate_manifest(manifest_path, manifest_sha256, deep_models=True)
        if manifest is None
        else manifest
    )
    validate_authorization(
        authorization_path,
        authorization_sha256,
        manifest_path,
        manifest_sha256,
    )
    root = manifest_path.parent
    receipt = base._read_json(root / "run/run_receipt.json", label="run receipt")
    base._validate_integrity(receipt, label="run receipt")
    base._require(
        receipt.get("schema_version") == RUN_RECEIPT_SCHEMA
        and receipt.get("status") == "complete"
        and receipt.get("evaluation_id") == EVALUATION_ID
        and receipt.get("manifest") == base._file_record(manifest_path)
        and receipt.get("authorization") == base._file_record(authorization_path)
        and receipt.get("rows") == EXPECTED_ROWS
        and receipt.get("panel_rows") == {PANEL: EXPECTED_ROWS}
        and receipt.get("models") == MODEL_COUNT
        and receipt.get("prior_generation_rows_reused") == 0,
        "training diagnostic run receipt role/population drift",
    )
    results, worker_receipts = base._load_worker_rows(
        root=root,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        authorization_path=authorization_path,
    )
    base._require(
        len(results) == EXPECTED_ROWS
        and len({int(row["absolute_case_index"]) for row in results}) == EXPECTED_ROWS
        and Counter(row["panel"] for row in results) == Counter({PANEL: EXPECTED_ROWS})
        and Counter(row["model_label"] for row in results)
        == Counter({model["label"]: MEETING_COUNT * REPLICATES for model in MODEL_SPECS}),
        "training diagnostic exact 8,440-row/four-model union drift",
    )
    result_path = root / "run/results.jsonl"
    base._require(
        receipt.get("workers") == worker_receipts
        and receipt.get("results") == base._file_record(result_path, relative_to=root)
        and base._read_jsonl(result_path, label="assembled material results") == results,
        "run receipt worker/material replay drift",
    )
    runtime_path = root / "run/runtime_summary.json"
    expected_runtime = base._expected_runtime_summary(root, manifest)
    base._require(
        receipt.get("runtime") == base._file_record(runtime_path, relative_to=root)
        and base._read_json(runtime_path, label="runtime summary") == expected_runtime,
        "run receipt runtime replay drift",
    )
    return receipt, results


def run(
    manifest_path: Path,
    manifest_sha256: str,
    *,
    authorization_path: Path,
    authorization_sha256: str,
    resume: bool,
) -> dict[str, Any]:
    manifest = validate_manifest(manifest_path, manifest_sha256, deep_models=True)
    validate_authorization(
        authorization_path,
        authorization_sha256,
        manifest_path,
        manifest_sha256,
    )
    root = manifest_path.parent
    with base._run_lock(root):
        receipt_path = root / "run/run_receipt.json"
        if receipt_path.exists():
            _validate_run_receipt(
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
                authorization_path=authorization_path,
                authorization_sha256=authorization_sha256,
                manifest=manifest,
            )
            return {
                "status": "already_complete",
                "run_receipt": base._file_record(receipt_path),
            }
        for model in manifest["models"]:
            _launch_worker_pair(
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
                authorization_path=authorization_path,
                authorization_sha256=authorization_sha256,
                model_label=str(model["label"]),
                resume=resume,
                scope="formal",
            )
        results, worker_receipts = base._load_worker_rows(
            root=root,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            authorization_path=authorization_path,
        )
        base._require(len(results) == EXPECTED_ROWS, "material result closure drift")
        result_path = root / "run/results.jsonl"
        base._write_or_validate_jsonl(
            result_path, results, label="assembled material results"
        )
        runtime_path = root / "run/runtime_summary.json"
        base._write_or_validate_json(
            runtime_path,
            base._expected_runtime_summary(root, manifest),
            label="runtime summary",
        )
        receipt = base._sealed(
            {
                "schema_version": RUN_RECEIPT_SCHEMA,
                "status": "complete",
                "created_at_utc": base._utc_now(),
                "evaluation_id": EVALUATION_ID,
                "manifest": base._file_record(manifest_path),
                "authorization": base._file_record(authorization_path),
                "workers": worker_receipts,
                "results": base._file_record(result_path, relative_to=root),
                "runtime": base._file_record(runtime_path, relative_to=root),
                "rows": len(results),
                "panel_rows": dict(Counter(row["panel"] for row in results)),
                "models": MODEL_COUNT,
                "prior_generation_rows_reused": 0,
            }
        )
        base._write_exclusive_json(receipt_path, receipt)
        _validate_run_receipt(
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            authorization_path=authorization_path,
            authorization_sha256=authorization_sha256,
            manifest=manifest,
        )
    return {
        "status": "generation_complete",
        "results": base._file_record(root / "run/results.jsonl", relative_to=root),
        "run_receipt": base._file_record(
            root / "run/run_receipt.json", relative_to=root
        ),
        "rows": EXPECTED_ROWS,
    }


def _launch_worker_pair(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    authorization_path: Path | None,
    authorization_sha256: str | None,
    model_label: str,
    resume: bool,
    scope: str = "formal",
) -> None:
    root = manifest_path.parent
    base._require(scope in {"formal", "smoke"}, "invalid launch scope")
    logs = root / ("run/logs" if scope == "formal" else "smoke/logs")
    logs.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[str]] = []
    handles = []
    try:
        for shard_id in range(base.SHARD_COUNT):
            log_path = logs / f"{model_label}.shard{shard_id}.log"
            handle = log_path.open("a", encoding="utf-8")
            handles.append(handle)
            environment = {
                **os.environ,
                **base.REQUIRED_VLLM_ENV,
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(shard_id),
                "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
            command = [
                str(base.VLLM_PYTHON),
                "-u",
                "-m",
                "jobs.retrain_v2.evaluate_chk4_training_in_sample_stochastic_vllm_k10",
                "worker",
                "--manifest",
                str(manifest_path),
                "--manifest-sha256",
                manifest_sha256,
                "--model-label",
                model_label,
                "--shard-id",
                str(shard_id),
                "--physical-gpu-index",
                str(shard_id),
                "--scope",
                scope,
            ]
            if authorization_path is not None:
                command += [
                    "--authorization",
                    str(authorization_path),
                    "--authorization-sha256",
                    str(authorization_sha256),
                ]
            if resume:
                command.append("--resume")
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        failures = []
        for shard_id, process in enumerate(processes):
            return_code = process.wait()
            if return_code:
                failures.append((shard_id, return_code))
        base._require(not failures, f"vLLM worker failures for {model_label}: {failures}")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for handle in handles:
            handle.close()


def _condition_rows(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = [
        {
            "condition_id": "all",
            "policy_direction": "all",
            "magnitude_bp": None,
            "meeting_ids": [str(row["sample_id"]) for row in samples],
        }
    ]
    for direction in ("cut", "hold", "hike"):
        direction_rows = [row for row in samples if row["direction"] == direction]
        magnitudes = sorted({int(row["magnitude_bp"]) for row in direction_rows})
        if len(magnitudes) > 1:
            result.append(
                {
                    "condition_id": f"{direction}_all",
                    "policy_direction": direction,
                    "magnitude_bp": None,
                    "meeting_ids": [str(row["sample_id"]) for row in direction_rows],
                }
            )
        for magnitude in magnitudes:
            result.append(
                {
                    "condition_id": f"{direction}_{magnitude}bp",
                    "policy_direction": direction,
                    "magnitude_bp": magnitude,
                    "meeting_ids": [
                        str(row["sample_id"])
                        for row in direction_rows
                        if int(row["magnitude_bp"]) == magnitude
                    ],
                }
            )
    return result


def exact_action_table(
    results: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compute meeting-grain rates; invalid predictions remain errors."""

    by_model_meeting: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in results:
        by_model_meeting[(str(row["model_label"]), str(row["sample_id"]))].append(row)
    import numpy as np

    rows = []
    model_labels = [str(model["label"]) for model in models]
    for condition in _condition_rows(samples):
        meeting_ids = list(condition["meeting_ids"])
        base._require(bool(meeting_ids), "empty exact-action condition")
        block: dict[str, Any] = {
            key: value for key, value in condition.items() if key != "meeting_ids"
        }
        block["support_meetings"] = len(meeting_ids)
        block["completions_per_model"] = len(meeting_ids) * REPLICATES
        block["models"] = {}
        exact_matrices: dict[str, Any] = {}
        invalid_counts: dict[str, int] = {}
        for model in models:
            label = str(model["label"])
            meeting_vectors = []
            invalid = 0
            for sample_id in meeting_ids:
                values = sorted(
                    by_model_meeting[(label, sample_id)],
                    key=lambda value: int(value["replicate_id"]),
                )
                base._require(
                    len(values) == REPLICATES
                    and [int(value["replicate_id"]) for value in values]
                    == list(range(REPLICATES)),
                    f"replicate closure drift: {label}/{sample_id}",
                )
                exact_values = [bool(value["decision_exact"]) for value in values]
                for value, exact in zip(values, exact_values, strict=True):
                    prediction = value.get("decision_prediction")
                    is_invalid = not (
                        isinstance(prediction, Mapping)
                        and prediction.get("direction") in base.DIRECTIONS
                        and isinstance(prediction.get("magnitude_bp"), int)
                    )
                    base._require(
                        not (is_invalid and exact),
                        "invalid output cannot be exact-action correct",
                    )
                    invalid += int(is_invalid)
                meeting_vectors.append(exact_values)
            exact_matrices[label] = np.asarray(meeting_vectors, dtype=np.float64)
            invalid_counts[label] = invalid

        # Shared-index two-stage bootstrap: select meetings within this action
        # cell, then select K replicate indices within every selected meeting.
        # The same indices are applied to all four model matrices, preserving
        # the paired decoding design for both marginal intervals and contrasts.
        rng = np.random.default_rng(
            BOOTSTRAP_SEED
            + int(base._sha256_text(str(condition["condition_id"]))[:8], 16)
        )
        draw_values = {
            label: np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
            for label in model_labels
        }
        cursor = 0
        while cursor < BOOTSTRAP_DRAWS:
            size = min(250, BOOTSTRAP_DRAWS - cursor)
            meeting_index = rng.integers(0, len(meeting_ids), size=(size, len(meeting_ids)))
            replicate_index = rng.integers(
                0,
                REPLICATES,
                size=(size, len(meeting_ids), REPLICATES),
            )
            expanded_meetings = meeting_index[:, :, None]
            for label in model_labels:
                sampled = exact_matrices[label][expanded_meetings, replicate_index]
                draw_values[label][cursor : cursor + size] = sampled.mean(axis=(1, 2))
            cursor += size

        for model in models:
            label = str(model["label"])
            matrix = exact_matrices[label]
            correct = int(matrix.sum())
            estimate = float(matrix.mean())
            lower, upper = np.quantile(
                draw_values[label], [0.025, 0.975], method="linear"
            )
            block["models"][label] = {
                "display_name": model["display_name"],
                "correct_completions": correct,
                "total_completions": len(meeting_ids) * REPLICATES,
                "exact_action_rate": estimate,
                "paired_two_stage_bootstrap_95_percentile_interval": [
                    float(lower),
                    float(upper),
                ],
                "invalid_completions_counted_as_errors": invalid_counts[label],
            }
        contrasts = []
        for left, right in itertools.combinations(model_labels, 2):
            differences = draw_values[left] - draw_values[right]
            lower, upper = np.quantile(differences, [0.025, 0.975], method="linear")
            contrasts.append(
                {
                    "model_a": left,
                    "model_b": right,
                    "difference_a_minus_b": float(
                        exact_matrices[left].mean() - exact_matrices[right].mean()
                    ),
                    "paired_two_stage_bootstrap_95_percentile_interval": [
                        float(lower),
                        float(upper),
                    ],
                    "p_value": None,
                }
            )
        block["pairwise_contrasts"] = contrasts
        rows.append(block)
    return rows


def _format_rate(value: Mapping[str, Any]) -> str:
    return (
        f"{100 * float(value['exact_action_rate']):.2f}% "
        f"({value['correct_completions']}/{value['total_completions']})"
    )


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Post-Hoc Training-Meeting Exact-Action Fit",
        "",
        "This is an in-sample diagnostic on the 211 unique Decision training meetings. It is not external, held-out, temporal, or out-of-sample evidence. Ten stochastic completions are nested within each meeting; invalid predictions are errors.",
        "",
        "| Target condition | Meetings | Model chk-0 | Model chk-1 cp200 | Model chk-3 SFT cp38 | Model chk-3 GRPO cp450 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = [str(model["label"]) for model in MODEL_SPECS]
    for row in rows:
        direction = str(row["policy_direction"])
        magnitude = row["magnitude_bp"]
        condition = (
            "All actions"
            if direction == "all"
            else f"{direction.title()} (all magnitudes)"
            if magnitude is None
            else f"{direction.title()} {magnitude} bp"
        )
        lines.append(
            "| "
            + " | ".join(
                [condition, str(row["support_meetings"])]
                + [_format_rate(row["models"][label]) for label in labels]
            )
            + " |"
        )
    lines += [
        "",
        "The two Model chk-3 states were trained on this Decision release. Model chk-0 and Model chk-1 did not receive the same Decision-label training and are background comparators; the exposure asymmetry prevents an out-of-sample model-ranking interpretation.",
        "",
        "Release caveats: the release is not canonical-DAG-bindable; 25 of 109 supplement meetings use a stored meeting-end date, so D-1 may coincide with the first meeting day; and all 67 supplement hold labels use the implemented missing-record-to-zero fallback. Cut 100 bp (N=1), cut 75 bp (N=2), and hike 50 bp (N=3) cells are descriptive only. Bootstrap intervals measure frozen-release and decoding-resampling sensitivity, not training-run or dataset-construction uncertainty.",
        "",
    ]
    return "\n".join(lines)


def _latex_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "% Generated by evaluate_chk4_training_in_sample_stochastic_vllm_k10.py.",
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\scriptsize",
        "    \\setlength{\\tabcolsep}{3pt}",
        "    \\renewcommand{\\arraystretch}{1.12}",
        "    \\caption{Post-Hoc Training-Meeting Exact-Action Fit by Policy Direction and Magnitude ($N=211$, $K=10$)}",
        "    \\label{tab:ch2:decision_training_exact_action_by_cell}",
        "    \\begin{tabular}{lrrrrr}",
        "        \\toprule",
        "        Target condition & $N$ & chk-0 & chk-1 cp200 & chk-3 SFT cp38 & chk-3 GRPO cp450 \\\\",
        "        \\midrule",
    ]
    labels = [str(model["label"]) for model in MODEL_SPECS]
    for row in rows:
        direction = str(row["policy_direction"])
        magnitude = row["magnitude_bp"]
        condition = (
            "All actions"
            if direction == "all"
            else f"{direction.title()} (all)"
            if magnitude is None
            else f"{direction.title()} {magnitude} bp"
        )
        rates = [
            f"{100 * float(row['models'][label]['exact_action_rate']):.2f}\\%"
            for label in labels
        ]
        lines.append(
            "        "
            + " & ".join([condition, str(row["support_meetings"]), *rates])
            + " \\\\"
        )
    lines += [
        "        \\bottomrule",
        "    \\end{tabular}",
        "    \\begin{minipage}{0.98\\textwidth}",
        "    \\footnotesize \\textit{Notes:} Each cell is the mean exact direction--magnitude correctness across ten stochastic completions per unique training meeting. Invalid outputs remain errors. The 312-row resampled SFT and GRPO schedules are excluded. This is an in-sample, post-hoc fit diagnostic and not evidence of held-out or temporal generalization. Only the two Model chk-3 states received Decision training on these meetings; chk-0 and chk-1 are background comparators. The cut-100, cut-75, and hike-50 cells have only 1, 2, and 3 meetings and are descriptive. The release is not canonical-DAG-bindable; 25/109 supplement cutoffs use stored end dates, and all 67 supplement hold labels use the missing-record-to-zero fallback. Bootstrap intervals do not include training-run or construction uncertainty.",
        "    \\end{minipage}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def score(
    manifest_path: Path,
    manifest_sha256: str,
    *,
    authorization_path: Path,
    authorization_sha256: str,
) -> dict[str, Any]:
    manifest = validate_manifest(manifest_path, manifest_sha256, deep_models=True)
    base._validate_authorization(
        authorization_path,
        authorization_sha256,
        manifest_path,
        manifest_sha256,
    )
    root = manifest_path.parent
    _, results = base._validate_run_receipt(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        authorization_path=authorization_path,
        authorization_sha256=authorization_sha256,
        manifest=manifest,
    )
    samples = base._read_jsonl(
        root / manifest["inputs"]["samples"]["path"], label="training samples"
    )
    table_rows = exact_action_table(results, samples, manifest["models"])
    summary = base._sealed(
        {
            "schema_version": SUMMARY_SCHEMA,
            "status": "complete",
            "created_at_utc": base._utc_now(),
            "evaluation_manifest_sha256": manifest_sha256,
            "run_receipt": base._file_record(
                root / "run/run_receipt.json", relative_to=root
            ),
            "independent_meetings": MEETING_COUNT,
            "models": MODEL_COUNT,
            "replicates_per_meeting_model": REPLICATES,
            "rows": EXPECTED_ROWS,
            "metric": "exact_direction_and_magnitude_accuracy",
            "invalid_outputs_count_as_errors": True,
            "conditions": table_rows,
            "interpretation": "post_hoc_in_sample_fit_diagnostic_not_generalization",
        }
    )
    report_root = root / "report"
    base._require(
        not report_root.exists() and not report_root.is_symlink(),
        "report output exists",
    )
    report_root.mkdir()
    base._write_exclusive_json(report_root / "summary.json", summary)
    base._write_exclusive_jsonl(
        report_root / "exact_action_by_direction_magnitude.jsonl", table_rows
    )
    base._write_exclusive_text(
        report_root / "technical_report.md", _markdown_table(table_rows)
    )
    base._write_exclusive_text(
        report_root / "training_exact_action_table.tex", _latex_table(table_rows)
    )
    receipt = base._sealed(
        {
            "status": "complete",
            "summary": base._file_record(report_root / "summary.json", relative_to=root),
            "table_ledger": base._file_record(
                report_root / "exact_action_by_direction_magnitude.jsonl",
                relative_to=root,
            ),
            "technical_report": base._file_record(
                report_root / "technical_report.md", relative_to=root
            ),
            "latex_table": base._file_record(
                report_root / "training_exact_action_table.tex", relative_to=root
            ),
        }
    )
    base._write_exclusive_json(report_root / "report_receipt.json", receipt)
    return {
        "status": "scored",
        "summary": base._file_record(report_root / "summary.json", relative_to=root),
        "latex_table": base._file_record(
            report_root / "training_exact_action_table.tex", relative_to=root
        ),
        "rows": EXPECTED_ROWS,
    }


def status(output_root: Path) -> dict[str, Any]:
    root = output_root.resolve()
    worker_status: dict[str, Any] = {}
    receipts = 0
    wal_rows = 0
    for model in MODEL_SPECS:
        label = str(model["label"])
        worker_status[label] = {}
        for shard_id in range(base.SHARD_COUNT):
            paths = base._worker_paths(root, label, shard_id)
            count = 0
            if paths["wal"].is_file():
                with paths["wal"].open("rb") as handle:
                    count = sum(bool(line.strip()) for line in handle)
            complete = paths["receipt"].is_file()
            receipts += int(complete)
            wal_rows += count
            worker_status[label][f"shard{shard_id}"] = {
                "wal_rows": count,
                "expected_rows": base.EXPECTED_BLOCKS_PER_SHARD,
                "complete": complete,
            }
    return {
        "output_root": str(root),
        "prepared": (root / "evaluation_manifest.json").is_file(),
        "authorized": (root / "formal_generation_authorization.json").is_file(),
        "worker_receipts": receipts,
        "expected_worker_receipts": MODEL_COUNT * base.SHARD_COUNT,
        "wal_rows": wal_rows,
        "expected_rows": EXPECTED_ROWS,
        "generation_complete": (root / "run/run_receipt.json").is_file(),
        "scored": (root / "report/report_receipt.json").is_file(),
        "workers": worker_status,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("prepare", "smoke", "authorize", "run", "worker", "score", "status"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--audit-statement")
    parser.add_argument("--model-label")
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--scope", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _configure_static()
    args = _parser().parse_args(argv)
    manifest_path = args.manifest or args.output_root / "evaluation_manifest.json"
    if args.stage == "prepare":
        result = prepare(args.output_root)
    elif args.stage == "status":
        result = status(args.output_root)
    else:
        base._require(bool(args.manifest_sha256), "--manifest-sha256 is required")
        if args.stage == "smoke":
            result = smoke(
                manifest_path, args.manifest_sha256, resume=args.resume
            )
        elif args.stage == "authorize":
            base._require(bool(args.audit_statement), "--audit-statement is required")
            result = authorize(
                manifest_path,
                args.manifest_sha256,
                audit_statement=args.audit_statement,
            )
        elif args.stage == "worker":
            base._require(bool(args.model_label), "--model-label is required")
            base._require(args.shard_id is not None, "--shard-id is required")
            base._require(
                args.physical_gpu_index is not None,
                "--physical-gpu-index is required",
            )
            result = base.worker(
                manifest_path=manifest_path,
                manifest_sha256=args.manifest_sha256,
                authorization_path=args.authorization,
                authorization_sha256=args.authorization_sha256,
                model_label=args.model_label,
                shard_id=args.shard_id,
                physical_gpu_index=args.physical_gpu_index,
                resume=args.resume,
                scope=args.scope,
            )
        else:
            base._require(args.authorization is not None, "--authorization is required")
            base._require(
                bool(args.authorization_sha256),
                "--authorization-sha256 is required",
            )
            if args.stage == "run":
                result = run(
                    manifest_path,
                    args.manifest_sha256,
                    authorization_path=args.authorization,
                    authorization_sha256=args.authorization_sha256,
                    resume=args.resume,
                )
            else:
                result = score(
                    manifest_path,
                    args.manifest_sha256,
                    authorization_path=args.authorization,
                    authorization_sha256=args.authorization_sha256,
                )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


_configure_static()


if __name__ == "__main__":
    raise SystemExit(main())
