"""Evaluate the frozen Decision artifacts on two deterministic Core8 panels.

This is an evaluation-only, no-retraining workflow.  It keeps two populations
separate:

* ``historical_n19``: regular 1993--2008 meetings absent from the earlier
  Decision supplement; and
* ``postcutoff_n12``: regular meetings after the frozen Decision training-data
  endpoint.

Both panels use the same deterministic input construction: exactly eight
point-in-time source-analysis blocks are concatenated in a frozen topic order.
This contract deliberately differs from the teacher-compressed brief used by
the already-opened 13-meeting Decision test and results must not be pooled.

Stages are create-only.  ``prepare`` seals inputs and official label sources,
``run`` writes resumable greedy inference batches, and ``score`` creates the
separate-panel statistical report.  No stage trains or modifies a model.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import html
import itertools
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from datetime import date, datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import yaml

from jobs.eval.build_chk3_external_holdout_1993_2008 import (
    CORE_TOPICS,
    _analysis_text,
)
from jobs.generation.prepare_chk4_supplement import compress_indicator_row
from open_r1.provenance import fingerprint_artifact_path, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_CONFIG = (
    REPO_ROOT / "configs/main/decision_historical_missing_n19_20260824.json"
)
POSTCUTOFF_CONFIG = (
    REPO_ROOT / "configs/main/decision_postcutoff_regular_n12_20260824.json"
)
HISTORICAL_RELEASE = REPO_ROOT / (
    "dataset/processed/retrain_v2/"
    "chk3_minutes_external_holdout_1993_2008_all_regular_v1"
)
HISTORICAL_CORE8 = HISTORICAL_RELEASE / "panels/core8.jsonl"
HISTORICAL_ROSTER = HISTORICAL_RELEASE / "official_meeting_roster.jsonl"
HISTORICAL_MINUTES_INVENTORY = (
    HISTORICAL_RELEASE / "references/official_minutes_inventory.jsonl"
)
HISTORICAL_CANDIDATE_ROOT = REPO_ROOT / (
    "output/data/retrain_v2/chk3/"
    "chk3_minutes_external_holdout_1993_2008_all_regular_v1_candidate"
)
HISTORICAL_SOURCE_HANDOFF = (
    HISTORICAL_CANDIDATE_ROOT / "sources/materialized/source_handoff.json"
)
POSTCUTOFF_LEDGER = REPO_ROOT / (
    "output/evaluation/retrain_v2/chk4_postcutoff_regular_n12_v1_20260824/source_ledger"
)
POSTCUTOFF_LEDGER_MANIFEST = POSTCUTOFF_LEDGER / "ledger_manifest.json"
POSTCUTOFF_INDICATOR_INPUTS = POSTCUTOFF_LEDGER / "indicator_inputs.jsonl"
DECISION_RELEASE = REPO_ROOT / (
    "dataset/processed/retrain_v2/chk4_decision_pre2009_train_balanced_v1_20260811"
)
TRAINING_CONFIG = REPO_ROOT / (
    "configs/retrain_v2/"
    "chk4_decision_grpo_from_pre2009_cp38_direct_full_no_smoke_v1_20260812.yaml"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    "output/evaluation/retrain_v2/chk4_external_deterministic_core8_n19_n12_v1_20260824"
)
TOKENIZER_BUNDLE_FILES = (
    "tokenizer.json",
    "chat_template.jinja",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
POSTCUTOFF_INITIAL_TARGET_RANGE = "4.25--4.50"
IMPLEMENTATION_FILES = {
    "evaluator": Path(__file__).resolve(),
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
    "core8_source_analysis_builder": REPO_ROOT
    / "jobs/eval/build_chk3_external_holdout_1993_2008.py",
    "source_compression": REPO_ROOT / "jobs/generation/prepare_chk4_supplement.py",
}

SYSTEM_PROMPT_SHA256 = (
    "426b64532dfb955a3941db2cc2bcfc9a785c0540ffba9ee94afabc189fe4ce18"
)
TRAINING_CONFIG_SHA256 = (
    "222c70d76823681b7ff07971704c38fd153abcac4620a4e0fab4fce00d77f06f"
)
USER_PREFIX = (
    "Make one policy decision using only the following pre-meeting analysis:\n\n"
)
USER_PREFIX_SHA256 = "3098efe835849e568346d2b4768b8e1bf604014a482fa0a29e4d24634033c43c"
POSTCUTOFF_LEDGER_PAYLOAD_SHA256 = (
    "6122eabead034857af09d84a4f55d58ec32450652897f0d949267c40df18d3ea"
)

MODELS: tuple[dict[str, Any], ...] = (
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
    },
)

PANELS = ("historical_n19", "postcutoff_n12")
PANEL_COUNTS = {"historical_n19": 19, "postcutoff_n12": 12}
DIRECTIONS = ("cut", "hold", "hike")
PREDICTION_LABELS = (*DIRECTIONS, "invalid")
MAX_PROMPT_TOKENS = 2560
MAX_NEW_TOKENS = 1536
TAIL_TOKENS = 256
BATCH_SIZE = 8
EXPECTED_RESULT_ROWS = sum(PANEL_COUNTS.values()) * len(MODELS)

MANIFEST_SCHEMA = "chk4-external-deterministic-core8-evaluation-manifest-v1"
SAMPLE_SCHEMA = "chk4-external-deterministic-core8-sample-v1"
RESULT_SCHEMA = "chk4-external-deterministic-core8-result-v1"
SUMMARY_SCHEMA = "chk4-external-deterministic-core8-summary-v1"
RUN_RECEIPT_SCHEMA = "chk4-external-deterministic-core8-run-receipt-v1"

# Every field below is reconstructed from the sealed raw completion token IDs
# by the bound tokenizer, completion decoder, delivery analyzer, and reward
# parser.  Stored values are never trusted by run resumption or scoring.
MATERIAL_REPLAY_FIELDS = (
    "generated_token_ids_first_eos_inclusive",
    "batched_padding_normalization",
    "completion",
    "completion_sha256",
    "status",
    "raw_generated_token_count",
    "completion_token_count",
    "hit_eos",
    "cap_reached",
    "think_boundary_count",
    "has_nonempty_reasoning",
    "plain_json",
    "json_object",
    "exact_keys",
    "decision_domain_valid",
    "parsed_decision",
    "json_error",
    "contract_valid",
    "full_token_4gram_repetition",
    "tail_token_count",
    "tail_token_4gram_repetition",
    "strict_periodic_tail",
    "repetition_valid",
    "delivery_valid",
    "failure_reasons",
    "decision_dense_v3_reward",
    "decision_prediction",
    "decision_direction_correct",
    "decision_exact",
    "decision_rejection_reason",
    "decision_forced_zero_reason",
    "response_format",
    "strict_json",
    "fenced_json",
)


class ExternalCore8EvaluationError(RuntimeError):
    """A frozen input, inference batch, or score failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExternalCore8EvaluationError(message)


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


def _package_versions() -> dict[str, str]:
    packages = (
        "torch",
        "transformers",
        "bitsandbytes",
        "tokenizers",
        "accelerate",
        "safetensors",
        "numpy",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError as exc:
            raise ExternalCore8EvaluationError(
                f"required runtime package is missing: {package}"
            ) from exc
    return versions


def _nvidia_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ExternalCore8EvaluationError(
            f"cannot inventory NVIDIA runtime: {exc}"
        ) from exc
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        _require(len(values) == 6, "unexpected nvidia-smi inventory schema")
        rows.append(
            {
                "index": int(values[0]),
                "uuid": values[1],
                "name": values[2],
                "driver_version": values[3],
                "memory_mib": int(values[4]),
                "compute_capability": values[5],
            }
        )
    _require(bool(rows), "NVIDIA inventory is empty")
    return rows


def _runtime_prebinding() -> dict[str, Any]:
    import torch

    return {
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": _package_versions(),
        "cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "physical_gpu_inventory": _nvidia_inventory(),
        "required_execution": {
            "visible_cuda_devices": 1,
            "load_in_4bit": True,
            "bnb_quant_type": "nf4",
            "bnb_double_quant": True,
            "bnb_compute_dtype": "bfloat16",
            "bnb_storage_dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "generation_mode": "greedy",
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "torch_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
            "cublas_workspace_config": ":4096:8",
            "python_no_user_site": "1",
            "thread_environment": {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
        },
    }


def _configure_deterministic_runtime(torch: Any) -> None:
    _require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
        "CUBLAS_WORKSPACE_CONFIG was not established before CUDA execution",
    )
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _runtime_record(torch: Any) -> dict[str, Any]:
    _require(torch.cuda.device_count() == 1, "runtime must expose one CUDA device")
    properties = torch.cuda.get_device_properties(0)
    return {
        **_runtime_prebinding(),
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "visible_cuda": {
            "device_count": torch.cuda.device_count(),
            "device_name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "major": properties.major,
            "minor": properties.minor,
        },
        "determinism_observed": {
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label} missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalCore8EvaluationError(f"cannot read {label}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must contain an object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"{label} missing or unsafe")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                _require(isinstance(value, dict), f"{label} row {number} invalid")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalCore8EvaluationError(f"cannot read {label}: {exc}") from exc
    return rows


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_exclusive_text(
        path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    )


def _write_exclusive_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_exclusive_text(path, "\n".join(canonical_json(row) for row in rows))


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


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"unsafe file: {path}")
    rendered = str(path.resolve())
    if relative_to is not None:
        rendered = path.resolve().relative_to(relative_to.resolve()).as_posix()
    record: dict[str, Any] = {
        "path": rendered,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".jsonl":
        with path.open("rb") as handle:
            record["rows"] = sum(bool(line.strip()) for line in handle)
    return record


def _model_by_label(label: str) -> dict[str, Any]:
    values = [model for model in MODELS if model["label"] == label]
    _require(len(values) == 1, f"unknown model: {label}")
    return values[0]


def _training_contract() -> tuple[str, dict[str, Any]]:
    _require(
        sha256_file(TRAINING_CONFIG) == TRAINING_CONFIG_SHA256,
        "training config drift",
    )
    value = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "training config is not a mapping")
    system_prompt = str(value.get("system_prompt") or "")
    _require(_sha256_text(system_prompt) == SYSTEM_PROMPT_SHA256, "system prompt drift")
    _require(_sha256_text(USER_PREFIX) == USER_PREFIX_SHA256, "user prefix drift")
    return system_prompt, value


def _messages(system_prompt: str, prompt: str) -> list[dict[str, str]]:
    _require(bool(system_prompt) and bool(prompt), "empty decision message")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def _prompt_ids(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> list[int]:
    try:
        value = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            truncation=False,
            return_dict=False,
        )
    except Exception as exc:
        raise ExternalCore8EvaluationError(
            f"chat-template tokenization failed: {exc}"
        ) from exc
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        value
        and isinstance(value[0], Sequence)
        and not isinstance(value[0], (str, bytes))
    ):
        _require(len(value) == 1, "chat template returned an unexpected batch")
        value = value[0]
    _require(
        not isinstance(value, (str, bytes)) and isinstance(value, Sequence),
        "chat template returned invalid token IDs",
    )
    try:
        result = [int(token_id) for token_id in value]
    except (TypeError, ValueError) as exc:
        raise ExternalCore8EvaluationError(
            "chat template returned non-integer token IDs"
        ) from exc
    _require(
        bool(result) and all(token_id >= 0 for token_id in result), "invalid token IDs"
    )
    return result


def _final_protocol() -> Any:
    # Imported only by GPU inference.  CPU preparation and scoring remain usable
    # in minimal environments that omit optional training-reward dependencies.
    from jobs.retrain_v2 import chk4_cp450_final_evaluation

    return chk4_cp450_final_evaluation


def _load_tokenizer(model: Mapping[str, Any]) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model["path"]), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _shared_tokenizer_bundle_binding() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for filename in TOKENIZER_BUNDLE_FILES:
        records = [_file_record(Path(model["path"]) / filename) for model in MODELS]
        signatures = {(record["bytes"], record["sha256"]) for record in records}
        _require(
            len(signatures) == 1,
            f"model tokenizer bundle differs: {filename}",
        )
        size, sha256 = next(iter(signatures))
        result[filename] = {"bytes": size, "sha256": sha256}
    return result


def _make_user_prompt(analysis: str) -> str:
    _require(bool(analysis.strip()), "source analysis is empty")
    return USER_PREFIX + canonical_json({"analysis": analysis})


def _assert_no_target_leak(
    *, prompt: str, meeting: Mapping[str, Any], analysis: str
) -> None:
    decoded = json.loads(prompt.removeprefix(USER_PREFIX))
    _require(decoded == {"analysis": analysis}, "decision user-message schema drift")
    for field in (
        "meeting_id",
        "meeting_date",
        "current_rate",
        "gold",
        "reward",
        "split",
    ):
        _require(f'"{field}"' not in prompt, f"forbidden prompt field: {field}")
    identities = {
        str(meeting.get(key) or "")
        for key in (
            "meeting_id",
            "meeting_start_date",
            "meeting_end_date",
            "decision_date",
        )
    }
    _require(
        not any(value and value in prompt for value in identities),
        "meeting identity leaked into decision prompt",
    )


def _decision_exposure_dates() -> set[str]:
    result: set[str] = set()
    for split in ("train", "validation", "test"):
        path = DECISION_RELEASE / f"manifests/unique/{split}.jsonl"
        for row in _read_jsonl(path, label=f"Decision {split} unique manifest"):
            value = str(row.get("meeting_date") or "")
            _require(bool(value), f"Decision {split} row lacks meeting date")
            result.add(value)
    _require(len(result) == 237, "Decision exposure roster drift")
    return result


def _historical_documents() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    release_manifest = _read_json(
        HISTORICAL_RELEASE / "release_manifest.json",
        label="historical release manifest",
    )
    _require(
        release_manifest.get("evaluation_only") is True
        and release_manifest.get("checkpoint_selection_allowed") is False,
        "historical release role drift",
    )
    for relative, path in (
        ("official_meeting_roster.jsonl", HISTORICAL_ROSTER),
        ("panels/core8.jsonl", HISTORICAL_CORE8),
        (
            "references/official_minutes_inventory.jsonl",
            HISTORICAL_MINUTES_INVENTORY,
        ),
    ):
        expected = release_manifest.get("files", {}).get(relative)
        _require(
            isinstance(expected, Mapping),
            f"historical release does not bind {relative}",
        )
        observed = _file_record(path, relative_to=HISTORICAL_RELEASE)
        _require(observed == expected, f"historical release file drift: {relative}")
    config = _read_json(HISTORICAL_CONFIG, label="historical N19 config")
    meetings = config.get("meetings")
    _require(isinstance(meetings, list) and len(meetings) == 19, "historical N19 drift")
    by_id = {str(row["meeting_id"]): row for row in meetings}
    _require(len(by_id) == 19, "duplicate historical meeting")
    rows = _read_jsonl(HISTORICAL_CORE8, label="historical Core8 panel")
    topic_map: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        meeting_id = str(row.get("meeting_id") or "")
        if meeting_id not in by_id:
            continue
        topic = str(row.get("topic") or "")
        _require(topic in CORE_TOPICS, f"non-Core8 topic in historical panel: {topic}")
        _require(
            topic not in topic_map[meeting_id], "duplicate historical meeting/topic"
        )
        topic_map[meeting_id][topic] = row
    roster = {
        str(row["meeting_id"]): row
        for row in _read_jsonl(HISTORICAL_ROSTER, label="historical official roster")
    }
    inventory = {
        str(row["meeting_id"]): row
        for row in _read_jsonl(
            HISTORICAL_MINUTES_INVENTORY, label="historical Minutes inventory"
        )
    }
    documents: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for meeting_id, meeting in by_id.items():
        _require(
            set(topic_map.get(meeting_id, {})) == set(CORE_TOPICS),
            f"Core8 closure failed: {meeting_id}",
        )
        official = roster.get(meeting_id)
        source = inventory.get(meeting_id)
        _require(
            isinstance(official, Mapping), f"official roster missing: {meeting_id}"
        )
        _require(isinstance(source, Mapping), f"official Minutes missing: {meeting_id}")
        for field in ("meeting_start_date", "meeting_end_date", "evidence_cutoff"):
            _require(
                meeting[field] == official[field],
                f"historical {field} drift: {meeting_id}",
            )
        _require(
            meeting["label_source"] == "official_minutes_directive",
            "historical label source drift",
        )
        _require(
            meeting["missing_record_fallback_used"] is False,
            "forbidden historical label fallback",
        )
        analyses: list[str] = []
        topic_hashes: list[dict[str, str]] = []
        for topic in CORE_TOPICS:
            row = topic_map[meeting_id][topic]
            _require(
                row["evidence_cutoff"] == meeting["evidence_cutoff"],
                f"historical cutoff drift: {meeting_id}",
            )
            value = str(row.get("source_analysis") or "").strip()
            _require(bool(value), f"empty historical analysis: {meeting_id}::{topic}")
            analyses.append(value)
            topic_hashes.append(
                {"topic": topic, "source_analysis_sha256": _sha256_text(value)}
            )
        raw_path = HISTORICAL_RELEASE / str(source["path"])
        _require(
            sha256_file(raw_path) == source["sha256"],
            f"historical official source drift: {meeting_id}",
        )
        documents.append(
            {
                **dict(meeting),
                "panel": "historical_n19",
                "analysis": "\n\n".join(analyses),
                "topic_order": list(CORE_TOPICS),
                "topic_source_hashes": topic_hashes,
                "official_source_url": source["url"],
                "official_source_sha256": source["sha256"],
            }
        )
        sources.append(
            {
                "panel": "historical_n19",
                "meeting_id": meeting_id,
                "source_path": raw_path,
                "url": source["url"],
                "sha256": source["sha256"],
                "meeting_start_date": meeting["meeting_start_date"],
                "meeting_end_date": meeting["meeting_end_date"],
                "direction": meeting["direction"],
                "magnitude_bp": meeting["magnitude_bp"],
            }
        )
    _require(len(documents) == 19, "historical document count drift")
    return documents, sources


def _validate_ledger_manifest() -> dict[str, Any]:
    manifest = _read_json(
        POSTCUTOFF_LEDGER_MANIFEST, label="post-cutoff ledger manifest"
    )
    integrity = manifest.get("integrity")
    _require(isinstance(integrity, Mapping), "post-cutoff ledger integrity missing")
    unsigned = dict(manifest)
    unsigned.pop("integrity", None)
    observed = _sha256_text(canonical_json(unsigned))
    _require(
        observed == integrity.get("payload_sha256"), "post-cutoff ledger payload drift"
    )
    _require(
        observed == POSTCUTOFF_LEDGER_PAYLOAD_SHA256, "unexpected post-cutoff ledger"
    )
    return manifest


def _historical_source_lineage_bindings() -> dict[str, Any]:
    release = _read_json(
        HISTORICAL_RELEASE / "release_manifest.json",
        label="historical release manifest",
    )
    lineage = release.get("source_lineage")
    _require(isinstance(lineage, Mapping), "historical source lineage missing")
    handoff = _read_json(HISTORICAL_SOURCE_HANDOFF, label="historical source handoff")
    _require(
        sha256_file(HISTORICAL_SOURCE_HANDOFF) == lineage.get("source_handoff_sha256")
        and handoff.get("payload_sha256")
        == lineage.get("source_handoff_payload_sha256"),
        "historical source handoff drift",
    )
    ledger_rows = handoff.get("ledgers")
    _require(
        isinstance(ledger_rows, list) and bool(ledger_rows),
        "historical source handoff has no ledgers",
    )
    ledger_records: dict[str, dict[str, Any]] = {}
    for row in ledger_rows:
        _require(isinstance(row, Mapping), "historical handoff ledger row invalid")
        ledger_dir = Path(str(row.get("ledger_dir") or ""))
        manifest_path = ledger_dir / "ledger_manifest.json"
        _require(
            sha256_file(manifest_path) == row.get("ledger_manifest_sha256"),
            f"historical ledger manifest drift: {ledger_dir.name}",
        )
        ledger_manifest = _read_json(
            manifest_path, label=f"historical ledger {ledger_dir.name}"
        )
        payload = ledger_manifest.get("integrity", {}).get("payload_sha256")
        if payload is None:
            payload = ledger_manifest.get("payload_sha256")
        _require(
            payload == row.get("ledger_payload_sha256"),
            f"historical ledger payload drift: {ledger_dir.name}",
        )
        ledger_records[ledger_dir.name] = _file_record(manifest_path)
    return {
        "source_handoff": _file_record(HISTORICAL_SOURCE_HANDOFF),
        "ledger_manifests": dict(sorted(ledger_records.items())),
    }


def _postcutoff_documents() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = _read_json(POSTCUTOFF_CONFIG, label="post-cutoff N12 config")
    meetings = config.get("meetings")
    _require(
        isinstance(meetings, list) and len(meetings) == 12, "post-cutoff N12 drift"
    )
    by_start = {str(row["meeting_start_date"]): row for row in meetings}
    _require(len(by_start) == 12, "duplicate post-cutoff meeting")
    _validate_ledger_manifest()
    rows = _read_jsonl(
        POSTCUTOFF_INDICATOR_INPUTS, label="post-cutoff indicator ledger"
    )
    topic_map: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        start = str(row.get("meeting_date") or "")
        if start not in by_start:
            continue
        topic = str(row.get("indicator") or "")
        if topic not in CORE_TOPICS:
            continue
        _require(topic not in topic_map[start], "duplicate post-cutoff meeting/topic")
        cutoff = str(by_start[start]["evidence_cutoff"])
        for field in (
            "requested_vintage_date",
            "availability_as_of_date",
            "information_as_of_date",
        ):
            _require(
                str(row.get(field) or "") == cutoff,
                f"post-cutoff D-1 drift: {start}::{topic}::{field}",
            )
        evidence = compress_indicator_row(row)
        _require(
            evidence is not None,
            f"cannot compress post-cutoff evidence: {start}::{topic}",
        )
        for series in evidence["provenance"]:
            _require(
                series["latest_observation_date"] <= cutoff,
                f"future observation: {start}::{topic}",
            )
        topic_map[start][topic] = {
            key: value for key, value in evidence.items() if key != "provenance"
        }
    documents: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    previous_range = POSTCUTOFF_INITIAL_TARGET_RANGE
    for start, meeting in sorted(by_start.items()):
        _require(
            set(topic_map.get(start, {})) == set(CORE_TOPICS),
            f"post-cutoff Core8 closure failed: {start}",
        )
        analyses = [
            _analysis_text(topic, topic_map[start][topic]) for topic in CORE_TOPICS
        ]
        _require(
            all(value.strip() for value in analyses),
            f"empty post-cutoff analysis: {start}",
        )
        previous_bounds = [float(value) for value in previous_range.split("--")]
        current_bounds = [
            float(value) for value in str(meeting["resulting_target_range"]).split("--")
        ]
        _require(
            len(previous_bounds) == len(current_bounds) == 2,
            f"invalid target-range encoding: {start}",
        )
        change_bp = round(
            100 * (statistics.fmean(current_bounds) - statistics.fmean(previous_bounds))
        )
        derived_direction = (
            "cut" if change_bp < 0 else "hike" if change_bp > 0 else "hold"
        )
        _require(
            derived_direction == meeting["direction"]
            and abs(change_bp) == int(meeting["magnitude_bp"]),
            f"post-cutoff target-range transition disagrees with label: {start}",
        )
        previous_range = str(meeting["resulting_target_range"])
        documents.append(
            {
                **dict(meeting),
                "meeting_id": f"fomc-{start.replace('-', '')}-{str(meeting['decision_date']).replace('-', '')}",
                "meeting_end_date": meeting["decision_date"],
                "panel": "postcutoff_n12",
                "analysis": "\n\n".join(analyses),
                "topic_order": list(CORE_TOPICS),
                "topic_source_hashes": [
                    {"topic": topic, "source_analysis_sha256": _sha256_text(value)}
                    for topic, value in zip(CORE_TOPICS, analyses, strict=True)
                ],
                "official_source_url": meeting["official_statement_url"],
                "label_source": "official_fomc_statement",
                "missing_record_fallback_used": False,
            }
        )
        sources.append(
            {
                "panel": "postcutoff_n12",
                "meeting_id": documents[-1]["meeting_id"],
                "url": meeting["official_statement_url"],
                "meeting_start_date": meeting["meeting_start_date"],
                "meeting_end_date": meeting["decision_date"],
                "direction": meeting["direction"],
                "magnitude_bp": meeting["magnitude_bp"],
                "resulting_target_range": meeting["resulting_target_range"],
            }
        )
    _require(len(documents) == 12, "post-cutoff document count drift")
    return documents, sources


def _build_samples(
    tokenizer: Any, system_prompt: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    historical, historical_sources = _historical_documents()
    postcutoff, postcutoff_sources = _postcutoff_documents()
    exposure = _decision_exposure_dates()
    samples: list[dict[str, Any]] = []
    for document in [*historical, *postcutoff]:
        identities = {
            str(document.get(key) or "")
            for key in ("meeting_start_date", "meeting_end_date", "decision_date")
        }
        _require(
            not (identities & exposure),
            f"direct Decision-release exposure: {document['meeting_id']}",
        )
        analysis = str(document.pop("analysis"))
        prompt = _make_user_prompt(analysis)
        _assert_no_target_leak(prompt=prompt, meeting=document, analysis=analysis)
        prompt_ids = _prompt_ids(tokenizer, _messages(system_prompt, prompt))
        _require(
            len(prompt_ids) <= MAX_PROMPT_TOKENS,
            f"prompt exceeds cap: {document['meeting_id']}",
        )
        sample_id = f"decision-core8::{document['panel']}::{document['meeting_id']}"
        samples.append(
            {
                "schema_version": SAMPLE_SCHEMA,
                "sample_id": sample_id,
                **document,
                "source_analysis": analysis,
                "source_analysis_sha256": _sha256_text(analysis),
                "prompt": prompt,
                "prompt_sha256": _sha256_text(prompt),
                "prompt_token_count": len(prompt_ids),
                "input_contract": "deterministic_core8_source_analysis_concatenation_v1",
            }
        )
    _require(
        Counter(row["panel"] for row in samples) == Counter(PANEL_COUNTS),
        "panel counts drift",
    )
    _require(
        len({row["sample_id"] for row in samples}) == 31, "duplicate external sample"
    )
    return samples, [*historical_sources, *postcutoff_sources]


def _official_text(raw: bytes) -> str:
    value = raw.decode("utf-8", errors="replace")
    value = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value)
    value = re.sub(r"(?s)<[^>]+>", " ", value)
    value = html.unescape(value)
    value = value.translate(str.maketrans({"‑": "-", "–": "-", "—": "-", "−": "-"}))
    return " ".join(value.split())


def _require_federal_reserve_url(url: str, *, meeting_id: str) -> None:
    host = (urlparse(url).hostname or "").lower()
    _require(
        host == "federalreserve.gov" or host.endswith(".federalreserve.gov"),
        f"official source is outside the Federal Reserve domain: {meeting_id}",
    )


def _rate_text_variants(value: float) -> set[str]:
    whole = int(value)
    fraction = round(value - whole, 2)
    decimals = {str(value), f"{value:.1f}", f"{value:.2f}"}
    if fraction == 0:
        decimals.add(str(whole))
    elif fraction == 0.25:
        decimals.add(f"{whole}-1/4")
    elif fraction == 0.5:
        decimals.add(f"{whole}-1/2")
    elif fraction == 0.75:
        decimals.add(f"{whole}-3/4")
    return decimals


def _validate_postcutoff_statement(
    raw: bytes, source: Mapping[str, Any], final_url: str
) -> dict[str, Any]:
    meeting_id = str(source["meeting_id"])
    _require_federal_reserve_url(str(source["url"]), meeting_id=meeting_id)
    _require_federal_reserve_url(final_url, meeting_id=meeting_id)
    decision_date = date.fromisoformat(str(source["meeting_end_date"]))
    compact_date = decision_date.strftime("%Y%m%d")
    _require(
        compact_date in str(source["url"]) and compact_date in final_url,
        f"official statement URL does not bind the decision date: {meeting_id}",
    )
    text = _official_text(raw)
    lower_text = text.lower()
    rendered_date = (
        f"{decision_date.strftime('%B')} {decision_date.day}, {decision_date.year}"
    )
    _require(
        rendered_date.lower() in lower_text,
        f"official statement body does not contain its decision date: {meeting_id}",
    )
    lower, upper = [
        float(value) for value in str(source["resulting_target_range"]).split("--")
    ]
    _require(
        any(value in lower_text for value in _rate_text_variants(lower))
        and any(value in lower_text for value in _rate_text_variants(upper)),
        f"official statement body does not contain the resulting target range: {meeting_id}",
    )
    direction = str(source["direction"])
    if direction == "hold":
        _require(
            "maintain the target range" in lower_text,
            f"official statement does not support a hold label: {meeting_id}",
        )
    else:
        verb = "lower" if direction == "cut" else "raise"
        _require(
            f"{verb} the target range" in lower_text,
            f"official statement does not support a {direction} label: {meeting_id}",
        )
        magnitude_phrase = {
            25: "1/4 percentage point",
            50: "1/2 percentage point",
            75: "3/4 percentage point",
            100: "1 percentage point",
        }.get(int(source["magnitude_bp"]))
        _require(
            magnitude_phrase is not None and magnitude_phrase in lower_text,
            f"official statement does not support the action magnitude: {meeting_id}",
        )
    return {
        "domain_check": "passed",
        "url_decision_date_check": "passed",
        "body_decision_date_check": "passed",
        "direction_check": "passed",
        "magnitude_check": "passed",
        "resulting_target_range_check": "passed",
        "method": "deterministic_official_statement_text_contract_v1",
    }


def _archive_official_sources(
    staging: Path, sources: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "fomc-trainer-research/1.0"})
    for source in sources:
        panel = str(source["panel"])
        meeting_id = str(source["meeting_id"])
        destination = staging / "sources/official" / panel / f"{meeting_id}.html"
        destination.parent.mkdir(parents=True, exist_ok=True)
        _require_federal_reserve_url(str(source["url"]), meeting_id=meeting_id)
        if panel == "historical_n19":
            source_path = Path(str(source["source_path"]))
            _require(
                sha256_file(source_path) == source["sha256"],
                f"historical source drift: {meeting_id}",
            )
            shutil.copyfile(source_path, destination)
            final_url = str(source["url"])
            status_code = None
            validation = {
                "domain_check": "passed",
                "source_hash_and_official_inventory_check": "passed",
                "meeting_identity_check": "passed_via_bound_official_roster_and_inventory",
                "direction_and_magnitude_check": "manual_official_directive_audit_bound_in_historical_config",
                "automated_historical_directive_extraction": "not_attempted_due_to_nonuniform_reserve_pressure_and_target_rate_language",
            }
        else:
            try:
                response = session.get(str(source["url"]), timeout=60)
                response.raise_for_status()
            except requests.RequestException as exc:
                raise ExternalCore8EvaluationError(
                    f"cannot archive official statement {meeting_id}: {exc}"
                ) from exc
            _require(
                len(response.content) >= 1000,
                f"official statement is unexpectedly short: {meeting_id}",
            )
            destination.write_bytes(response.content)
            final_url = response.url
            status_code = response.status_code
            validation = _validate_postcutoff_statement(
                response.content, source, final_url
            )
        result.append(
            {
                "panel": panel,
                "meeting_id": meeting_id,
                "requested_url": str(source["url"]),
                "final_url": final_url,
                "http_status": status_code,
                **_file_record(destination, relative_to=staging),
                "role": "gold_label_source_only_not_model_input",
                "validation": validation,
            }
        )
    _require(len(result) == 31, "official-source archive count drift")
    return result


def _sealed_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    return {
        **payload,
        "integrity": {
            "algorithm": "sha256(canonical-json-without-integrity)",
            "payload_sha256": _sha256_text(canonical_json(payload)),
        },
    }


def prepare(output_root: Path) -> dict[str, Any]:
    root = output_root.resolve()
    _require(not root.exists() and not root.is_symlink(), f"output root exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        system_prompt, _ = _training_contract()
        model_bindings = []
        for model in MODELS:
            observed = fingerprint_artifact_path(Path(model["path"]))
            _require(
                observed.get("sha256") == model["sha256"],
                f"model drift: {model['label']}",
            )
            model_bindings.append(
                {key: value for key, value in model.items() if key != "path"}
                | {"artifact": observed}
            )
        tokenizer_bundle = _shared_tokenizer_bundle_binding()
        tokenizer = _load_tokenizer(MODELS[-1])
        samples, source_specs = _build_samples(tokenizer, system_prompt)
        historical_source_lineage = _historical_source_lineage_bindings()
        runtime_prebinding = _runtime_prebinding()
        sample_path = staging / "inputs/panel_samples.jsonl"
        _write_exclusive_jsonl(sample_path, samples)
        official_sources = _archive_official_sources(staging, source_specs)
        official_path = staging / "sources/official_source_manifest.jsonl"
        _write_exclusive_jsonl(official_path, official_sources)
        manifest = _sealed_payload(
            {
                "schema_version": MANIFEST_SCHEMA,
                "status": "prepared",
                "created_at_utc": _utc_now(),
                "purpose": "separate_panel_greedy_decision_evaluation_without_retraining",
                "authorization_basis": "explicit_user_request_2026-08-24_no_retraining_external_extension",
                "output_root": str(root),
                "implementation": {
                    label: _file_record(path)
                    for label, path in sorted(IMPLEMENTATION_FILES.items())
                },
                "models": model_bindings,
                "inputs": {
                    "samples": _file_record(sample_path, relative_to=staging),
                    "historical_config": _file_record(HISTORICAL_CONFIG),
                    "postcutoff_config": _file_record(POSTCUTOFF_CONFIG),
                    "historical_release_manifest": _file_record(
                        HISTORICAL_RELEASE / "release_manifest.json"
                    ),
                    "historical_core8": _file_record(HISTORICAL_CORE8),
                    "historical_source_handoff": historical_source_lineage[
                        "source_handoff"
                    ],
                    "historical_ledger_manifests": historical_source_lineage[
                        "ledger_manifests"
                    ],
                    "postcutoff_ledger_manifest": _file_record(
                        POSTCUTOFF_LEDGER_MANIFEST
                    ),
                    "postcutoff_indicator_inputs": _file_record(
                        POSTCUTOFF_INDICATOR_INPUTS
                    ),
                    "official_sources": _file_record(
                        official_path, relative_to=staging
                    ),
                },
                "population": {
                    "panels": PANEL_COUNTS,
                    "class_counts": {
                        panel: dict(
                            sorted(
                                Counter(
                                    row["direction"]
                                    for row in samples
                                    if row["panel"] == panel
                                ).items()
                            )
                        )
                        for panel in PANELS
                    },
                    "direct_decision_release_overlap": 0,
                    "training_performed": False,
                },
                "input_contract": {
                    "name": "Deterministic Core8 Source-Analysis Decision Diagnostic",
                    "version": "deterministic_core8_source_analysis_concatenation_v1",
                    "topic_order": list(CORE_TOPICS),
                    "blocks_per_meeting": 8,
                    "separator": "two_newlines",
                    "information_cutoff": "official_meeting_start_date_minus_one_calendar_day",
                    "meeting_identity_in_prompt": False,
                    "gold_in_prompt": False,
                    "official_text_in_prompt": False,
                    "distinct_from_teacher_compressed_n13": True,
                    "byte_comparable_to_teacher_compressed_n13": False,
                },
                "prompt_contract": {
                    "training_config": _file_record(TRAINING_CONFIG),
                    "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
                    "user_prefix_sha256": USER_PREFIX_SHA256,
                    "shared_tokenizer_bundle": tokenizer_bundle,
                    "max_prompt_tokens": MAX_PROMPT_TOKENS,
                },
                "runtime_prebinding": runtime_prebinding,
                "generation": {
                    "mode": "greedy",
                    "completions_per_meeting_model": 1,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "batch_size": BATCH_SIZE,
                    "expected_rows": EXPECTED_RESULT_ROWS,
                    "post_run_receipt_required": True,
                    "load_in_4bit": True,
                    "dtype": "bfloat16",
                },
                "reporting": {
                    "panels_reported_separately": True,
                    "historical_primary_balanced_accuracy": "fixed_three_class",
                    "postcutoff_primary_balanced_accuracy": "supported_classes_cut_hold",
                    "pairwise_test": "exact_two_sided_mcnemar_direction_correctness",
                    "pairwise_multiplicity": "Holm adjustment within each panel's family of three model-pair comparisons",
                    "pooled_result_prohibited": True,
                },
                "limitations": [
                    "the input contract differs from the teacher-compressed N13 contract",
                    "historical public events may have been present in base-model pretraining",
                    "post-cutoff decisions are evaluated retrospectively rather than prospectively",
                    "postcutoff_n12 contains no hike meeting",
                    "small panels support diagnostics rather than strong model ranking",
                ],
            }
        )
        manifest_path = staging / "evaluation_manifest.json"
        _write_exclusive_json(manifest_path, manifest)
        os.rename(staging, root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "prepared",
        "manifest": _file_record(root / "evaluation_manifest.json"),
        "panels": PANEL_COUNTS,
        "expected_result_rows": EXPECTED_RESULT_ROWS,
    }


def validate_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    _require(sha256_file(path) == expected_sha256, "evaluation manifest SHA drift")
    value = _read_json(path, label="evaluation manifest")
    _require(value.get("schema_version") == MANIFEST_SCHEMA, "manifest schema drift")
    integrity = value.get("integrity")
    _require(isinstance(integrity, Mapping), "manifest integrity missing")
    unsigned = dict(value)
    unsigned.pop("integrity", None)
    _require(
        _sha256_text(canonical_json(unsigned)) == integrity.get("payload_sha256"),
        "manifest payload drift",
    )
    root = path.parent.resolve()
    _require(str(root) == value.get("output_root"), "manifest output root drift")
    expected_implementation = value.get("implementation")
    _require(
        isinstance(expected_implementation, Mapping)
        and set(expected_implementation) == set(IMPLEMENTATION_FILES),
        "implementation binding is missing or incomplete",
    )
    for label, implementation_path in IMPLEMENTATION_FILES.items():
        _require(
            _file_record(implementation_path) == expected_implementation[label],
            f"implementation drift: {label}",
        )
    for key in ("samples", "official_sources"):
        record = value["inputs"][key]
        observed = _file_record(root / record["path"], relative_to=root)
        _require(observed == record, f"prepared input drift: {key}")
    for key in (
        "historical_config",
        "postcutoff_config",
        "historical_release_manifest",
        "historical_core8",
        "historical_source_handoff",
        "postcutoff_ledger_manifest",
        "postcutoff_indicator_inputs",
    ):
        record = value["inputs"][key]
        observed = _file_record(Path(str(record["path"])))
        _require(observed == record, f"external input drift: {key}")
    ledger_records = value["inputs"].get("historical_ledger_manifests")
    _require(
        isinstance(ledger_records, Mapping) and bool(ledger_records),
        "historical ledger-manifest bindings missing",
    )
    for label, record in ledger_records.items():
        observed = _file_record(Path(str(record["path"])))
        _require(observed == record, f"historical ledger drift: {label}")
    official_rows = _read_jsonl(
        root / value["inputs"]["official_sources"]["path"],
        label="prepared official-source manifest",
    )
    _require(len(official_rows) == 31, "prepared official-source count drift")
    for record in official_rows:
        observed = _file_record(root / str(record["path"]), relative_to=root)
        for field in ("path", "bytes", "sha256"):
            _require(
                observed[field] == record[field],
                f"archived official source drift: {record.get('meeting_id')}",
            )
        validation = record.get("validation")
        _require(
            isinstance(validation, Mapping)
            and validation.get("domain_check") == "passed",
            f"official-source validation missing: {record.get('meeting_id')}",
        )
    _training_contract()
    _require(
        _shared_tokenizer_bundle_binding()
        == value.get("prompt_contract", {}).get("shared_tokenizer_bundle"),
        "shared tokenizer bundle drift",
    )
    _require(
        _runtime_prebinding() == value.get("runtime_prebinding"),
        "prebound software or hardware runtime drift",
    )
    for model in MODELS:
        _require(
            fingerprint_artifact_path(Path(model["path"])).get("sha256")
            == model["sha256"],
            f"model drift: {model['label']}",
        )
    return value


def _chunks(
    values: Sequence[Mapping[str, Any]], size: int
) -> list[list[Mapping[str, Any]]]:
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def _batch_seed(model: str, panel: str, index: int) -> int:
    text = f"chk4-external-core8-v1:{model}:{panel}:{index}"
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**31)


def batch_contracts(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for model in MODELS:
        for panel in PANELS:
            values = [row for row in samples if row["panel"] == panel]
            for index, batch in enumerate(_chunks(values, BATCH_SIZE)):
                result.append(
                    {
                        "model_label": model["label"],
                        "panel": panel,
                        "batch_index": index,
                        "sample_ids": [row["sample_id"] for row in batch],
                        "seed": _batch_seed(model["label"], panel, index),
                    }
                )
    return result


def _batch_path(run_root: Path, contract: Mapping[str, Any]) -> Path:
    return (
        run_root
        / "batches"
        / str(contract["model_label"])
        / f"{contract['panel']}_{int(contract['batch_index']):02d}.jsonl"
    )


def _eos_ids_for_model(
    model: Mapping[str, Any], tokenizer: Any, final_protocol: Any
) -> set[int]:
    generation_config = _read_json(
        Path(model["path"]) / "generation_config.json",
        label=f"{model['label']} generation config",
    )
    eos_value = generation_config.get("eos_token_id")
    if eos_value is None:
        eos_value = tokenizer.eos_token_id
    eos_ids = set(final_protocol.generation_probe._normalize_eos_ids(eos_value))
    _require(bool(eos_ids), f"{model['label']} has no bound EOS token")
    vocabulary_size = len(tokenizer)
    _require(
        all(0 <= value < vocabulary_size for value in eos_ids),
        f"{model['label']} EOS token is outside tokenizer vocabulary",
    )
    return eos_ids


def _fixed_result_metadata(
    model: Mapping[str, Any], sample: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA,
        "model_label": model["label"],
        "model_display_name": model["display_name"],
        "paper_model": model["paper_model"],
        "training_state": model["training_state"],
        "checkpoint_step": model["checkpoint_step"],
        "panel": sample["panel"],
        "split": "external_evaluation",
        "meeting_id": sample["meeting_id"],
        "meeting_start_date": sample["meeting_start_date"],
        "result_origin": "fresh_greedy_generation",
    }


def _assert_exact_replay(
    stored: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    missing_material = sorted(set(MATERIAL_REPLAY_FIELDS) - set(expected))
    _require(
        not missing_material,
        "bound replay omitted material fields: " + ",".join(missing_material),
    )
    differing = sorted(
        key
        for key in set(stored) | set(expected)
        if key not in stored
        or key not in expected
        or canonical_json(stored[key]) != canonical_json(expected[key])
    )
    _require(
        not differing,
        "result row differs from raw-token replay: " + ",".join(differing),
    )


def _replay_result_row(
    *,
    stored: Mapping[str, Any],
    sample: Mapping[str, Any],
    model: Mapping[str, Any],
    contract: Mapping[str, Any],
    manifest_sha: str,
    tokenizer: Any,
    eos_ids: set[int],
    final_protocol: Any,
) -> dict[str, Any]:
    raw_ids = stored.get("generated_token_ids_raw_padded")
    _require(isinstance(raw_ids, list) and bool(raw_ids), "raw token IDs missing")
    _require(
        len(raw_ids) <= MAX_NEW_TOKENS,
        "raw completion exceeds the sealed completion cap",
    )
    vocabulary_size = len(tokenizer)
    _require(
        all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value < vocabulary_size
            for value in raw_ids
        ),
        "raw completion contains a non-integer or out-of-vocabulary token ID",
    )
    expected = final_protocol._result_row(
        manifest_sha=manifest_sha,
        merged_sha=str(model["sha256"]),
        sample=sample,
        mode="greedy",
        generation_index=0,
        batch_index=int(contract["batch_index"]),
        seed=int(contract["seed"]),
        raw_ids=raw_ids,
        eos_ids=eos_ids,
        tokenizer=tokenizer,
    )
    expected.update(_fixed_result_metadata(model, sample))
    _assert_exact_replay(stored, expected)
    return expected


def _validate_batch(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    manifest_sha: str,
    sample_by_id: Mapping[str, Mapping[str, Any]],
    tokenizer: Any,
    eos_ids: set[int],
    final_protocol: Any,
) -> list[dict[str, Any]]:
    _require(len(rows) == len(contract["sample_ids"]), "batch row count drift")
    _require(
        [row.get("sample_id") for row in rows] == contract["sample_ids"],
        "batch sample order drift",
    )
    model = _model_by_label(str(contract["model_label"]))
    replayed_rows: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id"))
        _require(sample_id in sample_by_id, "result sample is outside prepared panel")
        sample = sample_by_id[sample_id]
        _require(row.get("schema_version") == RESULT_SCHEMA, "result schema drift")
        _require(row.get("model_label") == model["label"], "result model drift")
        _require(
            row.get("merged_model_sha256") == model["sha256"], "result model SHA drift"
        )
        _require(row.get("panel") == contract["panel"], "result panel drift")
        _require(row.get("split") == "external_evaluation", "result split-role drift")
        _require(row.get("generation_mode") == "greedy", "result mode drift")
        _require(row.get("generation_index") == 0, "result generation index drift")
        _require(
            row.get("batch_index") == contract["batch_index"]
            and row.get("batch_seed") == contract["seed"],
            "result batch index/seed drift",
        )
        _require(
            row.get("checkpoint_step") == model["checkpoint_step"],
            "result checkpoint drift",
        )
        _require(
            row.get("prompt_sha256") == sample["prompt_sha256"]
            and row.get("target_direction") == sample["direction"]
            and row.get("target_magnitude_bp") == sample["magnitude_bp"],
            "result prompt/target drift",
        )
        _require(
            row.get("evaluation_manifest_sha256") == manifest_sha,
            "result manifest drift",
        )
        replayed_rows.append(
            _replay_result_row(
                stored=row,
                sample=sample,
                model=model,
                contract=contract,
                manifest_sha=manifest_sha,
                tokenizer=tokenizer,
                eos_ids=eos_ids,
                final_protocol=final_protocol,
            )
        )
    return replayed_rows


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


@contextmanager
def _exclusive_lock(output_root: Path) -> Iterable[None]:
    lock_path = output_root / ".external_core8.lock"
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ExternalCore8EvaluationError("evaluation lock is held") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_integrity(value: Mapping[str, Any], *, label: str) -> None:
    integrity = value.get("integrity")
    _require(isinstance(integrity, Mapping), f"{label} integrity missing")
    unsigned = dict(value)
    unsigned.pop("integrity", None)
    _require(
        integrity.get("payload_sha256") == _sha256_text(canonical_json(unsigned)),
        f"{label} payload drift",
    )


def _run_receipt_payload(
    *,
    root: Path,
    manifest_sha: str,
    contracts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    run_root = root / "run"
    batch_records: dict[str, dict[str, Any]] = {}
    for contract in contracts:
        path = _batch_path(run_root, contract)
        relative = path.relative_to(root).as_posix()
        _require(relative not in batch_records, "duplicate batch receipt path")
        batch_records[relative] = _file_record(path, relative_to=root)
    _require(
        sum(int(record.get("rows", 0)) for record in batch_records.values())
        == EXPECTED_RESULT_ROWS,
        "run receipt batch-row closure drift",
    )
    return {
        "schema_version": RUN_RECEIPT_SCHEMA,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "evaluation_manifest_sha256": manifest_sha,
        "batch_contract": _file_record(
            run_root / "batch_contract.json", relative_to=root
        ),
        "runtime": _file_record(run_root / "runtime.json", relative_to=root),
        "results": _file_record(run_root / "results.jsonl", relative_to=root),
        "batches": dict(sorted(batch_records.items())),
    }


def _validate_run_receipt(
    *,
    root: Path,
    manifest_sha: str,
    contracts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    path = root / "run/run_receipt.json"
    value = _read_json(path, label="post-run receipt")
    _require(
        value.get("schema_version") == RUN_RECEIPT_SCHEMA
        and value.get("status") == "complete"
        and value.get("evaluation_manifest_sha256") == manifest_sha,
        "post-run receipt role drift",
    )
    _validate_integrity(value, label="post-run receipt")
    expected_paths = {
        _batch_path(root / "run", contract).relative_to(root).as_posix()
        for contract in contracts
    }
    batch_records = value.get("batches")
    _require(
        isinstance(batch_records, Mapping) and set(batch_records) == expected_paths,
        "post-run receipt batch closure drift",
    )
    for relative, record in batch_records.items():
        _require(
            _file_record(root / relative, relative_to=root) == record,
            f"post-run receipt batch drift: {relative}",
        )
    for key, relative in (
        ("batch_contract", "run/batch_contract.json"),
        ("runtime", "run/runtime.json"),
        ("results", "run/results.jsonl"),
    ):
        _require(
            _file_record(root / relative, relative_to=root) == value.get(key),
            f"post-run receipt {key} drift",
        )
    _require(
        int(value["results"].get("rows", 0)) == EXPECTED_RESULT_ROWS,
        "post-run receipt result-row drift",
    )
    return value


def _replay_all_batches(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    manifest_sha: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples = _read_jsonl(
        root / manifest["inputs"]["samples"]["path"],
        label="prepared panel samples",
    )
    sample_by_id = {str(row["sample_id"]): row for row in samples}
    _require(len(sample_by_id) == len(samples) == 31, "prepared sample closure drift")
    contracts = batch_contracts(samples)
    contract_value = {"schema_version": MANIFEST_SCHEMA, "batches": contracts}
    _require(
        _read_json(root / "run/batch_contract.json", label="batch contract")
        == contract_value,
        "batch contract replay drift",
    )
    final_protocol = _final_protocol()
    tokenizer = _load_tokenizer(MODELS[-1])
    eos_by_model = {
        str(model["label"]): _eos_ids_for_model(model, tokenizer, final_protocol)
        for model in MODELS
    }
    rows: list[dict[str, Any]] = []
    for contract in contracts:
        batch_rows = _read_jsonl(
            _batch_path(root / "run", contract), label="replayed result batch"
        )
        rows.extend(
            _validate_batch(
                batch_rows,
                contract,
                manifest_sha,
                sample_by_id,
                tokenizer,
                eos_by_model[str(contract["model_label"])],
                final_protocol,
            )
        )
    _require(len(rows) == EXPECTED_RESULT_ROWS, "replayed result-row closure drift")
    stored_results = _read_jsonl(
        root / "run/results.jsonl", label="assembled external Core8 results"
    )
    _require(
        canonical_json(stored_results) == canonical_json(rows),
        "assembled results differ from independently replayed batches",
    )
    return rows, contracts


def run(manifest_path: Path, manifest_sha: str) -> dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    manifest = validate_manifest(manifest_path, manifest_sha)
    root = manifest_path.parent
    samples = _read_jsonl(
        root / manifest["inputs"]["samples"]["path"], label="prepared panel samples"
    )
    sample_by_id = {str(row["sample_id"]): row for row in samples}
    contracts = batch_contracts(samples)
    run_root = root / "run"
    run_root.mkdir(parents=True, exist_ok=True)
    contract_path = run_root / "batch_contract.json"
    contract_value = {"schema_version": MANIFEST_SCHEMA, "batches": contracts}
    if contract_path.exists():
        _require(
            _read_json(contract_path, label="batch contract") == contract_value,
            "batch contract drift",
        )
    else:
        _write_exclusive_json(contract_path, contract_value)
    system_prompt, _ = _training_contract()
    import torch
    import transformers

    _configure_deterministic_runtime(torch)
    final_protocol = _final_protocol()
    replay_tokenizer = _load_tokenizer(MODELS[-1])
    replay_eos_ids = {
        str(model["label"]): _eos_ids_for_model(model, replay_tokenizer, final_protocol)
        for model in MODELS
    }
    _require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "run must expose exactly one GPU",
    )
    runtime = _runtime_record(torch)
    for key, expected in manifest["runtime_prebinding"].items():
        _require(runtime.get(key) == expected, f"run-time prebinding drift: {key}")
    required_runtime = manifest["runtime_prebinding"]["required_execution"]
    _require(
        runtime["environment"]["PYTHONNOUSERSITE"]
        == required_runtime["python_no_user_site"],
        "PYTHONNOUSERSITE runtime drift",
    )
    for variable, expected in required_runtime["thread_environment"].items():
        _require(
            runtime["environment"][variable] == expected,
            f"thread runtime drift: {variable}",
        )
    with _exclusive_lock(root):
        runtime_path = run_root / "runtime.json"
        if runtime_path.exists():
            _require(
                _read_json(runtime_path, label="frozen run-time record") == runtime,
                "frozen run-time record drift",
            )
        else:
            _write_exclusive_json(runtime_path, runtime)
        for model_spec in MODELS:
            pending = [
                contract
                for contract in contracts
                if contract["model_label"] == model_spec["label"]
                and not _batch_path(run_root, contract).exists()
            ]
            if not pending:
                continue
            tokenizer = _load_tokenizer(model_spec)
            model = None
            try:
                model = _load_model(Path(model_spec["path"]))
                eos_value = (
                    getattr(model.generation_config, "eos_token_id", None)
                    or tokenizer.eos_token_id
                )
                eos_ids = set(
                    final_protocol.generation_probe._normalize_eos_ids(eos_value)
                )
                _require(bool(eos_ids), "model/tokenizer has no EOS")
                _require(
                    eos_ids == replay_eos_ids[str(model_spec["label"])],
                    f"loaded-model EOS drift: {model_spec['label']}",
                )
                for contract in [
                    value
                    for value in contracts
                    if value["model_label"] == model_spec["label"]
                ]:
                    path = _batch_path(run_root, contract)
                    if path.exists():
                        _validate_batch(
                            _read_jsonl(path, label="existing batch"),
                            contract,
                            manifest_sha,
                            sample_by_id,
                            tokenizer,
                            eos_ids,
                            final_protocol,
                        )
                        continue
                    transformers.set_seed(int(contract["seed"]))
                    batch = [
                        sample_by_id[str(sample_id)]
                        for sample_id in contract["sample_ids"]
                    ]
                    rendered = [
                        tokenizer.apply_chat_template(
                            _messages(system_prompt, str(sample["prompt"])),
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        for sample in batch
                    ]
                    encoded = tokenizer(
                        rendered,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    input_ids = encoded["input_ids"].to("cuda:0")
                    attention_mask = encoded["attention_mask"].to("cuda:0")
                    for index, sample in enumerate(batch):
                        _require(
                            int(attention_mask[index].sum())
                            == sample["prompt_token_count"],
                            f"prompt token drift: {sample['sample_id']}",
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
                    width = int(input_ids.shape[1])
                    rows = []
                    for index, sample in enumerate(batch):
                        row = final_protocol._result_row(
                            manifest_sha=manifest_sha,
                            merged_sha=str(model_spec["sha256"]),
                            sample=sample,
                            mode="greedy",
                            generation_index=0,
                            batch_index=int(contract["batch_index"]),
                            seed=int(contract["seed"]),
                            raw_ids=sequences[index, width:].tolist(),
                            eos_ids=eos_ids,
                            tokenizer=tokenizer,
                        )
                        row.update(_fixed_result_metadata(model_spec, sample))
                        rows.append(row)
                    _validate_batch(
                        rows,
                        contract,
                        manifest_sha,
                        sample_by_id,
                        tokenizer,
                        eos_ids,
                        final_protocol,
                    )
                    _write_exclusive_jsonl(path, rows)
                    print(
                        canonical_json(
                            {
                                "status": "batch_complete",
                                "model": model_spec["label"],
                                "panel": contract["panel"],
                                "batch_index": contract["batch_index"],
                                "rows": len(rows),
                            }
                        ),
                        flush=True,
                    )
                    del input_ids, attention_mask, sequences
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()
        rows: list[dict[str, Any]] = []
        for contract in contracts:
            batch_rows = _read_jsonl(
                _batch_path(run_root, contract), label="completed batch"
            )
            rows.extend(
                _validate_batch(
                    batch_rows,
                    contract,
                    manifest_sha,
                    sample_by_id,
                    replay_tokenizer,
                    replay_eos_ids[str(contract["model_label"])],
                    final_protocol,
                )
            )
        _require(len(rows) == EXPECTED_RESULT_ROWS, "result closure drift")
        result_path = run_root / "results.jsonl"
        if result_path.exists():
            _require(
                _read_jsonl(result_path, label="assembled results") == rows,
                "assembled results drift",
            )
        else:
            _write_exclusive_jsonl(result_path, rows)
        receipt_path = run_root / "run_receipt.json"
        if receipt_path.exists():
            _validate_run_receipt(
                root=root, manifest_sha=manifest_sha, contracts=contracts
            )
        else:
            _write_exclusive_json(
                receipt_path,
                _sealed_payload(
                    _run_receipt_payload(
                        root=root,
                        manifest_sha=manifest_sha,
                        contracts=contracts,
                    )
                ),
            )
            _validate_run_receipt(
                root=root, manifest_sha=manifest_sha, contracts=contracts
            )
    return {
        "status": "generation_complete",
        "results": _file_record(root / "run/results.jsonl", relative_to=root),
        "run_receipt": _file_record(root / "run/run_receipt.json", relative_to=root),
        "rows": EXPECTED_RESULT_ROWS,
    }


def _prediction_label(row: Mapping[str, Any]) -> str:
    prediction = row.get("decision_prediction")
    if isinstance(prediction, Mapping) and prediction.get("direction") in DIRECTIONS:
        return str(prediction["direction"])
    return "invalid"


def score_block(
    rows: Sequence[Mapping[str, Any]], *, supported_only: bool
) -> dict[str, Any]:
    _require(bool(rows), "cannot score empty panel/model block")
    matrix = {
        actual: {guess: 0 for guess in PREDICTION_LABELS} for actual in DIRECTIONS
    }
    for row in rows:
        actual = str(row["target_direction"])
        _require(actual in DIRECTIONS, "invalid target direction")
        matrix[actual][_prediction_label(row)] += 1
    per_class = {}
    for label in DIRECTIONS:
        support = sum(matrix[label].values())
        correct = matrix[label][label]
        per_class[label] = {
            "support": support,
            "correct": correct,
            "recall": correct / support if support else None,
        }
    supported = [label for label in DIRECTIONS if per_class[label]["support"]]
    supported_ba = statistics.fmean(
        float(per_class[label]["recall"]) for label in supported
    )
    mechanical_zero_insertion = statistics.fmean(
        float(per_class[label]["recall"] or 0.0) for label in DIRECTIONS
    )
    fixed_three = (
        mechanical_zero_insertion if len(supported) == len(DIRECTIONS) else None
    )
    if not supported_only:
        _require(fixed_three is not None, "fixed-three-class BA requires all classes")
    return {
        "cases": len(rows),
        "direction_correct": sum(matrix[label][label] for label in DIRECTIONS),
        "direction_accuracy": sum(matrix[label][label] for label in DIRECTIONS)
        / len(rows),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "supported_classes": supported,
        "supported_class_balanced_accuracy": supported_ba,
        "fixed_three_class_balanced_accuracy": fixed_three,
        "mechanical_zero_insertion_balanced_accuracy": mechanical_zero_insertion,
        "mechanical_zero_insertion_is_estimand": fixed_three is not None,
        "primary_balanced_accuracy_definition": "supported_classes"
        if supported_only
        else "fixed_three_class",
        "primary_balanced_accuracy": supported_ba if supported_only else fixed_three,
    }


def exact_mcnemar(
    rows_a: Sequence[Mapping[str, Any]], rows_b: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_a = {str(row["sample_id"]): row for row in rows_a}
    by_b = {str(row["sample_id"]): row for row in rows_b}
    _require(set(by_a) == set(by_b) and bool(by_a), "McNemar pairing drift")
    a_only = b_only = both = neither = 0
    for sample_id in sorted(by_a):
        a = bool(by_a[sample_id].get("decision_direction_correct"))
        b = bool(by_b[sample_id].get("decision_direction_correct"))
        if a and b:
            both += 1
        elif a:
            a_only += 1
        elif b:
            b_only += 1
        else:
            neither += 1
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, value) for value in range(min(a_only, b_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "both_correct": both,
        "neither_correct": neither,
        "discordant_pairs": discordant,
        "test": "exact_two_sided_mcnemar",
        "p_value_raw": p_value,
    }


def holm_adjust_pairwise(
    comparisons: Sequence[Mapping[str, Any]], *, panel: str
) -> list[dict[str, Any]]:
    _require(panel in PANELS, "Holm family has an unknown panel")
    _require(len(comparisons) == 3, "Holm family must contain three comparisons")
    family = "within_panel_three_pairwise_model_comparisons"
    ordered = sorted(
        enumerate(comparisons),
        key=lambda value: (float(value[1]["p_value_raw"]), value[0]),
    )
    adjusted = [0.0] * len(comparisons)
    running_maximum = 0.0
    family_size = len(comparisons)
    for rank, (original_index, comparison) in enumerate(ordered):
        candidate = min(
            1.0,
            (family_size - rank) * float(comparison["p_value_raw"]),
        )
        running_maximum = max(running_maximum, candidate)
        adjusted[original_index] = running_maximum
    return [
        {
            **dict(comparison),
            "p_value_holm": adjusted[index],
            "holm_family": family,
            "holm_family_panel": panel,
            "holm_family_size": family_size,
        }
        for index, comparison in enumerate(comparisons)
    ]


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _markdown_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Deterministic Core8 Source-Analysis Decision Diagnostics",
        "",
        "## Result",
        "",
        "The two panels are reported separately. They use the same frozen eight-block source-analysis input contract but are not byte-equivalent to the teacher-compressed input used in the opened N=13 Decision test.",
        "",
    ]
    for panel, title in (
        ("historical_n19", "Historical Missing-Meeting Panel (N=19)"),
        ("postcutoff_n12", "Post-Cutoff Regular-Meeting Panel (N=12)"),
    ):
        lines += [
            f"## {title}",
            "",
            "| Model/state | Direction accuracy | Cut recall | Hold recall | Hike recall | Balanced accuracy |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for model in MODELS:
            block = summary["panels"][panel]["models"][model["label"]]
            recall = block["per_class"]

            def rendered(label: str) -> str:
                if recall[label]["recall"] is None:
                    return "N/A"
                return (
                    f"{recall[label]['correct']}/{recall[label]['support']} = "
                    f"{_pct(recall[label]['recall'])}"
                )

            ba_label = (
                "supported-class" if panel == "postcutoff_n12" else "fixed-three-class"
            )
            lines.append(
                f"| {model['display_name']} | {block['direction_correct']}/{block['cases']} = {_pct(block['direction_accuracy'])} | {rendered('cut')} | {rendered('hold')} | {rendered('hike')} | {_pct(block['primary_balanced_accuracy'])} ({ba_label}) |"
            )
        lines += [
            "",
            "Confusion matrices use rows as official directions and columns as predicted `cut`, `hold`, `hike`, and `invalid`.",
            "",
        ]
        for model in MODELS:
            matrix = summary["panels"][panel]["models"][model["label"]][
                "confusion_matrix"
            ]
            lines += [
                f"### {model['display_name']}",
                "",
                "| Actual | Cut | Hold | Hike | Invalid |",
                "|---|---:|---:|---:|---:|",
            ]
            for actual in DIRECTIONS:
                lines.append(
                    f"| {actual.title()} | {matrix[actual]['cut']} | {matrix[actual]['hold']} | {matrix[actual]['hike']} | {matrix[actual]['invalid']} |"
                )
            lines.append("")
        lines += [
            "Exact paired tests:",
            "",
            "| Model A | Model B | A only correct | B only correct | Exact McNemar p (raw) | Holm-adjusted p |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for comparison in summary["panels"][panel]["pairwise"]:
            lines.append(
                f"| {comparison['model_a_display']} | {comparison['model_b_display']} | {comparison['a_only_correct']} | {comparison['b_only_correct']} | {comparison['p_value_raw']:.6f} | {comparison['p_value_holm']:.6f} |"
            )
        lines.append("")
    lines += [
        "## Interpretation constraints",
        "",
        "- `postcutoff_n12` contains three cuts, nine holds, and no hikes; its primary balanced accuracy therefore averages cut and hold recall only. A zero inserted for the absent hike class is not used as the headline statistic.",
        "- `historical_n19` contains all three classes and uses fixed-three-class balanced accuracy.",
        "- Official Minutes/statements are label sources only and never enter the model prompt.",
        "- The 19 historical events are public and may have appeared in base-model pretraining. The 12 post-cutoff predictions are retrospective, not prospectively sealed.",
        "- Exact McNemar tests address paired direction correctness within each small frozen panel. The three model-pair p-values are Holm-adjusted within that panel; there is no cross-panel family. A non-significant result does not establish equivalence.",
        "- No pooled N=31 or pooled-with-N=13 performance estimate is authorized.",
        "",
    ]
    return "\n".join(lines)


def _latex_table(summary: Mapping[str, Any]) -> str:
    lines = [
        "% Generated by evaluate_chk4_external_core8_panels.py; do not edit by hand.",
        "\\begin{table}[H]",
        "    \\centering",
        "    \\scriptsize",
        "    \\caption{Decision Accuracy on the Separate Deterministic Core8 Source-Analysis Panels}",
        "    \\label{tab:ch2:decision_external_core8_panels}",
        "    \\begin{tabular}{llrrrrr}",
        "        \\toprule",
        "        Panel & Model/state & Direction & Cut recall & Hold recall & Hike recall & Balanced accuracy \\\\",
        "        \\midrule",
    ]
    for panel in PANELS:
        panel_label = (
            "Historical $N=19$" if panel == "historical_n19" else "Post-cutoff $N=12$"
        )
        for index, model in enumerate(MODELS):
            block = summary["panels"][panel]["models"][model["label"]]

            def recall(label: str) -> str:
                value = block["per_class"][label]
                return (
                    "--"
                    if value["recall"] is None
                    else f"{100 * value['recall']:.2f}\\%"
                )

            lines.append(
                "        "
                + " & ".join(
                    [
                        panel_label if index == 0 else "",
                        model["display_name"],
                        f"{100 * block['direction_accuracy']:.2f}\\%",
                        recall("cut"),
                        recall("hold"),
                        recall("hike"),
                        f"{100 * block['primary_balanced_accuracy']:.2f}\\%",
                    ]
                )
                + " \\\\"
            )
        if panel == "historical_n19":
            lines.append("        \\addlinespace")
    lines += [
        "        \\bottomrule",
        "    \\end{tabular}",
        "    \\begin{minipage}{0.98\\textwidth}",
        "    \\footnotesize \\textit{Notes:} The historical panel reports fixed-three-class balanced accuracy. The post-cutoff panel contains no hikes, so its balanced accuracy averages cut and hold recall only. Inputs concatenate exactly eight deterministic point-in-time source analyses. This contract differs from the teacher-compressed $N=13$ Decision test.",
        "    \\end{minipage}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def score(manifest_path: Path, manifest_sha: str) -> dict[str, Any]:
    manifest = validate_manifest(manifest_path, manifest_sha)
    root = manifest_path.parent
    samples = _read_jsonl(
        root / manifest["inputs"]["samples"]["path"],
        label="prepared panel samples",
    )
    contracts = batch_contracts(samples)
    receipt = _validate_run_receipt(
        root=root, manifest_sha=manifest_sha, contracts=contracts
    )
    rows, replayed_contracts = _replay_all_batches(
        root=root,
        manifest=manifest,
        manifest_sha=manifest_sha,
    )
    _require(replayed_contracts == contracts, "score replay contract drift")
    _require(len(rows) == EXPECTED_RESULT_ROWS, "score result count drift")
    _require(
        len({(row["model_label"], row["sample_id"]) for row in rows})
        == EXPECTED_RESULT_ROWS,
        "duplicate scored result",
    )
    expected_keys = {
        (str(model["label"]), str(sample["sample_id"]))
        for model in MODELS
        for sample in samples
    }
    _require(
        {(str(row["model_label"]), str(row["sample_id"])) for row in rows}
        == expected_keys,
        "scored result population drift",
    )
    panels: dict[str, Any] = {}
    for panel in PANELS:
        by_model = {
            model["label"]: [
                row
                for row in rows
                if row["panel"] == panel and row["model_label"] == model["label"]
            ]
            for model in MODELS
        }
        for model in MODELS:
            _require(
                len(by_model[model["label"]]) == PANEL_COUNTS[panel],
                f"score panel closure drift: {panel}/{model['label']}",
            )
        pairwise = []
        for model_a, model_b in itertools.combinations(MODELS, 2):
            comparison = exact_mcnemar(
                by_model[model_a["label"]], by_model[model_b["label"]]
            )
            pairwise.append(
                {
                    "model_a": model_a["label"],
                    "model_a_display": model_a["display_name"],
                    "model_b": model_b["label"],
                    "model_b_display": model_b["display_name"],
                    **comparison,
                }
            )
        pairwise = holm_adjust_pairwise(pairwise, panel=panel)
        panels[panel] = {
            "population": {
                "meetings": PANEL_COUNTS[panel],
                "class_counts": dict(
                    sorted(
                        Counter(
                            row["target_direction"]
                            for row in by_model[MODELS[0]["label"]]
                        ).items()
                    )
                ),
            },
            "models": {
                model["label"]: score_block(
                    by_model[model["label"]], supported_only=panel == "postcutoff_n12"
                )
                for model in MODELS
            },
            "pairwise": pairwise,
        }
    summary = _sealed_payload(
        {
            "schema_version": SUMMARY_SCHEMA,
            "status": "complete",
            "created_at_utc": _utc_now(),
            "evaluation_manifest_sha256": manifest_sha,
            "results": _file_record(root / "run/results.jsonl", relative_to=root),
            "run_receipt": _file_record(
                root / "run/run_receipt.json", relative_to=root
            ),
            "raw_token_replay_verification": {
                "status": "passed",
                "rows_replayed": len(rows),
                "source": "generated_token_ids_raw_padded",
                "stored_derived_fields_trusted": False,
                "material_fields_checked": list(MATERIAL_REPLAY_FIELDS),
                "receipt_payload_sha256": receipt["integrity"]["payload_sha256"],
            },
            "panels": panels,
            "pooled_result": None,
            "interpretation": {
                "input_contract": manifest["input_contract"],
                "historical_n19": "historical external diagnostic; no pretraining-independence claim",
                "postcutoff_n12": "retrospective post-training-data-cutoff chronological diagnostic; no hike support",
                "teacher_compressed_n13": "separate opened diagnostic; not byte-comparable and not pooled",
            },
        }
    )
    report_root = root / "report"
    _require(not report_root.exists(), "score/report output directory already exists")
    staging = Path(tempfile.mkdtemp(prefix=".report.", dir=root))
    try:
        summary_path = staging / "summary.json"
        report_path = staging / "technical_report.md"
        latex_path = staging / "results_table.tex"
        _write_exclusive_json(summary_path, summary)
        _write_exclusive_text(report_path, _markdown_report(summary))
        _write_exclusive_text(latex_path, _latex_table(summary))
        os.rename(staging, report_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "scored",
        "summary": _file_record(report_root / "summary.json", relative_to=root),
        "report": _file_record(report_root / "technical_report.md", relative_to=root),
        "latex_table": _file_record(
            report_root / "results_table.tex", relative_to=root
        ),
    }


def status(output_root: Path) -> dict[str, Any]:
    root = output_root.resolve()
    batches = (
        list((root / "run/batches").glob("*/*.jsonl"))
        if (root / "run/batches").is_dir()
        else []
    )
    return {
        "output_root": str(root),
        "prepared": (root / "evaluation_manifest.json").is_file(),
        "batches_complete": len(batches),
        "generation_complete": (root / "run/results.jsonl").is_file(),
        "scored": (root / "report/summary.json").is_file(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "score", "status"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.stage == "prepare":
        result = prepare(args.output_root)
    elif args.stage == "status":
        result = status(args.output_root)
    else:
        manifest = args.manifest or args.output_root / "evaluation_manifest.json"
        _require(bool(args.manifest_sha256), "--manifest-sha256 is required")
        result = (
            run(manifest, args.manifest_sha256)
            if args.stage == "run"
            else score(manifest, args.manifest_sha256)
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
