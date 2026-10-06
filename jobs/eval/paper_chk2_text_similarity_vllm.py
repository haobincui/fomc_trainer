"""Two-GPU vLLM generation worker for the paper chk-2 text-similarity audit.

This module owns only generation and transport validation.  Preparation and
semantic/style scoring live outside it.  The fixed experiment contains 391
prepared samples, ten paired stochastic replicates, and three model views:

* ``chk0`` is loaded as a separate full model;
* ``chk1`` is the exact chk-1 full model with no adapter request; and
* ``chk2`` is the same chk-1 engine with the checkpoint-50 LoRA request.

Two independent DP=1/TP=1 processes are used, one per physical GPU.  A
``(sample_id, replicate_index)`` pair is assigned to a shard without using the
model ID.  Consequently all three model outputs for a paired comparison are
always generated on the same physical GPU.  Every completed tuple is appended
and fsynced to a WAL before the next completion is acknowledged; resume
regenerates only missing tuple keys.

The worker never terminates foreign GPU processes.  Its vLLM memory fraction is
computed from the live free memory as ``floor_0.01((free-2048)/total)`` and it
fails closed below 0.78.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import gc
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO


ROOT = Path(__file__).resolve().parents[2]

SAMPLE_COUNT = 391
REPLICATES = 10
MODEL_IDS = ("chk0", "chk1", "chk2")
SHARD_IDS = (0, 1)

TEMPERATURE = 0.6
TOP_P = 0.9
TOP_K = -1
REPETITION_PENALTY = 1.0
MAX_TOKENS = 2048
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 16
MAX_NUM_BATCHED_TOKENS = 4096
GPU_HEADROOM_MIB = 2048
MIN_GPU_MEMORY_UTILIZATION = 0.78
DEFAULT_REPLICATE_SEEDS = tuple(20_260_901 + index for index in range(REPLICATES))
EXPECTED_VLLM_VERSION = "0.8.5.post1"

ROW_SCHEMA = "paper-chk2-text-similarity-vllm-generation-row-v1"
LAUNCH_SCHEMA = "paper-chk2-text-similarity-vllm-launch-v1"
STATE_SCHEMA = "paper-chk2-text-similarity-vllm-state-v1"
SHARD_SCHEMA = "paper-chk2-text-similarity-vllm-shard-v1"
VALIDATION_SCHEMA = "paper-chk2-text-similarity-vllm-validation-v1"
TORN_TAIL_RECOVERY_SCHEMA = "paper-chk2-text-similarity-wal-torn-tail-recovery-v1"
CONSOLIDATED_ROW_SCHEMA = "paper-chk2-text-similarity-generation-row-v1"
COMPLETE_EVALUATION_SCHEMA = "paper-chk2-text-similarity-evaluation-complete-v1"

THINK_BOUNDARY = "</think>"
FULL_4GRAM_REPETITION_LIMIT = 0.50
TAIL_4GRAM_REPETITION_LIMIT = 0.60
REPETITION_TAIL_TOKENS = 1024
CONTROL_MARKER_RE = re.compile(
    r"(?:</?think>|</?answer>|<\|(?:assistant|user|system|channel|im_start|im_end)[^>]*\|>"
    r"|<｜(?:Assistant|User|System|begin▁of▁sentence|end▁of▁sentence)｜>)",
    re.IGNORECASE,
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LOCK_TEMPLATE = "/tmp/fomc_trainer_paper_chk2_similarity_gpu{shard}.lock"


class PaperChk2VllmError(RuntimeError):
    """The prepared input, GPU runtime, WAL, or shard closure drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PaperChk2VllmError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PaperChk2VllmError(f"invalid {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                _require(raw.endswith("\n"), f"unterminated {label}:{line_number}")
                _require(bool(raw.strip()), f"blank {label}:{line_number}")
                value = json.loads(raw)
                _require(
                    isinstance(value, dict), f"non-object {label}:{line_number}"
                )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PaperChk2VllmError(f"invalid {label}: {path}") from exc
    return rows


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    _require(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_new_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Create a canonical JSONL artifact once and return its exact binding."""

    _require(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    row_count = 0
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            row_count += 1
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": row_count,
    }


def _append_fsync(handle: IO[str], value: Mapping[str, Any]) -> None:
    handle.write(canonical_json(value) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _resolve_path(raw: object, *, manifest_path: Path, label: str) -> Path:
    _require(isinstance(raw, str) and bool(raw.strip()), f"{label} path is invalid")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        repo_candidate = (ROOT / candidate).resolve()
        manifest_candidate = (manifest_path.parent / candidate).resolve()
        candidate = repo_candidate if repo_candidate.exists() else manifest_candidate
    return candidate.resolve()


def _file_record(
    raw: object, *, manifest_path: Path, label: str, expected_rows: int | None = None
) -> tuple[Path, dict[str, Any]]:
    _require(isinstance(raw, Mapping), f"{label} record is missing")
    path = _resolve_path(raw.get("path"), manifest_path=manifest_path, label=label)
    _require(path.is_file() and not path.is_symlink(), f"unsafe {label}: {path}")
    expected_sha = raw.get("sha256")
    _require(
        isinstance(expected_sha, str) and SHA256_RE.fullmatch(expected_sha) is not None,
        f"{label} SHA256 is invalid",
    )
    observed_sha = sha256_file(path)
    _require(observed_sha == expected_sha, f"{label} SHA256 drift")
    if expected_rows is not None:
        _require(raw.get("rows") == expected_rows, f"{label} row-count binding drift")
    return path, {
        "path": str(path),
        "sha256": observed_sha,
        "bytes": path.stat().st_size,
        **({"rows": expected_rows} if expected_rows is not None else {}),
    }


def _directory_path(
    raw: object,
    *,
    manifest_path: Path,
    label: str,
    adapter: bool = False,
) -> tuple[Path, dict[str, Any]]:
    _require(isinstance(raw, Mapping), f"{label} binding is missing")
    path_value = raw.get("adapter_path") if adapter else raw.get("path")
    if adapter and path_value is None:
        path_value = raw.get("path")
    if not adapter and path_value is None:
        path_value = raw.get("model_path")
    path = _resolve_path(path_value, manifest_path=manifest_path, label=label)
    _require(path.is_dir() and not path.is_symlink(), f"unsafe {label}: {path}")
    critical = (
        ("adapter_config.json", "adapter_model.safetensors")
        if adapter
        else ("config.json", "model.safetensors.index.json", "tokenizer.json")
    )
    observed_files: dict[str, str] = {}
    for name in critical:
        item = path / name
        _require(item.is_file() and not item.is_symlink(), f"missing {label}/{name}")
        observed_files[name] = sha256_file(item)

    expected_files = raw.get("files")
    if expected_files is not None:
        _require(isinstance(expected_files, Mapping), f"{label}.files is invalid")
        for name, expected in expected_files.items():
            _require(
                isinstance(name, str)
                and isinstance(expected, str)
                and SHA256_RE.fullmatch(expected) is not None,
                f"{label}.files entry is invalid",
            )
            item = path / name
            _require(item.is_file() and not item.is_symlink(), f"missing {label}/{name}")
            observed = sha256_file(item)
            _require(observed == expected, f"{label}/{name} SHA256 drift")
            observed_files[name] = observed

    for field, name in (
        ("config_sha256", "adapter_config.json" if adapter else "config.json"),
        ("adapter_model_sha256", "adapter_model.safetensors"),
    ):
        expected = raw.get(field)
        if expected is None or (field == "adapter_model_sha256" and not adapter):
            continue
        _require(
            isinstance(expected, str) and SHA256_RE.fullmatch(expected) is not None,
            f"{label}.{field} is invalid",
        )
        observed = sha256_file(path / name)
        _require(observed == expected, f"{label}.{field} drift")
        observed_files[name] = observed

    return path, {
        "path": str(path),
        "declared_sha256": raw.get("sha256"),
        "verified_files": dict(sorted(observed_files.items())),
    }


def _token_ids_hash(values: Sequence[int]) -> str:
    return sha256_text(canonical_json(list(values)))


@dataclass(frozen=True)
class EvaluationInputs:
    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    samples: tuple[Mapping[str, Any], ...]
    samples_binding: Mapping[str, Any]
    prompt_ledger_binding: Mapping[str, Any]
    replicate_seeds: tuple[int, ...]
    model_paths: Mapping[str, Path]
    model_bindings: Mapping[str, Mapping[str, Any]]


def _validate_generation_declaration(manifest: Mapping[str, Any]) -> None:
    generation = manifest.get("generation") or manifest.get("generation_contract")
    if generation is None:
        return
    _require(isinstance(generation, Mapping), "manifest generation contract is invalid")
    expected = {
        "k": REPLICATES,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "max_tokens": MAX_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": MAX_NUM_SEQS,
        "tensor_parallel_size_per_engine": 1,
        "data_parallel_size_per_engine": 1,
    }
    aliases = {"max_new_tokens": "max_tokens"}
    for key, observed in generation.items():
        canonical_key = aliases.get(str(key), str(key))
        if canonical_key in expected:
            _require(
                observed == expected[canonical_key],
                f"manifest generation drift at {key}: {observed!r}",
            )


def _load_ledger_rows(
    samples: Sequence[Mapping[str, Any]],
    *,
    ledger_path: Path,
    samples_path: Path,
) -> dict[str, Sequence[int]]:
    if ledger_path == samples_path:
        ledger_rows = samples
    else:
        ledger_rows = _read_jsonl(ledger_path, label="prompt token ledger")
        _require(len(ledger_rows) == SAMPLE_COUNT, "prompt ledger must have 391 rows")
    ledger: dict[str, Sequence[int]] = {}
    for index, row in enumerate(ledger_rows):
        sample_id = row.get("sample_id")
        values = row.get("prompt_token_ids")
        _require(isinstance(sample_id, str) and bool(sample_id), f"ledger ID {index}")
        _require(sample_id not in ledger, f"duplicate prompt ledger ID: {sample_id}")
        _require(
            isinstance(values, list)
            and bool(values)
            and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values),
            f"invalid prompt token IDs: {sample_id}",
        )
        declared_count = row.get("prompt_token_count")
        if declared_count is not None:
            _require(declared_count == len(values), f"prompt token count drift: {sample_id}")
        declared_sha = row.get("prompt_token_ids_sha256")
        if declared_sha is not None:
            _require(
                declared_sha == _token_ids_hash(values),
                f"prompt token hash drift: {sample_id}",
            )
        ledger[sample_id] = tuple(values)
    return ledger


def load_evaluation_inputs(manifest_path: Path) -> EvaluationInputs:
    """Load and hash-bind the 391-row prepared input without importing vLLM."""

    manifest_path = manifest_path.expanduser().resolve()
    manifest = _read_json(manifest_path, label="evaluation manifest")
    manifest_sha = sha256_file(manifest_path)
    inputs = manifest.get("inputs")
    _require(isinstance(inputs, Mapping), "manifest.inputs is missing")
    samples_path, samples_binding = _file_record(
        inputs.get("samples"),
        manifest_path=manifest_path,
        label="prepared samples",
        expected_rows=SAMPLE_COUNT,
    )
    ledger_path, ledger_binding = _file_record(
        inputs.get("prompt_token_ledger"),
        manifest_path=manifest_path,
        label="prompt token ledger",
        expected_rows=SAMPLE_COUNT,
    )
    rows = _read_jsonl(samples_path, label="prepared samples")
    _require(len(rows) == SAMPLE_COUNT, "prepared samples must contain 391 rows")
    ledger = _load_ledger_rows(rows, ledger_path=ledger_path, samples_path=samples_path)

    normalized: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for source_index, row in enumerate(rows):
        required = {
            "sample_id",
            "split",
            "meeting_id",
            "prompt",
            "reference",
            "prompt_sha256",
            "reference_sha256",
        }
        _require(required <= set(row), f"sample fields missing at row {source_index}")
        sample_id = row["sample_id"]
        _require(isinstance(sample_id, str) and bool(sample_id), "invalid sample ID")
        _require(sample_id not in seen, f"duplicate sample ID: {sample_id}")
        seen.add(sample_id)
        prompt = row["prompt"]
        reference = row["reference"]
        _require(isinstance(prompt, str) and bool(prompt), f"empty prompt: {sample_id}")
        _require(
            isinstance(reference, str) and bool(reference), f"empty reference: {sample_id}"
        )
        _require(row["prompt_sha256"] == sha256_text(prompt), f"prompt SHA drift: {sample_id}")
        _require(
            row["reference_sha256"] == sha256_text(reference),
            f"reference SHA drift: {sample_id}",
        )
        token_ids = ledger.get(sample_id)
        _require(token_ids is not None, f"sample missing from prompt ledger: {sample_id}")
        if row.get("prompt_token_ids") is not None:
            _require(
                tuple(row["prompt_token_ids"]) == tuple(token_ids),
                f"sample/ledger token mismatch: {sample_id}",
            )
        _require(
            len(token_ids) + MAX_TOKENS <= MAX_MODEL_LEN,
            f"context overflow for {sample_id}: {len(token_ids)}+{MAX_TOKENS}",
        )
        normalized.append(
            {
                "source_index": source_index,
                "sample_id": sample_id,
                "split": row["split"],
                "meeting_id": row["meeting_id"],
                "prompt": prompt,
                "reference": reference,
                "prompt_sha256": row["prompt_sha256"],
                "reference_sha256": row["reference_sha256"],
                "prompt_token_ids": tuple(token_ids),
                "prompt_token_ids_sha256": _token_ids_hash(token_ids),
                "prompt_token_count": len(token_ids),
            }
        )
    _require(set(ledger) == seen, "prompt ledger/sample ID partition drift")

    declared_seeds = manifest.get("replicate_seeds")
    if declared_seeds is None:
        generation = manifest.get("generation") or manifest.get("generation_contract") or {}
        if isinstance(generation, Mapping):
            declared_seeds = generation.get("replicate_seeds")
    seeds = tuple(DEFAULT_REPLICATE_SEEDS if declared_seeds is None else declared_seeds)
    _require(
        len(seeds) == REPLICATES
        and len(set(seeds)) == REPLICATES
        and all(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0 for seed in seeds),
        "replicate seeds must be ten unique non-negative integers",
    )

    _validate_generation_declaration(manifest)
    models = manifest.get("models")
    _require(isinstance(models, Mapping), "manifest.models is missing")
    _require(set(MODEL_IDS) <= set(models), "manifest must bind chk0/chk1/chk2")
    chk0_path, chk0_binding = _directory_path(
        models["chk0"], manifest_path=manifest_path, label="chk0 model"
    )
    chk1_path, chk1_binding = _directory_path(
        models["chk1"], manifest_path=manifest_path, label="chk1 model"
    )
    chk2_path, chk2_binding = _directory_path(
        models["chk2"], manifest_path=manifest_path, label="chk2 adapter", adapter=True
    )
    chk2_raw = models["chk2"]
    assert isinstance(chk2_raw, Mapping)
    declared_base = chk2_raw.get("base_model_path") or chk2_raw.get("base_path")
    if declared_base is not None:
        resolved_base = _resolve_path(
            declared_base, manifest_path=manifest_path, label="chk2 base model"
        )
        _require(resolved_base == chk1_path, "chk2 base must be exact chk1 model")
    adapter_config = _read_json(chk2_path / "adapter_config.json", label="chk2 adapter config")
    _require(adapter_config.get("peft_type") == "LORA", "chk2 is not a LoRA adapter")
    _require(adapter_config.get("r") == 32, "chk2 LoRA rank must be 32")
    _require(adapter_config.get("lora_alpha") == 64, "chk2 LoRA alpha must be 64")
    _require(adapter_config.get("bias") == "none", "chk2 LoRA bias must be none")
    _require(adapter_config.get("modules_to_save") is None, "chk2 modules_to_save forbidden")
    _require(adapter_config.get("use_dora") is False, "chk2 DoRA forbidden")
    declared_parent = adapter_config.get("base_model_name_or_path")
    if isinstance(declared_parent, str) and declared_parent:
        parent_path = Path(declared_parent).expanduser()
        if not parent_path.is_absolute():
            parent_path = (ROOT / parent_path).resolve()
        _require(parent_path.resolve() == chk1_path, "adapter parent path is not chk1")

    return EvaluationInputs(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        manifest=manifest,
        samples=tuple(normalized),
        samples_binding=samples_binding,
        prompt_ledger_binding=ledger_binding,
        replicate_seeds=seeds,
        model_paths={"chk0": chk0_path, "chk1": chk1_path, "chk2": chk2_path},
        model_bindings={
            "chk0": chk0_binding,
            "chk1": chk1_binding,
            "chk2": chk2_binding,
        },
    )


def paired_tuple_id(sample_id: str, replicate_index: int) -> str:
    _require(isinstance(sample_id, str) and bool(sample_id), "invalid sample ID")
    _require(0 <= replicate_index < REPLICATES, "invalid replicate index")
    return f"{sample_id}|{replicate_index}"


def tuple_shard(sample_id: str, replicate_index: int) -> int:
    """Assign a paired tuple without model_id so all three outputs colocate."""

    digest = hashlib.sha256(paired_tuple_id(sample_id, replicate_index).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big") % len(SHARD_IDS)


def derive_shared_seed(sample_id: str, replicate_index: int, replicate_seed: int) -> int:
    payload = f"{replicate_seed}|{paired_tuple_id(sample_id, replicate_index)}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % (
        2**31
    )


def generation_key(model_id: str, sample_id: str, replicate_index: int) -> str:
    _require(model_id in MODEL_IDS, f"invalid model ID: {model_id}")
    return f"{model_id}|{paired_tuple_id(sample_id, replicate_index)}"


def build_case_matrix(
    inputs: EvaluationInputs,
    *,
    smoke: bool = False,
    smoke_samples: int = 6,
    smoke_replicates: int = 2,
) -> list[dict[str, Any]]:
    """Return the canonical 11,730-case matrix (or a deterministic smoke prefix)."""

    if smoke:
        _require(1 <= smoke_samples <= SAMPLE_COUNT, "invalid smoke sample count")
        _require(1 <= smoke_replicates <= REPLICATES, "invalid smoke replicate count")
        split_order = ("train", "validation", "test")
        by_split = {
            split: [row for row in inputs.samples if row["split"] == split]
            for split in split_order
        }
        base, remainder = divmod(smoke_samples, len(split_order))
        split_quotas = {
            split: base + (1 if index < remainder else 0)
            for index, split in enumerate(split_order)
        }
        _require(
            all(len(by_split[split]) >= split_quotas[split] for split in split_order),
            "insufficient rows for stratified smoke selection",
        )
        samples = tuple(
            sorted(
                (
                    row
                    for split in split_order
                    for row in by_split[split][: split_quotas[split]]
                ),
                key=lambda row: int(row["source_index"]),
            )
        )
        replicate_indexes = range(smoke_replicates)
    else:
        samples = inputs.samples
        replicate_indexes = range(REPLICATES)
    cases: list[dict[str, Any]] = []
    absolute_case_index = 0
    for sample in samples:
        sample_id = str(sample["sample_id"])
        for replicate_index in replicate_indexes:
            replicate_seed = inputs.replicate_seeds[replicate_index]
            shared_seed = derive_shared_seed(sample_id, replicate_index, replicate_seed)
            shard_id = tuple_shard(sample_id, replicate_index)
            pair_index = int(sample["source_index"]) * REPLICATES + replicate_index
            for model_id in MODEL_IDS:
                cases.append(
                    {
                        "absolute_case_index": absolute_case_index,
                        "pair_index": pair_index,
                        "tuple_id": paired_tuple_id(sample_id, replicate_index),
                        "generation_key": generation_key(model_id, sample_id, replicate_index),
                        "model_id": model_id,
                        "sample_id": sample_id,
                        "source_index": sample["source_index"],
                        "split": sample["split"],
                        "meeting_id": sample["meeting_id"],
                        "replicate_index": replicate_index,
                        "replicate_seed": replicate_seed,
                        "row_seed": shared_seed,
                        "shard_id": shard_id,
                        "prompt_sha256": sample["prompt_sha256"],
                        "reference_sha256": sample["reference_sha256"],
                        "prompt_token_ids": sample["prompt_token_ids"],
                        "prompt_token_ids_sha256": sample["prompt_token_ids_sha256"],
                        "prompt_token_count": sample["prompt_token_count"],
                    }
                )
                absolute_case_index += 1
    expected = (smoke_samples * smoke_replicates * 3) if smoke else 11_730
    _require(len(cases) == expected, "case matrix cardinality drift")
    return cases


def compute_gpu_memory_utilization(free_mib: int, total_mib: int) -> float:
    """Floor the live free-memory budget to 0.01 while reserving 2 GiB."""

    _require(
        isinstance(free_mib, int)
        and isinstance(total_mib, int)
        and 0 < free_mib <= total_mib,
        "invalid GPU memory observation",
    )
    raw = (free_mib - GPU_HEADROOM_MIB) / total_mib
    value = math.floor(raw * 100.0) / 100.0
    _require(
        value >= MIN_GPU_MEMORY_UTILIZATION,
        f"insufficient GPU memory after {GPU_HEADROOM_MIB} MiB headroom: {value:.2f}",
    )
    _require(value < 1.0, "computed GPU memory utilization must be below 1")
    return value


def gpu_snapshot(physical_gpu_index: int) -> dict[str, Any]:
    _require(physical_gpu_index in SHARD_IDS, "physical GPU must be 0 or 1")
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_gpu_index}",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    _require(completed.returncode == 0, f"nvidia-smi failed: {completed.stderr.strip()}")
    fields = [item.strip() for item in completed.stdout.strip().split(",")]
    _require(len(fields) == 7, "unexpected nvidia-smi GPU row")
    result = {
        "physical_gpu_index": int(fields[0]),
        "uuid": fields[1],
        "name": fields[2],
        "memory_total_mib": int(fields[3]),
        "memory_used_mib": int(fields[4]),
        "memory_free_mib": int(fields[5]),
        "utilization_gpu_percent": int(fields[6]),
    }
    result["vllm_gpu_memory_utilization"] = compute_gpu_memory_utilization(
        result["memory_free_mib"], result["memory_total_mib"]
    )
    return result


def _validate_visible_gpu(shard_id: int) -> None:
    _require(
        os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
        "CUDA_DEVICE_ORDER must be PCI_BUS_ID",
    )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    _require(
        visible == str(shard_id),
        f"CUDA_VISIBLE_DEVICES must be exactly {shard_id!r}; observed={visible!r}",
    )


@contextlib.contextmanager
def _worker_lock(shard_id: int) -> Iterable[Mapping[str, Any]]:
    path = Path(LOCK_TEMPLATE.format(shard=shard_id))
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    handle = os.fdopen(descriptor, "a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PaperChk2VllmError(f"paper chk2 GPU{shard_id} worker lock is held") from exc
        yield {"path": str(path), "pid": os.getpid()}
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _generation_contract() -> dict[str, Any]:
    return {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "max_tokens": MAX_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "tensor_parallel_size_per_engine": 1,
        "data_parallel_size_per_engine": 1,
        "pipeline_parallel_size_per_engine": 1,
        "independent_engine_count": 2,
        "checkpoint_pairing": "same_sample_replicate_seed_and_physical_gpu",
    }


def _validate_wal_row(row: Mapping[str, Any], case: Mapping[str, Any], *, manifest_sha: str) -> None:
    _require(row.get("schema_version") == ROW_SCHEMA, "WAL row schema drift")
    _require(row.get("evaluation_manifest_sha256") == manifest_sha, "WAL manifest drift")
    for field in (
        "absolute_case_index",
        "pair_index",
        "tuple_id",
        "generation_key",
        "model_id",
        "sample_id",
        "source_index",
        "split",
        "meeting_id",
        "replicate_index",
        "replicate_seed",
        "row_seed",
        "shard_id",
        "prompt_sha256",
        "reference_sha256",
        "prompt_token_ids_sha256",
        "prompt_token_count",
    ):
        _require(row.get(field) == case.get(field), f"WAL case drift at {field}")
    completion = row.get("completion")
    token_ids = row.get("output_token_ids")
    _require(isinstance(completion, str), "WAL completion is not text")
    _require(
        isinstance(token_ids, list)
        and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in token_ids),
        "WAL output token IDs are invalid",
    )
    _require(row.get("completion_sha256") == sha256_text(completion), "completion SHA drift")
    _require(row.get("output_token_ids_sha256") == _token_ids_hash(token_ids), "token SHA drift")
    _require(row.get("raw_generated_tokens") == len(token_ids), "token count drift")
    _require(row.get("generation_contract") == _generation_contract(), "WAL generation contract drift")


def load_wal(
    path: Path,
    *,
    cases_by_key: Mapping[str, Mapping[str, Any]],
    manifest_sha: str,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    raw_payload = path.read_bytes()
    if raw_payload and not raw_payload.endswith(b"\n"):
        recovered_bytes = raw_payload.rfind(b"\n") + 1
        discarded = raw_payload[recovered_bytes:]
        _require(bool(discarded), "WAL torn-tail detection drift")
        with path.open("r+b") as handle:
            handle.truncate(recovered_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        recovery_path = path.with_name(f"{path.name}.torn-tail-recoveries.jsonl")
        _require(not recovery_path.is_symlink(), "unsafe WAL recovery ledger")
        recovery_path.parent.mkdir(parents=True, exist_ok=True)
        with recovery_path.open("a", encoding="utf-8") as recovery_handle:
            _append_fsync(
                recovery_handle,
                {
                    "schema_version": TORN_TAIL_RECOVERY_SCHEMA,
                    "recovered_at_utc": _utc_now(),
                    "wal_path": str(path.resolve()),
                    "original_bytes": len(raw_payload),
                    "recovered_bytes": recovered_bytes,
                    "discarded_bytes": len(discarded),
                    "discarded_sha256": hashlib.sha256(discarded).hexdigest(),
                    "policy": "discard_only_final_unterminated_jsonl_fragment",
                },
            )
    rows = _read_jsonl(path, label="generation WAL")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("generation_key")
        _require(isinstance(key, str) and key in cases_by_key, f"unknown WAL key: {key}")
        _require(key not in indexed, f"duplicate WAL key: {key}")
        _validate_wal_row(row, cases_by_key[key], manifest_sha=manifest_sha)
        indexed[key] = row
    return indexed


def _raw_result(
    *,
    case: Mapping[str, Any],
    completion: Any,
    manifest_sha: str,
    request_id: str,
) -> dict[str, Any]:
    text = getattr(completion, "text", None)
    token_ids = list(getattr(completion, "token_ids", []) or [])
    _require(isinstance(text, str), f"vLLM returned non-text: {request_id}")
    _require(
        all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in token_ids),
        f"vLLM returned invalid token IDs: {request_id}",
    )
    return {
        "schema_version": ROW_SCHEMA,
        "evaluation_manifest_sha256": manifest_sha,
        **{key: value for key, value in case.items() if key != "prompt_token_ids"},
        "request_id": request_id,
        "completion": text,
        "completion_sha256": sha256_text(text),
        "output_token_ids": token_ids,
        "output_token_ids_sha256": _token_ids_hash(token_ids),
        "raw_generated_tokens": len(token_ids),
        "finish_reason": getattr(completion, "finish_reason", None),
        "stop_reason": getattr(completion, "stop_reason", None),
        "generation_contract": _generation_contract(),
        "completed_at_utc": _utc_now(),
    }


async def _generate_cases(
    *,
    engine: Any,
    sampling_params_cls: Any,
    cases: Sequence[Mapping[str, Any]],
    manifest_sha: str,
    lora_request: Any,
    on_result: Any,
) -> None:
    async def consume(case: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any, str]:
        params = sampling_params_cls(
            n=1,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            max_tokens=MAX_TOKENS,
            seed=int(case["row_seed"]),
            detokenize=True,
            skip_special_tokens=False,
            spaces_between_special_tokens=True,
        )
        request_id = f"paper-chk2-{case['generation_key']}-{case['row_seed']}"
        final = None
        async for output in engine.generate(
            {"prompt_token_ids": list(case["prompt_token_ids"])},
            params,
            request_id,
            lora_request=lora_request,
        ):
            final = output
        _require(final is not None, f"vLLM returned no output: {request_id}")
        _require(len(getattr(final, "outputs", []) or []) == 1, f"vLLM n drift: {request_id}")
        observed_prompt = list(getattr(final, "prompt_token_ids", []) or [])
        _require(
            observed_prompt == list(case["prompt_token_ids"]),
            f"vLLM prompt token drift: {request_id}",
        )
        return case, final.outputs[0], request_id

    for start in range(0, len(cases), 64):
        chunk = cases[start : start + 64]
        tasks = [asyncio.create_task(consume(case)) for case in chunk]
        try:
            for future in asyncio.as_completed(tasks):
                case, completion, request_id = await future
                await on_result(
                    _raw_result(
                        case=case,
                        completion=completion,
                        manifest_sha=manifest_sha,
                        request_id=request_id,
                    )
                )
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def _shutdown_engine(engine: Any) -> None:
    if engine is None:
        return
    if hasattr(engine, "shutdown"):
        engine.shutdown()
    elif hasattr(engine, "shutdown_background_loop"):
        engine.shutdown_background_loop()


async def _run_engine_group(
    *,
    model_path: Path,
    adapter_path: Path | None,
    pending_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    manifest_sha: str,
    gpu_memory_utilization: float,
    on_result: Any,
) -> dict[str, Any]:
    import torch
    import vllm
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.lora.request import LoRARequest

    _require(vllm.__version__ == EXPECTED_VLLM_VERSION, f"vLLM version drift: {vllm.__version__}")
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "worker needs one visible CUDA GPU")
    enable_lora = adapter_path is not None
    args = AsyncEngineArgs(
        model=str(model_path),
        tokenizer=str(model_path),
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="bfloat16",
        quantization=None,
        load_format="safetensors",
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        data_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        swap_space=0,
        cpu_offload_gb=0,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        generation_config="vllm",
        seed=0,
        disable_log_requests=True,
        disable_log_stats=True,
        enable_lora=enable_lora,
        max_lora_rank=32,
        max_loras=1,
        max_cpu_loras=1,
        lora_dtype="bfloat16",
    )
    load_started = time.perf_counter()
    engine = AsyncLLMEngine.from_engine_args(args)
    loaded_seconds = time.perf_counter() - load_started
    generated = Counter()
    started = time.perf_counter()
    try:
        for model_id in MODEL_IDS:
            cases = list(pending_by_model.get(model_id, ()))
            if not cases:
                continue
            if model_id == "chk0":
                _require(adapter_path is None, "chk0 engine cannot receive LoRA")
                request = None
            elif model_id == "chk1":
                _require(adapter_path is not None, "chk1 must share LoRA-capable base engine")
                request = None
            else:
                _require(adapter_path is not None, "chk2 adapter path is missing")
                request = LoRARequest("paper-chk2-cp50", 50, lora_path=str(adapter_path))

            async def record(row: Mapping[str, Any]) -> None:
                await on_result(row)
                generated[model_id] += 1

            await _generate_cases(
                engine=engine,
                sampling_params_cls=SamplingParams,
                cases=cases,
                manifest_sha=manifest_sha,
                lora_request=request,
                on_result=record,
            )
    finally:
        _shutdown_engine(engine)
    elapsed = time.perf_counter() - started
    return {
        "model_path": str(model_path),
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "enable_lora": enable_lora,
        "model_load_wall_seconds": loaded_seconds,
        "generation_wall_seconds": elapsed,
        "new_rows_by_model": dict(generated),
    }


def _launch_contract(
    inputs: EvaluationInputs,
    *,
    shard_id: int,
    smoke: bool,
    smoke_samples: int,
    smoke_replicates: int,
) -> dict[str, Any]:
    return {
        "schema_version": LAUNCH_SCHEMA,
        "evaluation_manifest": {
            "path": str(inputs.manifest_path),
            "sha256": inputs.manifest_sha256,
        },
        "inputs": {
            "samples": dict(inputs.samples_binding),
            "prompt_token_ledger": dict(inputs.prompt_ledger_binding),
        },
        "models": {key: dict(value) for key, value in inputs.model_bindings.items()},
        "mode": "smoke" if smoke else "generate",
        "shard_id": shard_id,
        "physical_gpu_index": shard_id,
        "smoke_samples": smoke_samples if smoke else None,
        "smoke_replicates": smoke_replicates if smoke else None,
        "generation_contract": _generation_contract(),
        "replicate_seeds": list(inputs.replicate_seeds),
        "shard_function": "sha256(sample_id|replicate_index)[0:8] mod 2",
    }


def _case_index(cases: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for case in cases:
        key = str(case["generation_key"])
        _require(key not in result, f"duplicate expected generation key: {key}")
        result[key] = case
    return result


def _write_canonical_shard(
    path: Path,
    *,
    rows: Mapping[str, Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = [rows[str(case["generation_key"])] for case in cases]
    encoded = "".join(canonical_json(row) + "\n" for row in ordered).encode("utf-8")
    if path.exists():
        _require(path.is_file() and not path.is_symlink(), "unsafe canonical shard")
        _require(path.read_bytes() == encoded, "canonical shard content drift")
    else:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": len(ordered),
    }


def run_worker(
    *,
    manifest_path: Path,
    output_root: Path,
    shard_id: int,
    smoke: bool = False,
    smoke_samples: int = 6,
    smoke_replicates: int = 2,
    resume: bool = False,
) -> Mapping[str, Any]:
    """Run one independent single-GPU shard, resuming only absent tuple keys."""

    _require(shard_id in SHARD_IDS, "shard ID must be 0 or 1")
    _validate_visible_gpu(shard_id)
    inputs = load_evaluation_inputs(manifest_path)
    all_cases = build_case_matrix(
        inputs,
        smoke=smoke,
        smoke_samples=smoke_samples,
        smoke_replicates=smoke_replicates,
    )
    cases = [case for case in all_cases if case["shard_id"] == shard_id]
    case_index = _case_index(cases)
    output_root = output_root.expanduser().resolve()
    shard_root = output_root / f"shard-{shard_id}"
    launch_path = shard_root / "launch.json"
    wal_path = shard_root / "generations.wal.jsonl"
    canonical_path = shard_root / "generations.canonical.jsonl"
    state_path = shard_root / "state.json"
    shard_manifest_path = shard_root / "manifest.json"
    expected_launch = _launch_contract(
        inputs,
        shard_id=shard_id,
        smoke=smoke,
        smoke_samples=smoke_samples,
        smoke_replicates=smoke_replicates,
    )

    if shard_root.exists():
        _require(resume, f"shard output exists; pass --resume: {shard_root}")
        _require(shard_root.is_dir() and not shard_root.is_symlink(), "unsafe shard root")
        launch = _read_json(launch_path, label="worker launch")
        stable_launch = {key: launch.get(key) for key in expected_launch}
        _require(stable_launch == expected_launch, "resume launch contract drift")
    else:
        shard_root.mkdir(parents=True, exist_ok=False)
        _write_new_json(
            launch_path,
            {
                **expected_launch,
                "status": "initialized",
                "created_at_utc": _utc_now(),
                "python": sys.executable,
                "python_version": platform.python_version(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        )

    rows = load_wal(wal_path, cases_by_key=case_index, manifest_sha=inputs.manifest_sha256)
    if len(rows) == len(cases):
        canonical_binding = _write_canonical_shard(
            canonical_path, rows=rows, cases=cases
        )
        result = {
            "schema_version": SHARD_SCHEMA,
            "status": "complete",
            "shard_id": shard_id,
            "mode": "smoke" if smoke else "generate",
            "expected_rows": len(cases),
            "completed_rows": len(rows),
            "resume_noop": True,
            "canonical": canonical_binding,
            "wal": {
                "path": str(wal_path.resolve()),
                "sha256": sha256_file(wal_path),
                "bytes": wal_path.stat().st_size,
                "rows": len(rows),
            },
        }
        _atomic_json(shard_manifest_path, result)
        return result

    pending_by_model: dict[str, list[Mapping[str, Any]]] = {model: [] for model in MODEL_IDS}
    for case in cases:
        if case["generation_key"] not in rows:
            pending_by_model[str(case["model_id"])].append(case)

    state: dict[str, Any] = {
        "schema_version": STATE_SCHEMA,
        "status": "running",
        "updated_at_utc": _utc_now(),
        "shard_id": shard_id,
        "expected_rows": len(cases),
        "resumed_rows": len(rows),
        "completed_rows": len(rows),
        "pending_rows_by_model": {key: len(value) for key, value in pending_by_model.items()},
        "engine_runs": [],
    }
    _atomic_json(state_path, state)

    with _worker_lock(shard_id) as lock:
        state["worker_lock"] = dict(lock)
        wal_path.parent.mkdir(parents=True, exist_ok=True)
        with wal_path.open("a", encoding="utf-8") as wal_handle:
            async def on_result(row: Mapping[str, Any]) -> None:
                key = str(row["generation_key"])
                _require(key not in rows, f"duplicate newly generated key: {key}")
                _append_fsync(wal_handle, row)
                rows[key] = dict(row)
                state["completed_rows"] = len(rows)
                state["updated_at_utc"] = _utc_now()
                if len(rows) % 16 == 0 or len(rows) == len(cases):
                    _atomic_json(state_path, state)

            async def execute() -> None:
                if pending_by_model["chk0"]:
                    snapshot = gpu_snapshot(shard_id)
                    report = await _run_engine_group(
                        model_path=inputs.model_paths["chk0"],
                        adapter_path=None,
                        pending_by_model={"chk0": pending_by_model["chk0"]},
                        manifest_sha=inputs.manifest_sha256,
                        gpu_memory_utilization=float(snapshot["vllm_gpu_memory_utilization"]),
                        on_result=on_result,
                    )
                    state["engine_runs"].append({"gpu_before": snapshot, **report})
                    _atomic_json(state_path, state)
                    gc.collect()
                    with contextlib.suppress(ImportError):
                        import torch

                        torch.cuda.empty_cache()
                if pending_by_model["chk1"] or pending_by_model["chk2"]:
                    snapshot = gpu_snapshot(shard_id)
                    report = await _run_engine_group(
                        model_path=inputs.model_paths["chk1"],
                        adapter_path=inputs.model_paths["chk2"],
                        pending_by_model={
                            "chk1": pending_by_model["chk1"],
                            "chk2": pending_by_model["chk2"],
                        },
                        manifest_sha=inputs.manifest_sha256,
                        gpu_memory_utilization=float(snapshot["vllm_gpu_memory_utilization"]),
                        on_result=on_result,
                    )
                    state["engine_runs"].append({"gpu_before": snapshot, **report})
                    _atomic_json(state_path, state)

            asyncio.run(execute())

    _require(len(rows) == len(cases), "worker ended without shard closure")
    canonical_binding = _write_canonical_shard(canonical_path, rows=rows, cases=cases)
    counts = Counter(str(row["model_id"]) for row in rows.values())
    _require(len(set(counts.values())) == 1, "shard model counts are not paired")
    result = {
        "schema_version": SHARD_SCHEMA,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "shard_id": shard_id,
        "mode": "smoke" if smoke else "generate",
        "expected_rows": len(cases),
        "completed_rows": len(rows),
        "resumed_rows": state["resumed_rows"],
        "new_rows": len(rows) - int(state["resumed_rows"]),
        "model_counts": dict(sorted(counts.items())),
        "canonical": canonical_binding,
        "wal": {
            "path": str(wal_path.resolve()),
            "sha256": sha256_file(wal_path),
            "bytes": wal_path.stat().st_size,
            "rows": len(rows),
        },
        "engine_runs": state["engine_runs"],
    }
    _atomic_json(shard_manifest_path, result)
    state.update({"status": "complete", "updated_at_utc": _utc_now()})
    _atomic_json(state_path, state)
    return result


def validate_outputs(
    *,
    manifest_path: Path,
    output_root: Path,
    smoke: bool = False,
    smoke_samples: int = 6,
    smoke_replicates: int = 2,
) -> Mapping[str, Any]:
    """Validate both shard WALs and prove paired three-model closure."""

    inputs = load_evaluation_inputs(manifest_path)
    cases = build_case_matrix(
        inputs,
        smoke=smoke,
        smoke_samples=smoke_samples,
        smoke_replicates=smoke_replicates,
    )
    expected = _case_index(cases)
    combined: dict[str, Mapping[str, Any]] = {}
    shard_bindings: dict[str, Mapping[str, Any]] = {}
    for shard_id in SHARD_IDS:
        shard_cases = [case for case in cases if case["shard_id"] == shard_id]
        shard_index = _case_index(shard_cases)
        shard_root = output_root.expanduser().resolve() / f"shard-{shard_id}"
        rows = load_wal(
            shard_root / "generations.wal.jsonl",
            cases_by_key=shard_index,
            manifest_sha=inputs.manifest_sha256,
        )
        _require(rows.keys() == shard_index.keys(), f"shard-{shard_id} is incomplete")
        for key, row in rows.items():
            _require(key not in combined, f"cross-shard duplicate key: {key}")
            combined[key] = row
        canonical = _write_canonical_shard(
            shard_root / "generations.canonical.jsonl", rows=rows, cases=shard_cases
        )
        shard_bindings[f"shard-{shard_id}"] = {
            "rows": len(rows),
            "canonical": canonical,
            "wal": {
                "path": str((shard_root / "generations.wal.jsonl").resolve()),
                "sha256": sha256_file(shard_root / "generations.wal.jsonl"),
                "bytes": (shard_root / "generations.wal.jsonl").stat().st_size,
                "rows": len(rows),
            },
        }
    _require(combined.keys() == expected.keys(), "two-shard union is incomplete")

    pairs: dict[str, list[Mapping[str, Any]]] = {}
    for row in combined.values():
        pairs.setdefault(str(row["tuple_id"]), []).append(row)
    base_lora_differences = 0
    for tuple_id, group in pairs.items():
        _require({str(row["model_id"]) for row in group} == set(MODEL_IDS), f"model closure: {tuple_id}")
        _require(len(group) == 3, f"pair cardinality drift: {tuple_id}")
        _require(len({int(row["row_seed"]) for row in group}) == 1, f"seed pairing drift: {tuple_id}")
        _require(len({int(row["shard_id"]) for row in group}) == 1, f"GPU pairing drift: {tuple_id}")
        _require(
            len({str(row["prompt_token_ids_sha256"]) for row in group}) == 1,
            f"prompt-token pairing drift: {tuple_id}",
        )
        by_model = {str(row["model_id"]): row for row in group}
        if by_model["chk1"]["output_token_ids"] != by_model["chk2"]["output_token_ids"]:
            base_lora_differences += 1

    base_lora_outputs_not_all_identical = base_lora_differences > 0
    if smoke:
        _require(
            base_lora_outputs_not_all_identical,
            "smoke route proof failed: every chk1/chk2 output is token-identical",
        )

    model_counts = Counter(str(row["model_id"]) for row in combined.values())
    expected_per_model = (
        smoke_samples * smoke_replicates if smoke else SAMPLE_COUNT * REPLICATES
    )
    _require(
        model_counts == Counter({model_id: expected_per_model for model_id in MODEL_IDS}),
        "three-model count closure drift",
    )
    finish_reasons = Counter(str(row.get("finish_reason")) for row in combined.values())
    return {
        "schema_version": VALIDATION_SCHEMA,
        "status": "complete",
        "validated_at_utc": _utc_now(),
        "mode": "smoke" if smoke else "generate",
        "evaluation_manifest": {
            "path": str(inputs.manifest_path),
            "sha256": inputs.manifest_sha256,
        },
        "samples": smoke_samples if smoke else SAMPLE_COUNT,
        "replicates": smoke_replicates if smoke else REPLICATES,
        "paired_tuples": len(pairs),
        "generation_rows": len(combined),
        "model_counts": dict(sorted(model_counts.items())),
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "shards": shard_bindings,
        "gates": {
            "all_expected_keys_present": True,
            "no_duplicate_keys": True,
            "same_seed_across_three_models": True,
            "same_raw_prompt_tokens_across_three_models": True,
            "same_physical_gpu_across_three_models": True,
            "base_lora_outputs_not_all_identical": base_lora_outputs_not_all_identical,
            "wal_rows_fsynced_by_worker": True,
        },
        "base_lora_different_paired_tuples": base_lora_differences,
    }


def _ngram_repetition(token_ids: Sequence[int], *, n: int = 4) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    if len(token_ids) < n:
        return 0.0
    grams = [
        tuple(token_ids[index : index + n])
        for index in range(len(token_ids) - n + 1)
    ]
    return 1.0 - len(set(grams)) / len(grams)


def project_delivery(row: Mapping[str, Any]) -> dict[str, Any]:
    """Recover a scoring candidate and apply transport-only delivery gates."""

    completion = row.get("completion")
    token_ids = row.get("output_token_ids")
    _require(isinstance(completion, str), "completion must be text")
    _require(
        isinstance(token_ids, list)
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in token_ids
        ),
        "completion token IDs are invalid",
    )
    boundary_count = completion.count(THINK_BOUNDARY)
    if boundary_count == 1:
        recovered = completion.split(THINK_BOUNDARY, 1)[1].strip()
        recovery_method = "unique_think_boundary"
    else:
        candidates = [
            candidate.strip()
            for candidate in re.split(r"[\r\n\u2028\u2029]+", completion)
            if candidate.strip()
        ]
        recovered = candidates[-1] if candidates else ""
        recovery_method = "raw_best_effort_last_nonempty_paragraph"

    full_repetition = _ngram_repetition(token_ids, n=4)
    tail_repetition = _ngram_repetition(
        token_ids[-REPETITION_TAIL_TOKENS:], n=4
    )
    raw_finish_reason = row.get("finish_reason")
    finish_reason = (
        raw_finish_reason
        if isinstance(raw_finish_reason, str) and raw_finish_reason.strip()
        else "unknown"
    )
    failures: list[str] = []
    if boundary_count != 1:
        failures.append("think_boundary_count_not_one")
    if finish_reason != "stop":
        failures.append("finish_reason_not_stop")
    if not recovered:
        failures.append("empty_recovered_text")
    if recovered and re.search(r"[\r\n\u2028\u2029]", recovered):
        failures.append("recovered_text_multi_paragraph")
    if recovered and CONTROL_MARKER_RE.search(recovered):
        failures.append("recovered_text_control_marker")
    if full_repetition >= FULL_4GRAM_REPETITION_LIMIT:
        failures.append("full_4gram_repetition_ge_0.50")
    if tail_repetition >= TAIL_4GRAM_REPETITION_LIMIT:
        failures.append("tail_4gram_repetition_ge_0.60")
    failures = sorted(set(failures))
    return {
        "replicate_id": int(row["replicate_index"]),
        "completion_token_ids": list(token_ids),
        "recovered_text": recovered,
        "recovered_text_sha256": sha256_text(recovered),
        "recovery_method": recovery_method,
        "think_boundary_count": boundary_count,
        "finish_reason": finish_reason,
        "raw_finish_reason": raw_finish_reason,
        "full_token_4gram_repetition": round(full_repetition, 8),
        "tail_token_count": min(len(token_ids), REPETITION_TAIL_TOKENS),
        "tail_token_4gram_repetition": round(tail_repetition, 8),
        "delivery_valid": not failures,
        "delivery_failures": failures,
    }


def consolidate_outputs(
    *,
    manifest_path: Path,
    generation_output_root: Path,
    destination_root: Path,
    resume: bool = False,
) -> Mapping[str, Any]:
    """Merge the two complete shard WALs into the scoring-layer artifacts."""

    inputs = load_evaluation_inputs(manifest_path)
    cases = build_case_matrix(inputs)
    validation = validate_outputs(
        manifest_path=manifest_path,
        output_root=generation_output_root,
    )
    destination = destination_root.expanduser().resolve()
    if destination.suffix == ".jsonl":
        generation_rows_path = destination
        complete_manifest_path = destination.parent / "complete_evaluation_manifest.json"
    else:
        generation_rows_path = destination / "generation_rows.jsonl"
        complete_manifest_path = destination / "complete_evaluation_manifest.json"
    _require(not generation_rows_path.is_symlink(), "unsafe generation rows symlink")
    _require(not complete_manifest_path.is_symlink(), "unsafe complete manifest symlink")
    generation_exists = generation_rows_path.exists()
    manifest_exists = complete_manifest_path.exists()
    _require(
        generation_exists == manifest_exists,
        "partial consolidation exists; both artifacts are required for resume",
    )
    if generation_exists:
        _require(resume, f"refusing overwrite: {generation_rows_path}")
        _require(generation_rows_path.is_file(), "generation rows is not a file")
        _require(complete_manifest_path.is_file(), "complete manifest is not a file")

    case_index = _case_index(cases)
    combined: dict[str, Mapping[str, Any]] = {}
    for shard_id in SHARD_IDS:
        shard_cases = [case for case in cases if int(case["shard_id"]) == shard_id]
        shard_index = _case_index(shard_cases)
        wal_path = (
            generation_output_root.expanduser().resolve()
            / f"shard-{shard_id}"
            / "generations.wal.jsonl"
        )
        rows = load_wal(
            wal_path,
            cases_by_key=shard_index,
            manifest_sha=inputs.manifest_sha256,
        )
        _require(rows.keys() == shard_index.keys(), f"shard-{shard_id} is incomplete")
        for key, row in rows.items():
            _require(key not in combined, f"cross-shard duplicate key: {key}")
            combined[key] = row
    _require(combined.keys() == case_index.keys(), "two-shard union is incomplete")

    delivery_counts: Counter[str] = Counter()
    projected_rows: list[Mapping[str, Any]] = []
    valid_rows = 0
    for case in cases:
        source = combined[str(case["generation_key"])]
        delivery = project_delivery(source)
        delivery_counts.update(delivery["delivery_failures"])
        valid_rows += int(bool(delivery["delivery_valid"]))
        projected_rows.append(
            {
                **source,
                "schema_version": CONSOLIDATED_ROW_SCHEMA,
                "source_generation_schema_version": source["schema_version"],
                **delivery,
                "input_truncated": False,
            }
        )

    expected_total = SAMPLE_COUNT * REPLICATES * len(MODEL_IDS)
    _require(
        len(projected_rows) == expected_total,
        "consolidated generation row count drift",
    )
    samples_binding = dict(inputs.samples_binding)
    delivery_summary = {
        "valid_rows": valid_rows,
        "invalid_rows": expected_total - valid_rows,
        "failure_counts": dict(sorted(delivery_counts.items())),
        "recovery_rule": "unique </think> suffix; otherwise final nonempty raw paragraph",
    }

    if generation_exists:
        observed_rows = _read_jsonl(
            generation_rows_path, label="consolidated generation rows"
        )
        _require(
            observed_rows == projected_rows,
            "resume generation rows differ from the shard WAL projection",
        )
        generation_binding = {
            "path": str(generation_rows_path),
            "sha256": sha256_file(generation_rows_path),
            "bytes": generation_rows_path.stat().st_size,
            "rows": len(observed_rows),
        }
        prior = _read_json(
            complete_manifest_path, label="complete evaluation manifest"
        )
        if "integrity" in prior:
            from open_r1.validator.loo_generation_spec import validate_manifest_integrity

            validate_manifest_integrity(prior)
        _require(prior.get("schema_version") == COMPLETE_EVALUATION_SCHEMA, "complete schema drift")
        _require(prior.get("status") == "complete", "complete status drift")
        _require(prior.get("model_order") == list(MODEL_IDS), "model order drift")
        _require(prior.get("models") == list(MODEL_IDS), "model list drift")
        _require(prior.get("k") == REPLICATES, "replicate count drift")
        _require(prior.get("temperature") == TEMPERATURE, "temperature drift")
        _require(prior.get("top_p") == TOP_P, "top-p drift")
        _require(prior.get("total_generations") == expected_total, "generation total drift")
        _require(prior.get("generation_rows") == generation_binding, "generation binding drift")
        _require(prior.get("delivery") == delivery_summary, "delivery summary drift")
        prior_inputs = prior.get("inputs")
        prior_artifacts = prior.get("artifacts")
        _require(isinstance(prior_inputs, Mapping), "complete inputs missing")
        _require(isinstance(prior_artifacts, Mapping), "complete artifacts missing")
        _require(prior_inputs.get("samples") == samples_binding, "sample binding drift")
        _require(prior_artifacts.get("samples") == samples_binding, "artifact sample binding drift")
        _require(
            prior_artifacts.get("generation_rows") == generation_binding,
            "artifact generation binding drift",
        )
        prior_validation = prior.get("source_generation_validation")
        _require(isinstance(prior_validation, Mapping), "source validation missing")
        for key in (
            "mode",
            "samples",
            "replicates",
            "paired_tuples",
            "generation_rows",
            "model_counts",
            "finish_reason_counts",
            "shards",
            "gates",
            "base_lora_different_paired_tuples",
        ):
            _require(prior_validation.get(key) == validation.get(key), f"source validation drift: {key}")
        return {
            "status": "complete",
            "resume_noop": True,
            "generation_rows": generation_binding,
            "complete_evaluation_manifest": {
                "path": str(complete_manifest_path),
                "sha256": sha256_file(complete_manifest_path),
                "bytes": complete_manifest_path.stat().st_size,
            },
            "delivery": delivery_summary,
        }

    generation_binding = _write_new_jsonl(generation_rows_path, projected_rows)
    from open_r1.validator.loo_generation_spec import seal_manifest

    complete_manifest = seal_manifest({
        "schema_version": COMPLETE_EVALUATION_SCHEMA,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "model_order": list(MODEL_IDS),
        "k": REPLICATES,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "total_generations": expected_total,
        "inputs": {
            "samples": samples_binding,
            "prompt_token_ledger": dict(inputs.prompt_ledger_binding),
            "source_evaluation_manifest": {
                "path": str(inputs.manifest_path),
                "sha256": inputs.manifest_sha256,
                "bytes": inputs.manifest_path.stat().st_size,
            },
        },
        "artifacts": {
            "samples": samples_binding,
            "generation_rows": generation_binding,
        },
        "generation_rows": generation_binding,
        "models": list(MODEL_IDS),
        "generation_contract": _generation_contract(),
        "source_generation_validation": validation,
        "delivery": delivery_summary,
    })
    _write_new_json(complete_manifest_path, complete_manifest)
    return {
        "status": "complete",
        "resume_noop": False,
        "generation_rows": generation_binding,
        "complete_evaluation_manifest": {
            "path": str(complete_manifest_path),
            "sha256": sha256_file(complete_manifest_path),
            "bytes": complete_manifest_path.stat().st_size,
        },
        "delivery": complete_manifest["delivery"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("smoke", "generate"):
        child = subparsers.add_parser(command)
        child.add_argument("--manifest", required=True, type=Path)
        child.add_argument("--output-root", required=True, type=Path)
        child.add_argument("--shard-id", required=True, type=int, choices=SHARD_IDS)
        child.add_argument("--resume", action="store_true")
        if command == "smoke":
            child.add_argument("--smoke-samples", type=int, default=6)
            child.add_argument("--smoke-replicates", type=int, default=2)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True, type=Path)
    validate.add_argument("--output-root", required=True, type=Path)
    validate.add_argument("--smoke", action="store_true")
    validate.add_argument("--smoke-samples", type=int, default=6)
    validate.add_argument("--smoke-replicates", type=int, default=2)
    validate.add_argument("--receipt", type=Path)
    consolidate = subparsers.add_parser("consolidate")
    consolidate.add_argument("--manifest", required=True, type=Path)
    consolidate.add_argument(
        "--generation-output-root", "--output-root", required=True, type=Path
    )
    consolidate.add_argument(
        "--destination", "--destination-root", required=True, type=Path
    )
    consolidate.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"smoke", "generate"}:
        smoke = args.command == "smoke"
        result = run_worker(
            manifest_path=args.manifest,
            output_root=args.output_root,
            shard_id=args.shard_id,
            smoke=smoke,
            smoke_samples=getattr(args, "smoke_samples", 6),
            smoke_replicates=getattr(args, "smoke_replicates", 2),
            resume=args.resume,
        )
    elif args.command == "validate":
        result = validate_outputs(
            manifest_path=args.manifest,
            output_root=args.output_root,
            smoke=args.smoke,
            smoke_samples=args.smoke_samples,
            smoke_replicates=args.smoke_replicates,
        )
        if args.receipt is not None:
            _write_new_json(args.receipt.expanduser().resolve(), result)
    else:
        result = consolidate_outputs(
            manifest_path=args.manifest,
            generation_output_root=args.generation_output_root,
            destination_root=args.destination,
            resume=args.resume,
        )
    print(canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
