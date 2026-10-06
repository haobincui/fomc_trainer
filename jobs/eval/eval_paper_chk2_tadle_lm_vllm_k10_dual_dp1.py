"""Dual-DP1 vLLM generation for the paper-chk2 Tadle-form K=10 rerun.

Each invocation owns one physical GPU shard.  The first-stage target loads
chk0 alone; the second-stage target loads the chk1 base exactly once and uses
that engine first without LoRA for chk1 and then with the cp50 ``LoRARequest``
for paper chk2.  All three arms share exact prompt token IDs, per-row seeds,
and shard assignment, while each model retains its own fsynced generation WAL,
transport-attempt ledger, canonical rows, and immutable worker manifest.

Generation quality checks are diagnostic only.  In particular, a completion
without exactly one ``</think>`` boundary receives an empty parsed answer and
is retained rather than repaired, filtered, or heuristically recovered.
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
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import IO, Any

from jobs.eval import paper_chk2_tadle_lm_k10_common as common
from jobs.eval import paper_chk2_text_similarity_common as model_contract
from jobs.generation.generate_chk3_sft_targets import _attribution_categories
from jobs.retrain_v2 import probe_chk3_sft_degeneration as fidelity_probe
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import derive_row_seed, seal_manifest


ROOT = Path(__file__).resolve().parents[2]
DUAL_GPU_LAUNCHER = ROOT / "run/eval_paper_chk2_tadle_lm_vllm_k10_dual_gpu.sh"
SHARD_IDS = (0, 1)
SMOKE_MEETINGS = 2
SMOKE_REPLICATES = 2
MAX_TRANSPORT_ATTEMPTS = 3
EXPECTED_VLLM_VERSION = "0.8.5.post1"
EXPECTED_TRANSFORMERS_VERSION = "4.51.3"
EXPECTED_TOKENIZERS_VERSION = "0.21.1"
EXPECTED_TOKENIZER_CLASS = (
    "transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast"
)
GPU_HEADROOM_MIB = 2_048
TARGET_GPU_MEMORY_MIB = 20 * 1_024
MAX_GPU_COMPUTE_PERCENT = 10
LEGACY_IDLE_GATE_GENERATION_BYTES = 131_009
LEGACY_IDLE_GATE_GENERATION_SHA256 = (
    "7c8efecc034b84aeb08d2ce1e419499f436eae10a4154c3e791f9c2ba2446f0a"
)
LEGACY_IDLE_GATE_LAUNCHER_BYTES = 7_213
LEGACY_IDLE_GATE_LAUNCHER_SHA256 = (
    "6e9c4d56e8cc66bc77ded7ad8067e5f7fab32807575c9899e2d516d793b0861b"
)

ROW_SCHEMA = "paper-chk2-tadle-lm-k10-vllm-generation-row-v1"
ATTEMPT_SCHEMA = "paper-chk2-tadle-lm-k10-transport-attempt-v1"
SHARD_SCHEMA = "paper-chk2-tadle-lm-k10-vllm-shard-v1"
ENGINE_GROUP_SCHEMA = "paper-chk2-tadle-lm-k10-vllm-engine-group-v1"
ENGINE_INVOCATION_SCHEMA = "paper-chk2-tadle-lm-k10-engine-invocation-v1"
MODEL_VALIDATION_SCHEMA = "paper-chk2-tadle-lm-k10-model-validation-v1"
ALL_VALIDATION_SCHEMA = "paper-chk2-tadle-lm-k10-all-model-validation-v1"
DOCUMENT_ROW_SCHEMA = "paper-chk2-tadle-lm-k10-generated-document-v1"
DOCUMENT_MANIFEST_SCHEMA = "paper-chk2-tadle-lm-k10-document-manifest-v1"
TORN_TAIL_SCHEMA = "paper-chk2-tadle-lm-k10-wal-tail-recovery-v1"
LOCK_TEMPLATE = "/tmp/fomc_trainer_paper_chk2_tadle_gpu{shard}.lock"
CHK1_CP50_GROUP_ID = "chk1_paper_chk2_cp50"
WORKER_TARGET_IDS = ("chk0", CHK1_CP50_GROUP_ID)
SMOKE_AUTHORIZATION_GATES = frozenset(
    {
        "all_96_smoke_rows_present",
        "three_model_tuple_closure",
        "same_seed_prompt_and_gpu_across_models",
        "chk1_has_no_lora",
        "paper_chk2_cp50_has_required_lora",
        "no_quality_filtering_or_resampling",
        "all_rows_deep_replayed",
    }
)
BOUNDARY = "</think>"
CONTROL_MARKER_RE = re.compile(
    r"<(?:/?think|/?answer|/?analysis)>|<\|[^>]+\|>", re.IGNORECASE
)
DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b|"
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s+(?:19|20)\d{2}\b",
    re.IGNORECASE,
)


class PaperChk2TadleGenerationError(RuntimeError):
    """The immutable generation, WAL, route, or assembly contract failed."""


@dataclass(frozen=True, slots=True)
class GenerationInputs:
    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    ledger_rows: tuple[Mapping[str, Any], ...]
    model_paths: Mapping[str, Path]
    tokenizer_path: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PaperChk2TadleGenerationError(message)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    candidate = path.expanduser()
    _require(not candidate.is_symlink(), f"refusing JSON symlink: {candidate}")
    resolved = candidate.resolve()
    _require(resolved.is_file(), f"missing JSON: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperChk2TadleGenerationError(
            f"cannot read JSON {resolved}: {exc}"
        ) from exc
    _require(isinstance(value, dict), f"JSON object required: {resolved}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    candidate = path.expanduser()
    _require(not candidate.is_symlink(), f"refusing JSONL symlink: {candidate}")
    resolved = candidate.resolve()
    _require(resolved.is_file(), f"missing JSONL: {resolved}")
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            _require(
                raw.endswith("\n"), f"unterminated JSONL row: {resolved}:{line_number}"
            )
            _require(bool(raw.strip()), f"blank JSONL row: {resolved}:{line_number}")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise PaperChk2TadleGenerationError(
                    f"invalid JSONL row: {resolved}:{line_number}"
                ) from exc
            _require(
                isinstance(row, dict),
                f"JSONL object required: {resolved}:{line_number}",
            )
            rows.append(row)
    return rows


def _write_new_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.is_symlink(), f"refusing output symlink: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        published = True
        _fsync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        if published:
            _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_new_bytes(
        path,
        (
            json.dumps(
                value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
            )
            + "\n"
        ).encode("utf-8"),
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.is_symlink(), f"unsafe JSON symlink: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _append_fsync(handle: IO[str], value: Mapping[str, Any]) -> None:
    handle.write(common.canonical_json(dict(value)) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    candidate = path.expanduser()
    _require(not candidate.is_symlink(), f"refusing bound-file symlink: {candidate}")
    resolved = candidate.resolve()
    _require(resolved.is_file(), f"bound file missing: {resolved}")
    result: dict[str, Any] = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _require_bound_record(
    value: Any,
    path: Path,
    *,
    rows: int | None = None,
    label: str,
) -> dict[str, Any]:
    """Deep-replay one file binding rather than trusting a worker manifest."""

    _require(isinstance(value, dict), f"{label} binding must be an object")
    actual = _record(path, rows=rows)
    _require(value == actual, f"{label} binding drift")
    return actual


def _implementation_bindings() -> dict[str, Any]:
    return {
        "generation_module": _record(Path(__file__)),
        "dual_gpu_launcher": _record(DUAL_GPU_LAUNCHER),
    }


def _legacy_idle_gate_implementation_bindings() -> dict[str, Any]:
    """Binding used by sealed rows generated before the busy-GPU override."""

    return {
        "generation_module": {
            "path": str(Path(__file__).resolve()),
            "sha256": LEGACY_IDLE_GATE_GENERATION_SHA256,
            "bytes": LEGACY_IDLE_GATE_GENERATION_BYTES,
        },
        "dual_gpu_launcher": {
            "path": str(DUAL_GPU_LAUNCHER.resolve()),
            "sha256": LEGACY_IDLE_GATE_LAUNCHER_SHA256,
            "bytes": LEGACY_IDLE_GATE_LAUNCHER_BYTES,
        },
    }


def _implementation_binding_is_accepted(value: Any) -> bool:
    return value in (
        _implementation_bindings(),
        _legacy_idle_gate_implementation_bindings(),
    )


def _resume_launch_contract_matches(observed: Any, expected: Any) -> bool:
    """Allow sealed pre-override work while keeping every other field exact."""

    if not isinstance(observed, dict) or not isinstance(expected, dict):
        return False
    observed_without_implementation = dict(observed)
    expected_without_implementation = dict(expected)
    observed_implementation = observed_without_implementation.pop(
        "implementation", None
    )
    expected_without_implementation.pop("implementation", None)
    return (
        observed_without_implementation == expected_without_implementation
        and _implementation_binding_is_accepted(observed_implementation)
    )


def _write_or_verify_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        _require(
            path.is_file() and not path.is_symlink(), f"unsafe existing JSON: {path}"
        )
        _require(
            _read_json(path) == dict(value), f"existing JSON content drift: {path}"
        )
    else:
        _write_new_json(path, value)


def _write_or_verify_jsonl(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    encoded = "".join(common.canonical_json(dict(row)) + "\n" for row in rows).encode(
        "utf-8"
    )
    if path.exists():
        _require(
            path.is_file() and not path.is_symlink(), f"unsafe existing JSONL: {path}"
        )
        _require(path.read_bytes() == encoded, f"existing JSONL content drift: {path}")
    else:
        _write_new_bytes(path, encoded)
    return _record(path, rows=len(rows))


def _route_for_model(model_id: str) -> dict[str, Any]:
    _require(model_id in common.MODEL_IDS, f"invalid model ID: {model_id}")
    if model_id == "chk0":
        return {
            "base_model_path": str(model_contract.CHK0_MODEL.resolve()),
            "base_lineage_sha256": model_contract.CHK0_DIGEST,
            "base_runtime_payload_sha256": model_contract.CHK0_RUNTIME_PAYLOAD_DIGEST,
            "adapter_path": None,
            "adapter_model_sha256": None,
            "adapter_config_sha256": None,
            "lora_request": None,
            "tokenizer_path": str(model_contract.CHK1_MODEL.resolve()),
            "tokenizer_role": "single_frozen_chk1_cp200_tokenizer_for_all_models",
        }
    if model_id == "chk1":
        return {
            "base_model_path": str(model_contract.CHK1_MODEL.resolve()),
            "base_lineage_sha256": model_contract.CHK1_DIGEST,
            "base_runtime_payload_sha256": model_contract.CHK1_DIGEST,
            "adapter_path": None,
            "adapter_model_sha256": None,
            "adapter_config_sha256": None,
            "lora_request": None,
            "tokenizer_path": str(model_contract.CHK1_MODEL.resolve()),
            "tokenizer_role": "single_frozen_chk1_cp200_tokenizer_for_all_models",
        }
    return {
        "base_model_path": str(model_contract.CHK1_MODEL.resolve()),
        "base_lineage_sha256": model_contract.CHK1_DIGEST,
        "base_runtime_payload_sha256": model_contract.CHK1_DIGEST,
        "adapter_path": str(model_contract.CHK2_ADAPTER.resolve()),
        "adapter_model_sha256": model_contract.CHK2_ADAPTER_MODEL_SHA256,
        "adapter_config_sha256": model_contract.CHK2_ADAPTER_CONFIG_SHA256,
        "lora_request": {
            "name": "paper-chk2-cp50",
            "integer_id": 50,
            "max_lora_rank": 32,
            "max_loras": 1,
            "max_cpu_loras": 1,
            "lora_dtype": "bfloat16",
        },
        "tokenizer_path": str(model_contract.CHK1_MODEL.resolve()),
        "tokenizer_role": "single_frozen_chk1_cp200_tokenizer_for_all_models",
    }


def load_generation_inputs(manifest_path: Path, output_root: Path) -> GenerationInputs:
    """Load the prepared ledger and re-verify every model runtime binding."""

    prepared = common.validate_preparation_artifacts(
        output_root, verify_static_bindings=False
    )
    manifest_candidate = manifest_path.expanduser()
    _require(
        not manifest_candidate.is_symlink(), "refusing evaluation-manifest symlink"
    )
    resolved_manifest = manifest_candidate.resolve()
    _require(
        resolved_manifest == prepared.manifest_path.resolve(),
        "manifest/output-root binding drift",
    )
    manifest_sha = sha256_file(resolved_manifest)
    # This validates chk0's runtime allowlist, the exact chk1 directory digest,
    # and cp50's adapter payload before every worker starts.
    model_contract._verify_static_bindings()
    models = prepared.manifest.get("models") or {}
    _require(set(models) == set(common.MODEL_IDS), "prepared model inventory drift")
    _require(
        models["chk0"].get("lineage_full_directory_sha256")
        == model_contract.CHK0_DIGEST,
        "prepared chk0 lineage drift",
    )
    _require(
        models["chk1"].get("sha256") == model_contract.CHK1_DIGEST,
        "prepared chk1 digest drift",
    )
    _require(
        models["paper_chk2_cp50"].get("adapter_model_sha256")
        == model_contract.CHK2_ADAPTER_MODEL_SHA256,
        "prepared cp50 adapter drift",
    )
    return GenerationInputs(
        manifest_path=resolved_manifest,
        manifest_sha256=manifest_sha,
        manifest=prepared.manifest,
        ledger_rows=prepared.ledger_rows,
        model_paths={
            "chk0": model_contract.CHK0_MODEL.resolve(),
            "chk1": model_contract.CHK1_MODEL.resolve(),
            "paper_chk2_cp50": model_contract.CHK1_MODEL.resolve(),
            "paper_chk2_cp50_adapter": model_contract.CHK2_ADAPTER.resolve(),
        },
        tokenizer_path=model_contract.CHK1_MODEL.resolve(),
    )


def tuple_id(meeting_id: str, topic_rank: int, replicate_id: int) -> str:
    return f"{meeting_id}|topic-{topic_rank:02d}|replicate-{replicate_id:02d}"


def paired_block_id(meeting_id: str, replicate_id: int) -> str:
    return f"{meeting_id}|replicate-{replicate_id:02d}"


def generation_key(
    model_id: str, meeting_id: str, topic_rank: int, replicate_id: int
) -> str:
    _require(model_id in common.MODEL_IDS, f"invalid model ID: {model_id}")
    return f"{model_id}|{tuple_id(meeting_id, topic_rank, replicate_id)}"


def assigned_shard(meeting_id: str, replicate_id: int) -> int:
    payload = paired_block_id(meeting_id, replicate_id).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(SHARD_IDS)


def build_case_matrix(
    inputs: GenerationInputs,
    *,
    model_id: str,
    smoke: bool = False,
    smoke_meetings: int = SMOKE_MEETINGS,
    smoke_replicates: int = SMOKE_REPLICATES,
) -> list[dict[str, Any]]:
    """Build one arm's canonical 6,720 cases (or 32-row smoke prefix)."""

    _require(model_id in common.MODEL_IDS, f"invalid model ID: {model_id}")
    _require(
        1 <= smoke_meetings <= common.EXPECTED_MEETINGS, "invalid smoke meeting count"
    )
    _require(
        1 <= smoke_replicates <= len(common.REPLICATE_SEEDS),
        "invalid smoke replicate count",
    )
    meeting_ids = list(inputs.manifest["population"]["meeting_ids"])
    selected_meetings = set(meeting_ids[:smoke_meetings] if smoke else meeting_ids)
    replicate_ids = range(smoke_replicates if smoke else len(common.REPLICATE_SEEDS))
    route = _route_for_model(model_id)
    cases: list[dict[str, Any]] = []
    for ledger_row in inputs.ledger_rows:
        meeting_id = str(ledger_row["meeting_id"])
        if meeting_id not in selected_meetings:
            continue
        topic = str(ledger_row["topic"])
        topic_rank = int(ledger_row["topic_rank"])
        seed_key = f"{meeting_id}|{topic}"
        for replicate_id in replicate_ids:
            replicate_seed = common.REPLICATE_SEEDS[replicate_id]
            row_seed = derive_row_seed(replicate_seed, seed_key)
            case = {
                "absolute_case_index": int(ledger_row["absolute_prompt_index"])
                * len(common.REPLICATE_SEEDS)
                + replicate_id,
                "tuple_id": tuple_id(meeting_id, topic_rank, replicate_id),
                "paired_block_id": paired_block_id(meeting_id, replicate_id),
                "generation_key": generation_key(
                    model_id, meeting_id, topic_rank, replicate_id
                ),
                "model_id": model_id,
                "paper_label": common.MODEL_LABELS[model_id],
                "sample_id": ledger_row["sample_id"],
                "meeting_id": meeting_id,
                "meeting_start_date": ledger_row["meeting_start_date"],
                "meeting_end_date": ledger_row["meeting_end_date"],
                "meeting_rank": ledger_row["meeting_rank"],
                "topic": topic,
                "topic_rank": topic_rank,
                "replicate_id": replicate_id,
                "replicate_seed": replicate_seed,
                "row_seed": row_seed,
                "shard_id": assigned_shard(meeting_id, replicate_id),
                "source_analysis": ledger_row["source_analysis"],
                "source_analysis_sha256": ledger_row["source_analysis_sha256"],
                "user_prompt_sha256": ledger_row["user_prompt_sha256"],
                "messages_sha256": ledger_row["messages_sha256"],
                "prompt_token_count": ledger_row["prompt_token_count"],
                "prompt_token_ids": ledger_row["prompt_token_ids"],
                "prompt_token_ids_sha256": ledger_row["prompt_token_ids_sha256"],
                "route": route,
            }
            cases.append(case)
    expected = (
        smoke_meetings * len(common.TOPICS) * smoke_replicates
        if smoke
        else common.EXPECTED_GENERATIONS_PER_MODEL
    )
    _require(
        len(cases) == expected,
        f"case matrix cardinality drift: {len(cases)} != {expected}",
    )
    _require(
        len({str(case["generation_key"]) for case in cases}) == expected,
        "duplicate generation key",
    )
    blocks: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        blocks[str(case["paired_block_id"])].append(case)
    for block, rows in blocks.items():
        _require(len(rows) == len(common.TOPICS), f"paired block is not Core8: {block}")
        _require(
            len({int(row["shard_id"]) for row in rows}) == 1,
            f"paired block crosses GPUs: {block}",
        )
    return cases


def _case_index(cases: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for case in cases:
        key = str(case["generation_key"])
        _require(key not in result, f"duplicate expected generation key: {key}")
        result[key] = case
    return result


def _retry_start_index(generation_key_value: str, failure_count: int) -> int:
    _require(
        isinstance(failure_count, int)
        and not isinstance(failure_count, bool)
        and 0 <= failure_count < MAX_TRANSPORT_ATTEMPTS,
        f"transport retry budget exhausted: {generation_key_value}",
    )
    return failure_count


def _validate_paired_tuple_group(
    tuple_key: str, rows: Sequence[Mapping[str, Any]]
) -> bool:
    """Validate one cross-model common-random-number tuple."""

    _require(
        len(rows) == len(common.MODEL_IDS),
        f"three-model tuple closure drift: {tuple_key}",
    )
    _require(
        {str(row["model_id"]) for row in rows} == set(common.MODEL_IDS),
        f"tuple model inventory drift: {tuple_key}",
    )
    _require(
        len({int(row["row_seed"]) for row in rows}) == 1,
        f"tuple seed drift: {tuple_key}",
    )
    _require(
        len({int(row["shard_id"]) for row in rows}) == 1,
        f"tuple GPU drift: {tuple_key}",
    )
    _require(
        len({str(row["prompt_token_ids_sha256"]) for row in rows}) == 1,
        f"tuple prompt-token drift: {tuple_key}",
    )
    by_model = {str(row["model_id"]): row for row in rows}
    return (
        by_model["chk1"]["output_token_ids"]
        != by_model["paper_chk2_cp50"]["output_token_ids"]
    )


def compute_gpu_memory_utilization(
    free_mib: int,
    total_mib: int,
    *,
    allow_busy_gpu: bool = False,
) -> float:
    _require(0 < free_mib <= total_mib, "invalid GPU memory observation")
    required_free_mib = TARGET_GPU_MEMORY_MIB + (
        0 if allow_busy_gpu else GPU_HEADROOM_MIB
    )
    _require(
        free_mib >= required_free_mib,
        "insufficient GPU memory for the fixed 20 GiB budget"
        + ("" if allow_busy_gpu else " plus 2 GiB headroom"),
    )
    value = math.floor((TARGET_GPU_MEMORY_MIB / total_mib) * 100.0) / 100.0
    _require(
        0.80 <= value <= 0.83,
        f"GPU is not compatible with the ~20 GiB A30 budget: {value:.2f}",
    )
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
    _require(
        completed.returncode == 0, f"nvidia-smi failed: {completed.stderr.strip()}"
    )
    fields = [value.strip() for value in completed.stdout.strip().split(",")]
    _require(len(fields) == 7, "unexpected nvidia-smi output")
    result = {
        "physical_gpu_index": int(fields[0]),
        "uuid": fields[1],
        "name": fields[2],
        "memory_total_mib": int(fields[3]),
        "memory_used_mib": int(fields[4]),
        "memory_free_mib": int(fields[5]),
        "utilization_gpu_percent": int(fields[6]),
    }
    allow_busy_gpu_raw = os.environ.get("PAPER_CHK2_ALLOW_BUSY_GPU", "0")
    _require(
        allow_busy_gpu_raw in {"0", "1"},
        "PAPER_CHK2_ALLOW_BUSY_GPU must be 0 or 1",
    )
    allow_busy_gpu = allow_busy_gpu_raw == "1"
    _require(
        allow_busy_gpu
        or result["utilization_gpu_percent"] <= MAX_GPU_COMPUTE_PERCENT,
        "GPU is busy; launcher must wait rather than interrupt existing work",
    )
    result["vllm_gpu_memory_utilization"] = compute_gpu_memory_utilization(
        result["memory_free_mib"],
        result["memory_total_mib"],
        allow_busy_gpu=allow_busy_gpu,
    )
    result["vllm_target_memory_mib"] = TARGET_GPU_MEMORY_MIB
    result["busy_gpu_override"] = allow_busy_gpu
    return result


def _load_tokenizer(path: Path) -> tuple[Any, list[int], int]:
    _require(
        _package_version("transformers") == EXPECTED_TRANSFORMERS_VERSION,
        "worker transformers version drift",
    )
    _require(
        _package_version("tokenizers") == EXPECTED_TOKENIZERS_VERSION,
        "worker tokenizers version drift",
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
        fix_mistral_regex=False,
    )
    class_name = f"{type(tokenizer).__module__}.{type(tokenizer).__name__}"
    _require(
        class_name == EXPECTED_TOKENIZER_CLASS,
        f"worker tokenizer class drift: {class_name}",
    )
    _require(
        isinstance(tokenizer.chat_template, str)
        and common.sha256_text(tokenizer.chat_template)
        == "56a1447ad31926fdc21fb07e56e5642bd9c850c4f52d8c8af7bbe5f079a84f5f",
        "worker tokenizer chat-template drift",
    )
    eos = tokenizer.eos_token_id
    eos_ids = [int(value) for value in eos] if isinstance(eos, list) else [int(eos)]
    _require(eos_ids and all(value >= 0 for value in eos_ids), "invalid EOS token IDs")
    pad = tokenizer.pad_token_id
    return tokenizer, eos_ids, int(eos_ids[0] if pad is None else pad)


def _ngram_repetition(token_ids: Sequence[int], n: int = 4) -> float:
    if len(token_ids) < n:
        return 0.0
    grams = [
        tuple(token_ids[index : index + n]) for index in range(len(token_ids) - n + 1)
    ]
    return 1.0 - len(set(grams)) / len(grams)


def _strict_answer(completion: str) -> tuple[str, int]:
    count = completion.count(BOUNDARY)
    answer = completion.split(BOUNDARY, 1)[1].strip() if count == 1 else ""
    return answer, count


def _diagnostics(
    *,
    completion: str,
    output_token_ids: Sequence[int],
    finish_reason: Any,
    source_analysis: str,
) -> dict[str, Any]:
    answer, boundary_count = _strict_answer(completion)
    source_numbers = fidelity_probe._numeric_values(source_analysis)
    answer_numbers = fidelity_probe._numeric_values(answer)
    source_dates = fidelity_probe._date_values(source_analysis)
    answer_dates = fidelity_probe._date_values(answer)
    source_attributions = _attribution_categories(source_analysis)
    answer_attributions = _attribution_categories(answer)
    full_repetition = _ngram_repetition(output_token_ids)
    tail_repetition = _ngram_repetition(output_token_ids[-1024:])
    normalized_finish = finish_reason if isinstance(finish_reason, str) else "unknown"
    failures: list[str] = []
    if normalized_finish != "stop":
        failures.append("finish_reason_not_stop")
    if boundary_count != 1:
        failures.append("think_boundary_count_not_one")
    if not answer:
        failures.append("empty_strict_answer")
    if answer and re.search(r"[\r\n\u2028\u2029]", answer):
        failures.append("answer_not_single_paragraph")
    if answer and CONTROL_MARKER_RE.search(answer):
        failures.append("answer_contains_control_marker")
    if source_numbers != answer_numbers:
        failures.append("numeric_multiset_not_preserved")
    if source_dates != answer_dates:
        failures.append("date_set_not_preserved")
    if source_attributions != answer_attributions:
        failures.append("attribution_set_not_preserved")
    if full_repetition >= 0.50:
        failures.append("full_4gram_repetition_ge_0.50")
    if tail_repetition >= 0.60:
        failures.append("tail_4gram_repetition_ge_0.60")
    failures = sorted(set(failures))
    return {
        "answer": answer,
        "answer_sha256": common.sha256_text(answer),
        "think_boundary_count": boundary_count,
        "normal_finish": normalized_finish == "stop",
        "answer_nonempty": bool(answer),
        "answer_single_paragraph": bool(answer)
        and not bool(re.search(r"[\r\n\u2028\u2029]", answer)),
        "answer_control_marker_free": bool(answer)
        and CONTROL_MARKER_RE.search(answer) is None,
        "numeric_multiset_preserved": source_numbers == answer_numbers,
        "date_set_preserved": source_dates == answer_dates,
        "attribution_set_preserved": source_attributions == answer_attributions,
        "full_token_4gram_repetition": round(full_repetition, 8),
        "tail_token_4gram_repetition": round(tail_repetition, 8),
        "degeneration_free": full_repetition < 0.50 and tail_repetition < 0.60,
        "diagnostic_pass": not failures,
        "diagnostic_failures": failures,
        "diagnostics_used_for_filtering_or_resampling": False,
    }


def _delivery_funnel(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize diagnostic delivery/fidelity outcomes without filtering."""

    finish_reasons = Counter(str(row.get("finish_reason")) for row in rows)
    failure_counts: Counter[str] = Counter()
    for row in rows:
        failure_counts.update(
            str(value) for value in row.get("diagnostic_failures") or ()
        )
    return {
        "rows": len(rows),
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "normal_finish": sum(bool(row.get("normal_finish")) for row in rows),
        "unique_think_boundary": sum(
            int(row.get("think_boundary_count", -1)) == 1 for row in rows
        ),
        "nonempty_answer": sum(bool(row.get("answer_nonempty")) for row in rows),
        "single_paragraph_answer": sum(
            bool(row.get("answer_single_paragraph")) for row in rows
        ),
        "numeric_multiset_preserved": sum(
            bool(row.get("numeric_multiset_preserved")) for row in rows
        ),
        "date_set_preserved": sum(bool(row.get("date_set_preserved")) for row in rows),
        "attribution_set_preserved": sum(
            bool(row.get("attribution_set_preserved")) for row in rows
        ),
        "degeneration_free": sum(bool(row.get("degeneration_free")) for row in rows),
        "diagnostic_pass": sum(bool(row.get("diagnostic_pass")) for row in rows),
        "diagnostic_fail": sum(not bool(row.get("diagnostic_pass")) for row in rows),
        "diagnostic_failure_counts": dict(sorted(failure_counts.items())),
        "diagnostic_only": True,
        "rows_filtered": 0,
        "rows_resampled_for_quality": 0,
    }


def _build_result(
    *,
    case: Mapping[str, Any],
    generated_text: str,
    generated_token_ids: Sequence[int],
    raw_finish_reason: Any,
    raw_stop_reason: Any,
    attempt_index: int,
    manifest_sha: str,
    tokenizer: Any,
    eos_token_ids: Sequence[int],
) -> dict[str, Any]:
    token_ids = [int(value) for value in generated_token_ids]
    _require(all(value >= 0 for value in token_ids), "invalid generated token IDs")
    try:
        decoded = fidelity_probe.common_probe.decode_completion_preserving_boundary(
            tokenizer, token_ids, eos_token_ids
        )
    except fidelity_probe.common_probe.ProbeError as exc:
        raise PaperChk2TadleGenerationError(str(exc)) from exc
    _require(decoded == generated_text, "vLLM completion text/token deep-replay drift")
    diagnostics = _diagnostics(
        completion=generated_text,
        output_token_ids=token_ids,
        finish_reason=raw_finish_reason,
        source_analysis=str(case["source_analysis"]),
    )
    return {
        "schema_version": ROW_SCHEMA,
        "evaluation_id": common.EVALUATION_ID,
        "evaluation_manifest_sha256": manifest_sha,
        **{key: value for key, value in case.items() if key != "prompt_token_ids"},
        "physical_gpu_index": case["shard_id"],
        "request_id": f"tadle-k10-{case['generation_key']}-attempt-{attempt_index}",
        "attempt_index": attempt_index,
        "attempt_count": attempt_index + 1,
        "completion": generated_text,
        "completion_sha256": common.sha256_text(generated_text),
        "output_token_ids": token_ids,
        "output_token_ids_sha256": common.sha256_text(common.canonical_json(token_ids)),
        "raw_generated_tokens": len(token_ids),
        "finish_reason": raw_finish_reason,
        "stop_reason": raw_stop_reason,
        **diagnostics,
        "generation_contract": common.generation_contract(),
        "completed_at_utc": _utc_now(),
    }


def _validate_wal_row(
    row: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    manifest_sha: str,
    tokenizer: Any,
    eos_token_ids: Sequence[int],
) -> None:
    _require(row.get("schema_version") == ROW_SCHEMA, "generation WAL schema drift")
    _require(
        row.get("evaluation_manifest_sha256") == manifest_sha,
        "generation WAL manifest drift",
    )
    for field in (
        "absolute_case_index",
        "tuple_id",
        "paired_block_id",
        "generation_key",
        "model_id",
        "paper_label",
        "sample_id",
        "meeting_id",
        "meeting_start_date",
        "meeting_end_date",
        "meeting_rank",
        "topic",
        "topic_rank",
        "replicate_id",
        "replicate_seed",
        "row_seed",
        "shard_id",
        "source_analysis",
        "source_analysis_sha256",
        "user_prompt_sha256",
        "messages_sha256",
        "prompt_token_count",
        "prompt_token_ids_sha256",
        "route",
    ):
        _require(
            row.get(field) == case.get(field), f"generation WAL case drift at {field}"
        )
    text = row.get("completion")
    token_ids = row.get("output_token_ids")
    _require(isinstance(text, str), "generation WAL completion is not text")
    _require(
        isinstance(token_ids, list)
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in token_ids
        ),
        "generation WAL output token IDs are invalid",
    )
    _require(
        row.get("completion_sha256") == common.sha256_text(text), "completion SHA drift"
    )
    _require(
        row.get("output_token_ids_sha256")
        == common.sha256_text(common.canonical_json(token_ids)),
        "output token SHA drift",
    )
    _require(
        row.get("raw_generated_tokens") == len(token_ids), "generated token count drift"
    )
    _require(
        row.get("generation_contract") == common.generation_contract(),
        "generation contract drift",
    )
    attempt_index = row.get("attempt_index")
    _require(
        isinstance(attempt_index, int) and 0 <= attempt_index < MAX_TRANSPORT_ATTEMPTS,
        "attempt index drift",
    )
    _require(row.get("attempt_count") == attempt_index + 1, "attempt count drift")
    expected = _build_result(
        case=case,
        generated_text=text,
        generated_token_ids=token_ids,
        raw_finish_reason=row.get("finish_reason"),
        raw_stop_reason=row.get("stop_reason"),
        attempt_index=attempt_index,
        manifest_sha=manifest_sha,
        tokenizer=tokenizer,
        eos_token_ids=eos_token_ids,
    )
    for key, value in expected.items():
        if key != "completed_at_utc":
            _require(
                row.get(key) == value, f"generation WAL deep-replay drift at {key}"
            )


def _recover_torn_tail(path: Path) -> None:
    if not path.exists():
        return
    _require(path.is_file() and not path.is_symlink(), f"unsafe WAL: {path}")
    payload = path.read_bytes()
    if not payload or payload.endswith(b"\n"):
        return
    recovered_bytes = payload.rfind(b"\n") + 1
    discarded = payload[recovered_bytes:]
    _require(bool(discarded), "torn-tail detection drift")
    with path.open("r+b") as handle:
        handle.truncate(recovered_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    recovery_path = path.with_name(f"{path.name}.torn-tail-recoveries.jsonl")
    _require(not recovery_path.is_symlink(), "unsafe torn-tail recovery ledger")
    with recovery_path.open("a", encoding="utf-8") as handle:
        _append_fsync(
            handle,
            {
                "schema_version": TORN_TAIL_SCHEMA,
                "recovered_at_utc": _utc_now(),
                "wal_path": str(path.resolve()),
                "original_bytes": len(payload),
                "recovered_bytes": recovered_bytes,
                "discarded_bytes": len(discarded),
                "discarded_sha256": hashlib.sha256(discarded).hexdigest(),
                "policy": "discard_only_final_unterminated_jsonl_fragment",
            },
        )


def load_attempt_ledger(
    path: Path,
    *,
    cases_by_key: Mapping[str, Mapping[str, Any]],
    manifest_sha: str,
) -> dict[str, list[dict[str, Any]]]:
    _recover_torn_tail(path)
    if not path.exists():
        return {}
    indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(path):
        _require(
            row.get("schema_version") == ATTEMPT_SCHEMA, "attempt ledger schema drift"
        )
        _require(
            row.get("evaluation_manifest_sha256") == manifest_sha,
            "attempt manifest drift",
        )
        key = row.get("generation_key")
        _require(
            isinstance(key, str) and key in cases_by_key, f"unknown attempt key: {key}"
        )
        case = cases_by_key[key]
        failures = indexed[key]
        _require(
            row.get("status") == "transport_failure",
            "attempt ledger contains nonfailure",
        )
        _require(
            row.get("attempt_index") == len(failures),
            f"noncontiguous attempt index: {key}",
        )
        _require(row.get("row_seed") == case["row_seed"], f"attempt seed drift: {key}")
        _require(row.get("model_id") == case["model_id"], f"attempt model drift: {key}")
        _require(row.get("route") == case["route"], f"attempt route drift: {key}")
        failures.append(row)
    return dict(indexed)


def load_engine_invocation_ledger(
    path: Path,
    *,
    manifest_sha: str,
    shard_id: int,
) -> list[dict[str, Any]]:
    _recover_torn_tail(path)
    if not path.exists():
        return []
    rows = _read_jsonl(path)
    seen: set[str] = set()
    for index, row in enumerate(rows):
        _require(
            row.get("schema_version") == ENGINE_INVOCATION_SCHEMA,
            "engine invocation schema drift",
        )
        _require(
            row.get("evaluation_manifest_sha256") == manifest_sha,
            "engine invocation manifest drift",
        )
        _require(
            row.get("engine_group_id") == CHK1_CP50_GROUP_ID,
            "engine invocation group drift",
        )
        _require(row.get("shard_id") == shard_id, "engine invocation shard drift")
        _require(row.get("invocation_index") == index, "engine invocation index drift")
        invocation_id = row.get("invocation_id")
        _require(
            isinstance(invocation_id, str) and invocation_id not in seen,
            "duplicate/invalid engine invocation ID",
        )
        seen.add(invocation_id)
        _require(
            row.get("model_load_count") == 1,
            "engine invocation must record one base load",
        )
        _require(
            row.get("model_order") == ["chk1", "paper_chk2_cp50"],
            "engine invocation order drift",
        )
    return rows


def load_generation_wal(
    path: Path,
    *,
    cases_by_key: Mapping[str, Mapping[str, Any]],
    manifest_sha: str,
    tokenizer: Any,
    eos_token_ids: Sequence[int],
) -> dict[str, dict[str, Any]]:
    _recover_torn_tail(path)
    if not path.exists():
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        key = row.get("generation_key")
        _require(
            isinstance(key, str) and key in cases_by_key,
            f"unknown generation WAL key: {key}",
        )
        _require(key not in indexed, f"duplicate generation WAL key: {key}")
        _validate_wal_row(
            row,
            cases_by_key[key],
            manifest_sha=manifest_sha,
            tokenizer=tokenizer,
            eos_token_ids=eos_token_ids,
        )
        indexed[key] = row
    return indexed


def _validate_visible_gpu(shard_id: int) -> None:
    _require(
        os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
        "CUDA_DEVICE_ORDER must be PCI_BUS_ID",
    )
    _require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == str(shard_id),
        f"CUDA_VISIBLE_DEVICES must be exactly {shard_id}",
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
            raise PaperChk2TadleGenerationError(
                f"GPU{shard_id} Tadle worker lock is held"
            ) from exc
        yield {"path": str(path), "pid": os.getpid()}
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


async def _run_engine(
    *,
    model_id: str,
    inputs: GenerationInputs,
    cases: Sequence[Mapping[str, Any]],
    manifest_sha: str,
    tokenizer: Any,
    eos_token_ids: Sequence[int],
    gpu_memory_utilization: float,
    initial_failure_counts: Mapping[str, int],
    on_failure: Any,
    on_result: Any,
) -> dict[str, Any]:
    import torch
    import vllm
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.lora.request import LoRARequest

    _require(
        vllm.__version__ == EXPECTED_VLLM_VERSION,
        f"vLLM version drift: {vllm.__version__}",
    )
    _require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "worker requires one visible CUDA GPU",
    )
    route = _route_for_model(model_id)
    enable_lora = model_id == "paper_chk2_cp50"
    _require(enable_lora == (route["lora_request"] is not None), "LoRA route drift")
    engine_args = AsyncEngineArgs(
        model=str(inputs.model_paths[model_id]),
        tokenizer=str(inputs.tokenizer_path),
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
        max_model_len=common.MAX_MODEL_LEN,
        max_num_seqs=common.MAX_NUM_SEQS,
        max_num_batched_tokens=common.MAX_NUM_BATCHED_TOKENS,
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
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    load_seconds = time.perf_counter() - load_started
    lora_request = (
        LoRARequest(
            "paper-chk2-cp50",
            50,
            lora_path=str(inputs.model_paths["paper_chk2_cp50_adapter"]),
        )
        if enable_lora
        else None
    )

    async def consume(case: Mapping[str, Any]) -> None:
        key = str(case["generation_key"])
        start_attempt = _retry_start_index(key, int(initial_failure_counts.get(key, 0)))
        last_error: BaseException | None = None
        for attempt_index in range(start_attempt, MAX_TRANSPORT_ATTEMPTS):
            request_id = f"tadle-k10-{key}-attempt-{attempt_index}"
            try:
                params = SamplingParams(
                    n=1,
                    temperature=common.TEMPERATURE,
                    top_p=common.TOP_P,
                    top_k=common.TOP_K,
                    repetition_penalty=common.REPETITION_PENALTY,
                    max_tokens=common.MAX_NEW_TOKENS,
                    seed=int(case["row_seed"]),
                    detokenize=True,
                    skip_special_tokens=False,
                    spaces_between_special_tokens=True,
                )
                final = None
                async for output in engine.generate(
                    {"prompt_token_ids": list(case["prompt_token_ids"])},
                    params,
                    request_id,
                    lora_request=lora_request,
                ):
                    final = output
                _require(final is not None, f"vLLM returned no output: {request_id}")
                _require(
                    len(getattr(final, "outputs", []) or []) == 1,
                    f"vLLM n drift: {request_id}",
                )
                _require(
                    list(getattr(final, "prompt_token_ids", []) or [])
                    == list(case["prompt_token_ids"]),
                    f"vLLM prompt-token drift: {request_id}",
                )
                completion = final.outputs[0]
                text = getattr(completion, "text", None)
                _require(isinstance(text, str), f"vLLM returned non-text: {request_id}")
                row = _build_result(
                    case=case,
                    generated_text=text,
                    generated_token_ids=list(
                        getattr(completion, "token_ids", []) or []
                    ),
                    raw_finish_reason=getattr(completion, "finish_reason", None),
                    raw_stop_reason=getattr(completion, "stop_reason", None),
                    attempt_index=attempt_index,
                    manifest_sha=manifest_sha,
                    tokenizer=tokenizer,
                    eos_token_ids=eos_token_ids,
                )
            except Exception as exc:  # transport/runtime retry boundary is intentional
                last_error = exc
                await on_failure(
                    {
                        "schema_version": ATTEMPT_SCHEMA,
                        "evaluation_manifest_sha256": manifest_sha,
                        "generation_key": key,
                        "model_id": model_id,
                        "shard_id": case["shard_id"],
                        "row_seed": case["row_seed"],
                        "attempt_index": attempt_index,
                        "request_id": request_id,
                        "status": "transport_failure",
                        "error_type": type(exc).__name__,
                        "error_message_sha256": common.sha256_text(str(exc)),
                        "route": route,
                        "failed_at_utc": _utc_now(),
                    }
                )
                continue
            # Persistence failures are not provider/transport failures and must
            # never consume the model retry budget after a successful response.
            await on_result(row)
            return
        assert last_error is not None
        raise PaperChk2TadleGenerationError(
            f"transport retry budget exhausted for {key}: {type(last_error).__name__}"
        ) from last_error

    started = time.perf_counter()
    try:
        for start in range(0, len(cases), 64):
            tasks = [
                asyncio.create_task(consume(case)) for case in cases[start : start + 64]
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        if hasattr(engine, "shutdown"):
            engine.shutdown()
        elif hasattr(engine, "shutdown_background_loop"):
            engine.shutdown_background_loop()
    return {
        "model_id": model_id,
        "model_path": str(inputs.model_paths[model_id]),
        "tokenizer_path": str(inputs.tokenizer_path),
        "adapter_path": route["adapter_path"],
        "enable_lora": enable_lora,
        "lora_request": route["lora_request"],
        "model_load_wall_seconds": load_seconds,
        "generation_wall_seconds": time.perf_counter() - started,
        "new_rows": len(cases),
    }


async def _run_chk1_cp50_engine_group(
    *,
    inputs: GenerationInputs,
    cases_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    manifest_sha: str,
    tokenizer: Any,
    eos_token_ids: Sequence[int],
    gpu_memory_utilization: float,
    initial_failure_counts: Mapping[str, Mapping[str, int]],
    shard_id: int,
    invocation_id: str,
    invocation_index: int,
    on_engine_loaded: Any,
    on_failure: Any,
    on_result: Any,
) -> dict[str, Any]:
    """Load chk1 once, then generate base and cp50 routes in strict order."""

    import torch
    import vllm
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.lora.request import LoRARequest

    _require(
        vllm.__version__ == EXPECTED_VLLM_VERSION,
        f"vLLM version drift: {vllm.__version__}",
    )
    _require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "worker requires one visible CUDA GPU",
    )
    model_order = ("chk1", "paper_chk2_cp50")
    _require(
        set(cases_by_model) == set(model_order), "chk1/cp50 engine-group model drift"
    )
    route_cp50 = _route_for_model("paper_chk2_cp50")
    engine_args = AsyncEngineArgs(
        model=str(inputs.model_paths["chk1"]),
        tokenizer=str(inputs.tokenizer_path),
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
        max_model_len=common.MAX_MODEL_LEN,
        max_num_seqs=common.MAX_NUM_SEQS,
        max_num_batched_tokens=common.MAX_NUM_BATCHED_TOKENS,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        generation_config="vllm",
        seed=0,
        disable_log_requests=True,
        disable_log_stats=True,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
        max_cpu_loras=1,
        lora_dtype="bfloat16",
    )
    load_started = time.perf_counter()
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    load_seconds = time.perf_counter() - load_started
    lora_request = LoRARequest(
        "paper-chk2-cp50",
        50,
        lora_path=str(inputs.model_paths["paper_chk2_cp50_adapter"]),
    )
    try:
        await on_engine_loaded(
            {
                "schema_version": ENGINE_INVOCATION_SCHEMA,
                "evaluation_manifest_sha256": manifest_sha,
                "engine_group_id": CHK1_CP50_GROUP_ID,
                "shard_id": shard_id,
                "invocation_id": invocation_id,
                "invocation_index": invocation_index,
                "model_order": ["chk1", "paper_chk2_cp50"],
                "pending_rows_by_model": {
                    model_id: len(cases_by_model[model_id])
                    for model_id in ("chk1", "paper_chk2_cp50")
                },
                "base_model_path": str(inputs.model_paths["chk1"]),
                "adapter_path": str(inputs.model_paths["paper_chk2_cp50_adapter"]),
                "model_load_count": 1,
                "model_load_wall_seconds": load_seconds,
                "loaded_at_utc": _utc_now(),
            }
        )
    except BaseException:
        if hasattr(engine, "shutdown"):
            engine.shutdown()
        elif hasattr(engine, "shutdown_background_loop"):
            engine.shutdown_background_loop()
        raise
    per_model: dict[str, dict[str, Any]] = {}
    group_started = time.perf_counter()

    async def generate_model(model_id: str) -> None:
        cases = list(cases_by_model[model_id])
        request = None if model_id == "chk1" else lora_request
        route = _route_for_model(model_id)
        _require(
            (request is None) == (model_id == "chk1"),
            "grouped base/LoRA request route drift",
        )

        async def consume(case: Mapping[str, Any]) -> None:
            key = str(case["generation_key"])
            start_attempt = _retry_start_index(
                key,
                int(initial_failure_counts.get(model_id, {}).get(key, 0)),
            )
            last_error: BaseException | None = None
            for attempt_index in range(start_attempt, MAX_TRANSPORT_ATTEMPTS):
                request_id = f"tadle-k10-{key}-attempt-{attempt_index}"
                try:
                    params = SamplingParams(
                        n=1,
                        temperature=common.TEMPERATURE,
                        top_p=common.TOP_P,
                        top_k=common.TOP_K,
                        repetition_penalty=common.REPETITION_PENALTY,
                        max_tokens=common.MAX_NEW_TOKENS,
                        seed=int(case["row_seed"]),
                        detokenize=True,
                        skip_special_tokens=False,
                        spaces_between_special_tokens=True,
                    )
                    final = None
                    async for output in engine.generate(
                        {"prompt_token_ids": list(case["prompt_token_ids"])},
                        params,
                        request_id,
                        lora_request=request,
                    ):
                        final = output
                    _require(
                        final is not None, f"vLLM returned no output: {request_id}"
                    )
                    _require(
                        len(getattr(final, "outputs", []) or []) == 1,
                        f"vLLM n drift: {request_id}",
                    )
                    _require(
                        list(getattr(final, "prompt_token_ids", []) or [])
                        == list(case["prompt_token_ids"]),
                        f"vLLM prompt-token drift: {request_id}",
                    )
                    completion = final.outputs[0]
                    generated_text = getattr(completion, "text", None)
                    _require(
                        isinstance(generated_text, str),
                        f"vLLM returned non-text: {request_id}",
                    )
                    row = _build_result(
                        case=case,
                        generated_text=generated_text,
                        generated_token_ids=list(
                            getattr(completion, "token_ids", []) or []
                        ),
                        raw_finish_reason=getattr(completion, "finish_reason", None),
                        raw_stop_reason=getattr(completion, "stop_reason", None),
                        attempt_index=attempt_index,
                        manifest_sha=manifest_sha,
                        tokenizer=tokenizer,
                        eos_token_ids=eos_token_ids,
                    )
                except Exception as exc:  # transport/runtime retry boundary
                    last_error = exc
                    await on_failure(
                        {
                            "schema_version": ATTEMPT_SCHEMA,
                            "evaluation_manifest_sha256": manifest_sha,
                            "generation_key": key,
                            "model_id": model_id,
                            "shard_id": case["shard_id"],
                            "row_seed": case["row_seed"],
                            "attempt_index": attempt_index,
                            "request_id": request_id,
                            "status": "transport_failure",
                            "error_type": type(exc).__name__,
                            "error_message_sha256": common.sha256_text(str(exc)),
                            "route": route,
                            "failed_at_utc": _utc_now(),
                        }
                    )
                    continue
                await on_result(row)
                return
            assert last_error is not None
            raise PaperChk2TadleGenerationError(
                f"transport retry budget exhausted for {key}: {type(last_error).__name__}"
            ) from last_error

        started = time.perf_counter()
        for start in range(0, len(cases), 64):
            tasks = [
                asyncio.create_task(consume(case)) for case in cases[start : start + 64]
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        per_model[model_id] = {
            "new_rows": len(cases),
            "generation_wall_seconds": time.perf_counter() - started,
            "lora_request": route["lora_request"],
        }

    try:
        # This ordering is a frozen part of the experiment contract.
        for model_id in model_order:
            await generate_model(model_id)
    finally:
        if hasattr(engine, "shutdown"):
            engine.shutdown()
        elif hasattr(engine, "shutdown_background_loop"):
            engine.shutdown_background_loop()
    return {
        "engine_group_id": CHK1_CP50_GROUP_ID,
        "invocation_id": invocation_id,
        "invocation_index": invocation_index,
        "model_order": list(model_order),
        "base_model_path": str(inputs.model_paths["chk1"]),
        "adapter_path": route_cp50["adapter_path"],
        "enable_lora": True,
        "max_loras": 1,
        "model_load_count": 1,
        "model_load_wall_seconds": load_seconds,
        "generation_wall_seconds": time.perf_counter() - group_started,
        "per_model": per_model,
    }


def _mode_root(output_root: Path, smoke: bool) -> Path:
    return output_root.expanduser().resolve() / ("smoke" if smoke else "generation")


def _model_root(output_root: Path, model_id: str, smoke: bool) -> Path:
    return _mode_root(output_root, smoke) / model_id


def _engine_group_root(output_root: Path, smoke: bool, shard_id: int) -> Path:
    return (
        _mode_root(output_root, smoke)
        / "engine-groups"
        / CHK1_CP50_GROUP_ID
        / f"shard-{shard_id}"
    )


def _smoke_authorization(output_root: Path, manifest_sha: str) -> Mapping[str, Any]:
    path = _mode_root(output_root, True) / "validation.json"
    preparation_manifest = (
        output_root.expanduser().resolve() / "preparation" / "evaluation_manifest.json"
    )
    replayed = validate_all_models(
        manifest_path=preparation_manifest,
        output_root=output_root,
        smoke=True,
    )
    value = _read_json(path)
    _require(value == replayed, "smoke validation receipt differs from deep replay")
    _require(
        value.get("schema_version") == ALL_VALIDATION_SCHEMA,
        "smoke validation schema drift",
    )
    _require(
        value.get("status") == "complete" and value.get("mode") == "smoke",
        "smoke is not complete",
    )
    _require(
        value.get("evaluation_manifest_sha256") == manifest_sha,
        "smoke manifest binding drift",
    )
    gates = value.get("gates")
    _require(isinstance(gates, dict), "smoke authorization gates are missing")
    _require(
        set(gates) == SMOKE_AUTHORIZATION_GATES,
        "smoke authorization gate inventory drift",
    )
    _require(
        all(item is True for item in gates.values()), "smoke authorization gate failed"
    )
    return _record(path)


def _validate_completed_shard_manifest(
    *,
    path: Path,
    inputs: GenerationInputs,
    model_id: str,
    shard_id: int,
    smoke: bool,
    cases: Sequence[Mapping[str, Any]],
    rows: Mapping[str, Mapping[str, Any]],
    failures: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Deep-replay a sealed worker manifest and every file it binds."""

    shard_root = path.parent
    value = _read_json(path)
    _require(value.get("schema_version") == SHARD_SCHEMA, "shard manifest schema drift")
    _require(value.get("status") == "complete", "shard manifest is not complete")
    _require(
        value.get("mode") == ("smoke" if smoke else "formal"),
        "shard manifest mode drift",
    )
    _require(
        value.get("evaluation_manifest_sha256") == inputs.manifest_sha256,
        "shard manifest preparation binding drift",
    )
    _require(value.get("model_id") == model_id, "shard manifest model drift")
    _require(value.get("shard_id") == shard_id, "shard manifest shard drift")
    _require(
        int(value.get("expected_rows", -1)) == len(cases),
        "shard manifest expected-row drift",
    )
    _require(
        int(value.get("completed_rows", -1)) == len(rows),
        "shard manifest completed-row drift",
    )
    if model_id in {"chk1", "paper_chk2_cp50"}:
        _require(
            value.get("engine_group_id") == CHK1_CP50_GROUP_ID,
            "chk1/cp50 shard was not produced by the required shared engine group",
        )
        mode_root = path.parent.parent.parent
        group_state_path = (
            mode_root
            / "engine-groups"
            / CHK1_CP50_GROUP_ID
            / f"shard-{shard_id}"
            / "state.json"
        )
        _require_bound_record(
            value.get("engine_group_state"),
            group_state_path,
            label="shared engine-group state",
        )
        group_state = _read_json(group_state_path)
        invocation_wal_path = group_state_path.parent / "invocations.wal.jsonl"
        invocations = load_engine_invocation_ledger(
            invocation_wal_path,
            manifest_sha=inputs.manifest_sha256,
            shard_id=shard_id,
        )
        _require_bound_record(
            value.get("engine_invocation_ledger"),
            invocation_wal_path,
            rows=len(invocations),
            label="engine invocation ledger",
        )
        observed_arm_invocations = sorted(
            {str(row.get("engine_invocation_id")) for row in rows.values()}
        )
        _require(
            value.get("arm_engine_invocation_ids") == observed_arm_invocations,
            "arm invocation binding drift",
        )
        _require(
            value.get("cumulative_model_load_count") == len(invocations),
            "cumulative model-load count drift",
        )
        _require(
            group_state.get("schema_version") == ENGINE_GROUP_SCHEMA,
            "engine-group state schema drift",
        )
        _require(
            group_state.get("status") == "complete",
            "engine-group state is not complete",
        )
        _require(
            group_state.get("engine_group_id") == CHK1_CP50_GROUP_ID,
            "engine-group state ID drift",
        )
        _require(
            group_state.get("model_order") == ["chk1", "paper_chk2_cp50"],
            "engine-group state order drift",
        )
        _require(
            group_state.get("cumulative_model_load_count") == len(invocations),
            "engine-group state cumulative-load drift",
        )
    _require_bound_record(
        value.get("evaluation_manifest"),
        inputs.manifest_path,
        label="shard evaluation manifest",
    )
    _require_bound_record(
        value.get("worker_launch"),
        shard_root / "launch.json",
        label="worker launch",
    )
    worker_launch = _read_json(shard_root / "launch.json")
    _require(
        _implementation_binding_is_accepted(
            (worker_launch.get("contract") or {}).get("implementation")
        ),
        "worker implementation binding drift",
    )
    _require_bound_record(
        value.get("worker_state"),
        shard_root / "state.json",
        label="worker state",
    )
    _require_bound_record(
        value.get("generation_wal"),
        shard_root / "generations.wal.jsonl",
        rows=len(rows),
        label="generation WAL",
    )
    _require_bound_record(
        value.get("attempt_wal"),
        shard_root / "attempts.wal.jsonl",
        rows=sum(len(items) for items in failures.values()),
        label="attempt WAL",
    )
    canonical_path = shard_root / "generations.canonical.jsonl"
    _require_bound_record(
        value.get("canonical"),
        canonical_path,
        rows=len(rows),
        label="canonical generation rows",
    )
    expected_canonical = [dict(rows[str(case["generation_key"])]) for case in cases]
    _require(
        _read_jsonl(canonical_path) == expected_canonical,
        "canonical generation content/order drift",
    )
    state = _read_json(shard_root / "state.json")
    _require(state.get("schema_version") == SHARD_SCHEMA, "worker state schema drift")
    _require(state.get("status") == "complete", "worker state is not complete")
    _require(state.get("model_id") == model_id, "worker state model drift")
    _require(state.get("shard_id") == shard_id, "worker state shard drift")
    _require(
        int(state.get("expected_rows", -1)) == len(cases),
        "worker state expected-row drift",
    )
    _require(
        int(state.get("completed_rows", -1)) == len(rows),
        "worker state completed-row drift",
    )
    return value


def run_worker(
    *,
    manifest_path: Path,
    output_root: Path,
    shard_id: int,
    model_id: str,
    smoke: bool = False,
    resume: bool = False,
) -> Mapping[str, Any]:
    """Run one model arm on one independent single-GPU DP1 shard."""

    _require(shard_id in SHARD_IDS, "shard ID must be 0 or 1")
    _require(model_id in common.MODEL_IDS, f"invalid model ID: {model_id}")
    _require(
        model_id == "chk0",
        "chk1 and paper_chk2_cp50 must use the shared chk1_paper_chk2_cp50 worker",
    )
    _validate_visible_gpu(shard_id)
    inputs = load_generation_inputs(manifest_path, output_root)
    tokenizer, eos_ids, _pad_id = _load_tokenizer(inputs.tokenizer_path)
    all_cases = build_case_matrix(inputs, model_id=model_id, smoke=smoke)
    cases = [case for case in all_cases if int(case["shard_id"]) == shard_id]
    case_index = _case_index(cases)
    authorization = (
        None if smoke else _smoke_authorization(output_root, inputs.manifest_sha256)
    )
    shard_root = _model_root(output_root, model_id, smoke) / f"shard-{shard_id}"
    launch_path = shard_root / "launch.json"
    generation_wal = shard_root / "generations.wal.jsonl"
    attempt_wal = shard_root / "attempts.wal.jsonl"
    canonical_path = shard_root / "generations.canonical.jsonl"
    state_path = shard_root / "state.json"
    shard_manifest_path = shard_root / "manifest.json"
    expected_launch = {
        "schema_version": "paper-chk2-tadle-lm-k10-worker-launch-v1",
        "evaluation_manifest": {
            "path": str(inputs.manifest_path),
            "sha256": inputs.manifest_sha256,
        },
        "mode": "smoke" if smoke else "formal",
        "model_id": model_id,
        "shard_id": shard_id,
        "physical_gpu_index": shard_id,
        "route": _route_for_model(model_id),
        "implementation": _implementation_bindings(),
        "generation_contract": common.generation_contract(),
        "replicate_seeds": list(
            common.REPLICATE_SEEDS[:SMOKE_REPLICATES]
            if smoke
            else common.REPLICATE_SEEDS
        ),
        "formal_smoke_authorization": authorization,
    }
    prior_shard_manifest: Mapping[str, Any] | None = None
    if shard_root.exists():
        _require(resume, f"shard output exists; pass --resume: {shard_root}")
        _require(
            shard_root.is_dir() and not shard_root.is_symlink(),
            "unsafe shard output root",
        )
        _require(
            _resume_launch_contract_matches(
                _read_json(launch_path).get("contract"), expected_launch
            ),
            "resume launch contract drift",
        )
        if shard_manifest_path.is_file():
            prior_shard_manifest = _read_json(shard_manifest_path)
    else:
        shard_root.mkdir(parents=True, exist_ok=False)
        _write_new_json(
            launch_path,
            {
                "contract": expected_launch,
                "runtime": {
                    "python": sys.executable,
                    "python_version": platform.python_version(),
                    "vllm": _package_version("vllm"),
                    "transformers": _package_version("transformers"),
                    "tokenizers": _package_version("tokenizers"),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                },
            },
        )
    rows = load_generation_wal(
        generation_wal,
        cases_by_key=case_index,
        manifest_sha=inputs.manifest_sha256,
        tokenizer=tokenizer,
        eos_token_ids=eos_ids,
    )
    failures = load_attempt_ledger(
        attempt_wal, cases_by_key=case_index, manifest_sha=inputs.manifest_sha256
    )
    for key, row in rows.items():
        _require(
            len(failures.get(key, ())) == int(row["attempt_index"]),
            f"success/failure attempt history drift: {key}",
        )
    for key, items in failures.items():
        _require(
            key not in rows or len(items) < MAX_TRANSPORT_ATTEMPTS,
            f"failure ledger exceeds successful attempt: {key}",
        )
    pending = [case for case in cases if str(case["generation_key"]) not in rows]
    for case in pending:
        key = str(case["generation_key"])
        _retry_start_index(key, len(failures.get(key, ())))

    if not pending:
        if prior_shard_manifest is not None:
            sealed = _validate_completed_shard_manifest(
                path=shard_manifest_path,
                inputs=inputs,
                model_id=model_id,
                shard_id=shard_id,
                smoke=smoke,
                cases=cases,
                rows=rows,
                failures=failures,
            )
            gc.collect()
            return sealed

    if pending:
        state: dict[str, Any] = {
            "schema_version": SHARD_SCHEMA,
            "status": "running",
            "model_id": model_id,
            "shard_id": shard_id,
            "expected_rows": len(cases),
            "resumed_rows": len(rows),
            "pending_rows": len(pending),
            "completed_rows": len(rows),
            "transport_failures": sum(len(value) for value in failures.values()),
        }
        _atomic_json(state_path, state)
        with _worker_lock(shard_id) as lock:
            state["worker_lock"] = dict(lock)
            snapshot = gpu_snapshot(shard_id)
            with (
                generation_wal.open("a", encoding="utf-8") as generation_handle,
                attempt_wal.open("a", encoding="utf-8") as attempt_handle,
            ):

                async def on_failure(event: Mapping[str, Any]) -> None:
                    key = str(event["generation_key"])
                    _append_fsync(attempt_handle, event)
                    failures.setdefault(key, []).append(dict(event))
                    state["transport_failures"] = sum(
                        len(value) for value in failures.values()
                    )
                    _atomic_json(state_path, state)

                async def on_result(row: Mapping[str, Any]) -> None:
                    key = str(row["generation_key"])
                    _require(key not in rows, f"duplicate newly generated key: {key}")
                    _append_fsync(generation_handle, row)
                    rows[key] = dict(row)
                    state["completed_rows"] = len(rows)
                    if len(rows) % 16 == 0 or len(rows) == len(cases):
                        _atomic_json(state_path, state)

                report = asyncio.run(
                    _run_engine(
                        model_id=model_id,
                        inputs=inputs,
                        cases=pending,
                        manifest_sha=inputs.manifest_sha256,
                        tokenizer=tokenizer,
                        eos_token_ids=eos_ids,
                        gpu_memory_utilization=float(
                            snapshot["vllm_gpu_memory_utilization"]
                        ),
                        initial_failure_counts={
                            key: len(value) for key, value in failures.items()
                        },
                        on_failure=on_failure,
                        on_result=on_result,
                    )
                )
    else:
        state = {
            "schema_version": SHARD_SCHEMA,
            "status": "complete",
            "model_id": model_id,
            "shard_id": shard_id,
            "expected_rows": len(cases),
            "resumed_rows": len(rows),
            "completed_rows": len(rows),
            "transport_failures": sum(len(value) for value in failures.values()),
        }
        report = None
        snapshot = None

    _require(len(rows) == len(cases), "worker ended without shard closure")
    ordered_rows = [rows[str(case["generation_key"])] for case in cases]
    canonical = _write_or_verify_jsonl(canonical_path, ordered_rows)
    generation_wall_seconds = (
        float(report["generation_wall_seconds"])
        if report is not None
        else float((prior_shard_manifest or {}).get("generation_wall_seconds", 0.0))
    )
    model_load_wall_seconds = (
        float(report["model_load_wall_seconds"])
        if report is not None
        else float((prior_shard_manifest or {}).get("model_load_wall_seconds", 0.0))
    )
    newly_generated_rows = (
        int(report["new_rows"])
        if report is not None
        else int((prior_shard_manifest or {}).get("new_rows", 0))
    )
    result = {
        "schema_version": SHARD_SCHEMA,
        "status": "complete",
        "mode": "smoke" if smoke else "formal",
        "evaluation_manifest_sha256": inputs.manifest_sha256,
        "model_id": model_id,
        "shard_id": shard_id,
        "expected_rows": len(cases),
        "completed_rows": len(rows),
        "resumed_rows": state["resumed_rows"],
        "new_rows": newly_generated_rows,
        "new_rows_this_invocation": len(rows) - int(state["resumed_rows"]),
        "generation_wall_seconds": generation_wall_seconds,
        "model_load_wall_seconds": model_load_wall_seconds,
        "transport_failure_attempts": sum(len(value) for value in failures.values()),
        "resume_noop": not pending,
        "receipt_recovered_from_closed_wal": not pending,
        "gpu_before": snapshot,
        "engine": report,
        "evaluation_manifest": _record(inputs.manifest_path),
        "worker_launch": _record(launch_path),
        "generation_wal": _record(generation_wal, rows=len(rows)),
        "attempt_wal": _record(
            attempt_wal, rows=sum(len(value) for value in failures.values())
        ),
        "canonical": canonical,
    }
    state.update({"status": "complete", "completed_rows": len(rows)})
    _atomic_json(state_path, state)
    result["worker_state"] = _record(state_path)
    _require(
        not shard_manifest_path.exists(), "refusing to overwrite sealed shard manifest"
    )
    _write_new_json(shard_manifest_path, result)
    gc.collect()
    return result


def _validate_engine_group_manifest(
    *,
    path: Path,
    inputs: GenerationInputs,
    shard_id: int,
    smoke: bool,
    model_manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    value = _read_json(path)
    _require(
        value.get("schema_version") == ENGINE_GROUP_SCHEMA, "engine-group schema drift"
    )
    _require(value.get("status") == "complete", "engine-group is not complete")
    _require(
        value.get("mode") == ("smoke" if smoke else "formal"), "engine-group mode drift"
    )
    _require(
        value.get("engine_group_id") == CHK1_CP50_GROUP_ID, "engine-group ID drift"
    )
    _require(
        value.get("model_order") == ["chk1", "paper_chk2_cp50"],
        "engine-group order drift",
    )
    _require(value.get("shard_id") == shard_id, "engine-group shard drift")
    _require(
        value.get("evaluation_manifest_sha256") == inputs.manifest_sha256,
        "engine-group preparation binding drift",
    )
    _require_bound_record(
        value.get("evaluation_manifest"),
        inputs.manifest_path,
        label="engine-group evaluation manifest",
    )
    _require_bound_record(
        value.get("group_launch"),
        path.parent / "launch.json",
        label="engine-group launch",
    )
    group_launch = _read_json(path.parent / "launch.json")
    _require(
        _implementation_binding_is_accepted(
            (group_launch.get("contract") or {}).get("implementation")
        ),
        "engine-group implementation binding drift",
    )
    _require_bound_record(
        value.get("group_state"),
        path.parent / "state.json",
        label="engine-group state",
    )
    group_state = _read_json(path.parent / "state.json")
    invocation_wal_path = path.parent / "invocations.wal.jsonl"
    invocations = load_engine_invocation_ledger(
        invocation_wal_path,
        manifest_sha=inputs.manifest_sha256,
        shard_id=shard_id,
    )
    _require_bound_record(
        value.get("engine_invocation_ledger"),
        invocation_wal_path,
        rows=len(invocations),
        label="engine-group invocation ledger",
    )
    _require(
        value.get("cumulative_model_load_count") == len(invocations),
        "engine-group cumulative-load count drift",
    )
    _require(
        group_state.get("cumulative_model_load_count") == len(invocations),
        "engine-group state cumulative-load count drift",
    )
    declared_models = value.get("models")
    _require(isinstance(declared_models, dict), "engine-group model bindings missing")
    for model_id in ("chk1", "paper_chk2_cp50"):
        model_manifest_path = (
            path.parent.parent.parent.parent
            / model_id
            / f"shard-{shard_id}"
            / "manifest.json"
        )
        _require_bound_record(
            declared_models.get(model_id),
            model_manifest_path,
            label=f"engine-group {model_id} worker manifest",
        )
        _require(
            dict(model_manifests[model_id]) == _read_json(model_manifest_path),
            f"engine-group {model_id} manifest replay drift",
        )
    arm_invocations = {
        model_id: model_manifests[model_id].get("arm_engine_invocation_ids")
        for model_id in ("chk1", "paper_chk2_cp50")
    }
    expected_single_shared = (
        len(invocations) == 1
        and arm_invocations["chk1"] == arm_invocations["paper_chk2_cp50"]
    )
    _require(
        value.get("uninterrupted_single_shared_load") is expected_single_shared,
        "engine-group uninterrupted-load verdict drift",
    )
    _require(
        group_state.get("arm_invocation_ids") == arm_invocations
        and group_state.get("uninterrupted_single_shared_load")
        is expected_single_shared,
        "engine-group state arm-invocation provenance drift",
    )
    return value


def _grouped_model_manifest_value(
    *,
    inputs: GenerationInputs,
    context: Mapping[str, Any],
    group_state: Mapping[str, Any],
    group_state_path: Path,
    invocation_wal_path: Path,
    report: Mapping[str, Any],
    snapshot: Mapping[str, Any] | None,
    model_id: str,
    shard_id: int,
    smoke: bool,
    recovered_from_closed_wal: bool,
) -> dict[str, Any]:
    per_model_report = report["per_model"][model_id]
    return {
        "schema_version": SHARD_SCHEMA,
        "status": "complete",
        "mode": "smoke" if smoke else "formal",
        "evaluation_manifest_sha256": inputs.manifest_sha256,
        "model_id": model_id,
        "shard_id": shard_id,
        "expected_rows": len(context["cases"]),
        "completed_rows": len(context["rows"]),
        "resumed_rows": group_state["resumed_rows_by_model"][model_id],
        "new_rows": int(per_model_report["new_rows"]),
        "new_rows_this_invocation": (
            0 if recovered_from_closed_wal else int(per_model_report["new_rows"])
        ),
        "generation_wall_seconds": float(per_model_report["generation_wall_seconds"]),
        "model_load_wall_seconds": float(report["model_load_wall_seconds"]),
        "model_load_accounting": "one_shared_load_do_not_sum_across_group_models",
        "transport_failure_attempts": sum(
            len(items) for items in context["failures"].values()
        ),
        "resume_noop": recovered_from_closed_wal,
        "receipt_recovered_from_closed_wal": recovered_from_closed_wal,
        "gpu_before": snapshot,
        "engine": {
            "engine_group_id": CHK1_CP50_GROUP_ID,
            "latest_engine_invocation_model_load_count": 1,
            "model_order": report["model_order"],
            "per_model": per_model_report,
        },
        "engine_group_id": CHK1_CP50_GROUP_ID,
        "shared_base_model_load": group_state["uninterrupted_single_shared_load"],
        "cumulative_model_load_count": group_state["cumulative_model_load_count"],
        "arm_engine_invocation_ids": group_state["arm_invocation_ids"][model_id],
        "engine_invocation_ledger": _record(
            invocation_wal_path,
            rows=group_state["cumulative_model_load_count"],
        ),
        "engine_group_state": _record(group_state_path),
        "evaluation_manifest": _record(inputs.manifest_path),
        "worker_launch": _record(context["launch_path"]),
        "worker_state": _record(context["state_path"]),
        "generation_wal": _record(context["generation_wal"], rows=len(context["rows"])),
        "attempt_wal": _record(
            context["attempt_wal"],
            rows=sum(len(items) for items in context["failures"].values()),
        ),
        "canonical": _record(context["canonical_path"], rows=len(context["rows"])),
    }


def _seal_chk1_cp50_group_receipts(
    *,
    inputs: GenerationInputs,
    contexts: Mapping[str, Mapping[str, Any]],
    group_launch_path: Path,
    group_state_path: Path,
    group_manifest_path: Path,
    invocation_wal_path: Path,
    invocations: Sequence[Mapping[str, Any]],
    group_state: dict[str, Any],
    report: Mapping[str, Any],
    snapshot: Mapping[str, Any] | None,
    shard_id: int,
    smoke: bool,
    recovered_from_closed_wal: bool,
) -> dict[str, Any]:
    model_ids = ("chk1", "paper_chk2_cp50")
    _require(
        bool(invocations), "cannot seal grouped outputs without an engine invocation"
    )
    invocation_ids = {str(row["invocation_id"]) for row in invocations}
    arm_invocation_ids = {
        model_id: sorted(
            {str(row.get("engine_invocation_id")) for row in context["rows"].values()}
        )
        for model_id, context in contexts.items()
    }
    _require(
        all(ids and set(ids) <= invocation_ids for ids in arm_invocation_ids.values()),
        "generation rows are not bound to the invocation ledger",
    )
    uninterrupted_single_shared_load = (
        len(invocations) == 1
        and arm_invocation_ids["chk1"] == arm_invocation_ids["paper_chk2_cp50"]
    )
    for model_id, context in contexts.items():
        _require(
            len(context["rows"]) == len(context["cases"]),
            f"{model_id} grouped worker ended without closure",
        )
        _write_or_verify_jsonl(
            context["canonical_path"],
            [context["rows"][str(case["generation_key"])] for case in context["cases"]],
        )
        state = (
            _read_json(context["state_path"])
            if context["state_path"].is_file()
            else {
                "schema_version": SHARD_SCHEMA,
                "model_id": model_id,
                "shard_id": shard_id,
                "expected_rows": len(context["cases"]),
                "resumed_rows": len(context["rows"]),
            }
        )
        state.update(
            {
                "status": "complete",
                "completed_rows": len(context["rows"]),
                "transport_failures": sum(
                    len(items) for items in context["failures"].values()
                ),
            }
        )
        _atomic_json(context["state_path"], state)
    group_state.update(
        {
            "status": "complete",
            "completed_rows_by_model": {
                model_id: len(context["rows"]) for model_id, context in contexts.items()
            },
            "engine": dict(report),
            "gpu_before": snapshot,
            "receipt_recovered_from_closed_wal": recovered_from_closed_wal,
            "cumulative_model_load_count": len(invocations),
            "arm_invocation_ids": arm_invocation_ids,
            "uninterrupted_single_shared_load": uninterrupted_single_shared_load,
        }
    )
    _atomic_json(group_state_path, group_state)

    model_manifests: dict[str, Mapping[str, Any]] = {}
    for model_id, context in contexts.items():
        value = _grouped_model_manifest_value(
            inputs=inputs,
            context=context,
            group_state=group_state,
            group_state_path=group_state_path,
            invocation_wal_path=invocation_wal_path,
            report=report,
            snapshot=snapshot,
            model_id=model_id,
            shard_id=shard_id,
            smoke=smoke,
            recovered_from_closed_wal=recovered_from_closed_wal,
        )
        _require(
            not context["manifest_path"].exists(),
            "refusing to overwrite grouped model manifest",
        )
        _write_new_json(context["manifest_path"], value)
        model_manifests[model_id] = value

    group_manifest = {
        "schema_version": ENGINE_GROUP_SCHEMA,
        "status": "complete",
        "mode": "smoke" if smoke else "formal",
        "evaluation_manifest_sha256": inputs.manifest_sha256,
        "engine_group_id": CHK1_CP50_GROUP_ID,
        "model_order": list(model_ids),
        "cumulative_model_load_count": len(invocations),
        "uninterrupted_single_shared_load": uninterrupted_single_shared_load,
        "arm_invocation_ids": arm_invocation_ids,
        "shard_id": shard_id,
        "expected_rows": sum(len(context["cases"]) for context in contexts.values()),
        "completed_rows": sum(len(context["rows"]) for context in contexts.values()),
        "new_rows": sum(
            int(report["per_model"][model_id]["new_rows"]) for model_id in model_ids
        ),
        "generation_wall_seconds": float(report["generation_wall_seconds"]),
        "model_load_wall_seconds": float(report["model_load_wall_seconds"]),
        "receipt_recovered_from_closed_wal": recovered_from_closed_wal,
        "evaluation_manifest": _record(inputs.manifest_path),
        "group_launch": _record(group_launch_path),
        "group_state": _record(group_state_path),
        "engine_invocation_ledger": _record(invocation_wal_path, rows=len(invocations)),
        "models": {
            model_id: _record(contexts[model_id]["manifest_path"])
            for model_id in model_ids
        },
    }
    _require(
        not group_manifest_path.exists(),
        "refusing to overwrite sealed engine-group manifest",
    )
    _write_new_json(group_manifest_path, group_manifest)
    return group_manifest


def run_chk1_cp50_group_worker(
    *,
    manifest_path: Path,
    output_root: Path,
    shard_id: int,
    smoke: bool = False,
    resume: bool = False,
) -> Mapping[str, Any]:
    """Generate chk1 then cp50 with one shared LoRA-capable chk1 engine."""

    _require(shard_id in SHARD_IDS, "shard ID must be 0 or 1")
    _validate_visible_gpu(shard_id)
    inputs = load_generation_inputs(manifest_path, output_root)
    tokenizer, eos_ids, _pad_id = _load_tokenizer(inputs.tokenizer_path)
    authorization = (
        None if smoke else _smoke_authorization(output_root, inputs.manifest_sha256)
    )
    model_ids = ("chk1", "paper_chk2_cp50")
    group_root = _engine_group_root(output_root, smoke, shard_id)
    group_launch_path = group_root / "launch.json"
    group_state_path = group_root / "state.json"
    group_manifest_path = group_root / "manifest.json"
    invocation_wal_path = group_root / "invocations.wal.jsonl"
    group_launch_contract = {
        "schema_version": "paper-chk2-tadle-lm-k10-engine-group-launch-v1",
        "evaluation_manifest": {
            "path": str(inputs.manifest_path),
            "sha256": inputs.manifest_sha256,
        },
        "mode": "smoke" if smoke else "formal",
        "engine_group_id": CHK1_CP50_GROUP_ID,
        "model_order": list(model_ids),
        "shard_id": shard_id,
        "physical_gpu_index": shard_id,
        "base_model_path": str(inputs.model_paths["chk1"]),
        "adapter_path": str(inputs.model_paths["paper_chk2_cp50_adapter"]),
        "enable_lora": True,
        "max_loras": 1,
        "model_loads_per_engine_invocation": 1,
        "engine_invocation_ledger": {
            "path": str(invocation_wal_path.resolve()),
            "schema_version": ENGINE_INVOCATION_SCHEMA,
            "append_only": True,
        },
        "implementation": _implementation_bindings(),
        "generation_contract": common.generation_contract(),
        "replicate_seeds": list(
            common.REPLICATE_SEEDS[:SMOKE_REPLICATES]
            if smoke
            else common.REPLICATE_SEEDS
        ),
        "formal_smoke_authorization": authorization,
    }
    if group_root.exists():
        _require(resume, f"engine-group output exists; pass --resume: {group_root}")
        _require(
            group_root.is_dir() and not group_root.is_symlink(),
            "unsafe engine-group root",
        )
        _require(
            _resume_launch_contract_matches(
                _read_json(group_launch_path).get("contract"), group_launch_contract
            ),
            "engine-group resume contract drift",
        )
    else:
        group_root.mkdir(parents=True, exist_ok=False)
        _write_new_json(
            group_launch_path,
            {
                "contract": group_launch_contract,
                "runtime": {
                    "python": sys.executable,
                    "python_version": platform.python_version(),
                    "vllm": _package_version("vllm"),
                    "transformers": _package_version("transformers"),
                    "tokenizers": _package_version("tokenizers"),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                },
            },
        )
    invocations = load_engine_invocation_ledger(
        invocation_wal_path,
        manifest_sha=inputs.manifest_sha256,
        shard_id=shard_id,
    )

    contexts: dict[str, dict[str, Any]] = {}
    any_existing_model_manifest = False
    for model_id in model_ids:
        cases = [
            case
            for case in build_case_matrix(inputs, model_id=model_id, smoke=smoke)
            if int(case["shard_id"]) == shard_id
        ]
        index = _case_index(cases)
        root = _model_root(output_root, model_id, smoke) / f"shard-{shard_id}"
        launch_path = root / "launch.json"
        generation_wal = root / "generations.wal.jsonl"
        attempt_wal = root / "attempts.wal.jsonl"
        canonical_path = root / "generations.canonical.jsonl"
        state_path = root / "state.json"
        manifest_out = root / "manifest.json"
        expected_launch = {
            "schema_version": "paper-chk2-tadle-lm-k10-worker-launch-v1",
            "evaluation_manifest": {
                "path": str(inputs.manifest_path),
                "sha256": inputs.manifest_sha256,
            },
            "mode": "smoke" if smoke else "formal",
            "model_id": model_id,
            "shard_id": shard_id,
            "physical_gpu_index": shard_id,
            "route": _route_for_model(model_id),
            "implementation": _implementation_bindings(),
            "generation_contract": common.generation_contract(),
            "replicate_seeds": list(
                common.REPLICATE_SEEDS[:SMOKE_REPLICATES]
                if smoke
                else common.REPLICATE_SEEDS
            ),
            "formal_smoke_authorization": authorization,
            "engine_group": {
                "id": CHK1_CP50_GROUP_ID,
                "model_order": list(model_ids),
                "shared_base_model_load": True,
                "group_launch": _record(group_launch_path),
            },
        }
        if root.exists():
            _require(resume, f"model shard output exists; pass --resume: {root}")
            _require(root.is_dir() and not root.is_symlink(), "unsafe model shard root")
            _require(
                _resume_launch_contract_matches(
                    _read_json(launch_path).get("contract"), expected_launch
                ),
                "grouped model resume contract drift",
            )
        else:
            root.mkdir(parents=True, exist_ok=False)
            _write_new_json(
                launch_path,
                {
                    "contract": expected_launch,
                    "runtime": {
                        "python": sys.executable,
                        "python_version": platform.python_version(),
                        "vllm": _package_version("vllm"),
                        "transformers": _package_version("transformers"),
                        "tokenizers": _package_version("tokenizers"),
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    },
                },
            )
        rows = load_generation_wal(
            generation_wal,
            cases_by_key=index,
            manifest_sha=inputs.manifest_sha256,
            tokenizer=tokenizer,
            eos_token_ids=eos_ids,
        )
        failures = load_attempt_ledger(
            attempt_wal,
            cases_by_key=index,
            manifest_sha=inputs.manifest_sha256,
        )
        for key, row in rows.items():
            _require(
                len(failures.get(key, ())) == int(row["attempt_index"]),
                f"success/failure attempt history drift: {key}",
            )
        pending = [case for case in cases if str(case["generation_key"]) not in rows]
        for case in pending:
            key = str(case["generation_key"])
            _retry_start_index(key, len(failures.get(key, ())))
        any_existing_model_manifest = (
            any_existing_model_manifest or manifest_out.exists()
        )
        contexts[model_id] = {
            "cases": cases,
            "index": index,
            "root": root,
            "launch_path": launch_path,
            "generation_wal": generation_wal,
            "attempt_wal": attempt_wal,
            "canonical_path": canonical_path,
            "state_path": state_path,
            "manifest_path": manifest_out,
            "rows": rows,
            "failures": failures,
            "pending": pending,
        }

    pending_total = sum(len(context["pending"]) for context in contexts.values())
    if pending_total == 0:
        all_model_manifests = all(
            context["manifest_path"].is_file() for context in contexts.values()
        )
        if all_model_manifests and group_manifest_path.is_file():
            model_manifests = {
                model_id: _validate_completed_shard_manifest(
                    path=context["manifest_path"],
                    inputs=inputs,
                    model_id=model_id,
                    shard_id=shard_id,
                    smoke=smoke,
                    cases=context["cases"],
                    rows=context["rows"],
                    failures=context["failures"],
                )
                for model_id, context in contexts.items()
            }
            sealed = _validate_engine_group_manifest(
                path=group_manifest_path,
                inputs=inputs,
                shard_id=shard_id,
                smoke=smoke,
                model_manifests=model_manifests,
            )
            gc.collect()
            return sealed
        if any_existing_model_manifest:
            state = _read_json(group_state_path)
            engine_report = state.get("engine") or {}
            _require(
                engine_report.get("model_load_count") == 1,
                "group receipt recovery lacks one-load proof",
            )
            model_manifests: dict[str, Mapping[str, Any]] = {}
            for model_id, context in contexts.items():
                if context["manifest_path"].is_file():
                    model_manifests[model_id] = _validate_completed_shard_manifest(
                        path=context["manifest_path"],
                        inputs=inputs,
                        model_id=model_id,
                        shard_id=shard_id,
                        smoke=smoke,
                        cases=context["cases"],
                        rows=context["rows"],
                        failures=context["failures"],
                    )
                    continue
                _require(
                    context["canonical_path"].is_file()
                    and context["state_path"].is_file(),
                    f"missing unsealed {model_id} canonical/state receipt inputs",
                )
                missing_value = _grouped_model_manifest_value(
                    inputs=inputs,
                    context=context,
                    group_state=state,
                    group_state_path=group_state_path,
                    invocation_wal_path=invocation_wal_path,
                    report=engine_report,
                    snapshot=state.get("gpu_before"),
                    model_id=model_id,
                    shard_id=shard_id,
                    smoke=smoke,
                    recovered_from_closed_wal=True,
                )
                _write_new_json(context["manifest_path"], missing_value)
                model_manifests[model_id] = missing_value
            recovered_group_manifest = {
                "schema_version": ENGINE_GROUP_SCHEMA,
                "status": "complete",
                "mode": "smoke" if smoke else "formal",
                "evaluation_manifest_sha256": inputs.manifest_sha256,
                "engine_group_id": CHK1_CP50_GROUP_ID,
                "model_order": list(model_ids),
                "cumulative_model_load_count": len(invocations),
                "uninterrupted_single_shared_load": state[
                    "uninterrupted_single_shared_load"
                ],
                "arm_invocation_ids": state["arm_invocation_ids"],
                "shard_id": shard_id,
                "expected_rows": sum(
                    len(context["cases"]) for context in contexts.values()
                ),
                "completed_rows": sum(
                    len(context["rows"]) for context in contexts.values()
                ),
                "new_rows": sum(
                    int(value["new_rows"]) for value in model_manifests.values()
                ),
                "generation_wall_seconds": float(
                    engine_report["generation_wall_seconds"]
                ),
                "model_load_wall_seconds": float(
                    engine_report["model_load_wall_seconds"]
                ),
                "receipt_recovered_from_closed_wal": True,
                "evaluation_manifest": _record(inputs.manifest_path),
                "group_launch": _record(group_launch_path),
                "group_state": _record(group_state_path),
                "engine_invocation_ledger": _record(
                    invocation_wal_path, rows=len(invocations)
                ),
                "models": {
                    model_id: _record(contexts[model_id]["manifest_path"])
                    for model_id in model_ids
                },
            }
            _write_new_json(group_manifest_path, recovered_group_manifest)
            sealed = _validate_engine_group_manifest(
                path=group_manifest_path,
                inputs=inputs,
                shard_id=shard_id,
                smoke=smoke,
                model_manifests=model_manifests,
            )
            gc.collect()
            return sealed
        _require(
            not any_existing_model_manifest and not group_manifest_path.exists(),
            "partial engine-group receipt publication requires manual audit",
        )
        _require(group_state_path.is_file(), "closed grouped WALs lack recovery state")
        recovered_state = _read_json(group_state_path)
        _require(
            recovered_state.get("schema_version") == ENGINE_GROUP_SCHEMA,
            "recovery state schema drift",
        )
        _require(
            recovered_state.get("engine_group_id") == CHK1_CP50_GROUP_ID,
            "recovery state group drift",
        )
        recovered_state.setdefault(
            "resumed_rows_by_model",
            {model_id: 0 for model_id in model_ids},
        )
        report = recovered_state.get("engine")
        if not isinstance(report, dict):
            report = {
                "engine_group_id": CHK1_CP50_GROUP_ID,
                "model_order": list(model_ids),
                "base_model_path": str(inputs.model_paths["chk1"]),
                "adapter_path": str(inputs.model_paths["paper_chk2_cp50_adapter"]),
                "enable_lora": True,
                "max_loras": 1,
                "model_load_count": 1,
                "model_load_wall_seconds": 0.0,
                "generation_wall_seconds": 0.0,
                "timing_unavailable_after_pre_receipt_crash": True,
                "per_model": {
                    model_id: {
                        "new_rows": len(contexts[model_id]["rows"])
                        - int(
                            recovered_state["resumed_rows_by_model"].get(model_id, 0)
                        ),
                        "generation_wall_seconds": 0.0,
                        "lora_request": _route_for_model(model_id)["lora_request"],
                    }
                    for model_id in model_ids
                },
            }
        sealed = _seal_chk1_cp50_group_receipts(
            inputs=inputs,
            contexts=contexts,
            group_launch_path=group_launch_path,
            group_state_path=group_state_path,
            group_manifest_path=group_manifest_path,
            invocation_wal_path=invocation_wal_path,
            invocations=invocations,
            group_state=recovered_state,
            report=report,
            snapshot=recovered_state.get("gpu_before"),
            shard_id=shard_id,
            smoke=smoke,
            recovered_from_closed_wal=True,
        )
        gc.collect()
        return sealed
    _require(
        not any_existing_model_manifest and not group_manifest_path.exists(),
        "partial engine group cannot mutate an already sealed model/group manifest",
    )

    group_state: dict[str, Any] = {
        "schema_version": ENGINE_GROUP_SCHEMA,
        "status": "running",
        "engine_group_id": CHK1_CP50_GROUP_ID,
        "mode": "smoke" if smoke else "formal",
        "model_order": list(model_ids),
        "shard_id": shard_id,
        "engine_invocation_ledger_path": str(invocation_wal_path.resolve()),
        "cumulative_model_load_count": len(invocations),
        "expected_rows_by_model": {
            model_id: len(context["cases"]) for model_id, context in contexts.items()
        },
        "resumed_rows_by_model": {
            model_id: len(context["rows"]) for model_id, context in contexts.items()
        },
        "completed_rows_by_model": {
            model_id: len(context["rows"]) for model_id, context in contexts.items()
        },
        "transport_failures_by_model": {
            model_id: sum(len(items) for items in context["failures"].values())
            for model_id, context in contexts.items()
        },
    }
    _atomic_json(group_state_path, group_state)
    for model_id, context in contexts.items():
        _atomic_json(
            context["state_path"],
            {
                "schema_version": SHARD_SCHEMA,
                "status": "running",
                "model_id": model_id,
                "shard_id": shard_id,
                "expected_rows": len(context["cases"]),
                "resumed_rows": len(context["rows"]),
                "completed_rows": len(context["rows"]),
                "transport_failures": sum(
                    len(items) for items in context["failures"].values()
                ),
            },
        )

    invocation_index = len(invocations)
    invocation_id = (
        f"{CHK1_CP50_GROUP_ID}-shard-{shard_id}-invocation-{invocation_index:03d}"
    )
    with _worker_lock(shard_id) as lock, contextlib.ExitStack() as stack:
        group_state["worker_lock"] = dict(lock)
        invocation_handle = stack.enter_context(
            invocation_wal_path.open("a", encoding="utf-8")
        )
        generation_handles = {
            model_id: stack.enter_context(
                context["generation_wal"].open("a", encoding="utf-8")
            )
            for model_id, context in contexts.items()
        }
        attempt_handles = {
            model_id: stack.enter_context(
                context["attempt_wal"].open("a", encoding="utf-8")
            )
            for model_id, context in contexts.items()
        }

        async def on_failure(event: Mapping[str, Any]) -> None:
            model_id = str(event["model_id"])
            context = contexts[model_id]
            key = str(event["generation_key"])
            _append_fsync(attempt_handles[model_id], event)
            context["failures"].setdefault(key, []).append(dict(event))
            group_state["transport_failures_by_model"][model_id] = sum(
                len(items) for items in context["failures"].values()
            )
            _atomic_json(group_state_path, group_state)

        async def on_result(row: Mapping[str, Any]) -> None:
            model_id = str(row["model_id"])
            context = contexts[model_id]
            key = str(row["generation_key"])
            _require(
                key not in context["rows"], f"duplicate newly generated key: {key}"
            )
            _append_fsync(generation_handles[model_id], row)
            context["rows"][key] = dict(row)
            group_state["completed_rows_by_model"][model_id] = len(context["rows"])
            if len(context["rows"]) % 16 == 0 or len(context["rows"]) == len(
                context["cases"]
            ):
                _atomic_json(group_state_path, group_state)

        async def on_engine_loaded(event: Mapping[str, Any]) -> None:
            _require(
                event["invocation_id"] == invocation_id, "engine invocation ID drift"
            )
            _append_fsync(invocation_handle, event)
            invocations.append(dict(event))
            group_state["cumulative_model_load_count"] = len(invocations)
            group_state["latest_invocation_id"] = invocation_id
            _atomic_json(group_state_path, group_state)

        snapshot = gpu_snapshot(shard_id)
        report = asyncio.run(
            _run_chk1_cp50_engine_group(
                inputs=inputs,
                cases_by_model={
                    model_id: [
                        {**case, "engine_invocation_id": invocation_id}
                        for case in context["pending"]
                    ]
                    for model_id, context in contexts.items()
                },
                manifest_sha=inputs.manifest_sha256,
                tokenizer=tokenizer,
                eos_token_ids=eos_ids,
                gpu_memory_utilization=float(snapshot["vllm_gpu_memory_utilization"]),
                initial_failure_counts={
                    model_id: {
                        key: len(items) for key, items in context["failures"].items()
                    }
                    for model_id, context in contexts.items()
                },
                shard_id=shard_id,
                invocation_id=invocation_id,
                invocation_index=invocation_index,
                on_engine_loaded=on_engine_loaded,
                on_failure=on_failure,
                on_result=on_result,
            )
        )

    group_manifest = _seal_chk1_cp50_group_receipts(
        inputs=inputs,
        contexts=contexts,
        group_launch_path=group_launch_path,
        group_state_path=group_state_path,
        group_manifest_path=group_manifest_path,
        invocation_wal_path=invocation_wal_path,
        invocations=invocations,
        group_state=group_state,
        report=report,
        snapshot=snapshot,
        shard_id=shard_id,
        smoke=smoke,
        recovered_from_closed_wal=False,
    )
    gc.collect()
    return group_manifest


def validate_model(
    *,
    manifest_path: Path,
    output_root: Path,
    model_id: str,
    smoke: bool = False,
) -> Mapping[str, Any]:
    inputs = load_generation_inputs(manifest_path, output_root)
    tokenizer, eos_ids, _pad_id = _load_tokenizer(inputs.tokenizer_path)
    cases = build_case_matrix(inputs, model_id=model_id, smoke=smoke)
    combined: dict[str, Mapping[str, Any]] = {}
    shards: dict[str, Any] = {}
    for shard_id in SHARD_IDS:
        shard_cases = [case for case in cases if int(case["shard_id"]) == shard_id]
        index = _case_index(shard_cases)
        shard_root = _model_root(output_root, model_id, smoke) / f"shard-{shard_id}"
        generation_wal = shard_root / "generations.wal.jsonl"
        attempt_wal = shard_root / "attempts.wal.jsonl"
        rows = load_generation_wal(
            generation_wal,
            cases_by_key=index,
            manifest_sha=inputs.manifest_sha256,
            tokenizer=tokenizer,
            eos_token_ids=eos_ids,
        )
        failures = load_attempt_ledger(
            attempt_wal, cases_by_key=index, manifest_sha=inputs.manifest_sha256
        )
        _require(rows.keys() == index.keys(), f"{model_id}/shard-{shard_id} incomplete")
        shard_manifest_path = shard_root / "manifest.json"
        shard_manifest = _validate_completed_shard_manifest(
            path=shard_manifest_path,
            inputs=inputs,
            model_id=model_id,
            shard_id=shard_id,
            smoke=smoke,
            cases=shard_cases,
            rows=rows,
            failures=failures,
        )
        for key, row in rows.items():
            _require(key not in combined, f"cross-shard duplicate: {key}")
            _require(
                len(failures.get(key, ())) == int(row["attempt_index"]),
                f"attempt history drift: {key}",
            )
            combined[key] = row
        shards[f"shard-{shard_id}"] = {
            "rows": len(rows),
            "new_rows": int(shard_manifest.get("new_rows", 0)),
            "generation_wall_seconds": float(
                shard_manifest.get("generation_wall_seconds", 0.0)
            ),
            "transport_failure_attempts": sum(
                len(value) for value in failures.values()
            ),
            "worker_manifest": _record(shard_manifest_path),
            "generation_wal": _record(generation_wal, rows=len(rows)),
            "attempt_wal": _record(
                attempt_wal, rows=sum(len(value) for value in failures.values())
            ),
        }
    if model_id in {"chk1", "paper_chk2_cp50"}:
        for shard_id in SHARD_IDS:
            model_manifests = {
                grouped_model_id: _read_json(
                    _model_root(output_root, grouped_model_id, smoke)
                    / f"shard-{shard_id}"
                    / "manifest.json"
                )
                for grouped_model_id in ("chk1", "paper_chk2_cp50")
            }
            group_path = (
                _engine_group_root(output_root, smoke, shard_id) / "manifest.json"
            )
            _validate_engine_group_manifest(
                path=group_path,
                inputs=inputs,
                shard_id=shard_id,
                smoke=smoke,
                model_manifests=model_manifests,
            )
            shards[f"shard-{shard_id}"]["engine_group_manifest"] = _record(group_path)
    _require(len(combined) == len(cases), f"{model_id} two-shard union incomplete")
    ordered_combined = [combined[str(case["generation_key"])] for case in cases]
    delivery = _delivery_funnel(ordered_combined)
    result = {
        "schema_version": MODEL_VALIDATION_SCHEMA,
        "status": "complete",
        "mode": "smoke" if smoke else "formal",
        "evaluation_manifest_sha256": inputs.manifest_sha256,
        "model_id": model_id,
        "generation_rows": len(combined),
        "meetings": SMOKE_MEETINGS if smoke else common.EXPECTED_MEETINGS,
        "replicates": SMOKE_REPLICATES if smoke else len(common.REPLICATE_SEEDS),
        "new_rows": sum(int(item["new_rows"]) for item in shards.values()),
        "generation_wall_seconds": sum(
            float(item["generation_wall_seconds"]) for item in shards.values()
        ),
        "two_gpu_phase_wall_seconds": max(
            float(item["generation_wall_seconds"]) for item in shards.values()
        ),
        "shards": shards,
        "delivery_funnel": delivery,
        "gates": {
            "all_expected_keys_present": True,
            "no_duplicate_or_cross_shard_keys": True,
            "all_rows_use_declared_route": all(
                row["route"] == _route_for_model(model_id) for row in combined.values()
            ),
            "all_rows_deep_replayed_from_token_ids": True,
            "no_unresolved_transport_failure": True,
            "quality_diagnostics_not_used_for_filtering": True,
        },
    }
    path = _model_root(output_root, model_id, smoke) / "validation.json"
    _write_or_verify_json(path, result)
    return result


def validate_shard(
    *,
    manifest_path: Path,
    output_root: Path,
    model_id: str,
    shard_id: int,
    smoke: bool = False,
) -> Mapping[str, Any]:
    inputs = load_generation_inputs(manifest_path, output_root)
    tokenizer, eos_ids, _pad_id = _load_tokenizer(inputs.tokenizer_path)
    cases = [
        case
        for case in build_case_matrix(inputs, model_id=model_id, smoke=smoke)
        if int(case["shard_id"]) == shard_id
    ]
    index = _case_index(cases)
    root = _model_root(output_root, model_id, smoke) / f"shard-{shard_id}"
    rows = load_generation_wal(
        root / "generations.wal.jsonl",
        cases_by_key=index,
        manifest_sha=inputs.manifest_sha256,
        tokenizer=tokenizer,
        eos_token_ids=eos_ids,
    )
    failures = load_attempt_ledger(
        root / "attempts.wal.jsonl",
        cases_by_key=index,
        manifest_sha=inputs.manifest_sha256,
    )
    _require(rows.keys() == index.keys(), "shard validation found missing rows")
    shard_manifest = _validate_completed_shard_manifest(
        path=root / "manifest.json",
        inputs=inputs,
        model_id=model_id,
        shard_id=shard_id,
        smoke=smoke,
        cases=cases,
        rows=rows,
        failures=failures,
    )
    return {
        "status": "complete",
        "mode": "smoke" if smoke else "formal",
        "model_id": model_id,
        "shard_id": shard_id,
        "generation_rows": len(rows),
        "generation_wall_seconds": shard_manifest["generation_wall_seconds"],
        "new_rows": shard_manifest["new_rows"],
        "worker_manifest": _record(root / "manifest.json"),
        "generation_wal": _record(root / "generations.wal.jsonl", rows=len(rows)),
    }


def validate_all_models(
    *,
    manifest_path: Path,
    output_root: Path,
    smoke: bool = False,
) -> Mapping[str, Any]:
    """Deep-validate all three arms and write the terminal generation closure."""

    inputs = load_generation_inputs(manifest_path, output_root)
    tokenizer, eos_ids, _pad_id = _load_tokenizer(inputs.tokenizer_path)
    all_rows: list[Mapping[str, Any]] = []
    model_validations: dict[str, Any] = {}
    tuple_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for model_id in common.MODEL_IDS:
        model_validations[model_id] = validate_model(
            manifest_path=manifest_path,
            output_root=output_root,
            model_id=model_id,
            smoke=smoke,
        )
        cases = build_case_matrix(inputs, model_id=model_id, smoke=smoke)
        by_key: dict[str, Mapping[str, Any]] = {}
        for shard_id in SHARD_IDS:
            selected = [case for case in cases if int(case["shard_id"]) == shard_id]
            selected_index = _case_index(selected)
            rows = load_generation_wal(
                _model_root(output_root, model_id, smoke)
                / f"shard-{shard_id}/generations.wal.jsonl",
                cases_by_key=selected_index,
                manifest_sha=inputs.manifest_sha256,
                tokenizer=tokenizer,
                eos_token_ids=eos_ids,
            )
            by_key.update(rows)
        ordered = [by_key[str(case["generation_key"])] for case in cases]
        all_rows.extend(ordered)
        for row in ordered:
            tuple_groups[str(row["tuple_id"])].append(row)

    expected_per_model = (
        SMOKE_MEETINGS * len(common.TOPICS) * SMOKE_REPLICATES
        if smoke
        else common.EXPECTED_GENERATIONS_PER_MODEL
    )
    expected_total = expected_per_model * len(common.MODEL_IDS)
    _require(len(all_rows) == expected_total, "all-model generation row count drift")
    different_base_lora = 0
    for tuple_key, rows in tuple_groups.items():
        if _validate_paired_tuple_group(tuple_key, rows):
            different_base_lora += 1
    if smoke:
        _require(
            different_base_lora > 0,
            "smoke route proof failed: chk1/cp50 outputs all identical",
        )

    mode_root = _mode_root(output_root, smoke)
    combined_binding = _write_or_verify_jsonl(
        mode_root / "generation_rows.jsonl", all_rows
    )
    overall_delivery = _delivery_funnel(all_rows)
    sequential_two_gpu_wall_seconds = sum(
        float(model_validations[model_id]["two_gpu_phase_wall_seconds"])
        for model_id in common.MODEL_IDS
    )
    timed_new_rows = sum(
        int(model_validations[model_id]["new_rows"]) for model_id in common.MODEL_IDS
    )
    observed_rows_per_second = (
        timed_new_rows / sequential_two_gpu_wall_seconds
        if sequential_two_gpu_wall_seconds > 0.0
        else None
    )
    rough_formal_eta_seconds = (
        common.EXPECTED_GENERATIONS / observed_rows_per_second
        if smoke and observed_rows_per_second
        else None
    )
    result = {
        "schema_version": ALL_VALIDATION_SCHEMA,
        "status": "complete",
        "mode": "smoke" if smoke else "formal",
        "evaluation_id": common.EVALUATION_ID,
        "evaluation_manifest": str(inputs.manifest_path),
        "evaluation_manifest_sha256": inputs.manifest_sha256,
        "generation_rows": len(all_rows),
        "rows_per_model": expected_per_model,
        "paired_tuples": len(tuple_groups),
        "models": model_validations,
        "timing": {
            "new_rows_observed": timed_new_rows,
            "sequential_model_phases_two_gpu_wall_seconds": sequential_two_gpu_wall_seconds,
            "observed_atomic_rows_per_second": observed_rows_per_second,
            "rough_formal_generation_eta_seconds": rough_formal_eta_seconds,
            "rough_formal_generation_eta_hours": (
                rough_formal_eta_seconds / 3_600.0
                if rough_formal_eta_seconds is not None
                else None
            ),
            "eta_is_rough_smoke_extrapolation": bool(smoke),
            "eta_caveat": (
                "Linear extrapolation from a 96-row smoke; output-length and runtime contention may change formal throughput."
                if smoke
                else "Formal observed generation timing; no ETA extrapolation."
            ),
        },
        "delivery_funnel": {
            "overall": overall_delivery,
            "by_model": {
                model_id: model_validations[model_id]["delivery_funnel"]
                for model_id in common.MODEL_IDS
            },
        },
        "combined_generation_rows": combined_binding,
        "base_lora_different_paired_tuples": different_base_lora,
        "gates": {
            "all_20_160_rows_present"
            if not smoke
            else "all_96_smoke_rows_present": True,
            "three_model_tuple_closure": True,
            "same_seed_prompt_and_gpu_across_models": True,
            "chk1_has_no_lora": all(
                row["route"]["lora_request"] is None
                for row in all_rows
                if row["model_id"] == "chk1"
            ),
            "paper_chk2_cp50_has_required_lora": all(
                row["route"]["lora_request"]
                == _route_for_model("paper_chk2_cp50")["lora_request"]
                for row in all_rows
                if row["model_id"] == "paper_chk2_cp50"
            ),
            "no_quality_filtering_or_resampling": True,
            "all_rows_deep_replayed": True,
        },
    }
    _write_or_verify_json(mode_root / "validation.json", result)
    return result


def validate_chk1_cp50_group_shard(
    *,
    manifest_path: Path,
    output_root: Path,
    shard_id: int,
    smoke: bool = False,
) -> Mapping[str, Any]:
    """Deep-validate both model shards plus their shared-engine receipt."""

    inputs = load_generation_inputs(manifest_path, output_root)
    model_results = {
        model_id: validate_shard(
            manifest_path=manifest_path,
            output_root=output_root,
            model_id=model_id,
            shard_id=shard_id,
            smoke=smoke,
        )
        for model_id in ("chk1", "paper_chk2_cp50")
    }
    model_manifests = {
        model_id: _read_json(
            _model_root(output_root, model_id, smoke)
            / f"shard-{shard_id}"
            / "manifest.json"
        )
        for model_id in ("chk1", "paper_chk2_cp50")
    }
    group_path = _engine_group_root(output_root, smoke, shard_id) / "manifest.json"
    group = _validate_engine_group_manifest(
        path=group_path,
        inputs=inputs,
        shard_id=shard_id,
        smoke=smoke,
        model_manifests=model_manifests,
    )
    return {
        "status": "complete",
        "mode": "smoke" if smoke else "formal",
        "engine_group_id": CHK1_CP50_GROUP_ID,
        "shard_id": shard_id,
        "cumulative_model_load_count": group["cumulative_model_load_count"],
        "uninterrupted_single_shared_load": group["uninterrupted_single_shared_load"],
        "models": model_results,
        "engine_group_manifest": _record(group_path),
        "generation_wall_seconds": group["generation_wall_seconds"],
        "new_rows": group["new_rows"],
    }


def assemble_generated_documents(
    *,
    output_root: Path,
    resume: bool = False,
) -> Mapping[str, Any]:
    """Assemble 20,160 validated atomic rows into 2,520 eight-topic documents."""

    output_root = output_root.expanduser().resolve()
    manifest_path = output_root / "preparation/evaluation_manifest.json"
    validate_all_models(
        manifest_path=manifest_path, output_root=output_root, smoke=False
    )
    rows = _read_jsonl(output_root / "generation/generation_rows.jsonl")
    _require(
        len(rows) == common.EXPECTED_GENERATIONS,
        "formal generation rows are not N=20,160",
    )
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["model_id"]), str(row["meeting_id"]), int(row["replicate_id"]))
        ].append(row)
    _require(
        len(grouped) == common.EXPECTED_DOCUMENTS, "document group count is not N=2,520"
    )
    documents: list[dict[str, Any]] = []
    for model_id in common.MODEL_IDS:
        meeting_ids = common.load_prepared(output_root).manifest["population"][
            "meeting_ids"
        ]
        for meeting_id in meeting_ids:
            for replicate_id, replicate_seed in enumerate(common.REPLICATE_SEEDS):
                sections = sorted(
                    grouped[(model_id, str(meeting_id), replicate_id)],
                    key=lambda row: int(row["topic_rank"]),
                )
                _require(
                    len(sections) == len(common.TOPICS),
                    f"document section count drift: {model_id}/{meeting_id}/{replicate_id}",
                )
                _require(
                    tuple(str(row["topic"]) for row in sections) == common.TOPICS,
                    f"document topic order drift: {model_id}/{meeting_id}/{replicate_id}",
                )
                _require(
                    len({int(row["replicate_seed"]) for row in sections}) == 1,
                    "document seed drift",
                )
                answers = [str(row["answer"]) for row in sections]
                text = "\n\n".join(answers)
                documents.append(
                    {
                        "schema_version": DOCUMENT_ROW_SCHEMA,
                        "document_id": f"{model_id}::{meeting_id}::replicate-{replicate_id:02d}",
                        "arm": model_id,
                        "paper_label": common.MODEL_LABELS[model_id],
                        "deterministic": False,
                        "meeting_id": meeting_id,
                        "generation_meeting_id": meeting_id,
                        "meeting_start_date": sections[0]["meeting_start_date"],
                        "meeting_end_date": meeting_id,
                        "replicate_id": replicate_id,
                        "replicate_seed": replicate_seed,
                        "section_count": len(common.TOPICS),
                        "nonempty_section_count": sum(
                            bool(answer) for answer in answers
                        ),
                        "topic_order": list(common.TOPICS),
                        "assembly_separator": "two_newlines_exact_section_answers",
                        "source_generation_keys": [
                            row["generation_key"] for row in sections
                        ],
                        "document_text": text,
                        "document_text_sha256": common.sha256_text(text),
                        "contains_delivery_failure": any(
                            not bool(row["diagnostic_pass"]) for row in sections
                        ),
                        "quality_diagnostics_used_for_exclusion": False,
                    }
                )
    _require(
        len(documents) == common.EXPECTED_DOCUMENTS, "assembled document count drift"
    )
    documents_root = output_root / "documents"
    documents_path = documents_root / "generated_documents.jsonl"
    manifest_out = documents_root / "manifest.json"
    if documents_root.exists():
        _require(resume, f"documents output exists; pass --resume: {documents_root}")
        _require(
            documents_root.is_dir() and not documents_root.is_symlink(),
            "unsafe documents root",
        )
    else:
        documents_root.mkdir(parents=True, exist_ok=False)
    document_binding = _write_or_verify_jsonl(documents_path, documents)
    existing_manifest = _read_json(manifest_out) if manifest_out.exists() else None
    created_at = (
        str(existing_manifest.get("created_at_utc"))
        if existing_manifest is not None
        else _utc_now()
    )
    manifest = seal_manifest(
        {
            "schema_version": DOCUMENT_MANIFEST_SCHEMA,
            "status": "complete",
            "created_at_utc": created_at,
            "evaluation_id": common.EVALUATION_ID,
            "source_generation_validation": _record(
                output_root / "generation/validation.json"
            ),
            "source_generation_rows": _record(
                output_root / "generation/generation_rows.jsonl",
                rows=common.EXPECTED_GENERATIONS,
            ),
            "generated_documents": document_binding,
            "coverage": {
                "models": list(common.MODEL_IDS),
                "meetings": common.EXPECTED_MEETINGS,
                "replicates": len(common.REPLICATE_SEEDS),
                "topics_per_document": len(common.TOPICS),
                "documents_per_model": common.EXPECTED_MEETINGS
                * len(common.REPLICATE_SEEDS),
                "documents_total": common.EXPECTED_DOCUMENTS,
            },
            "assembly": {
                "answer_policy": "unique_think_boundary_suffix_else_empty_string",
                "separator": "two_newlines_exact_section_answers",
                "topic_order": list(common.TOPICS),
                "diagnostic_failures_retained": True,
                "sampling_or_repair_during_assembly": False,
            },
        }
    )
    _write_or_verify_json(manifest_out, manifest)
    return {
        "status": "complete",
        "generated_documents": document_binding,
        "manifest": _record(manifest_out),
        "documents": len(documents),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "validate"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, choices=SHARD_IDS, required=True)
    parser.add_argument("--model-id", choices=WORKER_TARGET_IDS, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "worker":
        if args.model_id == "chk0":
            result = run_worker(
                manifest_path=args.manifest,
                output_root=args.output_root,
                shard_id=args.shard_id,
                model_id="chk0",
                smoke=args.smoke,
                resume=args.resume,
            )
        else:
            result = run_chk1_cp50_group_worker(
                manifest_path=args.manifest,
                output_root=args.output_root,
                shard_id=args.shard_id,
                smoke=args.smoke,
                resume=args.resume,
            )
    else:
        if args.model_id == "chk0":
            result = validate_shard(
                manifest_path=args.manifest,
                output_root=args.output_root,
                shard_id=args.shard_id,
                model_id="chk0",
                smoke=args.smoke,
            )
        else:
            result = validate_chk1_cp50_group_shard(
                manifest_path=args.manifest,
                output_root=args.output_root,
                shard_id=args.shard_id,
                smoke=args.smoke,
            )
    print(common.canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "GenerationInputs",
    "PaperChk2TadleGenerationError",
    "assemble_generated_documents",
    "assigned_shard",
    "build_case_matrix",
    "generation_key",
    "load_attempt_ledger",
    "load_generation_inputs",
    "load_generation_wal",
    "run_chk1_cp50_group_worker",
    "run_worker",
    "tuple_id",
    "validate_all_models",
    "validate_chk1_cp50_group_shard",
    "validate_model",
    "validate_shard",
]
