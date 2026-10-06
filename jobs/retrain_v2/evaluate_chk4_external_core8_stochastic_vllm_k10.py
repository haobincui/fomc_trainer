"""Two-GPU vLLM stochastic Decision diagnostics on the frozen N19/N12 panels.

This create-only workflow extends the deterministic Core8 Decision diagnostic
with ten *fresh* stochastic completions per meeting and model.  It never
trains a model and never opens, resumes, or mutates the predecessor greedy-v4
run.  The frozen decoding contract is ``temperature=0.6``, ``top_p=0.9``,
``top_k=-1``, and ``max_tokens=1536``.

Preparation copies the already sealed v4 panel inputs and materialises their
exact chat-template token IDs.  Formal generation uses two independent vLLM
DP=1 workers.  A complete ``(meeting, replicate)`` block is assigned to one
physical GPU, and the request-level seed is identical across all three frozen
model states.  Each worker persists completion-order raw-token WAL rows with
fsync, immutable chunk receipts, a canonical result, and a terminal receipt.

Scoring reconstructs every material field from the raw completion token IDs.
All statistics use meetings as the independent units.  Replicates remain
nested within meetings, and the N19 and N12 panels are never pooled.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import fcntl
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_V4_ROOT = REPO_ROOT / (
    "output/evaluation/retrain_v2/"
    "chk4_external_deterministic_core8_n19_n12_v4_20260824"
)
SOURCE_V4_MANIFEST = SOURCE_V4_ROOT / "evaluation_manifest.json"
SOURCE_V4_MANIFEST_SHA256 = (
    "689706912c74f800a7e6ad02aa62a8b0616c19b874bd845c1adc74b7dc6d2d22"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    "output/evaluation/retrain_v2/"
    "chk4_external_stochastic_core8_n19_n12_vllm_t06_p09_k10_v3_20260824"
)
FAILED_SMOKE_V1_ROOT = REPO_ROOT / (
    "output/evaluation/retrain_v2/"
    "chk4_external_stochastic_core8_n19_n12_vllm_t06_p09_k10_v1_20260824"
)
FAILED_SMOKE_V1_MANIFEST_SHA256 = (
    "1709ddc8dbfcddcfabb7fbf1db5a08b749e3bde3a9acfa42a75b73cc6295dd91"
)
FAILED_FORMAL_V2_ROOT = REPO_ROOT / (
    "output/evaluation/retrain_v2/"
    "chk4_external_stochastic_core8_n19_n12_vllm_t06_p09_k10_v2_20260824"
)
FAILED_FORMAL_V2_MANIFEST_SHA256 = (
    "cbda2747aa4a1ef24ceb790be20ad10e92727a069341435e3a2c7bb747c0eb51"
)
FAILED_FORMAL_V2_SMOKE_RECEIPT_SHA256 = (
    "ac733889d0fe6f8b5dc5b5b718221eac0d920804e29c2a175d12efef1633cb12"
)
FAILED_FORMAL_V2_AUTHORIZATION_SHA256 = (
    "d9e6a6ad221b9de6374bf01907714e2caa887b784740e601b3443257ff8a1690"
)
VLLM_PYTHON = Path("/home/haobin_cui/.conda/envs/vllm_env/bin/python")

PANELS = ("historical_n19", "postcutoff_n12")
PANEL_COUNTS = {"historical_n19": 19, "postcutoff_n12": 12}
DIRECTIONS = ("cut", "hold", "hike")
PREDICTION_LABELS = (*DIRECTIONS, "invalid")
REPLICATES = 10
REPLICATE_SEEDS = tuple(202_608_240 + value for value in range(REPLICATES))
MODEL_COUNT = 3
MEETING_COUNT = sum(PANEL_COUNTS.values())
PAIRED_BLOCKS = MEETING_COUNT * REPLICATES
EXPECTED_RAW_ROWS = PAIRED_BLOCKS * MODEL_COUNT
EXPECTED_PANEL_ROWS = {
    panel: count * REPLICATES * MODEL_COUNT for panel, count in PANEL_COUNTS.items()
}
SHARD_COUNT = 2
EXPECTED_BLOCKS_PER_SHARD = PAIRED_BLOCKS // SHARD_COUNT
EXPECTED_ROWS_PER_SHARD = EXPECTED_BLOCKS_PER_SHARD * MODEL_COUNT
SMOKE_MODEL_LABEL = "model_chk1_cp200"
SMOKE_MEETING_RANKS = (0, PANEL_COUNTS["historical_n19"])
SMOKE_ROWS = len(SMOKE_MEETING_RANKS) * REPLICATES
SMOKE_ROWS_PER_SHARD = SMOKE_ROWS // SHARD_COUNT

TEMPERATURE = 0.6
TOP_P = 0.9
TOP_K = -1
REPETITION_PENALTY = 1.0
MAX_NEW_TOKENS = 1536
TAIL_TOKENS = 256
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 16
MAX_NUM_BATCHED_TOKENS = 4096
# 0.95 is the established safe maximum on the two 24-GiB A30 cards.  The
# remaining five percent is deliberate CUDA/runtime headroom, not unused
# capacity accidentally left by the scheduler.
GPU_MEMORY_UTILIZATION = 0.95
CHUNK_CASES = 16
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 202_608_241
GLOBAL_PROMPT_TOKEN_IDS_SHA256 = (
    "660d104fd472ab82c7e7ba344ba26dea2b4d51580131e7beb2e86d8223fd63c5"
)
MIN_PROMPT_TOKENS = 781
MAX_OBSERVED_PROMPT_TOKENS = 1060

EXPECTED_VLLM_VERSION = "0.8.5.post1"
EXPECTED_VLLM_PACKAGES = {
    "vllm": "0.8.5.post1",
    "torch": "2.6.0",
    "transformers": "4.51.3",
}
REQUIRED_VLLM_ENV = {
    "VLLM_USE_V1": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "PYTHONNOUSERSITE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
GPU_LOCK_TEMPLATE = "/tmp/fomc_trainer_chk3_beta_core8_vllm_k5_gpu{index}.lock"

EVALUATION_ID = "chk4-external-core8-n19-n12-vllm-t06-p09-k10-dual-dp1-v3"
MANIFEST_SCHEMA = "chk4-external-core8-stochastic-vllm-k10-manifest-v1"
TOKEN_LEDGER_SCHEMA = "chk4-external-core8-stochastic-vllm-k10-token-ledger-v1"
AUTHORIZATION_SCHEMA = "chk4-external-core8-stochastic-vllm-k10-authorization-v1"
SMOKE_RECEIPT_SCHEMA = "chk4-external-core8-stochastic-vllm-k10-smoke-receipt-v1"
RAW_ROW_SCHEMA = "chk4-external-core8-stochastic-vllm-k10-raw-row-v1"
RESULT_ROW_SCHEMA = "chk4-external-core8-stochastic-vllm-k10-result-row-v1"
CHUNK_RECEIPT_SCHEMA = "chk4-external-core8-stochastic-vllm-k10-chunk-receipt-v1"
WORKER_RECEIPT_SCHEMA = "chk4-external-core8-stochastic-vllm-k10-worker-receipt-v1"
RUN_RECEIPT_SCHEMA = "chk4-external-core8-stochastic-vllm-k10-run-receipt-v1"
SUMMARY_SCHEMA = "chk4-external-core8-stochastic-vllm-k10-summary-v1"
BOOTSTRAP_SCHEMA = "chk4-external-core8-stochastic-vllm-k10-bootstrap-draw-v1"

IMPLEMENTATION_FILES = {
    "stochastic_vllm_evaluator": Path(__file__).resolve(),
    "source_v4_evaluator": REPO_ROOT
    / "jobs/retrain_v2/evaluate_chk4_external_core8_panels.py",
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
}


class StochasticDecisionEvaluationError(RuntimeError):
    """A sealed input, GPU worker, replay, or statistic failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StochasticDecisionEvaluationError(message)


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label} missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StochasticDecisionEvaluationError(f"cannot read {label}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must contain an object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"{label} missing or unsafe")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                _require(bool(line.strip()), f"blank line in {label}:{number}")
                value = json.loads(line)
                _require(isinstance(value, dict), f"invalid {label} row {number}")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise StochasticDecisionEvaluationError(f"cannot read {label}: {exc}") from exc
    return rows


def _read_wal_jsonl_recover_torn_final(
    path: Path, *, label: str
) -> list[dict[str, Any]]:
    """Read canonical WAL rows and discard only an uncommitted final fragment.

    A newline is the WAL record commit marker.  Every newline-terminated row
    must be valid canonical JSON.  The only recoverable corruption is a final
    byte fragment without its terminating newline; that fragment is truncated
    after all preceding rows have been fully validated.
    """

    _require(path.is_file() and not path.is_symlink(), f"{label} missing or unsafe")
    raw = path.read_bytes()
    if not raw:
        return []
    parts = raw.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    committed_bytes = 0
    for index, part in enumerate(parts):
        terminal = index == len(parts) - 1
        committed = part.endswith(b"\n")
        if terminal and not committed:
            # Valid-looking but unterminated JSON is also uncommitted.  Never
            # guess whether a crash happened before flush/fsync completion.
            with path.open("r+b") as handle:
                handle.truncate(committed_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            return rows
        _require(committed, f"non-final torn record in {label}")
        payload = part[:-1]
        _require(bool(payload), f"blank committed row in {label}:{index + 1}")
        try:
            text = payload.decode("utf-8")
            value = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StochasticDecisionEvaluationError(
                f"invalid committed row in {label}:{index + 1}: {exc}"
            ) from exc
        _require(isinstance(value, dict), f"non-object committed row in {label}")
        _require(
            text == canonical_json(value),
            f"non-canonical committed row in {label}:{index + 1}",
        )
        rows.append(value)
        committed_bytes += len(part)
    return rows


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


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_exclusive_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _write_exclusive_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_exclusive_text(path, "\n".join(canonical_json(row) for row in rows))


def _write_or_validate_json(path: Path, value: Mapping[str, Any], *, label: str) -> None:
    if path.exists() or path.is_symlink():
        _require(
            _read_json(path, label=label) == dict(value),
            f"existing {label} drift",
        )
        return
    _write_exclusive_json(path, value)


def _write_or_validate_jsonl(
    path: Path, rows: Sequence[Mapping[str, Any]], *, label: str
) -> None:
    expected = [dict(row) for row in rows]
    if path.exists() or path.is_symlink():
        _require(_read_jsonl(path, label=label) == expected, f"existing {label} drift")
        return
    _write_exclusive_jsonl(path, expected)


def _write_exclusive_empty(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _append_fsync(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"unsafe file: {path}")
    rendered = str(path.resolve())
    if relative_to is not None:
        rendered = path.resolve().relative_to(relative_to.resolve()).as_posix()
    result: dict[str, Any] = {
        "path": rendered,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if path.suffix == ".jsonl":
        with path.open("rb") as handle:
            result["rows"] = sum(bool(line.strip()) for line in handle)
    return result


def _failed_smoke_v1_supersession_record() -> dict[str, Any]:
    """Bind the failed v1 smoke attempt without admitting any of its rows."""

    root = FAILED_SMOKE_V1_ROOT.resolve()
    _require(root.is_dir() and not root.is_symlink(), "failed v1 smoke root missing")
    manifest_path = root / "evaluation_manifest.json"
    _require(
        _sha256_file(manifest_path) == FAILED_SMOKE_V1_MANIFEST_SHA256,
        "failed v1 smoke manifest drift",
    )
    wal_paths = {
        f"shard{shard_id}": root
        / "smoke/workers/model_chk1_cp200"
        / f"shard{shard_id}/.partial/completion_order_wal.jsonl"
        for shard_id in range(SHARD_COUNT)
    }
    wal_rows = {
        shard: _read_jsonl(path, label=f"failed v1 smoke {shard} WAL")
        for shard, path in wal_paths.items()
    }
    _require(
        {shard: len(rows) for shard, rows in wal_rows.items()}
        == {"shard0": 10, "shard1": 10}
        and len(
            {
                int(row["absolute_case_index"])
                for rows in wal_rows.values()
                for row in rows
            }
        )
        == SMOKE_ROWS,
        "failed v1 smoke row closure drift",
    )
    _require(
        not (root / "smoke/smoke_receipt.json").exists()
        and not (root / "formal_generation_authorization.json").exists()
        and not (root / "run/run_receipt.json").exists(),
        "failed v1 smoke unexpectedly progressed",
    )
    log_paths = {
        f"shard{shard_id}": root
        / f"smoke/logs/model_chk1_cp200.shard{shard_id}.log"
        for shard_id in range(SHARD_COUNT)
    }
    return {
        "status": "failed_two_gpu_smoke_superseded",
        "output_root": str(root),
        "manifest": _file_record(manifest_path),
        "failure_stage": "post_generation_vllm_v1_engine_teardown",
        "failure_reason": "legacy_shutdown_background_loop_api_unavailable",
        "smoke_wal_rows_generated": SMOKE_ROWS,
        "smoke_wal": {
            shard: _file_record(path, relative_to=root)
            for shard, path in sorted(wal_paths.items())
        },
        "smoke_logs": {
            shard: _file_record(path, relative_to=root)
            for shard, path in sorted(log_paths.items())
        },
        "smoke_rows_reused": 0,
        "formal_rows_reused": 0,
        "formal_generation_started": False,
    }


def _failed_formal_v2_supersession_record() -> dict[str, Any]:
    """Bind v2's passed smoke and dependency-only zero-row launch failure."""

    root = FAILED_FORMAL_V2_ROOT.resolve()
    _require(root.is_dir() and not root.is_symlink(), "failed v2 root missing")
    manifest_path = root / "evaluation_manifest.json"
    smoke_path = root / "smoke/smoke_receipt.json"
    authorization_path = root / "formal_generation_authorization.json"
    _require(
        _sha256_file(manifest_path) == FAILED_FORMAL_V2_MANIFEST_SHA256
        and _sha256_file(smoke_path) == FAILED_FORMAL_V2_SMOKE_RECEIPT_SHA256
        and _sha256_file(authorization_path)
        == FAILED_FORMAL_V2_AUTHORIZATION_SHA256,
        "failed v2 manifest/smoke/authorization drift",
    )
    smoke = _read_json(smoke_path, label="failed v2 smoke receipt")
    authorization = _read_json(
        authorization_path, label="failed v2 authorization"
    )
    _validate_integrity(smoke, label="failed v2 smoke receipt")
    _validate_integrity(authorization, label="failed v2 authorization")
    _require(
        smoke.get("status") == "passed"
        and smoke.get("rows") == SMOKE_ROWS
        and authorization.get("status") == "authorized_after_independent_audit",
        "failed v2 pre-formal evidence drift",
    )
    log_paths = {
        f"shard{shard_id}": root
        / f"run/logs/model_chk1_cp200.shard{shard_id}.log"
        for shard_id in range(SHARD_COUNT)
    }
    for shard, path in log_paths.items():
        _require(path.is_file() and not path.is_symlink(), f"failed v2 {shard} log missing")
        _require(
            "ModuleNotFoundError: No module named 'latex2sympy2_extended'"
            in path.read_text(encoding="utf-8"),
            f"failed v2 {shard} reason drift",
        )
    _require(
        not (root / "run/run_receipt.json").exists()
        and not (root / "run/results.jsonl").exists()
        and not (root / "run/workers").exists(),
        "failed v2 unexpectedly produced formal rows",
    )
    return {
        "status": "failed_formal_worker_dependency_check_superseded",
        "output_root": str(root),
        "manifest": _file_record(manifest_path),
        "smoke_receipt": _file_record(smoke_path),
        "authorization": _file_record(authorization_path),
        "failure_stage": "formal_worker_pre_generation_authorization_validation",
        "failure_reason": "generation_worker_imported_parent_only_reward_dependency",
        "formal_logs": {
            shard: _file_record(path, relative_to=root)
            for shard, path in sorted(log_paths.items())
        },
        "smoke_rows_generated": SMOKE_ROWS,
        "smoke_rows_reused": 0,
        "formal_rows_generated": 0,
        "formal_rows_reused": 0,
    }


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    return {
        **payload,
        "integrity": {
            "algorithm": "sha256(canonical-json-without-integrity)",
            "payload_sha256": _sha256_text(canonical_json(payload)),
        },
    }


def _validate_integrity(value: Mapping[str, Any], *, label: str) -> None:
    integrity = value.get("integrity")
    _require(isinstance(integrity, Mapping), f"{label} integrity missing")
    unsigned = dict(value)
    unsigned.pop("integrity", None)
    _require(
        integrity.get("payload_sha256") == _sha256_text(canonical_json(unsigned)),
        f"{label} payload drift",
    )


def _implementation_records() -> dict[str, dict[str, Any]]:
    return {
        label: _file_record(path)
        for label, path in sorted(IMPLEMENTATION_FILES.items())
    }


def _vllm_runtime_probe() -> dict[str, Any]:
    _require(VLLM_PYTHON.is_file(), f"vLLM Python missing: {VLLM_PYTHON}")
    script = """
import importlib.metadata, json, platform, sys
from pathlib import Path
names = ['vllm','torch','transformers','tokenizers','safetensors','numpy']
print(json.dumps({
  'python': {'executable': str(Path(sys.executable).resolve()), 'version': platform.python_version()},
  'packages': {name: importlib.metadata.version(name) for name in names},
}, sort_keys=True))
"""
    completed = subprocess.run(
        [str(VLLM_PYTHON), "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    _require(bool(lines), "vLLM runtime probe returned no JSON")
    value = json.loads(lines[-1])
    for package, expected in EXPECTED_VLLM_PACKAGES.items():
        observed = str(value["packages"][package])
        _require(
            observed == expected or (package == "torch" and observed.startswith(expected)),
            f"unexpected vLLM runtime {package}: {observed}",
        )
    return value


def _physical_gpu_inventory() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        _require(len(values) == 6, "unexpected NVIDIA inventory schema")
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
    _require(len(rows) == 2, "formal topology requires exactly two physical GPUs")
    _require(
        all(row["name"] == "NVIDIA A30" and row["memory_mib"] == 24576 for row in rows),
        "formal topology requires the two bound 24-GiB A30 GPUs",
    )
    return rows


def _model_path(model: Mapping[str, Any]) -> Path:
    artifact = model.get("artifact")
    _require(isinstance(artifact, Mapping), "model artifact binding missing")
    return Path(str(artifact.get("path") or "")).resolve()


def _model_by_label(manifest: Mapping[str, Any], label: str) -> dict[str, Any]:
    models = [row for row in manifest["models"] if row["label"] == label]
    _require(len(models) == 1, f"unknown model label: {label}")
    return dict(models[0])


def _row_seed(sample_id: str, replicate_id: int) -> int:
    _require(0 <= replicate_id < REPLICATES, "invalid replicate ID")
    value = f"{EVALUATION_ID}:{REPLICATE_SEEDS[replicate_id]}:{sample_id}"
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16) % (2**31)


def canonical_cases(
    samples: Sequence[Mapping[str, Any]], models: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    _require(len(samples) == MEETING_COUNT and len(models) == MODEL_COUNT, "case inputs drift")
    result = []
    for meeting_rank, sample in enumerate(samples):
        for replicate_id in range(REPLICATES):
            paired_block_id = meeting_rank * REPLICATES + replicate_id
            seed = _row_seed(str(sample["sample_id"]), replicate_id)
            for model_rank, model in enumerate(models):
                result.append(
                    {
                        "absolute_case_index": paired_block_id * MODEL_COUNT + model_rank,
                        "paired_block_id": paired_block_id,
                        "assigned_shard": paired_block_id % SHARD_COUNT,
                        "meeting_rank": meeting_rank,
                        "model_rank": model_rank,
                        "model_label": model["label"],
                        "sample_id": sample["sample_id"],
                        "panel": sample["panel"],
                        "replicate_id": replicate_id,
                        "replicate_seed": REPLICATE_SEEDS[replicate_id],
                        "row_seed": seed,
                    }
                )
    _require(len(result) == EXPECTED_RAW_ROWS, "case closure drift")
    _require(
        len({row["absolute_case_index"] for row in result}) == EXPECTED_RAW_ROWS,
        "duplicate absolute case",
    )
    for paired_block_id in range(PAIRED_BLOCKS):
        block = [row for row in result if row["paired_block_id"] == paired_block_id]
        _require(
            len(block) == MODEL_COUNT
            and len({row["row_seed"] for row in block}) == 1
            and len({row["assigned_shard"] for row in block}) == 1,
            "model-shared paired-block seed/shard drift",
        )
    return result


def _copy_regular_file(source: Path, destination: Path) -> None:
    _require(source.is_file() and not source.is_symlink(), f"unsafe source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _safe_prepared_path(root: Path, rendered: Any, *, label: str) -> Path:
    """Resolve one canonical relative path without accepting symlink escapes."""

    _require(isinstance(rendered, str) and bool(rendered), f"{label} path missing")
    relative = Path(rendered)
    _require(
        not relative.is_absolute()
        and relative.as_posix() == rendered
        and all(part not in {"", ".", ".."} for part in relative.parts),
        f"unsafe relative path for {label}: {rendered}",
    )
    candidate = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        _require(not current.is_symlink(), f"symlink in prepared path for {label}")
    try:
        candidate.resolve().relative_to(root)
    except ValueError as error:
        raise StochasticDecisionEvaluationError(
            f"prepared path escapes root for {label}: {rendered}"
        ) from error
    return candidate


def _validate_official_source_archive(
    *, root: Path, manifest_path: Path, samples: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Deeply replay the 31 archived official-source file bindings."""

    rows = _read_jsonl(manifest_path, label="prepared official-source manifest")
    _require(len(rows) == MEETING_COUNT, "prepared official-source count drift")
    sample_by_meeting = {str(row.get("meeting_id") or ""): row for row in samples}
    _require(
        len(samples) == len(sample_by_meeting) == MEETING_COUNT,
        "prepared sample meeting identity drift",
    )
    seen_meetings: set[str] = set()
    seen_paths: set[str] = set()
    for record in rows:
        meeting_id = str(record.get("meeting_id") or "")
        panel = str(record.get("panel") or "")
        rendered_path = str(record.get("path") or "")
        _require(
            meeting_id in sample_by_meeting
            and meeting_id not in seen_meetings
            and rendered_path not in seen_paths,
            f"duplicate or unknown official source: {meeting_id}",
        )
        sample = sample_by_meeting[meeting_id]
        _require(
            panel in PANELS
            and sample.get("panel") == panel
            and rendered_path
            == f"sources/official/{panel}/{meeting_id}.html",
            f"official-source meeting/path binding drift: {meeting_id}",
        )
        source_path = _safe_prepared_path(
            root, rendered_path, label=f"official source {meeting_id}"
        )
        observed = _file_record(source_path, relative_to=root)
        _require(
            all(observed.get(field) == record.get(field) for field in ("path", "bytes", "sha256")),
            f"archived official source drift: {meeting_id}",
        )
        validation = record.get("validation")
        _require(
            isinstance(validation, Mapping)
            and validation.get("domain_check") == "passed",
            f"official-source validation missing: {meeting_id}",
        )
        for field in ("requested_url", "final_url"):
            host = (urlparse(str(record.get(field) or "")).hostname or "").lower()
            _require(
                host == "federalreserve.gov" or host.endswith(".federalreserve.gov"),
                f"official-source domain drift: {meeting_id}::{field}",
            )
        _require(
            record.get("role") == "gold_label_source_only_not_model_input"
            and sample.get("official_source_url")
            in {record.get("requested_url"), record.get("final_url")},
            f"official-source role/URL binding drift: {meeting_id}",
        )
        if sample.get("official_source_sha256") is not None:
            _require(
                sample.get("official_source_sha256") == record.get("sha256"),
                f"official-source sample hash drift: {meeting_id}",
            )
        seen_meetings.add(meeting_id)
        seen_paths.add(rendered_path)
    _require(
        seen_meetings == set(sample_by_meeting)
        and Counter(row["panel"] for row in rows) == Counter(PANEL_COUNTS),
        "official-source population closure drift",
    )
    return rows


def prepare(output_root: Path) -> dict[str, Any]:
    """Seal a fresh stochastic root while leaving source v4 byte-untouched."""

    from jobs.retrain_v2 import evaluate_chk4_external_core8_panels as source

    root = output_root.resolve()
    _require(not root.exists() and not root.is_symlink(), f"output root exists: {root}")
    _require(
        _sha256_file(SOURCE_V4_MANIFEST) == SOURCE_V4_MANIFEST_SHA256,
        "source v4 manifest drift",
    )
    source_manifest = source.validate_manifest(
        SOURCE_V4_MANIFEST, SOURCE_V4_MANIFEST_SHA256
    )
    source_tree_before = {
        "manifest": _file_record(SOURCE_V4_MANIFEST),
        "samples": _file_record(SOURCE_V4_ROOT / source_manifest["inputs"]["samples"]["path"]),
        "official_sources": _file_record(
            SOURCE_V4_ROOT / source_manifest["inputs"]["official_sources"]["path"]
        ),
    }
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        source_samples_path = SOURCE_V4_ROOT / source_manifest["inputs"]["samples"]["path"]
        samples_path = staging / "inputs/panel_samples.jsonl"
        _copy_regular_file(source_samples_path, samples_path)
        samples = _read_jsonl(samples_path, label="copied v4 samples")
        _require(Counter(row["panel"] for row in samples) == Counter(PANEL_COUNTS), "panel drift")

        system_prompt, _ = source._training_contract()
        tokenizer = source._load_tokenizer(source.MODELS[-1])
        token_rows = []
        for sample in samples:
            ids = source._prompt_ids(
                tokenizer,
                source._messages(system_prompt, str(sample["prompt"])),
            )
            _require(len(ids) == sample["prompt_token_count"], "v4 prompt-token count drift")
            token_rows.append(
                {
                    "schema_version": TOKEN_LEDGER_SCHEMA,
                    "sample_id": sample["sample_id"],
                    "prompt_sha256": sample["prompt_sha256"],
                    "prompt_token_count": len(ids),
                    "prompt_token_ids": ids,
                    "prompt_token_ids_sha256": _sha256_text(canonical_json(ids)),
                }
            )
        global_prompt_preimage = [
            [row["sample_id"], row["prompt_token_ids"]] for row in token_rows
        ]
        _require(
            _sha256_text(
                json.dumps(
                    global_prompt_preimage,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            )
            == GLOBAL_PROMPT_TOKEN_IDS_SHA256,
            "default/unfixed global prompt-token ledger SHA drift",
        )
        observed_counts = [int(row["prompt_token_count"]) for row in token_rows]
        _require(
            min(observed_counts) == MIN_PROMPT_TOKENS
            and max(observed_counts) == MAX_OBSERVED_PROMPT_TOKENS,
            "default/unfixed prompt-token range drift",
        )
        token_path = staging / "inputs/prompt_token_ledger.jsonl"
        _write_exclusive_jsonl(token_path, token_rows)

        source_official_manifest = (
            SOURCE_V4_ROOT / source_manifest["inputs"]["official_sources"]["path"]
        )
        official_rows = _read_jsonl(source_official_manifest, label="source v4 official manifest")
        copied_official = []
        for row in official_rows:
            source_path = SOURCE_V4_ROOT / str(row["path"])
            destination = staging / str(row["path"])
            _copy_regular_file(source_path, destination)
            copied = dict(row)
            observed = _file_record(destination, relative_to=staging)
            for key in ("path", "bytes", "sha256"):
                _require(observed[key] == row[key], f"official-source copy drift: {row['meeting_id']}")
            copied_official.append(copied)
        official_path = staging / "sources/official_source_manifest.jsonl"
        _write_exclusive_jsonl(official_path, copied_official)

        models = copy.deepcopy(source_manifest["models"])
        cases = canonical_cases(samples, models)
        case_path = staging / "inputs/case_schedule.jsonl"
        _write_exclusive_jsonl(case_path, cases)
        vllm_runtime = _vllm_runtime_probe()
        gpu_inventory = _physical_gpu_inventory()
        _require(
            gpu_inventory == source_manifest["runtime_prebinding"]["physical_gpu_inventory"],
            "source-v4 GPU inventory drift",
        )
        manifest = _sealed(
            {
                "schema_version": MANIFEST_SCHEMA,
                "status": "prepared_pending_independent_audit",
                "created_at_utc": _utc_now(),
                "evaluation_id": EVALUATION_ID,
                "purpose": "stochastic_decoding_robustness_without_retraining",
                "output_root": str(root),
                "source_v4": {
                    "role": "sealed_input_source_only_greedy_run_never_reused",
                    "manifest": _file_record(SOURCE_V4_MANIFEST),
                    "manifest_expected_sha256": SOURCE_V4_MANIFEST_SHA256,
                    "source_tree_before": source_tree_before,
                    "greedy_generation_rows_reused": 0,
                    "source_run_mutation_authorized": False,
                },
                "supersedes_failed_attempts": [
                    _failed_smoke_v1_supersession_record(),
                    _failed_formal_v2_supersession_record(),
                ],
                "implementation": _implementation_records(),
                "models": models,
                "checkpoint_governance": {
                    "model_chk3_grpo_cp450": {
                        "selected_checkpoint_step": None,
                        "release_verdict": "quality_not_demonstrated",
                        "paper_facing_status": "best-observed exploratory candidate used for evaluation",
                        "selected_promoted_or_final_language_prohibited": True,
                    }
                },
                "inputs": {
                    "samples": _file_record(samples_path, relative_to=staging),
                    "prompt_token_ledger": _file_record(token_path, relative_to=staging),
                    "case_schedule": _file_record(case_path, relative_to=staging),
                    "official_sources": _file_record(official_path, relative_to=staging),
                },
                "prompt_contract": {
                    **copy.deepcopy(source_manifest["prompt_contract"]),
                    "tokenizer_fix_mistral_regex": False,
                    "fix_mistral_regex_true_forbidden": True,
                    "global_prompt_token_ids_sha256": GLOBAL_PROMPT_TOKEN_IDS_SHA256,
                    "global_prompt_token_ids_preimage": (
                        "json.dumps([[sample_id,token_ids],...],"
                        "separators=(',',':'),ensure_ascii=False) in physical sample order"
                    ),
                    "observed_prompt_token_range": [
                        MIN_PROMPT_TOKENS,
                        MAX_OBSERVED_PROMPT_TOKENS,
                    ],
                },
                "input_contract": copy.deepcopy(source_manifest["input_contract"]),
                "population": {
                    "panels": PANEL_COUNTS,
                    "class_counts": copy.deepcopy(source_manifest["population"]["class_counts"]),
                    "meetings": MEETING_COUNT,
                    "models": MODEL_COUNT,
                    "replicates_per_meeting_model": REPLICATES,
                    "expected_rows": EXPECTED_RAW_ROWS,
                    "expected_panel_rows": EXPECTED_PANEL_ROWS,
                    "training_performed": False,
                },
                "generation": {
                    "backend": "vllm-async-engine-v1-two-independent-dp1-workers",
                    "generation_mode": "stochastic_sampling",
                    "do_sample": True,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                    "repetition_penalty": REPETITION_PENALTY,
                    "max_tokens": MAX_NEW_TOKENS,
                    "max_model_len": MAX_MODEL_LEN,
                    "max_num_seqs_per_worker": MAX_NUM_SEQS,
                    "max_num_batched_tokens_per_worker": MAX_NUM_BATCHED_TOKENS,
                    "gpu_memory_utilization_per_worker": GPU_MEMORY_UTILIZATION,
                    "gpu_memory_headroom_fraction": 1.0 - GPU_MEMORY_UTILIZATION,
                    "dtype": "bfloat16",
                    "quantization": None,
                    "data_parallel_size_per_worker": 1,
                    "tensor_parallel_size_per_worker": 1,
                    "pipeline_parallel_size_per_worker": 1,
                    "physical_gpu_indexes": [0, 1],
                    "workers_per_model": 2,
                    "models_processed_sequentially": True,
                    "paired_block": "meeting_x_replicate_all_three_models",
                    "shard_function": "paired_block_id_mod_2",
                    "paired_block_never_crosses_workers": True,
                    "replicate_seeds": list(REPLICATE_SEEDS),
                    "row_seed": "sha256(evaluation_id:replicate_seed:sample_id)_first64_mod_2^31",
                    "row_seed_shared_across_models": True,
                    "exact_prompt_token_ids_only": True,
                    "fresh_generations_only": True,
                    "expected_paired_blocks": PAIRED_BLOCKS,
                    "expected_blocks_per_shard": EXPECTED_BLOCKS_PER_SHARD,
                    "expected_rows_per_shard": EXPECTED_ROWS_PER_SHARD,
                },
                "runtime_prebinding": {
                    "vllm_python": vllm_runtime,
                    "physical_gpu_inventory": gpu_inventory,
                    "required_environment": REQUIRED_VLLM_ENV,
                    "gpu_lock_template": GPU_LOCK_TEMPLATE,
                },
                "persistence": {
                    "single_writer_completion_order_wal_per_model_shard": True,
                    "append_flush_fsync_per_completed_request": True,
                    "resume_only_missing_tuple_keys": True,
                    "fsynced_rows_never_redispatched": True,
                    "chunk_receipt_after_exact_chunk_union": True,
                    "raw_completion_token_ids_retained": True,
                    "full_raw_token_replay_required": True,
                    "post_run_receipt_required": True,
                },
                "statistics": {
                    "independent_unit": "meeting",
                    "generation_rows_are_not_independent": True,
                    "panels_reported_separately": True,
                    "pooled_n31_prohibited": True,
                    "bootstrap": "10000 class-stratified paired two-stage hierarchical draws; meetings within class and replicate indices within meeting; draws shared across models",
                    "accuracy_and_balanced_accuracy_test": "exact full-meeting-vector label-swap; six-test Holm family per panel",
                    "delivery_test": "exact full-meeting-vector label-swap; separate three-test Holm family per panel",
                    "modal_test": "exact McNemar with separate three-test Holm family per panel",
                    "postcutoff_balanced_accuracy": "supported cut/hold classes only",
                },
                "limitations": [
                    "K=10 measures decoding variability and does not increase the number of independent meetings",
                    "postcutoff_n12 contains no hike meeting",
                    "historical_n19 may be represented in base-model pretraining",
                    "both panels are retrospective diagnostics",
                    "the deterministic Core8 input contract differs from the opened teacher-compressed N13 contract",
                ],
            }
        )
        manifest_path = staging / "evaluation_manifest.json"
        _write_exclusive_json(manifest_path, manifest)
        os.rename(staging, root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    _require(
        source_tree_before
        == {
            "manifest": _file_record(SOURCE_V4_MANIFEST),
            "samples": _file_record(SOURCE_V4_ROOT / source_manifest["inputs"]["samples"]["path"]),
            "official_sources": _file_record(
                SOURCE_V4_ROOT / source_manifest["inputs"]["official_sources"]["path"]
            ),
        },
        "source v4 changed during preparation",
    )
    return {
        "status": "prepared_pending_independent_audit",
        "manifest": _file_record(root / "evaluation_manifest.json"),
        "expected_rows": EXPECTED_RAW_ROWS,
        "source_v4_unchanged": True,
    }


def validate_manifest(path: Path, expected_sha256: str, *, deep_models: bool = False) -> dict[str, Any]:
    _require(_sha256_file(path) == expected_sha256, "manifest SHA drift")
    value = _read_json(path, label="stochastic manifest")
    _require(value.get("schema_version") == MANIFEST_SCHEMA, "manifest schema drift")
    _validate_integrity(value, label="stochastic manifest")
    root = path.parent.resolve()
    _require(value.get("output_root") == str(root), "manifest root drift")
    _require(value.get("implementation") == _implementation_records(), "implementation drift")
    _require(
        _sha256_file(SOURCE_V4_MANIFEST) == SOURCE_V4_MANIFEST_SHA256
        and value["source_v4"]["manifest"] == _file_record(SOURCE_V4_MANIFEST),
        "source v4 binding drift",
    )
    _require(
        value.get("supersedes_failed_attempts")
        == [
            _failed_smoke_v1_supersession_record(),
            _failed_formal_v2_supersession_record(),
        ],
        "failed-attempt supersession binding drift",
    )
    for name in ("samples", "prompt_token_ledger", "case_schedule", "official_sources"):
        record = value["inputs"][name]
        prepared_path = _safe_prepared_path(
            root, record.get("path"), label=f"prepared input {name}"
        )
        _require(
            _file_record(prepared_path, relative_to=root) == record,
            f"prepared input drift: {name}",
        )
    samples = _read_jsonl(
        _safe_prepared_path(
            root,
            value["inputs"]["samples"].get("path"),
            label="prepared samples",
        ),
        label="prepared samples",
    )
    _validate_official_source_archive(
        root=root,
        manifest_path=_safe_prepared_path(
            root,
            value["inputs"]["official_sources"].get("path"),
            label="prepared official-source manifest",
        ),
        samples=samples,
    )
    _require(value["generation"]["expected_rows_per_shard"] == EXPECTED_ROWS_PER_SHARD, "shard closure drift")
    if deep_models:
        from open_r1.provenance import fingerprint_artifact_path

        for model in value["models"]:
            _require(
                fingerprint_artifact_path(_model_path(model)) == model["artifact"],
                f"model drift: {model['label']}",
            )
    return value


def authorize(
    manifest_path: Path,
    manifest_sha256: str,
    *,
    audit_statement: str,
) -> dict[str, Any]:
    manifest = validate_manifest(manifest_path, manifest_sha256, deep_models=True)
    _require(len(audit_statement.strip()) >= 20, "independent audit statement is too short")
    smoke_path = manifest_path.parent / "smoke/smoke_receipt.json"
    smoke = _validate_smoke_receipt(
        smoke_path,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )
    path = manifest_path.parent / "formal_generation_authorization.json"
    _require(not path.exists() and not path.is_symlink(), "authorization already exists")
    value = _sealed(
        {
            "schema_version": AUTHORIZATION_SCHEMA,
            "status": "authorized_after_independent_audit",
            "created_at_utc": _utc_now(),
            "evaluation_id": EVALUATION_ID,
            "manifest": {"path": str(manifest_path.resolve()), "sha256": manifest_sha256},
            "audit_statement": audit_statement.strip(),
            "authorized_scope": "formal_930_row_two_gpu_vllm_generation_only",
            "validated_infrastructure_smoke": {
                "receipt": _file_record(smoke_path),
                "rows": smoke["rows"],
                "shards": smoke["shards"],
                "gates": copy.deepcopy(smoke["gates"]),
            },
            "generation_contract": copy.deepcopy(manifest["generation"]),
        }
    )
    _write_exclusive_json(path, value)
    return {"status": value["status"], "authorization": _file_record(path)}


def _validate_authorization(
    path: Path,
    expected_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    *,
    deep_smoke: bool = True,
) -> dict[str, Any]:
    _require(_sha256_file(path) == expected_sha256, "authorization SHA drift")
    value = _read_json(path, label="formal authorization")
    _validate_integrity(value, label="formal authorization")
    _require(
        value.get("schema_version") == AUTHORIZATION_SCHEMA
        and value.get("status") == "authorized_after_independent_audit"
        and value.get("manifest")
        == {"path": str(manifest_path.resolve()), "sha256": manifest_sha256},
        "authorization scope drift",
    )
    smoke_binding = value.get("validated_infrastructure_smoke", {}).get("receipt")
    _require(isinstance(smoke_binding, Mapping), "authorization lacks smoke binding")
    smoke_path = Path(str(smoke_binding.get("path") or ""))
    _require(_file_record(smoke_path) == smoke_binding, "authorized smoke receipt drift")
    if deep_smoke:
        smoke = _validate_smoke_receipt(
            smoke_path,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
        )
    else:
        # GPU workers are raw-generation processes.  The parent process has
        # already deep-replayed the smoke receipt before launch; workers only
        # need a dependency-light cryptographic authorization check and must
        # not import the reward/scoring stack from the training environment.
        smoke = _read_json(smoke_path, label="authorized smoke receipt")
        _validate_integrity(smoke, label="authorized smoke receipt")
        _require(
            smoke.get("schema_version") == SMOKE_RECEIPT_SCHEMA
            and smoke.get("status") == "passed"
            and smoke.get("evaluation_id") == EVALUATION_ID
            and smoke.get("manifest")
            == {"path": str(manifest_path.resolve()), "sha256": manifest_sha256}
            and smoke.get("rows") == SMOKE_ROWS
            and smoke.get("shards") == {"shard0": 10, "shard1": 10},
            "authorized smoke receipt shallow scope drift",
        )
    _require(
        value["validated_infrastructure_smoke"]["rows"] == smoke["rows"]
        and value["validated_infrastructure_smoke"]["shards"] == smoke["shards"]
        and value["validated_infrastructure_smoke"]["gates"] == smoke["gates"],
        "authorized smoke evidence drift",
    )
    return value


def _load_prepared(
    manifest: Mapping[str, Any], root: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    samples = _read_jsonl(root / manifest["inputs"]["samples"]["path"], label="samples")
    tokens = _read_jsonl(
        root / manifest["inputs"]["prompt_token_ledger"]["path"], label="token ledger"
    )
    token_by_id = {str(row["sample_id"]): row for row in tokens}
    _require(len(samples) == len(token_by_id) == MEETING_COUNT, "sample/token closure drift")
    for sample in samples:
        token = token_by_id[str(sample["sample_id"])]
        ids = token.get("prompt_token_ids")
        _require(
            isinstance(ids, list)
            and len(ids) == sample["prompt_token_count"]
            and token["prompt_sha256"] == sample["prompt_sha256"]
            and token["prompt_token_ids_sha256"] == _sha256_text(canonical_json(ids)),
            f"prompt-token ledger drift: {sample['sample_id']}",
        )
    _require(
        _sha256_text(
            json.dumps(
                [
                    [row["sample_id"], row["prompt_token_ids"]]
                    for row in tokens
                ],
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
        == GLOBAL_PROMPT_TOKEN_IDS_SHA256,
        "global default/unfixed prompt-token ledger SHA drift",
    )
    counts = [int(row["prompt_token_count"]) for row in tokens]
    _require(
        min(counts) == MIN_PROMPT_TOKENS
        and max(counts) == MAX_OBSERVED_PROMPT_TOKENS,
        "global prompt-token range drift",
    )
    cases = canonical_cases(samples, manifest["models"])
    stored_cases = _read_jsonl(
        root / manifest["inputs"]["case_schedule"]["path"], label="case schedule"
    )
    _require(stored_cases == cases, "case schedule replay drift")
    return samples, token_by_id, cases


def _raw_worker_row(
    *,
    case: Mapping[str, Any],
    token: Mapping[str, Any],
    manifest_sha256: str,
    model: Mapping[str, Any],
    shard_id: int,
    physical_gpu_index: int,
    generated_token_ids: Sequence[int],
    generated_text: str,
    finish_reason: Any,
    stop_reason: Any,
    request_id: str,
) -> dict[str, Any]:
    ids = [int(value) for value in generated_token_ids]
    _require(bool(ids) and len(ids) <= MAX_NEW_TOKENS, "invalid generated token IDs")
    return {
        "schema_version": RAW_ROW_SCHEMA,
        "evaluation_id": EVALUATION_ID,
        "evaluation_manifest_sha256": manifest_sha256,
        "absolute_case_index": case["absolute_case_index"],
        "paired_block_id": case["paired_block_id"],
        "meeting_rank": case["meeting_rank"],
        "model_rank": case["model_rank"],
        "model_label": case["model_label"],
        "model_sha256": model["sha256"],
        "sample_id": case["sample_id"],
        "panel": case["panel"],
        "replicate_id": case["replicate_id"],
        "replicate_seed": case["replicate_seed"],
        "row_seed": case["row_seed"],
        "shard_id": shard_id,
        "physical_gpu_index": physical_gpu_index,
        "prompt_token_count": token["prompt_token_count"],
        "prompt_token_ids_sha256": token["prompt_token_ids_sha256"],
        "request_id": request_id,
        "generation_parameters": {
            "do_sample": True,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "repetition_penalty": REPETITION_PENALTY,
            "max_tokens": MAX_NEW_TOKENS,
            "seed": case["row_seed"],
        },
        "generated_token_ids_raw": ids,
        "generated_token_ids_sha256": _sha256_text(canonical_json(ids)),
        "vllm_generated_text": generated_text,
        "vllm_generated_text_sha256": _sha256_text(generated_text),
        "vllm_text_semantics": (
            "vllm_v1_completion_text_excludes_a_retained_terminal_stop_token"
        ),
        "vllm_finish_reason": finish_reason,
        "vllm_stop_reason": stop_reason,
    }


def _normalise_token_id_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int) and not isinstance(value, bool):
        return {value}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = {
            int(item)
            for item in value
            if isinstance(item, int) and not isinstance(item, bool)
        }
        _require(len(result) == len(value), "non-integer stop-token ID")
        return result
    raise StochasticDecisionEvaluationError("invalid stop-token ID contract")


def _bound_eos_ids(model: Mapping[str, Any], tokenizer: Any) -> set[int]:
    path = _model_path(model) / "generation_config.json"
    config = _read_json(path, label=f"{model['label']} generation config")
    result = _normalise_token_id_set(config.get("eos_token_id"))
    if not result:
        result = _normalise_token_id_set(getattr(tokenizer, "eos_token_id", None))
    _require(bool(result), f"{model['label']} has no bound EOS token")
    return result


def _accepted_vllm_text_from_raw_ids(
    *,
    tokenizer: Any,
    generated_token_ids: Sequence[int],
    finish_reason: Any,
    stop_reason: Any,
    eos_ids: set[int],
) -> tuple[str, dict[str, Any]]:
    """Mirror vLLM V1's CompletionOutput.text stop-token semantics.

    vLLM retains a terminal EOS/stop token in ``token_ids`` but omits that
    token from ``CompletionOutput.text``.  Non-stop and length completions
    decode every retained token.  Raw IDs remain authoritative in both cases.
    """

    ids = [int(value) for value in generated_token_ids]
    stop_ids = set(eos_ids)
    if isinstance(stop_reason, int) and not isinstance(stop_reason, bool):
        stop_ids.add(stop_reason)
    omitted_terminal_stop_id: int | None = None
    accepted_ids = ids
    if finish_reason == "stop" and ids and ids[-1] in stop_ids:
        omitted_terminal_stop_id = ids[-1]
        accepted_ids = ids[:-1]
    text = tokenizer.decode(
        accepted_ids,
        skip_special_tokens=False,
        spaces_between_special_tokens=True,
    )
    return text, {
        "finish_reason": finish_reason,
        "raw_token_count": len(ids),
        "accepted_text_token_count": len(accepted_ids),
        "omitted_terminal_stop_id": omitted_terminal_stop_id,
        "rule": (
            "omit_one_retained_terminal_stop_token_iff_finish_reason_is_stop"
        ),
    }


def _validate_raw_row(
    row: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    token: Mapping[str, Any],
    model: Mapping[str, Any],
    manifest_sha256: str,
    shard_id: int,
    physical_gpu_index: int,
    tokenizer: Any | None = None,
    eos_ids: set[int] | None = None,
) -> dict[str, Any]:
    ids = row.get("generated_token_ids_raw")
    _require(isinstance(ids, list) and bool(ids), "raw generated token IDs missing")
    accepted_text = str(row.get("vllm_generated_text") or "")
    if tokenizer is not None:
        _require(bool(eos_ids), "EOS IDs are required for vLLM text replay")
        accepted_text, _ = _accepted_vllm_text_from_raw_ids(
            tokenizer=tokenizer,
            generated_token_ids=ids,
            finish_reason=row.get("vllm_finish_reason"),
            stop_reason=row.get("vllm_stop_reason"),
            eos_ids=set(eos_ids or ()),
        )
        _require(
            accepted_text == row.get("vllm_generated_text"),
            "vLLM V1 accepted-text/raw-token drift",
        )
    expected = _raw_worker_row(
        case=case,
        token=token,
        manifest_sha256=manifest_sha256,
        model=model,
        shard_id=shard_id,
        physical_gpu_index=physical_gpu_index,
        generated_token_ids=ids,
        generated_text=accepted_text,
        finish_reason=row.get("vllm_finish_reason"),
        stop_reason=row.get("vllm_stop_reason"),
        request_id=str(row.get("request_id") or ""),
    )
    _require(dict(row) == expected, f"raw WAL replay drift: {case['absolute_case_index']}")
    return expected


def _worker_paths(
    root: Path, model_label: str, shard_id: int, *, scope: str = "formal"
) -> dict[str, Path]:
    _require(scope in {"formal", "smoke"}, "invalid worker scope")
    stage_root = "run" if scope == "formal" else "smoke"
    worker = root / stage_root / "workers" / model_label / f"shard{shard_id}"
    return {
        "root": worker,
        "partial": worker / ".partial",
        "launch": worker / "launch.json",
        "wal": worker / ".partial/completion_order_wal.jsonl",
        "receipts": worker / ".partial/chunk_receipts.jsonl",
        "state": worker / "state.json",
        "runtime_progress": worker / "runtime.progress.json",
        "canonical": worker / "raw_generations.canonical.jsonl",
        "runtime": worker / "runtime.json",
        "receipt": worker / "worker_receipt.json",
    }


def _worker_cases(
    cases: Sequence[Mapping[str, Any]], model_label: str, shard_id: int,
    *, scope: str = "formal"
) -> list[dict[str, Any]]:
    _require(scope in {"formal", "smoke"}, "invalid worker scope")
    if scope == "smoke":
        _require(model_label == SMOKE_MODEL_LABEL, "smoke model drift")
    selected = [
        dict(row)
        for row in cases
        if row["model_label"] == model_label and row["assigned_shard"] == shard_id
        and (scope == "formal" or int(row["meeting_rank"]) in SMOKE_MEETING_RANKS)
    ]
    expected = EXPECTED_BLOCKS_PER_SHARD if scope == "formal" else SMOKE_ROWS_PER_SHARD
    _require(len(selected) == expected, "worker case count drift")
    return selected


def _chunks(values: Sequence[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    return [
        [dict(row) for row in values[index : index + CHUNK_CASES]]
        for index in range(0, len(values), CHUNK_CASES)
    ]


def _chunk_receipt(
    chunk_id: int, chunk: Sequence[Mapping[str, Any]], rows: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    indexes = [int(case["absolute_case_index"]) for case in chunk]
    _require(all(index in rows for index in indexes), "cannot receipt incomplete chunk")
    bindings = [
        {
            "absolute_case_index": index,
            "sample_id": rows[index]["sample_id"],
            "replicate_id": rows[index]["replicate_id"],
            "row_seed": rows[index]["row_seed"],
            "generated_token_ids_sha256": rows[index]["generated_token_ids_sha256"],
        }
        for index in indexes
    ]
    return {
        "schema_version": CHUNK_RECEIPT_SCHEMA,
        "chunk_id": chunk_id,
        "cases": len(indexes),
        "absolute_case_indexes": indexes,
        "canonical_bindings_sha256": _sha256_text(canonical_json(bindings)),
    }


def _validate_worker_receipt(
    *, paths: Mapping[str, Path], root: Path, manifest: Mapping[str, Any],
    manifest_sha256: str, model: Mapping[str, Any], shard_id: int,
    scope: str, authorization_path: Path | None,
    canonical_rows: Sequence[Mapping[str, Any]],
    selected_cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    receipt = _read_json(paths["receipt"], label="worker receipt")
    _validate_integrity(receipt, label="worker receipt")
    expected_authorization = (
        _file_record(authorization_path) if authorization_path is not None else None
    )
    _require(
        receipt.get("schema_version") == WORKER_RECEIPT_SCHEMA
        and receipt.get("status") == "complete"
        and receipt.get("evaluation_scope") == scope
        and receipt.get("manifest_sha256") == manifest_sha256
        and receipt.get("authorization") == expected_authorization
        and receipt.get("model_label") == model["label"]
        and receipt.get("model_sha256") == model["sha256"]
        and receipt.get("shard_id") == shard_id
        and receipt.get("physical_gpu_index") == shard_id
        and receipt.get("rows") == len(canonical_rows),
        "worker receipt role drift",
    )
    for name, key in (
        ("launch", "launch"),
        ("wal", "wal"),
        ("receipts", "chunk_receipts"),
        ("runtime_progress", "runtime_progress"),
        ("canonical", "canonical_raw_results"),
        ("runtime", "runtime"),
    ):
        _require(
            _file_record(paths[name], relative_to=root) == receipt.get(key),
            f"worker receipt file drift: {model['label']} shard{shard_id} {name}",
        )
    _require(
        _read_jsonl(paths["canonical"], label="canonical worker output")
        == [dict(row) for row in canonical_rows],
        "canonical worker output drift",
    )
    wal_rows = _read_jsonl(paths["wal"], label="terminal worker WAL")
    wal_by_index = {
        int(row["absolute_case_index"]): row for row in wal_rows
    }
    canonical_by_index = {
        int(row["absolute_case_index"]): dict(row) for row in canonical_rows
    }
    _require(
        len(wal_rows) == len(wal_by_index) == len(canonical_by_index)
        and wal_by_index == canonical_by_index,
        "terminal worker WAL/canonical exact-union drift",
    )
    chunks = _chunks(selected_cases)
    chunk_receipts = _read_jsonl(
        paths["receipts"], label="terminal worker chunk receipts"
    )
    _require(
        len(chunk_receipts) == len(chunks)
        and all(
            receipt
            == _chunk_receipt(chunk_id, chunk, canonical_by_index)
            for chunk_id, (chunk, receipt) in enumerate(
                zip(chunks, chunk_receipts, strict=True)
            )
        ),
        "terminal worker chunk-receipt closure drift",
    )
    _validate_worker_runtime_file(
        paths["runtime"],
        manifest=manifest,
        model_label=str(model["label"]),
        shard_id=shard_id,
        expected_rows=len(canonical_rows),
    )
    return receipt


def _finalize_worker(
    *, paths: Mapping[str, Path], root: Path, manifest: Mapping[str, Any],
    manifest_sha256: str, model: Mapping[str, Any], shard_id: int,
    scope: str, authorization_path: Path | None,
    canonical_rows: Sequence[Mapping[str, Any]],
    selected_cases: Sequence[Mapping[str, Any]], runtime: Mapping[str, Any]
) -> dict[str, Any]:
    _write_or_validate_jsonl(
        paths["canonical"], canonical_rows, label="canonical worker output"
    )
    _write_or_validate_json(paths["runtime"], runtime, label="worker runtime")
    receipt_payload = {
        "schema_version": WORKER_RECEIPT_SCHEMA,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "evaluation_id": EVALUATION_ID,
        "manifest_sha256": manifest_sha256,
        "evaluation_scope": scope,
        "authorization": (
            _file_record(authorization_path) if authorization_path is not None else None
        ),
        "model_label": model["label"],
        "model_sha256": model["sha256"],
        "shard_id": shard_id,
        "physical_gpu_index": shard_id,
        "rows": len(canonical_rows),
        "launch": _file_record(paths["launch"], relative_to=root),
        "wal": _file_record(paths["wal"], relative_to=root),
        "chunk_receipts": _file_record(paths["receipts"], relative_to=root),
        "runtime_progress": _file_record(
            paths["runtime_progress"], relative_to=root
        ),
        "canonical_raw_results": _file_record(paths["canonical"], relative_to=root),
        "runtime": _file_record(paths["runtime"], relative_to=root),
    }
    if paths["receipt"].exists():
        existing = _read_json(paths["receipt"], label="worker receipt")
        # Creation time is evidentiary but intentionally not reconstructed on
        # an idempotent resume after the receipt commit.
        receipt_payload["created_at_utc"] = existing.get("created_at_utc")
        expected = _sealed(receipt_payload)
        _require(existing == expected, "existing worker receipt drift")
    else:
        _write_exclusive_json(paths["receipt"], _sealed(receipt_payload))
    _atomic_json(
        paths["state"],
        _sealed(
            {
                "status": "complete",
                "updated_at_utc": _utc_now(),
                "model_label": model["label"],
                "shard_id": shard_id,
                "completed_cases": len(canonical_rows),
                "expected_cases": len(canonical_rows),
            }
        ),
    )
    return _validate_worker_receipt(
        paths=paths,
        root=root,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        model=model,
        shard_id=shard_id,
        scope=scope,
        authorization_path=authorization_path,
        canonical_rows=canonical_rows,
        selected_cases=selected_cases,
    )


def _nvidia_memory(index: int) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=uuid,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    _require(len(values) == 5, "unexpected GPU memory schema")
    return {
        "uuid": values[0],
        "memory_total_mib": int(values[1]),
        "memory_used_mib": int(values[2]),
        "memory_free_mib": int(values[3]),
        "utilization_percent": int(values[4]),
    }


def _worker_runtime_observation(
    manifest: Mapping[str, Any], physical_gpu_index: int
) -> dict[str, Any]:
    package_names = ("vllm", "torch", "transformers", "tokenizers", "safetensors", "numpy")
    observation = {
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": ".".join(str(value) for value in sys.version_info[:3]),
        },
        "packages": {
            name: importlib.metadata.version(name) for name in package_names
        },
        "physical_gpu_inventory": _physical_gpu_inventory(),
        "selected_physical_gpu": _nvidia_memory(physical_gpu_index),
        "environment": {
            **{key: os.environ.get(key) for key in REQUIRED_VLLM_ENV},
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
    }
    _validate_worker_runtime_observation(
        observation, manifest=manifest, physical_gpu_index=physical_gpu_index
    )
    return observation


def _validate_worker_runtime_observation(
    observation: Mapping[str, Any], *, manifest: Mapping[str, Any],
    physical_gpu_index: int
) -> None:
    prebound = manifest["runtime_prebinding"]
    expected_python = prebound["vllm_python"]["python"]
    expected_packages = prebound["vllm_python"]["packages"]
    _require(observation.get("python") == expected_python, "worker Python runtime drift")
    _require(
        observation.get("packages") == expected_packages,
        "worker package-version runtime drift",
    )
    _require(
        observation.get("physical_gpu_inventory") == prebound["physical_gpu_inventory"],
        "worker physical GPU inventory drift",
    )
    selected = observation.get("selected_physical_gpu")
    expected_gpu = prebound["physical_gpu_inventory"][physical_gpu_index]
    _require(
        isinstance(selected, Mapping)
        and selected.get("uuid") == expected_gpu["uuid"]
        and selected.get("memory_total_mib") == expected_gpu["memory_mib"],
        "worker selected physical GPU UUID/memory drift",
    )
    environment = observation.get("environment")
    _require(isinstance(environment, Mapping), "worker environment record missing")
    for key, expected in REQUIRED_VLLM_ENV.items():
        _require(environment.get(key) == expected, f"worker environment drift: {key}")
    _require(
        environment.get("CUDA_VISIBLE_DEVICES") == str(physical_gpu_index)
        and environment.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
        "worker CUDA environment drift",
    )
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        _require(environment.get(key) == "1", f"worker thread environment drift: {key}")


def _validate_worker_runtime_file(
    path: Path, *, manifest: Mapping[str, Any], model_label: str,
    shard_id: int, expected_rows: int
) -> dict[str, Any]:
    runtime = _read_json(path, label="worker runtime")
    observation = runtime.get("prebound_runtime_observation")
    _require(isinstance(observation, Mapping), "worker prebound runtime evidence missing")
    _validate_worker_runtime_observation(
        observation,
        manifest=manifest,
        physical_gpu_index=shard_id,
    )
    _require(
        runtime.get("model_label") == model_label
        and runtime.get("shard_id") == shard_id
        and runtime.get("physical_gpu_index") == shard_id
        and runtime.get("completed_rows") == expected_rows
        and runtime.get("generation_contract") == manifest["generation"],
        "worker terminal runtime scope drift",
    )
    for key in ("gpu_before_load", "gpu_after_load", "gpu_before_shutdown"):
        record = runtime.get(key)
        expected_gpu = manifest["runtime_prebinding"]["physical_gpu_inventory"][shard_id]
        _require(
            isinstance(record, Mapping)
            and record.get("uuid") == expected_gpu["uuid"]
            and record.get("memory_total_mib") == expected_gpu["memory_mib"],
            f"worker runtime GPU binding drift: {key}",
        )
    return runtime


@contextlib.contextmanager
def _gpu_lease(index: int) -> Iterable[dict[str, Any]]:
    path = Path(GPU_LOCK_TEMPLATE.format(index=index))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StochasticDecisionEvaluationError(f"GPU{index} canonical lock is held") from exc
        yield {"path": str(path), "acquired_at_utc": _utc_now()}
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


async def _generate_chunk(
    *, engine: Any, sampling_params_cls: Any, chunk: Sequence[Mapping[str, Any]],
    token_by_id: Mapping[str, Mapping[str, Any]], on_result: Any
) -> None:
    async def consume(case: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any, str]:
        params = sampling_params_cls(
            n=1,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            max_tokens=MAX_NEW_TOKENS,
            seed=int(case["row_seed"]),
            detokenize=True,
            skip_special_tokens=False,
            spaces_between_special_tokens=True,
            ignore_eos=False,
        )
        request_id = (
            f"decision-{int(case['absolute_case_index']):04d}-"
            f"seed-{int(case['row_seed'])}"
        )
        final = None
        async for output in engine.generate(
            {"prompt_token_ids": token_by_id[str(case["sample_id"])]["prompt_token_ids"]},
            params,
            request_id,
        ):
            final = output
        _require(final is not None and len(final.outputs) == 1, "vLLM request did not finish")
        _require(
            list(final.prompt_token_ids)
            == token_by_id[str(case["sample_id"])]["prompt_token_ids"],
            "vLLM prompt-token drift",
        )
        return case, final.outputs[0], request_id

    tasks = [asyncio.create_task(consume(case)) for case in chunk]
    try:
        for future in asyncio.as_completed(tasks):
            case, completion, request_id = await future
            await on_result(case, completion, request_id)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _shutdown_vllm_v1_engine(engine: Any) -> None:
    """Close the pinned vLLM V1 engine through its public teardown API."""

    shutdown = getattr(engine, "shutdown", None)
    _require(callable(shutdown), "vLLM V1 engine shutdown API drift")
    result = shutdown()
    _require(result is None, "unexpected asynchronous vLLM shutdown result")


def worker(
    *, manifest_path: Path, manifest_sha256: str,
    authorization_path: Path | None, authorization_sha256: str | None,
    model_label: str, shard_id: int, physical_gpu_index: int, resume: bool,
    scope: str = "formal",
) -> dict[str, Any]:
    """Generate one model/shard in a dedicated vLLM process."""

    _require(shard_id in (0, 1) and physical_gpu_index == shard_id, "worker/GPU mapping drift")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(physical_gpu_index), "CUDA visibility drift")
    for key, expected in REQUIRED_VLLM_ENV.items():
        _require(os.environ.get(key) == expected, f"required environment drift: {key}")
    _require(scope in {"formal", "smoke"}, "invalid worker scope")
    manifest = validate_manifest(manifest_path, manifest_sha256, deep_models=False)
    if scope == "formal":
        _require(
            authorization_path is not None and bool(authorization_sha256),
            "formal worker requires authorization",
        )
        authorization = _validate_authorization(
            authorization_path,
            str(authorization_sha256),
            manifest_path,
            manifest_sha256,
            deep_smoke=False,
        )
    else:
        _require(
            authorization_path is None and authorization_sha256 is None,
            "smoke must precede and must not carry formal authorization",
        )
        authorization = None
    root = manifest_path.parent
    samples, token_by_id, cases = _load_prepared(manifest, root)
    del samples
    model = _model_by_label(manifest, model_label)
    selected = _worker_cases(cases, model_label, shard_id, scope=scope)
    paths = _worker_paths(root, model_label, shard_id, scope=scope)
    launch_payload = _sealed(
        {
            "schema_version": WORKER_RECEIPT_SCHEMA,
            "status": "launched",
            "created_at_utc": _utc_now(),
            "evaluation_id": EVALUATION_ID,
            "manifest_sha256": manifest_sha256,
            "evaluation_scope": scope,
            "authorization_sha256": authorization_sha256,
            "authorization_scope": (
                authorization["authorized_scope"] if authorization is not None else None
            ),
            "model_label": model_label,
            "model_sha256": model["sha256"],
            "shard_id": shard_id,
            "physical_gpu_index": physical_gpu_index,
            "expected_cases": len(selected),
            "generation": copy.deepcopy(manifest["generation"]),
        }
    )
    if not paths["root"].exists():
        paths["partial"].mkdir(parents=True)
        _write_exclusive_json(paths["launch"], launch_payload)
        for name in ("wal", "receipts"):
            _write_exclusive_empty(paths[name])
    else:
        _require(resume, f"worker output exists without --resume: {paths['root']}")
        existing_launch = _read_json(paths["launch"], label="worker launch")
        _validate_integrity(existing_launch, label="worker launch")
        for key in launch_payload:
            if key not in {"created_at_utc", "integrity"}:
                _require(existing_launch.get(key) == launch_payload.get(key), f"worker launch drift: {key}")

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    _require(vllm.__version__ == EXPECTED_VLLM_VERSION, "vLLM version drift")
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "worker requires one visible GPU")
    expected_gpu = manifest["runtime_prebinding"]["physical_gpu_inventory"][physical_gpu_index]
    _require(torch.cuda.get_device_name(0) == expected_gpu["name"], "visible GPU identity drift")
    runtime_observation = _worker_runtime_observation(
        manifest, physical_gpu_index
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(_model_path(model)), local_files_only=True, trust_remote_code=False
    )
    eos_ids = _bound_eos_ids(model, tokenizer)
    rows_list = _read_wal_jsonl_recover_torn_final(
        paths["wal"], label="worker WAL"
    )
    by_case = {int(case["absolute_case_index"]): case for case in selected}
    rows: dict[int, dict[str, Any]] = {}
    for row in rows_list:
        index = int(row.get("absolute_case_index", -1))
        _require(index in by_case and index not in rows, "duplicate/out-of-scope WAL row")
        rows[index] = _validate_raw_row(
            row,
            case=by_case[index],
            token=token_by_id[str(by_case[index]["sample_id"])],
            model=model,
            manifest_sha256=manifest_sha256,
            shard_id=shard_id,
            physical_gpu_index=physical_gpu_index,
            tokenizer=tokenizer,
            eos_ids=eos_ids,
        )
    chunks = _chunks(selected)
    receipt_rows = _read_wal_jsonl_recover_torn_final(
        paths["receipts"], label="chunk receipts"
    )
    _require(len(receipt_rows) <= len(chunks), "too many chunk receipts")
    for chunk_id, receipt in enumerate(receipt_rows):
        _require(receipt == _chunk_receipt(chunk_id, chunks[chunk_id], rows), "chunk receipt drift")
    for chunk_id, chunk in enumerate(chunks[: len(receipt_rows)]):
        _require(all(int(case["absolute_case_index"]) in rows for case in chunk), "receipted chunk incomplete")
    while len(receipt_rows) < len(chunks):
        chunk_id = len(receipt_rows)
        chunk = chunks[chunk_id]
        if not all(int(case["absolute_case_index"]) in rows for case in chunk):
            break
        recovered_receipt = _chunk_receipt(chunk_id, chunk, rows)
        _append_fsync(paths["receipts"], recovered_receipt)
        receipt_rows.append(recovered_receipt)
    canonical_rows = [rows[int(case["absolute_case_index"])] for case in selected if int(case["absolute_case_index"]) in rows]
    if len(rows) == len(selected):
        _require(
            len(receipt_rows) == len(chunks),
            "complete worker WAL lacks complete chunk receipts",
        )
        _require(
            paths["runtime_progress"].is_file(),
            "complete WAL lacks crash-recovery runtime progress",
        )
        if paths["runtime"].is_file():
            recovered_runtime = _read_json(paths["runtime"], label="worker runtime")
        else:
            recovered_runtime = _read_json(
                paths["runtime_progress"], label="worker runtime progress"
            )
            recovered_runtime.update(
                {
                    "recovered_terminalization_without_redispatch": True,
                    "recovered_at_utc": _utc_now(),
                    "gpu_before_shutdown": _nvidia_memory(physical_gpu_index),
                    "completed_at_utc": _utc_now(),
                    "completed_rows": len(rows),
                }
            )
        _finalize_worker(
            paths=paths,
            root=root,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            model=model,
            shard_id=shard_id,
            scope=scope,
            authorization_path=authorization_path,
            canonical_rows=canonical_rows,
            selected_cases=selected,
            runtime=recovered_runtime,
        )
        return {
            "status": "already_complete",
            "worker_receipt": _file_record(paths["receipt"]),
        }
    _require(
        not paths["receipt"].exists()
        and not paths["canonical"].exists()
        and not paths["runtime"].exists(),
        "terminal worker artifact exists before WAL closure",
    )

    if paths["runtime_progress"].is_file():
        runtime = _read_json(
            paths["runtime_progress"], label="worker runtime progress"
        )
        prior_observation = runtime.get("prebound_runtime_observation")
        _require(
            isinstance(prior_observation, Mapping),
            "runtime progress lacks prebound observation",
        )
        _validate_worker_runtime_observation(
            prior_observation,
            manifest=manifest,
            physical_gpu_index=physical_gpu_index,
        )
        runtime["resume_count"] = int(runtime.get("resume_count", 0)) + 1
        runtime["resumed_rows"] = len(rows)
        runtime["current_session_started_at_utc"] = _utc_now()
        runtime["gpu_before_load"] = _nvidia_memory(physical_gpu_index)
    else:
        runtime = {
            "schema_version": WORKER_RECEIPT_SCHEMA,
            "started_at_utc": _utc_now(),
            "prebound_runtime_observation": runtime_observation,
            "model_label": model_label,
            "shard_id": shard_id,
            "physical_gpu_index": physical_gpu_index,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "generation_contract": copy.deepcopy(manifest["generation"]),
            "gpu_before_load": _nvidia_memory(physical_gpu_index),
            "resumed_rows": len(rows),
            "resume_count": 0,
        }
        _atomic_json(paths["runtime_progress"], runtime)

    async def execute() -> None:
        args = AsyncEngineArgs(
            model=str(_model_path(model)),
            tokenizer=str(_model_path(model)),
            tokenizer_mode="auto",
            trust_remote_code=False,
            dtype="bfloat16",
            quantization=None,
            load_format="safetensors",
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            swap_space=0,
            cpu_offload_gb=0,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            enforce_eager=False,
            disable_custom_all_reduce=True,
            generation_config="vllm",
            seed=0,
            disable_log_requests=True,
            disable_log_stats=True,
        )
        loaded = time.perf_counter()
        engine = AsyncLLMEngine.from_engine_args(args)
        runtime["model_load_wall_seconds"] = time.perf_counter() - loaded
        runtime["gpu_after_load"] = _nvidia_memory(physical_gpu_index)
        _atomic_json(paths["runtime_progress"], runtime)
        started = time.perf_counter()
        new_tokens = 0
        new_rows = 0
        try:
            for chunk_id, chunk in enumerate(chunks):
                missing = [
                    case for case in chunk if int(case["absolute_case_index"]) not in rows
                ]
                if missing:
                    async def on_result(case: Mapping[str, Any], completion: Any, request_id: str) -> None:
                        nonlocal new_tokens, new_rows
                        raw = _raw_worker_row(
                            case=case,
                            token=token_by_id[str(case["sample_id"])],
                            manifest_sha256=manifest_sha256,
                            model=model,
                            shard_id=shard_id,
                            physical_gpu_index=physical_gpu_index,
                            generated_token_ids=list(completion.token_ids),
                            generated_text=str(completion.text),
                            finish_reason=completion.finish_reason,
                            stop_reason=completion.stop_reason,
                            request_id=request_id,
                        )
                        _validate_raw_row(
                            raw,
                            case=case,
                            token=token_by_id[str(case["sample_id"])],
                            model=model,
                            manifest_sha256=manifest_sha256,
                            shard_id=shard_id,
                            physical_gpu_index=physical_gpu_index,
                            tokenizer=tokenizer,
                            eos_ids=eos_ids,
                        )
                        index = int(case["absolute_case_index"])
                        _require(index not in rows, "duplicate vLLM completion")
                        _append_fsync(paths["wal"], raw)
                        rows[index] = raw
                        new_tokens += len(raw["generated_token_ids_raw"])
                        new_rows += 1
                        runtime["new_rows_current_session"] = new_rows
                        runtime["new_output_tokens_current_session"] = new_tokens
                        runtime["last_completion_at_utc"] = _utc_now()
                        _atomic_json(paths["runtime_progress"], runtime)
                        _atomic_json(
                            paths["state"],
                            _sealed(
                                {
                                    "status": "running",
                                    "updated_at_utc": _utc_now(),
                                    "model_label": model_label,
                                    "shard_id": shard_id,
                                    "completed_cases": len(rows),
                                    "expected_cases": len(selected),
                                }
                            ),
                        )
                    await _generate_chunk(
                        engine=engine,
                        sampling_params_cls=SamplingParams,
                        chunk=missing,
                        token_by_id=token_by_id,
                        on_result=on_result,
                    )
                expected_receipt = _chunk_receipt(chunk_id, chunk, rows)
                if chunk_id < len(receipt_rows):
                    _require(receipt_rows[chunk_id] == expected_receipt, "existing receipt drift")
                else:
                    _append_fsync(paths["receipts"], expected_receipt)
                    receipt_rows.append(expected_receipt)
        finally:
            # With ``VLLM_USE_V1=1`` in vLLM 0.8.5.post1,
            # ``AsyncLLMEngine.from_engine_args`` returns an ``AsyncLLM``
            # instance.  Its public teardown method is ``shutdown()``; the
            # legacy V0-only ``shutdown_background_loop()`` is unavailable.
            _shutdown_vllm_v1_engine(engine)
        runtime["generation_wall_seconds"] = time.perf_counter() - started
        runtime["new_rows"] = new_rows
        runtime["new_output_tokens"] = new_tokens
        runtime["output_tokens_per_second"] = (
            new_tokens / runtime["generation_wall_seconds"]
            if runtime["generation_wall_seconds"] > 0
            else None
        )
        runtime["gpu_before_shutdown"] = _nvidia_memory(physical_gpu_index)
        _atomic_json(paths["runtime_progress"], runtime)

    with _gpu_lease(physical_gpu_index) as lease:
        runtime["gpu_lease"] = lease
        asyncio.run(execute())
    _require(len(rows) == len(selected), "worker result closure drift")
    canonical_rows = [rows[int(case["absolute_case_index"])] for case in selected]
    runtime["completed_at_utc"] = _utc_now()
    runtime["completed_rows"] = len(rows)
    _atomic_json(paths["runtime_progress"], runtime)
    _finalize_worker(
        paths=paths,
        root=root,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        model=model,
        shard_id=shard_id,
        scope=scope,
        authorization_path=authorization_path,
        canonical_rows=canonical_rows,
        selected_cases=selected,
        runtime=runtime,
    )
    return {"status": "complete", "worker_receipt": _file_record(paths["receipt"])}


def _material_result(
    *, raw: Mapping[str, Any], sample: Mapping[str, Any], model: Mapping[str, Any],
    tokenizer: Any, eos_ids: set[int], manifest_sha256: str, final_protocol: Any
) -> dict[str, Any]:
    prior_temperature = final_protocol.TEMPERATURE
    try:
        final_protocol.TEMPERATURE = TEMPERATURE
        result = final_protocol._result_row(
            manifest_sha=manifest_sha256,
            merged_sha=str(model["sha256"]),
            sample=sample,
            mode="sampled",
            generation_index=int(raw["replicate_id"]) + 1,
            batch_index=int(raw["paired_block_id"]),
            seed=int(raw["row_seed"]),
            raw_ids=raw["generated_token_ids_raw"],
            eos_ids=eos_ids,
            tokenizer=tokenizer,
        )
    finally:
        final_protocol.TEMPERATURE = prior_temperature
    result.update(
        {
            "schema_version": RESULT_ROW_SCHEMA,
            "model_label": model["label"],
            "model_display_name": model["display_name"],
            "paper_model": model["paper_model"],
            "training_state": model["training_state"],
            "checkpoint_step": model["checkpoint_step"],
            "panel": sample["panel"],
            "split": "external_evaluation",
            "meeting_id": sample["meeting_id"],
            "meeting_start_date": sample["meeting_start_date"],
            "result_origin": "fresh_vllm_stochastic_generation",
            "absolute_case_index": raw["absolute_case_index"],
            "paired_block_id": raw["paired_block_id"],
            "meeting_rank": raw["meeting_rank"],
            "model_rank": raw["model_rank"],
            "replicate_id": raw["replicate_id"],
            "replicate_seed": raw["replicate_seed"],
            "row_seed": raw["row_seed"],
            "shard_id": raw["shard_id"],
            "physical_gpu_index": raw["physical_gpu_index"],
            "prompt_token_ids_sha256": raw["prompt_token_ids_sha256"],
            "vllm_finish_reason": raw["vllm_finish_reason"],
            "vllm_stop_reason": raw["vllm_stop_reason"],
            "vllm_request_id": raw["request_id"],
            "raw_worker_row_sha256": _sha256_text(canonical_json(raw)),
        }
    )
    result["generation_parameters"] = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "request_level_seed": raw["row_seed"],
    }
    return result


def _load_worker_rows(
    *, root: Path, manifest: Mapping[str, Any], manifest_sha256: str,
    authorization_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples, token_by_id, cases = _load_prepared(manifest, root)
    sample_by_id = {str(row["sample_id"]): row for row in samples}
    case_by_index = {int(row["absolute_case_index"]): row for row in cases}
    from jobs.retrain_v2 import evaluate_chk4_external_core8_panels as source
    final_protocol = source._final_protocol()
    tokenizer = source._load_tokenizer(source.MODELS[-1])
    eos_by_model = {
        model["label"]: source._eos_ids_for_model(model, tokenizer, final_protocol)
        for model in source.MODELS
    }
    raw_rows = []
    worker_receipts = []
    for model in manifest["models"]:
        for shard_id in range(SHARD_COUNT):
            paths = _worker_paths(root, str(model["label"]), shard_id)
            rows = _read_jsonl(paths["canonical"], label="canonical raw worker output")
            selected = _worker_cases(cases, str(model["label"]), shard_id)
            _require(len(rows) == len(selected), "canonical worker row count drift")
            _validate_worker_receipt(
                paths=paths,
                root=root,
                manifest=manifest,
                manifest_sha256=manifest_sha256,
                model=model,
                shard_id=shard_id,
                scope="formal",
                authorization_path=authorization_path,
                canonical_rows=rows,
                selected_cases=selected,
            )
            for raw, case in zip(rows, selected, strict=True):
                _validate_raw_row(
                    raw,
                    case=case,
                    token=token_by_id[str(case["sample_id"])],
                    model=model,
                    manifest_sha256=manifest_sha256,
                    shard_id=shard_id,
                    physical_gpu_index=shard_id,
                    tokenizer=tokenizer,
                    eos_ids=eos_by_model[str(model["label"])],
                )
                raw_rows.append(raw)
            worker_receipts.append(_file_record(paths["receipt"], relative_to=root))
    _require(len(raw_rows) == EXPECTED_RAW_ROWS, "raw worker union count drift")
    _require(
        {int(row["absolute_case_index"]) for row in raw_rows} == set(range(EXPECTED_RAW_ROWS)),
        "raw worker exact-union drift",
    )
    raw_rows.sort(key=lambda row: int(row["absolute_case_index"]))
    results = []
    for raw in raw_rows:
        case = case_by_index[int(raw["absolute_case_index"])]
        model = _model_by_label(manifest, str(raw["model_label"]))
        results.append(
            _material_result(
                raw=raw,
                sample=sample_by_id[str(case["sample_id"])],
                model=model,
                tokenizer=tokenizer,
                eos_ids=eos_by_model[str(model["label"])],
                manifest_sha256=manifest_sha256,
                final_protocol=final_protocol,
            )
        )
    return results, worker_receipts


def _load_smoke_rows(
    *, root: Path, manifest: Mapping[str, Any], manifest_sha256: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Deep-replay both smoke shards from their fsynced raw-token evidence."""

    samples, token_by_id, cases = _load_prepared(manifest, root)
    sample_by_id = {str(row["sample_id"]): row for row in samples}
    from jobs.retrain_v2 import evaluate_chk4_external_core8_panels as source

    final_protocol = source._final_protocol()
    tokenizer = source._load_tokenizer(source.MODELS[-1])
    model = _model_by_label(manifest, SMOKE_MODEL_LABEL)
    replay_model = {**model, "path": _model_path(model)}
    eos_ids = source._eos_ids_for_model(replay_model, tokenizer, final_protocol)
    raw_rows: list[dict[str, Any]] = []
    worker_receipts = []
    for shard_id in range(SHARD_COUNT):
        paths = _worker_paths(
            root, SMOKE_MODEL_LABEL, shard_id, scope="smoke"
        )
        rows = _read_jsonl(
            paths["canonical"], label="canonical smoke raw output"
        )
        selected = _worker_cases(
            cases, SMOKE_MODEL_LABEL, shard_id, scope="smoke"
        )
        _require(len(rows) == len(selected), "smoke shard count drift")
        _validate_worker_receipt(
            paths=paths,
            root=root,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            model=model,
            shard_id=shard_id,
            scope="smoke",
            authorization_path=None,
            canonical_rows=rows,
            selected_cases=selected,
        )
        for raw, case in zip(rows, selected, strict=True):
            _validate_raw_row(
                raw,
                case=case,
                token=token_by_id[str(case["sample_id"])],
                model=model,
                manifest_sha256=manifest_sha256,
                shard_id=shard_id,
                physical_gpu_index=shard_id,
                tokenizer=tokenizer,
                eos_ids=eos_ids,
            )
            raw_rows.append(raw)
        worker_receipts.append(_file_record(paths["receipt"], relative_to=root))
    _require(
        len(raw_rows) == SMOKE_ROWS
        and Counter(row["shard_id"] for row in raw_rows) == Counter({0: 10, 1: 10})
        and len({row["absolute_case_index"] for row in raw_rows}) == SMOKE_ROWS,
        "smoke exact-union drift",
    )
    raw_rows.sort(key=lambda row: int(row["absolute_case_index"]))
    material = [
        _material_result(
            raw=raw,
            sample=sample_by_id[str(raw["sample_id"])],
            model=model,
            tokenizer=tokenizer,
            eos_ids=eos_ids,
            manifest_sha256=manifest_sha256,
            final_protocol=final_protocol,
        )
        for raw in raw_rows
    ]
    return material, raw_rows, worker_receipts


def _validate_smoke_receipt(
    path: Path, *, manifest_path: Path, manifest_sha256: str
) -> dict[str, Any]:
    value = _read_json(path, label="infrastructure smoke receipt")
    _validate_integrity(value, label="infrastructure smoke receipt")
    _require(
        value.get("schema_version") == SMOKE_RECEIPT_SCHEMA
        and value.get("status") == "passed"
        and value.get("manifest")
        == {"path": str(manifest_path.resolve()), "sha256": manifest_sha256}
        and value.get("rows") == SMOKE_ROWS
        and value.get("shards") == {"shard0": 10, "shard1": 10},
        "infrastructure smoke role drift",
    )
    required_gates = {
        "exact_formal_engine_and_sampling_contract": True,
        "exact_default_unfixed_prompt_token_ids": True,
        "returned_prompt_token_identity": True,
        "two_gpu_receipts": True,
        "exact_twenty_row_union": True,
        "normal_finish_reason_and_token_volume": True,
        "full_raw_token_replay": True,
        "source_v4_generation_rows_reused_zero": True,
    }
    _require(value.get("gates") == required_gates, "smoke gates drift")
    manifest = validate_manifest(manifest_path, manifest_sha256, deep_models=False)
    material, raw, worker_receipts = _load_smoke_rows(
        root=manifest_path.parent,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )
    root = manifest_path.parent
    _require(value.get("workers") == worker_receipts, "smoke worker binding drift")
    _require(
        value.get("raw_results")
        == _file_record(root / "smoke/raw_results.jsonl", relative_to=root)
        and value.get("material_results")
        == _file_record(root / "smoke/results.jsonl", relative_to=root)
        and _read_jsonl(root / "smoke/raw_results.jsonl", label="smoke raw assembly")
        == raw
        and _read_jsonl(root / "smoke/results.jsonl", label="smoke material assembly")
        == material,
        "smoke assembled-result replay drift",
    )
    return value


def smoke(
    manifest_path: Path, manifest_sha256: str, *, resume: bool
) -> dict[str, Any]:
    """Run the create-only 20-case, two-GPU infrastructure gate."""

    manifest = validate_manifest(manifest_path, manifest_sha256, deep_models=True)
    root = manifest_path.parent
    _require(
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
        return {"status": "already_passed", "smoke_receipt": _file_record(receipt_path), "rows": value["rows"]}
    _launch_worker_pair(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        authorization_path=None,
        authorization_sha256=None,
        model_label=SMOKE_MODEL_LABEL,
        resume=resume,
        scope="smoke",
    )
    material, raw, worker_receipts = _load_smoke_rows(
        root=root,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )
    _require(
        all(
            row["vllm_finish_reason"] in {"stop", "length"}
            and 1 <= len(row["generated_token_ids_raw"]) <= MAX_NEW_TOKENS
            for row in raw
        ),
        "smoke finish/token-volume gate failed",
    )
    raw_path = root / "smoke/raw_results.jsonl"
    result_path = root / "smoke/results.jsonl"
    _write_or_validate_jsonl(raw_path, raw, label="assembled smoke raw results")
    _write_or_validate_jsonl(
        result_path, material, label="assembled smoke material results"
    )
    receipt_payload = {
            "schema_version": SMOKE_RECEIPT_SCHEMA,
            "status": "passed",
            "created_at_utc": _utc_now(),
            "evaluation_id": EVALUATION_ID,
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": manifest_sha256,
            },
            "scope": "two_meetings_times_ten_replicates_one_model_two_gpus",
            "model_label": SMOKE_MODEL_LABEL,
            "meeting_ranks": list(SMOKE_MEETING_RANKS),
            "rows": len(raw),
            "shards": {"shard0": 10, "shard1": 10},
            "workers": worker_receipts,
            "raw_results": _file_record(raw_path, relative_to=root),
            "material_results": _file_record(result_path, relative_to=root),
            "source_v4_generation_rows_reused": 0,
            "gates": {
                "exact_formal_engine_and_sampling_contract": True,
                "exact_default_unfixed_prompt_token_ids": True,
                "returned_prompt_token_identity": True,
                "two_gpu_receipts": True,
                "exact_twenty_row_union": True,
                "normal_finish_reason_and_token_volume": True,
                "full_raw_token_replay": True,
                "source_v4_generation_rows_reused_zero": True,
            },
        }
    if receipt_path.exists():
        existing = _read_json(receipt_path, label="infrastructure smoke receipt")
        receipt_payload["created_at_utc"] = existing.get("created_at_utc")
        _require(existing == _sealed(receipt_payload), "existing smoke receipt drift")
    else:
        _write_exclusive_json(receipt_path, _sealed(receipt_payload))
    _validate_smoke_receipt(
        receipt_path,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )
    return {
        "status": "passed",
        "smoke_receipt": _file_record(receipt_path),
        "rows": SMOKE_ROWS,
    }


def _launch_worker_pair(
    *, manifest_path: Path, manifest_sha256: str,
    authorization_path: Path | None, authorization_sha256: str | None,
    model_label: str, resume: bool, scope: str = "formal"
) -> None:
    root = manifest_path.parent
    _require(scope in {"formal", "smoke"}, "invalid launch scope")
    logs = root / ("run/logs" if scope == "formal" else "smoke/logs")
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    handles = []
    try:
        for shard_id in range(SHARD_COUNT):
            log_path = logs / f"{model_label}.shard{shard_id}.log"
            handle = log_path.open("a", encoding="utf-8")
            handles.append(handle)
            env = {
                **os.environ,
                **REQUIRED_VLLM_ENV,
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(shard_id),
                "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
            command = [
                str(VLLM_PYTHON),
                "-u",
                "-m",
                "jobs.retrain_v2.evaluate_chk4_external_core8_stochastic_vllm_k10",
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
                command.extend(
                    [
                        "--authorization",
                        str(authorization_path),
                        "--authorization-sha256",
                        str(authorization_sha256),
                    ]
                )
            if resume:
                command.append("--resume")
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        failures = []
        for shard_id, process in enumerate(processes):
            code = process.wait()
            if code:
                failures.append((shard_id, code))
        _require(not failures, f"vLLM worker failures for {model_label}: {failures}")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for handle in handles:
            handle.close()


@contextlib.contextmanager
def _run_lock(root: Path) -> Iterable[None]:
    path = root / ".stochastic_vllm_run.lock"
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StochasticDecisionEvaluationError("stochastic run lock is held") from exc
        yield
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _expected_runtime_summary(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    runtime_rows = []
    for model in manifest["models"]:
        for shard_id in range(SHARD_COUNT):
            paths = _worker_paths(root, str(model["label"]), shard_id)
            runtime_rows.append(
                _validate_worker_runtime_file(
                    paths["runtime"],
                    manifest=manifest,
                    model_label=str(model["label"]),
                    shard_id=shard_id,
                    expected_rows=EXPECTED_BLOCKS_PER_SHARD,
                )
            )
    return {
        "workers": runtime_rows,
        "gpu_memory_utilization_configured": GPU_MEMORY_UTILIZATION,
        "headroom_fraction": 1.0 - GPU_MEMORY_UTILIZATION,
    }


def _validate_run_receipt(
    *, manifest_path: Path, manifest_sha256: str,
    authorization_path: Path, authorization_sha256: str,
    manifest: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = (
        validate_manifest(manifest_path, manifest_sha256, deep_models=True)
        if manifest is None
        else manifest
    )
    _validate_authorization(
        authorization_path,
        authorization_sha256,
        manifest_path,
        manifest_sha256,
    )
    root = manifest_path.parent
    receipt_path = root / "run/run_receipt.json"
    receipt = _read_json(receipt_path, label="run receipt")
    _validate_integrity(receipt, label="run receipt")
    _require(
        receipt.get("schema_version") == RUN_RECEIPT_SCHEMA
        and receipt.get("status") == "complete"
        and receipt.get("evaluation_id") == EVALUATION_ID
        and receipt.get("manifest") == _file_record(manifest_path)
        and receipt.get("authorization") == _file_record(authorization_path)
        and receipt.get("rows") == EXPECTED_RAW_ROWS
        and receipt.get("panel_rows") == EXPECTED_PANEL_ROWS
        and receipt.get("source_v4_generation_rows_reused") == 0,
        "run receipt role/population drift",
    )
    results, worker_receipts = _load_worker_rows(
        root=root,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        authorization_path=authorization_path,
    )
    _require(
        len(results) == EXPECTED_RAW_ROWS
        and len({int(row["absolute_case_index"]) for row in results})
        == EXPECTED_RAW_ROWS
        and Counter(row["panel"] for row in results) == Counter(EXPECTED_PANEL_ROWS),
        "run receipt exact 930-row union drift",
    )
    result_path = root / "run/results.jsonl"
    _require(
        receipt.get("workers") == worker_receipts
        and receipt.get("results") == _file_record(result_path, relative_to=root)
        and _read_jsonl(result_path, label="assembled material results") == results,
        "run receipt worker/material replay drift",
    )
    runtime_path = root / "run/runtime_summary.json"
    expected_runtime = _expected_runtime_summary(root, manifest)
    _require(
        receipt.get("runtime") == _file_record(runtime_path, relative_to=root)
        and _read_json(runtime_path, label="runtime summary") == expected_runtime,
        "run receipt runtime replay drift",
    )
    return receipt, results


def run(
    manifest_path: Path, manifest_sha256: str, *, authorization_path: Path,
    authorization_sha256: str, resume: bool
) -> dict[str, Any]:
    manifest = validate_manifest(manifest_path, manifest_sha256, deep_models=True)
    _validate_authorization(
        authorization_path, authorization_sha256, manifest_path, manifest_sha256
    )
    root = manifest_path.parent
    with _run_lock(root):
        receipt_path = root / "run/run_receipt.json"
        if receipt_path.exists():
            _validate_run_receipt(
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
                authorization_path=authorization_path,
                authorization_sha256=authorization_sha256,
                manifest=manifest,
            )
            return {"status": "already_complete", "run_receipt": _file_record(receipt_path)}
        for model in manifest["models"]:
            _launch_worker_pair(
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
                authorization_path=authorization_path,
                authorization_sha256=authorization_sha256,
                model_label=str(model["label"]),
                resume=resume,
            )
        results, worker_receipts = _load_worker_rows(
            root=root,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            authorization_path=authorization_path,
        )
        _require(len(results) == EXPECTED_RAW_ROWS, "material result closure drift")
        result_path = root / "run/results.jsonl"
        _write_or_validate_jsonl(
            result_path, results, label="assembled material results"
        )
        runtime_path = root / "run/runtime_summary.json"
        _write_or_validate_json(
            runtime_path,
            _expected_runtime_summary(root, manifest),
            label="runtime summary",
        )
        receipt_payload = {
            "schema_version": RUN_RECEIPT_SCHEMA,
            "status": "complete",
            "created_at_utc": _utc_now(),
            "evaluation_id": EVALUATION_ID,
            "manifest": _file_record(manifest_path),
            "authorization": _file_record(authorization_path),
            "workers": worker_receipts,
            "results": _file_record(result_path, relative_to=root),
            "runtime": _file_record(runtime_path, relative_to=root),
            "rows": len(results),
            "panel_rows": dict(Counter(row["panel"] for row in results)),
            "source_v4_generation_rows_reused": 0,
        }
        if receipt_path.exists():
            existing = _read_json(receipt_path, label="run receipt")
            receipt_payload["created_at_utc"] = existing.get("created_at_utc")
            _require(existing == _sealed(receipt_payload), "existing run receipt drift")
        else:
            _write_exclusive_json(receipt_path, _sealed(receipt_payload))
        _validate_run_receipt(
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            authorization_path=authorization_path,
            authorization_sha256=authorization_sha256,
            manifest=manifest,
        )
    return {
        "status": "generation_complete",
        "results": _file_record(root / "run/results.jsonl", relative_to=root),
        "run_receipt": _file_record(root / "run/run_receipt.json", relative_to=root),
        "rows": EXPECTED_RAW_ROWS,
    }


def _prediction_label(row: Mapping[str, Any]) -> str:
    prediction = row.get("decision_prediction")
    if isinstance(prediction, Mapping) and prediction.get("direction") in DIRECTIONS:
        return str(prediction["direction"])
    return "invalid"


def meeting_vectors(
    rows: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, dict[str, Any]]]:
    sample_by_id = {str(row["sample_id"]): row for row in samples}
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model_label"]), str(row["sample_id"]))].append(row)
    expected = {
        (str(model["label"]), str(sample["sample_id"]))
        for model in models
        for sample in samples
    }
    _require(set(groups) == expected, "meeting-vector population drift")
    for (model_label, sample_id), values in groups.items():
        values = sorted(values, key=lambda row: int(row["replicate_id"]))
        _require(
            [row["replicate_id"] for row in values] == list(range(REPLICATES)),
            "meeting replicate closure drift",
        )
        sample = sample_by_id[sample_id]
        labels = [_prediction_label(row) for row in values]
        counts = Counter(labels)
        largest = max(counts.values())
        winners = sorted(label for label, count in counts.items() if count == largest)
        probabilities = [counts[label] / REPLICATES for label in PREDICTION_LABELS]
        entropy = -sum(p * math.log(p) for p in probabilities if p > 0) / math.log(4)
        result[model_label][sample_id] = {
            "sample_id": sample_id,
            "panel": sample["panel"],
            "target_direction": sample["direction"],
            "correctness": [bool(row["decision_direction_correct"]) for row in values],
            "delivery": [bool(row["delivery_valid"]) for row in values],
            "labels": labels,
            "correctness_rate": statistics.fmean(bool(row["decision_direction_correct"]) for row in values),
            "delivery_rate": statistics.fmean(bool(row["delivery_valid"]) for row in values),
            "modal_label": winners[0] if len(winners) == 1 else None,
            "modal_tie": len(winners) > 1,
            "modal_share": largest / REPLICATES,
            "modal_correct": len(winners) == 1 and winners[0] == sample["direction"],
            "normalized_four_label_entropy": entropy,
            "unanimous": largest == REPLICATES,
        }
    # A request seed is shared across all model states for every meeting/replicate.
    for sample in samples:
        sample_rows = [row for row in rows if row["sample_id"] == sample["sample_id"]]
        for replicate_id in range(REPLICATES):
            block = [row for row in sample_rows if row["replicate_id"] == replicate_id]
            _require(
                len(block) == MODEL_COUNT and len({row["row_seed"] for row in block}) == 1,
                "cross-model request-seed pairing drift",
            )
    return {key: dict(value) for key, value in result.items()}


def _metric_from_meetings(
    meetings: Sequence[Mapping[str, Any]], *, panel: str
) -> dict[str, Any]:
    _require(bool(meetings), "cannot score empty meeting block")
    per_class = {}
    for direction in DIRECTIONS:
        values = [row for row in meetings if row["target_direction"] == direction]
        per_class[direction] = {
            "support_meetings": len(values),
            "expected_correct_completions": sum(
                sum(bool(value) for value in row["correctness"]) for row in values
            ),
            "total_completions": len(values) * REPLICATES,
            "recall": statistics.fmean(row["correctness_rate"] for row in values)
            if values
            else None,
        }
    supported = [direction for direction in DIRECTIONS if per_class[direction]["support_meetings"]]
    primary = (
        DIRECTIONS if panel == "historical_n19" else tuple(direction for direction in DIRECTIONS if direction in {"cut", "hold"})
    )
    _require(all(per_class[direction]["support_meetings"] for direction in primary), "primary BA class missing")
    return {
        "meetings": len(meetings),
        "replicates_per_meeting": REPLICATES,
        "generated_completions": len(meetings) * REPLICATES,
        "meeting_averaged_direction_accuracy": statistics.fmean(row["correctness_rate"] for row in meetings),
        "meeting_averaged_delivery_rate": statistics.fmean(row["delivery_rate"] for row in meetings),
        "per_class": per_class,
        "balanced_accuracy": statistics.fmean(per_class[direction]["recall"] for direction in primary),
        "balanced_accuracy_classes": list(primary),
        "fixed_three_class_balanced_accuracy": (
            statistics.fmean(per_class[direction]["recall"] for direction in DIRECTIONS)
            if len(supported) == 3
            else None
        ),
        "modal_accuracy": statistics.fmean(bool(row["modal_correct"]) for row in meetings),
        "modal_tie_rate": statistics.fmean(bool(row["modal_tie"]) for row in meetings),
        "mean_modal_share": statistics.fmean(row["modal_share"] for row in meetings),
        "mean_normalized_four_label_entropy": statistics.fmean(
            row["normalized_four_label_entropy"] for row in meetings
        ),
        "unanimous_rate": statistics.fmean(bool(row["unanimous"]) for row in meetings),
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def hierarchical_bootstrap(
    vectors: Mapping[str, Mapping[str, Mapping[str, Any]]],
    samples: Sequence[Mapping[str, Any]], models: Sequence[Mapping[str, Any]],
    *, panel: str, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    panel_samples = [row for row in samples if row["panel"] == panel]
    by_class = {
        direction: [str(row["sample_id"]) for row in panel_samples if row["direction"] == direction]
        for direction in DIRECTIONS
    }
    supported = [direction for direction in DIRECTIONS if by_class[direction]]
    rng = random.Random(seed + PANELS.index(panel))
    model_labels = [str(model["label"]) for model in models]
    distributions = {
        model: {metric: [] for metric in ("accuracy", "balanced_accuracy", "delivery")}
        for model in model_labels
    }
    pair_distributions: dict[str, dict[str, list[float]]] = {}
    for left, right in itertools.combinations(model_labels, 2):
        pair_distributions[f"{left}__minus__{right}"] = {
            metric: [] for metric in ("accuracy", "balanced_accuracy", "delivery")
        }
    ledger = []
    for draw_id in range(draws):
        plan = []
        selected_by_class: dict[str, list[tuple[str, list[int]]]] = {}
        for direction in supported:
            population = by_class[direction]
            selected = []
            for _ in population:
                sample_id = population[rng.randrange(len(population))]
                replicate_indexes = [rng.randrange(REPLICATES) for _ in range(REPLICATES)]
                selected.append((sample_id, replicate_indexes))
                plan.append(
                    {
                        "target_direction": direction,
                        "sample_id": sample_id,
                        "replicate_indexes": replicate_indexes,
                    }
                )
            selected_by_class[direction] = selected
        draw_metrics = {}
        for model in model_labels:
            class_accuracy = {}
            class_delivery = {}
            for direction, selected in selected_by_class.items():
                accuracy_values = []
                delivery_values = []
                for sample_id, replicate_indexes in selected:
                    vector = vectors[model][sample_id]
                    accuracy_values.append(
                        statistics.fmean(vector["correctness"][index] for index in replicate_indexes)
                    )
                    delivery_values.append(
                        statistics.fmean(vector["delivery"][index] for index in replicate_indexes)
                    )
                class_accuracy[direction] = statistics.fmean(accuracy_values)
                class_delivery[direction] = statistics.fmean(delivery_values)
            total_meetings = sum(len(value) for value in selected_by_class.values())
            accuracy = sum(
                class_accuracy[direction] * len(selected_by_class[direction])
                for direction in supported
            ) / total_meetings
            delivery = sum(
                class_delivery[direction] * len(selected_by_class[direction])
                for direction in supported
            ) / total_meetings
            ba = statistics.fmean(class_accuracy[direction] for direction in supported)
            draw_metrics[model] = {
                "accuracy": accuracy,
                "balanced_accuracy": ba,
                "delivery": delivery,
            }
            for metric, value in draw_metrics[model].items():
                distributions[model][metric].append(value)
        contrasts = {}
        for left, right in itertools.combinations(model_labels, 2):
            key = f"{left}__minus__{right}"
            contrasts[key] = {}
            for metric in ("accuracy", "balanced_accuracy", "delivery"):
                value = draw_metrics[left][metric] - draw_metrics[right][metric]
                pair_distributions[key][metric].append(value)
                contrasts[key][metric] = value
        ledger.append(
            {
                "schema_version": BOOTSTRAP_SCHEMA,
                "panel": panel,
                "draw_id": draw_id,
                "draw_plan": plan,
                "model_metrics": draw_metrics,
                "paired_contrasts": contrasts,
            }
        )
    def intervals(items: Mapping[str, Sequence[float]]) -> dict[str, Any]:
        return {
            metric: {
                "lower": _percentile(values, 0.025),
                "upper": _percentile(values, 0.975),
            }
            for metric, values in items.items()
        }
    return {
        "method": "class_stratified_paired_two_stage_hierarchical_percentile_bootstrap",
        "draws": draws,
        "seed": seed + PANELS.index(panel),
        "meeting_draws_within_target_class": True,
        "replicate_draws_within_meeting": True,
        "draws_shared_across_models": True,
        "model_intervals": {model: intervals(values) for model, values in distributions.items()},
        "contrast_intervals": {key: intervals(values) for key, values in pair_distributions.items()},
    }, ledger


def exact_vector_label_swap(
    meetings_a: Sequence[Mapping[str, Any]], meetings_b: Sequence[Mapping[str, Any]],
    *, metric: str
) -> dict[str, Any]:
    by_a = {str(row["sample_id"]): row for row in meetings_a}
    by_b = {str(row["sample_id"]): row for row in meetings_b}
    _require(set(by_a) == set(by_b) and bool(by_a), "label-swap pairing drift")
    ids = sorted(by_a)
    if metric == "accuracy":
        differences = [by_a[key]["correctness_rate"] - by_b[key]["correctness_rate"] for key in ids]
        weights = [1.0 / len(ids)] * len(ids)
    elif metric == "delivery":
        differences = [by_a[key]["delivery_rate"] - by_b[key]["delivery_rate"] for key in ids]
        weights = [1.0 / len(ids)] * len(ids)
    elif metric == "balanced_accuracy":
        differences = [by_a[key]["correctness_rate"] - by_b[key]["correctness_rate"] for key in ids]
        counts = Counter(str(by_a[key]["target_direction"]) for key in ids)
        weights = [1.0 / (len(counts) * counts[str(by_a[key]["target_direction"])]) for key in ids]
    else:
        raise StochasticDecisionEvaluationError(f"unknown label-swap metric: {metric}")
    observed = sum(weight * difference for weight, difference in zip(weights, differences, strict=True))
    extreme = 0
    total = 2 ** len(ids)
    threshold = abs(observed) - 1e-12
    for mask in range(total):
        value = 0.0
        for index, (weight, difference) in enumerate(zip(weights, differences, strict=True)):
            value += weight * difference * (1.0 if mask & (1 << index) else -1.0)
        extreme += int(abs(value) >= threshold)
    return {
        "test": "exact_full_meeting_vector_model_label_swap_two_sided",
        "metric": metric,
        "meetings": len(ids),
        "assignments": total,
        "difference_a_minus_b": observed,
        "p_value_raw": extreme / total,
    }


def exact_modal_mcnemar(
    meetings_a: Sequence[Mapping[str, Any]], meetings_b: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_a = {str(row["sample_id"]): row for row in meetings_a}
    by_b = {str(row["sample_id"]): row for row in meetings_b}
    _require(set(by_a) == set(by_b) and bool(by_a), "modal McNemar pairing drift")
    a_only = b_only = both = neither = 0
    for key in sorted(by_a):
        a, b = bool(by_a[key]["modal_correct"]), bool(by_b[key]["modal_correct"])
        if a and b:
            both += 1
        elif a:
            a_only += 1
        elif b:
            b_only += 1
        else:
            neither += 1
    discordant = a_only + b_only
    if discordant:
        tail = sum(math.comb(discordant, value) for value in range(min(a_only, b_only) + 1)) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    else:
        p_value = 1.0
    return {
        "test": "exact_modal_decision_mcnemar_two_sided_ties_counted_incorrect",
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "both_correct": both,
        "neither_correct": neither,
        "discordant": discordant,
        "p_value_raw": p_value,
    }


def holm_adjust(rows: Sequence[Mapping[str, Any]], *, family: str) -> list[dict[str, Any]]:
    ordered = sorted(enumerate(rows), key=lambda item: (float(item[1]["p_value_raw"]), item[0]))
    adjusted = [0.0] * len(rows)
    running = 0.0
    for rank, (index, row) in enumerate(ordered):
        running = max(running, min(1.0, (len(rows) - rank) * float(row["p_value_raw"])))
        adjusted[index] = running
    return [
        {**dict(row), "p_value_holm": adjusted[index], "holm_family": family, "holm_family_size": len(rows)}
        for index, row in enumerate(rows)
    ]


def _markdown_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Stochastic Decision Diagnostics (T=0.6, top-p=0.9, K=10)",
        "",
        "This report uses ten fresh stochastic completions per model and meeting. Meetings, not generations, are the independent statistical units; N19 and N12 are reported separately.",
        "",
    ]
    for panel, title in (("historical_n19", "Historical N=19"), ("postcutoff_n12", "Post-cutoff N=12")):
        lines += [
            f"## {title}",
            "",
            "| Model/state | Meeting-averaged direction accuracy | Balanced accuracy | Delivery | Modal accuracy | Modal ties | Mean modal share | Entropy | Unanimous |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for label, block in summary["panels"][panel]["models"].items():
            display = block["display_name"]
            lines.append(
                f"| {display} | {100*block['meeting_averaged_direction_accuracy']:.2f}% | {100*block['balanced_accuracy']:.2f}% | {100*block['meeting_averaged_delivery_rate']:.2f}% | {100*block['modal_accuracy']:.2f}% | {100*block['modal_tie_rate']:.2f}% | {100*block['mean_modal_share']:.2f}% | {block['mean_normalized_four_label_entropy']:.3f} | {100*block['unanimous_rate']:.2f}% |"
            )
        lines += [
            "",
            "The hierarchical bootstrap resamples meetings within target class and replicate indices within each selected meeting, with every draw shared across models. Exact tests swap complete ten-replicate meeting vectors, never individual generations.",
            "",
        ]
    lines += [
        "## Interpretation constraints",
        "",
        "- K=10 measures decoding uncertainty; it does not enlarge N=19 or N=12.",
        "- N12 has no hike meeting, so its balanced accuracy averages cut and hold recall only.",
        "- The panels are retrospective and use deterministic Core8 source-analysis inputs rather than the opened teacher-compressed N=13 input contract.",
        "- Model chk-3 GRPO cp450 remains the best-observed exploratory candidate used for evaluation; selected_checkpoint_step is null and quality is not demonstrated.",
        "- No pooled N=31 estimate is reported.",
        "",
    ]
    return "\n".join(lines)


def _latex_table(summary: Mapping[str, Any]) -> str:
    lines = [
        "% Generated by evaluate_chk4_external_core8_stochastic_vllm_k10.py; do not edit.",
        "\\begin{table}[H]",
        "    \\centering",
        "    \\scriptsize",
        "    \\caption{Stochastic Decision Diagnostics at $T=0.6$, $p=0.9$, and $K=10$}",
        "    \\label{tab:ch2:decision_external_core8_stochastic_k10}",
        "    \\begin{tabular}{llrrrr}",
        "        \\toprule",
        "        Panel & Model/state & Direction & Balanced & Delivery & Modal direction \\\\",
        "        \\midrule",
    ]
    for panel in PANELS:
        for index, block in enumerate(summary["panels"][panel]["models"].values()):
            label = "Historical $N=19$" if panel == "historical_n19" else "Post-cutoff $N=12$"
            lines.append(
                "        "
                + " & ".join(
                    [
                        label if index == 0 else "",
                        block["display_name"],
                        f"{100*block['meeting_averaged_direction_accuracy']:.2f}\\%",
                        f"{100*block['balanced_accuracy']:.2f}\\%",
                        f"{100*block['meeting_averaged_delivery_rate']:.2f}\\%",
                        f"{100*block['modal_accuracy']:.2f}\\%",
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
        "    \\footnotesize \\textit{Notes:} Values average the ten completions within each meeting before averaging across meetings. The historical panel uses three-class balanced accuracy; the post-cutoff panel averages cut and hold recall because it contains no hikes. $K=10$ quantifies decoding variation and does not increase the independent meeting count.",
        "    \\end{minipage}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def score(
    manifest_path: Path, manifest_sha256: str, *, authorization_path: Path,
    authorization_sha256: str
) -> dict[str, Any]:
    manifest = validate_manifest(manifest_path, manifest_sha256, deep_models=True)
    _validate_authorization(
        authorization_path, authorization_sha256, manifest_path, manifest_sha256
    )
    root = manifest_path.parent
    receipt, results = _validate_run_receipt(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        authorization_path=authorization_path,
        authorization_sha256=authorization_sha256,
        manifest=manifest,
    )
    samples = _read_jsonl(root / manifest["inputs"]["samples"]["path"], label="samples")
    vectors = meeting_vectors(results, samples, manifest["models"])
    summary_panels = {}
    bootstrap_rows = []
    for panel in PANELS:
        panel_ids = [str(row["sample_id"]) for row in samples if row["panel"] == panel]
        model_blocks = {}
        meetings_by_model = {}
        for model in manifest["models"]:
            label = str(model["label"])
            meetings = [vectors[label][sample_id] for sample_id in panel_ids]
            meetings_by_model[label] = meetings
            block = _metric_from_meetings(meetings, panel=panel)
            block["display_name"] = model["display_name"]
            model_blocks[label] = block
        bootstrap, ledger = hierarchical_bootstrap(
            vectors, samples, manifest["models"], panel=panel
        )
        bootstrap_rows.extend(ledger)
        accuracy_ba_tests = []
        delivery_tests = []
        modal_tests = []
        for left, right in itertools.combinations(
            [str(model["label"]) for model in manifest["models"]], 2
        ):
            for metric in ("accuracy", "balanced_accuracy"):
                accuracy_ba_tests.append(
                    {
                        "model_a": left,
                        "model_b": right,
                        **exact_vector_label_swap(
                            meetings_by_model[left], meetings_by_model[right], metric=metric
                        ),
                    }
                )
            delivery_tests.append(
                {
                    "model_a": left,
                    "model_b": right,
                    **exact_vector_label_swap(
                        meetings_by_model[left], meetings_by_model[right], metric="delivery"
                    ),
                }
            )
            modal_tests.append(
                {
                    "model_a": left,
                    "model_b": right,
                    **exact_modal_mcnemar(meetings_by_model[left], meetings_by_model[right]),
                }
            )
        summary_panels[panel] = {
            "models": model_blocks,
            "bootstrap": bootstrap,
            "accuracy_and_balanced_accuracy_tests": holm_adjust(
                accuracy_ba_tests,
                family=f"{panel}:six_accuracy_and_balanced_accuracy_tests",
            ),
            "delivery_tests": holm_adjust(
                delivery_tests, family=f"{panel}:three_delivery_tests"
            ),
            "modal_mcnemar_tests": holm_adjust(
                modal_tests, family=f"{panel}:three_modal_mcnemar_tests"
            ),
        }
    summary = _sealed(
        {
            "schema_version": SUMMARY_SCHEMA,
            "status": "complete",
            "created_at_utc": _utc_now(),
            "evaluation_manifest_sha256": manifest_sha256,
            "run_receipt": _file_record(root / "run/run_receipt.json", relative_to=root),
            "rows": EXPECTED_RAW_ROWS,
            "independent_unit": "meeting",
            "replicates_nested_within_meeting": True,
            "pooled_panel_result": None,
            "panels": summary_panels,
            "checkpoint_governance": copy.deepcopy(manifest["checkpoint_governance"]),
            "interpretation": "stochastic_decoding_robustness_diagnostic_not_new_training_or_prospective_validation",
        }
    )
    report_root = root / "report"
    _require(not report_root.exists() and not report_root.is_symlink(), "report output exists")
    report_root.mkdir()
    _write_exclusive_json(report_root / "summary.json", summary)
    _write_exclusive_jsonl(report_root / "bootstrap_ledger.jsonl", bootstrap_rows)
    _write_exclusive_text(report_root / "technical_report.md", _markdown_report(summary))
    _write_exclusive_text(report_root / "results_table.tex", _latex_table(summary))
    _write_exclusive_json(
        report_root / "report_receipt.json",
        _sealed(
            {
                "status": "complete",
                "summary": _file_record(report_root / "summary.json", relative_to=root),
                "bootstrap_ledger": _file_record(
                    report_root / "bootstrap_ledger.jsonl", relative_to=root
                ),
                "technical_report": _file_record(
                    report_root / "technical_report.md", relative_to=root
                ),
                "latex_table": _file_record(report_root / "results_table.tex", relative_to=root),
            }
        ),
    )
    return {
        "status": "scored",
        "summary": _file_record(report_root / "summary.json", relative_to=root),
        "technical_report": _file_record(report_root / "technical_report.md", relative_to=root),
        "rows": EXPECTED_RAW_ROWS,
    }


def status(output_root: Path) -> dict[str, Any]:
    root = output_root.resolve()
    worker_status = {}
    completed = 0
    wal_rows = 0
    for model_label in (
        "model_chk1_cp200",
        "model_chk3_sft_cp38",
        "model_chk3_grpo_cp450",
    ):
        worker_status[model_label] = {}
        for shard_id in range(SHARD_COUNT):
            paths = _worker_paths(root, model_label, shard_id)
            count = 0
            if paths["wal"].is_file():
                with paths["wal"].open("rb") as handle:
                    count = sum(bool(line.strip()) for line in handle)
            done = paths["receipt"].is_file()
            completed += int(done)
            wal_rows += count
            worker_status[model_label][f"shard{shard_id}"] = {
                "wal_rows": count,
                "expected_rows": EXPECTED_BLOCKS_PER_SHARD,
                "complete": done,
            }
    return {
        "output_root": str(root),
        "prepared": (root / "evaluation_manifest.json").is_file(),
        "authorized": (root / "formal_generation_authorization.json").is_file(),
        "worker_receipts": completed,
        "expected_worker_receipts": MODEL_COUNT * SHARD_COUNT,
        "wal_rows": wal_rows,
        "expected_rows": EXPECTED_RAW_ROWS,
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
    args = _parser().parse_args(argv)
    manifest_path = args.manifest or args.output_root / "evaluation_manifest.json"
    if args.stage == "prepare":
        result = prepare(args.output_root)
    elif args.stage == "status":
        result = status(args.output_root)
    else:
        _require(bool(args.manifest_sha256), "--manifest-sha256 is required")
        if args.stage == "authorize":
            _require(bool(args.audit_statement), "--audit-statement is required")
            result = authorize(
                manifest_path,
                args.manifest_sha256,
                audit_statement=args.audit_statement,
            )
        elif args.stage == "smoke":
            _require(
                args.authorization is None and args.authorization_sha256 is None,
                "smoke must not carry formal authorization",
            )
            result = smoke(
                manifest_path,
                args.manifest_sha256,
                resume=args.resume,
            )
        else:
            if args.stage == "worker":
                _require(bool(args.model_label), "--model-label is required")
                _require(args.shard_id is not None, "--shard-id is required")
                _require(
                    args.physical_gpu_index is not None,
                    "--physical-gpu-index is required",
                )
                result = worker(
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
            elif args.stage == "run":
                _require(args.authorization is not None, "--authorization is required")
                _require(
                    bool(args.authorization_sha256),
                    "--authorization-sha256 is required",
                )
                result = run(
                    manifest_path,
                    args.manifest_sha256,
                    authorization_path=args.authorization,
                    authorization_sha256=args.authorization_sha256,
                    resume=args.resume,
                )
            else:
                _require(args.authorization is not None, "--authorization is required")
                _require(
                    bool(args.authorization_sha256),
                    "--authorization-sha256 is required",
                )
                result = score(
                    manifest_path,
                    args.manifest_sha256,
                    authorization_path=args.authorization,
                    authorization_sha256=args.authorization_sha256,
                )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
