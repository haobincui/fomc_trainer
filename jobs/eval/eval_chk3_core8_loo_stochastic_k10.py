"""Generate the versioned CHK3 cp318 Core8 LOO N128 x K10 cohort.

This runner consumes the already sealed 2,176-row Core8 LOO input matrix.  It
does not alter the historical greedy run.  Every meeting has one full prompt
and sixteen interventions; within a meeting and replicate all seventeen arms
share the row seed derived from the meeting's frozen full-sample ID.  That
common-random-number design reduces decoding noise in paired LOO contrasts.

GPU execution is fail-closed on physical GPU1.  ``CUDA_DEVICE_ORDER`` must be
``PCI_BUS_ID`` and ``CUDA_VISIBLE_DEVICES`` must be exactly ``1`` before the
process starts.  Logical ``cuda:0`` is then UUID-checked against physical GPU1.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jobs.eval import eval_chk3_core8_loo_full as full_profile
from jobs.eval import eval_chk3_core8_loo_smoke as core8
from jobs.eval import eval_chk3_native_three_model as native_eval
from jobs.eval import eval_chk3_stochastic_bootstrap_generation as stochastic
from open_r1.validator import loo_generation_spec as loo_spec
from open_r1.validator.loo_generation_spec import (
    derive_row_seed,
    seal_manifest,
    validate_manifest_integrity,
)


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ID = "chk3-cp318-core8-loo-stochastic-n128-k10-1993-2008-v1"
ROW_SCHEMA = "chk3-core8-loo-stochastic-k10-generation-row-v1"
LAUNCH_SCHEMA = "chk3-core8-loo-stochastic-k10-launch-v1"
STATE_SCHEMA = "chk3-core8-loo-stochastic-k10-state-v1"
RUN_SCHEMA = "chk3-core8-loo-stochastic-k10-run-v1"
MODEL_ID = "chk3"
MODEL_LABEL = "chk3-cp318-exact-merged-core8-loo-stochastic-k10"
EXPECTED_SAMPLES = 2176
EXPECTED_MEETINGS = 128
REPLICATE_SEEDS = (
    20260811,
    21260811,
    22260811,
    23260811,
    24260811,
    25260811,
    26260811,
    27260811,
    28260811,
    29260811,
)
EXPECTED_CASES = EXPECTED_SAMPLES * len(REPLICATE_SEEDS)
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 50
REPETITION_PENALTY = 1.0
MAX_NEW_TOKENS = 2560
TAIL_TOKENS = 1024
ATTN_IMPLEMENTATION = "sdpa"
PHYSICAL_GPU_INDEX = 1
GPU_LOCK = Path("/tmp/fomc_trainer_chk3_core8_loo_stochastic_k10_gpu1.lock")
DEFAULT_SAMPLE_MANIFEST = ROOT / (
    "output/evaluation/main/"
    "chk3_cp318_core8_loo_full_n128_1993_2008_20260814_v1/"
    "inputs_v1/samples.json"
)
DEFAULT_SAMPLE_MANIFEST_SHA256 = (
    "5ddaf97ef9ee7d957ca65a977d6527ea5cedc93e2b22fc15bc08627ea9e91efe"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Core8StochasticK10Error(RuntimeError):
    """The frozen stochastic Core8 contract cannot be satisfied."""


class ExternalGpu1ProcessError(Core8StochasticK10Error):
    """An external process appeared on GPU1 after the model was loaded."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise Core8StochasticK10Error(f"non-canonical JSON value: {exc}") from exc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise Core8StochasticK10Error(f"missing regular JSON file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Core8StochasticK10Error(f"invalid JSON file {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise Core8StochasticK10Error(f"JSON root is not an object: {resolved}")
    return value


def _read_canonical_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[bytes]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise Core8StochasticK10Error(f"missing regular JSONL file: {resolved}")
    rows: list[dict[str, Any]] = []
    encoded_lines: list[bytes] = []
    with resolved.open("rb") as handle:
        for line_number, encoded in enumerate(handle, 1):
            if not encoded.endswith(b"\n") or not encoded.strip():
                raise Core8StochasticK10Error(
                    f"non-canonical/blank JSONL line: {resolved}:{line_number}"
                )
            try:
                value = json.loads(encoded.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise Core8StochasticK10Error(
                    f"invalid JSONL line {resolved}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise Core8StochasticK10Error(
                    f"JSONL row is not an object: {resolved}:{line_number}"
                )
            expected = (_canonical_json(value) + "\n").encode("utf-8")
            if encoded != expected:
                raise Core8StochasticK10Error(
                    f"JSONL line is not canonical: {resolved}:{line_number}"
                )
            rows.append(value)
            encoded_lines.append(encoded)
    return rows, encoded_lines


def _file_binding(
    path: Path,
    *,
    rows: int | None = None,
    sha256: str | None = None,
    byte_count: int | None = None,
    payload_sha256: str | None = None,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise Core8StochasticK10Error(f"cannot bind missing regular file: {resolved}")
    binding: dict[str, Any] = {
        "path": str(resolved),
        "sha256": sha256 or _sha256_file(resolved),
        "bytes": byte_count if byte_count is not None else resolved.stat().st_size,
    }
    if rows is not None:
        binding["rows"] = rows
    if payload_sha256 is not None:
        binding["payload_sha256"] = payload_sha256
    return binding


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Core8StochasticK10Error(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise Core8StochasticK10Error(
                f"refusing to overwrite artifact: {path}"
            ) from exc
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_gpu_environment() -> str:
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise Core8StochasticK10Error(
            "CUDA_DEVICE_ORDER must be exactly PCI_BUS_ID before launch"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "1":
        raise Core8StochasticK10Error(
            "CUDA_VISIBLE_DEVICES must be exactly 1; refusing to expose GPU0"
        )
    return visible


def _external_gpu1_compute_processes() -> list[dict[str, Any]]:
    command = (
        "nvidia-smi",
        "--id=1",
        "--query-compute-apps=pid,process_name",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Core8StochasticK10Error(
            f"cannot inspect physical GPU1 with nvidia-smi: {exc}"
        ) from exc
    processes: list[dict[str, Any]] = []
    for raw in completed.stdout.splitlines():
        line = raw.strip()
        if not line or "No running processes" in line or line.startswith("N/A"):
            continue
        columns = [part.strip() for part in line.split(",", 1)]
        if not columns[0].isdigit():
            raise Core8StochasticK10Error(
                f"unexpected nvidia-smi GPU1 process row: {line!r}"
            )
        pid = int(columns[0])
        if pid != os.getpid():
            processes.append(
                {"pid": pid, "process_name": columns[1] if len(columns) > 1 else ""}
            )
    return processes


def _require_no_external_gpu1_processes(*, phase: str) -> None:
    """Fail closed whenever an unleased process is observed on physical GPU1."""

    processes = _external_gpu1_compute_processes()
    if processes:
        raise ExternalGpu1ProcessError(
            f"external GPU1 process detected {phase}: {processes}"
        )


def _physical_gpu1_identity() -> dict[str, str]:
    command = (
        "nvidia-smi",
        "--id=1",
        "--query-gpu=uuid,pci.bus_id",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Core8StochasticK10Error(
            f"cannot bind physical GPU1 identity: {exc}"
        ) from exc
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise Core8StochasticK10Error("physical GPU1 identity is ambiguous")
    columns = [part.strip() for part in rows[0].split(",")]
    if len(columns) != 2 or not columns[0].startswith("GPU-"):
        raise Core8StochasticK10Error("invalid physical GPU1 identity")
    return {"uuid": columns[0], "pci_bus_id": columns[1].lower()}


def _normalise_gpu_uuid(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.hex()
    compact = re.sub(r"[^0-9a-f]", "", str(value).casefold())
    if len(compact) != 32:
        raise Core8StochasticK10Error(f"invalid CUDA UUID: {value!r}")
    return compact


def _verify_logical_cuda0_is_physical_gpu1(
    torch_module: Any, physical_identity: Mapping[str, str]
) -> str:
    if torch_module.cuda.device_count() != 1:
        raise Core8StochasticK10Error("exactly one logical CUDA device must be visible")
    properties = torch_module.cuda.get_device_properties(0)
    logical_uuid = getattr(properties, "uuid", None)
    expected = str(physical_identity["uuid"])
    if _normalise_gpu_uuid(logical_uuid) != _normalise_gpu_uuid(expected):
        raise Core8StochasticK10Error(
            "logical cuda:0 UUID does not match nvidia-smi physical GPU1"
        )
    return expected


@contextlib.contextmanager
def exclusive_gpu1_lease(
    *,
    lock_path: Path = GPU_LOCK,
    timeout_seconds: int,
    poll_seconds: int,
    on_wait: Callable[[str, Sequence[Mapping[str, Any]]], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Wait for a separate flock and two consecutive idle GPU1 polls."""

    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise Core8StochasticK10Error("GPU wait timeout and poll must be positive")
    lock_path = lock_path.expanduser().resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.is_symlink():
        raise Core8StochasticK10Error(f"GPU1 lock is a symlink: {lock_path}")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    handle = os.fdopen(descriptor, "a+", encoding="utf-8")
    started = time.monotonic()
    last_notice = -60.0
    acquired = False
    try:
        while not acquired:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                elapsed = time.monotonic() - started
                if elapsed >= timeout_seconds:
                    raise Core8StochasticK10Error(
                        "timed out waiting for the exclusive GPU1 flock"
                    )
                if on_wait is not None and elapsed - last_notice >= 60:
                    on_wait("waiting_for_gpu1_flock", [])
                    last_notice = elapsed
                time.sleep(min(poll_seconds, max(0.1, timeout_seconds - elapsed)))
        consecutive_idle = 0
        while consecutive_idle < 2:
            processes = _external_gpu1_compute_processes()
            elapsed = time.monotonic() - started
            if processes:
                consecutive_idle = 0
                if on_wait is not None and elapsed - last_notice >= 60:
                    on_wait("waiting_for_external_gpu1_processes", processes)
                    last_notice = elapsed
            else:
                consecutive_idle += 1
            if consecutive_idle < 2:
                if elapsed >= timeout_seconds:
                    raise Core8StochasticK10Error(
                        "timed out waiting for physical GPU1 to become idle"
                    )
                time.sleep(min(poll_seconds, max(0.1, timeout_seconds - elapsed)))
        yield {
            "lock_path": str(lock_path),
            "acquired_at_utc": _utc_now(),
            "wait_seconds": round(time.monotonic() - started, 3),
            "physical_gpu_index": PHYSICAL_GPU_INDEX,
            "external_compute_pids_at_acquire": [],
        }
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _sampling_contract() -> dict[str, Any]:
    return {
        "generation_mode": "stochastic_sampling",
        "do_sample": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "max_new_tokens": MAX_NEW_TOKENS,
        "tail_tokens": TAIL_TOKENS,
        "batch_size": 1,
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_storage": "bfloat16",
        "bnb_4bit_use_double_quant": True,
        "model_dtype": "bfloat16",
        "attn_implementation": ATTN_IMPLEMENTATION,
        "replicate_seeds": list(REPLICATE_SEEDS),
        "row_seed": "derive_row_seed(replicate_seed,paired_seed_key)",
        "paired_seed_key": "meeting_full_sample_id",
        "common_random_numbers": "same-meeting-replicate-across-all-17-arms-v1",
        "rng_reset": (
            "transformers.set_seed+torch.manual_seed+"
            "torch.cuda.manual_seed_all-per-tuple-v1"
        ),
        "canonical_tuple_order": "sealed_sample_order_then_replicate_id",
    }


def _load_inputs(
    sample_manifest_path: Path,
    expected_sha256: str,
    *,
    load_tokenizer: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], Any | None, str]:
    observed = _sha256_file(sample_manifest_path)
    if observed != expected_sha256:
        raise Core8StochasticK10Error(
            f"sample manifest SHA drift: expected={expected_sha256}, observed={observed}"
        )
    # ``eval_chk3_core8_loo_full`` is an older profile implemented by mutating
    # module globals in the smoke implementation.  Scope and restore those
    # globals so importing/running this new runner cannot alter the old smoke
    # contract or make its tests order-dependent.
    profile_names = (
        "CONFIG_SCHEMA",
        "SAMPLE_SCHEMA",
        "INPUT_ROW_SCHEMA",
        "RUN_SCHEMA",
        "STATE_SCHEMA",
        "SCORE_ROW_SCHEMA",
        "SCORE_SCHEMA",
        "EVALUATION_ID",
        "EXPECTED_MEETINGS",
        "EXPECTED_ROWS",
        "MODEL_LABEL",
        "DEFAULT_CONFIG",
        "GPU_LOCK",
        "EVALUATION_SCOPE",
        "PREPARE_LIMITATIONS",
        "RUN_LIMITATIONS",
        "SCORE_LIMITATIONS",
        "FAILURE_ROW_SCHEMA",
        "_select_meetings",
    )
    previous = {name: getattr(core8, name) for name in profile_names}
    try:
        full_profile.configure_implementation()
        manifest, input_rows, tokenizer = core8._load_sample_manifest(
            sample_manifest_path, expected_sha256, load_tokenizer=load_tokenizer
        )
    except core8.Core8LooSmokeError as exc:
        raise Core8StochasticK10Error(str(exc)) from exc
    finally:
        for name, value in previous.items():
            setattr(core8, name, value)
    samples = manifest.get("samples")
    selection = manifest.get("selection")
    if (
        not isinstance(samples, list)
        or len(samples) != EXPECTED_SAMPLES
        or len(input_rows) != EXPECTED_SAMPLES
        or not isinstance(selection, Mapping)
        or selection.get("meetings") != EXPECTED_MEETINGS
        or selection.get("variants_per_meeting") != 17
    ):
        raise Core8StochasticK10Error("sealed Core8 input matrix is not N128 x 17")
    return manifest, input_rows, tokenizer, observed


def _paired_seed_keys(samples: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    meetings: dict[str, list[Mapping[str, Any]]] = {}
    for sample in samples:
        meeting = sample.get("meeting_id")
        sample_id = sample.get("sample_id")
        if not isinstance(meeting, str) or not isinstance(sample_id, str):
            raise Core8StochasticK10Error("sample meeting/sample identity is invalid")
        meetings.setdefault(meeting, []).append(sample)
    if len(meetings) != EXPECTED_MEETINGS:
        raise Core8StochasticK10Error("meeting inventory drift")
    result: dict[str, str] = {}
    for meeting, variants in meetings.items():
        full = [row for row in variants if row.get("arm") == "full"]
        if len(variants) != 17 or len(full) != 1:
            raise Core8StochasticK10Error(
                f"meeting {meeting} does not have exactly 17 variants and one full"
            )
        result[meeting] = str(full[0]["sample_id"])
    return result


def _canonical_cases(
    samples: Sequence[Mapping[str, Any]], *, smoke: bool
) -> list[tuple[Mapping[str, Any], int, int, str]]:
    paired = _paired_seed_keys(samples)
    selected_samples = list(samples[:1] if smoke else samples)
    selected_seeds = REPLICATE_SEEDS[:1] if smoke else REPLICATE_SEEDS
    return [
        (sample, replicate_id, seed, paired[str(sample["meeting_id"])])
        for sample in selected_samples
        for replicate_id, seed in enumerate(selected_seeds)
    ]


def _case_key(
    case: tuple[Mapping[str, Any], int, int, str]
) -> tuple[str, str, int]:
    sample, replicate_id, _, _ = case
    return MODEL_ID, str(sample["sample_id"]), replicate_id


def _validate_tuple_prefix(
    rows: Sequence[Mapping[str, Any]],
    cases: Sequence[tuple[Mapping[str, Any], int, int, str]],
    *,
    complete: bool = False,
) -> None:
    if len(rows) > len(cases) or (complete and len(rows) != len(cases)):
        raise Core8StochasticK10Error("generation tuple count drift")
    seen: set[tuple[Any, ...]] = set()
    for index, row in enumerate(rows):
        observed = (row.get("model_id"), row.get("sample_id"), row.get("replicate_id"))
        if observed in seen:
            raise Core8StochasticK10Error("duplicate generation tuple")
        seen.add(observed)
        if observed != _case_key(cases[index]):
            raise Core8StochasticK10Error(
                f"out-of-order/missing generation tuple at index {index}"
            )


def _source_hashes(
    manifest: Mapping[str, Any], sample_manifest_sha256: str
) -> dict[str, Any]:
    tokenizer_files = manifest["tokenizer"]["files"]
    bindings = {
        name: {
            key: binding.get(key)
            for key in ("path", "sha256", "payload_sha256", "rows", "bytes")
            if binding.get(key) is not None
        }
        for name, binding in (
            ("input_dataset", manifest["dataset"]),
            ("source_release", manifest["source_release"]),
            ("source_panel", manifest["source_panel"]),
            ("model_anchor", manifest["model_anchor"]),
            ("exact_merge_evidence", manifest["exact_merge_evidence"]),
            ("config", manifest["config"]),
        )
    }
    return {
        "sample_manifest_sha256": sample_manifest_sha256,
        "sample_manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
        "bindings": bindings,
        "training_config_sha256": manifest["prompt_contract"][
            "training_config_sha256"
        ],
        "tokenizer_file_inventory_sha256": _sha256_text(
            _canonical_json(tokenizer_files)
        ),
        "implementation_sources": {
            "runner": _file_binding(Path(__file__).resolve()),
            "core8_input_contract": _file_binding(Path(str(core8.__file__)).resolve()),
            "full_profile": _file_binding(Path(str(full_profile.__file__)).resolve()),
            "native_generation_contract": _file_binding(
                Path(str(native_eval.__file__)).resolve()
            ),
            "row_seed_contract": _file_binding(Path(str(loo_spec.__file__)).resolve()),
        },
    }


def _load_anchor() -> dict[str, Any]:
    try:
        anchor = stochastic.load_frozen_anchor(MODEL_ID)
    except stochastic.StochasticBootstrapGenerationError as exc:
        raise Core8StochasticK10Error(str(exc)) from exc
    if anchor["model_label"] != "chk3-cp318-exact-merged":
        raise Core8StochasticK10Error("CHK3 anchor label drift")
    return anchor


def _build_result(
    *,
    text: str,
    generated_ids: Sequence[int],
    eos_ids: Sequence[int],
    pad_token_id: int,
    sample: Mapping[str, Any],
    input_row: Mapping[str, Any],
    replicate_id: int,
    replicate_seed: int,
    paired_seed_key: str,
    sample_manifest_sha256: str,
    source_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    row_seed = derive_row_seed(replicate_seed, paired_seed_key)
    try:
        base = native_eval.build_full_result(
            text=text,
            generated_token_ids=generated_ids,
            eos_token_ids=eos_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            tail_tokens=TAIL_TOKENS,
            source_prompt=str(input_row["prompt"]),
            source_analysis=str(input_row["source_analysis"]),
            reference_response=str(input_row["response"]),
            load_in_4bit=True,
            attn_implementation=ATTN_IMPLEMENTATION,
            pad_token_id=pad_token_id,
            stage_id=MODEL_ID,
            model_label=MODEL_LABEL,
            sample_manifest_sha256=sample_manifest_sha256,
            sample=sample,
            seed=row_seed,
        )
    except native_eval.NativeThreeModelEvalError as exc:
        raise Core8StochasticK10Error(str(exc)) from exc
    contract = _sampling_contract()
    row: dict[str, Any] = {
        **base,
        "schema_version": ROW_SCHEMA,
        "evaluation_id": EVALUATION_ID,
        "model_id": MODEL_ID,
        "variant_sample_id": sample["sample_id"],
        "meeting_id": input_row["meeting_id"],
        "arm": input_row["arm"],
        "intervention_topic": input_row["intervention_topic"],
        "replicate_id": replicate_id,
        "replicate_seed": replicate_seed,
        "paired_seed_key": paired_seed_key,
        "row_seed": row_seed,
        "generation_mode": "stochastic_sampling",
        "do_sample": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "batch_size": 1,
        "input_token_count": sample["prompt_token_count"],
        "input_truncated": False,
        "sampling_contract_sha256": _sha256_text(_canonical_json(contract)),
        "source_artifact_sha256s": dict(source_hashes),
        "full_source_analysis_sha256": input_row["full_source_analysis_sha256"],
        "full_reference_sha256": input_row["full_reference_sha256"],
        "full_prompt_token_count": input_row["full_prompt_token_count"],
        "semantic_status": "pending_offline",
    }
    try:
        generation_metrics, audit = stochastic._generation_metrics(row)
    except stochastic.StochasticBootstrapGenerationError as exc:
        raise Core8StochasticK10Error(str(exc)) from exc
    row["generation_metrics"] = generation_metrics
    row["preregistered_core_valid"] = bool(audit["preregistered_core_valid"])
    row["preregistered_core_failures"] = list(audit["preregistered_core_failures"])
    row["six_metrics"] = {
        **generation_metrics,
        "mpnet_cosine": None,
        "bertscore_f1": None,
    }
    return row


def _validate_result(
    row: Mapping[str, Any],
    *,
    sample: Mapping[str, Any],
    input_row: Mapping[str, Any],
    replicate_id: int,
    replicate_seed: int,
    paired_seed_key: str,
    sample_manifest_sha256: str,
    source_hashes: Mapping[str, Any],
    tokenizer: Any | None,
) -> None:
    row_seed = derive_row_seed(replicate_seed, paired_seed_key)
    expected = {
        "schema_version": ROW_SCHEMA,
        "evaluation_id": EVALUATION_ID,
        "model_id": MODEL_ID,
        "stage_id": MODEL_ID,
        "model_label": MODEL_LABEL,
        "sample_id": sample["sample_id"],
        "variant_sample_id": sample["sample_id"],
        "meeting_id": sample["meeting_id"],
        "arm": sample["arm"],
        "intervention_topic": sample["intervention_topic"],
        "replicate_id": replicate_id,
        "replicate_seed": replicate_seed,
        "paired_seed_key": paired_seed_key,
        "seed": row_seed,
        "row_seed": row_seed,
        "generation_mode": "stochastic_sampling",
        "do_sample": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "max_new_tokens": MAX_NEW_TOKENS,
        "tail_tokens": TAIL_TOKENS,
        "load_in_4bit": True,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "batch_size": 1,
        "input_token_count": sample["prompt_token_count"],
        "input_truncated": False,
        "sampling_contract_sha256": _sha256_text(
            _canonical_json(_sampling_contract())
        ),
        "source_artifact_sha256s": dict(source_hashes),
        "full_source_analysis_sha256": input_row["full_source_analysis_sha256"],
        "full_reference_sha256": input_row["full_reference_sha256"],
        "full_prompt_token_count": input_row["full_prompt_token_count"],
        "semantic_status": "pending_offline",
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise Core8StochasticK10Error(f"generation row drift at {key}")
    native_row = copy.deepcopy(dict(row))
    native_row["schema_version"] = native_eval.ROW_SCHEMA_VERSION
    native_row["do_sample"] = False
    try:
        native_eval.validate_full_result(
            native_row,
            stage_id=MODEL_ID,
            model_label=MODEL_LABEL,
            sample_manifest_sha256=sample_manifest_sha256,
            sample=sample,
            source_analysis=str(input_row["source_analysis"]),
            reference_response=str(input_row["response"]),
            tokenizer=tokenizer,
        )
    except native_eval.NativeThreeModelEvalError as exc:
        raise Core8StochasticK10Error(str(exc)) from exc
    try:
        metrics, audit = stochastic._generation_metrics(row)
    except stochastic.StochasticBootstrapGenerationError as exc:
        raise Core8StochasticK10Error(str(exc)) from exc
    if row.get("generation_metrics") != metrics:
        raise Core8StochasticK10Error("generation metric drift")
    if row.get("preregistered_core_valid") is not bool(
        audit["preregistered_core_valid"]
    ) or row.get("preregistered_core_failures") != list(
        audit["preregistered_core_failures"]
    ):
        raise Core8StochasticK10Error("core diagnostic drift")
    if row.get("six_metrics") != {
        **metrics,
        "mpnet_cosine": None,
        "bertscore_f1": None,
    }:
        raise Core8StochasticK10Error("six-metric placeholder drift")


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    cases: Sequence[tuple[Mapping[str, Any], int, int, str]],
    input_rows_by_id: Mapping[str, Mapping[str, Any]],
    sample_manifest_sha256: str,
    source_hashes: Mapping[str, Any],
    tokenizer: Any | None,
    complete: bool = False,
) -> None:
    _validate_tuple_prefix(rows, cases, complete=complete)
    for index, row in enumerate(rows):
        sample, replicate_id, replicate_seed, paired_seed_key = cases[index]
        _validate_result(
            row,
            sample=sample,
            input_row=input_rows_by_id[str(sample["sample_id"])],
            replicate_id=replicate_id,
            replicate_seed=replicate_seed,
            paired_seed_key=paired_seed_key,
            sample_manifest_sha256=sample_manifest_sha256,
            source_hashes=source_hashes,
            tokenizer=tokenizer,
        )


def _prefix_digest(lines: Sequence[bytes], count: int | None = None) -> tuple[str, int]:
    selected = lines if count is None else lines[:count]
    digest = hashlib.sha256()
    byte_count = 0
    for line in selected:
        digest.update(line)
        byte_count += len(line)
    return digest.hexdigest(), byte_count


def _completion_state(
    cases: Sequence[tuple[Mapping[str, Any], int, int, str]],
    completed: int,
    *,
    tuple_prefix_sha256: str | None = None,
) -> dict[str, Any]:
    if tuple_prefix_sha256 is None:
        tuple_digest = hashlib.sha256()
        for case in cases[:completed]:
            tuple_digest.update(
                (_canonical_json(_case_key(case)) + "\n").encode("utf-8")
            )
        tuple_prefix_sha256 = tuple_digest.hexdigest()
    current = None
    if completed and completed % len(REPLICATE_SEEDS):
        sample = cases[completed - 1][0]
        current = {
            "sample_id": sample["sample_id"],
            "replicate_ids": list(range(completed % len(REPLICATE_SEEDS))),
        }
    return {
        "canonical_prefix_rows": completed,
        "fully_completed_samples": completed // len(REPLICATE_SEEDS),
        "current_sample": current,
        "tuple_prefix_sha256": tuple_prefix_sha256,
    }


def _tuple_prefix_sha256s(
    cases: Sequence[tuple[Mapping[str, Any], int, int, str]],
) -> list[str]:
    """Precompute every canonical tuple-prefix digest once (O(number of cases))."""

    digest = hashlib.sha256()
    prefixes = [digest.hexdigest()]
    for case in cases:
        digest.update((_canonical_json(_case_key(case)) + "\n").encode("utf-8"))
        prefixes.append(digest.hexdigest())
    return prefixes


def _state_payload(
    *,
    status: str,
    completed: int,
    expected: int,
    resume_count: int,
    results_path: Path,
    results_sha256: str,
    results_bytes: int,
    cases: Sequence[tuple[Mapping[str, Any], int, int, str]],
    tuple_prefix_sha256: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return seal_manifest(
        {
            "schema_version": STATE_SCHEMA,
            "status": status,
            "updated_at_utc": _utc_now(),
            "evaluation_id": EVALUATION_ID,
            "model_id": MODEL_ID,
            "completed_cases": completed,
            "expected_cases": expected,
            "resume_count": resume_count,
            "completion_matrix": _completion_state(
                cases,
                completed,
                tuple_prefix_sha256=tuple_prefix_sha256,
            ),
            "results": _file_binding(
                results_path,
                rows=completed,
                sha256=results_sha256,
                byte_count=results_bytes,
            ),
            **extra,
        }
    )


def _validate_resume_state(
    state: Mapping[str, Any],
    *,
    lines: Sequence[bytes],
    active_path: Path,
    partial_path: Path,
    cases: Sequence[tuple[Mapping[str, Any], int, int, str]],
) -> tuple[int, int]:
    try:
        validate_manifest_integrity(state)
    except Exception as exc:
        raise Core8StochasticK10Error(f"resume state integrity failed: {exc}") from exc
    if (
        state.get("schema_version") != STATE_SCHEMA
        or state.get("evaluation_id") != EVALUATION_ID
        or state.get("model_id") != MODEL_ID
        or state.get("expected_cases") != len(cases)
    ):
        raise Core8StochasticK10Error("resume state contract drift")
    recorded = state.get("completed_cases")
    if (
        not isinstance(recorded, int)
        or isinstance(recorded, bool)
        or len(lines) not in {recorded, recorded + 1}
    ):
        raise Core8StochasticK10Error(
            "partial/state mismatch exceeds one-row fsync-before-state window"
        )
    binding = state.get("results")
    if not isinstance(binding, Mapping):
        raise Core8StochasticK10Error("resume state lacks results binding")
    bound_path = Path(str(binding.get("path"))).expanduser().resolve()
    active_resolved = active_path.expanduser().resolve()
    rename_window = (
        active_path.name == "generations.jsonl"
        and bound_path == partial_path.expanduser().resolve()
        and recorded == len(cases)
        and len(lines) == len(cases)
    )
    if bound_path != active_resolved and not rename_window:
        raise Core8StochasticK10Error("resume results path binding drift")
    prefix_sha, prefix_bytes = _prefix_digest(lines, recorded)
    if binding.get("sha256") != prefix_sha or binding.get("bytes") != prefix_bytes:
        raise Core8StochasticK10Error("resume results SHA/byte binding drift")
    if binding.get("rows") != recorded:
        raise Core8StochasticK10Error("resume results row binding drift")
    if state.get("completion_matrix") != _completion_state(cases, recorded):
        raise Core8StochasticK10Error("resume completion matrix drift")
    resume_count = state.get("resume_count")
    if not isinstance(resume_count, int) or isinstance(resume_count, bool):
        raise Core8StochasticK10Error("resume count is invalid")
    return resume_count + 1, recorded


def _summary(rows: Sequence[Mapping[str, Any]], *, smoke: bool) -> dict[str, Any]:
    failures: Counter[str] = Counter()
    for row in rows:
        failures.update(row["preregistered_core_failures"])
    return {
        "status": "complete",
        "scope": "one-row-one-seed-smoke" if smoke else "formal-n128-k10",
        "cases": len(rows),
        "expected_cases": 1 if smoke else EXPECTED_CASES,
        "unique_samples": len({row["sample_id"] for row in rows}),
        "unique_meetings": len({row["meeting_id"] for row in rows}),
        "replicate_ids": sorted({row["replicate_id"] for row in rows}),
        "input_truncation_cases": sum(bool(row["input_truncated"]) for row in rows),
        "completion_cap_cases": sum(bool(row["cap_reached"]) for row in rows),
        "empty_answer_cases": sum(not bool(row["answer"].strip()) for row in rows),
        "strict_periodic_tail_cases": sum(
            bool(row["strict_periodic_tail"]) for row in rows
        ),
        "generation_metric_rates": {
            metric: sum(bool(row["generation_metrics"][metric]) for row in rows)
            / len(rows)
            for metric in (
                "structure_delivery",
                "numeric_fidelity",
                "date_fidelity",
                "degeneration_free",
            )
        },
        "core_failure_counts": dict(sorted(failures.items())),
    }


def _validate_runtime_evidence(
    runtime_evidence: Any, launch_runtime: Any
) -> None:
    if not isinstance(runtime_evidence, Mapping) or not isinstance(
        launch_runtime, Mapping
    ):
        raise Core8StochasticK10Error("final state lacks GPU1 runtime evidence")
    physical = launch_runtime.get("physical_gpu_identity")
    lease = runtime_evidence.get("gpu_lease")
    if not isinstance(physical, Mapping) or not isinstance(lease, Mapping):
        raise Core8StochasticK10Error("final GPU1 runtime evidence is incomplete")
    if (
        runtime_evidence.get("cuda_device_order") != "PCI_BUS_ID"
        or runtime_evidence.get("cuda_visible_devices") != "1"
        or runtime_evidence.get("physical_gpu_index") != PHYSICAL_GPU_INDEX
        or runtime_evidence.get("logical_device") != "cuda:0"
        or runtime_evidence.get("physical_gpu_identity") != dict(physical)
        or runtime_evidence.get("logical_cuda0_uuid") != physical.get("uuid")
        or lease.get("physical_gpu_index") != PHYSICAL_GPU_INDEX
        or lease.get("lock_path") != str(GPU_LOCK.expanduser().resolve())
    ):
        raise Core8StochasticK10Error("final GPU1 runtime evidence drift")


def _deep_validate_artifacts(
    *,
    output_dir: Path,
    sample_manifest_path: Path,
    sample_manifest_sha256: str,
    smoke: bool,
    load_tokenizer: bool,
) -> dict[str, Any]:
    manifest, input_rows, tokenizer, observed_sha = _load_inputs(
        sample_manifest_path, sample_manifest_sha256, load_tokenizer=load_tokenizer
    )
    samples = manifest["samples"]
    cases = _canonical_cases(samples, smoke=smoke)
    source_hashes = _source_hashes(manifest, observed_sha)
    input_by_id = {str(row["sample_id"]): row for row in input_rows}
    run_path = output_dir / "manifest.json"
    final_path = output_dir / "generations.jsonl"
    state_path = output_dir / "state.progress.v1.json"
    launch_path = output_dir / "launch.json"
    for path in (run_path, final_path, state_path, launch_path):
        if not path.is_file() or path.is_symlink():
            raise Core8StochasticK10Error(f"missing sealed run artifact: {path}")
    launch = _read_json(launch_path)
    state = _read_json(state_path)
    run = _read_json(run_path)
    for label, value in (("launch", launch), ("state", state), ("run", run)):
        try:
            validate_manifest_integrity(value)
        except Exception as exc:
            raise Core8StochasticK10Error(f"{label} integrity failed: {exc}") from exc
    rows, lines = _read_canonical_jsonl(final_path)
    _validate_rows(
        rows,
        cases=cases,
        input_rows_by_id=input_by_id,
        sample_manifest_sha256=observed_sha,
        source_hashes=source_hashes,
        tokenizer=tokenizer,
        complete=True,
    )
    final_sha, final_bytes = _prefix_digest(lines)
    expected_binding = _file_binding(
        final_path,
        rows=len(rows),
        sha256=final_sha,
        byte_count=final_bytes,
    )
    if state.get("status") != "generation_complete" or state.get(
        "results"
    ) != expected_binding:
        raise Core8StochasticK10Error("sealed state/final generation binding drift")
    if run.get("status") != "complete" or run.get("generations") != expected_binding:
        raise Core8StochasticK10Error("run manifest/final generation binding drift")
    if run.get("state") != _file_binding(
        state_path, payload_sha256=state["integrity"]["payload_sha256"]
    ):
        raise Core8StochasticK10Error("run manifest/state binding drift")
    if run.get("launch") != _file_binding(
        launch_path, payload_sha256=launch["integrity"]["payload_sha256"]
    ):
        raise Core8StochasticK10Error("run manifest/launch binding drift")
    if run.get("summary") != _summary(rows, smoke=smoke):
        raise Core8StochasticK10Error("run summary drift")
    runtime_evidence = state.get("runtime_evidence")
    _validate_runtime_evidence(runtime_evidence, launch.get("runtime"))
    if run.get("runtime_evidence") != runtime_evidence:
        raise Core8StochasticK10Error("run/state runtime evidence drift")
    return {
        "status": "validated",
        "rows": len(rows),
        "sha256": final_sha,
        "bytes": final_bytes,
        "manifest": str(run_path),
    }


def run_generation(
    *,
    sample_manifest_path: Path,
    sample_manifest_sha256: str,
    output_dir: Path,
    smoke: bool,
    resume: bool,
    gpu_wait_timeout_seconds: int,
    gpu_poll_seconds: int,
) -> dict[str, Any]:
    _validate_gpu_environment()
    physical_identity = _physical_gpu1_identity()
    manifest, input_rows, tokenizer, observed_sha = _load_inputs(
        sample_manifest_path, sample_manifest_sha256, load_tokenizer=True
    )
    if tokenizer is None:
        raise Core8StochasticK10Error("validated tokenizer was not loaded")
    anchor = _load_anchor()
    samples = manifest["samples"]
    cases = _canonical_cases(samples, smoke=smoke)
    tuple_prefix_sha256s = _tuple_prefix_sha256s(cases)
    expected = len(cases)
    input_by_id = {str(row["sample_id"]): row for row in input_rows}
    source_hashes = _source_hashes(manifest, observed_sha)
    output_dir = output_dir.expanduser().resolve()
    partial_dir = output_dir / ".partial"
    progress_path = partial_dir / "generations.progress.v1.jsonl"
    final_path = output_dir / "generations.jsonl"
    state_path = output_dir / "state.progress.v1.json"
    launch_path = output_dir / "launch.json"
    run_path = output_dir / "manifest.json"
    launch_payload = seal_manifest(
        {
            "schema_version": LAUNCH_SCHEMA,
            "status": "initialized",
            "created_at_utc": _utc_now(),
            "evaluation_id": EVALUATION_ID,
            "scope": "one-row-one-seed-smoke" if smoke else "formal-n128-k10",
            "model_id": MODEL_ID,
            "model_label": MODEL_LABEL,
            "model": anchor["model"],
            "adapter": None,
            "sealed_anchor_manifest": anchor["anchor"],
            "sample_manifest": {
                "path": str(sample_manifest_path.expanduser().resolve()),
                "sha256": observed_sha,
                "payload_sha256": manifest["integrity"]["payload_sha256"],
            },
            "source_artifact_sha256s": source_hashes,
            "generation": {
                **_sampling_contract(),
                "expected_samples": 1 if smoke else EXPECTED_SAMPLES,
                "expected_meetings": 1 if smoke else EXPECTED_MEETINGS,
                "expected_cases": expected,
            },
            "runtime": {
                "python": sys.executable,
                "python_version": platform.python_version(),
                "cuda_device_order": "PCI_BUS_ID",
                "cuda_visible_devices": "1",
                "physical_gpu_index": PHYSICAL_GPU_INDEX,
                "physical_gpu_identity": physical_identity,
                "logical_device": "cuda:0",
                "gpu_lock_path": str(GPU_LOCK),
                "gpu_busy_policy": "wait-before-load-fail-closed-after-load-v1",
            },
            "persistence": {
                "append_flush_fsync_per_row": True,
                "full_generated_text": True,
                "answer": True,
                "generated_token_ids": True,
                "source_text_and_hashes": True,
                "sealed_state": True,
                "canonical_prefix_resume": True,
                "one_row_wal_recovery": True,
            },
        }
    )
    compare_keys = tuple(key for key in launch_payload if key not in {"created_at_utc", "integrity"})
    rows: list[dict[str, Any]] = []
    lines: list[bytes] = []
    resume_count = 0
    runtime_evidence: dict[str, Any] | None = None
    active_path = progress_path
    if resume:
        if run_path.exists() or run_path.is_symlink():
            raise Core8StochasticK10Error("refusing to resume a sealed run")
        if not launch_path.is_file() or launch_path.is_symlink():
            raise Core8StochasticK10Error("resume launch manifest is missing")
        prior_launch = _read_json(launch_path)
        try:
            validate_manifest_integrity(prior_launch)
        except Exception as exc:
            raise Core8StochasticK10Error(f"launch integrity failed: {exc}") from exc
        for key in compare_keys:
            if prior_launch.get(key) != launch_payload.get(key):
                raise Core8StochasticK10Error(f"resume launch drift at {key}")
        if progress_path.is_file() and not progress_path.is_symlink():
            active_path = progress_path
        elif final_path.is_file() and not final_path.is_symlink():
            active_path = final_path
        else:
            raise Core8StochasticK10Error("resume generation prefix is missing")
        if progress_path.exists() and final_path.exists():
            raise Core8StochasticK10Error("partial and final artifacts coexist")
        rows, lines = _read_canonical_jsonl(active_path)
        _validate_rows(
            rows,
            cases=cases,
            input_rows_by_id=input_by_id,
            sample_manifest_sha256=observed_sha,
            source_hashes=source_hashes,
            tokenizer=tokenizer,
        )
        state = _read_json(state_path)
        resume_count, _ = _validate_resume_state(
            state,
            lines=lines,
            active_path=active_path,
            partial_path=progress_path,
            cases=cases,
        )
        persisted_runtime = state.get("runtime_evidence")
        if persisted_runtime is not None:
            if not isinstance(persisted_runtime, Mapping):
                raise Core8StochasticK10Error("resume runtime evidence is invalid")
            runtime_evidence = dict(persisted_runtime)
    else:
        if output_dir.exists() or output_dir.is_symlink():
            raise Core8StochasticK10Error(f"refusing to overwrite: {output_dir}")
        output_dir.mkdir(parents=True)
        partial_dir.mkdir()
        _fsync_directory(output_dir.parent)
        _fsync_directory(output_dir)
        _write_new_json(launch_path, launch_payload)
        with progress_path.open("xb") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(partial_dir)
        empty_sha = hashlib.sha256().hexdigest()
        _write_new_json(
            state_path,
            _state_payload(
                status="initialized",
                completed=0,
                expected=expected,
                resume_count=0,
                results_path=progress_path,
                results_sha256=empty_sha,
                results_bytes=0,
                cases=cases,
                tuple_prefix_sha256=tuple_prefix_sha256s[0],
            ),
        )

    # Recover finalization without loading a model after a rename-before-state crash.
    if active_path == final_path and len(rows) == expected:
        if runtime_evidence is None:
            raise Core8StochasticK10Error(
                "finalization recovery lacks verified GPU1 runtime evidence"
            )
        final_sha, final_bytes = _prefix_digest(lines)
        _atomic_write_json(
            state_path,
            _state_payload(
                status="generation_complete",
                completed=expected,
                expected=expected,
                resume_count=max(0, resume_count - 1),
                results_path=final_path,
                results_sha256=final_sha,
                results_bytes=final_bytes,
                cases=cases,
                tuple_prefix_sha256=tuple_prefix_sha256s[expected],
                runtime_evidence=runtime_evidence,
            ),
        )
    else:
        prompt_contract = manifest["prompt_contract"]
        try:
            system_prompt, suffix, config_sha = native_eval.native_probe._load_prompt_contract(
                Path(str(prompt_contract["training_config"]))
            )
        except native_eval.native_probe.Chk3ProbeError as exc:
            raise Core8StochasticK10Error(str(exc)) from exc
        if config_sha != prompt_contract["training_config_sha256"]:
            raise Core8StochasticK10Error("training prompt contract drift")

        def on_wait(status: str, processes: Sequence[Mapping[str, Any]]) -> None:
            current_sha, current_bytes = _prefix_digest(lines)
            _atomic_write_json(
                state_path,
                _state_payload(
                    status=status,
                    completed=len(rows),
                    expected=expected,
                    resume_count=resume_count,
                    results_path=progress_path,
                    results_sha256=current_sha,
                    results_bytes=current_bytes,
                    cases=cases,
                    tuple_prefix_sha256=tuple_prefix_sha256s[len(rows)],
                    external_compute_processes=[dict(value) for value in processes],
                ),
            )
            print(_canonical_json({"status": status, "gpu": 1, "processes": processes}),
                  flush=True)

        with exclusive_gpu1_lease(
            timeout_seconds=gpu_wait_timeout_seconds,
            poll_seconds=gpu_poll_seconds,
            on_wait=on_wait,
        ) as lease:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig

            if not torch.cuda.is_available():
                raise Core8StochasticK10Error("CUDA is unavailable")
            logical_uuid = _verify_logical_cuda0_is_physical_gpu1(
                torch, physical_identity
            )
            runtime_evidence = {
                "cuda_device_order": os.environ["CUDA_DEVICE_ORDER"],
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "physical_gpu_index": PHYSICAL_GPU_INDEX,
                "physical_gpu_identity": dict(physical_identity),
                "logical_device": "cuda:0",
                "logical_cuda0_uuid": logical_uuid,
                "gpu_lease": dict(lease),
                "torch": str(torch.__version__),
                "transformers": str(transformers.__version__),
            }
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(anchor["model_path"]),
                dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
                quantization_config=quantization,
                device_map={"": 0},
                local_files_only=True,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            model.eval()
            model.config.use_cache = True
            eos_value = model.generation_config.eos_token_id
            if eos_value is None:
                eos_value = tokenizer.eos_token_id
            try:
                eos_ids = sorted(
                    native_eval.native_probe.common_probe._normalize_eos_ids(eos_value)
                )
            except native_eval.native_probe.common_probe.ProbeError as exc:
                raise Core8StochasticK10Error(str(exc)) from exc
            if not eos_ids:
                raise Core8StochasticK10Error("model/tokenizer has no EOS ID")
            for row in rows:
                if row.get("eos_token_ids") != eos_ids or row.get(
                    "pad_token_id"
                ) != int(tokenizer.pad_token_id):
                    raise Core8StochasticK10Error("resume EOS/pad contract drift")
            digest = hashlib.sha256()
            byte_count = 0
            for line in lines:
                digest.update(line)
                byte_count += len(line)
            if active_path == final_path:
                raise Core8StochasticK10Error("cannot append to final artifact")
            torch.cuda.reset_peak_memory_stats()
            with progress_path.open("ab") as output_handle:
                for case in cases[len(rows):]:
                    sample, replicate_id, replicate_seed, paired_seed_key = case
                    sample_id = str(sample["sample_id"])
                    input_row = input_by_id[sample_id]
                    external = _external_gpu1_compute_processes()
                    if external:
                        _atomic_write_json(
                            state_path,
                            _state_payload(
                                status="external_gpu1_process_detected_fail_closed",
                                completed=len(rows),
                                expected=expected,
                                resume_count=resume_count,
                                results_path=progress_path,
                                results_sha256=digest.hexdigest(),
                                results_bytes=byte_count,
                                cases=cases,
                                tuple_prefix_sha256=tuple_prefix_sha256s[len(rows)],
                                external_compute_processes=external,
                                runtime_evidence=runtime_evidence,
                            ),
                        )
                        raise ExternalGpu1ProcessError(
                            f"external GPU1 process appeared after model load: {external}"
                        )
                    prompt_ids = core8._chat_prompt_ids(
                        tokenizer,
                        prompt=str(input_row["prompt"]),
                        system_prompt=system_prompt,
                        suffix=suffix,
                    )
                    if len(prompt_ids) != sample["prompt_token_count"]:
                        raise Core8StochasticK10Error(
                            f"runtime prompt token-count drift for {sample_id}"
                        )
                    context_limit = int(
                        getattr(model.config, "max_position_embeddings", 0) or 0
                    )
                    if context_limit and len(prompt_ids) + MAX_NEW_TOKENS > context_limit:
                        raise Core8StochasticK10Error(
                            f"input would truncate/overflow for {sample_id}"
                        )
                    input_ids = torch.tensor(
                        [prompt_ids], dtype=torch.long, device="cuda:0"
                    )
                    attention_mask = torch.ones_like(input_ids)
                    row_seed = derive_row_seed(replicate_seed, paired_seed_key)
                    stochastic.reset_row_rng(transformers, torch, row_seed)
                    with torch.inference_mode():
                        sequences = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            do_sample=True,
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            top_k=TOP_K,
                            repetition_penalty=REPETITION_PENALTY,
                            max_new_tokens=MAX_NEW_TOKENS,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=eos_ids,
                            use_cache=True,
                        )
                    # This post-generate check includes the final tuple.  If a
                    # process entered while generate was running, its output is
                    # discarded and the canonical prefix is left unchanged.
                    external = _external_gpu1_compute_processes()
                    if external:
                        _atomic_write_json(
                            state_path,
                            _state_payload(
                                status="external_gpu1_process_detected_post_generate_fail_closed",
                                completed=len(rows),
                                expected=expected,
                                resume_count=resume_count,
                                results_path=progress_path,
                                results_sha256=digest.hexdigest(),
                                results_bytes=byte_count,
                                cases=cases,
                                tuple_prefix_sha256=tuple_prefix_sha256s[len(rows)],
                                external_compute_processes=external,
                                runtime_evidence=runtime_evidence,
                            ),
                        )
                        raise ExternalGpu1ProcessError(
                            "external GPU1 process appeared during generation; "
                            f"discarding uncommitted tuple: {external}"
                        )
                    generated_ids = sequences[0, input_ids.shape[1]:].tolist()
                    try:
                        text = native_eval.native_probe.common_probe.decode_completion_preserving_boundary(
                            tokenizer, generated_ids, eos_ids
                        )
                    except native_eval.native_probe.common_probe.ProbeError as exc:
                        raise Core8StochasticK10Error(str(exc)) from exc
                    result = _build_result(
                        text=text,
                        generated_ids=generated_ids,
                        eos_ids=eos_ids,
                        pad_token_id=int(tokenizer.pad_token_id),
                        sample=sample,
                        input_row=input_row,
                        replicate_id=replicate_id,
                        replicate_seed=replicate_seed,
                        paired_seed_key=paired_seed_key,
                        sample_manifest_sha256=observed_sha,
                        source_hashes=source_hashes,
                    )
                    _validate_result(
                        result,
                        sample=sample,
                        input_row=input_row,
                        replicate_id=replicate_id,
                        replicate_seed=replicate_seed,
                        paired_seed_key=paired_seed_key,
                        sample_manifest_sha256=observed_sha,
                        source_hashes=source_hashes,
                        tokenizer=tokenizer,
                    )
                    encoded = (_canonical_json(result) + "\n").encode("utf-8")
                    output_handle.write(encoded)
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
                    rows.append(result)
                    lines.append(encoded)
                    digest.update(encoded)
                    byte_count += len(encoded)
                    _atomic_write_json(
                        state_path,
                        _state_payload(
                            status="generating",
                            completed=len(rows),
                            expected=expected,
                            resume_count=resume_count,
                            results_path=progress_path,
                            results_sha256=digest.hexdigest(),
                            results_bytes=byte_count,
                            cases=cases,
                            tuple_prefix_sha256=tuple_prefix_sha256s[len(rows)],
                            runtime_evidence=runtime_evidence,
                            last_tuple={
                                "model_id": MODEL_ID,
                                "sample_id": sample_id,
                                "replicate_id": replicate_id,
                                "paired_seed_key": paired_seed_key,
                                "row_seed": row_seed,
                            },
                        ),
                    )
                    print(
                        _canonical_json(
                            {
                                "status": "generating",
                                "completed_cases": len(rows),
                                "expected_cases": expected,
                                "sample_id": sample_id,
                                "replicate_id": replicate_id,
                                "row_seed": row_seed,
                                "progress_sha256": digest.hexdigest(),
                            }
                        ),
                        flush=True,
                    )
            del model
            torch.cuda.empty_cache()
        if runtime_evidence is None:
            raise Core8StochasticK10Error("GPU1 runtime evidence was not captured")
        _validate_rows(
            rows,
            cases=cases,
            input_rows_by_id=input_by_id,
            sample_manifest_sha256=observed_sha,
            source_hashes=source_hashes,
            tokenizer=tokenizer,
            complete=True,
        )
        final_sha, final_bytes = _prefix_digest(lines)
        os.replace(progress_path, final_path)
        _fsync_directory(partial_dir)
        _fsync_directory(output_dir)
        _atomic_write_json(
            state_path,
            _state_payload(
                status="generation_complete",
                completed=expected,
                expected=expected,
                resume_count=resume_count,
                results_path=final_path,
                results_sha256=final_sha,
                results_bytes=final_bytes,
                cases=cases,
                tuple_prefix_sha256=tuple_prefix_sha256s[expected],
                runtime_evidence=runtime_evidence,
            ),
        )

    rows, lines = _read_canonical_jsonl(final_path)
    final_sha, final_bytes = _prefix_digest(lines)
    state = _read_json(state_path)
    try:
        state_payload_sha = validate_manifest_integrity(state)
    except Exception as exc:
        raise Core8StochasticK10Error(f"final state integrity failed: {exc}") from exc
    run_manifest = seal_manifest(
        {
            "schema_version": RUN_SCHEMA,
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "evaluation_id": EVALUATION_ID,
            "scope": "one-row-one-seed-smoke" if smoke else "formal-n128-k10",
            "model_id": MODEL_ID,
            "model_label": MODEL_LABEL,
            "sample_manifest": launch_payload["sample_manifest"],
            "source_artifact_sha256s": source_hashes,
            "generation_contract": _sampling_contract(),
            "runtime_evidence": runtime_evidence,
            "generations": _file_binding(
                final_path,
                rows=len(rows),
                sha256=final_sha,
                byte_count=final_bytes,
            ),
            "state": _file_binding(state_path, payload_sha256=state_payload_sha),
            "launch": _file_binding(
                launch_path,
                payload_sha256=launch_payload["integrity"]["payload_sha256"],
            ),
            "summary": _summary(rows, smoke=smoke),
            "limitations": [
                "This run measures stochastic decoding robustness, not causal contribution.",
                "K=10 estimates within-prompt decoding variability; meetings remain the inferential clusters.",
                "Semantic MPNet/BERTScore scoring is a separate offline stage.",
            ],
        }
    )
    _write_new_json(run_path, run_manifest)
    validation = _deep_validate_artifacts(
        output_dir=output_dir,
        sample_manifest_path=sample_manifest_path,
        sample_manifest_sha256=sample_manifest_sha256,
        smoke=smoke,
        load_tokenizer=True,
    )
    return {"status": "complete", "output_dir": str(output_dir), **validation}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "validate"):
        child = subparsers.add_parser(command)
        child.add_argument("--sample-manifest", type=Path,
                           default=DEFAULT_SAMPLE_MANIFEST)
        child.add_argument("--sample-manifest-sha256",
                           default=DEFAULT_SAMPLE_MANIFEST_SHA256)
        child.add_argument("--output-dir", type=Path, required=True)
        child.add_argument("--smoke", action="store_true")
        if command == "run":
            child.add_argument("--resume", action="store_true")
            child.add_argument("--gpu-wait-timeout-seconds", type=int, default=86400)
            child.add_argument("--gpu-poll-seconds", type=int, default=5)
        else:
            child.add_argument("--no-tokenizer", action="store_true")
    status = subparsers.add_parser("status")
    status.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "run":
        result = run_generation(
            sample_manifest_path=args.sample_manifest,
            sample_manifest_sha256=args.sample_manifest_sha256,
            output_dir=args.output_dir,
            smoke=args.smoke,
            resume=args.resume,
            gpu_wait_timeout_seconds=args.gpu_wait_timeout_seconds,
            gpu_poll_seconds=args.gpu_poll_seconds,
        )
    elif args.command == "validate":
        result = _deep_validate_artifacts(
            output_dir=args.output_dir.expanduser().resolve(),
            sample_manifest_path=args.sample_manifest,
            sample_manifest_sha256=args.sample_manifest_sha256,
            smoke=args.smoke,
            load_tokenizer=not args.no_tokenizer,
        )
    else:
        output_dir = args.output_dir.expanduser().resolve()
        state_path = output_dir / "state.progress.v1.json"
        manifest_path = output_dir / "manifest.json"
        if manifest_path.is_file():
            result = {"status": "sealed", "manifest": _read_json(manifest_path)}
        elif state_path.is_file():
            state = _read_json(state_path)
            validate_manifest_integrity(state)
            result = {"status": "active", "state": state}
        else:
            result = {"status": "not_started", "output_dir": str(output_dir)}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
