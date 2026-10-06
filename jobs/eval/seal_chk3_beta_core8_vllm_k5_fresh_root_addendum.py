"""Seal the create-time absence of the remediated vLLM K5 benchmark root.

The first remediation receipt preserved a zero-row vLLM V1 compatibility
failure and selected a new, versioned benchmark root.  It did not, however,
measure that the new root was absent when the receipt was sealed.  This
create-only addendum closes that provenance gap without replacing either of
the two historical receipts.

The addendum binds the historical migration and remediation receipts, the
fixed runner, the post-addendum launcher, and the benchmark selector.  At
creation it also proves that the v2 benchmark root does not lexically exist
(including as a dangling symlink), that both physical GPUs have no compute
processes, and that the ordered GPU lock pair can be acquired.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    ROOT / "output/evaluation/main/chk3_beta_core8_merged_n2048_vllm_k5_20260816_v1"
)
ORIGINAL_MIGRATION_RECEIPT = RUN_ROOT / "migration/nf4_k10_superseded_receipt.json"
ASYNC_OUTPUT_REMEDIATION_RECEIPT = (
    RUN_ROOT / "migration/vllm_v1_async_output_remediation_receipt.json"
)
DEFAULT_RECEIPT = (
    RUN_ROOT / "migration/vllm_v1_async_output_fresh_root_addendum_receipt.json"
)
PATCHED_BENCHMARK_ROOT = RUN_ROOT / "benchmark_max_num_seqs_chk1_core8_k5_v2"
PATCHED_RUNNER = ROOT / "jobs/eval/eval_chk3_beta_core8_merged_vllm_k5.py"
PATCHED_LAUNCHER = ROOT / "run/eval_chk3_beta_core8_merged_vllm_k5.sh"
BENCHMARK_SELECTOR = ROOT / "jobs/eval/select_chk3_beta_core8_vllm_k5_benchmark.py"
ASYNC_OUTPUT_REMEDIATION_TOOL = (
    ROOT / "jobs/eval/remediate_chk3_beta_core8_vllm_k5_v1_async_output.py"
)
GPU_LOCKS = (
    Path("/tmp/fomc_trainer_chk3_beta_core8_vllm_k5_gpu0.lock"),
    Path("/tmp/fomc_trainer_chk3_beta_core8_vllm_k5_gpu1.lock"),
)
PIPELINE_LOCK = Path("/tmp/fomc_trainer_chk3_beta_core8_merged_vllm_k5_pipeline.lock")

SCHEMA_VERSION = "chk3-beta-core8-vllm-k5-v1-fresh-root-addendum-receipt-v1"
EVALUATION_ID = "chk3-beta-core8-merged-1993-2025-n2048-vllm-k5-v1"
CONFIRMATION = "SEAL-VLLM-K5-FRESH-V2-ROOT-ADDENDUM"

ORIGINAL_MIGRATION_SHA256 = (
    "81b41fbcb9456a25e886377b23f22783c873235ccae4b794c3f954949247e07f"
)
ORIGINAL_MIGRATION_PAYLOAD_SHA256 = (
    "7c75660f0d0d9d39b618486c806346f597edcfeda2843fba4216b5b3ca494947"
)
ASYNC_OUTPUT_REMEDIATION_SHA256 = (
    "c271e8c396d6526cd8f0b06147638bbf69ab3497ed03367359695ae53daa9a5c"
)
ASYNC_OUTPUT_REMEDIATION_PAYLOAD_SHA256 = (
    "35e4c3a8398e583f8d708c66ae01ce5e230708e755d0fbad95c78e8a46c4e04f"
)
PRE_ADDENDUM_LAUNCHER_SHA256 = (
    "a65a86058abc0fb8acdf3489ac14c785585f481fc6e8fea1b4c7e761ca83ae3e"
)
PATCHED_RUNNER_SHA256 = (
    "1bc77b941c278a53c88189f32f1a158c4c607600bd795b5f86af9474fffd2632"
)
BENCHMARK_SELECTOR_SHA256 = (
    "2298b8d6d19051897af338a883a8dba2568af288ebbb97f99f3871a6e932b8cf"
)
ASYNC_OUTPUT_REMEDIATION_TOOL_SHA256 = (
    "a111a188aeea7798b06ac3ff366cfaecf81ee94a3a0cde1d599f2b97a10b9072"
)


class FreshRootAddendumError(RuntimeError):
    """A fresh-root addendum invariant was not proven."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise FreshRootAddendumError(f"bound artifact is a symlink: {unresolved}")
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise FreshRootAddendumError(f"bound artifact is missing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _read_sealed_json(path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    binding = _binding(path)
    resolved = Path(str(binding["path"]))
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FreshRootAddendumError(f"cannot read sealed JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FreshRootAddendumError(f"sealed JSON is not an object: {path}")
    try:
        payload_sha256 = validate_manifest_integrity(value)
    except Exception as exc:
        raise FreshRootAddendumError(f"sealed JSON integrity failed: {path}") from exc
    return value, binding, payload_sha256


def _readonly_receipt_binding(
    path: Path,
    *,
    expected_sha256: str,
    expected_payload_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    value, binding, payload_sha256 = _read_sealed_json(path)
    mode = stat.S_IMODE(path.stat().st_mode)
    if (
        binding["sha256"] != expected_sha256
        or payload_sha256 != expected_payload_sha256
        or mode != 0o444
    ):
        raise FreshRootAddendumError(f"historical receipt drift: {path}")
    return value, {
        **binding,
        "payload_sha256": payload_sha256,
        "mode": oct(mode),
    }


def audit_historical_receipts() -> dict[str, Any]:
    migration, migration_binding = _readonly_receipt_binding(
        ORIGINAL_MIGRATION_RECEIPT,
        expected_sha256=ORIGINAL_MIGRATION_SHA256,
        expected_payload_sha256=ORIGINAL_MIGRATION_PAYLOAD_SHA256,
    )
    remediation, remediation_binding = _readonly_receipt_binding(
        ASYNC_OUTPUT_REMEDIATION_RECEIPT,
        expected_sha256=ASYNC_OUTPUT_REMEDIATION_SHA256,
        expected_payload_sha256=ASYNC_OUTPUT_REMEDIATION_PAYLOAD_SHA256,
    )
    original_in_remediation = remediation.get("original_migration_receipt")
    failed_attempt = remediation.get("failed_attempt")
    repair = remediation.get("remediation")
    historical_gpu_idle = remediation.get("gpu_idle_evidence")
    safety = remediation.get("safety")
    if (
        migration.get("status") != "passed"
        or remediation.get("schema_version")
        != "chk3-beta-core8-vllm-k5-v1-async-output-remediation-receipt-v1"
        or remediation.get("status") != "passed"
        or remediation.get("evaluation_id") != EVALUATION_ID
        or not isinstance(original_in_remediation, Mapping)
        or original_in_remediation.get("sha256") != ORIGINAL_MIGRATION_SHA256
        or original_in_remediation.get("payload_sha256")
        != ORIGINAL_MIGRATION_PAYLOAD_SHA256
        or original_in_remediation.get("mode") != "0o444"
        or not isinstance(failed_attempt, Mapping)
        or failed_attempt.get("generated_rows") != 0
        or failed_attempt.get("completed_chunks") != 0
        or failed_attempt.get("model_load_started") is not False
        or failed_attempt.get("preserved_unchanged") is not True
        or failed_attempt.get("canonical_generations_absent") is not True
        or failed_attempt.get("sealed_run_manifest_absent") is not True
        or not isinstance(repair, Mapping)
        or repair.get("new_benchmark_root") != str(PATCHED_BENCHMARK_ROOT.resolve())
        or repair.get("new_benchmark_root_must_be_fresh") is not True
        or repair.get("unsupported_explicit_engine_argument_removed") is not True
        or repair.get("patched_runner", {}).get("sha256") != PATCHED_RUNNER_SHA256
        or repair.get("patched_launcher", {}).get("sha256")
        != PRE_ADDENDUM_LAUNCHER_SHA256
        or repair.get("bridge_implementation", {}).get("sha256")
        != ASYNC_OUTPUT_REMEDIATION_TOOL_SHA256
        or not isinstance(historical_gpu_idle, Mapping)
        or historical_gpu_idle.get("both_gpus_idle") is not True
        or historical_gpu_idle.get("compute_processes") != {"0": [], "1": []}
        or not isinstance(historical_gpu_idle.get("physical_gpu_identities"), list)
        or len(historical_gpu_idle["physical_gpu_identities"]) != 2
        or not isinstance(safety, Mapping)
        or safety.get("generated_rows_reused") != 0
        or safety.get("failed_directory_deleted_or_modified") is not False
        or safety.get("gpu_work_launched_by_remediation") is not False
        or safety.get("receipt_create_only") is not True
        or safety.get("receipt_mode") != "0o444"
    ):
        raise FreshRootAddendumError("historical remediation chain drift")
    return {
        "original_migration_receipt": migration_binding,
        "async_output_remediation_receipt": remediation_binding,
        "historical_remediation_launcher": dict(repair["patched_launcher"]),
        "historical_physical_gpu_identities": list(
            historical_gpu_idle["physical_gpu_identities"]
        ),
        "zero_row_failure_bound": True,
        "failed_generated_rows": 0,
    }


def audit_implementations() -> dict[str, Any]:
    runner = _binding(PATCHED_RUNNER)
    launcher = _binding(PATCHED_LAUNCHER)
    selector = _binding(BENCHMARK_SELECTOR)
    remediation_tool = _binding(ASYNC_OUTPUT_REMEDIATION_TOOL)
    launcher_source = PATCHED_LAUNCHER.read_text(encoding="utf-8")
    required_launcher_fragments = (
        "vllm_v1_async_output_fresh_root_addendum_receipt.json",
        "jobs.eval.seal_chk3_beta_core8_vllm_k5_fresh_root_addendum",
        "benchmark_max_num_seqs_chk1_core8_k5_v2",
    )
    if (
        runner["sha256"] != PATCHED_RUNNER_SHA256
        or selector["sha256"] != BENCHMARK_SELECTOR_SHA256
        or remediation_tool["sha256"] != ASYNC_OUTPUT_REMEDIATION_TOOL_SHA256
        or launcher["sha256"] == PRE_ADDENDUM_LAUNCHER_SHA256
        or any(
            fragment not in launcher_source for fragment in required_launcher_fragments
        )
    ):
        raise FreshRootAddendumError("post-addendum implementation drift")
    return {
        "patched_runner": runner,
        "patched_launcher": launcher,
        "benchmark_selector": selector,
        "async_output_remediation_tool": remediation_tool,
        "fresh_root_addendum_tool": _binding(Path(__file__)),
    }


def audit_fresh_root_absence() -> dict[str, Any]:
    path = PATCHED_BENCHMARK_ROOT.expanduser()
    lexical_exists = os.path.lexists(path)
    is_symlink = path.is_symlink()
    if lexical_exists or is_symlink:
        raise FreshRootAddendumError(
            f"v2 benchmark root must not exist at addendum seal: {path}"
        )
    return {
        "checked_at_utc": _utc_now(),
        "path": str(path.resolve(strict=False)),
        "lexists": False,
        "exists": False,
        "is_symlink": False,
        "new_benchmark_root_absent_at_seal": True,
    }


def _gpu_identity_rows() -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,memory.total",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FreshRootAddendumError(f"cannot inspect GPU identities: {exc}") from exc
    rows = []
    for line in completed.stdout.splitlines():
        columns = [column.strip() for column in line.split(",")]
        if len(columns) != 5 or not columns[0].isdigit():
            raise FreshRootAddendumError(f"invalid GPU identity row: {line!r}")
        rows.append(
            {
                "physical_gpu_index": int(columns[0]),
                "uuid": columns[1],
                "pci_bus_id": columns[2].lower(),
                "name": columns[3],
                "memory_total_mib": int(columns[4]),
            }
        )
    if [row["physical_gpu_index"] for row in rows] != [0, 1]:
        raise FreshRootAddendumError("exact physical GPU0/GPU1 identities are required")
    return rows


def _gpu_compute_processes(index: int) -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                f"--id={index}",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FreshRootAddendumError(
            f"cannot inspect GPU{index} processes: {exc}"
        ) from exc
    rows = []
    for line in completed.stdout.splitlines():
        columns = [column.strip() for column in line.split(",", 1)]
        if not columns or not columns[0].isdigit():
            if not line.strip() or "No running processes" in line:
                continue
            raise FreshRootAddendumError(f"invalid GPU{index} process row: {line!r}")
        rows.append(
            {
                "pid": int(columns[0]),
                "process_name": columns[1] if len(columns) == 2 else "",
            }
        )
    return rows


def _prove_lock_pair_available() -> list[str]:
    handles = []
    acquired = []
    try:
        for path in GPU_LOCKS:
            if path.is_symlink() or not path.is_file():
                raise FreshRootAddendumError(f"GPU lock is missing or unsafe: {path}")
            handle = path.open("r+", encoding="utf-8")
            handles.append(handle)
        for handle in handles:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise FreshRootAddendumError(
                    "GPU0/GPU1 lock pair is not available"
                ) from exc
            acquired.append(handle)
        return [str(path.resolve()) for path in GPU_LOCKS]
    finally:
        for handle in reversed(acquired):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        for handle in handles:
            handle.close()


def audit_idle_gpus(
    expected_identities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    identities = _gpu_identity_rows()
    if identities != [dict(identity) for identity in expected_identities]:
        raise FreshRootAddendumError(
            "physical GPU UUID/PCI/index identities drifted from first remediation"
        )
    processes = {str(index): _gpu_compute_processes(index) for index in range(2)}
    if any(processes.values()):
        raise FreshRootAddendumError(
            f"both GPUs must be idle at addendum seal: {processes}"
        )
    return {
        "audited_at_utc": _utc_now(),
        "physical_gpu_identities": identities,
        "compute_processes": processes,
        "both_gpus_idle": True,
        "no_compute_pids": True,
        "lock_pair_available": True,
        "gpu_lock_paths": _prove_lock_pair_available(),
    }


def build_receipt(
    *,
    historical_receipts: Mapping[str, Any],
    implementations: Mapping[str, Any],
    fresh_root: Mapping[str, Any],
    gpu_idle: Mapping[str, Any],
) -> dict[str, Any]:
    return seal_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_at_utc": _utc_now(),
            "evaluation_id": EVALUATION_ID,
            "purpose": "prove_v2_benchmark_root_absent_before_any_remediated_gpu_work",
            "historical_receipts": dict(historical_receipts),
            "implementations": dict(implementations),
            "fresh_root_evidence": dict(fresh_root),
            "gpu_idle_evidence": dict(gpu_idle),
            "safety": {
                "historical_receipts_overwritten": False,
                "historical_failed_directory_modified": False,
                "v2_benchmark_root_created_by_addendum": False,
                "gpu_work_launched_by_addendum": False,
                "receipt_create_only": True,
                "receipt_mode": "0o444",
            },
        }
    )


def write_readonly_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FreshRootAddendumError(f"receipt is create-only: {destination}")
    encoded = (_canonical(receipt) + "\n").encode("utf-8")
    staging = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(staging, 0o444)
        try:
            os.link(staging, destination)
        except FileExistsError as exc:
            raise FreshRootAddendumError(
                f"receipt publication race: {destination}"
            ) from exc
        with destination.open("rb") as handle:
            os.fsync(handle.fileno())
    finally:
        staging.unlink(missing_ok=True)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def validate_receipt(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    receipt, binding, payload_sha256 = _read_sealed_json(path)
    mode = stat.S_IMODE(path.stat().st_mode)
    historical = audit_historical_receipts()
    implementations = audit_implementations()
    fresh = receipt.get("fresh_root_evidence")
    idle = receipt.get("gpu_idle_evidence")
    safety = receipt.get("safety")
    if (
        mode != 0o444
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("status") != "passed"
        or receipt.get("evaluation_id") != EVALUATION_ID
        or receipt.get("purpose")
        != "prove_v2_benchmark_root_absent_before_any_remediated_gpu_work"
        or receipt.get("historical_receipts") != historical
        or receipt.get("implementations") != implementations
        or not isinstance(fresh, Mapping)
        or fresh.get("path") != str(PATCHED_BENCHMARK_ROOT.resolve(strict=False))
        or fresh.get("lexists") is not False
        or fresh.get("exists") is not False
        or fresh.get("is_symlink") is not False
        or fresh.get("new_benchmark_root_absent_at_seal") is not True
        or not isinstance(fresh.get("checked_at_utc"), str)
        or not isinstance(idle, Mapping)
        or idle.get("both_gpus_idle") is not True
        or idle.get("no_compute_pids") is not True
        or idle.get("compute_processes") != {"0": [], "1": []}
        or idle.get("physical_gpu_identities")
        != historical["historical_physical_gpu_identities"]
        or not isinstance(idle.get("audited_at_utc"), str)
        or idle.get("lock_pair_available") is not True
        or idle.get("gpu_lock_paths") != [str(path.resolve()) for path in GPU_LOCKS]
        or not isinstance(safety, Mapping)
        or safety.get("historical_receipts_overwritten") is not False
        or safety.get("historical_failed_directory_modified") is not False
        or safety.get("v2_benchmark_root_created_by_addendum") is not False
        or safety.get("gpu_work_launched_by_addendum") is not False
        or safety.get("receipt_create_only") is not True
        or safety.get("receipt_mode") != "0o444"
    ):
        raise FreshRootAddendumError("sealed fresh-root addendum contract drift")
    return {
        "status": "passed",
        "path": str(path.resolve()),
        "sha256": binding["sha256"],
        "payload_sha256": payload_sha256,
        "mode": oct(mode),
        "new_benchmark_root_absent_at_seal": True,
        "patched_runner_sha256": implementations["patched_runner"]["sha256"],
        "patched_launcher_sha256": implementations["patched_launcher"]["sha256"],
        "benchmark_selector_sha256": implementations["benchmark_selector"]["sha256"],
    }


def create_receipt() -> dict[str, Any]:
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        lock_fd = os.open(PIPELINE_LOCK, flags, 0o600)
    except OSError as exc:
        raise FreshRootAddendumError(
            f"cannot safely open pipeline lock: {PIPELINE_LOCK}"
        ) from exc
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FreshRootAddendumError(
                "pipeline lock is busy; addendum creation must be serialized"
            ) from exc
        if DEFAULT_RECEIPT.exists() or DEFAULT_RECEIPT.is_symlink():
            raise FreshRootAddendumError(
                f"production fresh-root addendum already exists: {DEFAULT_RECEIPT}"
            )
        historical = audit_historical_receipts()
        implementations = audit_implementations()
        fresh = audit_fresh_root_absence()
        gpu_idle = audit_idle_gpus(historical["historical_physical_gpu_identities"])
        receipt = build_receipt(
            historical_receipts=historical,
            implementations=implementations,
            fresh_root=fresh,
            gpu_idle=gpu_idle,
        )
        write_readonly_receipt(DEFAULT_RECEIPT, receipt)
        audit_fresh_root_absence()
        result = validate_receipt(DEFAULT_RECEIPT)
        audit_fresh_root_absence()
        return result
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--confirm", required=True)
    subparsers.add_parser("validate")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            if args.confirm != CONFIRMATION:
                raise FreshRootAddendumError("exact addendum confirmation is required")
            result = create_receipt()
        else:
            result = validate_receipt()
    except (FreshRootAddendumError, OSError, ValueError) as exc:
        print(
            _canonical(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 1
    print(_canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
