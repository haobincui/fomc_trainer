"""Losslessly publish the sealed v4 fixed-16 generations as meeting documents.

This is a CPU-only, versioned adapter around the established K=5 lossless
reshape core.  It accepts only the sealed stochastic-schedule-v2 formal suite
and its exact cohort.  All 30,720 generation rows are retained as eight-topic
sections in 3,840 model/meeting/replicate documents; no quality or hard-gate
filter is applied.

Publication is create-only.  A unique sibling staging directory is fsynced
and renamed with Linux ``RENAME_NOREPLACE``.  A failed staging directory is
never removed, so partial evidence remains inspectable and the final path can
never be overwritten.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import fcntl
import json
import os
import secrets
import shutil
import stat
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, IO, Iterator

from jobs.eval import assemble_chk3_beta_core8_meeting_documents_vllm_k5 as legacy
from jobs.eval import chk3_beta_core8_merged_contract as data_contract
from jobs.eval import prepare_chk3_beta_core8_merged_vllm_k5 as preparation
from jobs.eval import (
    remediate_chk3_beta_core8_vllm_k5_dual_dp1_stochastic_schedule_v2 as amendment,
)
from jobs.eval import (
    seal_chk3_beta_core8_vllm_k5_dual_dp1_stochastic_schedule_v2_suite as suite,
)
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    ROOT / "output/evaluation/main/chk3_beta_core8_merged_n2048_vllm_k5_20260816_v1"
)
DEFAULT_COHORT = RUN_ROOT / "preparation/cohort_n2048_k5.v1.json"
COHORT_SHA256 = "a815b5af8e6b33e3d1a2b211e346393f155d1d45acaafae80b822ee08128ab8e"
DEFAULT_SUITE = (
    RUN_ROOT
    / "generation_formal_n2048_k5_three_models_v4_stochastic_schedule_v2/manifest.json"
)
DEFAULT_OUTPUT = RUN_ROOT / "meeting_documents_n3840_v4_stochastic_schedule_v2"
ASSEMBLY_LOCK = Path(
    "/tmp/fomc_trainer_chk3_beta_core8_vllm_k5_stochastic_schedule_v2_assembly.lock"
)
FIXED_MAX_NUM_SEQS = 16
DOCUMENT_SCHEMA = (
    "chk3-beta-core8-merged-vllm-k5-dual-dp1-stochastic-schedule-meeting-document-v2"
)
MANIFEST_SCHEMA = (
    "chk3-beta-core8-merged-vllm-k5-dual-dp1-"
    "stochastic-schedule-meeting-documents-manifest-v2"
)
DOCUMENT_FILENAME = "meeting_documents.v2.jsonl"
EXPECTED_DOCUMENTS = (
    data_contract.EXPECTED_MEETINGS
    * len(preparation.REPLICATE_SEEDS)
    * len(preparation.MODEL_ORDER)
)
DOCUMENTS_PER_MODEL = data_contract.EXPECTED_MEETINGS * len(preparation.REPLICATE_SEEDS)
EXPECTED_SECTIONS = preparation.EXPECTED_TOTAL_CASES
MIN_FREE_BYTES = 8 * 1024**3
AT_FDCWD = -100
RENAME_NOREPLACE = 1


class StochasticScheduleMeetingAssemblyError(RuntimeError):
    """The sealed v4 generation matrix cannot be published losslessly."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _binding(
    physical_path: Path,
    *,
    reported_path: Path | None = None,
    rows: int | None = None,
    payload_sha256: str | None = None,
) -> dict[str, Any]:
    unresolved = physical_path.expanduser()
    if unresolved.is_symlink():
        raise StochasticScheduleMeetingAssemblyError(
            f"bound path is a symlink: {unresolved}"
        )
    path = unresolved.resolve()
    if not path.is_file():
        raise StochasticScheduleMeetingAssemblyError(f"bound file missing: {path}")
    report = path if reported_path is None else reported_path.expanduser().resolve()
    result: dict[str, Any] = {
        "path": str(report),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        result["rows"] = rows
    if payload_sha256 is not None:
        result["payload_sha256"] = payload_sha256
    return result


def _retag_documents(documents: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for document in documents:
        copied = copy.deepcopy(dict(document))
        copied["schema_version"] = DOCUMENT_SCHEMA
        result.append(copied)
    return result


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_jsonl_readonly(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if os.path.lexists(path):
        raise StochasticScheduleMeetingAssemblyError(f"refusing overwrite: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_canonical(dict(row)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Preserve a partial file as failure evidence.
        raise
    path.chmod(0o444)
    _fsync_directory(path.parent)


def _write_new_json_readonly(path: Path, value: Mapping[str, Any]) -> None:
    if os.path.lexists(path):
        raise StochasticScheduleMeetingAssemblyError(f"refusing overwrite: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_canonical(dict(value)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise
    path.chmod(0o444)
    _fsync_directory(path.parent)


def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically publish a directory without replacing any target."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise StochasticScheduleMeetingAssemblyError(
            "Linux renameat2 is required for atomic no-replace publication"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(target),
        RENAME_NOREPLACE,
    )
    if result != 0:
        observed_errno = ctypes.get_errno()
        if observed_errno == errno.EEXIST:
            raise StochasticScheduleMeetingAssemblyError(
                f"atomic publish target already exists: {target}"
            )
        raise StochasticScheduleMeetingAssemblyError(
            "atomic no-replace publication failed: "
            f"errno={observed_errno} {os.strerror(observed_errno)}"
        )
    _fsync_directory(target.parent)


@contextmanager
def _assembly_lock() -> Iterator[IO[str]]:
    if ASSEMBLY_LOCK.is_symlink():
        raise StochasticScheduleMeetingAssemblyError("assembly lock is a symlink")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(ASSEMBLY_LOCK, flags, 0o600)
    handle = os.fdopen(descriptor, "a+", encoding="utf-8")
    try:
        path_stat = os.stat(ASSEMBLY_LOCK, follow_symlinks=False)
        fd_stat = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or path_stat.st_nlink != 1
            or (path_stat.st_dev, path_stat.st_ino) != (fd_stat.st_dev, fd_stat.st_ino)
        ):
            raise StochasticScheduleMeetingAssemblyError("unsafe assembly lock")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StochasticScheduleMeetingAssemblyError(
                "another v4 meeting assembly holds the canonical lock"
            ) from exc
        yield handle
    finally:
        handle.close()


def _require_exact_target_contract(
    *,
    cohort_manifest: Path,
    cohort_manifest_sha256: str,
    suite_manifest: Path,
    output_dir: Path,
    max_num_seqs: int,
) -> tuple[Path, Path, Path]:
    cohort = cohort_manifest.expanduser().resolve()
    formal_suite = suite_manifest.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    if (
        cohort != DEFAULT_COHORT.resolve()
        or cohort_manifest_sha256 != COHORT_SHA256
        or formal_suite != DEFAULT_SUITE.resolve()
        or output != DEFAULT_OUTPUT.resolve()
        or max_num_seqs != FIXED_MAX_NUM_SEQS
    ):
        raise StochasticScheduleMeetingAssemblyError(
            "assembler accepts only the sealed fixed-16 v4 target contract"
        )
    for unresolved in (cohort_manifest, suite_manifest, output_dir):
        if unresolved.expanduser().is_symlink():
            raise StochasticScheduleMeetingAssemblyError(
                f"target contract path is a symlink: {unresolved}"
            )
    return cohort, formal_suite, output


def _load_inputs(
    *, cohort_manifest: Path, suite_manifest: Path
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
]:
    try:
        cohort, samples, ledger = legacy._load_cohort(cohort_manifest, COHORT_SHA256)
        validated = suite.load_and_validate_suite(
            suite_manifest,
            cohort_path=cohort_manifest,
            cohort_sha256=COHORT_SHA256,
            expected_scope="formal_merged_panel",
            max_num_seqs=FIXED_MAX_NUM_SEQS,
        )
    except Exception as exc:
        raise StochasticScheduleMeetingAssemblyError(str(exc)) from exc
    rows_by_model = {
        model_id: [dict(row) for row in validated["runs"][model_id]["results"]]
        for model_id in preparation.MODEL_ORDER
    }
    return cohort, samples, ledger, validated, rows_by_model


def _audit_document_matrix(
    documents: Sequence[Mapping[str, Any]],
    *,
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    meeting_order: list[str] = []
    for sample in samples:
        meeting_id = str(sample["meeting_id"])
        if not meeting_order or meeting_order[-1] != meeting_id:
            meeting_order.append(meeting_id)
    if len(meeting_order) != data_contract.EXPECTED_MEETINGS:
        raise StochasticScheduleMeetingAssemblyError("meeting inventory is not N=256")
    expected_order = [
        (model_id, meeting_id, replicate_id)
        for model_id in preparation.MODEL_ORDER
        for meeting_id in meeting_order
        for replicate_id in range(len(preparation.REPLICATE_SEEDS))
    ]
    observed_order: list[tuple[str, str, int]] = []
    seen_ids: set[str] = set()
    source_lines: dict[str, set[int]] = defaultdict(set)
    sections = 0
    exposed = 0
    normal_finishes = 0
    for document in documents:
        model_id = str(document.get("model_id") or "")
        meeting_id = str(document.get("meeting_id") or "")
        replicate_id = document.get("replicate_id")
        if isinstance(replicate_id, bool) or not isinstance(replicate_id, int):
            raise StochasticScheduleMeetingAssemblyError("invalid replicate ID")
        observed_order.append((model_id, meeting_id, replicate_id))
        expected_id = f"{model_id}::{meeting_id}::replicate-{replicate_id:02d}"
        if document.get("document_id") != expected_id or expected_id in seen_ids:
            raise StochasticScheduleMeetingAssemblyError(
                "duplicate or noncanonical document ID"
            )
        seen_ids.add(expected_id)
        topic_sections = document.get("sections")
        if (
            document.get("schema_version") != DOCUMENT_SCHEMA
            or not isinstance(topic_sections, list)
            or len(topic_sections) != len(data_contract.CORE_TOPICS)
            or document.get("section_count") != len(data_contract.CORE_TOPICS)
            or tuple(section.get("topic") for section in topic_sections)
            != tuple(data_contract.CORE_TOPICS)
        ):
            raise StochasticScheduleMeetingAssemblyError(
                "document Core8/schema closure drift"
            )
        answers: list[str] = []
        for section in topic_sections:
            if (
                section.get("model_id") != model_id
                or section.get("meeting_id") != meeting_id
                or section.get("replicate_id") != replicate_id
                or section.get("input_truncated") is not False
            ):
                raise StochasticScheduleMeetingAssemblyError(
                    "section identity/truncation drift"
                )
            source_line = section.get("source_generation_line_number")
            if (
                isinstance(source_line, bool)
                or not isinstance(source_line, int)
                or source_line in source_lines[model_id]
            ):
                raise StochasticScheduleMeetingAssemblyError(
                    "source generation line is invalid or duplicated"
                )
            source_lines[model_id].add(source_line)
            answers.append(str(section["answer"]))
            sections += 1
            normal_finishes += section.get("status") == "ok" and section.get(
                "finish_reason"
            ) in {"eos", "length"}
        expected_text = "\n\n".join(answers)
        if document.get("document_text") != expected_text or document.get(
            "document_text_sha256"
        ) != legacy.sha256_text(expected_text):
            raise StochasticScheduleMeetingAssemblyError(
                "document text/hash binding drift"
            )
        exposed += bool(document.get("cp318_selection_exposed"))
    if observed_order != expected_order:
        raise StochasticScheduleMeetingAssemblyError(
            "model/meeting/replicate ordering drift"
        )
    expected_lines = set(range(1, preparation.EXPECTED_CASES_PER_MODEL + 1))
    if any(
        source_lines[model_id] != expected_lines for model_id in preparation.MODEL_ORDER
    ):
        raise StochasticScheduleMeetingAssemblyError(
            "source generation line coverage drift"
        )
    documents_per_model = Counter(document["model_id"] for document in documents)
    if (
        len(documents) != EXPECTED_DOCUMENTS
        or sections != EXPECTED_SECTIONS
        or len(seen_ids) != EXPECTED_DOCUMENTS
        or documents_per_model
        != Counter(
            {model_id: DOCUMENTS_PER_MODEL for model_id in preparation.MODEL_ORDER}
        )
        or normal_finishes != EXPECTED_SECTIONS
    ):
        raise StochasticScheduleMeetingAssemblyError("document matrix closure drift")
    return {
        "models": len(preparation.MODEL_ORDER),
        "meetings": len(meeting_order),
        "replicates": len(preparation.REPLICATE_SEEDS),
        "topics_per_document": len(data_contract.CORE_TOPICS),
        "documents": len(documents),
        "documents_per_model": dict(documents_per_model),
        "sections": sections,
        "normal_finish_sections": normal_finishes,
        "input_truncation_sections": 0,
        "unique_document_ids": len(seen_ids),
        "cp318_selection_exposed_documents": exposed,
        "source_generation_lines_exact_per_model": True,
        "model_meeting_replicate_order_exact": True,
        "core8_topic_order_exact": True,
    }


def _source_generation_files(validated: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model_id in preparation.MODEL_ORDER:
        artifact = validated["runs"][model_id]["manifest"]["artifacts"][
            "canonical_generations"
        ]
        path = Path(str(artifact["path"]))
        observed = _binding(path, rows=preparation.EXPECTED_CASES_PER_MODEL)
        if dict(artifact) != observed:
            raise StochasticScheduleMeetingAssemblyError(
                f"canonical generation binding drift: {model_id}"
            )
        result[model_id] = copy.deepcopy(observed)
    return result


def _validated_publication(
    publication: Mapping[str, Any], *, final_root: Path
) -> tuple[str, str, int]:
    """Validate the create-only publication evidence before rebuilding it."""

    attempt = publication.get("publish_attempt_id")
    staging_basename = publication.get("staging_basename")
    free_bytes = publication.get("preflight_free_bytes")
    if (
        not isinstance(attempt, str)
        or len(attempt) != 32
        or any(character not in "0123456789abcdef" for character in attempt)
        or not isinstance(staging_basename, str)
        or "/" in staging_basename
        or not staging_basename.startswith(f"{final_root.name}.staging.")
        or not staging_basename.endswith(f".{attempt}")
        or isinstance(free_bytes, bool)
        or not isinstance(free_bytes, int)
        or free_bytes < MIN_FREE_BYTES
        or publication.get("minimum_required_free_bytes") != MIN_FREE_BYTES
        or publication.get("final_root") != str(final_root)
    ):
        raise StochasticScheduleMeetingAssemblyError(
            "assembly publication evidence drift"
        )
    pid_text = staging_basename.removeprefix(
        f"{final_root.name}.staging."
    ).removesuffix(f".{attempt}")
    if not pid_text.isdigit() or int(pid_text) <= 0:
        raise StochasticScheduleMeetingAssemblyError(
            "assembly staging process identity drift"
        )
    return attempt, staging_basename, free_bytes


def _manifest_payload(
    *,
    created_at_utc: str,
    final_root: Path,
    documents_physical_path: Path,
    cohort_manifest: Path,
    cohort: Mapping[str, Any],
    validated: Mapping[str, Any],
    matrix: Mapping[str, Any],
    publish_attempt_id: str,
    staging_basename: str,
    preflight_free_bytes: int,
) -> dict[str, Any]:
    suite_manifest = validated["manifest"]
    return {
        "schema_version": MANIFEST_SCHEMA,
        "status": "complete",
        "immutable": True,
        "created_at_utc": created_at_utc,
        "evaluation_id": suite.runner.EVALUATION_ID,
        "evaluation_scope": "formal_merged_panel_meeting_document_reshape",
        "model_order": list(preparation.MODEL_ORDER),
        "cohort_manifest": {
            **_binding(cohort_manifest),
            "payload_sha256": cohort["integrity"]["payload_sha256"],
        },
        "formal_suite_manifest": copy.deepcopy(validated["manifest_binding"]),
        "amendment_receipt": copy.deepcopy(suite_manifest["remediation_receipt"]),
        "speed_selection": copy.deepcopy(suite_manifest["speed_selection"]),
        "schedule_sensitivity_diagnostic": copy.deepcopy(
            suite_manifest["schedule_sensitivity_diagnostic"]
        ),
        "generation_gates": copy.deepcopy(suite_manifest["generation_gates"]),
        "official_smoke_suite": copy.deepcopy(suite_manifest["official_smoke_suite"]),
        "generation_contract": copy.deepcopy(suite_manifest["generation_contract"]),
        "source_generation_files": _source_generation_files(validated),
        "meeting_documents": {
            **_binding(
                documents_physical_path,
                reported_path=final_root / DOCUMENT_FILENAME,
                rows=EXPECTED_DOCUMENTS,
            ),
            "schema_version": DOCUMENT_SCHEMA,
        },
        "coverage": copy.deepcopy(dict(matrix)),
        "assembly_contract": {
            "ordering": "model_then_source_meeting_then_replicate_then_core8_topic",
            "topic_order": list(data_contract.CORE_TOPICS),
            "section_payload": "complete_generation_record_plus_assembly_provenance",
            "full_generation_records_preserved": True,
            "prompt_and_generated_token_ids_preserved": True,
            "source_and_generation_hashes_preserved": True,
            "hard_gate_filtering": False,
            "stochastic_rows_filtered_or_deduplicated": False,
            "schedule_sensitive_token_differences_preserved": True,
            "document_text": "exact_answers_joined_by_two_newlines",
            "expected_documents_formula": "256_meetings_x_5_replicates_x_3_models",
            "expected_sections_formula": "3840_documents_x_8_core_topics",
        },
        "publication": {
            "publish_attempt_id": publish_attempt_id,
            "staging_basename": staging_basename,
            "final_root": str(final_root),
            "method": "renameat2_RENAME_NOREPLACE_after_file_and_directory_fsync",
            "create_only": True,
            "failure_staging_preserved": True,
            "preflight_free_bytes": preflight_free_bytes,
            "minimum_required_free_bytes": MIN_FREE_BYTES,
            "gpu_work": False,
        },
        "implementation_sources": {
            "v4_assembler": _binding(Path(__file__).resolve()),
            "lossless_assembly_core": _binding(Path(legacy.__file__).resolve()),
            "v4_suite_validator": _binding(Path(suite.__file__).resolve()),
            "v4_generation_runner": _binding(Path(suite.runner.__file__).resolve()),
            "v4_amendment": _binding(Path(amendment.__file__).resolve()),
            "cohort_preparer": _binding(Path(preparation.__file__).resolve()),
            "merged_data_contract": _binding(Path(data_contract.__file__).resolve()),
        },
    }


def assemble(
    *,
    cohort_manifest: Path,
    cohort_manifest_sha256: str,
    suite_manifest: Path,
    output_dir: Path,
    max_num_seqs: int,
) -> dict[str, Any]:
    cohort_path, suite_path, final_root = _require_exact_target_contract(
        cohort_manifest=cohort_manifest,
        cohort_manifest_sha256=cohort_manifest_sha256,
        suite_manifest=suite_manifest,
        output_dir=output_dir,
        max_num_seqs=max_num_seqs,
    )
    with _assembly_lock():
        if os.path.lexists(final_root):
            raise StochasticScheduleMeetingAssemblyError(
                f"final output is create-only and already exists: {final_root}"
            )
        free_bytes = shutil.disk_usage(final_root.parent).free
        if free_bytes < MIN_FREE_BYTES:
            raise StochasticScheduleMeetingAssemblyError(
                f"insufficient free disk before assembly: {free_bytes}"
            )
        cohort, samples, ledger, validated, rows_by_model = _load_inputs(
            cohort_manifest=cohort_path,
            suite_manifest=suite_path,
        )
        try:
            documents = _retag_documents(
                legacy.assemble_documents(
                    samples=samples,
                    token_ledger_rows=ledger,
                    rows_by_model=rows_by_model,
                )
            )
        except Exception as exc:
            raise StochasticScheduleMeetingAssemblyError(str(exc)) from exc
        matrix = _audit_document_matrix(documents, samples=samples)
        attempt = secrets.token_hex(16)
        staging = final_root.with_name(
            f"{final_root.name}.staging.{os.getpid()}.{attempt}"
        )
        if os.path.lexists(staging):
            raise StochasticScheduleMeetingAssemblyError(
                f"unexpected staging collision: {staging}"
            )
        staging.mkdir(mode=0o700, parents=False, exist_ok=False)
        _fsync_directory(staging.parent)
        documents_path = staging / DOCUMENT_FILENAME
        _write_new_jsonl_readonly(documents_path, documents)
        manifest = seal_manifest(
            _manifest_payload(
                created_at_utc=suite.runner.core._utc_now(),
                final_root=final_root,
                documents_physical_path=documents_path,
                cohort_manifest=cohort_path,
                cohort=cohort,
                validated=validated,
                matrix=matrix,
                publish_attempt_id=attempt,
                staging_basename=staging.name,
                preflight_free_bytes=free_bytes,
            )
        )
        _write_new_json_readonly(staging / "manifest.json", manifest)
        staging.chmod(0o555)
        _fsync_directory(staging)
        _fsync_directory(staging.parent)
        _rename_noreplace(staging, final_root)
    loaded = validate_assembly(final_root / "manifest.json")
    return loaded["manifest"]


def validate_assembly(manifest_path: Path) -> dict[str, Any]:
    unresolved = manifest_path.expanduser()
    if (
        unresolved.is_symlink()
        or unresolved.parent.is_symlink()
        or not unresolved.is_file()
    ):
        raise StochasticScheduleMeetingAssemblyError(
            "assembly manifest is missing or a symlink"
        )
    manifest_path = unresolved.resolve()
    final_root = manifest_path.parent
    if (
        final_root != DEFAULT_OUTPUT.resolve()
        or manifest_path != final_root / "manifest.json"
    ):
        raise StochasticScheduleMeetingAssemblyError("assembly final root drift")
    if (
        stat.S_IMODE(final_root.stat().st_mode) != 0o555
        or stat.S_IMODE(manifest_path.stat().st_mode) != 0o444
    ):
        raise StochasticScheduleMeetingAssemblyError("assembly modes are not sealed")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StochasticScheduleMeetingAssemblyError(str(exc)) from exc
    if not isinstance(manifest, dict):
        raise StochasticScheduleMeetingAssemblyError("assembly manifest is not object")
    payload_sha = validate_manifest_integrity(manifest)
    publication = manifest.get("publication")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("immutable") is not True
        or manifest.get("evaluation_id") != suite.runner.EVALUATION_ID
        or manifest.get("evaluation_scope")
        != "formal_merged_panel_meeting_document_reshape"
        or manifest.get("model_order") != list(preparation.MODEL_ORDER)
        or not isinstance(publication, Mapping)
        or publication.get("method")
        != "renameat2_RENAME_NOREPLACE_after_file_and_directory_fsync"
        or publication.get("create_only") is not True
        or publication.get("failure_staging_preserved") is not True
        or publication.get("gpu_work") is not False
    ):
        raise StochasticScheduleMeetingAssemblyError("assembly header/publish drift")
    publish_attempt_id, staging_basename, preflight_free_bytes = _validated_publication(
        publication, final_root=final_root
    )
    if set(path.name for path in final_root.iterdir()) != {
        "manifest.json",
        DOCUMENT_FILENAME,
    }:
        raise StochasticScheduleMeetingAssemblyError("assembly file inventory drift")
    cohort_binding = manifest.get("cohort_manifest")
    suite_binding = manifest.get("formal_suite_manifest")
    documents_binding = manifest.get("meeting_documents")
    if not all(
        isinstance(value, Mapping)
        for value in (cohort_binding, suite_binding, documents_binding)
    ):
        raise StochasticScheduleMeetingAssemblyError("assembly bindings are missing")
    if (
        cohort_binding.get("path") != str(DEFAULT_COHORT.resolve())
        or cohort_binding.get("sha256") != COHORT_SHA256
        or suite_binding.get("path") != str(DEFAULT_SUITE.resolve())
    ):
        raise StochasticScheduleMeetingAssemblyError("cohort/suite path binding drift")
    document_path = final_root / DOCUMENT_FILENAME
    if (
        document_path.is_symlink()
        or not document_path.is_file()
        or stat.S_IMODE(document_path.stat().st_mode) != 0o444
        or documents_binding.get("schema_version") != DOCUMENT_SCHEMA
        or {
            key: documents_binding.get(key)
            for key in ("path", "bytes", "sha256", "rows")
        }
        != _binding(document_path, rows=EXPECTED_DOCUMENTS)
    ):
        raise StochasticScheduleMeetingAssemblyError("document file binding drift")
    cohort, samples, ledger, validated, rows_by_model = _load_inputs(
        cohort_manifest=DEFAULT_COHORT.resolve(),
        suite_manifest=DEFAULT_SUITE.resolve(),
    )
    if suite_binding != validated["manifest_binding"]:
        raise StochasticScheduleMeetingAssemblyError("formal suite binding drift")
    suite_manifest = validated["manifest"]
    for key in (
        "remediation_receipt",
        "speed_selection",
        "schedule_sensitivity_diagnostic",
        "generation_gates",
        "official_smoke_suite",
        "generation_contract",
    ):
        manifest_key = "amendment_receipt" if key == "remediation_receipt" else key
        if manifest.get(manifest_key) != suite_manifest.get(key):
            raise StochasticScheduleMeetingAssemblyError(f"suite binding drift: {key}")
    if manifest.get("source_generation_files") != _source_generation_files(validated):
        raise StochasticScheduleMeetingAssemblyError("generation file bindings drift")
    actual = legacy._read_canonical_jsonl(document_path)
    try:
        expected = _retag_documents(
            legacy.assemble_documents(
                samples=samples,
                token_ledger_rows=ledger,
                rows_by_model=rows_by_model,
            )
        )
    except Exception as exc:
        raise StochasticScheduleMeetingAssemblyError(str(exc)) from exc
    if actual != expected:
        raise StochasticScheduleMeetingAssemblyError(
            "meeting documents are not the exact lossless reshape"
        )
    matrix = _audit_document_matrix(actual, samples=samples)
    if manifest.get("coverage") != matrix:
        raise StochasticScheduleMeetingAssemblyError("coverage matrix drift")
    rebuilt = seal_manifest(
        _manifest_payload(
            created_at_utc=str(manifest.get("created_at_utc")),
            final_root=final_root,
            documents_physical_path=document_path,
            cohort_manifest=DEFAULT_COHORT.resolve(),
            cohort=cohort,
            validated=validated,
            matrix=matrix,
            publish_attempt_id=publish_attempt_id,
            staging_basename=staging_basename,
            preflight_free_bytes=preflight_free_bytes,
        )
    )
    if manifest != rebuilt:
        raise StochasticScheduleMeetingAssemblyError("assembly manifest drift")
    return {
        "status": "valid",
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "payload_sha256": payload_sha,
        "documents": len(actual),
        "sections": matrix["sections"],
        "documents_per_model": matrix["documents_per_model"],
        "meeting_documents_sha256": documents_binding["sha256"],
        "meeting_documents_bytes": documents_binding["bytes"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("assemble")
    build.add_argument("--cohort-manifest", required=True, type=Path)
    build.add_argument("--cohort-manifest-sha256", required=True)
    build.add_argument("--suite-manifest", required=True, type=Path)
    build.add_argument("--output-dir", required=True, type=Path)
    build.add_argument(
        "--max-num-seqs", required=True, type=int, choices=(FIXED_MAX_NUM_SEQS,)
    )
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "assemble":
            manifest = assemble(
                cohort_manifest=args.cohort_manifest,
                cohort_manifest_sha256=args.cohort_manifest_sha256,
                suite_manifest=args.suite_manifest,
                output_dir=args.output_dir,
                max_num_seqs=args.max_num_seqs,
            )
            summary = {
                "status": manifest["status"],
                "documents": manifest["coverage"]["documents"],
                "sections": manifest["coverage"]["sections"],
                "payload_sha256": manifest["integrity"]["payload_sha256"],
            }
        else:
            summary = validate_assembly(args.manifest)
            summary.pop("manifest", None)
        print(_canonical(summary))
        return 0
    except (
        StochasticScheduleMeetingAssemblyError,
        suite.StochasticScheduleSuiteError,
        legacy.VllmK5MeetingAssemblyError,
        OSError,
        ValueError,
    ) as exc:
        print(
            _canonical(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DOCUMENT_SCHEMA",
    "EXPECTED_DOCUMENTS",
    "EXPECTED_SECTIONS",
    "MANIFEST_SCHEMA",
    "StochasticScheduleMeetingAssemblyError",
    "assemble",
    "main",
    "validate_assembly",
]
