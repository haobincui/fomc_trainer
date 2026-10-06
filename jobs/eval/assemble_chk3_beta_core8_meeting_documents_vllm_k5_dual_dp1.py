"""Losslessly assemble dual-independent-DP1 K5 Core8 rows by meeting."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.eval import assemble_chk3_beta_core8_meeting_documents_vllm_k5 as legacy
from jobs.eval import chk3_beta_core8_merged_contract as data_contract
from jobs.eval import prepare_chk3_beta_core8_merged_vllm_k5 as preparation
from jobs.eval import seal_chk3_beta_core8_vllm_k5_dual_dp1_suite as suite
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


DOCUMENT_SCHEMA = "chk3-beta-core8-merged-vllm-k5-dual-dp1-meeting-document-v1"
MANIFEST_SCHEMA = (
    "chk3-beta-core8-merged-vllm-k5-dual-dp1-meeting-documents-manifest-v1"
)
EXPECTED_DOCUMENTS = legacy.EXPECTED_DOCUMENTS
EXPECTED_SECTIONS = preparation.EXPECTED_TOTAL_CASES


class DualDp1MeetingAssemblyError(RuntimeError):
    """The dual-independent-DP1 generation matrix cannot be reshaped exactly."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _binding(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise DualDp1MeetingAssemblyError(f"bound path is a symlink: {unresolved}")
    path = unresolved.resolve()
    if not path.is_file():
        raise DualDp1MeetingAssemblyError(f"bound file is missing: {path}")
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _load_inputs(
    *,
    cohort_manifest: Path,
    cohort_manifest_sha256: str,
    suite_manifest: Path,
    max_num_seqs: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
]:
    try:
        cohort, samples, ledger = legacy._load_cohort(
            cohort_manifest, cohort_manifest_sha256
        )
        validated = suite.load_and_validate_suite(
            suite_manifest,
            cohort_path=cohort_manifest,
            cohort_sha256=cohort_manifest_sha256,
            expected_scope="formal_merged_panel",
            max_num_seqs=max_num_seqs,
        )
    except Exception as exc:
        raise DualDp1MeetingAssemblyError(str(exc)) from exc
    rows_by_model = {
        model_id: [dict(row) for row in validated["runs"][model_id]["results"]]
        for model_id in preparation.MODEL_ORDER
    }
    return cohort, samples, ledger, validated, rows_by_model


def _retag_documents(documents: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for document in documents:
        copied = copy.deepcopy(dict(document))
        copied["schema_version"] = DOCUMENT_SCHEMA
        result.append(copied)
    return result


def assemble(
    *,
    cohort_manifest: Path,
    cohort_manifest_sha256: str,
    suite_manifest: Path,
    output_dir: Path,
    max_num_seqs: int,
) -> dict[str, Any]:
    unresolved = output_dir.expanduser()
    if unresolved.is_symlink() or os.path.lexists(unresolved):
        raise DualDp1MeetingAssemblyError("assembly output root must be fresh")
    cohort, samples, ledger, validated, rows_by_model = _load_inputs(
        cohort_manifest=cohort_manifest,
        cohort_manifest_sha256=cohort_manifest_sha256,
        suite_manifest=suite_manifest,
        max_num_seqs=max_num_seqs,
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
        raise DualDp1MeetingAssemblyError(str(exc)) from exc
    output_dir = unresolved.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    documents_path = output_dir / "meeting_documents.v1.jsonl"
    legacy._write_new_jsonl(documents_path, documents)
    generation_files = {
        model_id: copy.deepcopy(
            validated["runs"][model_id]["manifest"]["artifacts"][
                "canonical_generations"
            ]
        )
        for model_id in preparation.MODEL_ORDER
    }
    exposed_documents = sum(
        bool(document.get("cp318_selection_exposed")) for document in documents
    )
    manifest = seal_manifest(
        {
            "schema_version": MANIFEST_SCHEMA,
            "status": "complete",
            "immutable": True,
            "created_at_utc": validated["manifest"]["created_at_utc"],
            "evaluation_id": suite.runner.EVALUATION_ID,
            "backend_contract": {
                "backend": "vllm-async-engine-v1-two-independent-dp1-workers",
                "independent_engine_count": 2,
                "data_parallel_size_per_worker": 1,
                "tensor_parallel_size_per_worker": 1,
                "pipeline_parallel_size_per_worker": 1,
                "physical_gpu_indexes": [0, 1],
                "shard_function": "absolute_case_index_mod_2",
                "weight_precision": "bfloat16",
                "quantization": None,
                "gpu_memory_utilization_per_worker": 0.95,
                "max_num_seqs_per_worker": max_num_seqs,
                "enforce_eager": False,
                "cuda_graphs": True,
                "async_output_processing": True,
                "mixed_with_dp2_or_nf4_rows": False,
            },
            "model_order": list(preparation.MODEL_ORDER),
            "remediation_receipt": copy.deepcopy(
                validated["manifest"]["remediation_receipt"]
            ),
            "cohort_manifest": {
                **_binding(cohort_manifest),
                "payload_sha256": cohort["integrity"]["payload_sha256"],
            },
            "suite_manifest": copy.deepcopy(validated["manifest_binding"]),
            "source_generation_files": generation_files,
            "meeting_documents": {
                **_binding(documents_path, rows=EXPECTED_DOCUMENTS),
                "schema_version": DOCUMENT_SCHEMA,
            },
            "coverage": {
                "models": 3,
                "meetings": data_contract.EXPECTED_MEETINGS,
                "replicates": len(preparation.REPLICATE_SEEDS),
                "topics_per_document": len(data_contract.CORE_TOPICS),
                "documents": EXPECTED_DOCUMENTS,
                "documents_per_model": legacy.DOCUMENTS_PER_MODEL,
                "sections": EXPECTED_SECTIONS,
                "input_truncation_sections": 0,
                "cp318_selection_exposed_documents": exposed_documents,
            },
            "assembly": {
                "ordering": "model_then_source_meeting_then_replicate",
                "topic_order": list(data_contract.CORE_TOPICS),
                "full_generation_records_preserved": True,
                "prompt_and_generated_token_ids_preserved": True,
                "source_and_generation_hashes_preserved": True,
                "hard_gate_filtering": False,
                "document_text": "exact_answers_joined_by_two_newlines",
            },
            "implementation_sources": {
                "dual_dp1_assembler": _binding(Path(__file__).resolve()),
                "lossless_assembly_core": _binding(Path(legacy.__file__).resolve()),
                "suite_validator": _binding(Path(suite.__file__).resolve()),
            },
        }
    )
    legacy._write_new_json(output_dir / "manifest.json", manifest)
    return manifest


def validate_assembly(manifest_path: Path) -> dict[str, Any]:
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise DualDp1MeetingAssemblyError("assembly manifest is missing or a symlink")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise DualDp1MeetingAssemblyError("assembly manifest is not an object")
    payload_sha = validate_manifest_integrity(manifest)
    backend = manifest.get("backend_contract")
    coverage = manifest.get("coverage")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("immutable") is not True
        or manifest.get("evaluation_id") != suite.runner.EVALUATION_ID
        or manifest.get("model_order") != list(preparation.MODEL_ORDER)
        or not isinstance(manifest.get("remediation_receipt"), Mapping)
        or not isinstance(backend, Mapping)
        or backend.get("backend") != "vllm-async-engine-v1-two-independent-dp1-workers"
        or backend.get("independent_engine_count") != 2
        or backend.get("data_parallel_size_per_worker") != 1
        or backend.get("tensor_parallel_size_per_worker") != 1
        or backend.get("shard_function") != "absolute_case_index_mod_2"
        or backend.get("enforce_eager") is not False
        or backend.get("cuda_graphs") is not True
        or backend.get("async_output_processing") is not True
        or backend.get("mixed_with_dp2_or_nf4_rows") is not False
        or not isinstance(coverage, Mapping)
        or coverage.get("documents") != EXPECTED_DOCUMENTS
        or coverage.get("sections") != EXPECTED_SECTIONS
        or coverage.get("input_truncation_sections") != 0
    ):
        raise DualDp1MeetingAssemblyError("assembly contract drift")
    cohort_binding = manifest.get("cohort_manifest")
    suite_binding = manifest.get("suite_manifest")
    documents_binding = manifest.get("meeting_documents")
    if not all(
        isinstance(value, Mapping)
        for value in (cohort_binding, suite_binding, documents_binding)
    ):
        raise DualDp1MeetingAssemblyError("assembly source bindings are missing")
    cohort_path = Path(str(cohort_binding["path"]))
    suite_path = Path(str(suite_binding["path"]))
    max_num_seqs = int(backend["max_num_seqs_per_worker"])
    cohort, samples, ledger, validated, rows_by_model = _load_inputs(
        cohort_manifest=cohort_path,
        cohort_manifest_sha256=str(cohort_binding["sha256"]),
        suite_manifest=suite_path,
        max_num_seqs=max_num_seqs,
    )
    del cohort
    if suite_binding != validated["manifest_binding"]:
        raise DualDp1MeetingAssemblyError("suite manifest binding drift")
    if manifest.get("remediation_receipt") != validated["manifest"].get(
        "remediation_receipt"
    ):
        raise DualDp1MeetingAssemblyError("remediation receipt binding drift")
    expected_generation_files = {
        model_id: validated["runs"][model_id]["manifest"]["artifacts"][
            "canonical_generations"
        ]
        for model_id in preparation.MODEL_ORDER
    }
    if manifest.get("source_generation_files") != expected_generation_files:
        raise DualDp1MeetingAssemblyError("source generation bindings drift")
    documents_path = Path(str(documents_binding["path"]))
    if documents_binding.get("schema_version") != DOCUMENT_SCHEMA or {
        key: documents_binding.get(key) for key in ("path", "bytes", "sha256", "rows")
    } != _binding(documents_path, rows=EXPECTED_DOCUMENTS):
        raise DualDp1MeetingAssemblyError("meeting-document file binding drift")
    actual = legacy._read_canonical_jsonl(documents_path)
    expected = _retag_documents(
        legacy.assemble_documents(
            samples=samples,
            token_ledger_rows=ledger,
            rows_by_model=rows_by_model,
        )
    )
    if actual != expected:
        raise DualDp1MeetingAssemblyError(
            "meeting documents are not a lossless reshape"
        )
    return {
        "status": "valid",
        "manifest_sha256": sha256_file(manifest_path),
        "payload_sha256": payload_sha,
        "documents": len(actual),
        "sections": sum(len(document["sections"]) for document in actual),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("assemble")
    build.add_argument("--cohort-manifest", type=Path, required=True)
    build.add_argument("--cohort-manifest-sha256", required=True)
    build.add_argument("--suite-manifest", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--max-num-seqs", type=int, choices=(8, 12, 16), required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "assemble":
            result = assemble(
                cohort_manifest=args.cohort_manifest,
                cohort_manifest_sha256=args.cohort_manifest_sha256,
                suite_manifest=args.suite_manifest,
                output_dir=args.output_dir,
                max_num_seqs=args.max_num_seqs,
            )
            summary = {
                "status": result["status"],
                "documents": result["coverage"]["documents"],
                "sections": result["coverage"]["sections"],
            }
        else:
            summary = validate_assembly(args.manifest)
        print(_canonical(summary))
        return 0
    except (
        DualDp1MeetingAssemblyError,
        suite.DualDp1SuiteError,
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
    "DualDp1MeetingAssemblyError",
    "MANIFEST_SCHEMA",
    "assemble",
    "main",
    "validate_assembly",
]
