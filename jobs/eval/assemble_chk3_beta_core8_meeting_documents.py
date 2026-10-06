"""Assemble eight persisted Core8 generations into meeting documents."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.eval import chk3_beta_core8_merged_contract as contract
from jobs.eval import eval_chk3_beta_core8_merged_stochastic_k10 as profile
from jobs.eval import eval_chk3_stochastic_bootstrap_generation as core
from jobs.eval.prepare_chk3_external_holdout_smoke import (
    ExternalSmokeError,
    _binding,
    _write_new_json,
    _write_new_jsonl,
)
from open_r1.provenance import sha256_file, sha256_text
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


DOCUMENT_SCHEMA = "chk3-beta-core8-merged-meeting-document-v1"
MANIFEST_SCHEMA = "chk3-beta-core8-merged-meeting-documents-manifest-v1"
DOWNSTREAM_SCORING_RECOMMENDATION = "score_each_topic_section_then_equal-weight"
EXPECTED_DOCUMENTS = (
    contract.EXPECTED_MEETINGS * len(profile.REPLICATE_SEEDS) * len(core.MODEL_ORDER)
)


class MeetingAssemblyError(RuntimeError):
    """The eight-topic meeting document cannot be assembled losslessly."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def assemble_documents(
    *,
    samples: Sequence[Mapping[str, Any]],
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Build canonical model -> meeting -> replicate documents in memory."""

    sample_by_id = {str(sample["sample_id"]): sample for sample in samples}
    if len(sample_by_id) != contract.EXPECTED_ROWS:
        raise MeetingAssemblyError("sample inventory is not uniquely N2048")
    meeting_ids = tuple(sorted({str(sample["meeting_id"]) for sample in samples}))
    if len(meeting_ids) != contract.EXPECTED_MEETINGS:
        raise MeetingAssemblyError("sample meeting inventory is not N256")
    documents: list[dict[str, Any]] = []
    for model_id in core.MODEL_ORDER:
        rows = rows_by_model.get(model_id)
        if rows is None:
            raise MeetingAssemblyError(f"generation rows missing for {model_id}")
        grouped: dict[tuple[str, int], list[tuple[int, Mapping[str, Any]]]] = (
            defaultdict(list)
        )
        for generation_line, row in enumerate(rows, 1):
            sample_id = str(row.get("sample_id") or "")
            sample = sample_by_id.get(sample_id)
            if sample is None:
                raise MeetingAssemblyError(
                    f"{model_id} generation has unknown sample: {sample_id}"
                )
            if row.get("model_id") != model_id:
                raise MeetingAssemblyError(
                    f"model identity drift at {model_id}:{generation_line}"
                )
            replicate_id = row.get("replicate_id")
            if (
                isinstance(replicate_id, bool)
                or not isinstance(replicate_id, int)
                or replicate_id < 0
                or replicate_id >= len(profile.REPLICATE_SEEDS)
            ):
                raise MeetingAssemblyError("invalid replicate identity")
            grouped[(str(sample["meeting_id"]), replicate_id)].append(
                (generation_line, row)
            )
        expected_group_count = contract.EXPECTED_MEETINGS * len(profile.REPLICATE_SEEDS)
        if len(grouped) != expected_group_count:
            raise MeetingAssemblyError(
                f"{model_id} meeting/replicate closure is not {expected_group_count}"
            )
        for meeting_id in meeting_ids:
            for replicate_id, replicate_seed in enumerate(profile.REPLICATE_SEEDS):
                entries = grouped.get((meeting_id, replicate_id))
                if entries is None or len(entries) != len(contract.CORE_TOPICS):
                    raise MeetingAssemblyError(
                        f"{model_id}:{meeting_id}:r{replicate_id} is not exactly Core8"
                    )
                ordered = sorted(entries, key=lambda item: int(item[1]["topic_order"]))
                topics = tuple(str(row["topic"]) for _, row in ordered)
                if topics != contract.CORE_TOPICS:
                    raise MeetingAssemblyError(
                        f"{model_id}:{meeting_id}:r{replicate_id} topic order drift"
                    )
                invariants = (
                    "era",
                    "source_split",
                    "original_post_split_role",
                    "original_qa_split",
                    "meeting_type",
                    "sensitivity_flag",
                    "sensitivity_reason",
                    "scheduled",
                    "meeting_start_date",
                    "meeting_end_date",
                    "cp318_selection_exposed",
                    "research_scope",
                    "transport_split_role",
                    "not_all_held_out",
                )
                first = ordered[0][1]
                for _, row in ordered[1:]:
                    if any(row.get(key) != first.get(key) for key in invariants):
                        raise MeetingAssemblyError(
                            f"{model_id}:{meeting_id}:r{replicate_id} meeting metadata drift"
                        )
                sections: list[dict[str, Any]] = []
                answers: list[str] = []
                for generation_line, row in ordered:
                    answer = row.get("answer")
                    if not isinstance(answer, str):
                        raise MeetingAssemblyError("generation answer is not text")
                    if sha256_text(answer) != row.get("answer_sha256"):
                        raise MeetingAssemblyError("generation answer SHA drift")
                    answers.append(answer.strip())
                    sections.append(
                        {
                            "topic": row["topic"],
                            "topic_order": row["topic_order"],
                            "sample_id": row["sample_id"],
                            "source_sample_id": row["source_sample_id"],
                            "generation_line_number": generation_line,
                            "row_seed": row["row_seed"],
                            "finish_reason": row["finish_reason"],
                            "answer": answer,
                            "answer_sha256": row["answer_sha256"],
                            "generated_text_sha256": row["completion_sha256"],
                            "generated_token_ids_sha256": row[
                                "generated_token_ids_sha256"
                            ],
                            "input_truncated": row["input_truncated"],
                        }
                    )
                document_text = "\n\n".join(answers)
                document_id = f"{model_id}::{meeting_id}::replicate-{replicate_id:02d}"
                documents.append(
                    {
                        "schema_version": DOCUMENT_SCHEMA,
                        "document_id": document_id,
                        "model_id": model_id,
                        "model_label": first["model_label"],
                        "meeting_id": meeting_id,
                        "era": first["era"],
                        "source_split": first["source_split"],
                        "original_post_split_role": first["original_post_split_role"],
                        "original_qa_split": first["original_qa_split"],
                        "meeting_type": first["meeting_type"],
                        "sensitivity_flag": first["sensitivity_flag"],
                        "sensitivity_reason": first["sensitivity_reason"],
                        "scheduled": first["scheduled"],
                        "meeting_start_date": first["meeting_start_date"],
                        "meeting_end_date": first["meeting_end_date"],
                        "cp318_selection_exposed": first["cp318_selection_exposed"],
                        "research_scope": first["research_scope"],
                        "transport_split_role": first["transport_split_role"],
                        "not_all_held_out": first["not_all_held_out"],
                        "replicate_id": replicate_id,
                        "replicate_seed": replicate_seed,
                        "topic_order": list(contract.CORE_TOPICS),
                        "section_count": len(sections),
                        "sections": sections,
                        "document_text": document_text,
                        "document_text_sha256": sha256_text(document_text),
                        "assembly_separator": "two_newlines_no_topic_labels",
                        "document_text_not_direct_512_token_model_input": True,
                        "recommended_downstream_scoring": (
                            DOWNSTREAM_SCORING_RECOMMENDATION
                        ),
                        "input_truncation_sections": sum(
                            bool(section["input_truncated"]) for section in sections
                        ),
                    }
                )
    if len(documents) != EXPECTED_DOCUMENTS:
        raise MeetingAssemblyError("meeting document closure is not N7680")
    return documents


def assemble(
    *,
    pre_release_manifest: Path,
    post_release_manifest: Path,
    pre_release_sha256: str | None,
    post_release_sha256: str | None,
    suite_manifest: Path,
    sample_manifest: Path,
    sample_manifest_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    unresolved_output = output_dir.expanduser()
    if unresolved_output.is_symlink():
        raise MeetingAssemblyError(
            f"refusing symlink output directory: {unresolved_output}"
        )
    output_dir = unresolved_output.resolve()
    if output_dir.exists():
        raise MeetingAssemblyError(f"refusing to reuse output directory: {output_dir}")
    sources = profile.configure_profile(
        pre_release_manifest=pre_release_manifest,
        post_release_manifest=post_release_manifest,
        pre_release_sha256=pre_release_sha256,
        post_release_sha256=post_release_sha256,
    )
    validated = core.load_and_validate_suite(
        suite_manifest,
        sample_manifest_path=sample_manifest,
        sample_manifest_sha256=sample_manifest_sha256,
        expected_scope="formal_full_test",
    )
    sample_payload, observed_sample_sha = core.load_full_test_sample_manifest(
        sample_manifest, sample_manifest_sha256
    )
    documents = assemble_documents(
        samples=sample_payload["samples"],
        rows_by_model={
            model_id: validated["runs"][model_id]["results"]
            for model_id in core.MODEL_ORDER
        },
    )
    exposed_documents = sum(bool(row["cp318_selection_exposed"]) for row in documents)
    if exposed_documents != 9 * len(profile.REPLICATE_SEEDS) * len(core.MODEL_ORDER):
        raise MeetingAssemblyError("cp318 selection-exposed document closure drift")
    documents_path = output_dir / "meeting_documents.jsonl"
    _write_new_jsonl(documents_path, documents)
    source_generation_files = {
        model_id: {
            **_binding(suite_manifest.parent / model_id / "generations.jsonl"),
            "rows": contract.EXPECTED_ROWS * len(profile.REPLICATE_SEEDS),
        }
        for model_id in core.MODEL_ORDER
    }
    payload = seal_manifest(
        {
            "schema_version": MANIFEST_SCHEMA,
            "status": "complete",
            "evaluation_id": core.EVALUATION_ID,
            "research_scope": profile.RESEARCH_SCOPE,
            "transport_split_role": profile.TRANSPORT_SPLIT_ROLE,
            "not_all_held_out": True,
            "model_order": list(core.MODEL_ORDER),
            "sample_manifest": {
                "path": str(sample_manifest.expanduser().resolve()),
                "sha256": observed_sample_sha,
                "payload_sha256": sample_payload["integrity"]["payload_sha256"],
            },
            "suite_manifest": validated["manifest_binding"],
            "source_generation_files": source_generation_files,
            "source_releases": dict(sources.source_bindings),
            "meeting_documents": {
                **_binding(documents_path, rows=EXPECTED_DOCUMENTS),
                "schema_version": DOCUMENT_SCHEMA,
            },
            "coverage": {
                "documents": EXPECTED_DOCUMENTS,
                "documents_per_model": (
                    contract.EXPECTED_MEETINGS * len(profile.REPLICATE_SEEDS)
                ),
                "meetings": contract.EXPECTED_MEETINGS,
                "replicates": len(profile.REPLICATE_SEEDS),
                "topics_per_document": len(contract.CORE_TOPICS),
                "input_truncation_sections": sum(
                    int(row["input_truncation_sections"]) for row in documents
                ),
                "cp318_selection_exposed_meetings": 9,
                "cp318_selection_exposed_documents": exposed_documents,
            },
            "assembly": {
                "ordering": "model_order_then_meeting_end_date_then_replicate_id",
                "topic_order": list(contract.CORE_TOPICS),
                "document_text": "eight_answers_joined_by_two_newlines_without_labels",
                "document_text_not_direct_512_token_model_input": True,
                "recommended_downstream_scoring": (DOWNSTREAM_SCORING_RECOMMENDATION),
                "direct_document_scoring_requires_versioned_chunk_policy": True,
                "hard_gate_filtering": False,
                "empty_answers_preserved": True,
            },
        }
    )
    manifest_path = output_dir / "manifest.json"
    _write_new_json(manifest_path, payload)
    return payload


def validate_assembly(manifest_path: Path) -> dict[str, Any]:
    unresolved_manifest = manifest_path.expanduser()
    if unresolved_manifest.is_symlink():
        raise MeetingAssemblyError("assembly manifest is a symlink")
    manifest_path = unresolved_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        payload_sha = validate_manifest_integrity(manifest)
    except Exception as exc:
        raise MeetingAssemblyError(
            f"assembly manifest integrity failed: {exc}"
        ) from exc
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("coverage", {}).get("documents") != EXPECTED_DOCUMENTS
        or manifest.get("coverage", {}).get("input_truncation_sections") != 0
        or manifest.get("assembly", {}).get(
            "document_text_not_direct_512_token_model_input"
        )
        is not True
        or manifest.get("assembly", {}).get("recommended_downstream_scoring")
        != DOWNSTREAM_SCORING_RECOMMENDATION
        or manifest.get("assembly", {}).get(
            "direct_document_scoring_requires_versioned_chunk_policy"
        )
        is not True
    ):
        raise MeetingAssemblyError("assembly manifest contract drift")
    binding = manifest.get("meeting_documents")
    if not isinstance(binding, Mapping):
        raise MeetingAssemblyError("assembly document binding missing")
    unresolved_path = Path(str(binding.get("path"))).expanduser()
    if unresolved_path.is_symlink():
        raise MeetingAssemblyError("assembly document file is a symlink")
    path = unresolved_path.resolve()
    if (
        not path.is_file()
        or path.parent != manifest_path.parent
        or binding.get("sha256") != sha256_file(path)
        or binding.get("bytes") != path.stat().st_size
        or binding.get("rows") != EXPECTED_DOCUMENTS
        or _jsonl_row_count(path) != EXPECTED_DOCUMENTS
    ):
        raise MeetingAssemblyError("assembly document binding drift")
    return {
        "status": "valid",
        "manifest_sha256": sha256_file(manifest_path),
        "payload_sha256": payload_sha,
        "documents": EXPECTED_DOCUMENTS,
    }


def _jsonl_row_count(path: Path) -> int:
    """Count complete JSONL records without loading large meeting texts."""

    rows = 0
    last_byte = b""
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            rows += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if path.stat().st_size and last_byte != b"\n":
        raise MeetingAssemblyError("assembly document JSONL has an incomplete tail")
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("assemble")
    build.add_argument(
        "--pre-release", type=Path, default=contract.PRE_RELEASE_MANIFEST
    )
    build.add_argument(
        "--post-release", type=Path, default=contract.POST_RELEASE_MANIFEST
    )
    build.add_argument("--pre-release-sha256")
    build.add_argument("--post-release-sha256", required=True)
    build.add_argument("--suite-manifest", required=True, type=Path)
    build.add_argument("--sample-manifest", required=True, type=Path)
    build.add_argument("--sample-manifest-sha256", required=True)
    build.add_argument("--output-dir", required=True, type=Path)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "assemble":
            result = assemble(
                pre_release_manifest=args.pre_release,
                post_release_manifest=args.post_release,
                pre_release_sha256=args.pre_release_sha256,
                post_release_sha256=args.post_release_sha256,
                suite_manifest=args.suite_manifest,
                sample_manifest=args.sample_manifest,
                sample_manifest_sha256=args.sample_manifest_sha256,
                output_dir=args.output_dir,
            )
            summary = {
                "status": result["status"],
                "documents": result["coverage"]["documents"],
                "payload_sha256": result["integrity"]["payload_sha256"],
            }
        else:
            summary = validate_assembly(args.manifest)
        print(_canonical(summary))
        return 0
    except (
        MeetingAssemblyError,
        ExternalSmokeError,
        core.StochasticBootstrapGenerationError,
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
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DOCUMENT_SCHEMA",
    "EXPECTED_DOCUMENTS",
    "MANIFEST_SCHEMA",
    "MeetingAssemblyError",
    "assemble",
    "assemble_documents",
    "validate_assembly",
]
