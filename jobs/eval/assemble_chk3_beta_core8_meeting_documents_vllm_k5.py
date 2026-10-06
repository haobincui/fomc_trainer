"""Losslessly assemble the sealed vLLM K=5 Core8 rows into meeting documents.

This module is intentionally independent of the older NF4/K=10 assembler.  A
section is the complete persisted generation record plus four assembly-only
provenance fields, so no generated text, token ID, hash, source field, or
selection-exposure flag is discarded during the topic-to-meeting reshape.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.eval import chk3_beta_core8_merged_contract as data_contract
from jobs.eval import prepare_chk3_beta_core8_merged_vllm_k5 as preparation
from jobs.eval.prepare_chk3_external_holdout_smoke import (
    ExternalSmokeError,
    _binding,
    _write_new_json,
    _write_new_jsonl,
)
from open_r1.provenance import sha256_file, sha256_text
from open_r1.validator.loo_generation_spec import (
    derive_row_seed,
    seal_manifest,
    validate_manifest_integrity,
)


DOCUMENT_SCHEMA = "chk3-beta-core8-merged-vllm-k5-meeting-document-v1"
MANIFEST_SCHEMA = "chk3-beta-core8-merged-vllm-k5-meeting-documents-manifest-v1"
EXPECTED_DOCUMENTS = (
    data_contract.EXPECTED_MEETINGS
    * len(preparation.REPLICATE_SEEDS)
    * len(preparation.MODEL_ORDER)
)
EXPECTED_SELECTION_EXPOSED_MEETINGS = len(
    data_contract.CP318_SELECTION_EXPOSED_MEETINGS
)
DOCUMENTS_PER_MODEL = data_contract.EXPECTED_MEETINGS * len(preparation.REPLICATE_SEEDS)
SECTION_ASSEMBLY_KEYS = frozenset(
    {
        "source_generation_line_number",
        "generation_record_sha256",
        "source_sample_record_sha256",
        "prompt_token_ledger_row_sha256",
        "input_prompt_token_ids",
        "input_prompt_token_ids_sha256",
    }
)
SOURCE_HASH_PAIRS = (
    ("source_prompt", "source_prompt_sha256"),
    ("source_analysis", "source_analysis_sha256"),
    ("reference_minutes", "reference_minutes_sha256"),
)


class VllmK5MeetingAssemblyError(RuntimeError):
    """The vLLM K=5 generation matrix cannot be reshaped losslessly."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _record_sha256(value: Mapping[str, Any]) -> str:
    return sha256_text(_canonical(dict(value)))


def _token_ids_sha256(value: Sequence[int]) -> str:
    return sha256_text(_canonical(list(value)))


def _read_json(path: Path, *, sealed: bool = True) -> dict[str, Any]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise VllmK5MeetingAssemblyError(f"refusing symlink JSON: {unresolved}")
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise VllmK5MeetingAssemblyError(f"missing JSON: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VllmK5MeetingAssemblyError(f"invalid JSON {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise VllmK5MeetingAssemblyError(f"JSON root is not an object: {resolved}")
    if sealed:
        try:
            validate_manifest_integrity(value)
        except Exception as exc:
            raise VllmK5MeetingAssemblyError(
                f"sealed JSON integrity failed for {resolved}: {exc}"
            ) from exc
    return value


def _read_canonical_jsonl(path: Path) -> list[dict[str, Any]]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise VllmK5MeetingAssemblyError(f"refusing symlink JSONL: {unresolved}")
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise VllmK5MeetingAssemblyError(f"missing JSONL: {resolved}")
    payload = resolved.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise VllmK5MeetingAssemblyError(f"incomplete JSONL tail: {resolved}")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(payload.splitlines(keepends=True), 1):
        try:
            text = raw[:-1].decode("utf-8")
            value = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VllmK5MeetingAssemblyError(
                f"invalid JSONL row {resolved}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise VllmK5MeetingAssemblyError(
                f"JSONL row is not an object: {resolved}:{line_number}"
            )
        if (_canonical(value) + "\n").encode("utf-8") != raw:
            raise VllmK5MeetingAssemblyError(
                f"noncanonical JSONL row: {resolved}:{line_number}"
            )
        rows.append(value)
    return rows


def _check_text_hash(row: Mapping[str, Any], text_key: str, hash_key: str) -> None:
    text = row.get(text_key)
    observed_hash = row.get(hash_key)
    if not isinstance(text, str) or observed_hash != sha256_text(text):
        raise VllmK5MeetingAssemblyError(f"{text_key}/{hash_key} content binding drift")


def _generation_text_hash_key(row: Mapping[str, Any]) -> str:
    for key in ("generated_text_sha256", "completion_sha256"):
        if isinstance(row.get(key), str):
            return key
    raise VllmK5MeetingAssemblyError("generated-text SHA field is missing")


def _validate_generation_record(
    row: Mapping[str, Any],
    *,
    model_id: str,
    sample: Mapping[str, Any],
    ledger: Mapping[str, Any],
    replicate_id: int,
    replicate_seed: int,
) -> None:
    sample_id = str(sample["sample_id"])
    exact_values = {
        "model_id": model_id,
        "sample_id": sample_id,
        "meeting_id": sample["meeting_id"],
        "topic": sample["topic"],
        "topic_order": sample["topic_order"],
        "replicate_id": replicate_id,
        "replicate_seed": replicate_seed,
        "row_seed": derive_row_seed(replicate_seed, sample_id),
        "cp318_selection_exposed": sample["cp318_selection_exposed"],
    }
    for key, expected in exact_values.items():
        if row.get(key) != expected:
            raise VllmK5MeetingAssemblyError(
                f"generation value drift for {model_id}:{sample_id}:r{replicate_id}:{key}"
            )
    for key in SECTION_ASSEMBLY_KEYS:
        if key in row:
            raise VllmK5MeetingAssemblyError(
                f"generation row already contains reserved assembly key: {key}"
            )
    if row.get("input_truncated") is not False:
        raise VllmK5MeetingAssemblyError("input truncation is not allowed")
    _check_text_hash(row, "answer", "answer_sha256")
    _check_text_hash(row, "generated_text", _generation_text_hash_key(row))
    for text_key, hash_key in SOURCE_HASH_PAIRS:
        _check_text_hash(row, text_key, hash_key)
        sample_hash = sample.get(hash_key)
        if isinstance(sample_hash, str) and row.get(hash_key) != sample_hash:
            raise VllmK5MeetingAssemblyError(
                f"source sample hash drift for {sample_id}:{hash_key}"
            )
    token_ids = row.get("generated_token_ids")
    if (
        not isinstance(token_ids, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in token_ids
        )
        or row.get("generated_token_ids_sha256") != _token_ids_sha256(token_ids)
    ):
        raise VllmK5MeetingAssemblyError("generated-token ID binding drift")
    prompt_token_ids = ledger.get("prompt_token_ids")
    if not isinstance(prompt_token_ids, list) or _token_ids_sha256(
        prompt_token_ids
    ) != ledger.get("prompt_token_ids_sha256"):
        raise VllmK5MeetingAssemblyError("prompt-token ID ledger drift")
    if row.get("prompt_token_ids_sha256") != ledger.get("prompt_token_ids_sha256"):
        raise VllmK5MeetingAssemblyError("generation prompt-token SHA drift")


def assemble_documents(
    *,
    samples: Sequence[Mapping[str, Any]],
    token_ledger_rows: Sequence[Mapping[str, Any]],
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Create canonical model -> meeting -> replicate documents in memory."""

    if len(samples) != data_contract.EXPECTED_ROWS:
        raise VllmK5MeetingAssemblyError("sample inventory is not N=2,048")
    sample_by_id = {str(row.get("sample_id") or ""): row for row in samples}
    ledger_by_id = {str(row.get("sample_id") or ""): row for row in token_ledger_rows}
    if len(sample_by_id) != len(samples) or len(ledger_by_id) != len(samples):
        raise VllmK5MeetingAssemblyError("sample/token-ledger identity is not unique")
    canonical_sample_ids = [str(row["sample_id"]) for row in samples]
    if [str(row.get("sample_id")) for row in token_ledger_rows] != canonical_sample_ids:
        raise VllmK5MeetingAssemblyError("token-ledger order differs from sample order")
    meeting_ids: list[str] = []
    for sample in samples:
        meeting_id = str(sample["meeting_id"])
        if not meeting_ids or meeting_ids[-1] != meeting_id:
            meeting_ids.append(meeting_id)
    if len(meeting_ids) != data_contract.EXPECTED_MEETINGS:
        raise VllmK5MeetingAssemblyError("meeting inventory is not N=256")

    documents: list[dict[str, Any]] = []
    expected_cases = len(samples) * len(preparation.REPLICATE_SEEDS)
    for model_id in preparation.MODEL_ORDER:
        rows = rows_by_model.get(model_id)
        if rows is None or len(rows) != expected_cases:
            raise VllmK5MeetingAssemblyError(
                f"{model_id} generation inventory is not N={expected_cases:,}"
            )
        grouped: dict[tuple[str, int], list[tuple[int, Mapping[str, Any]]]] = (
            defaultdict(list)
        )
        for zero_index, row in enumerate(rows):
            sample_index, replicate_id = divmod(
                zero_index, len(preparation.REPLICATE_SEEDS)
            )
            sample = samples[sample_index]
            ledger = token_ledger_rows[sample_index]
            replicate_seed = preparation.REPLICATE_SEEDS[replicate_id]
            _validate_generation_record(
                row,
                model_id=model_id,
                sample=sample,
                ledger=ledger,
                replicate_id=replicate_id,
                replicate_seed=replicate_seed,
            )
            grouped[(str(sample["meeting_id"]), replicate_id)].append(
                (zero_index + 1, row)
            )
        if len(grouped) != DOCUMENTS_PER_MODEL:
            raise VllmK5MeetingAssemblyError(
                f"{model_id} meeting/replicate closure is not N={DOCUMENTS_PER_MODEL}"
            )

        for meeting_id in meeting_ids:
            for replicate_id, replicate_seed in enumerate(preparation.REPLICATE_SEEDS):
                source_entries = grouped.get((meeting_id, replicate_id))
                if source_entries is None or len(source_entries) != len(
                    data_contract.CORE_TOPICS
                ):
                    raise VllmK5MeetingAssemblyError(
                        f"{model_id}:{meeting_id}:r{replicate_id} is not Core8"
                    )
                entries = sorted(
                    source_entries, key=lambda item: int(item[1]["topic_order"])
                )
                if tuple(str(row["topic"]) for _, row in entries) != tuple(
                    data_contract.CORE_TOPICS
                ):
                    raise VllmK5MeetingAssemblyError(
                        f"{model_id}:{meeting_id}:r{replicate_id} topic order drift"
                    )
                first = entries[0][1]
                meeting_fields = (
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
                    "evidence_cutoff",
                    "cp318_selection_exposed",
                    "research_scope",
                    "transport_split_role",
                    "not_all_held_out",
                )
                if any(
                    any(row.get(key) != first.get(key) for key in meeting_fields)
                    for _, row in entries[1:]
                ):
                    raise VllmK5MeetingAssemblyError(
                        f"{model_id}:{meeting_id}:r{replicate_id} metadata drift"
                    )
                sections: list[dict[str, Any]] = []
                for source_line, row in entries:
                    sample = sample_by_id[str(row["sample_id"])]
                    ledger = ledger_by_id[str(row["sample_id"])]
                    section = copy.deepcopy(dict(row))
                    section.update(
                        {
                            "source_generation_line_number": source_line,
                            "generation_record_sha256": _record_sha256(row),
                            "source_sample_record_sha256": _record_sha256(sample),
                            "prompt_token_ledger_row_sha256": _record_sha256(ledger),
                            "input_prompt_token_ids": copy.deepcopy(
                                ledger["prompt_token_ids"]
                            ),
                            "input_prompt_token_ids_sha256": ledger[
                                "prompt_token_ids_sha256"
                            ],
                        }
                    )
                    sections.append(section)
                document_text = "\n\n".join(str(row["answer"]) for row in sections)
                document_id = f"{model_id}::{meeting_id}::replicate-{replicate_id:02d}"
                documents.append(
                    {
                        "schema_version": DOCUMENT_SCHEMA,
                        "document_id": document_id,
                        "model_id": model_id,
                        "model_label": first.get("model_label"),
                        "meeting_id": meeting_id,
                        **{
                            key: copy.deepcopy(first.get(key)) for key in meeting_fields
                        },
                        "replicate_id": replicate_id,
                        "replicate_seed": replicate_seed,
                        "topic_order": list(data_contract.CORE_TOPICS),
                        "section_count": len(sections),
                        "sections": sections,
                        "document_text": document_text,
                        "document_text_sha256": sha256_text(document_text),
                        "assembly_separator": "two_newlines_exact_section_answers",
                        "full_generation_records_preserved": True,
                        "hard_gate_filtering": False,
                        "input_truncation_sections": 0,
                    }
                )
    if len(documents) != EXPECTED_DOCUMENTS:
        raise VllmK5MeetingAssemblyError(
            f"meeting document closure is not N={EXPECTED_DOCUMENTS:,}"
        )
    return documents


def _validate_binding(
    binding: Mapping[str, Any],
    *,
    expected_path: Path | None = None,
    rows: int | None = None,
) -> Path:
    path = Path(str(binding.get("path") or "")).expanduser()
    if path.is_symlink():
        raise VllmK5MeetingAssemblyError(f"bound path is a symlink: {path}")
    resolved = path.resolve()
    if expected_path is not None and resolved != expected_path.resolve():
        raise VllmK5MeetingAssemblyError("bound path drift")
    if (
        not resolved.is_file()
        or binding.get("sha256") != sha256_file(resolved)
        or binding.get("bytes") != resolved.stat().st_size
        or (rows is not None and binding.get("rows") != rows)
    ):
        raise VllmK5MeetingAssemblyError(f"file binding drift: {resolved}")
    return resolved


def _load_cohort(
    cohort_path: Path, cohort_sha256: str
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    cohort = _read_json(cohort_path)
    if sha256_file(cohort_path) != cohort_sha256:
        raise VllmK5MeetingAssemblyError("cohort manifest SHA drift")
    if (
        cohort.get("schema_version") != preparation.COHORT_SCHEMA
        or cohort.get("status") != "complete"
        or cohort.get("evaluation_id") != preparation.EVALUATION_ID
        or cohort.get("generation_design", {}).get("replicate_seeds")
        != list(preparation.REPLICATE_SEEDS)
        or cohort.get("generation_design", {}).get("models")
        != list(preparation.MODEL_ORDER)
        or cohort.get("generation_design", {}).get("total_rows")
        != preparation.EXPECTED_TOTAL_CASES
    ):
        raise VllmK5MeetingAssemblyError("K=5 cohort contract drift")
    sample_binding = cohort.get("source_sample_manifest")
    ledger_binding = cohort.get("token_ledger")
    if not isinstance(sample_binding, Mapping) or not isinstance(
        ledger_binding, Mapping
    ):
        raise VllmK5MeetingAssemblyError("cohort source bindings are missing")
    sample_path = Path(str(sample_binding.get("path") or "")).expanduser().resolve()
    if sample_path.is_symlink() or sha256_file(sample_path) != sample_binding.get(
        "sha256"
    ):
        raise VllmK5MeetingAssemblyError("source sample-manifest binding drift")
    sample_manifest = _read_json(sample_path)
    samples = sample_manifest.get("samples")
    if not isinstance(samples, list):
        raise VllmK5MeetingAssemblyError("source sample list is missing")
    ledger_path = _validate_binding(ledger_binding, rows=preparation.EXPECTED_PROMPTS)
    ledger_rows = _read_canonical_jsonl(ledger_path)
    if len(ledger_rows) != preparation.EXPECTED_PROMPTS:
        raise VllmK5MeetingAssemblyError("prompt-token ledger row count drift")
    return cohort, samples, ledger_rows


def _load_validated_suite(
    *,
    suite_manifest: Path,
    cohort_manifest: Path,
    cohort_sha256: str,
    max_num_seqs: int,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Delegate generation validation to the versioned vLLM runner."""

    try:
        from jobs.eval import eval_chk3_beta_core8_merged_vllm_k5 as runner
    except ImportError as exc:  # pragma: no cover - catches incomplete deployments
        raise VllmK5MeetingAssemblyError(
            "the versioned vLLM K=5 runner is not installed"
        ) from exc
    try:
        validated = runner.load_and_validate_suite(
            suite_manifest,
            cohort_path=cohort_manifest,
            cohort_sha256=cohort_sha256,
            expected_scope="formal_merged_panel",
            max_num_seqs=max_num_seqs,
        )
    except Exception as exc:
        raise VllmK5MeetingAssemblyError(
            f"vLLM K=5 suite validation failed: {exc}"
        ) from exc
    if not isinstance(validated, Mapping):
        raise VllmK5MeetingAssemblyError("runner returned no validated suite")
    runs = validated.get("runs")
    if not isinstance(runs, Mapping):
        raise VllmK5MeetingAssemblyError("validated suite has no model runs")
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for model_id in preparation.MODEL_ORDER:
        run = runs.get(model_id)
        results = run.get("results") if isinstance(run, Mapping) else None
        if not isinstance(results, list) or any(
            not isinstance(row, dict) for row in results
        ):
            raise VllmK5MeetingAssemblyError(
                f"validated suite has no {model_id} generation rows"
            )
        rows_by_model[model_id] = results
    return dict(validated), rows_by_model


def assemble(
    *,
    cohort_manifest: Path,
    cohort_manifest_sha256: str,
    suite_manifest: Path,
    output_dir: Path,
    max_num_seqs: int,
) -> dict[str, Any]:
    unresolved_output = output_dir.expanduser()
    if unresolved_output.is_symlink():
        raise VllmK5MeetingAssemblyError("refusing a symlink output directory")
    output_dir = unresolved_output.resolve()
    if output_dir.exists():
        raise VllmK5MeetingAssemblyError(f"refusing to reuse output: {output_dir}")
    cohort, samples, ledger_rows = _load_cohort(cohort_manifest, cohort_manifest_sha256)
    validated, rows_by_model = _load_validated_suite(
        suite_manifest=suite_manifest,
        cohort_manifest=cohort_manifest,
        cohort_sha256=cohort_manifest_sha256,
        max_num_seqs=max_num_seqs,
    )
    documents = assemble_documents(
        samples=samples,
        token_ledger_rows=ledger_rows,
        rows_by_model=rows_by_model,
    )
    documents_path = output_dir / "meeting_documents.v1.jsonl"
    _write_new_jsonl(documents_path, documents)
    source_generation_files = {
        model_id: _binding(
            suite_manifest.expanduser().resolve().parent
            / model_id
            / "generations.canonical.v1.jsonl",
            rows=preparation.EXPECTED_CASES_PER_MODEL,
        )
        for model_id in preparation.MODEL_ORDER
    }
    exposed_documents = sum(
        bool(document["cp318_selection_exposed"]) for document in documents
    )
    expected_exposed = (
        EXPECTED_SELECTION_EXPOSED_MEETINGS
        * len(preparation.REPLICATE_SEEDS)
        * len(preparation.MODEL_ORDER)
    )
    if exposed_documents != expected_exposed:
        raise VllmK5MeetingAssemblyError("selection-exposed document closure drift")
    suite_generation_contract = validated["manifest"].get("generation_contract")
    if (
        not isinstance(suite_generation_contract, Mapping)
        or suite_generation_contract.get("backend")
        != "vllm-async-engine-v1-continuous-batching"
        or suite_generation_contract.get("data_parallel_size") != 2
        or suite_generation_contract.get("tensor_parallel_size") != 1
        or suite_generation_contract.get("pipeline_parallel_size") != 1
        or suite_generation_contract.get("physical_gpu_indexes") != [0, 1]
        or suite_generation_contract.get("gpu_memory_utilization") != 0.95
        or suite_generation_contract.get("max_num_seqs") != max_num_seqs
        or suite_generation_contract.get("enforce_eager") is not True
    ):
        raise VllmK5MeetingAssemblyError("suite DP2 generation contract drift")
    manifest = seal_manifest(
        {
            "schema_version": MANIFEST_SCHEMA,
            "status": "complete",
            "immutable": True,
            "evaluation_id": preparation.EVALUATION_ID,
            "backend_contract": {
                "backend": "vllm-async-engine-v1-continuous-batching",
                "weight_precision": "bfloat16",
                "quantization": None,
                "replicates": 5,
                "data_parallel_size": 2,
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "physical_gpu_indexes": [0, 1],
                "gpu_memory_utilization_per_replica": 0.95,
                "max_num_seqs": max_num_seqs,
                "enforce_eager": True,
                "full_bf16_replica_per_gpu": True,
                "mixed_with_nf4_k10_rows": False,
            },
            "model_order": list(preparation.MODEL_ORDER),
            "cohort_manifest": {
                **_binding(cohort_manifest),
                "payload_sha256": cohort["integrity"]["payload_sha256"],
            },
            "suite_manifest": copy.deepcopy(validated["manifest_binding"]),
            "source_generation_files": source_generation_files,
            "meeting_documents": {
                **_binding(documents_path, rows=EXPECTED_DOCUMENTS),
                "schema_version": DOCUMENT_SCHEMA,
            },
            "coverage": {
                "models": len(preparation.MODEL_ORDER),
                "meetings": data_contract.EXPECTED_MEETINGS,
                "replicates": len(preparation.REPLICATE_SEEDS),
                "topics_per_document": len(data_contract.CORE_TOPICS),
                "documents": EXPECTED_DOCUMENTS,
                "documents_per_model": DOCUMENTS_PER_MODEL,
                "sections": preparation.EXPECTED_TOTAL_CASES,
                "input_truncation_sections": 0,
                "cp318_selection_exposed_meetings": (
                    EXPECTED_SELECTION_EXPOSED_MEETINGS
                ),
                "cp318_selection_exposed_documents": exposed_documents,
            },
            "assembly": {
                "ordering": "model_order_then_source_meeting_order_then_replicate_id",
                "topic_order": list(data_contract.CORE_TOPICS),
                "section_payload": "complete_generation_record_plus_assembly_provenance",
                "full_generation_records_preserved": True,
                "generated_text_preserved": True,
                "answers_preserved": True,
                "prompt_and_generated_token_ids_preserved": True,
                "source_metadata_preserved": True,
                "source_and_generation_hashes_preserved": True,
                "selection_exposure_flags_preserved": True,
                "hard_gate_filtering": False,
                "document_text": "exact_answers_joined_by_two_newlines",
            },
            "runtime_limitations": {
                "dp_request_output_does_not_expose_serving_replica_rank": True,
                "per_row_dp_replica_assignment_unavailable": True,
                "forced_eager_required_for_vllm_dp2": True,
                "throughput_not_comparable_to_cuda_graph_mode": True,
            },
        }
    )
    _write_new_json(output_dir / "manifest.json", manifest)
    return manifest


def _generation_record_from_section(section: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in section.items() if key not in SECTION_ASSEMBLY_KEYS
    }


def _load_bound_assembly_sources(
    manifest: Mapping[str, Any],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
]:
    """Reload every bound source so validation proves a lossless reshape."""

    cohort_binding = manifest.get("cohort_manifest")
    suite_binding = manifest.get("suite_manifest")
    generation_bindings = manifest.get("source_generation_files")
    if (
        not isinstance(cohort_binding, Mapping)
        or not isinstance(suite_binding, Mapping)
        or not isinstance(generation_bindings, Mapping)
    ):
        raise VllmK5MeetingAssemblyError("assembly source bindings are missing")
    cohort_path = _validate_binding(cohort_binding)
    cohort, samples, ledger_rows = _load_cohort(
        cohort_path, str(cohort_binding.get("sha256") or "")
    )
    if cohort_binding.get("payload_sha256") != cohort["integrity"]["payload_sha256"]:
        raise VllmK5MeetingAssemblyError("cohort payload binding drift")
    suite_path = _validate_binding(suite_binding)
    suite = _read_json(suite_path)
    if suite_binding.get("payload_sha256") != suite["integrity"]["payload_sha256"]:
        raise VllmK5MeetingAssemblyError("suite payload binding drift")
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for model_id in preparation.MODEL_ORDER:
        binding = generation_bindings.get(model_id)
        if not isinstance(binding, Mapping):
            raise VllmK5MeetingAssemblyError(
                f"source generation binding is missing for {model_id}"
            )
        expected_path = suite_path.parent / model_id / "generations.canonical.v1.jsonl"
        generation_path = _validate_binding(
            binding,
            expected_path=expected_path,
            rows=preparation.EXPECTED_CASES_PER_MODEL,
        )
        rows = _read_canonical_jsonl(generation_path)
        if len(rows) != preparation.EXPECTED_CASES_PER_MODEL:
            raise VllmK5MeetingAssemblyError(
                f"physical source generation count drift for {model_id}"
            )
        rows_by_model[model_id] = rows
    return (
        rows_by_model,
        {str(row["sample_id"]): row for row in samples},
        {str(row["sample_id"]): row for row in ledger_rows},
    )


def validate_assembly(manifest_path: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    payload_sha = validate_manifest_integrity(manifest)
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("immutable") is not True
        or manifest.get("evaluation_id") != preparation.EVALUATION_ID
        or manifest.get("model_order") != list(preparation.MODEL_ORDER)
        or manifest.get("coverage", {}).get("documents") != EXPECTED_DOCUMENTS
        or manifest.get("coverage", {}).get("sections")
        != preparation.EXPECTED_TOTAL_CASES
        or manifest.get("coverage", {}).get("input_truncation_sections") != 0
        or manifest.get("assembly", {}).get("full_generation_records_preserved")
        is not True
        or manifest.get("backend_contract", {}).get("mixed_with_nf4_k10_rows")
        is not False
        or manifest.get("backend_contract", {}).get("data_parallel_size") != 2
        or manifest.get("backend_contract", {}).get("tensor_parallel_size") != 1
        or manifest.get("backend_contract", {}).get("pipeline_parallel_size") != 1
        or manifest.get("backend_contract", {}).get("physical_gpu_indexes") != [0, 1]
        or manifest.get("backend_contract", {}).get("backend")
        != "vllm-async-engine-v1-continuous-batching"
        or manifest.get("backend_contract", {}).get("weight_precision") != "bfloat16"
        or manifest.get("backend_contract", {}).get("quantization") is not None
        or manifest.get("backend_contract", {}).get(
            "gpu_memory_utilization_per_replica"
        )
        != 0.95
        or manifest.get("backend_contract", {}).get("max_num_seqs") not in {8, 12, 16}
        or manifest.get("backend_contract", {}).get("enforce_eager") is not True
        or manifest.get("backend_contract", {}).get("full_bf16_replica_per_gpu")
        is not True
        or manifest.get("runtime_limitations", {}).get(
            "per_row_dp_replica_assignment_unavailable"
        )
        is not True
    ):
        raise VllmK5MeetingAssemblyError("assembly manifest contract drift")
    binding = manifest.get("meeting_documents")
    if not isinstance(binding, Mapping):
        raise VllmK5MeetingAssemblyError("meeting-document binding is missing")
    document_path = _validate_binding(
        binding,
        expected_path=manifest_path.expanduser().resolve().parent
        / "meeting_documents.v1.jsonl",
        rows=EXPECTED_DOCUMENTS,
    )
    documents = _read_canonical_jsonl(document_path)
    if len(documents) != EXPECTED_DOCUMENTS:
        raise VllmK5MeetingAssemblyError("physical meeting-document count drift")
    source_rows, sample_by_id, ledger_by_id = _load_bound_assembly_sources(manifest)
    seen_documents: set[str] = set()
    seen_tuples: set[tuple[str, str, int, str]] = set()
    seen_source_lines: dict[str, set[int]] = defaultdict(set)
    observed_model_order: list[str] = []
    meeting_replicate_order: dict[str, list[tuple[str, int]]] = defaultdict(list)
    last_model: str | None = None
    total_sections = 0
    exposed_documents = 0
    for document in documents:
        model_id = str(document.get("model_id") or "")
        if model_id != last_model:
            observed_model_order.append(model_id)
            last_model = model_id
        replicate_id = document.get("replicate_id")
        if (
            model_id not in preparation.MODEL_ORDER
            or isinstance(replicate_id, bool)
            or not isinstance(replicate_id, int)
            or not 0 <= replicate_id < len(preparation.REPLICATE_SEEDS)
        ):
            raise VllmK5MeetingAssemblyError("invalid document model/replicate")
        meeting_id = str(document.get("meeting_id") or "")
        expected_document_id = f"{model_id}::{meeting_id}::replicate-{replicate_id:02d}"
        if (
            document.get("document_id") != expected_document_id
            or expected_document_id in seen_documents
        ):
            raise VllmK5MeetingAssemblyError(
                "document identity is duplicate or noncanonical"
            )
        seen_documents.add(expected_document_id)
        if document.get("replicate_seed") != preparation.REPLICATE_SEEDS[replicate_id]:
            raise VllmK5MeetingAssemblyError("document replicate seed drift")
        meeting_replicate_order[model_id].append((meeting_id, replicate_id))
        sections = document.get("sections")
        if (
            not isinstance(sections, list)
            or len(sections) != len(data_contract.CORE_TOPICS)
            or document.get("section_count") != len(data_contract.CORE_TOPICS)
            or tuple(section.get("topic") for section in sections)
            != tuple(data_contract.CORE_TOPICS)
        ):
            raise VllmK5MeetingAssemblyError("document section/Core8 closure drift")
        answers: list[str] = []
        for section in sections:
            if not isinstance(section, Mapping):
                raise VllmK5MeetingAssemblyError("document section is not an object")
            generation = _generation_record_from_section(section)
            if section.get("generation_record_sha256") != _record_sha256(generation):
                raise VllmK5MeetingAssemblyError("section generation-record SHA drift")
            if (
                generation.get("model_id") != model_id
                or generation.get("meeting_id") != meeting_id
                or generation.get("replicate_id") != replicate_id
                or generation.get("replicate_seed")
                != preparation.REPLICATE_SEEDS[replicate_id]
                or generation.get("input_truncated") is not False
            ):
                raise VllmK5MeetingAssemblyError("section identity/contract drift")
            source_line = section.get("source_generation_line_number")
            if (
                isinstance(source_line, bool)
                or not isinstance(source_line, int)
                or not 1 <= source_line <= preparation.EXPECTED_CASES_PER_MODEL
                or source_line in seen_source_lines[model_id]
                or generation != source_rows[model_id][source_line - 1]
            ):
                raise VllmK5MeetingAssemblyError(
                    "section/source-generation line binding drift"
                )
            seen_source_lines[model_id].add(source_line)
            sample_id = str(generation.get("sample_id") or "")
            sample = sample_by_id.get(sample_id)
            ledger = ledger_by_id.get(sample_id)
            if (
                sample is None
                or ledger is None
                or section.get("source_sample_record_sha256") != _record_sha256(sample)
                or section.get("prompt_token_ledger_row_sha256")
                != _record_sha256(ledger)
                or section.get("input_prompt_token_ids")
                != ledger.get("prompt_token_ids")
            ):
                raise VllmK5MeetingAssemblyError(
                    "section cohort/token-ledger binding drift"
                )
            _check_text_hash(generation, "answer", "answer_sha256")
            _check_text_hash(
                generation,
                "generated_text",
                _generation_text_hash_key(generation),
            )
            token_ids = generation.get("generated_token_ids")
            if not isinstance(token_ids, list) or generation.get(
                "generated_token_ids_sha256"
            ) != _token_ids_sha256(token_ids):
                raise VllmK5MeetingAssemblyError("section generated-token SHA drift")
            prompt_token_ids = section.get("input_prompt_token_ids")
            if (
                not isinstance(prompt_token_ids, list)
                or section.get("input_prompt_token_ids_sha256")
                != _token_ids_sha256(prompt_token_ids)
                or generation.get("prompt_token_ids_sha256")
                != section.get("input_prompt_token_ids_sha256")
            ):
                raise VllmK5MeetingAssemblyError("section prompt-token SHA drift")
            tuple_key = (
                model_id,
                str(generation.get("sample_id") or ""),
                replicate_id,
                str(generation.get("topic") or ""),
            )
            if tuple_key in seen_tuples:
                raise VllmK5MeetingAssemblyError(
                    "duplicate generation tuple in documents"
                )
            seen_tuples.add(tuple_key)
            answers.append(str(generation["answer"]))
            total_sections += 1
        expected_text = "\n\n".join(answers)
        if document.get("document_text") != expected_text or document.get(
            "document_text_sha256"
        ) != sha256_text(expected_text):
            raise VllmK5MeetingAssemblyError("meeting document text binding drift")
        exposed_documents += bool(document.get("cp318_selection_exposed"))
    if observed_model_order != list(preparation.MODEL_ORDER):
        raise VllmK5MeetingAssemblyError("physical document model order drift")
    canonical_meeting_order: list[str] | None = None
    for model_id in preparation.MODEL_ORDER:
        observed_pairs = meeting_replicate_order[model_id]
        if len(observed_pairs) != DOCUMENTS_PER_MODEL:
            raise VllmK5MeetingAssemblyError("documents-per-model closure drift")
        observed_meetings: list[str] = []
        for offset in range(0, len(observed_pairs), len(preparation.REPLICATE_SEEDS)):
            block = observed_pairs[offset : offset + len(preparation.REPLICATE_SEEDS)]
            if [replicate_id for _, replicate_id in block] != list(
                range(len(preparation.REPLICATE_SEEDS))
            ) or len({meeting_id for meeting_id, _ in block}) != 1:
                raise VllmK5MeetingAssemblyError(
                    "physical meeting/replicate order drift"
                )
            observed_meetings.append(block[0][0])
        if len(set(observed_meetings)) != data_contract.EXPECTED_MEETINGS:
            raise VllmK5MeetingAssemblyError("physical meeting coverage drift")
        if canonical_meeting_order is None:
            canonical_meeting_order = observed_meetings
        elif observed_meetings != canonical_meeting_order:
            raise VllmK5MeetingAssemblyError("cross-model meeting order drift")
    if (
        total_sections != preparation.EXPECTED_TOTAL_CASES
        or len(seen_tuples) != total_sections
    ):
        raise VllmK5MeetingAssemblyError("document section tuple closure drift")
    expected_source_lines = set(range(1, preparation.EXPECTED_CASES_PER_MODEL + 1))
    if any(
        seen_source_lines[model_id] != expected_source_lines
        for model_id in preparation.MODEL_ORDER
    ):
        raise VllmK5MeetingAssemblyError("source generation line coverage drift")
    expected_exposed = (
        EXPECTED_SELECTION_EXPOSED_MEETINGS
        * len(preparation.REPLICATE_SEEDS)
        * len(preparation.MODEL_ORDER)
    )
    if exposed_documents != expected_exposed:
        raise VllmK5MeetingAssemblyError("selection-exposed document count drift")
    return {
        "status": "valid",
        "manifest_sha256": sha256_file(manifest_path),
        "payload_sha256": payload_sha,
        "documents": len(documents),
        "sections": total_sections,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("assemble")
    build.add_argument("--cohort-manifest", required=True, type=Path)
    build.add_argument("--cohort-manifest-sha256", required=True)
    build.add_argument("--suite-manifest", required=True, type=Path)
    build.add_argument("--output-dir", required=True, type=Path)
    build.add_argument("--max-num-seqs", required=True, type=int, choices=(8, 12, 16))
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True, type=Path)
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
                "payload_sha256": result["integrity"]["payload_sha256"],
            }
        else:
            summary = validate_assembly(args.manifest)
        print(_canonical(summary))
        return 0
    except (VllmK5MeetingAssemblyError, ExternalSmokeError, OSError, ValueError) as exc:
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
    "VllmK5MeetingAssemblyError",
    "assemble",
    "assemble_documents",
    "validate_assembly",
]
