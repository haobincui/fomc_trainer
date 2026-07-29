"""Validate and freeze a canonical, generation-only LOO release.

The finalizer is intentionally read-only with respect to models, tokenizers,
prompts, and generated outputs.  Its only write is the requested immutable
release manifest.  It performs no scoring, training, checkpoint merging, or
model loading.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from open_r1.provenance import (
    sha256_file,
    sha256_text,
    validate_sha256,
)
from open_r1.validator.loo_generation_spec import (
    derive_row_seed,
    load_and_validate_generation_spec,
    seal_manifest,
    validate_generation_completion,
)


RELEASE_SCHEMA_VERSION = "canonical-loo-generation-release-v1"
ANALYSIS_SCHEMA_VERSION = "indicator-analysis-generation-v1"
PROMPT_SCHEMA_VERSION = "loo-prompt-manifest-v1"
GENERATION_SCHEMA_VERSION = "loo-generation-v4"
SAMPLE_SEED_POLICY = "sample-id-sha256-v1"
DELETION_STRATEGY = "indicator_block_deletion"
NEUTRAL_STRATEGY = "indicator_block_neutral_replacement"
BASELINE_INDICATOR_DEFAULT = "None"


@dataclass(frozen=True)
class RunExpectation:
    name: str
    strategy: str
    prompt_folder: str
    replicate_count: int
    temperature: float
    top_p: float


EXPECTED_RUNS = (
    RunExpectation(
        "deletion_primary",
        DELETION_STRATEGY,
        "exact_delete",
        1,
        0.0,
        1.0,
    ),
    RunExpectation(
        "neutral_primary",
        NEUTRAL_STRATEGY,
        "neutral",
        1,
        0.0,
        1.0,
    ),
    RunExpectation(
        "deletion_stochastic",
        DELETION_STRATEGY,
        "exact_delete",
        5,
        0.6,
        0.9,
    ),
    RunExpectation(
        "neutral_stochastic",
        NEUTRAL_STRATEGY,
        "neutral",
        5,
        0.6,
        0.9,
    ),
)
EXPECTED_REPLICATE_SEEDS = {
    "deletion_primary": [20260728],
    "neutral_primary": [20260728],
    "deletion_stochastic": [
        20260729,
        21260729,
        22260729,
        23260729,
        24260729,
    ],
    "neutral_stochastic": [
        20260729,
        21260729,
        22260729,
        23260729,
        24260729,
    ],
}


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {label} {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"{label} row {line_number} must be a JSON object"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{label} contains no rows: {path}")
    return rows


def _resolve_within(
    root: Path,
    path: str | Path,
    *,
    label: str,
    base: Path | None = None,
) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (base or root) / candidate
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} escapes the run root: {resolved}")
    return resolved


def _relative_to_run_root(path: Path, run_root: Path) -> str:
    return path.resolve().relative_to(run_root).as_posix()


def _require_digest(value: Any, *, label: str) -> str:
    return validate_sha256(value, label=label)


def _assert_file_hash(path: Path, expected: Any, *, label: str) -> str:
    expected_digest = _require_digest(expected, label=f"{label}.sha256")
    observed = sha256_file(path)
    if observed != expected_digest:
        raise ValueError(
            f"{label} hash mismatch: expected={expected_digest}, observed={observed}"
        )
    return observed


def _find_source_hash(
    spec: Mapping[str, Any],
    digest: str,
    *,
    label: str,
) -> None:
    sources = spec.get("frozen_artifacts", {}).get("sources", {})
    if not isinstance(sources, Mapping):
        raise ValueError("Generation spec has no frozen source inventory")
    if digest not in {
        str(record.get("sha256"))
        for record in sources.values()
        if isinstance(record, Mapping)
    }:
        raise ValueError(
            f"Generation spec does not bind the active {label} hash {digest}"
        )


def _validate_analysis(
    *,
    manifest_path: Path,
    run_root: Path,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = _read_json(manifest_path, label="analysis manifest")
    if manifest.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported analysis manifest schema: "
            f"{manifest.get('schema_version')!r}"
        )
    if manifest.get("status") != "complete":
        raise ValueError("Analysis manifest status must be 'complete'")
    if (
        manifest.get("generation_only") is not True
        or manifest.get("training_performed") is not False
    ):
        raise ValueError(
            "Analysis manifest must declare generation_only=true and "
            "training_performed=false"
        )

    output = manifest.get("output")
    inventory = manifest.get("inventory")
    inputs = manifest.get("inputs")
    if (
        not isinstance(output, Mapping)
        or not isinstance(inventory, Mapping)
        or not isinstance(inputs, Mapping)
    ):
        raise ValueError(
            "Analysis manifest lacks input, output, or inventory metadata"
        )
    analysis_model = inputs.get("model")
    analysis_tokenizer = inputs.get("tokenizer")
    if not isinstance(analysis_model, Mapping) or not isinstance(
        analysis_tokenizer,
        Mapping,
    ):
        raise ValueError(
            "Analysis manifest lacks model/tokenizer fingerprints"
        )
    frozen_artifacts = spec.get("frozen_artifacts", {})
    frozen_model_hashes = {
        str(entry.get("sha256"))
        for entry in frozen_artifacts.get("models", {}).values()
        if isinstance(entry, Mapping)
    }
    frozen_tokenizer_hashes = {
        str(entry.get("sha256"))
        for entry in frozen_artifacts.get("tokenizers", {}).values()
        if isinstance(entry, Mapping)
    }
    if analysis_model.get("sha256") not in frozen_model_hashes:
        raise ValueError(
            "Analysis model fingerprint is not bound by the generation spec"
        )
    if analysis_tokenizer.get("sha256") not in frozen_tokenizer_hashes:
        raise ValueError(
            "Analysis tokenizer fingerprint is not bound by the generation spec"
        )
    cutoff_config = spec.get("generation_config", {}).get(
        "information_cutoff",
        {},
    )
    require_d1_provenance = (
        isinstance(cutoff_config, Mapping)
        and cutoff_config.get("policy") == "previous-calendar-day-v1"
    )
    ledger_provenance = manifest.get("ledger_provenance")
    if require_d1_provenance:
        required_input_names = (
            "ledger_manifest",
            "snapshot_manifest",
            "source_registry",
            "roster",
            "population",
        )
        missing_provenance = [
            name
            for name in required_input_names
            if not isinstance(inputs.get(name), Mapping)
        ]
        if missing_provenance:
            raise ValueError(
                "Analysis manifest lacks required D-1 provenance inputs: "
                f"{missing_provenance}"
            )
        if not isinstance(ledger_provenance, Mapping):
            raise ValueError(
                "Analysis manifest lacks its validated ledger_provenance binding"
            )
        for name in required_input_names:
            digest = _require_digest(
                inputs[name].get("sha256"),
                label=f"analysis inputs.{name}.sha256",
            )
            _find_source_hash(spec, digest, label=f"analysis {name}")
        expected_bindings = {
            "ledger_manifest_sha256": inputs["ledger_manifest"]["sha256"],
            "snapshot_manifest_sha256": inputs["snapshot_manifest"]["sha256"],
            "source_registry_sha256": inputs["source_registry"]["sha256"],
            "roster_sha256": inputs["roster"]["sha256"],
            "population_sha256": inputs["population"]["sha256"],
        }
        mismatches = {
            field: {
                "expected": digest,
                "observed": ledger_provenance.get(field),
            }
            for field, digest in expected_bindings.items()
            if ledger_provenance.get(field) != digest
        }
        if mismatches:
            raise ValueError(
                f"Analysis ledger provenance differs from its inputs: {mismatches}"
            )
    output_path = _resolve_within(
        run_root,
        str(output.get("path") or ""),
        label="analysis output",
        base=manifest_path.parent,
    )
    output_sha256 = _assert_file_hash(
        output_path,
        output.get("sha256"),
        label="analysis output",
    )
    rows = _read_jsonl(output_path, label="analysis output")
    declared_count = inventory.get("row_count")
    if (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count != len(rows)
    ):
        raise ValueError(
            "Analysis output row count does not match "
            f"inventory.row_count: declared={declared_count!r}, "
            f"observed={len(rows)}"
        )
    inventory_rows = inventory.get("rows")
    if isinstance(inventory_rows, list) and len(inventory_rows) != len(rows):
        raise ValueError(
            "Analysis inventory row list does not match analysis output count"
        )
    manifest_sha256 = sha256_file(manifest_path)
    _find_source_hash(spec, manifest_sha256, label="analysis manifest")
    _find_source_hash(spec, output_sha256, label="analysis output")
    return {
        "manifest_relative_path": _relative_to_run_root(
            manifest_path,
            run_root,
        ),
        "manifest_sha256": manifest_sha256,
        "output_relative_path": _relative_to_run_root(output_path, run_root),
        "output_sha256": output_sha256,
        "row_count": len(rows),
        "run_id": manifest.get("run_id"),
        "population_id": inventory.get("population_id"),
        "model_sha256": analysis_model.get("sha256"),
        "tokenizer_sha256": analysis_tokenizer.get("sha256"),
        "ledger_provenance": ledger_provenance,
    }


def _validate_prompt_rows(
    path: Path,
    *,
    declared_count: Any,
    label: str,
) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path, label=label)
    if (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count != len(rows)
    ):
        raise ValueError(
            f"{label} row count mismatch: declared={declared_count!r}, "
            f"observed={len(rows)}"
        )
    by_sample: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, 1):
        sample_id = str(row.get("sample_id") or "").strip()
        prompt = row.get("prompt")
        if not sample_id or not isinstance(prompt, str) or not prompt:
            raise ValueError(
                f"{label} row {row_number} lacks sample_id or prompt"
            )
        if sample_id in by_sample:
            raise ValueError(f"{label} contains duplicate sample_id {sample_id!r}")
        prompt_digest = sha256_text(prompt)
        if row.get("prompt_sha256") not in {None, prompt_digest}:
            raise ValueError(
                f"{label} has an invalid prompt_sha256 for {sample_id!r}"
            )
        by_sample[sample_id] = row
    return by_sample


def _validate_prompt_reference(
    *,
    manifest_path: Path,
    run_root: Path,
    reference: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    relative_path = str(reference.get("relative_path") or "").strip()
    if not relative_path:
        raise ValueError(f"Prompt manifest {label} has no relative_path")
    path = _resolve_within(
        run_root,
        relative_path,
        label=f"prompt {label}",
        base=manifest_path.parent,
    )
    digest = _assert_file_hash(
        path,
        reference.get("sha256"),
        label=f"prompt {label}",
    )
    return {
        "relative_path": _relative_to_run_root(path, run_root),
        "sha256": digest,
    }


def _validate_prompts(
    *,
    manifest_path: Path,
    run_root: Path,
    spec: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    manifest = _read_json(manifest_path, label="prompt manifest")
    if manifest.get("schema_version") != PROMPT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported prompt manifest schema: "
            f"{manifest.get('schema_version')!r}"
        )
    manifest_sha256 = sha256_file(manifest_path)
    _find_source_hash(spec, manifest_sha256, label="prompt manifest")

    baseline = str(
        manifest.get("baseline_indicator") or BASELINE_INDICATOR_DEFAULT
    ).strip()
    indicators = manifest.get("indicators")
    if (
        not baseline
        or not isinstance(indicators, list)
        or not indicators
        or any(not isinstance(value, str) or not value.strip() for value in indicators)
        or len(indicators) != len(set(indicators))
        or baseline in indicators
    ):
        raise ValueError("Prompt manifest has an invalid indicator roster")
    indicator_roster = [baseline, *indicators]
    population_context = str(manifest.get("population_context") or "").strip()
    if not population_context:
        raise ValueError("Prompt manifest lacks population_context")
    population_dates = manifest.get("population_dates")
    section_families = manifest.get("section_families")
    counts = manifest.get("counts")
    if (
        not isinstance(population_dates, list)
        or not population_dates
        or any(not isinstance(value, str) or not value for value in population_dates)
        or population_dates != sorted(set(population_dates))
    ):
        raise ValueError(
            "Prompt manifest requires unique ordered population_dates"
        )
    if (
        not isinstance(section_families, list)
        or not section_families
        or any(not isinstance(value, str) or not value for value in section_families)
        or len(section_families) != len(set(section_families))
    ):
        raise ValueError(
            "Prompt manifest requires unique non-empty section_families"
        )
    if not isinstance(counts, Mapping):
        raise ValueError("Prompt manifest lacks counts")

    prompt_sources = manifest.get("source_artifacts")
    if not isinstance(prompt_sources, list):
        raise ValueError("Prompt manifest lacks source_artifacts")
    analysis_source_matches = [
        entry
        for entry in prompt_sources
        if isinstance(entry, Mapping) and entry.get("role") == "analysis_blocks"
    ]
    if len(analysis_source_matches) != 1:
        raise ValueError(
            "Prompt manifest must contain one analysis_blocks source artifact"
        )
    analysis_source = analysis_source_matches[0]
    if analysis_source.get("sha256") != analysis["output_sha256"]:
        raise ValueError(
            "Prompt manifest is not bound to the validated analysis output hash"
        )
    analysis_source_path = _resolve_within(
        run_root,
        str(analysis_source.get("path") or ""),
        label="prompt analysis source",
        base=manifest_path.parent,
    )
    if (
        _relative_to_run_root(analysis_source_path, run_root)
        != analysis["output_relative_path"]
    ):
        raise ValueError(
            "Prompt manifest analysis_blocks path differs from the validated "
            "analysis output"
        )

    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise ValueError("Prompt manifest lacks artifact inventory")
    inventory: dict[str, dict[str, dict[str, Any]]] = {
        "exact_delete": {},
        "neutral": {},
    }
    expected_arms = {
        "exact_delete": {
            baseline: "full",
            **{indicator: "delete" for indicator in indicators},
        },
        "neutral": {
            baseline: "full_export",
            **{indicator: "neutral" for indicator in indicators},
        },
    }
    for entry in raw_artifacts:
        if not isinstance(entry, Mapping):
            raise ValueError("Prompt manifest artifact entries must be objects")
        relative_path = str(entry.get("relative_path") or "").strip()
        parts = Path(relative_path).parts
        if not parts or parts[0] not in inventory:
            continue
        folder = parts[0]
        indicator = str(entry.get("indicator") or "").strip()
        if indicator not in expected_arms[folder]:
            raise ValueError(
                f"Unexpected prompt indicator {indicator!r} in {relative_path}"
            )
        if entry.get("arm") != expected_arms[folder][indicator]:
            raise ValueError(
                f"Unexpected prompt arm for {relative_path}: {entry.get('arm')!r}"
            )
        if indicator in inventory[folder]:
            raise ValueError(
                f"Duplicate prompt artifact for {folder}/{indicator}"
            )
        path = _resolve_within(
            run_root,
            relative_path,
            label="prompt artifact",
            base=manifest_path.parent,
        )
        digest = _assert_file_hash(
            path,
            entry.get("sha256"),
            label=f"prompt artifact {relative_path}",
        )
        rows = _validate_prompt_rows(
            path,
            declared_count=entry.get("row_count"),
            label=f"prompt artifact {relative_path}",
        )
        inventory[folder][indicator] = {
            "relative_path": relative_path,
            "path": path,
            "sha256": digest,
            "row_count": len(rows),
            "rows": rows,
            "context": str(entry.get("context") or population_context),
        }

    for folder in inventory:
        if set(inventory[folder]) != set(indicator_roster):
            raise ValueError(
                f"Prompt {folder} inventory mismatch: "
                f"missing={sorted(set(indicator_roster) - set(inventory[folder]))}, "
                f"extra={sorted(set(inventory[folder]) - set(indicator_roster))}"
            )
        folder_path = _resolve_within(
            run_root,
            folder,
            label=f"prompt folder {folder}",
            base=manifest_path.parent,
        )
        observed = {
            path.resolve()
            for path in folder_path.glob("*.jsonl")
            if path.is_file()
        }
        declared = {
            entry["path"].resolve() for entry in inventory[folder].values()
        }
        if observed != declared:
            raise ValueError(
                f"Prompt folder {folder} contains undeclared or missing JSONL files"
            )

    references: dict[str, Any] = {}
    for key in (
        "run_intervention_roster",
        "prompt_ledger",
        "intervention_manifest",
        "neutral_intervention_manifest",
    ):
        reference = manifest.get(key)
        if reference is not None:
            if not isinstance(reference, Mapping):
                raise ValueError(f"Prompt manifest {key} must be an object")
            references[key] = _validate_prompt_reference(
                manifest_path=manifest_path,
                run_root=run_root,
                reference=reference,
                label=key,
            )

    summary = {
        "manifest_relative_path": _relative_to_run_root(
            manifest_path,
            run_root,
        ),
        "manifest_sha256": manifest_sha256,
        "population_id": manifest.get("population_id"),
        "population_context": population_context,
        "population_dates": population_dates,
        "section_families": section_families,
        "baseline_indicator": baseline,
        "indicator_count": len(indicators),
        "counts": dict(counts),
        "artifact_count": sum(len(values) for values in inventory.values()),
        "references": references,
    }
    return summary, inventory


def _validate_generation_row(
    row: Mapping[str, Any],
    *,
    artifact_label: str,
    source_row: Mapping[str, Any],
    indicator: str,
    replicate_id: str,
    replicate_seed: int,
    strategy: str,
    context: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    sample_id = str(row.get("sample_id") or "").strip()
    if not sample_id:
        raise ValueError(f"{artifact_label} contains a row without sample_id")
    source_prompt = str(source_row.get("prompt") or "")
    prompt = row.get("prompt")
    if prompt != source_prompt:
        raise ValueError(
            f"{artifact_label} prompt differs from its source for {sample_id!r}"
        )
    prompt_sha256 = sha256_text(source_prompt)
    if row.get("source_prompt_sha256") != prompt_sha256:
        raise ValueError(
            f"{artifact_label} has an invalid source_prompt_sha256 for "
            f"{sample_id!r}"
        )
    generated = row.get("generated")
    if not isinstance(generated, str) or not generated.strip():
        raise ValueError(
            f"{artifact_label} has empty generated text for {sample_id!r}"
        )
    generated_sha256 = sha256_text(generated)
    if row.get("generated_sha256") != generated_sha256:
        raise ValueError(
            f"{artifact_label} has an invalid generated_sha256 for {sample_id!r}"
        )
    expected_seed = derive_row_seed(replicate_seed, sample_id)
    expected_metadata = {
        "indicator": indicator,
        "replicate_id": replicate_id,
        "generation_seed": expected_seed,
        "generation_seed_policy": SAMPLE_SEED_POLICY,
        "masking_strategy": strategy,
        "evaluation_context": context,
        "decoding_temperature": float(manifest["temperature"]),
        "decoding_top_p": float(manifest["top_p"]),
        "max_new_tokens": int(manifest["max_new_tokens"]),
        "max_model_len": int(manifest["max_model_len"]),
        "generation_model_sha256": manifest["model_artifact"]["sha256"],
        "generation_tokenizer_sha256": manifest["tokenizer_artifact"]["sha256"],
        "generation_system_prompt_sha256": manifest["system_prompt_sha256"],
    }
    mismatches = {
        key: {"expected": value, "observed": row.get(key)}
        for key, value in expected_metadata.items()
        if row.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"{artifact_label} has incompatible row metadata for "
            f"{sample_id!r}: {mismatches}"
        )
    try:
        completion = validate_generation_completion(
            input_token_count=int(row["prompt_preflight_token_count"]),
            output_token_count=int(row["output_token_count"]),
            max_new_tokens=int(manifest["max_new_tokens"]),
            context_limit=int(manifest["max_model_len"]),
            finish_reason=str(row.get("generation_finish_reason") or ""),
            input_was_truncated=row.get("input_was_truncated"),
            consumed_input_token_count=int(row["prompt_token_count"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{artifact_label} has invalid completion metadata for "
            f"{sample_id!r}: {exc}"
        ) from exc
    return {
        "sample_id": sample_id,
        "prompt": source_prompt,
        "prompt_sha256": prompt_sha256,
        "generation_seed": expected_seed,
        "generated": generated,
        "generated_sha256": generated_sha256,
        "finish_reason": completion["finish_reason"],
    }


def _validate_generation_run(
    *,
    run_root: Path,
    expectation: RunExpectation,
    spec: Mapping[str, Any],
    spec_path: Path,
    spec_file_sha256: str,
    prompt_summary: Mapping[str, Any],
    prompt_inventory: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    directory = (run_root / "generations" / expectation.name).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Expected generation directory does not exist: {directory}"
        )
    manifest_path = directory / "generation_manifest.json"
    manifest = _read_json(
        manifest_path,
        label=f"{expectation.name} generation manifest",
    )
    if manifest.get("schema_version") != GENERATION_SCHEMA_VERSION:
        raise ValueError(
            f"{expectation.name} has unsupported generation schema "
            f"{manifest.get('schema_version')!r}"
        )
    expected_fields = {
        "masking_strategy": expectation.strategy,
        "simulation_step": expectation.replicate_count,
        "temperature": expectation.temperature,
        "top_p": expectation.top_p,
        "seed_policy": SAMPLE_SEED_POLICY,
        "require_normal_finish": True,
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected_fields.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"{expectation.name} generation configuration mismatch: {mismatches}"
        )
    if Path(str(manifest.get("output_dir") or "")).expanduser().resolve() != directory:
        raise ValueError(f"{expectation.name} output_dir is not its run directory")

    prompt_folder_path = (
        Path(prompt_inventory[expectation.prompt_folder][
            str(prompt_summary["baseline_indicator"])
        ]["path"]).parent.resolve()
    )
    if (
        Path(str(manifest.get("input_folder") or "")).expanduser().resolve()
        != prompt_folder_path
    ):
        raise ValueError(
            f"{expectation.name} input_folder differs from the prompt inventory"
        )

    prompt_binding = manifest.get("prompt_manifest")
    spec_binding = manifest.get("generation_spec")
    if not isinstance(prompt_binding, Mapping) or not isinstance(
        spec_binding,
        Mapping,
    ):
        raise ValueError(
            f"{expectation.name} lacks prompt/spec manifest bindings"
        )
    if (
        prompt_binding.get("sha256") != prompt_summary["manifest_sha256"]
        or prompt_binding.get("population_id") != prompt_summary["population_id"]
    ):
        raise ValueError(
            f"{expectation.name} is not bound to the active prompt manifest"
        )
    if (
        Path(str(spec_binding.get("path") or "")).expanduser().resolve()
        != spec_path
        or spec_binding.get("file_sha256") != spec_file_sha256
        or spec_binding.get("payload_sha256")
        != spec["integrity"]["payload_sha256"]
        or spec_binding.get("run_id") != spec["run_id"]
    ):
        raise ValueError(
            f"{expectation.name} is not bound to the active generation spec"
        )

    for kind in ("model_artifact", "tokenizer_artifact"):
        before = manifest.get(kind)
        after = manifest.get(f"{kind}_after_generation")
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise ValueError(f"{expectation.name} lacks {kind} fingerprints")
        if (
            before.get("sha256") != after.get("sha256")
            or before.get("kind") != after.get("kind")
            or before.get("file_count") != after.get("file_count")
            or before.get("total_bytes") != after.get("total_bytes")
        ):
            raise ValueError(
                f"{expectation.name} {kind} changed during generation"
            )
    frozen_artifacts = spec.get("frozen_artifacts")
    if not isinstance(frozen_artifacts, Mapping):
        raise ValueError("Generation spec lacks frozen_artifacts")
    frozen_model_hashes = {
        str(entry.get("sha256"))
        for entry in frozen_artifacts.get("models", {}).values()
        if isinstance(entry, Mapping)
    }
    frozen_tokenizer_hashes = {
        str(entry.get("sha256"))
        for entry in frozen_artifacts.get("tokenizers", {}).values()
        if isinstance(entry, Mapping)
    }
    if manifest["model_artifact"].get("sha256") not in frozen_model_hashes:
        raise ValueError(
            f"{expectation.name} model is not bound by the generation spec"
        )
    if (
        manifest["tokenizer_artifact"].get("sha256")
        not in frozen_tokenizer_hashes
    ):
        raise ValueError(
            f"{expectation.name} tokenizer is not bound by the generation spec"
        )

    replicate_seeds = manifest.get("replicate_seeds")
    if (
        not isinstance(replicate_seeds, list)
        or len(replicate_seeds) != expectation.replicate_count
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in replicate_seeds
        )
        or len(replicate_seeds) != len(set(replicate_seeds))
    ):
        raise ValueError(
            f"{expectation.name} has an invalid replicate seed inventory"
        )
    if replicate_seeds != EXPECTED_REPLICATE_SEEDS[expectation.name]:
        raise ValueError(
            f"{expectation.name} replicate seeds differ from the frozen "
            f"design: expected={EXPECTED_REPLICATE_SEEDS[expectation.name]}, "
            f"observed={replicate_seeds}"
        )
    spec_replicate_seeds = spec.get("seed_policy", {}).get(
        "replicate_seeds"
    )
    if (
        not isinstance(spec_replicate_seeds, list)
        or not set(replicate_seeds).issubset(spec_replicate_seeds)
    ):
        raise ValueError(
            f"{expectation.name} replicate seeds are not bound by the "
            "generation spec"
        )
    expected_replicates = {
        str(index): seed for index, seed in enumerate(replicate_seeds)
    }

    source_inventory = prompt_inventory[expectation.prompt_folder]
    input_artifacts = manifest.get("input_artifacts")
    if not isinstance(input_artifacts, list):
        raise ValueError(f"{expectation.name} lacks input_artifacts")
    inputs_by_indicator: dict[str, Mapping[str, Any]] = {}
    for entry in input_artifacts:
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"{expectation.name} input artifact entries must be objects"
            )
        indicator = str(entry.get("indicator") or "").strip()
        if indicator in inputs_by_indicator:
            raise ValueError(
                f"{expectation.name} has duplicate input artifact {indicator!r}"
            )
        inputs_by_indicator[indicator] = entry
    if set(inputs_by_indicator) != set(source_inventory):
        raise ValueError(
            f"{expectation.name} input artifact indicator inventory mismatch"
        )
    for indicator, source in source_inventory.items():
        entry = inputs_by_indicator[indicator]
        if (
            entry.get("source_file_sha256") != source["sha256"]
            or entry.get("source_row_count") != source["row_count"]
            or str(entry.get("context")) != str(source["context"])
        ):
            raise ValueError(
                f"{expectation.name} input artifact differs from prompt "
                f"inventory for {indicator!r}"
            )

    expected_artifacts = manifest.get("expected_artifacts")
    if not isinstance(expected_artifacts, list):
        raise ValueError(f"{expectation.name} lacks expected_artifacts")
    expected_keys = {
        (indicator, replicate_id)
        for indicator in source_inventory
        for replicate_id in expected_replicates
    }
    artifacts_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    output_paths: set[Path] = set()
    baseline_rows: dict[str, dict[str, dict[str, Any]]] = {}
    total_rows = 0
    for entry in expected_artifacts:
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"{expectation.name} expected artifact entries must be objects"
            )
        indicator = str(entry.get("indicator") or "").strip()
        replicate_id = str(entry.get("replicate_id") or "")
        key = (indicator, replicate_id)
        if key in artifacts_by_key:
            raise ValueError(
                f"{expectation.name} has duplicate expected artifact {key!r}"
            )
        artifacts_by_key[key] = entry
        if key not in expected_keys:
            continue
        source = source_inventory[indicator]
        replicate_seed = expected_replicates[replicate_id]
        relative_path = str(entry.get("relative_path") or "").strip()
        output_path = _resolve_within(
            run_root,
            relative_path,
            label=f"{expectation.name} output artifact",
            base=directory,
        )
        declared_absolute = Path(
            str(entry.get("path") or "")
        ).expanduser().resolve()
        if declared_absolute != output_path:
            raise ValueError(
                f"{expectation.name} output path binding mismatch for {key!r}"
            )
        if output_path in output_paths:
            raise ValueError(
                f"{expectation.name} output path is declared more than once: "
                f"{output_path}"
            )
        output_paths.add(output_path)
        _assert_file_hash(
            output_path,
            entry.get("output_sha256"),
            label=f"{expectation.name} output {relative_path}",
        )
        if (
            entry.get("source_file_sha256") != source["sha256"]
            or entry.get("source_row_count") != source["row_count"]
            or entry.get("replicate_seed") != replicate_seed
            or str(entry.get("context")) != str(source["context"])
        ):
            raise ValueError(
                f"{expectation.name} expected artifact metadata mismatch for "
                f"{key!r}"
            )
        rows = _read_jsonl(
            output_path,
            label=f"{expectation.name} output {relative_path}",
        )
        if len(rows) != source["row_count"]:
            raise ValueError(
                f"{expectation.name} output row count mismatch for {key!r}: "
                f"expected={source['row_count']}, observed={len(rows)}"
            )
        rows_by_sample: dict[str, dict[str, Any]] = {}
        source_rows = source["rows"]
        for row in rows:
            sample_id = str(row.get("sample_id") or "").strip()
            if sample_id not in source_rows:
                raise ValueError(
                    f"{expectation.name} output {key!r} has unexpected "
                    f"sample_id {sample_id!r}"
                )
            if sample_id in rows_by_sample:
                raise ValueError(
                    f"{expectation.name} output {key!r} duplicates "
                    f"sample_id {sample_id!r}"
                )
            rows_by_sample[sample_id] = _validate_generation_row(
                row,
                artifact_label=f"{expectation.name} output {relative_path}",
                source_row=source_rows[sample_id],
                indicator=indicator,
                replicate_id=replicate_id,
                replicate_seed=replicate_seed,
                strategy=expectation.strategy,
                context=str(source["context"]),
                manifest=manifest,
            )
        if set(rows_by_sample) != set(source_rows):
            raise ValueError(
                f"{expectation.name} output {key!r} has incomplete sample coverage"
            )
        if indicator == prompt_summary["baseline_indicator"]:
            baseline_rows[replicate_id] = rows_by_sample
        total_rows += len(rows)

    if set(artifacts_by_key) != expected_keys:
        raise ValueError(
            f"{expectation.name} expected artifact inventory mismatch: "
            f"missing={sorted(expected_keys - set(artifacts_by_key))}, "
            f"extra={sorted(set(artifacts_by_key) - expected_keys)}"
        )
    observed_jsonl = {
        path.resolve() for path in directory.glob("*.jsonl") if path.is_file()
    }
    if observed_jsonl != output_paths:
        raise ValueError(
            f"{expectation.name} contains undeclared or missing output JSONL files"
        )
    if set(baseline_rows) != set(expected_replicates):
        raise ValueError(
            f"{expectation.name} lacks one full baseline per replicate"
        )

    intervention = manifest.get("intervention_manifest")
    if not isinstance(intervention, Mapping):
        raise ValueError(f"{expectation.name} lacks intervention_manifest")
    intervention_path = _resolve_within(
        run_root,
        str(intervention.get("relative_path") or ""),
        label=f"{expectation.name} intervention manifest",
        base=directory,
    )
    intervention_sha256 = _assert_file_hash(
        intervention_path,
        intervention.get("sha256"),
        label=f"{expectation.name} intervention manifest",
    )
    summary = {
        "directory": _relative_to_run_root(directory, run_root),
        "manifest_relative_path": _relative_to_run_root(
            manifest_path,
            run_root,
        ),
        "manifest_sha256": sha256_file(manifest_path),
        "strategy": expectation.strategy,
        "replicate_count": expectation.replicate_count,
        "replicate_seeds": replicate_seeds,
        "artifact_count": len(expected_artifacts),
        "row_count": total_rows,
        "model_sha256": manifest["model_artifact"]["sha256"],
        "tokenizer_sha256": manifest["tokenizer_artifact"]["sha256"],
        "system_prompt_sha256": manifest["system_prompt_sha256"],
        "max_new_tokens": manifest["max_new_tokens"],
        "max_model_len": manifest["max_model_len"],
        "temperature": manifest["temperature"],
        "top_p": manifest["top_p"],
        "intervention_manifest_sha256": intervention_sha256,
        "expected_artifacts": [
            {
                "relative_path": _relative_to_run_root(path, run_root),
                "sha256": sha256_file(path),
                "row_count": len(
                    _read_jsonl(path, label=f"{expectation.name} output")
                ),
            }
            for path in sorted(output_paths)
        ],
    }
    return summary, baseline_rows


def _compare_baselines(
    *,
    regime: str,
    deletion_summary: Mapping[str, Any],
    neutral_summary: Mapping[str, Any],
    deletion_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    neutral_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    comparable_run_fields = (
        "replicate_seeds",
        "model_sha256",
        "tokenizer_sha256",
        "system_prompt_sha256",
        "max_new_tokens",
        "max_model_len",
        "temperature",
        "top_p",
    )
    run_mismatches = {
        field: {
            "deletion": deletion_summary.get(field),
            "neutral": neutral_summary.get(field),
        }
        for field in comparable_run_fields
        if deletion_summary.get(field) != neutral_summary.get(field)
    }
    if run_mismatches:
        raise ValueError(
            f"{regime} deletion/neutral generation settings differ: "
            f"{run_mismatches}"
        )
    if set(deletion_rows) != set(neutral_rows):
        raise ValueError(
            f"{regime} deletion/neutral baseline replicate inventory differs"
        )

    compared = 0
    for replicate_id in sorted(deletion_rows, key=int):
        deletion_samples = deletion_rows[replicate_id]
        neutral_samples = neutral_rows[replicate_id]
        if set(deletion_samples) != set(neutral_samples):
            raise ValueError(
                f"{regime} baseline sample inventory differs for replicate "
                f"{replicate_id}"
            )
        for sample_id in sorted(deletion_samples):
            deletion = deletion_samples[sample_id]
            neutral = neutral_samples[sample_id]
            comparable_row_fields = (
                "prompt",
                "prompt_sha256",
                "generation_seed",
                "generated",
                "generated_sha256",
                "finish_reason",
            )
            mismatches = {
                field: {
                    "deletion": deletion.get(field),
                    "neutral": neutral.get(field),
                }
                for field in comparable_row_fields
                if deletion.get(field) != neutral.get(field)
            }
            if mismatches:
                raise ValueError(
                    f"{regime} full-baseline mismatch for replicate "
                    f"{replicate_id}, sample_id={sample_id!r}: {mismatches}"
                )
            compared += 1
    return {
        "status": "exact_match",
        "regime": regime,
        "replicate_count": len(deletion_rows),
        "sample_comparison_count": compared,
        "compared_fields": [
            "prompt",
            "prompt_sha256",
            "generation_seed",
            "generated",
            "generated_sha256",
            "finish_reason",
        ],
    }


def _write_immutable_release(
    output: Path,
    release: Mapping[str, Any],
) -> str:
    serialised = (
        json.dumps(
            release,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if output.is_file():
        if output.read_text(encoding="utf-8") != serialised:
            raise ValueError(
                f"Refusing to overwrite incompatible canonical release "
                f"{output}; use a new output path"
            )
        return sha256_file(output)
    if output.exists():
        raise ValueError(f"Release output is not a regular file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(serialised, encoding="utf-8")
    temporary.replace(output)
    return sha256_file(output)


def finalize_loo_generation(
    *,
    run_root: str | Path,
    generation_spec: str | Path,
    generation_spec_sha256: str,
    analysis_manifest: str | Path,
    prompt_manifest: str | Path,
    output: str | Path,
) -> tuple[dict[str, Any], str]:
    """Validate a complete run and write its immutable release manifest."""

    root = Path(run_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {root}")
    spec_path = _resolve_within(
        root,
        generation_spec,
        label="generation spec",
    )
    analysis_path = _resolve_within(
        root,
        analysis_manifest,
        label="analysis manifest",
    )
    prompt_path = _resolve_within(
        root,
        prompt_manifest,
        label="prompt manifest",
    )
    expected_spec_file_sha256 = _require_digest(
        generation_spec_sha256,
        label="generation_spec_sha256",
    )
    spec = load_and_validate_generation_spec(
        spec_path,
        verify_artifact_paths=False,
        expected_file_sha256=expected_spec_file_sha256,
    )
    generation_config = spec.get("generation_config")
    if not isinstance(generation_config, Mapping):
        raise ValueError("Generation spec lacks generation_config")
    if (
        generation_config.get("generation_only") is not True
        or generation_config.get("training_performed") is not False
    ):
        raise ValueError(
            "Generation spec config must declare generation_only=true and "
            "training_performed=false"
        )

    analysis = _validate_analysis(
        manifest_path=analysis_path,
        run_root=root,
        spec=spec,
    )
    prompts, prompt_inventory = _validate_prompts(
        manifest_path=prompt_path,
        run_root=root,
        spec=spec,
        analysis=analysis,
    )
    if spec.get("population_id") != prompts.get("population_id"):
        raise ValueError(
            "Generation spec population_id differs from the prompt manifest"
        )
    if analysis.get("population_id") not in {None, prompts.get("population_id")}:
        raise ValueError(
            "Analysis population_id differs from the prompt manifest"
        )
    populations = generation_config.get("populations")
    invariants = generation_config.get("invariants")
    configured_sections = generation_config.get("sections")
    if (
        not isinstance(populations, Mapping)
        or not isinstance(invariants, Mapping)
        or not isinstance(configured_sections, list)
    ):
        raise ValueError(
            "Generation spec config must freeze populations, sections, and "
            "invariants"
        )
    population_config = populations.get(spec["population_id"])
    if not isinstance(population_config, Mapping):
        raise ValueError(
            "Generation spec config does not contain its population_id"
        )
    if population_config.get("phase") != spec.get("phase"):
        raise ValueError(
            "Generation spec phase differs from its frozen population config"
        )
    expected_dates = population_config.get("meeting_dates")
    expected_indicator_count = invariants.get(
        "indicator_count_per_meeting"
    )
    expected_section_count = invariants.get("section_count_per_meeting")
    configured_section_names = [
        str(entry.get("section_name") or "")
        for entry in configured_sections
        if isinstance(entry, Mapping)
    ]
    if (
        not isinstance(expected_dates, list)
        or not expected_dates
        or isinstance(expected_indicator_count, bool)
        or not isinstance(expected_indicator_count, int)
        or expected_indicator_count <= 0
        or isinstance(expected_section_count, bool)
        or not isinstance(expected_section_count, int)
        or expected_section_count <= 0
        or len(configured_section_names) != expected_section_count
        or any(not value for value in configured_section_names)
    ):
        raise ValueError(
            "Generation spec config contains an invalid frozen population or "
            "roster invariant"
        )
    if prompts["population_dates"] != expected_dates:
        raise ValueError(
            "Prompt population dates differ from the generation spec config"
        )
    if prompts["section_families"] != configured_section_names:
        raise ValueError(
            "Prompt section families differ from the generation spec config"
        )
    if prompts["indicator_count"] != expected_indicator_count:
        raise ValueError(
            "Prompt indicator count differs from the generation spec config"
        )
    expected_meeting_count = len(expected_dates)
    expected_unit_count = expected_meeting_count * expected_section_count
    expected_analysis_count = (
        expected_meeting_count * expected_indicator_count
    )
    expected_prompt_counts = {
        "meeting_count": expected_meeting_count,
        "section_count": expected_section_count,
        "unit_count": expected_unit_count,
        "indicator_count": expected_indicator_count,
        "analysis_block_count": expected_analysis_count,
        "full_prompt_count": expected_unit_count,
        "delete_prompt_count": expected_unit_count * expected_indicator_count,
        "neutral_prompt_count": expected_unit_count * expected_indicator_count,
    }
    prompt_count_mismatches = {
        field: {"expected": expected, "observed": prompts["counts"].get(field)}
        for field, expected in expected_prompt_counts.items()
        if prompts["counts"].get(field) != expected
    }
    if prompt_count_mismatches:
        raise ValueError(
            f"Prompt counts differ from the frozen design: "
            f"{prompt_count_mismatches}"
        )
    if analysis["row_count"] != expected_analysis_count:
        raise ValueError(
            "Analysis row count differs from the frozen design: "
            f"expected={expected_analysis_count}, "
            f"observed={analysis['row_count']}"
        )
    for folder, artifacts in prompt_inventory.items():
        for indicator, artifact in artifacts.items():
            if artifact["row_count"] != expected_unit_count:
                raise ValueError(
                    f"Prompt artifact {folder}/{indicator} has "
                    f"{artifact['row_count']} rows; expected "
                    f"{expected_unit_count}"
                )

    run_summaries: dict[str, dict[str, Any]] = {}
    baselines: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for expectation in EXPECTED_RUNS:
        summary, baseline_rows = _validate_generation_run(
            run_root=root,
            expectation=expectation,
            spec=spec,
            spec_path=spec_path,
            spec_file_sha256=expected_spec_file_sha256,
            prompt_summary=prompts,
            prompt_inventory=prompt_inventory,
        )
        run_summaries[expectation.name] = summary
        baselines[expectation.name] = baseline_rows

    primary_equivalence = _compare_baselines(
        regime="primary",
        deletion_summary=run_summaries["deletion_primary"],
        neutral_summary=run_summaries["neutral_primary"],
        deletion_rows=baselines["deletion_primary"],
        neutral_rows=baselines["neutral_primary"],
    )
    stochastic_equivalence = _compare_baselines(
        regime="stochastic",
        deletion_summary=run_summaries["deletion_stochastic"],
        neutral_summary=run_summaries["neutral_stochastic"],
        deletion_rows=baselines["deletion_stochastic"],
        neutral_rows=baselines["neutral_stochastic"],
    )

    payload = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "status": "complete",
        "generation_only": True,
        "training_performed": False,
        "scoring_performed": False,
        "run_root": str(root),
        "run_id": spec["run_id"],
        "phase": spec["phase"],
        "population_id": spec["population_id"],
        "generation_spec": {
            "relative_path": _relative_to_run_root(spec_path, root),
            "file_sha256": expected_spec_file_sha256,
            "payload_sha256": spec["integrity"]["payload_sha256"],
            "schema_version": spec["schema_version"],
        },
        "analysis": analysis,
        "prompts": prompts,
        "generation_runs": run_summaries,
        "baseline_equivalence": {
            "primary": primary_equivalence,
            "stochastic": stochastic_equivalence,
        },
        "validation_policy": {
            "expected_run_names": [
                expectation.name for expectation in EXPECTED_RUNS
            ],
            "expected_replicate_counts": {
                expectation.name: expectation.replicate_count
                for expectation in EXPECTED_RUNS
            },
            "normal_finish_required": True,
            "input_truncation": "forbidden",
            "token_limit_finish": "forbidden",
            "row_seed_inputs": ["replicate_seed", "sample_id"],
            "full_baseline_cross_strategy_match": "exact_text_and_sha256",
            "information_cutoff_policy": generation_config.get(
                "information_cutoff"
            ),
        },
    }
    release = seal_manifest(payload)
    output_path = Path(output).expanduser().resolve()
    release_file_sha256 = _write_immutable_release(output_path, release)
    return release, release_file_sha256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a complete generation-only canonical LOO run and write "
            "an immutable release manifest. No scoring or training is performed."
        )
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--generation-spec", required=True)
    parser.add_argument("--generation-spec-sha256", required=True)
    parser.add_argument("--analysis-manifest", required=True)
    parser.add_argument("--prompt-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        release, release_file_sha256 = finalize_loo_generation(
            run_root=args.run_root,
            generation_spec=args.generation_spec,
            generation_spec_sha256=args.generation_spec_sha256,
            analysis_manifest=args.analysis_manifest,
            prompt_manifest=args.prompt_manifest,
            output=args.output,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(f"status={release['status']}")
    print(f"output={Path(args.output).expanduser().resolve()}")
    print(f"file_sha256={release_file_sha256}")
    print(f"payload_sha256={release['integrity']['payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
