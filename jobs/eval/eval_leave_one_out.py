"""Canonical paired leave-one-out evaluation for Chapter 2.

This module replaces the historical ``shapley`` evaluation path for new runs.
It preserves the full and masked scores separately and defines the primary
estimand as

    delta = cos(full_output, target) - cos(masked_output, target).

Rows are paired by stable identifiers and replicate, never by physical line
position.  Summaries first average repeated generations within meeting and
then use the meeting as the unit of inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from scipy import stats

from open_r1.provenance import fingerprint_artifact_path, sha256_text, validate_sha256
from open_r1.validator.intervention import (
    has_line_block_boundaries,
    match_indicator_marker,
    single_contiguous_deletion,
)
from open_r1.validator.leave_one_out import leave_one_out_metrics_from_similarities
from process_fomc_report.generate_prompt_and_response.algo.common.response_templates import (
    parse_response_text,
)


SCHEMA_VERSION = "loo-paired-v3"
CANONICAL_GENERATION_SCHEMA_VERSION = "loo-generation-v4"
LEGACY_GENERATION_SCHEMA_VERSIONS = {
    "loo-generation-v2",
    "loo-generation-v3",
}
INTERVENTION_SCHEMA_VERSION = "loo-intervention-v2"
DEFAULT_BASELINE_INDICATOR = "None"
DEFAULT_REFERENCE_KEYS = ("meeting_date", "section_name")


class TripletScorer(Protocol):
    """Interface used by the evaluator and lightweight unit-test scorers."""

    def score_triplets(
        self,
        targets: list[str],
        full_outputs: list[str],
        masked_outputs: list[str],
        *,
        target_mode: str,
    ) -> list[tuple[float, float, float]]:
        """Return ``(s_full, s_masked, s_self)`` for each aligned triplet."""


class EmbeddingCosineTripletScorer:
    """Batched cosine scorer backed by the configured embedding model."""

    def __init__(self, model_path: str, *, embedding_batch_size: int | None = 8):
        # Import lazily so pairing and summary tests do not load torch/transformers.
        from open_r1.validator.cos.embedding_model import get_cached_embedding_model

        self.model_path = model_path
        self.model = get_cached_embedding_model(model_path, bs=embedding_batch_size)

    def score_triplets(
        self,
        targets: list[str],
        full_outputs: list[str],
        masked_outputs: list[str],
        *,
        target_mode: str,
    ) -> list[tuple[float, float, float]]:
        if not (len(targets) == len(full_outputs) == len(masked_outputs)):
            raise ValueError("targets, full_outputs, and masked_outputs must have equal lengths")
        if not targets:
            return []

        if target_mode == "full-output":
            full_embeddings = self.model.get_embeddings(full_outputs)
            masked_embeddings = self.model.get_embeddings(masked_outputs)
            self_scores = self.model.get_similarities(full_embeddings, masked_embeddings)
            return [(1.0, float(score), float(score)) for score in self_scores]

        target_embeddings = self.model.get_embeddings(targets)
        full_embeddings = self.model.get_embeddings(full_outputs)
        masked_embeddings = self.model.get_embeddings(masked_outputs)
        full_scores = self.model.get_similarities(target_embeddings, full_embeddings)
        masked_scores = self.model.get_similarities(target_embeddings, masked_embeddings)
        self_scores = self.model.get_similarities(full_embeddings, masked_embeddings)
        return [
            (float(full_score), float(masked_score), float(self_score))
            for full_score, masked_score, self_score in zip(
                full_scores,
                masked_scores,
                self_scores,
                strict=True,
            )
        ]


@dataclass(frozen=True)
class GeneratedArtifact:
    path: Path
    indicator: str
    context: str
    replicate_id: str
    generation_seed: int | None
    row_count: int
    sha256: str

    @property
    def pairing_key(self) -> tuple[str, str]:
        return self.context, self.replicate_id


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}")
            row = dict(row)
            row["_source_line"] = line_number
            rows.append(row)
    return rows


def _read_first_json_object(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}")
            return row
    raise ValueError(f"Generated artifact is empty: {path}")


def _parse_legacy_artifact_name(path: Path) -> tuple[str, str, str]:
    stem = path.stem
    if "_masked_" not in stem:
        raise ValueError(
            f"Cannot infer indicator/replicate from {path.name!r}; expected "
            "'<indicator>_masked_<context>_<replicate>.jsonl' or row metadata"
        )

    indicator, suffix = stem.split("_masked_", 1)
    replicate_match = re.match(r"^(?P<context>.+)_(?P<replicate>\d+)$", suffix)
    if replicate_match:
        return indicator, replicate_match.group("context"), replicate_match.group("replicate")
    return indicator, suffix, "legacy-single"


def discover_generated_artifacts(input_folder: Path) -> list[GeneratedArtifact]:
    """Discover generated JSONL artifacts without collapsing replicate files."""

    if not input_folder.is_dir():
        raise FileNotFoundError(f"Generated-output folder does not exist: {input_folder}")

    artifacts: list[GeneratedArtifact] = []
    parse_errors: list[str] = []
    for path in sorted(input_folder.rglob("*.jsonl")):
        try:
            first_row = _read_first_json_object(path)
        except ValueError as exc:
            parse_errors.append(str(exc))
            continue

        row_metadata_complete = all(
            str(first_row.get(field) or "").strip()
            for field in ("indicator", "evaluation_context", "replicate_id")
        )
        if row_metadata_complete:
            filename_indicator = str(first_row["indicator"]).strip()
            filename_context = str(first_row["evaluation_context"]).strip()
            filename_replicate = str(first_row["replicate_id"]).strip()
        else:
            try:
                (
                    filename_indicator,
                    filename_context,
                    filename_replicate,
                ) = _parse_legacy_artifact_name(path)
            except ValueError as exc:
                parse_errors.append(str(exc))
                continue

        indicator = str(first_row.get("indicator") or filename_indicator).strip()
        context = str(first_row.get("evaluation_context") or filename_context).strip()
        replicate_id = str(first_row.get("replicate_id") or filename_replicate).strip()
        seed_value = first_row.get("generation_seed")
        generation_seed = int(seed_value) if seed_value is not None and str(seed_value) != "" else None
        with path.open("r", encoding="utf-8") as handle:
            row_count = sum(1 for line in handle if line.strip())

        artifacts.append(
            GeneratedArtifact(
                path=path,
                indicator=indicator,
                context=context,
                replicate_id=replicate_id,
                generation_seed=generation_seed,
                row_count=row_count,
                sha256=_sha256_file(path),
            )
        )

    if parse_errors:
        raise ValueError(
            f"{len(parse_errors)} JSONL artifacts in {input_folder} are empty, "
            "invalid, or lack usable identity metadata. Examples: "
            + "; ".join(parse_errors[:3])
        )
    if not artifacts:
        raise ValueError(f"No generated masking artifacts found in {input_folder}")

    identities: dict[tuple[str, str, str], Path] = {}
    for artifact in artifacts:
        identity = (artifact.indicator, artifact.context, artifact.replicate_id)
        previous = identities.get(identity)
        if previous is not None:
            raise ValueError(
                "Duplicate generated artifact identity "
                f"{identity}: {previous} and {artifact.path}"
            )
        identities[identity] = artifact.path

    return artifacts


def load_and_validate_generation_manifest(
    input_folder: Path,
    artifacts: list[GeneratedArtifact],
    *,
    required: bool,
) -> tuple[dict | None, dict]:
    """Validate the complete artifact inventory emitted by masking generation."""

    manifest_path = input_folder / "generation_manifest.json"
    if not manifest_path.is_file():
        if required:
            raise FileNotFoundError(
                f"Generation manifest is required for a canonical run: {manifest_path}. "
                "Use --allow-missing-generation-manifest only for explicitly "
                "labelled legacy inputs."
            )
        return None, {
            "status": "missing_legacy_override",
            "path": str(manifest_path),
            "sha256": None,
        }

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid generation manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Generation manifest must be a JSON object: {manifest_path}")
    manifest_schema = manifest.get("schema_version")
    if manifest_schema not in {
        CANONICAL_GENERATION_SCHEMA_VERSION,
        *LEGACY_GENERATION_SCHEMA_VERSIONS,
    }:
        raise ValueError(
            f"Unsupported generation manifest schema in {manifest_path}: "
            f"{manifest_schema!r}"
        )
    if required and manifest_schema != CANONICAL_GENERATION_SCHEMA_VERSION:
        raise ValueError(
            f"Canonical runs require {CANONICAL_GENERATION_SCHEMA_VERSION!r} with "
            f"a validated intervention manifest; observed {manifest_schema!r}. "
            "Use --allow-missing-generation-manifest only for explicitly labelled "
            "legacy inputs."
        )

    generation_model_audit = {
        "status": "not_available_in_legacy_generation_manifest",
        "path": None,
        "sha256": None,
        "kind": None,
        "file_count": None,
        "total_bytes": None,
        "algorithm": None,
    }
    if manifest_schema == CANONICAL_GENERATION_SCHEMA_VERSION:
        model_artifact = manifest.get("model_artifact")
        if not isinstance(model_artifact, dict):
            raise ValueError(
                f"Canonical generation manifest lacks model_artifact: "
                f"{manifest_path}"
            )
        model_sha256 = validate_sha256(
            model_artifact.get("sha256"),
            label="generation model artifact SHA-256",
        )
        model_artifact_path = str(model_artifact.get("path") or "").strip()
        model_kind = str(model_artifact.get("kind") or "").strip()
        model_algorithm = str(model_artifact.get("algorithm") or "").strip()
        try:
            model_file_count = int(model_artifact.get("file_count"))
            model_total_bytes = int(model_artifact.get("total_bytes"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Generation model artifact has invalid inventory counts: "
                f"{manifest_path}"
            ) from exc
        if (
            not model_artifact_path
            or model_kind not in {"file", "directory"}
            or not model_algorithm
            or model_file_count < 1
            or model_total_bytes < 1
        ):
            raise ValueError(
                f"Generation model artifact metadata is incomplete: {manifest_path}"
            )
        manifest_model_path = str(manifest.get("model_path") or "").strip()
        if not manifest_model_path:
            raise ValueError(
                f"Generation manifest has no model_path: {manifest_path}"
            )
        if Path(manifest_model_path).expanduser().resolve() != Path(
            model_artifact_path
        ).expanduser().resolve():
            raise ValueError(
                f"Generation model path differs from its fingerprinted artifact: "
                f"{manifest_path}"
            )
        generation_model_audit = {
            "status": "fingerprint_declared_by_generation_manifest",
            "path": model_artifact_path,
            "sha256": model_sha256,
            "kind": model_kind,
            "file_count": model_file_count,
            "total_bytes": model_total_bytes,
            "algorithm": model_algorithm,
        }

    expected_rows = manifest.get("expected_artifacts")
    if not isinstance(expected_rows, list) or not expected_rows:
        raise ValueError(f"Generation manifest has no expected_artifacts: {manifest_path}")

    actual_by_identity = {
        (artifact.indicator, artifact.context, artifact.replicate_id): artifact
        for artifact in artifacts
    }
    expected_by_identity: dict[tuple[str, str, str], dict] = {}
    for expected in expected_rows:
        if not isinstance(expected, dict):
            raise ValueError(f"Invalid expected_artifacts entry in {manifest_path}")
        identity = (
            str(expected.get("indicator") or "").strip(),
            str(expected.get("context") or "").strip(),
            str(expected.get("replicate_id") or "").strip(),
        )
        if not all(identity):
            raise ValueError(
                f"Expected artifact has incomplete identity in {manifest_path}: {expected}"
            )
        if identity in expected_by_identity:
            raise ValueError(f"Duplicate expected artifact identity {identity} in {manifest_path}")
        expected_by_identity[identity] = expected

    expected_identities = set(expected_by_identity)
    actual_identities = set(actual_by_identity)
    if expected_identities != actual_identities:
        missing = sorted(expected_identities - actual_identities)
        extra = sorted(actual_identities - expected_identities)
        raise ValueError(
            "Generated artifacts do not match generation_manifest.json: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    for identity, expected in expected_by_identity.items():
        artifact = actual_by_identity[identity]
        expected_path = input_folder / str(expected.get("relative_path") or "")
        if expected_path.resolve() != artifact.path.resolve():
            raise ValueError(
                f"Manifest path mismatch for {identity}: "
                f"expected {expected_path}, observed {artifact.path}"
            )
        expected_count = int(expected.get("source_row_count", -1))
        if artifact.row_count != expected_count:
            raise ValueError(
                f"Row-count mismatch for {artifact.path}: "
                f"expected {expected_count}, observed {artifact.row_count}"
            )
        expected_hash = str(expected.get("output_sha256") or "")
        if artifact.sha256 != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for generated artifact {artifact.path}: "
                f"expected {expected_hash}, observed {artifact.sha256}"
            )
        if manifest_schema == CANONICAL_GENERATION_SCHEMA_VERSION:
            expected_model_hash = validate_sha256(
                expected.get("model_artifact_sha256"),
                label=f"model_artifact_sha256 for {identity}",
            )
            if expected_model_hash != generation_model_audit["sha256"]:
                raise ValueError(
                    f"Generated artifact model fingerprint mismatch for {identity}: "
                    f"{manifest_path}"
                )

    intervention_audit = {
        "status": "not_available_in_legacy_generation_manifest",
        "path": None,
        "sha256": None,
        "roster_id": None,
        "roster_file_sha256": None,
    }
    if manifest_schema == CANONICAL_GENERATION_SCHEMA_VERSION:
        intervention_metadata = manifest.get("intervention_manifest")
        if not isinstance(intervention_metadata, dict):
            raise ValueError(
                f"Canonical generation manifest lacks intervention_manifest: "
                f"{manifest_path}"
            )
        relative_path = str(
            intervention_metadata.get("relative_path") or ""
        ).strip()
        expected_intervention_hash = str(
            intervention_metadata.get("sha256") or ""
        ).strip()
        if not relative_path or not expected_intervention_hash:
            raise ValueError(
                f"Invalid intervention_manifest metadata in {manifest_path}"
            )
        intervention_path = (input_folder / relative_path).resolve()
        if input_folder.resolve() not in intervention_path.parents:
            raise ValueError(
                f"Intervention manifest must be inside the generated-output folder: "
                f"{intervention_path}"
            )
        if not intervention_path.is_file():
            raise FileNotFoundError(
                f"Intervention manifest declared by generation run is missing: "
                f"{intervention_path}"
            )
        observed_intervention_hash = _sha256_file(intervention_path)
        if observed_intervention_hash != expected_intervention_hash:
            raise ValueError(
                f"SHA-256 mismatch for intervention manifest {intervention_path}: "
                f"expected {expected_intervention_hash}, "
                f"observed {observed_intervention_hash}"
            )
        try:
            intervention_manifest = json.loads(
                intervention_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid intervention manifest {intervention_path}: {exc}"
            ) from exc
        if not isinstance(intervention_manifest, dict):
            raise ValueError(
                f"Intervention manifest must be a JSON object: {intervention_path}"
            )
        if intervention_manifest.get("schema_version") != INTERVENTION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported intervention manifest schema in {intervention_path}: "
                f"{intervention_manifest.get('schema_version')!r}"
            )
        if (
            intervention_metadata.get("schema_version")
            != intervention_manifest.get("schema_version")
        ):
            raise ValueError(
                f"Generation/intervention schema mismatch for {intervention_path}"
            )
        if (
            intervention_manifest.get("validation")
            != "exact_single_contiguous_prompt_deletion"
        ):
            raise ValueError(
                f"Intervention manifest does not attest exact one-block deletion: "
                f"{intervention_path}"
            )

        baseline_indicator = str(
            intervention_manifest.get("baseline_indicator") or ""
        ).strip()
        indicators = {
            str(value).strip()
            for value in intervention_manifest.get("indicators", [])
            if str(value).strip()
        }
        contexts = {
            str(value).strip()
            for value in intervention_manifest.get("contexts", [])
            if str(value).strip()
        }
        if not baseline_indicator or not indicators or not contexts:
            raise ValueError(
                f"Intervention manifest has an incomplete roster: "
                f"{intervention_path}"
            )
        marker_payload = intervention_manifest.get("indicator_markers")
        if not isinstance(marker_payload, dict) or set(marker_payload) != indicators:
            raise ValueError(
                f"Intervention manifest does not define markers for its exact "
                f"indicator universe: {intervention_path}"
            )
        indicator_markers: dict[str, list[str]] = {}
        for indicator in indicators:
            marker_values = marker_payload.get(indicator)
            if (
                not isinstance(marker_values, list)
                or not marker_values
                or any(not str(value).strip() for value in marker_values)
            ):
                raise ValueError(
                    f"Intervention manifest has invalid markers for "
                    f"{indicator!r}: {intervention_path}"
                )
            indicator_markers[indicator] = [
                str(value).strip() for value in marker_values
            ]
        required_intervention_attestations = {
            "indicator_identity_validation": (
                "declared_marker_in_removed_block_header"
            ),
            "block_boundary_validation": "complete_line_or_block_boundaries",
            "indicator_block_span_validation": (
                "pairwise_non_overlapping_per_sample"
            ),
            "distinct_masked_prompt_per_indicator": True,
        }
        for field, expected_value in required_intervention_attestations.items():
            if intervention_manifest.get(field) != expected_value:
                raise ValueError(
                    f"Intervention manifest lacks required attestation "
                    f"{field}={expected_value!r}: {intervention_path}"
                )
        if (
            intervention_metadata.get("roster_id")
            != intervention_manifest.get("roster_id")
            or intervention_metadata.get("roster_file_sha256")
            != intervention_manifest.get("roster_file_sha256")
        ):
            raise ValueError(
                f"Generation/intervention roster metadata mismatch for "
                f"{intervention_path}"
            )
        declared_prompt_artifacts = intervention_manifest.get("input_artifacts")
        if not isinstance(declared_prompt_artifacts, list):
            raise ValueError(
                f"Intervention manifest has no input_artifacts: {intervention_path}"
            )
        intervention_input_by_identity: dict[tuple[str, str], dict] = {}
        for entry in declared_prompt_artifacts:
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Invalid input_artifacts entry in {intervention_path}"
                )
            identity = (
                str(entry.get("indicator") or "").strip(),
                str(entry.get("context") or "").strip(),
            )
            if not all(identity) or identity in intervention_input_by_identity:
                raise ValueError(
                    f"Invalid or duplicate prompt artifact identity {identity} in "
                    f"{intervention_path}"
                )
            intervention_input_by_identity[identity] = entry
        expected_prompt_identities = {
            (indicator, context)
            for indicator in {baseline_indicator, *indicators}
            for context in contexts
        }
        if set(intervention_input_by_identity) != expected_prompt_identities:
            raise ValueError(
                f"Intervention prompt inventory does not cover its declared roster: "
                f"{intervention_path}"
            )

        generation_input_artifacts = manifest.get("input_artifacts")
        if not isinstance(generation_input_artifacts, list):
            raise ValueError(
                f"Generation manifest has no input_artifacts: {manifest_path}"
            )
        generation_input_by_identity = {
            (
                str(entry.get("indicator") or "").strip(),
                str(entry.get("context") or "").strip(),
            ): entry
            for entry in generation_input_artifacts
            if isinstance(entry, dict)
        }
        if set(generation_input_by_identity) != expected_prompt_identities:
            raise ValueError(
                f"Generation input inventory differs from the intervention roster: "
                f"{manifest_path}"
            )
        for identity, intervention_entry in intervention_input_by_identity.items():
            generation_entry = generation_input_by_identity[identity]
            for field in (
                "relative_path",
                "source_row_count",
                "source_file_sha256",
            ):
                if generation_entry.get(field) != intervention_entry.get(field):
                    raise ValueError(
                        f"Generation/intervention input mismatch for {identity}, "
                        f"field={field}: generation={generation_entry.get(field)!r}, "
                        f"intervention={intervention_entry.get(field)!r}"
                    )

        interventions = intervention_manifest.get("interventions")
        expected_intervention_count = sum(
            int(intervention_input_by_identity[(baseline_indicator, context)].get(
                "source_row_count",
                -1,
            ))
            for context in contexts
        ) * len(indicators)
        if (
            not isinstance(interventions, list)
            or len(interventions) != expected_intervention_count
        ):
            raise ValueError(
                f"Intervention row inventory is incomplete in {intervention_path}: "
                f"expected {expected_intervention_count}, "
                f"observed {len(interventions) if isinstance(interventions, list) else 'invalid'}"
            )
        intervention_keys: set[tuple[str, str, tuple[str, ...]]] = set()
        intervention_counts: dict[tuple[str, str], int] = {}
        intervention_by_key: dict[
            tuple[str, str, tuple[str, ...]],
            dict,
        ] = {}
        masked_hashes_by_sample: dict[
            tuple[str, tuple[str, ...]],
            dict[str, str],
        ] = {}
        full_hashes_by_sample: dict[
            tuple[str, tuple[str, ...]],
            str,
        ] = {}
        deletion_spans_by_sample: dict[
            tuple[str, tuple[str, ...]],
            list[tuple[int, int, str]],
        ] = {}
        for entry in interventions:
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Invalid intervention entry in {intervention_path}"
                )
            indicator = str(entry.get("indicator") or "").strip()
            context = str(entry.get("context") or "").strip()
            row_key_value = entry.get("row_key")
            if (
                indicator not in indicators
                or context not in contexts
                or not isinstance(row_key_value, list)
                or not row_key_value
                or not str(entry.get("meeting_date") or "").strip()
                or not str(entry.get("section_name") or "").strip()
            ):
                raise ValueError(
                    f"Intervention entry has incomplete identity in "
                    f"{intervention_path}: {entry}"
                )
            hashes = (
                entry.get("full_prompt_sha256"),
                entry.get("masked_prompt_sha256"),
                entry.get("removed_block_sha256"),
            )
            for label, value in zip(
                (
                    "full_prompt_sha256",
                    "masked_prompt_sha256",
                    "removed_block_sha256",
                ),
                hashes,
                strict=True,
            ):
                validate_sha256(
                    value,
                    label=f"{label} in {intervention_path}",
                )
            matched_marker = str(
                entry.get("matched_indicator_marker") or ""
            ).strip()
            if (
                matched_marker not in indicator_markers[indicator]
                or entry.get("line_block_boundaries_validated") is not True
            ):
                raise ValueError(
                    f"Intervention entry lacks indicator/block identity evidence in "
                    f"{intervention_path}: {entry}"
                )
            intervention_key = (
                indicator,
                context,
                tuple(str(value) for value in row_key_value),
            )
            if intervention_key in intervention_keys:
                raise ValueError(
                    f"Duplicate intervention identity {intervention_key} in "
                    f"{intervention_path}"
                )
            intervention_keys.add(intervention_key)
            intervention_by_key[intervention_key] = entry
            intervention_counts[(indicator, context)] = (
                intervention_counts.get((indicator, context), 0) + 1
            )
            sample_key = (context, intervention_key[2])
            full_prompt_hash = str(entry["full_prompt_sha256"])
            previous_full_hash = full_hashes_by_sample.setdefault(
                sample_key,
                full_prompt_hash,
            )
            if previous_full_hash != full_prompt_hash:
                raise ValueError(
                    f"Intervention entries disagree on the full prompt for "
                    f"{sample_key!r}: {intervention_path}"
                )
            masked_prompt_hash = str(entry["masked_prompt_sha256"])
            previous_indicator = masked_hashes_by_sample.setdefault(
                sample_key,
                {},
            ).get(masked_prompt_hash)
            if previous_indicator is not None:
                raise ValueError(
                    f"Duplicate masked prompt for {sample_key!r}: indicators "
                    f"{previous_indicator!r} and {indicator!r} in "
                    f"{intervention_path}"
                )
            masked_hashes_by_sample[sample_key][masked_prompt_hash] = indicator
            try:
                deletion_span = (
                    int(entry.get("deletion_start")),
                    int(entry.get("deletion_end")),
                    indicator,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Intervention entry has an invalid deletion span in "
                    f"{intervention_path}: {entry}"
                ) from exc
            deletion_spans_by_sample.setdefault(sample_key, []).append(
                deletion_span
            )

        for sample_key, spans in deletion_spans_by_sample.items():
            sorted_spans = sorted(spans)
            for previous, current in zip(
                sorted_spans,
                sorted_spans[1:],
                strict=False,
            ):
                if current[0] < previous[1]:
                    raise ValueError(
                        f"Overlapping indicator-block deletions for "
                        f"{sample_key!r} in {intervention_path}: "
                        f"{previous[2]!r} overlaps {current[2]!r}"
                    )
        for indicator in indicators:
            for context in contexts:
                expected_rows = int(
                    intervention_input_by_identity[
                        (baseline_indicator, context)
                    ].get("source_row_count", -1)
                )
                if intervention_counts.get((indicator, context), 0) != expected_rows:
                    raise ValueError(
                        f"Intervention coverage mismatch for "
                        f"{indicator}/{context} in {intervention_path}"
                    )

        try:
            simulation_step = int(manifest.get("simulation_step"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Generation manifest has invalid simulation_step: {manifest_path}"
            ) from exc
        if simulation_step < 1:
            raise ValueError(
                f"Generation manifest has invalid simulation_step: {manifest_path}"
            )
        expected_output_identities = {
            (indicator, context, str(replicate))
            for indicator, context in expected_prompt_identities
            for replicate in range(simulation_step)
        }
        if set(expected_by_identity) != expected_output_identities:
            raise ValueError(
                f"Generated-output inventory does not cover the frozen "
                f"indicator/context/replicate universe in {manifest_path}"
            )
        for identity, output_entry in expected_by_identity.items():
            prompt_entry = intervention_input_by_identity[identity[:2]]
            if (
                output_entry.get("source_file_sha256")
                != prompt_entry.get("source_file_sha256")
                or output_entry.get("source_row_count")
                != prompt_entry.get("source_row_count")
            ):
                raise ValueError(
                    f"Generated artifact source provenance mismatch for {identity} "
                    f"in {manifest_path}"
                )

        representative_entry_by_sample: dict[
            tuple[str, tuple[str, ...]],
            dict,
        ] = {}
        for (_indicator, context, row_key), entry in intervention_by_key.items():
            sample_key = (context, row_key)
            representative = representative_entry_by_sample.setdefault(
                sample_key,
                entry,
            )
            for field in ("meeting_date", "section_name", "full_prompt_sha256"):
                if representative.get(field) != entry.get(field):
                    raise ValueError(
                        f"Intervention entries disagree on {field} for "
                        f"{sample_key!r}: {intervention_path}"
                    )

        prompt_by_intervention_identity: dict[
            tuple[str, str, tuple[str, ...]],
            str,
        ] = {}
        for artifact_identity, artifact in actual_by_identity.items():
            indicator, context, _replicate_id = artifact_identity
            artifact_rows = _read_jsonl(artifact.path)
            observed_row_keys: set[tuple[str, ...]] = set()
            expected_row_keys = {
                row_key
                for sample_context, row_key in representative_entry_by_sample
                if sample_context == context
            }
            for row in artifact_rows:
                row_key = _row_pair_key(row)
                if row_key in observed_row_keys:
                    raise ValueError(
                        f"Duplicate row key {row_key!r} in {artifact.path}"
                    )
                observed_row_keys.add(row_key)
                if indicator == baseline_indicator:
                    expected_entry = representative_entry_by_sample.get(
                        (context, row_key)
                    )
                    expected_prompt_hash = (
                        None
                        if expected_entry is None
                        else expected_entry.get("full_prompt_sha256")
                    )
                else:
                    expected_entry = intervention_by_key.get(
                        (indicator, context, row_key)
                    )
                    expected_prompt_hash = (
                        None
                        if expected_entry is None
                        else expected_entry.get("masked_prompt_sha256")
                    )
                if expected_entry is None or expected_prompt_hash is None:
                    raise ValueError(
                        f"Generated row {row_key!r} in {artifact.path} is not "
                        "declared by the intervention manifest"
                    )

                prompt = str(row.get("prompt") or "")
                observed_prompt_hash = sha256_text(prompt)
                if (
                    not prompt
                    or observed_prompt_hash != expected_prompt_hash
                    or row.get("source_prompt_sha256") != expected_prompt_hash
                ):
                    raise ValueError(
                        f"Generated row {row_key!r} in {artifact.path} is not "
                        "bound to the attested intervention prompt"
                    )
                if (
                    _normalise_key_value(
                        "meeting_date",
                        row.get("meeting_date"),
                    )
                    != str(expected_entry.get("meeting_date") or "")
                    or _normalise_key_value(
                        "section_name",
                        row.get("section_name"),
                    )
                    != str(expected_entry.get("section_name") or "")
                ):
                    raise ValueError(
                        f"Generated row identity differs from the intervention "
                        f"manifest for {row_key!r} in {artifact.path}"
                    )
                if (
                    validate_sha256(
                        row.get("generation_model_sha256"),
                        label=f"generation_model_sha256 in {artifact.path}",
                    )
                    != generation_model_audit["sha256"]
                ):
                    raise ValueError(
                        f"Generated row model fingerprint mismatch in "
                        f"{artifact.path}"
                    )

                prompt_identity = (indicator, context, row_key)
                previous_prompt = prompt_by_intervention_identity.setdefault(
                    prompt_identity,
                    prompt,
                )
                if previous_prompt != prompt:
                    raise ValueError(
                        f"Replicates disagree on the source prompt for "
                        f"{prompt_identity!r}"
                    )
            if observed_row_keys != expected_row_keys:
                missing_keys = sorted(expected_row_keys - observed_row_keys)
                extra_keys = sorted(observed_row_keys - expected_row_keys)
                raise ValueError(
                    f"Generated artifact row universe differs from the "
                    f"intervention manifest for {artifact.path}: "
                    f"missing={missing_keys[:5]}, extra={extra_keys[:5]}"
                )

        for (indicator, context, row_key), entry in intervention_by_key.items():
            full_prompt = prompt_by_intervention_identity[
                (baseline_indicator, context, row_key)
            ]
            masked_prompt = prompt_by_intervention_identity[
                (indicator, context, row_key)
            ]
            try:
                deletion_start, deletion_end, removed_block = (
                    single_contiguous_deletion(full_prompt, masked_prompt)
                )
            except ValueError as exc:
                raise ValueError(
                    f"Generated prompts do not reproduce the declared "
                    f"intervention for {(indicator, context, row_key)!r}"
                ) from exc
            matched_marker = match_indicator_marker(
                removed_block,
                indicator_markers[indicator],
            )
            if (
                sha256_text(full_prompt) != entry.get("full_prompt_sha256")
                or sha256_text(masked_prompt) != entry.get("masked_prompt_sha256")
                or sha256_text(removed_block) != entry.get("removed_block_sha256")
                or deletion_start != entry.get("deletion_start")
                or deletion_end != entry.get("deletion_end")
                or matched_marker != entry.get("matched_indicator_marker")
                or not has_line_block_boundaries(
                    full_prompt,
                    deletion_start,
                    deletion_end,
                )
            ):
                raise ValueError(
                    f"Generated prompts fail indicator/block intervention "
                    f"validation for {(indicator, context, row_key)!r}"
                )
        intervention_audit = {
            "status": "validated_against_generation_manifest",
            "path": str(intervention_path),
            "sha256": observed_intervention_hash,
            "roster_id": intervention_manifest.get("roster_id"),
            "roster_file_sha256": intervention_manifest.get(
                "roster_file_sha256"
            ),
            "indicators": len(indicators),
            "contexts": len(contexts),
            "interventions": len(interventions),
            "indicator_identity_validation": (
                "declared_marker_in_removed_block_header"
            ),
            "block_boundary_validation": (
                "complete_line_or_block_boundaries"
            ),
            "indicator_block_span_validation": (
                "pairwise_non_overlapping_per_sample"
            ),
            "generated_rows_bound_to_interventions": True,
        }

    return manifest, {
        "status": "validated",
        "path": str(manifest_path),
        "sha256": _sha256_file(manifest_path),
        "schema_version": manifest_schema,
        "generation_model_artifact": generation_model_audit,
        "intervention_manifest": intervention_audit,
    }


def pair_generated_artifacts(
    artifacts: list[GeneratedArtifact],
    *,
    baseline_indicator: str = DEFAULT_BASELINE_INDICATOR,
) -> list[tuple[GeneratedArtifact, GeneratedArtifact]]:
    """Match every masked artifact to the full artifact for the same replicate."""

    baselines = {
        artifact.pairing_key: artifact
        for artifact in artifacts
        if artifact.indicator == baseline_indicator
    }
    if not baselines:
        raise ValueError(
            f"No full-output artifact with indicator={baseline_indicator!r} was found"
        )

    baseline_replicates: dict[str, set[str]] = {}
    masked_replicates: dict[tuple[str, str], set[str]] = {}
    for artifact in artifacts:
        if artifact.indicator == baseline_indicator:
            baseline_replicates.setdefault(artifact.context, set()).add(
                artifact.replicate_id
            )
        else:
            masked_replicates.setdefault(
                (artifact.indicator, artifact.context),
                set(),
            ).add(artifact.replicate_id)

    coverage_errors: list[str] = []
    for (indicator, context), replicate_ids in sorted(masked_replicates.items()):
        expected_replicates = baseline_replicates.get(context, set())
        if replicate_ids != expected_replicates:
            coverage_errors.append(
                f"{indicator}/{context}: expected {sorted(expected_replicates)}, "
                f"observed {sorted(replicate_ids)}"
            )
    if coverage_errors:
        raise ValueError(
            "Masked indicators do not cover the complete baseline replicate set: "
            + "; ".join(coverage_errors[:5])
        )

    pairs: list[tuple[GeneratedArtifact, GeneratedArtifact]] = []
    missing: list[GeneratedArtifact] = []
    for masked_artifact in artifacts:
        if masked_artifact.indicator == baseline_indicator:
            continue
        full_artifact = baselines.get(masked_artifact.pairing_key)
        if full_artifact is None:
            missing.append(masked_artifact)
            continue
        pairs.append((full_artifact, masked_artifact))

    if missing:
        examples = ", ".join(str(item.path) for item in missing[:3])
        raise ValueError(
            f"{len(missing)} masked artifacts have no full-output artifact for the "
            f"same context/replicate. Examples: {examples}"
        )
    if not pairs:
        raise ValueError("No masked artifacts were found after excluding the full-output baseline")
    return pairs


def _normalise_key_value(field: str, value: object) -> str:
    text = str(value or "").strip()
    if field == "meeting_date":
        return text[:10]
    return text


def _row_pair_key(row: dict) -> tuple[str, ...]:
    meeting_date = _normalise_key_value("meeting_date", row.get("meeting_date"))
    section_name = _normalise_key_value("section_name", row.get("section_name"))
    if not meeting_date or not section_name:
        raise ValueError(
            "Generated row requires non-empty meeting_date and section_name"
        )

    sample_id = _normalise_key_value("sample_id", row.get("sample_id"))
    if sample_id:
        return "sample_id", sample_id

    index = row.get("source_index", row.get("index"))
    if index is not None and str(index) != "":
        return "legacy", meeting_date, section_name, str(index)
    raise ValueError(
        "Generated row has no stable key; expected sample_id or "
        "(meeting_date, section_name, source_index/index)"
    )


def _unique_row_index(rows: list[dict], *, path: Path) -> dict[tuple[str, ...], dict]:
    indexed: dict[tuple[str, ...], dict] = {}
    for row in rows:
        key = _row_pair_key(row)
        if key in indexed:
            raise ValueError(f"Duplicate generated row key {key!r} in {path}")
        indexed[key] = row
    return indexed


def _reference_key(row: dict, fields: tuple[str, ...]) -> tuple[str, ...]:
    values = tuple(_normalise_key_value(field, row.get(field)) for field in fields)
    if any(not value for value in values):
        raise ValueError(f"Reference key {fields!r} contains an empty value: {values!r}")
    return values


def build_reference_index(
    reference_rows: list[dict],
    *,
    key_fields: tuple[str, ...] = DEFAULT_REFERENCE_KEYS,
    text_field: str = "response",
    duplicate_policy: str = "error",
) -> dict[tuple[str, ...], str]:
    """Build a validated many-generated-rows-to-one-reference lookup."""

    grouped: dict[tuple[str, ...], list[str]] = {}
    for row in reference_rows:
        key = _reference_key(row, key_fields)
        text = parse_response_text(row.get(text_field)).answer.strip()
        if not text:
            raise ValueError(f"Reference row {key!r} has empty {text_field!r}")
        grouped.setdefault(key, []).append(text)

    index: dict[tuple[str, ...], str] = {}
    for key, texts in grouped.items():
        unique_texts = list(dict.fromkeys(texts))
        if len(unique_texts) == 1:
            index[key] = unique_texts[0]
        elif duplicate_policy == "concatenate":
            index[key] = "\n\n".join(unique_texts)
        else:
            raise ValueError(
                f"Reference key {key!r} maps to {len(unique_texts)} distinct texts; "
                "use --reference-duplicate-policy concatenate only when the source "
                "intentionally stores one section in multiple rows"
            )
    return index


def _record_exclusion(
    exclusions: list[dict],
    *,
    reason: str,
    artifact: GeneratedArtifact,
    row_key: tuple[str, ...] | None = None,
    details: str | None = None,
) -> None:
    exclusions.append(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "excluded",
            "reason": reason,
            "indicator": artifact.indicator,
            "context": artifact.context,
            "replicate_id": artifact.replicate_id,
            "row_key": None if row_key is None else list(row_key),
            "artifact": str(artifact.path),
            "details": details,
        }
    )


def _raise_or_exclude(
    exclusions: list[dict],
    *,
    unmatched_policy: str,
    reason: str,
    artifact: GeneratedArtifact,
    row_key: tuple[str, ...] | None = None,
    details: str | None = None,
) -> None:
    if unmatched_policy == "error":
        suffix = f" ({details})" if details else ""
        raise ValueError(
            f"{reason} for indicator={artifact.indicator!r}, "
            f"replicate={artifact.replicate_id!r}, row_key={row_key!r}{suffix}"
        )
    _record_exclusion(
        exclusions,
        reason=reason,
        artifact=artifact,
        row_key=row_key,
        details=details,
    )


def _pair_id(
    *,
    indicator: str,
    context: str,
    replicate_id: str,
    row_key: tuple[str, ...],
) -> str:
    payload = json.dumps(
        [indicator, context, replicate_id, list(row_key)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _metadata_is_present(value: object) -> bool:
    return value is not None and str(value).strip() != ""


def _normalise_generation_metadata(field: str, value: object) -> object:
    if field in {
        "decoding_temperature",
        "decoding_top_p",
    }:
        return float(value)
    if field in {
        "generation_batch_size",
        "generation_position",
        "max_new_tokens",
    }:
        return int(value)
    return str(value).strip()


def prepare_paired_rows(
    full_artifact: GeneratedArtifact,
    masked_artifact: GeneratedArtifact,
    *,
    target_mode: str,
    reference_index: dict[tuple[str, ...], str] | None,
    reference_key_fields: tuple[str, ...],
    unmatched_policy: str,
    exclusions: list[dict],
    full_row_cache: dict[Path, list[dict]] | None = None,
) -> list[dict]:
    """Join full/masked rows by stable key and attach the fixed target text."""

    full_row_cache = full_row_cache if full_row_cache is not None else {}
    if full_artifact.path not in full_row_cache:
        full_row_cache[full_artifact.path] = _read_jsonl(full_artifact.path)
    full_rows = full_row_cache[full_artifact.path]
    masked_rows = _read_jsonl(masked_artifact.path)
    full_index = _unique_row_index(full_rows, path=full_artifact.path)
    masked_index = _unique_row_index(masked_rows, path=masked_artifact.path)

    full_keys = set(full_index)
    masked_keys = set(masked_index)
    for key in sorted(full_keys - masked_keys):
        _raise_or_exclude(
            exclusions,
            unmatched_policy=unmatched_policy,
            reason="missing_masked_output",
            artifact=masked_artifact,
            row_key=key,
        )
    for key in sorted(masked_keys - full_keys):
        _raise_or_exclude(
            exclusions,
            unmatched_policy=unmatched_policy,
            reason="missing_full_output",
            artifact=masked_artifact,
            row_key=key,
        )

    prepared: list[dict] = []
    for row_key in sorted(full_keys & masked_keys):
        full_row = full_index[row_key]
        masked_row = masked_index[row_key]

        artifact_metadata_errors: list[str] = []
        for side, row, artifact in (
            ("full", full_row, full_artifact),
            ("masked", masked_row, masked_artifact),
        ):
            for field, expected in (
                ("indicator", artifact.indicator),
                ("evaluation_context", artifact.context),
                ("replicate_id", artifact.replicate_id),
            ):
                observed = row.get(field)
                if _metadata_is_present(observed) and str(observed).strip() != str(expected):
                    artifact_metadata_errors.append(
                        f"{side}.{field}: expected={expected!r}, observed={observed!r}"
                    )
        if artifact_metadata_errors:
            _raise_or_exclude(
                exclusions,
                unmatched_policy=unmatched_policy,
                reason="artifact_row_metadata_mismatch",
                artifact=masked_artifact,
                row_key=row_key,
                details="; ".join(artifact_metadata_errors),
            )
            continue

        row_identity_errors: list[str] = []
        for field in ("sample_id", "meeting_date", "section_name"):
            full_value = _normalise_key_value(field, full_row.get(field))
            masked_value = _normalise_key_value(field, masked_row.get(field))
            if full_value and masked_value and full_value != masked_value:
                row_identity_errors.append(
                    f"{field}: full={full_value!r}, masked={masked_value!r}"
                )
        if row_identity_errors:
            _raise_or_exclude(
                exclusions,
                unmatched_policy=unmatched_policy,
                reason="row_identity_metadata_mismatch",
                artifact=masked_artifact,
                row_key=row_key,
                details="; ".join(row_identity_errors),
            )
            continue

        full_seed = full_row.get("generation_seed", full_artifact.generation_seed)
        masked_seed = masked_row.get("generation_seed", masked_artifact.generation_seed)
        if _metadata_is_present(full_seed) != _metadata_is_present(masked_seed):
            _raise_or_exclude(
                exclusions,
                unmatched_policy=unmatched_policy,
                reason="incomplete_generation_seed_metadata",
                artifact=masked_artifact,
                row_key=row_key,
                details=f"full={full_seed}, masked={masked_seed}",
            )
            continue
        if (
            _metadata_is_present(full_seed)
            and _metadata_is_present(masked_seed)
            and int(full_seed) != int(masked_seed)
        ):
            _raise_or_exclude(
                exclusions,
                unmatched_policy=unmatched_policy,
                reason="generation_seed_mismatch",
                artifact=masked_artifact,
                row_key=row_key,
                details=f"full={full_seed}, masked={masked_seed}",
            )
            continue

        paired_generation_fields = (
            "replicate_id",
            "generation_model",
            "generation_model_sha256",
            "generation_batch_size",
            "generation_position",
            "decoding_temperature",
            "decoding_top_p",
            "max_new_tokens",
            "masking_strategy",
        )
        generation_metadata_errors: list[str] = []
        complete_generation_fields = 0
        for field in paired_generation_fields:
            full_value = full_row.get(field)
            masked_value = masked_row.get(field)
            full_present = _metadata_is_present(full_value)
            masked_present = _metadata_is_present(masked_value)
            if full_present != masked_present:
                generation_metadata_errors.append(
                    f"{field}: full={full_value!r}, masked={masked_value!r}"
                )
                continue
            if not full_present:
                continue
            complete_generation_fields += 1
            try:
                normalised_full = _normalise_generation_metadata(field, full_value)
                normalised_masked = _normalise_generation_metadata(field, masked_value)
            except (TypeError, ValueError):
                generation_metadata_errors.append(
                    f"{field}: invalid full={full_value!r}, masked={masked_value!r}"
                )
                continue
            if normalised_full != normalised_masked:
                generation_metadata_errors.append(
                    f"{field}: full={full_value!r}, masked={masked_value!r}"
                )
        if generation_metadata_errors:
            _raise_or_exclude(
                exclusions,
                unmatched_policy=unmatched_policy,
                reason="generation_configuration_mismatch",
                artifact=masked_artifact,
                row_key=row_key,
                details="; ".join(generation_metadata_errors),
            )
            continue

        full_output = parse_response_text(full_row.get("generated")).answer.strip()
        masked_output = parse_response_text(masked_row.get("generated")).answer.strip()
        if not full_output or not masked_output:
            _raise_or_exclude(
                exclusions,
                unmatched_policy=unmatched_policy,
                reason="empty_generated_answer",
                artifact=masked_artifact,
                row_key=row_key,
            )
            continue

        if target_mode == "full-output":
            target_text = full_output
            target_key: tuple[str, ...] | None = None
        else:
            if reference_index is None:
                raise ValueError("reference_index is required for actual-minutes mode")
            try:
                target_key = _reference_key(masked_row, reference_key_fields)
            except ValueError as exc:
                _raise_or_exclude(
                    exclusions,
                    unmatched_policy=unmatched_policy,
                    reason="invalid_reference_key",
                    artifact=masked_artifact,
                    row_key=row_key,
                    details=str(exc),
                )
                continue
            target_text = reference_index.get(target_key, "")
            if not target_text:
                _raise_or_exclude(
                    exclusions,
                    unmatched_policy=unmatched_policy,
                    reason="missing_actual_minutes",
                    artifact=masked_artifact,
                    row_key=row_key,
                    details=f"reference_key={target_key!r}",
                )
                continue

        meeting_date = _normalise_key_value(
            "meeting_date",
            masked_row.get("meeting_date") or full_row.get("meeting_date"),
        )
        section_name = _normalise_key_value(
            "section_name",
            masked_row.get("section_name") or full_row.get("section_name"),
        )
        sample_id = _normalise_key_value(
            "sample_id",
            masked_row.get("sample_id") or full_row.get("sample_id"),
        )
        generation_seed = full_seed if full_seed is not None else masked_seed
        if (
            _metadata_is_present(full_seed)
            and complete_generation_fields == len(paired_generation_fields)
        ):
            pairing_quality = "matched_complete_generation_metadata"
        elif _metadata_is_present(full_seed):
            pairing_quality = "matched_generation_seed_partial_metadata"
        else:
            pairing_quality = "legacy_replicate_label_only"

        prepared.append(
            {
                "schema_version": SCHEMA_VERSION,
                "metric": "signed_leave_one_out_alignment_delta",
                "target_mode": target_mode,
                "indicator": masked_artifact.indicator,
                "context": masked_artifact.context,
                "meeting_date": meeting_date,
                "section_name": section_name,
                "sample_id": sample_id or None,
                "source_index": masked_row.get(
                    "source_index",
                    masked_row.get("index"),
                ),
                "row_key": list(row_key),
                "reference_key": None if target_key is None else list(target_key),
                "replicate_id": masked_artifact.replicate_id,
                "generation_seed": None if generation_seed is None else int(generation_seed),
                "full_generation_seed": (
                    None if not _metadata_is_present(full_seed) else int(full_seed)
                ),
                "masked_generation_seed": (
                    None if not _metadata_is_present(masked_seed) else int(masked_seed)
                ),
                "pairing_quality": pairing_quality,
                "pair_id": _pair_id(
                    indicator=masked_artifact.indicator,
                    context=masked_artifact.context,
                    replicate_id=masked_artifact.replicate_id,
                    row_key=row_key,
                ),
                "full_output": full_output,
                "masked_output": masked_output,
                "target_text": target_text,
                "full_artifact": str(full_artifact.path),
                "masked_artifact": str(masked_artifact.path),
                "full_artifact_sha256": full_artifact.sha256,
                "masked_artifact_sha256": masked_artifact.sha256,
                "full_source_file_sha256": full_row.get("source_file_sha256"),
                "masked_source_file_sha256": masked_row.get("source_file_sha256"),
                "full_source_prompt_sha256": full_row.get(
                    "source_prompt_sha256"
                ),
                "masked_source_prompt_sha256": masked_row.get(
                    "source_prompt_sha256"
                ),
                "masking_strategy": masked_row.get("masking_strategy"),
                "generation_model": masked_row.get("generation_model")
                or full_row.get("generation_model"),
                "generation_model_sha256": masked_row.get(
                    "generation_model_sha256"
                )
                or full_row.get("generation_model_sha256"),
                "full_generation_model": full_row.get("generation_model"),
                "masked_generation_model": masked_row.get("generation_model"),
                "generation_batch_size": masked_row.get("generation_batch_size")
                or full_row.get("generation_batch_size"),
                "generation_position": masked_row.get("generation_position")
                if _metadata_is_present(masked_row.get("generation_position"))
                else full_row.get("generation_position"),
                "full_generation_position": full_row.get("generation_position"),
                "masked_generation_position": masked_row.get("generation_position"),
                "decoding_temperature": masked_row.get("decoding_temperature")
                if _metadata_is_present(masked_row.get("decoding_temperature"))
                else full_row.get("decoding_temperature"),
                "decoding_top_p": masked_row.get("decoding_top_p")
                if _metadata_is_present(masked_row.get("decoding_top_p"))
                else full_row.get("decoding_top_p"),
                "max_new_tokens": masked_row.get("max_new_tokens")
                or full_row.get("max_new_tokens"),
                "status": "paired",
            }
        )
    return prepared


def score_prepared_rows(
    prepared_rows: list[dict],
    *,
    scorer: TripletScorer,
    target_mode: str,
    embedding_model_path: str,
    embedding_model_sha256: str,
    score_chunk_size: int = 256,
) -> list[dict]:
    """Attach paired LOO metrics to prepared rows using batched scoring."""

    if score_chunk_size < 1:
        raise ValueError("score_chunk_size must be positive")

    scored_rows: list[dict] = []
    for start in range(0, len(prepared_rows), score_chunk_size):
        chunk = prepared_rows[start : start + score_chunk_size]
        score_triples = scorer.score_triplets(
            [row["target_text"] for row in chunk],
            [row["full_output"] for row in chunk],
            [row["masked_output"] for row in chunk],
            target_mode=target_mode,
        )
        if len(score_triples) != len(chunk):
            raise ValueError(
                f"Scorer returned {len(score_triples)} rows for a {len(chunk)}-row chunk"
            )

        for row, (similarity_full, similarity_masked, self_similarity) in zip(
            chunk,
            score_triples,
            strict=True,
        ):
            metrics = leave_one_out_metrics_from_similarities(
                similarity_full,
                similarity_masked,
                self_similarity=self_similarity,
            )
            scored = dict(row)
            scored.update(metrics)
            scored["embedding_model_path"] = embedding_model_path
            scored["embedding_model_sha256"] = embedding_model_sha256
            scored["status"] = "scored"
            scored_rows.append(scored)
    return scored_rows


def _cluster_bootstrap_interval(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if len(values) < 2:
        return math.nan, math.nan

    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for index in range(samples):
        means[index] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
    alpha = 1.0 - confidence
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def _holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [math.nan] * len(p_values)
    valid = [(index, value) for index, value in enumerate(p_values) if math.isfinite(value)]
    valid.sort(key=lambda item: item[1])
    running_max = 0.0
    total = len(valid)
    for rank, (original_index, p_value) in enumerate(valid):
        candidate = min(1.0, (total - rank) * p_value)
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted


def summarise_scored_rows(
    scored_rows: list[dict],
    *,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 20260728,
) -> list[dict]:
    """Summarise paired deltas using meeting-level clusters."""

    if not scored_rows:
        raise ValueError("No scored LOO rows are available to summarise")

    frame = pd.DataFrame(scored_rows)
    required = {
        "target_mode",
        "context",
        "indicator",
        "section_name",
        "meeting_date",
        "replicate_id",
        "row_key",
        "similarity_full",
        "similarity_masked",
        "delta",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Scored rows are missing required columns: {sorted(missing)}")
    if frame["meeting_date"].fillna("").eq("").any():
        raise ValueError("meeting_date is required for meeting-clustered summaries")

    summaries: list[dict] = []
    grouped = frame.groupby(
        ["target_mode", "context", "indicator", "section_name"],
        dropna=False,
        sort=True,
    )
    for group_index, (
        (target_mode, context, indicator, section_name),
        group,
    ) in enumerate(grouped):
        meeting_replicate_level = (
            group.groupby(["meeting_date", "replicate_id"], sort=True)
            .agg(
                similarity_full=("similarity_full", "mean"),
                similarity_masked=("similarity_masked", "mean"),
                delta=("delta", "mean"),
                n_rows=("delta", "size"),
            )
            .reset_index()
        )
        meeting_level = (
            meeting_replicate_level.groupby("meeting_date", sort=True)
            .agg(
                similarity_full=("similarity_full", "mean"),
                similarity_masked=("similarity_masked", "mean"),
                delta=("delta", "mean"),
            )
            .reset_index()
        )
        deltas = meeting_level["delta"].to_numpy(dtype=float)
        n_meetings = len(deltas)
        mean_delta = float(np.mean(deltas))
        median_delta = float(np.median(deltas))
        std_delta = float(np.std(deltas, ddof=1)) if n_meetings > 1 else math.nan
        stderr_delta = std_delta / math.sqrt(n_meetings) if n_meetings > 1 else math.nan

        if n_meetings > 1 and stderr_delta > 0:
            t_stat = mean_delta / stderr_delta
            p_value = float(2.0 * stats.t.sf(abs(t_stat), df=n_meetings - 1))
            inference_status = "meeting_level_inference_available"
        elif n_meetings < 2:
            t_stat = math.nan
            p_value = math.nan
            inference_status = "insufficient_meeting_clusters"
        else:
            t_stat = math.nan
            p_value = math.nan
            inference_status = "zero_between_meeting_variance"

        ci_lower, ci_upper = _cluster_bootstrap_interval(
            deltas,
            samples=bootstrap_samples,
            seed=bootstrap_seed + group_index,
        )
        mean_similarity_full = float(meeting_level["similarity_full"].mean())
        mean_similarity_masked = float(meeting_level["similarity_masked"].mean())
        replicates_per_meeting = meeting_replicate_level.groupby(
            "meeting_date"
        ).size()
        sample_keys = {
            tuple(value) if isinstance(value, list) else (str(value),)
            for value in group["row_key"]
        }
        summaries.append(
            {
                "schema_version": SCHEMA_VERSION,
                "metric": "signed_leave_one_out_alignment_delta",
                "target_mode": str(target_mode),
                "context": str(context),
                "indicator": str(indicator),
                "section_name": str(section_name),
                "n_pairs": int(len(group)),
                "n_sample_keys": int(len(sample_keys)),
                "n_meetings": int(n_meetings),
                "n_meeting_replicates": int(len(meeting_replicate_level)),
                "n_replicates": int(group["replicate_id"].astype(str).nunique()),
                "replicates_per_meeting_min": int(replicates_per_meeting.min()),
                "replicates_per_meeting_max": int(replicates_per_meeting.max()),
                "mean_similarity_full": mean_similarity_full,
                "mean_similarity_masked": mean_similarity_masked,
                "mean_distance_full": 1.0 - mean_similarity_full,
                "mean_distance_masked": 1.0 - mean_similarity_masked,
                "mean_delta": mean_delta,
                "median_delta": median_delta,
                "std_delta_meeting": std_delta,
                "stderr_delta_meeting": stderr_delta,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "t_stat": t_stat,
                "p_value": p_value,
                "inference_status": inference_status,
                "bootstrap_samples": int(bootstrap_samples),
                "bootstrap_seed": int(bootstrap_seed + group_index),
            }
        )

    adjusted = _holm_adjust([float(row["p_value"]) for row in summaries])
    holm_family_size = sum(math.isfinite(float(row["p_value"])) for row in summaries)
    for row, adjusted_p_value in zip(summaries, adjusted, strict=True):
        row["p_value_holm"] = adjusted_p_value
        row["holm_family"] = "all_indicator_section_context_cells_in_run"
        row["holm_family_size"] = holm_family_size
    return summaries


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _companion_path(summary_path: Path, label: str, suffix: str) -> Path:
    stem = summary_path.name
    if summary_path.suffix:
        stem = summary_path.name[: -len(summary_path.suffix)]
    return summary_path.with_name(f"{stem}.{label}{suffix}")


def run_paired_evaluation(
    *,
    input_folder: Path,
    summary_output_file: Path,
    target_mode: str,
    embedding_model_path: str,
    embedding_model_sha256: str | None = None,
    reference_file: Path | None = None,
    reference_key_fields: tuple[str, ...] = DEFAULT_REFERENCE_KEYS,
    reference_text_field: str = "response",
    reference_duplicate_policy: str = "error",
    baseline_indicator: str = DEFAULT_BASELINE_INDICATOR,
    unmatched_policy: str = "error",
    row_output_file: Path | None = None,
    exclusions_file: Path | None = None,
    audit_file: Path | None = None,
    embedding_batch_size: int = 8,
    score_chunk_size: int = 256,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 20260728,
    require_generation_manifest: bool = True,
    scorer: TripletScorer | None = None,
) -> dict:
    """Run discovery, keyed pairing, scoring, clustered summary, and audit."""

    if target_mode not in {"full-output", "actual-minutes"}:
        raise ValueError(f"Unsupported target_mode={target_mode!r}")
    if unmatched_policy not in {"error", "drop"}:
        raise ValueError(f"Unsupported unmatched_policy={unmatched_policy!r}")
    if reference_duplicate_policy not in {"error", "concatenate"}:
        raise ValueError(
            f"Unsupported reference_duplicate_policy={reference_duplicate_policy!r}"
        )
    if target_mode == "actual-minutes" and reference_file is None:
        raise ValueError("reference_file is required for target_mode='actual-minutes'")
    if embedding_batch_size < 1:
        raise ValueError("embedding_batch_size must be positive")
    if score_chunk_size < 1:
        raise ValueError("score_chunk_size must be positive")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    embedding_path = Path(embedding_model_path).expanduser()
    if embedding_path.exists():
        embedding_model_artifact = fingerprint_artifact_path(embedding_path)
        if embedding_model_sha256 is not None:
            expected_embedding_hash = validate_sha256(
                embedding_model_sha256,
                label="embedding_model_sha256",
            )
            if embedding_model_artifact["sha256"] != expected_embedding_hash:
                raise ValueError(
                    "Embedding model fingerprint differs from "
                    "--embedding-model-sha256"
                )
    else:
        if scorer is None:
            raise FileNotFoundError(
                "Canonical evaluation requires a local, fingerprintable embedding "
                f"model artifact: {embedding_path}. Snapshot the model locally "
                "instead of passing a mutable Hub alias."
            )
        declared_hash = validate_sha256(
            embedding_model_sha256,
            label="embedding_model_sha256 for injected scorer",
        )
        embedding_model_artifact = {
            "path": embedding_model_path,
            "kind": "injected_scorer",
            "sha256": declared_hash,
            "file_count": None,
            "total_bytes": None,
            "algorithm": "caller-declared test/custom scorer fingerprint",
        }
    resolved_embedding_hash = str(embedding_model_artifact["sha256"])

    artifacts = discover_generated_artifacts(input_folder)
    generation_manifest, generation_manifest_audit = (
        load_and_validate_generation_manifest(
            input_folder,
            artifacts,
            required=require_generation_manifest,
        )
    )
    artifact_pairs = pair_generated_artifacts(
        artifacts,
        baseline_indicator=baseline_indicator,
    )

    reference_index: dict[tuple[str, ...], str] | None = None
    if target_mode == "actual-minutes":
        assert reference_file is not None
        if not reference_file.is_file():
            raise FileNotFoundError(f"Actual-Minutes reference file does not exist: {reference_file}")
        reference_index = build_reference_index(
            _read_jsonl(reference_file),
            key_fields=reference_key_fields,
            text_field=reference_text_field,
            duplicate_policy=reference_duplicate_policy,
        )

    exclusions: list[dict] = []
    prepared_rows: list[dict] = []
    full_row_cache: dict[Path, list[dict]] = {}
    for full_artifact, masked_artifact in artifact_pairs:
        prepared_rows.extend(
            prepare_paired_rows(
                full_artifact,
                masked_artifact,
                target_mode=target_mode,
                reference_index=reference_index,
                reference_key_fields=reference_key_fields,
                unmatched_policy=unmatched_policy,
                exclusions=exclusions,
                full_row_cache=full_row_cache,
            )
        )
    if not prepared_rows:
        raise ValueError("No full/masked rows remained after keyed pairing")
    if generation_manifest is not None:
        incomplete_pairing = [
            row
            for row in prepared_rows
            if row["pairing_quality"] != "matched_complete_generation_metadata"
        ]
        if incomplete_pairing:
            raise ValueError(
                f"{len(incomplete_pairing)} pairs covered by the canonical generation "
                "manifest lack complete matched generation metadata"
            )

    scorer = scorer or EmbeddingCosineTripletScorer(
        embedding_model_path,
        embedding_batch_size=embedding_batch_size,
    )
    scored_rows = score_prepared_rows(
        prepared_rows,
        scorer=scorer,
        target_mode=target_mode,
        embedding_model_path=embedding_model_path,
        embedding_model_sha256=resolved_embedding_hash,
        score_chunk_size=score_chunk_size,
    )
    summaries = summarise_scored_rows(
        scored_rows,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )

    row_output_file = row_output_file or _companion_path(summary_output_file, "rows", ".jsonl")
    exclusions_file = exclusions_file or _companion_path(
        summary_output_file,
        "exclusions",
        ".jsonl",
    )
    audit_file = audit_file or _companion_path(summary_output_file, "audit", ".json")
    _atomic_write_jsonl(row_output_file, scored_rows)
    _atomic_write_jsonl(summary_output_file, summaries)
    _atomic_write_jsonl(exclusions_file, exclusions)

    audit = {
        "schema_version": SCHEMA_VERSION,
        "target_mode": target_mode,
        "primary_estimand": "delta = similarity_full - similarity_masked",
        "input_folder": str(input_folder),
        "generation_manifest": generation_manifest_audit,
        "generation_manifest_schema": (
            None if generation_manifest is None else generation_manifest.get("schema_version")
        ),
        "reference_file": None if reference_file is None else str(reference_file),
        "reference_file_sha256": (
            None if reference_file is None else _sha256_file(reference_file)
        ),
        "reference_key_fields": list(reference_key_fields),
        "reference_text_field": reference_text_field,
        "reference_duplicate_policy": reference_duplicate_policy,
        "embedding_model_path": embedding_model_path,
        "embedding_model_artifact": embedding_model_artifact,
        "embedding_batch_size": embedding_batch_size,
        "score_chunk_size": score_chunk_size,
        "baseline_indicator": baseline_indicator,
        "unmatched_policy": unmatched_policy,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "holm_family": "all_indicator_section_context_cells_in_run",
        "generated_artifacts": len(artifacts),
        "generated_artifact_inventory": [
            {
                "path": str(artifact.path),
                "indicator": artifact.indicator,
                "context": artifact.context,
                "replicate_id": artifact.replicate_id,
                "generation_seed": artifact.generation_seed,
                "row_count": artifact.row_count,
                "sha256": artifact.sha256,
            }
            for artifact in artifacts
        ],
        "artifact_pairs": len(artifact_pairs),
        "scored_pairs": len(scored_rows),
        "excluded_rows": len(exclusions),
        "exclusion_reason_counts": (
            {}
            if not exclusions
            else (
                pd.Series([row["reason"] for row in exclusions])
                .value_counts()
                .sort_index()
                .to_dict()
            )
        ),
        "summary_groups": len(summaries),
        "row_output_file": str(row_output_file),
        "summary_output_file": str(summary_output_file),
        "exclusions_file": str(exclusions_file),
        "audit_file": str(audit_file),
        "pairing_quality_counts": (
            pd.Series([row["pairing_quality"] for row in scored_rows])
            .value_counts()
            .sort_index()
            .to_dict()
        ),
    }
    _atomic_write_json(audit_file, audit)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score paired full/masked generations with "
            "delta = cos(full,target) - cos(masked,target)."
        )
    )
    parser.add_argument("--input-folder", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True, help="Summary JSONL output.")
    parser.add_argument(
        "--target-mode",
        choices=["full-output", "actual-minutes"],
        required=True,
    )
    parser.add_argument("--reference-file", type=Path)
    parser.add_argument(
        "--reference-key",
        action="append",
        dest="reference_keys",
        help="Reference join key; repeat for composite keys (default: meeting_date, section_name).",
    )
    parser.add_argument("--reference-text-field", default="response")
    parser.add_argument(
        "--reference-duplicate-policy",
        choices=["error", "concatenate"],
        default="error",
    )
    parser.add_argument("--embedding-model-path", required=True)
    parser.add_argument(
        "--embedding-model-sha256",
        help=(
            "Optional expected fingerprint for the local embedding artifact; "
            "evaluation always computes and records the observed fingerprint."
        ),
    )
    parser.add_argument("--baseline-indicator", default=DEFAULT_BASELINE_INDICATOR)
    parser.add_argument(
        "--allow-missing-generation-manifest",
        action="store_true",
        help=(
            "Allow legacy inputs without generation_manifest.json. "
            "Canonical new runs should not use this override."
        ),
    )
    parser.add_argument("--unmatched-policy", choices=["error", "drop"], default="error")
    parser.add_argument("--row-output-file", type=Path)
    parser.add_argument("--exclusions-file", type=Path)
    parser.add_argument("--audit-file", type=Path)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--score-chunk-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    reference_keys = tuple(args.reference_keys or DEFAULT_REFERENCE_KEYS)
    audit = run_paired_evaluation(
        input_folder=args.input_folder,
        summary_output_file=args.output_file,
        target_mode=args.target_mode,
        embedding_model_path=args.embedding_model_path,
        embedding_model_sha256=args.embedding_model_sha256,
        reference_file=args.reference_file,
        reference_key_fields=reference_keys,
        reference_text_field=args.reference_text_field,
        reference_duplicate_policy=args.reference_duplicate_policy,
        baseline_indicator=args.baseline_indicator,
        unmatched_policy=args.unmatched_policy,
        row_output_file=args.row_output_file,
        exclusions_file=args.exclusions_file,
        audit_file=args.audit_file,
        embedding_batch_size=args.embedding_batch_size,
        score_chunk_size=args.score_chunk_size,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        require_generation_manifest=not args.allow_missing_generation_manifest,
    )
    print("✅ Paired leave-one-out evaluation complete")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
