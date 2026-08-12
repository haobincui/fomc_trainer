"""Seal strict smoke gates and population releases for the legacy-six LOO run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.generation.finalize_loo_intervention_shard import (
    build_intervention_shard_manifest,
)
from jobs.generation.mask_generation import (
    DELETION_MASKING_STRATEGY,
    NEUTRAL_MASKING_STRATEGY,
    build_intervention_manifest,
    validate_prompt_manifest_binding,
)
from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.validator.loo_experiment_scope import (
    EXPECTED_POPULATIONS,
    EXPECTED_SECTIONS,
    load_and_validate_experiment_scope,
)
from open_r1.validator.loo_generation_spec import (
    DEFAULT_SUCCESS_FINISH_REASONS,
    seal_manifest,
    validate_manifest_integrity,
)
from open_r1.validator.neutral_intervention import (
    validate_neutral_intervention_manifest,
)


SCOPED_RELEASE_SCHEMA_VERSION = "loo-scoped-generation-release-v1"
SMOKE_GATE_SCHEMA_VERSION = "loo-scoped-smoke-gate-v1"
GENERATION_SCHEMA_VERSION = "loo-generation-v4"
ROW_SEED_POLICY = "sample-id-sha256-v1"
RUNS = {
    "deletion_primary": ("indicator_block_deletion", "primary", 1),
    "neutral_primary": (
        "indicator_block_neutral_replacement",
        "primary",
        1,
    ),
    "deletion_stochastic": ("indicator_block_deletion", "stochastic", 5),
    "neutral_stochastic": (
        "indicator_block_neutral_replacement",
        "stochastic",
        5,
    ),
}


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
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
                    f"Invalid {label} JSONL {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{label} row {line_number} must be an object")
            rows.append(row)
    return rows


def _resolve_artifact_path(value: object, *, manifest_path: Path) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _verify_fingerprint(record: object, *, label: str) -> None:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} fingerprint must be an object")
    path = Path(str(record.get("path") or "")).expanduser().resolve()
    observed = fingerprint_artifact_path(path)
    identity_fields = (
        "sha256",
        "kind",
        "file_count",
        "total_bytes",
        "algorithm",
    )
    mismatches = {
        key: {"expected": record.get(key), "observed": observed.get(key)}
        for key in identity_fields
        if record.get(key) != observed.get(key)
    }
    if mismatches:
        raise ValueError(f"{label} fingerprint mismatch: {mismatches}")


def _require_sha256(value: object, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return digest


def _validate_analysis_projection_binding(
    *,
    analysis_path: Path,
    projection_path: Path,
) -> dict[str, str]:
    """Bind the model that actually generated analysis, not an unused spec entry."""

    analysis = _read_json(analysis_path, label="indicator analysis manifest")
    projection = _read_json(projection_path, label="analysis projection manifest")
    if (
        analysis.get("schema_version") != "indicator-analysis-generation-v2"
        or analysis.get("status") != "complete"
        or analysis.get("generation_only") is not True
        or analysis.get("training_performed") is not False
    ):
        raise ValueError("Indicator analysis manifest is not a complete generation artifact")
    inventory = analysis.get("inventory")
    if not isinstance(inventory, Mapping) or (
        inventory.get("meeting_count"),
        inventory.get("indicator_count"),
        inventory.get("row_count"),
    ) != (13, 26, 338):
        raise ValueError("Indicator analysis does not cover the frozen 13x26 matrix")
    inputs = analysis.get("inputs")
    output = analysis.get("output")
    if not isinstance(inputs, Mapping) or not isinstance(output, Mapping):
        raise ValueError("Indicator analysis lacks provenance inputs/output")
    model = inputs.get("model")
    tokenizer = inputs.get("tokenizer")
    if not isinstance(model, Mapping) or not isinstance(tokenizer, Mapping):
        raise ValueError("Indicator analysis lacks actual model/tokenizer provenance")
    model_sha = _require_sha256(
        model.get("sha256"), label="indicator-analysis model sha256"
    )
    tokenizer_sha = _require_sha256(
        tokenizer.get("sha256"), label="indicator-analysis tokenizer sha256"
    )
    output_sha = _require_sha256(
        output.get("sha256"), label="indicator-analysis output sha256"
    )

    if (
        projection.get("schema_version") != "indicator-analysis-projection-v1"
        or projection.get("status") != "complete"
        or projection.get("generation_performed") is not False
        or projection.get("summarization_performed") is not False
        or projection.get("truncation_performed") is not False
    ):
        raise ValueError("Analysis projection is not a deterministic extraction artifact")
    projection_input = projection.get("input")
    projection_analysis = projection.get("input_analysis_manifest")
    if not isinstance(projection_input, Mapping) or not isinstance(
        projection_analysis, Mapping
    ):
        raise ValueError("Analysis projection lacks source bindings")
    if (
        projection_input.get("sha256") != output_sha
        or projection_analysis.get("sha256") != sha256_file(analysis_path)
        or projection_analysis.get("schema_version")
        != analysis.get("schema_version")
    ):
        raise ValueError("Analysis projection is not bound to the supplied analysis")
    return {
        "analysis_model_sha256": model_sha,
        "analysis_tokenizer_sha256": tokenizer_sha,
        "analysis_output_sha256": output_sha,
        "analysis_projection_sha256": _require_sha256(
            projection.get("output", {}).get("sha256")
            if isinstance(projection.get("output"), Mapping)
            else None,
            label="analysis projection output sha256",
        ),
    }


def load_and_validate_scoped_release(
    release_file: str | Path,
    *,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    """Validate a sealed scoped population release for safe resume/reuse."""

    path = Path(release_file).expanduser().resolve()
    manifest = _read_json(path, label="scoped generation release")
    if manifest.get("schema_version") != SCOPED_RELEASE_SCHEMA_VERSION:
        raise ValueError("Unsupported scoped generation release schema")
    validate_manifest_integrity(manifest)
    if (
        manifest.get("status") != "complete"
        or manifest.get("scoped_population_release") is not True
        or manifest.get("standalone_full_roster_canonical_release") is not False
        or manifest.get("complete_matrix") is not True
    ):
        raise ValueError("Scoped generation release is not complete")
    if verify_artifacts:
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("Scoped release lacks artifacts")
        for name, record in artifacts.items():
            _verify_fingerprint(record, label=f"artifacts.{name}")
        scope_path = Path(artifacts["experiment_config"]["path"]).resolve()
        roster_path = Path(artifacts["indicator_roster"]["path"]).resolve()
        spec_path = Path(artifacts["generation_spec"]["path"]).resolve()
        prompt_path = Path(artifacts["prompt_manifest"]["path"]).resolve()
        analysis_path = Path(artifacts["analysis_manifest"]["path"]).resolve()
        projection_path = Path(artifacts["projection_manifest"]["path"]).resolve()
        scope = load_and_validate_experiment_scope(
            scope_path, roster_file=roster_path
        )
        spec = _read_json(spec_path, label="generation spec")
        prompt = _read_json(prompt_path, label="prompt manifest")
        intervention_validation = _validate_protocol_bindings(
            scope=scope,
            scope_path=scope_path,
            spec=spec,
            prompt=prompt,
            prompt_path=prompt_path,
            roster_path=roster_path,
        )
        analysis_validation = _validate_analysis_projection_binding(
            analysis_path=analysis_path,
            projection_path=projection_path,
        )
        generation_manifest_path = Path(
            artifacts["deletion_primary"]["path"]
        ).resolve()
        run_root = generation_manifest_path.parents[2]
        validation = _validate_generation_runs(
            run_root=run_root,
            scope=scope,
            scope_path=scope_path,
            spec_path=spec_path,
            prompt_path=prompt_path,
            release_kind=str(manifest["release_kind"]),
            intervention_validation=intervention_validation,
        )
        if (
            manifest.get("prompt_template_sha256")
            != intervention_validation["prompt_template_sha256"]
            or manifest.get("minutes_model_sha256")
            != validation["minutes_artifacts"]["model_sha256"]
            or manifest.get("minutes_tokenizer_sha256")
            != validation["minutes_artifacts"]["tokenizer_sha256"]
            or manifest.get("matrix", {}).get("total_generation_attempts")
            != validation["total_generation_attempts"]
            or manifest.get("analysis_model_sha256")
            != analysis_validation["analysis_model_sha256"]
            or manifest.get("analysis_tokenizer_sha256")
            != analysis_validation["analysis_tokenizer_sha256"]
        ):
            raise ValueError("Scoped release does not match revalidated raw outputs")
        pilot_record = artifacts.get("pilot_scoped_release")
        if pilot_record is not None:
            load_and_validate_scoped_release(
                Path(str(pilot_record["path"])), verify_artifacts=True
            )
    return manifest


def load_and_validate_smoke_gate(
    gate_file: str | Path,
    *,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    """Validate a sealed, non-inferential smoke gate for safe resume."""

    path = Path(gate_file).expanduser().resolve()
    manifest = _read_json(path, label="scoped smoke gate")
    if manifest.get("schema_version") != SMOKE_GATE_SCHEMA_VERSION:
        raise ValueError("Unsupported scoped smoke-gate schema")
    validate_manifest_integrity(manifest)
    gate = manifest.get("gate")
    if (
        manifest.get("status") != "passed"
        or manifest.get("inferential_use_allowed") is not False
        or not isinstance(gate, Mapping)
        or not gate
        or any(value is not True for value in gate.values())
    ):
        raise ValueError("Scoped smoke gate did not pass every requirement")
    if verify_artifacts:
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("Scoped smoke gate lacks artifacts")
        for name, record in artifacts.items():
            _verify_fingerprint(record, label=f"artifacts.{name}")
        scope_path = Path(artifacts["experiment_config"]["path"]).resolve()
        roster_path = Path(artifacts["indicator_roster"]["path"]).resolve()
        spec_path = Path(artifacts["generation_spec"]["path"]).resolve()
        prompt_path = Path(artifacts["prompt_manifest"]["path"]).resolve()
        analysis_path = Path(artifacts["analysis_manifest"]["path"]).resolve()
        projection_path = Path(artifacts["projection_manifest"]["path"]).resolve()
        scope = load_and_validate_experiment_scope(
            scope_path, roster_file=roster_path
        )
        spec = _read_json(spec_path, label="generation spec")
        prompt = _read_json(prompt_path, label="prompt manifest")
        intervention_validation = _validate_protocol_bindings(
            scope=scope,
            scope_path=scope_path,
            spec=spec,
            prompt=prompt,
            prompt_path=prompt_path,
            roster_path=roster_path,
        )
        analysis_validation = _validate_analysis_projection_binding(
            analysis_path=analysis_path,
            projection_path=projection_path,
        )
        generation_manifest_path = Path(
            artifacts["deletion_primary"]["path"]
        ).resolve()
        run_root = generation_manifest_path.parents[2]
        validation = _validate_generation_runs(
            run_root=run_root,
            scope=scope,
            scope_path=scope_path,
            spec_path=spec_path,
            prompt_path=prompt_path,
            release_kind="smoke",
            intervention_validation=intervention_validation,
            smoke_sample_ids=manifest.get("sample_ids", []),
        )
        if manifest.get("matrix", {}).get("total_generation_attempts") != validation[
            "total_generation_attempts"
        ]:
            raise ValueError("Smoke gate does not match revalidated raw outputs")
        if (
            manifest.get("analysis_model_sha256")
            != analysis_validation["analysis_model_sha256"]
            or manifest.get("analysis_tokenizer_sha256")
            != analysis_validation["analysis_tokenizer_sha256"]
        ):
            raise ValueError("Smoke gate analysis provenance changed")
    return manifest


def _validate_protocol_bindings(
    *,
    scope: Mapping[str, Any],
    scope_path: Path,
    spec: Mapping[str, Any],
    prompt: Mapping[str, Any],
    prompt_path: Path,
    roster_path: Path,
) -> dict[str, Any]:
    prompt_policy = scope["minutes_system_prompt"]
    generation_config = spec.get("generation_config")
    if not isinstance(generation_config, Mapping):
        raise ValueError("Generation spec lacks generation_config")
    if generation_config.get("minutes_system_prompt") != prompt_policy:
        raise ValueError(
            "Generation spec Minutes system prompt differs from experiment scope"
        )
    sources = spec.get("frozen_artifacts", {}).get("sources", {})
    source_hashes = {
        str(record.get("sha256") or "")
        for record in sources.values()
        if isinstance(record, Mapping)
    }
    if sha256_file(scope_path) not in source_hashes:
        raise ValueError("Generation spec does not freeze the experiment scope")

    prompt_binding = prompt.get("experiment_config")
    if (
        not isinstance(prompt_binding, Mapping)
        or prompt_binding.get("sha256") != sha256_file(scope_path)
    ):
        raise ValueError("Prompt manifest experiment-config binding mismatch")
    context_budget = prompt.get("context_budget")
    if not isinstance(context_budget, Mapping):
        raise ValueError("Prompt manifest lacks context_budget")
    expected_budget = {
        "system_prompt_sha256": prompt_policy["sha256"],
        "requested_max_output_tokens": prompt_policy[
            "requested_max_output_tokens"
        ],
        "max_new_tokens": prompt_policy["hard_max_new_tokens"],
        "max_model_len": prompt_policy["max_model_len"],
        "input_truncation": "forbidden",
    }
    mismatches = {
        key: {"expected": value, "observed": context_budget.get(key)}
        for key, value in expected_budget.items()
        if context_budget.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Prompt context-budget binding mismatch: {mismatches}")
    return _validate_prompt_interventions(
        prompt_path=prompt_path,
        prompt=prompt,
        roster_path=roster_path,
        spec=spec,
    )


def _validate_prompt_interventions(
    *,
    prompt_path: Path,
    prompt: Mapping[str, Any],
    roster_path: Path,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute deletion and neutral proofs from the frozen prompt copies."""

    deletion_binding = validate_prompt_manifest_binding(
        prompt_manifest_file=prompt_path,
        input_dir=prompt_path.parent / "exact_delete",
        masking_strategy=DELETION_MASKING_STRATEGY,
    )
    neutral_binding = validate_prompt_manifest_binding(
        prompt_manifest_file=prompt_path,
        input_dir=prompt_path.parent / "neutral",
        masking_strategy=NEUTRAL_MASKING_STRATEGY,
    )
    deletion_path = Path(
        deletion_binding["references"]["intervention_manifest"]["path"]
    )
    run_roster_path = Path(
        deletion_binding["references"]["run_intervention_roster"]["path"]
    )
    declared_deletion = _read_json(
        deletion_path, label="deletion intervention manifest"
    )
    recomputed_deletion = build_intervention_manifest(
        input_dir=(prompt_path.parent / "exact_delete").resolve(),
        input_prompt_files=sorted((prompt_path.parent / "exact_delete").glob("*.jsonl")),
        roster_file=run_roster_path,
    )
    if declared_deletion != recomputed_deletion:
        raise ValueError(
            "Deletion intervention manifest differs from recomputed exact proofs"
        )

    tokenizer_record = (
        spec.get("frozen_artifacts", {})
        .get("tokenizers", {})
        .get("minutes_tokenizer")
    )
    if not isinstance(tokenizer_record, Mapping):
        raise ValueError("Generation spec lacks Minutes tokenizer binding")
    tokenizer_path = Path(str(tokenizer_record.get("path") or "")).expanduser().resolve()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    neutral_path = Path(
        neutral_binding["references"]["neutral_intervention_manifest"]["path"]
    )
    normalized_neutral = validate_neutral_intervention_manifest(
        manifest=neutral_path,
        input_dir=(prompt_path.parent / "neutral").resolve(),
        roster_file=run_roster_path,
        tokenizer=tokenizer,
        tokenizer_artifact_sha256=str(tokenizer_record.get("sha256") or ""),
    )
    prompt_template_sha256 = prompt.get("prompt_template", {}).get("sha256")
    if (
        not isinstance(prompt_template_sha256, str)
        or len(prompt_template_sha256) != 64
    ):
        raise ValueError("Prompt manifest lacks a frozen user-prompt template hash")
    return {
        "deletion_manifest_sha256": sha256_file(deletion_path),
        "deletion_proof_count": len(recomputed_deletion["interventions"]),
        "neutral_manifest_sha256": sha256_file(neutral_path),
        "neutral_proof_count": len(normalized_neutral["replacements"]),
        "deletion_runtime_manifest_sha256": hashlib.sha256(
            (
                json.dumps(
                    recomputed_deletion,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest(),
        "neutral_runtime_manifest_sha256": hashlib.sha256(
            (
                json.dumps(
                    normalized_neutral,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest(),
        "prompt_template_sha256": prompt_template_sha256,
        "exact_deletion_validated": True,
        "neutral_replacement_validated": True,
    }


def _validate_generation_runs(
    *,
    run_root: Path,
    scope: Mapping[str, Any],
    scope_path: Path,
    spec_path: Path,
    prompt_path: Path,
    release_kind: str,
    intervention_validation: Mapping[str, Any],
    smoke_sample_ids: Sequence[str] = (),
) -> dict[str, Any]:
    selected = list(scope["intervention_indicators"])
    baseline = str(scope["baseline_indicator"])
    universe = {baseline, *selected}
    requested_limit = int(
        scope["minutes_system_prompt"]["requested_max_output_tokens"]
    )
    hard_limit = int(scope["minutes_system_prompt"]["hard_max_new_tokens"])
    model_limit = int(scope["minutes_system_prompt"]["max_model_len"])
    scope_sha = sha256_file(scope_path)
    spec_sha = sha256_file(spec_path)
    prompt_sha = sha256_file(prompt_path)
    expected_population_rows = 13 * len(scope["section_names"])
    expected_rows = 3 if release_kind == "smoke" else expected_population_rows
    expected_sample_ids = set(smoke_sample_ids)
    if release_kind == "smoke":
        if len(smoke_sample_ids) != 3 or len(expected_sample_ids) != 3:
            raise ValueError("Smoke gate requires exactly three unique sample IDs")
    elif smoke_sample_ids:
        raise ValueError("Population releases cannot provide smoke sample IDs")

    generation_manifests: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    row_matrix_by_run: dict[
        str, dict[tuple[str, str, str], dict[str, Any]]
    ] = {}
    observed_population_sample_ids: set[str] | None = None
    observed_sections: Counter[str] = Counter()
    frozen_artifacts = spec_path and _read_json(
        spec_path, label="generation spec"
    ).get("frozen_artifacts", {})
    frozen_models = (
        frozen_artifacts.get("models", {})
        if isinstance(frozen_artifacts, Mapping)
        else {}
    )
    frozen_tokenizers = (
        frozen_artifacts.get("tokenizers", {})
        if isinstance(frozen_artifacts, Mapping)
        else {}
    )
    expected_model_sha = str(
        frozen_models.get("minutes_model", {}).get("sha256") or ""
    )
    expected_tokenizer_sha = str(
        frozen_tokenizers.get("minutes_tokenizer", {}).get("sha256") or ""
    )
    if len(expected_model_sha) != 64 or len(expected_tokenizer_sha) != 64:
        raise ValueError("Generation spec lacks frozen Minutes model/tokenizer hashes")

    for run_name, (strategy, regime, replicate_count) in RUNS.items():
        manifest_path = (
            run_root / "generations" / run_name / "generation_manifest.json"
        )
        manifest = _read_json(manifest_path, label=f"{run_name} manifest")
        generation_manifests[run_name] = manifest
        decoding = scope["decoding"][regime]
        expected_fields = {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "status": "complete",
            "generation_attempts_complete": True,
            "scorable_population_complete": True,
            "masking_strategy": strategy,
            "simulation_step": replicate_count,
            "temperature": decoding["temperature"],
            "top_p": decoding["top_p"],
            "replicate_seeds": decoding["replicate_seeds"],
            "max_new_tokens": hard_limit,
            "max_model_len": model_limit,
            "seed_policy": ROW_SEED_POLICY,
            "system_prompt_sha256": scope["minutes_system_prompt"]["sha256"],
        }
        mismatches = {
            key: {"expected": value, "observed": manifest.get(key)}
            for key, value in expected_fields.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"{run_name} protocol mismatch: {mismatches}")
        if manifest.get("completion_validation", {}).get("excluded_row_count") != 0:
            raise ValueError(f"{run_name} contains generation exclusions")
        if manifest.get("completion_validation", {}).get("token_limit_policy") != "error":
            raise ValueError(f"{run_name} did not use strict token-limit policy")
        if manifest.get("generation_spec", {}).get("file_sha256") != spec_sha:
            raise ValueError(f"{run_name} generation-spec binding mismatch")
        if manifest.get("prompt_manifest", {}).get("sha256") != prompt_sha:
            raise ValueError(f"{run_name} prompt-manifest binding mismatch")
        runtime_hash_key = (
            "deletion_runtime_manifest_sha256"
            if strategy == "indicator_block_deletion"
            else "neutral_runtime_manifest_sha256"
        )
        expected_intervention_sha = intervention_validation.get(
            runtime_hash_key
        )
        runtime_intervention_path = manifest_path.parent / "intervention_manifest.json"
        if (
            not isinstance(expected_intervention_sha, str)
            or manifest.get("intervention_manifest", {}).get("sha256")
            != expected_intervention_sha
            or not runtime_intervention_path.is_file()
            or sha256_file(runtime_intervention_path) != expected_intervention_sha
        ):
            raise ValueError(
                f"{run_name} intervention-manifest copy/hash binding mismatch"
            )
        if (
            manifest.get("experiment_config", {}).get("sha256") != scope_sha
            or manifest.get("minutes_system_prompt")
            != scope["minutes_system_prompt"]
        ):
            raise ValueError(f"{run_name} scoped Minutes prompt binding mismatch")
        artifact_hash_expectations = {
            "model_artifact": expected_model_sha,
            "model_artifact_after_generation": expected_model_sha,
            "tokenizer_artifact": expected_tokenizer_sha,
            "tokenizer_artifact_after_generation": expected_tokenizer_sha,
        }
        artifact_hash_mismatches = {
            key: {"expected": value, "observed": manifest.get(key, {}).get("sha256")}
            for key, value in artifact_hash_expectations.items()
            if not isinstance(manifest.get(key), Mapping)
            or manifest.get(key, {}).get("sha256") != value
        }
        if artifact_hash_mismatches:
            raise ValueError(
                f"{run_name} frozen Minutes artifact mismatch: "
                f"{artifact_hash_mismatches}"
            )
        prompt_experiment = (
            manifest.get("prompt_manifest", {})
            .get("references", {})
            .get("experiment_config", {})
        )
        if prompt_experiment and prompt_experiment.get("sha256") != scope_sha:
            raise ValueError(f"{run_name} experiment-config binding mismatch")

        sample_selection = manifest.get("sample_selection")
        if not isinstance(sample_selection, Mapping):
            raise ValueError(f"{run_name} lacks sample_selection metadata")
        if release_kind == "smoke":
            expected_selection = {
                "policy": "explicit-sample-id-filter-v1",
                "mode": "filtered_smoke",
                "requested_sample_ids": list(smoke_sample_ids),
                "selected_sample_ids": list(smoke_sample_ids),
                "selected_row_count_per_artifact": 3,
                "full_source_row_count_per_artifact": expected_population_rows,
                "population_release_allowed": False,
            }
        else:
            expected_selection = {
                "policy": "full-population-v1",
                "mode": "full_population",
                "requested_sample_ids": [],
                "selected_row_count_per_artifact": expected_population_rows,
                "full_source_row_count_per_artifact": expected_population_rows,
                "population_release_allowed": True,
            }
        selection_mismatches = {
            key: {"expected": value, "observed": sample_selection.get(key)}
            for key, value in expected_selection.items()
            if sample_selection.get(key) != value
        }
        if selection_mismatches:
            raise ValueError(
                f"{run_name} sample selection mismatch: {selection_mismatches}"
            )
        if release_kind != "smoke" and sample_selection.get(
            "selected_sample_ids"
        ) not in (None, []):
            selected_ids = sample_selection.get("selected_sample_ids")
            if not isinstance(selected_ids, list) or len(selected_ids) != expected_rows:
                raise ValueError(
                    f"{run_name} full-population selected sample inventory is invalid"
                )

        scope_record = manifest.get("intervention_scope")
        if not isinstance(scope_record, Mapping):
            raise ValueError(f"{run_name} lacks intervention_scope")
        if (
            scope_record.get("baseline_indicator") != baseline
            or scope_record.get("full_roster_indicators")
            != scope["full_context_indicators"]
            or scope_record.get("generated_intervention_indicators") != selected
            or scope_record.get("baseline_contains_full_roster") is not True
        ):
            raise ValueError(f"{run_name} intervention scope mismatch")

        artifacts = manifest.get("expected_artifacts")
        if not isinstance(artifacts, list) or len(artifacts) != len(universe) * replicate_count:
            raise ValueError(f"{run_name} output artifact matrix is incomplete")
        artifact_index: dict[tuple[str, str], list[dict[str, Any]]] = {}
        run_row_matrix: dict[tuple[str, str, str], dict[str, Any]] = {}
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                raise ValueError(f"{run_name} contains an invalid output artifact")
            indicator = str(artifact.get("indicator") or "")
            replicate_id = str(artifact.get("replicate_id") or "")
            if indicator not in universe:
                raise ValueError(f"{run_name} contains an unexpected indicator")
            key = (indicator, replicate_id)
            if key in artifact_index:
                raise ValueError(f"{run_name} contains a duplicate output artifact")
            output_path = _resolve_artifact_path(
                artifact.get("path"), manifest_path=manifest_path
            )
            if sha256_file(output_path) != artifact.get("output_sha256"):
                raise ValueError(f"{run_name} output artifact hash mismatch")
            rows = _read_jsonl(output_path, label=f"{run_name} output")
            if len(rows) != expected_rows:
                raise ValueError(
                    f"{run_name} {indicator}/{replicate_id} has {len(rows)} rows; "
                    f"expected {expected_rows}"
                )
            artifact_index[key] = rows
            for count_field in (
                "attempted_row_count",
                "scorable_row_count",
            ):
                if artifact.get(count_field) != expected_rows:
                    raise ValueError(
                        f"{run_name} artifact {count_field} is incomplete"
                    )
            if artifact.get("excluded_row_count") != 0:
                raise ValueError(f"{run_name} artifact contains exclusions")

            row_ids: set[str] = set()
            for row in rows:
                sample_id = str(row.get("sample_id") or "")
                generated = str(row.get("generated") or "")
                section_name = str(row.get("section_name") or "")
                if not sample_id or sample_id in row_ids:
                    raise ValueError(f"{run_name} has missing/duplicate sample IDs")
                row_ids.add(sample_id)
                run_row_matrix[(indicator, replicate_id, sample_id)] = row
                if not generated.strip():
                    raise ValueError(f"{run_name} contains an empty output")
                if "<think" in generated.lower() or "</think" in generated.lower():
                    raise ValueError(f"{run_name} output reveals a think wrapper")
                output_tokens = row.get("output_token_count")
                if (
                    isinstance(output_tokens, bool)
                    or not isinstance(output_tokens, int)
                    or not 1 <= output_tokens <= requested_limit
                ):
                    raise ValueError(
                        f"{run_name} output exceeds the requested 4096-token limit"
                    )
                if row.get("input_was_truncated") is not False:
                    raise ValueError(f"{run_name} contains truncated input")
                finish_reason = str(row.get("generation_finish_reason") or "").lower()
                if finish_reason not in DEFAULT_SUCCESS_FINISH_REASONS:
                    raise ValueError(f"{run_name} contains a non-normal finish")
                if row.get("generation_validation_status") != "passed":
                    raise ValueError(f"{run_name} contains a non-passing row")
                if row.get("generation_system_prompt_sha256") != scope[
                    "minutes_system_prompt"
                ]["sha256"]:
                    raise ValueError(f"{run_name} row system-prompt hash mismatch")
                if section_name not in scope["section_names"]:
                    raise ValueError(f"{run_name} contains an unknown section")
            if release_kind == "smoke" and row_ids != expected_sample_ids:
                raise ValueError(f"{run_name} smoke sample inventory mismatch")
            if observed_population_sample_ids is None:
                observed_population_sample_ids = row_ids
                observed_sections.update(
                    str(row["section_name"]) for row in rows
                )
            elif row_ids != observed_population_sample_ids:
                raise ValueError(f"{run_name} artifact sample inventory mismatch")

        expected_artifact_keys = {
            (indicator, str(replicate_id))
            for indicator in universe
            for replicate_id in range(replicate_count)
        }
        if set(artifact_index) != expected_artifact_keys:
            raise ValueError(f"{run_name} artifact keys are incomplete")

        for replicate_id in range(replicate_count):
            baseline_rows = {
                str(row["sample_id"]): row
                for row in artifact_index[(baseline, str(replicate_id))]
            }
            for indicator in selected:
                intervention_rows = {
                    str(row["sample_id"]): row
                    for row in artifact_index[(indicator, str(replicate_id))]
                }
                if set(intervention_rows) != set(baseline_rows):
                    raise ValueError(f"{run_name} LOO pairing matrix is incomplete")
                for sample_id, full_row in baseline_rows.items():
                    intervention_row = intervention_rows[sample_id]
                    if (
                        full_row.get("generation_seed")
                        != intervention_row.get("generation_seed")
                        or full_row.get("generation_seed_policy") != ROW_SEED_POLICY
                        or intervention_row.get("generation_seed_policy")
                        != ROW_SEED_POLICY
                    ):
                        raise ValueError(
                            f"{run_name} common-random-number seed mismatch"
                        )
        row_matrix_by_run[run_name] = run_row_matrix

        pair_count = len(selected) * expected_rows * replicate_count
        summaries[run_name] = {
            "artifact_count": len(artifacts),
            "attempted_row_count": len(artifacts) * expected_rows,
            "scored_pair_count": pair_count,
            "replicate_count": replicate_count,
            "sample_count_per_artifact": expected_rows,
            "generation_manifest": fingerprint_artifact_path(manifest_path),
        }

    for regime in ("primary", "stochastic"):
        deletion_rows = row_matrix_by_run[f"deletion_{regime}"]
        neutral_rows = row_matrix_by_run[f"neutral_{regime}"]
        if set(deletion_rows) != set(neutral_rows):
            raise ValueError(f"{regime} deletion/neutral row matrices differ")
        for key, deletion_row in deletion_rows.items():
            neutral_row = neutral_rows[key]
            if deletion_row.get("generation_seed") != neutral_row.get(
                "generation_seed"
            ):
                raise ValueError(
                    f"{regime} deletion/neutral common-random-number seed mismatch"
                )
            indicator, _, _ = key
            if indicator == baseline and (
                deletion_row.get("prompt_sha256")
                != neutral_row.get("prompt_sha256")
                or deletion_row.get("generated_sha256")
                != neutral_row.get("generated_sha256")
                or deletion_row.get("generated") != neutral_row.get("generated")
            ):
                raise ValueError(
                    f"{regime} deletion/neutral full-baseline outputs differ"
                )

    assert observed_population_sample_ids is not None
    if release_kind == "smoke":
        section_counts = Counter(
            sample_id.split("::", 1)[1]
            for sample_id in observed_population_sample_ids
            if "::" in sample_id
        )
        if section_counts != Counter({section: 1 for section in EXPECTED_SECTIONS}):
            raise ValueError("Smoke sample IDs must select one row per section")
    else:
        section_counts = Counter(
            sample_id.split("::", 1)[1]
            for sample_id in observed_population_sample_ids
            if "::" in sample_id
        )
        if section_counts != Counter({section: 13 for section in EXPECTED_SECTIONS}):
            raise ValueError("Population release does not cover 13 rows per section")

    return {
        "release_kind": release_kind,
        "complete_matrix": True,
        "sample_ids": sorted(observed_population_sample_ids),
        "section_counts": dict(sorted(section_counts.items())),
        "run_summaries": summaries,
        "total_generation_attempts": sum(
            summary["attempted_row_count"] for summary in summaries.values()
        ),
        "total_scored_pairs": sum(
            summary["scored_pair_count"] for summary in summaries.values()
        ),
        "generation_manifests": generation_manifests,
        "minutes_artifacts": {
            "model_sha256": expected_model_sha,
            "tokenizer_sha256": expected_tokenizer_sha,
        },
    }


def build_scoped_release_manifest(
    *,
    run_root: str | Path,
    generation_spec_file: str | Path,
    prompt_manifest_file: str | Path,
    analysis_manifest_file: str | Path,
    projection_manifest_file: str | Path,
    experiment_config_file: str | Path,
    roster_file: str | Path,
    release_kind: str,
    pilot_release_file: str | Path | None = None,
    smoke_sample_ids: Sequence[str] = (),
) -> dict[str, Any]:
    if release_kind not in {"smoke", "pilot", "formal"}:
        raise ValueError("release_kind must be smoke, pilot, or formal")
    root = Path(run_root).expanduser().resolve()
    spec_path = Path(generation_spec_file).expanduser().resolve()
    prompt_path = Path(prompt_manifest_file).expanduser().resolve()
    analysis_path = Path(analysis_manifest_file).expanduser().resolve()
    projection_path = Path(projection_manifest_file).expanduser().resolve()
    scope_path = Path(experiment_config_file).expanduser().resolve()
    roster_path = Path(roster_file).expanduser().resolve()
    scope = load_and_validate_experiment_scope(scope_path, roster_file=roster_path)
    spec = _read_json(spec_path, label="generation spec")
    prompt = _read_json(prompt_path, label="prompt manifest")
    analysis_validation = _validate_analysis_projection_binding(
        analysis_path=analysis_path,
        projection_path=projection_path,
    )
    expected_population = (
        "pilot_eval_13" if release_kind == "smoke" else f"{release_kind}_eval_13"
    )
    if release_kind == "formal":
        expected_population = "formal_test_13"
    if spec.get("population_id") != expected_population:
        raise ValueError(
            f"Generation spec population mismatch: {spec.get('population_id')!r}"
        )
    expected_phase = "pilot" if release_kind == "smoke" else release_kind
    if spec.get("phase") != expected_phase:
        raise ValueError("Generation spec phase mismatch")
    intervention_validation = _validate_protocol_bindings(
        scope=scope,
        scope_path=scope_path,
        spec=spec,
        prompt=prompt,
        prompt_path=prompt_path,
        roster_path=roster_path,
    )
    validation = _validate_generation_runs(
        run_root=root,
        scope=scope,
        scope_path=scope_path,
        spec_path=spec_path,
        prompt_path=prompt_path,
        release_kind=release_kind,
        intervention_validation=intervention_validation,
        smoke_sample_ids=smoke_sample_ids,
    )

    if release_kind == "smoke":
        artifact_records = {
            name: summary["generation_manifest"]
            for name, summary in validation["run_summaries"].items()
        }
        payload = {
            "schema_version": SMOKE_GATE_SCHEMA_VERSION,
            "status": "passed",
            "inferential_use_allowed": False,
            "experiment_id": scope["experiment_id"],
            "population_id": "pilot_eval_13",
            "sample_ids": validation["sample_ids"],
            "gate": {
                "complete_matrix": True,
                "zero_input_truncation": True,
                "zero_length_finishes": True,
                "zero_empty_outputs": True,
                "zero_generation_exclusions": True,
                "all_outputs_at_most_4096_tokens": True,
                "full_context_baseline": True,
                "paired_row_seeds": True,
                "exact_deletion_validated": intervention_validation[
                    "exact_deletion_validated"
                ],
                "neutral_replacement_validated": intervention_validation[
                    "neutral_replacement_validated"
                ],
                "prompt_model_tokenizer_hashes_validated": True,
            },
            "matrix": {
                "total_generation_attempts": validation[
                    "total_generation_attempts"
                ],
                "total_scored_pairs": validation["total_scored_pairs"],
                "generation_runs": validation["run_summaries"],
            },
            "intervention_validation": intervention_validation,
            "analysis_model_sha256": analysis_validation[
                "analysis_model_sha256"
            ],
            "analysis_tokenizer_sha256": analysis_validation[
                "analysis_tokenizer_sha256"
            ],
            "artifacts": {
                "experiment_config": fingerprint_artifact_path(scope_path),
                "indicator_roster": fingerprint_artifact_path(roster_path),
                "generation_spec": fingerprint_artifact_path(spec_path),
                "prompt_manifest": fingerprint_artifact_path(prompt_path),
                "analysis_manifest": fingerprint_artifact_path(analysis_path),
                "projection_manifest": fingerprint_artifact_path(projection_path),
                **artifact_records,
            },
        }
        return seal_manifest(payload)

    shard = build_intervention_shard_manifest(
        run_root=root,
        generation_spec_file=spec_path,
        prompt_manifest_file=prompt_path,
        analysis_manifest_file=analysis_path,
        projection_manifest_file=projection_path,
        selected_indicators=scope["intervention_indicators"],
    )
    if (
        shard.get("status") != "partial_complete"
        or shard.get("canonical_compatible") is not True
        or shard.get("scorable_population_complete") is not True
        or shard.get("completion_validation", {}).get("excluded_row_count") != 0
    ):
        raise ValueError("Underlying intervention shard is not strict and complete")

    pilot_prerequisite: dict[str, Any] | None = None
    if release_kind == "formal":
        if pilot_release_file is None:
            raise ValueError("Formal release requires a scoped pilot release")
        pilot_path = Path(pilot_release_file).expanduser().resolve()
        pilot = load_and_validate_scoped_release(pilot_path)
        if (
            pilot.get("release_kind") != "pilot"
            or pilot.get("experiment_id") != scope["experiment_id"]
            or pilot.get("experiment_config_sha256") != sha256_file(scope_path)
            or pilot.get("minutes_system_prompt")
            != scope["minutes_system_prompt"]
            or pilot.get("decoding") != scope["decoding"]
            or pilot.get("prompt_template_sha256")
            != intervention_validation["prompt_template_sha256"]
            or pilot.get("minutes_model_sha256")
            != validation["minutes_artifacts"]["model_sha256"]
            or pilot.get("minutes_tokenizer_sha256")
            != validation["minutes_artifacts"]["tokenizer_sha256"]
            or pilot.get("analysis_model_sha256")
            != analysis_validation["analysis_model_sha256"]
            or pilot.get("analysis_tokenizer_sha256")
            != analysis_validation["analysis_tokenizer_sha256"]
        ):
            raise ValueError("Formal release does not match the scoped pilot gate")
        pilot_prerequisite = fingerprint_artifact_path(pilot_path)
    elif pilot_release_file is not None:
        raise ValueError("Only formal release accepts a pilot prerequisite")

    run_manifest_artifacts = {
        name: summary["generation_manifest"]
        for name, summary in validation["run_summaries"].items()
    }
    artifacts = {
        "experiment_config": fingerprint_artifact_path(scope_path),
        "indicator_roster": fingerprint_artifact_path(roster_path),
        "generation_spec": fingerprint_artifact_path(spec_path),
        "prompt_manifest": fingerprint_artifact_path(prompt_path),
        "analysis_manifest": fingerprint_artifact_path(analysis_path),
        "projection_manifest": fingerprint_artifact_path(projection_path),
        **run_manifest_artifacts,
    }
    if pilot_prerequisite is not None:
        artifacts["pilot_scoped_release"] = pilot_prerequisite
    payload = {
        "schema_version": SCOPED_RELEASE_SCHEMA_VERSION,
        "status": "complete",
        "release_kind": release_kind,
        "scoped_population_release": True,
        "standalone_full_roster_canonical_release": False,
        "complete_matrix": True,
        "experiment_id": scope["experiment_id"],
        "experiment_config_sha256": sha256_file(scope_path),
        "run_id": spec.get("run_id"),
        "phase": spec.get("phase"),
        "population_id": spec.get("population_id"),
        "baseline_indicator": scope["baseline_indicator"],
        "full_context_indicators": scope["full_context_indicators"],
        "intervention_indicators": scope["intervention_indicators"],
        "baseline_contains_full_roster": True,
        "minutes_system_prompt": scope["minutes_system_prompt"],
        "decoding": scope["decoding"],
        "prompt_template_sha256": intervention_validation[
            "prompt_template_sha256"
        ],
        "minutes_model_sha256": validation["minutes_artifacts"][
            "model_sha256"
        ],
        "minutes_tokenizer_sha256": validation["minutes_artifacts"][
            "tokenizer_sha256"
        ],
        "analysis_model_sha256": analysis_validation[
            "analysis_model_sha256"
        ],
        "analysis_tokenizer_sha256": analysis_validation[
            "analysis_tokenizer_sha256"
        ],
        "claim_boundary": scope["claim_boundary"],
        "release_policy": scope["release_policy"][release_kind],
        "matrix": {
            "meeting_count": 13,
            "section_count": 3,
            "total_generation_attempts": validation["total_generation_attempts"],
            "total_scored_pairs": validation["total_scored_pairs"],
            "generation_runs": validation["run_summaries"],
        },
        "strict_gate": {
            "zero_input_truncation": True,
            "zero_length_finishes": True,
            "zero_empty_outputs": True,
            "zero_generation_exclusions": True,
            "all_outputs_at_most_4096_tokens": True,
            "full_context_baseline": True,
            "paired_row_seeds": True,
            "population_sample_filtering": "forbidden",
            "exact_deletion_validated": True,
            "neutral_replacement_validated": True,
            "prompt_model_tokenizer_hashes_validated": True,
        },
        "intervention_validation": intervention_validation,
        "underlying_partial_shard": shard,
        "artifacts": artifacts,
    }
    return seal_manifest(payload)


def write_scoped_release_manifest(
    *, output_file: str | Path, **kwargs: Any
) -> Path:
    output = Path(output_file).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite scoped release: {output}")
    manifest = build_scoped_release_manifest(**kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal or validate a strict legacy-six scoped LOO release."
    )
    parser.add_argument("--validate-release")
    parser.add_argument("--validate-smoke-gate")
    parser.add_argument("--run-root")
    parser.add_argument("--generation-spec")
    parser.add_argument("--prompt-manifest")
    parser.add_argument("--analysis-manifest")
    parser.add_argument("--projection-manifest")
    parser.add_argument("--experiment-config")
    parser.add_argument("--roster")
    parser.add_argument("--release-kind", choices=("smoke", "pilot", "formal"))
    parser.add_argument("--pilot-release")
    parser.add_argument("--smoke-sample-id", action="append", default=[])
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.validate_release or args.validate_smoke_gate:
        forbidden = (
            args.run_root,
            args.generation_spec,
            args.prompt_manifest,
            args.analysis_manifest,
            args.projection_manifest,
            args.experiment_config,
            args.roster,
            args.release_kind,
            args.pilot_release,
            args.output,
        )
        if args.validate_release and args.validate_smoke_gate:
            raise SystemExit("Choose only one validation mode")
        if any(value is not None for value in forbidden) or args.smoke_sample_id:
            raise SystemExit("Validation mode cannot be combined with build args")
        if args.validate_release:
            manifest = load_and_validate_scoped_release(args.validate_release)
            print(f"{manifest['release_kind']}:{manifest['population_id']}:complete")
        else:
            manifest = load_and_validate_smoke_gate(args.validate_smoke_gate)
            print(f"smoke:{manifest['population_id']}:passed")
        return 0
    required = {
        "run_root": args.run_root,
        "generation_spec": args.generation_spec,
        "prompt_manifest": args.prompt_manifest,
        "analysis_manifest": args.analysis_manifest,
        "projection_manifest": args.projection_manifest,
        "experiment_config": args.experiment_config,
        "roster": args.roster,
        "release_kind": args.release_kind,
        "output": args.output,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(f"Missing required build arguments: {missing}")
    output = write_scoped_release_manifest(
        output_file=args.output,
        run_root=args.run_root,
        generation_spec_file=args.generation_spec,
        prompt_manifest_file=args.prompt_manifest,
        analysis_manifest_file=args.analysis_manifest,
        projection_manifest_file=args.projection_manifest,
        experiment_config_file=args.experiment_config,
        roster_file=args.roster,
        release_kind=args.release_kind,
        pilot_release_file=args.pilot_release,
        smoke_sample_ids=args.smoke_sample_id,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
