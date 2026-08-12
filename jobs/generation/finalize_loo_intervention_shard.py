"""Seal a partial LOO intervention-generation shard.

A shard is not a standalone canonical release.  It keeps the full frozen
indicator roster in every baseline prompt but generates deletion and neutral
outputs for only a declared subset of intervention indicators.  This permits
compute sharding without silently changing the LOO estimand.

An opt-in operational retry may retain token-limit rows as raw provenance
while excluding them from scoring. Such a shard is explicitly non-canonical
and exposes its complete exclusion inventory.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_generation_completion,
)


SHARD_SCHEMA_VERSION = "loo-generation-intervention-shard-v1"
GENERATION_SCHEMA_VERSION = "loo-generation-v4"
SCOPE_SCHEMA_VERSION = "loo-intervention-generation-scope-v1"
TOKEN_LIMIT_EXCLUSION_POLICY = "record-and-exclude-token-limit-v1"

EXPECTED_RUNS = {
    "deletion_primary": {
        "strategy": "indicator_block_deletion",
        "replicate_count": 1,
        "temperature": 0.0,
        "top_p": 1.0,
    },
    "neutral_primary": {
        "strategy": "indicator_block_neutral_replacement",
        "replicate_count": 1,
        "temperature": 0.0,
        "top_p": 1.0,
    },
    "deletion_stochastic": {
        "strategy": "indicator_block_deletion",
        "replicate_count": 5,
        "temperature": 0.6,
        "top_p": 0.9,
    },
    "neutral_stochastic": {
        "strategy": "indicator_block_neutral_replacement",
        "replicate_count": 5,
        "temperature": 0.6,
        "top_p": 0.9,
    },
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


def _validate_selected_indicators(
    *,
    prompt_manifest: Mapping[str, Any],
    selected_indicators: Sequence[str],
) -> tuple[str, list[str], list[str]]:
    baseline = str(prompt_manifest.get("baseline_indicator") or "").strip()
    full_roster = [
        str(value).strip()
        for value in prompt_manifest.get("indicators", [])
        if str(value).strip()
    ]
    selected = [str(value).strip() for value in selected_indicators]
    if not baseline or not full_roster:
        raise ValueError("Prompt manifest has an invalid full indicator roster")
    if not selected or any(not value for value in selected):
        raise ValueError("At least one non-empty shard indicator is required")
    if len(selected) != len(set(selected)):
        raise ValueError("Shard indicator selection contains duplicates")
    if baseline in selected:
        raise ValueError("The baseline indicator is included automatically")
    unknown = sorted(set(selected) - set(full_roster))
    if unknown:
        raise ValueError(f"Shard indicators are outside the prompt roster: {unknown}")
    selected_set = set(selected)
    ordered = [value for value in full_roster if value in selected_set]
    if ordered == full_roster:
        raise ValueError(
            "A full-roster run must use the canonical generation finalizer, "
            "not the partial shard finalizer"
        )
    return baseline, full_roster, ordered


def _validate_artifact_hashes(artifacts: object, *, label: str) -> None:
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"{label} must contain non-empty expected_artifacts")
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            raise ValueError(f"{label} artifact {index} must be an object")
        path = Path(str(artifact.get("path") or "")).expanduser().resolve()
        expected_sha256 = str(artifact.get("output_sha256") or "")
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise ValueError(f"{label} generated artifact changed or is missing: {path}")


def _validate_minutes_artifact_binding(
    *,
    generation_manifest: Mapping[str, Any],
    generation_spec: Mapping[str, Any],
    label: str,
) -> None:
    frozen_artifacts = generation_spec.get("frozen_artifacts")
    if not isinstance(frozen_artifacts, Mapping):
        raise ValueError("Generation spec lacks frozen_artifacts")
    frozen_models = frozen_artifacts.get("models")
    frozen_tokenizers = frozen_artifacts.get("tokenizers")
    minutes_model = (
        frozen_models.get("minutes_model")
        if isinstance(frozen_models, Mapping)
        else None
    )
    minutes_tokenizer = (
        frozen_tokenizers.get("minutes_tokenizer")
        if isinstance(frozen_tokenizers, Mapping)
        else None
    )
    if not isinstance(minutes_model, Mapping):
        raise ValueError("Generation spec lacks frozen models.minutes_model")
    if not isinstance(minutes_tokenizer, Mapping):
        raise ValueError(
            "Generation spec lacks frozen tokenizers.minutes_tokenizer"
        )

    expected = {
        "model_artifact": minutes_model,
        "tokenizer_artifact": minutes_tokenizer,
    }
    identity_fields = (
        "sha256",
        "kind",
        "file_count",
        "total_bytes",
        "algorithm",
    )
    for field, frozen in expected.items():
        before = generation_manifest.get(field)
        after = generation_manifest.get(f"{field}_after_generation")
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise ValueError(f"{label} lacks complete {field} fingerprints")
        if before.get("sha256") != frozen.get("sha256"):
            raise ValueError(
                f"{label} {field} differs from the named frozen Minutes artifact"
            )
        changes = {
            key: {"before": before.get(key), "after": after.get(key)}
            for key in identity_fields
            if before.get(key) != after.get(key)
        }
        if changes:
            raise ValueError(
                f"{label} {field} changed during generation: {changes}"
            )


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
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
                raise ValueError(
                    f"{label} row must be an object: {path}:{line_number}"
                )
            rows.append(row)
    return rows


def _completion_error_record(
    row: Mapping[str, Any],
    *,
    artifact: Mapping[str, Any],
    finish_reason: str,
    output_token_count: int,
    max_new_tokens: int,
    max_model_len: int,
) -> dict[str, Any]:
    return {
        "error_type": "token_limit_finish",
        "sample_id": str(row.get("sample_id") or ""),
        "meeting_date": str(row.get("meeting_date") or ""),
        "section_name": str(row.get("section_name") or ""),
        "generation_position": int(row["generation_position"]),
        "finish_reason": finish_reason,
        "output_token_count": output_token_count,
        "max_new_tokens": max_new_tokens,
        "max_model_len": max_model_len,
        "generated_sha256": str(row.get("generated_sha256") or ""),
        "artifact_relative_path": str(artifact.get("relative_path") or ""),
        "indicator": str(artifact.get("indicator") or ""),
        "context": str(artifact.get("context") or ""),
        "replicate_id": str(artifact.get("replicate_id") or ""),
    }


def validate_generation_completion_manifest(
    *,
    manifest: Mapping[str, Any],
    expected_artifacts: object,
    label: str,
) -> dict[str, Any]:
    """Recompute an exclusion-aware completion summary from hashed raw rows."""

    declared = manifest.get("completion_validation")
    if declared is None:
        return {
            "policy": "strict-legacy-v1",
            "token_limit_policy": "error",
            "attempted_row_count": None,
            "passed_row_count": None,
            "excluded_row_count": 0,
            "exclusions": [],
        }
    if not isinstance(declared, Mapping):
        raise ValueError(f"{label} completion_validation must be an object")
    token_limit_policy = str(declared.get("token_limit_policy") or "")
    if token_limit_policy not in {"error", "exclude"}:
        raise ValueError(f"{label} has an invalid token-limit policy")
    expected_policy = (
        TOKEN_LIMIT_EXCLUSION_POLICY
        if token_limit_policy == "exclude"
        else "strict-token-limit-error-v1"
    )
    if declared.get("policy") != expected_policy:
        raise ValueError(f"{label} completion policy metadata is inconsistent")
    if not isinstance(expected_artifacts, list) or not expected_artifacts:
        raise ValueError(f"{label} lacks expected artifacts")

    attempted = 0
    passed = 0
    exclusions: list[dict[str, Any]] = []
    for artifact in expected_artifacts:
        if not isinstance(artifact, Mapping):
            raise ValueError(f"{label} has an invalid expected artifact")
        path = Path(str(artifact.get("path") or "")).expanduser().resolve()
        rows = _read_jsonl(path, label=f"{label} output")
        artifact_exclusions: list[dict[str, Any]] = []
        artifact_passed = 0
        for row in rows:
            try:
                prompt_tokens = int(row["prompt_token_count"])
                preflight_tokens = int(row["prompt_preflight_token_count"])
                output_tokens = int(row["output_token_count"])
                max_new_tokens = int(manifest["max_new_tokens"])
                max_model_len = int(manifest["max_model_len"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{label} output lacks valid completion token metadata"
                ) from exc
            finish_reason = str(row.get("generation_finish_reason") or "")
            completion = validate_generation_completion(
                input_token_count=preflight_tokens,
                output_token_count=output_tokens,
                max_new_tokens=max_new_tokens,
                context_limit=max_model_len,
                finish_reason=finish_reason,
                input_was_truncated=row.get("input_was_truncated"),
                consumed_input_token_count=prompt_tokens,
                token_limit_policy=token_limit_policy,
            )
            status = completion["generation_validation_status"]
            if row.get("generation_validation_status") != status:
                raise ValueError(
                    f"{label} output completion status differs from raw metadata"
                )
            if status == "excluded_token_limit_finish":
                error = _completion_error_record(
                    row,
                    artifact=artifact,
                    finish_reason=finish_reason,
                    output_token_count=output_tokens,
                    max_new_tokens=max_new_tokens,
                    max_model_len=max_model_len,
                )
                if row.get("generation_validation_error") != {
                    key: value
                    for key, value in error.items()
                    if key
                    not in {
                        "artifact_relative_path",
                        "indicator",
                        "context",
                        "replicate_id",
                    }
                }:
                    raise ValueError(
                        f"{label} output has invalid exclusion metadata"
                    )
                artifact_exclusions.append(error)
            else:
                artifact_passed += 1
                if "generation_validation_error" in row:
                    raise ValueError(
                        f"{label} passed output contains an exclusion error"
                    )
        attempted += len(rows)
        passed += artifact_passed
        exclusions.extend(artifact_exclusions)
        artifact_expected = {
            "attempted_row_count": len(rows),
            "scorable_row_count": artifact_passed,
            "excluded_row_count": len(artifact_exclusions),
            "completion_exclusions": artifact_exclusions,
        }
        mismatches = {
            key: {"expected": value, "observed": artifact.get(key)}
            for key, value in artifact_expected.items()
            if artifact.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"{label} artifact completion summary mismatch: {mismatches}"
            )

    expected_summary = {
        "attempted_row_count": attempted,
        "passed_row_count": passed,
        "excluded_row_count": len(exclusions),
        "exclusions": exclusions,
    }
    mismatches = {
        key: {"expected": value, "observed": declared.get(key)}
        for key, value in expected_summary.items()
        if declared.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{label} completion summary mismatch: {mismatches}")
    if manifest.get("generation_attempts_complete") is not True:
        raise ValueError(f"{label} generation attempts are not complete")
    if manifest.get("scorable_population_complete") is not (
        len(exclusions) == 0
    ):
        raise ValueError(f"{label} scorable population status is inconsistent")
    expected_status = "complete_with_exclusions" if exclusions else "complete"
    if manifest.get("status") != expected_status:
        raise ValueError(f"{label} generation status is inconsistent")
    return {
        "policy": expected_policy,
        "token_limit_policy": token_limit_policy,
        **expected_summary,
    }


def build_intervention_shard_manifest(
    *,
    run_root: str | Path,
    generation_spec_file: str | Path,
    prompt_manifest_file: str | Path,
    analysis_manifest_file: str | Path,
    projection_manifest_file: str | Path,
    selected_indicators: Sequence[str],
) -> dict[str, Any]:
    root = Path(run_root).expanduser().resolve()
    generation_spec_path = Path(generation_spec_file).expanduser().resolve()
    prompt_manifest_path = Path(prompt_manifest_file).expanduser().resolve()
    analysis_manifest_path = Path(analysis_manifest_file).expanduser().resolve()
    projection_manifest_path = Path(projection_manifest_file).expanduser().resolve()

    generation_spec = _read_json(generation_spec_path, label="generation spec")
    prompt_manifest = _read_json(prompt_manifest_path, label="prompt manifest")
    _read_json(analysis_manifest_path, label="analysis manifest")
    _read_json(projection_manifest_path, label="projection manifest")
    baseline, full_roster, selected = _validate_selected_indicators(
        prompt_manifest=prompt_manifest,
        selected_indicators=selected_indicators,
    )
    selected_universe = {baseline, *selected}
    omitted = [value for value in full_roster if value not in set(selected)]
    spec_sha256 = sha256_file(generation_spec_path)
    prompt_sha256 = sha256_file(prompt_manifest_path)

    run_artifacts: dict[str, dict[str, Any]] = {}
    execution_environments: list[dict[str, Any]] = []
    completion_summaries: dict[str, dict[str, Any]] = {}
    all_completion_exclusions: list[dict[str, Any]] = []
    for run_name, expectation in EXPECTED_RUNS.items():
        manifest_path = root / "generations" / run_name / "generation_manifest.json"
        manifest = _read_json(manifest_path, label=f"{run_name} generation manifest")
        if manifest.get("schema_version") != GENERATION_SCHEMA_VERSION:
            raise ValueError(f"{run_name} uses an unsupported generation schema")
        if manifest.get("masking_strategy") != expectation["strategy"]:
            raise ValueError(f"{run_name} masking strategy mismatch")
        if manifest.get("simulation_step") != expectation["replicate_count"]:
            raise ValueError(f"{run_name} replicate count mismatch")
        for field in ("temperature", "top_p"):
            observed = manifest.get(field)
            expected = expectation[field]
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not math.isclose(float(observed), float(expected), abs_tol=1e-12)
            ):
                raise ValueError(f"{run_name} {field} mismatch")

        scope = manifest.get("intervention_scope")
        expected_scope = {
            "schema_version": SCOPE_SCHEMA_VERSION,
            "mode": "partial_intervention_shard",
            "baseline_indicator": baseline,
            "full_roster_indicators": full_roster,
            "generated_intervention_indicators": selected,
            "omitted_intervention_indicators": omitted,
            "baseline_contains_full_roster": True,
            "standalone_canonical_release_allowed": False,
        }
        if scope != expected_scope:
            raise ValueError(f"{run_name} intervention scope mismatch")

        spec_binding = manifest.get("generation_spec")
        prompt_binding = manifest.get("prompt_manifest")
        if (
            not isinstance(spec_binding, Mapping)
            or spec_binding.get("file_sha256") != spec_sha256
        ):
            raise ValueError(f"{run_name} generation-spec binding mismatch")
        if (
            not isinstance(prompt_binding, Mapping)
            or prompt_binding.get("sha256") != prompt_sha256
        ):
            raise ValueError(f"{run_name} prompt-manifest binding mismatch")
        _validate_minutes_artifact_binding(
            generation_manifest=manifest,
            generation_spec=generation_spec,
            label=run_name,
        )

        input_artifacts = manifest.get("input_artifacts")
        if not isinstance(input_artifacts, list):
            raise ValueError(f"{run_name} lacks input_artifacts")
        observed_input_indicators = {
            str(record.get("indicator") or "")
            for record in input_artifacts
            if isinstance(record, Mapping)
        }
        if observed_input_indicators != selected_universe:
            raise ValueError(f"{run_name} input indicator inventory mismatch")

        expected_artifacts = manifest.get("expected_artifacts")
        _validate_artifact_hashes(expected_artifacts, label=run_name)
        completion_summary = validate_generation_completion_manifest(
            manifest=manifest,
            expected_artifacts=expected_artifacts,
            label=run_name,
        )
        completion_summaries[run_name] = completion_summary
        all_completion_exclusions.extend(
            {
                **record,
                "generation_run": run_name,
            }
            for record in completion_summary["exclusions"]
        )
        observed_output_indicators = {
            str(record.get("indicator") or "")
            for record in expected_artifacts
            if isinstance(record, Mapping)
        }
        if observed_output_indicators != selected_universe:
            raise ValueError(f"{run_name} output indicator inventory mismatch")

        execution_environment = manifest.get("execution_environment")
        if not isinstance(execution_environment, Mapping):
            raise ValueError(f"{run_name} lacks a canonical execution environment")
        if (
            str(execution_environment.get("physical_gpu_index")) != "1"
            or str(execution_environment.get("cuda_visible_devices")) != "1"
            or str(execution_environment.get("declared_visible_device_count")) != "1"
            or execution_environment.get("tensor_parallel_size") != 1
            or not str(execution_environment.get("physical_gpu_uuid") or "").startswith(
                "GPU-"
            )
        ):
            raise ValueError(f"{run_name} was not confined to physical GPU 1")
        execution_environments.append(dict(execution_environment))
        run_artifacts[run_name] = fingerprint_artifact_path(manifest_path)

    if any(value != execution_environments[0] for value in execution_environments[1:]):
        raise ValueError("Generation shard execution environments are inconsistent")

    token_limit_policies = {
        summary["token_limit_policy"]
        for summary in completion_summaries.values()
    }
    if len(token_limit_policies) != 1:
        raise ValueError("Generation shard token-limit policies are inconsistent")
    token_limit_policy = next(iter(token_limit_policies))
    frozen_minutes_policy = (
        generation_spec.get("generation_config", {})
        .get("invariants", {})
        .get("minutes_token_limit_finish")
    )
    protocol_amendment = (
        token_limit_policy == "exclude"
        and frozen_minutes_policy != {
            "policy": TOKEN_LIMIT_EXCLUSION_POLICY,
            "raw_outputs_retained": True,
            "scoring_eligibility": "excluded",
        }
    )
    excluded_row_count = len(all_completion_exclusions)
    status = (
        "partial_complete_with_generation_exclusions"
        if excluded_row_count
        else "partial_complete"
    )
    canonical_compatible = not protocol_amendment and excluded_row_count == 0
    if excluded_row_count:
        claim_boundary = (
            "Generation attempts completed, but token-limit outputs are retained "
            "only as raw provenance and must be excluded from every LOO score. "
            "This is a complete-case, non-canonical intervention shard and "
            "cannot support a full-population canonical causal claim."
        )
    elif protocol_amendment:
        claim_boundary = (
            "Generation completed under a post-specification token-limit "
            "exclusion policy. No exclusion was observed, but this operational "
            "retry is not a standalone canonical release."
        )
    else:
        claim_boundary = (
            "Canonical-compatible intervention shard only: every baseline "
            "retains the full frozen indicator universe, but the omitted "
            "interventions must be generated under the same specification "
            "before a complete canonical release can be sealed."
        )

    payload = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "status": status,
        "standalone_canonical_release": False,
        "canonical_compatible": canonical_compatible,
        "generation_attempts_complete": True,
        "scorable_population_complete": excluded_row_count == 0,
        "claim_boundary": claim_boundary,
        "run_id": generation_spec.get("run_id"),
        "phase": generation_spec.get("phase"),
        "population_id": generation_spec.get("population_id"),
        "baseline_indicator": baseline,
        "full_roster_indicators": full_roster,
        "generated_intervention_indicators": selected,
        "omitted_intervention_indicators": omitted,
        "baseline_contains_full_roster": True,
        "execution_environment": execution_environments[0],
        "completion_validation": {
            "token_limit_policy": token_limit_policy,
            "protocol_amendment": protocol_amendment,
            "frozen_minutes_token_limit_policy": frozen_minutes_policy,
            "attempted_row_count": sum(
                int(summary["attempted_row_count"] or 0)
                for summary in completion_summaries.values()
            ),
            "passed_row_count": sum(
                int(summary["passed_row_count"] or 0)
                for summary in completion_summaries.values()
            ),
            "excluded_row_count": excluded_row_count,
            "exclusions": all_completion_exclusions,
            "generation_runs": completion_summaries,
        },
        "artifacts": {
            "generation_spec": fingerprint_artifact_path(generation_spec_path),
            "prompt_manifest": fingerprint_artifact_path(prompt_manifest_path),
            "analysis_manifest": fingerprint_artifact_path(analysis_manifest_path),
            "projection_manifest": fingerprint_artifact_path(
                projection_manifest_path
            ),
            "generation_manifests": run_artifacts,
        },
    }
    return seal_manifest(payload)


def write_intervention_shard_manifest(
    *,
    output_file: str | Path,
    **kwargs: Any,
) -> Path:
    output = Path(output_file).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite shard manifest: {output}")
    manifest = build_intervention_shard_manifest(**kwargs)
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
        description="Seal a non-standalone canonical LOO intervention shard."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--generation-spec", required=True)
    parser.add_argument("--prompt-manifest", required=True)
    parser.add_argument("--analysis-manifest", required=True)
    parser.add_argument("--projection-manifest", required=True)
    parser.add_argument(
        "--indicator",
        action="append",
        dest="selected_indicators",
        required=True,
    )
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = write_intervention_shard_manifest(
        output_file=args.output,
        run_root=args.run_root,
        generation_spec_file=args.generation_spec,
        prompt_manifest_file=args.prompt_manifest,
        analysis_manifest_file=args.analysis_manifest,
        projection_manifest_file=args.projection_manifest,
        selected_indicators=args.selected_indicators,
    )
    print(output)


if __name__ == "__main__":
    main()
