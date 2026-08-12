"""Seal a complete two-stage legacy-six generation workflow."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jobs.generation.finalize_loo_scoped_release import (
    load_and_validate_scoped_release,
    load_and_validate_smoke_gate,
)
from jobs.eval.summarize_canonical_loo import validate_report_manifest
from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.validator.loo_experiment_scope import (
    load_and_validate_experiment_scope,
)
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


SCHEMA_VERSION = "loo-scoped-workflow-release-v1"
SCORING_PROTOCOL_SCHEMA_VERSION = "loo-scoring-protocol-signature-v1"


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _require_within(path: Path, *, root: Path, label: str) -> None:
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} must be stored under the workflow root")


def _validate_ledger(
    path: Path,
    *,
    population_id: str,
    source_registry_sha256: str,
    snapshot_sha256: str,
) -> dict[str, Any]:
    ledger = _read_json(path, label=f"{population_id} ledger")
    if (
        ledger.get("schema_version") != "loo-indicator-ledger-manifest-v1"
        or ledger.get("status") != "complete"
        or ledger.get("population_id") != population_id
    ):
        raise ValueError(f"{population_id} ledger is not complete")
    validate_manifest_integrity(ledger)
    inputs = ledger.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError(f"{population_id} ledger lacks input bindings")
    if (
        inputs.get("registry", {}).get("sha256") != source_registry_sha256
        or inputs.get("snapshot_manifest", {}).get("sha256") != snapshot_sha256
    ):
        raise ValueError(f"{population_id} ledger source binding mismatch")
    return ledger


def _validate_results_manifest(
    path: Path,
    *,
    scope_path: Path,
    population_id: str,
) -> dict[str, Any]:
    validate_report_manifest(
        path,
        scope_manifest_file=scope_path,
        population_id=population_id,
    )
    report = _read_json(path, label=f"{population_id} results manifest")
    arm_signatures: list[dict[str, Any]] = []
    for arm, record in report.get("inputs", {}).items():
        if not isinstance(record, Mapping):
            raise ValueError(f"Invalid {population_id} report input {arm}")
        audit = _read_json(
            Path(str(record.get("audit_file") or "")).expanduser().resolve(),
            label=f"{population_id} {arm} scoring audit",
        )
        embedding = audit.get("embedding_model_artifact")
        scoped = audit.get("scoped_experiment")
        if not isinstance(embedding, Mapping) or not isinstance(scoped, Mapping):
            raise ValueError(f"{population_id} {arm} lacks scoring provenance")
        reference_binding = scoped.get("reference_manifest")
        if not isinstance(reference_binding, Mapping):
            raise ValueError(f"{population_id} {arm} lacks reference provenance")
        reference_manifest_path = Path(
            str(reference_binding.get("path") or "")
        ).expanduser().resolve()
        reference_manifest = _read_json(
            reference_manifest_path,
            label=f"{population_id} {arm} reference manifest",
        )
        source = reference_manifest.get("source")
        if not isinstance(source, Mapping):
            raise ValueError(
                f"{population_id} {arm} reference manifest lacks source provenance"
            )
        source_path = Path(str(source.get("path") or "")).expanduser().resolve()
        source_sha256 = str(source.get("sha256") or "")
        if (
            len(source_sha256) != 64
            or not source_path.is_file()
            or sha256_file(source_path) != source_sha256
        ):
            raise ValueError(
                f"{population_id} {arm} Actual-Minutes source changed or is missing"
            )
        signature = {
            "schema_version": SCORING_PROTOCOL_SCHEMA_VERSION,
            "scorer_schema_version": audit.get("schema_version"),
            "target": {
                "mode": audit.get("target_mode"),
                "primary_estimand": audit.get("primary_estimand"),
            },
            "embedding": {
                "artifact_path": embedding.get("path")
                or audit.get("embedding_model_path"),
                "artifact_sha256": embedding.get("sha256"),
                "artifact_kind": embedding.get("kind"),
                "batch_size": audit.get("embedding_batch_size"),
                "max_tokens": audit.get("embedding_max_tokens"),
                "long_text_policy": audit.get("embedding_long_text_policy"),
                "score_chunk_size": audit.get("score_chunk_size"),
            },
            "reference": {
                "view_schema_version": reference_manifest.get("schema_version"),
                "key_fields": audit.get("reference_key_fields"),
                "text_field": audit.get("reference_text_field"),
                "duplicate_policy": audit.get("reference_duplicate_policy"),
                "source_sha256": source_sha256,
                "source_text_field": source.get("text_field"),
            },
            "report": {
                "estimand": report.get("estimand"),
                "aggregation_order": report.get("aggregation_order"),
                "inference_unit": report.get("inference_unit"),
                "bootstrap": report.get("bootstrap"),
                "multiplicity": report.get("multiplicity"),
            },
        }
        required_values = (
            signature["scorer_schema_version"],
            signature["target"]["mode"],
            signature["target"]["primary_estimand"],
            signature["embedding"]["artifact_sha256"],
            signature["embedding"]["artifact_path"],
            signature["embedding"]["artifact_kind"],
            signature["embedding"]["batch_size"],
            signature["embedding"]["max_tokens"],
            signature["embedding"]["long_text_policy"],
            signature["embedding"]["score_chunk_size"],
            signature["reference"]["view_schema_version"],
            signature["reference"]["key_fields"],
            signature["reference"]["text_field"],
            signature["reference"]["duplicate_policy"],
            signature["reference"]["source_text_field"],
            signature["report"]["estimand"],
            signature["report"]["aggregation_order"],
            signature["report"]["inference_unit"],
            signature["report"]["bootstrap"],
            signature["report"]["multiplicity"],
        )
        if any(value is None or value == "" for value in required_values):
            raise ValueError(
                f"{population_id} {arm} has an incomplete scoring protocol"
            )
        arm_signatures.append(signature)
    if not arm_signatures or any(
        signature != arm_signatures[0] for signature in arm_signatures[1:]
    ):
        raise ValueError(
            f"{population_id} scoring arms use different scoring protocols"
        )
    return arm_signatures[0]


def _require_matching_scoring_protocols(
    pilot: Mapping[str, Any],
    formal: Mapping[str, Any],
) -> None:
    if dict(pilot) != dict(formal):
        raise ValueError(
            "Pilot and formal scoring protocols differ; model, embedding settings, "
            "target/reference contract, Actual-Minutes source, and inference "
            "configuration must remain frozen"
        )


def build_scoped_workflow_manifest(
    *,
    run_id: str,
    mode: str,
    workflow_root: str | Path,
    experiment_config_file: str | Path,
    source_registry_file: str | Path,
    snapshot_manifest_file: str | Path,
    smoke_gate_file: str | Path | None,
    pilot_ledger_file: str | Path | None,
    pilot_release_file: str | Path,
    formal_ledger_file: str | Path | None,
    formal_release_file: str | Path | None,
    reused_pilot_analysis_manifest_file: str | Path | None = None,
    reused_pilot_analysis_output_file: str | Path | None = None,
    reused_pilot_projection_manifest_file: str | Path | None = None,
    reused_pilot_projection_output_file: str | Path | None = None,
    pilot_results_manifest_file: str | Path | None = None,
    formal_results_manifest_file: str | Path | None = None,
) -> dict[str, Any]:
    if mode not in {"pilot", "formal", "all"}:
        raise ValueError("mode must be pilot, formal, or all")
    root = Path(workflow_root).expanduser().resolve()
    scope_path = Path(experiment_config_file).expanduser().resolve()
    registry_path = Path(source_registry_file).expanduser().resolve()
    snapshot_path = Path(snapshot_manifest_file).expanduser().resolve()
    pilot_release_path = Path(pilot_release_file).expanduser().resolve()
    paths = [scope_path, registry_path, snapshot_path, pilot_release_path]
    optional_paths = [
        smoke_gate_file,
        pilot_ledger_file,
        formal_ledger_file,
        formal_release_file,
        reused_pilot_analysis_manifest_file,
        reused_pilot_analysis_output_file,
        reused_pilot_projection_manifest_file,
        reused_pilot_projection_output_file,
        pilot_results_manifest_file,
        formal_results_manifest_file,
    ]
    paths.extend(
        Path(value).expanduser().resolve()
        for value in optional_paths
        if value is not None
    )
    for path in paths:
        _require_within(path, root=root, label=str(path))

    scope = load_and_validate_experiment_scope(scope_path)
    registry = _read_json(registry_path, label="source registry")
    snapshot = _read_json(snapshot_path, label="source snapshot manifest")
    if registry.get("schema_version") != "loo-indicator-source-registry-v1":
        raise ValueError("Unexpected source registry schema")
    if (
        snapshot.get("schema_version") != "loo-source-snapshot-manifest-v1"
        or snapshot.get("status") != "complete"
    ):
        raise ValueError("Source snapshot manifest is not complete")
    validate_manifest_integrity(snapshot)
    registry_sha = sha256_file(registry_path)
    snapshot_sha = sha256_file(snapshot_path)
    if snapshot.get("registry", {}).get("sha256") != registry_sha:
        raise ValueError("Snapshot does not bind the frozen source registry")

    artifacts: dict[str, dict[str, Any]] = {
        "experiment_config": fingerprint_artifact_path(scope_path),
        "source_registry": fingerprint_artifact_path(registry_path),
        "snapshot_manifest": fingerprint_artifact_path(snapshot_path),
    }
    pilot = load_and_validate_scoped_release(pilot_release_path)
    if (
        pilot.get("release_kind") != "pilot"
        or pilot.get("population_id") != "pilot_eval_13"
        or pilot.get("experiment_config_sha256") != sha256_file(scope_path)
    ):
        raise ValueError("Pilot prerequisite does not match the workflow scope")
    artifacts["pilot_release"] = fingerprint_artifact_path(pilot_release_path)

    pilot_scoring_protocol: dict[str, Any] | None = None
    if mode in {"pilot", "all"}:
        if smoke_gate_file is None or pilot_ledger_file is None:
            raise ValueError("Pilot workflow requires smoke gate and pilot ledger")
        smoke_path = Path(smoke_gate_file).expanduser().resolve()
        pilot_ledger_path = Path(pilot_ledger_file).expanduser().resolve()
        smoke = load_and_validate_smoke_gate(smoke_path)
        if smoke.get("experiment_id") != scope["experiment_id"]:
            raise ValueError("Smoke gate experiment differs from workflow scope")
        _validate_ledger(
            pilot_ledger_path,
            population_id="pilot_eval_13",
            source_registry_sha256=registry_sha,
            snapshot_sha256=snapshot_sha,
        )
        artifacts["smoke_gate"] = fingerprint_artifact_path(smoke_path)
        artifacts["pilot_ledger"] = fingerprint_artifact_path(pilot_ledger_path)
        if pilot_results_manifest_file is None:
            raise ValueError("Pilot workflow requires a validated result manifest")
        pilot_results_path = Path(pilot_results_manifest_file).expanduser().resolve()
        pilot_scoring_protocol = _validate_results_manifest(
            pilot_results_path,
            scope_path=scope_path,
            population_id="pilot_eval_13",
        )
        artifacts["pilot_results_manifest"] = fingerprint_artifact_path(
            pilot_results_path
        )
        reused = {
            "reused_pilot_analysis_manifest": reused_pilot_analysis_manifest_file,
            "reused_pilot_analysis_output": reused_pilot_analysis_output_file,
            "reused_pilot_projection_manifest": reused_pilot_projection_manifest_file,
            "reused_pilot_projection_output": reused_pilot_projection_output_file,
        }
        if any(value is None for value in reused.values()):
            raise ValueError("Pilot workflow must freeze all reused analysis artifacts")
        for name, value in reused.items():
            assert value is not None
            artifacts[name] = fingerprint_artifact_path(
                Path(value).expanduser().resolve()
            )
    elif smoke_gate_file is not None or pilot_ledger_file is not None:
        raise ValueError("Formal-only workflow cannot bind a local pilot run")
    if mode == "formal":
        if pilot_results_manifest_file is None:
            raise ValueError(
                "Formal-only workflow requires its frozen pilot results manifest"
            )
        pilot_results_path = Path(
            pilot_results_manifest_file
        ).expanduser().resolve()
        pilot_scoring_protocol = _validate_results_manifest(
            pilot_results_path,
            scope_path=scope_path,
            population_id="pilot_eval_13",
        )
        artifacts["pilot_results_manifest"] = fingerprint_artifact_path(
            pilot_results_path
        )

    formal: dict[str, Any] | None = None
    if mode in {"formal", "all"}:
        if formal_ledger_file is None or formal_release_file is None:
            raise ValueError("Formal workflow requires formal ledger and release")
        formal_ledger_path = Path(formal_ledger_file).expanduser().resolve()
        formal_release_path = Path(formal_release_file).expanduser().resolve()
        _validate_ledger(
            formal_ledger_path,
            population_id="formal_test_13",
            source_registry_sha256=registry_sha,
            snapshot_sha256=snapshot_sha,
        )
        formal = load_and_validate_scoped_release(formal_release_path)
        if (
            formal.get("release_kind") != "formal"
            or formal.get("population_id") != "formal_test_13"
            or formal.get("experiment_config_sha256") != sha256_file(scope_path)
            or formal.get("minutes_model_sha256")
            != pilot.get("minutes_model_sha256")
            or formal.get("minutes_tokenizer_sha256")
            != pilot.get("minutes_tokenizer_sha256")
            or formal.get("analysis_model_sha256")
            != pilot.get("analysis_model_sha256")
            or formal.get("analysis_tokenizer_sha256")
            != pilot.get("analysis_tokenizer_sha256")
        ):
            raise ValueError("Formal release differs from its pilot prerequisite")
        artifacts["formal_ledger"] = fingerprint_artifact_path(formal_ledger_path)
        artifacts["formal_release"] = fingerprint_artifact_path(formal_release_path)
        if formal_results_manifest_file is None:
            raise ValueError("Formal workflow requires a validated result manifest")
        formal_results_path = Path(formal_results_manifest_file).expanduser().resolve()
        formal_scoring_protocol = _validate_results_manifest(
            formal_results_path,
            scope_path=scope_path,
            population_id="formal_test_13",
        )
        artifacts["formal_results_manifest"] = fingerprint_artifact_path(
            formal_results_path
        )
        assert pilot_scoring_protocol is not None
        _require_matching_scoring_protocols(
            pilot_scoring_protocol,
            formal_scoring_protocol,
        )
    elif formal_ledger_file is not None or formal_release_file is not None:
        raise ValueError("Pilot-only workflow cannot bind formal artifacts")

    expected_vintages = 26 if mode == "all" else 13
    if snapshot.get("vintage_count") != expected_vintages:
        raise ValueError("Snapshot population does not match workflow mode")
    assert pilot_scoring_protocol is not None
    final_scoring_protocol = (
        formal_scoring_protocol
        if formal is not None
        else pilot_scoring_protocol
    )
    return seal_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "run_id": run_id,
            "mode": mode,
            "generation_only": False,
            "training_performed": False,
            "workflow_components": {
                "generation": "complete",
                "validation": "complete",
                "scoring": "complete",
                "aggregation": "complete",
            },
            "experiment_id": scope["experiment_id"],
            "experiment_config_sha256": sha256_file(scope_path),
            "smoke_gate_required": mode in {"pilot", "all"},
            "pilot_gate_passed": True,
            "formal_release_present": formal is not None,
            "standalone_full_roster_canonical_release": False,
            "claim_boundary": scope["claim_boundary"],
            "minutes_model_sha256": pilot["minutes_model_sha256"],
            "minutes_tokenizer_sha256": pilot["minutes_tokenizer_sha256"],
            "analysis_model_sha256": pilot["analysis_model_sha256"],
            "analysis_tokenizer_sha256": pilot[
                "analysis_tokenizer_sha256"
            ],
            "embedding_model_sha256": final_scoring_protocol["embedding"][
                "artifact_sha256"
            ],
            "scoring_protocol_signature": final_scoring_protocol,
            "artifacts": artifacts,
        }
    )


def write_scoped_workflow_manifest(
    *, output_file: str | Path, **kwargs: Any
) -> Path:
    output = Path(output_file).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite scoped workflow: {output}")
    payload = build_scoped_workflow_manifest(**kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def load_and_validate_scoped_workflow(
    manifest_file: str | Path,
) -> dict[str, Any]:
    """Deeply revalidate a completed workflow and every bound artifact."""

    path = Path(manifest_file).expanduser().resolve()
    manifest = _read_json(path, label="scoped workflow release")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
    ):
        raise ValueError("Scoped workflow release is not complete")
    validate_manifest_integrity(manifest)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Scoped workflow release lacks artifacts")
    for name, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"Invalid workflow artifact {name}")
        artifact_path = Path(str(record.get("path") or "")).expanduser().resolve()
        observed = fingerprint_artifact_path(artifact_path)
        for field in ("sha256", "kind", "file_count", "total_bytes", "algorithm"):
            if observed.get(field) != record.get(field):
                raise ValueError(f"Workflow artifact changed: {name}")

    mode = str(manifest.get("mode") or "")
    common_names = {
        "experiment_config",
        "source_registry",
        "snapshot_manifest",
        "pilot_release",
    }
    pilot_names = {
        "smoke_gate",
        "pilot_ledger",
        "pilot_results_manifest",
        "reused_pilot_analysis_manifest",
        "reused_pilot_analysis_output",
        "reused_pilot_projection_manifest",
        "reused_pilot_projection_output",
    }
    formal_names = {
        "formal_ledger",
        "formal_release",
        "formal_results_manifest",
    }
    expected_names = common_names | (
        pilot_names if mode in {"pilot", "all"} else set()
    ) | (formal_names if mode in {"formal", "all"} else set()) | (
        {"pilot_results_manifest"} if mode == "formal" else set()
    )
    if mode not in {"pilot", "formal", "all"} or set(artifacts) != expected_names:
        raise ValueError("Scoped workflow artifact inventory does not match mode")

    scope_path = Path(artifacts["experiment_config"]["path"]).resolve()
    scope = load_and_validate_experiment_scope(scope_path)
    if manifest.get("experiment_config_sha256") != sha256_file(scope_path):
        raise ValueError("Workflow scope hash mismatch")
    registry_path = Path(artifacts["source_registry"]["path"]).resolve()
    snapshot_path = Path(artifacts["snapshot_manifest"]["path"]).resolve()
    registry_sha = sha256_file(registry_path)
    snapshot_sha = sha256_file(snapshot_path)
    if "pilot_ledger" in artifacts:
        _validate_ledger(
            Path(artifacts["pilot_ledger"]["path"]),
            population_id="pilot_eval_13",
            source_registry_sha256=registry_sha,
            snapshot_sha256=snapshot_sha,
        )
    if "formal_ledger" in artifacts:
        _validate_ledger(
            Path(artifacts["formal_ledger"]["path"]),
            population_id="formal_test_13",
            source_registry_sha256=registry_sha,
            snapshot_sha256=snapshot_sha,
        )
    if "smoke_gate" in artifacts:
        load_and_validate_smoke_gate(artifacts["smoke_gate"]["path"])
    pilot = load_and_validate_scoped_release(artifacts["pilot_release"]["path"])
    formal = None
    if "formal_release" in artifacts:
        formal = load_and_validate_scoped_release(
            artifacts["formal_release"]["path"]
        )
    scoring_protocols: list[dict[str, Any]] = []
    if "pilot_results_manifest" in artifacts:
        scoring_protocols.append(
            _validate_results_manifest(
                Path(artifacts["pilot_results_manifest"]["path"]),
                scope_path=scope_path,
                population_id="pilot_eval_13",
            )
        )
    if "formal_results_manifest" in artifacts:
        scoring_protocols.append(
            _validate_results_manifest(
                Path(artifacts["formal_results_manifest"]["path"]),
                scope_path=scope_path,
                population_id="formal_test_13",
            )
        )
    if not scoring_protocols:
        raise ValueError("Workflow lacks a validated scoring protocol")
    for scoring_protocol in scoring_protocols[1:]:
        _require_matching_scoring_protocols(
            scoring_protocols[0],
            scoring_protocol,
        )
    if (
        manifest.get("scoring_protocol_signature") != scoring_protocols[0]
        or manifest.get("embedding_model_sha256")
        != scoring_protocols[0]["embedding"]["artifact_sha256"]
    ):
        raise ValueError("Workflow scoring-protocol binding mismatch")
    if formal is not None and (
        formal.get("minutes_model_sha256") != pilot.get("minutes_model_sha256")
        or formal.get("minutes_tokenizer_sha256")
        != pilot.get("minutes_tokenizer_sha256")
        or formal.get("analysis_model_sha256")
        != pilot.get("analysis_model_sha256")
        or formal.get("analysis_tokenizer_sha256")
        != pilot.get("analysis_tokenizer_sha256")
    ):
        raise ValueError("Workflow Pilot/Formal generation artifacts differ")
    if (
        manifest.get("minutes_model_sha256") != pilot.get("minutes_model_sha256")
        or manifest.get("minutes_tokenizer_sha256")
        != pilot.get("minutes_tokenizer_sha256")
        or manifest.get("analysis_model_sha256")
        != pilot.get("analysis_model_sha256")
        or manifest.get("analysis_tokenizer_sha256")
        != pilot.get("analysis_tokenizer_sha256")
    ):
        raise ValueError("Workflow generation-artifact binding mismatch")
    if manifest.get("experiment_id") != scope["experiment_id"]:
        raise ValueError("Workflow experiment identity mismatch")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seal the legacy-six scoped two-stage generation workflow."
    )
    parser.add_argument("--validate-workflow")
    parser.add_argument("--run-id")
    parser.add_argument("--mode", choices=("pilot", "formal", "all"))
    parser.add_argument("--workflow-root")
    parser.add_argument("--experiment-config")
    parser.add_argument("--source-registry")
    parser.add_argument("--snapshot-manifest")
    parser.add_argument("--smoke-gate")
    parser.add_argument("--pilot-ledger")
    parser.add_argument("--pilot-release")
    parser.add_argument("--formal-ledger")
    parser.add_argument("--formal-release")
    parser.add_argument("--reused-pilot-analysis-manifest")
    parser.add_argument("--reused-pilot-analysis-output")
    parser.add_argument("--reused-pilot-projection-manifest")
    parser.add_argument("--reused-pilot-projection-output")
    parser.add_argument("--pilot-results-manifest")
    parser.add_argument("--formal-results-manifest")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.validate_workflow:
        other_values = [
            value
            for name, value in vars(args).items()
            if name != "validate_workflow" and value is not None
        ]
        if other_values:
            parser.error("--validate-workflow cannot be combined with build args")
        manifest = load_and_validate_scoped_workflow(args.validate_workflow)
        print(f"{manifest['run_id']}:{manifest['mode']}:complete")
        return
    required = {
        "run_id": args.run_id,
        "mode": args.mode,
        "workflow_root": args.workflow_root,
        "experiment_config": args.experiment_config,
        "source_registry": args.source_registry,
        "snapshot_manifest": args.snapshot_manifest,
        "pilot_release": args.pilot_release,
        "output": args.output,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error(f"Missing workflow build arguments: {missing}")
    output = write_scoped_workflow_manifest(
        output_file=args.output,
        run_id=args.run_id,
        mode=args.mode,
        workflow_root=args.workflow_root,
        experiment_config_file=args.experiment_config,
        source_registry_file=args.source_registry,
        snapshot_manifest_file=args.snapshot_manifest,
        smoke_gate_file=args.smoke_gate,
        pilot_ledger_file=args.pilot_ledger,
        pilot_release_file=args.pilot_release,
        formal_ledger_file=args.formal_ledger,
        formal_release_file=args.formal_release,
        reused_pilot_analysis_manifest_file=args.reused_pilot_analysis_manifest,
        reused_pilot_analysis_output_file=args.reused_pilot_analysis_output,
        reused_pilot_projection_manifest_file=args.reused_pilot_projection_manifest,
        reused_pilot_projection_output_file=args.reused_pilot_projection_output,
        pilot_results_manifest_file=args.pilot_results_manifest,
        formal_results_manifest_file=args.formal_results_manifest,
    )
    print(output)


if __name__ == "__main__":
    main()
