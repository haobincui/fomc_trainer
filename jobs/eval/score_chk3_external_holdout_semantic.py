"""Score pinned BERTScore and MPNet on a sealed external CHK3 smoke run.

This scorer is intentionally separate from the checkpoint-selection N12 scorer:
the external 1993--2008 sample contains no analysis/reference identity rows.  It
deep-validates the three native ``analysis -> Minutes`` generation artifacts,
scores every answer against its sealed reference, and reports both raw and
hard-gate-zero-penalized views.  Semantic similarity never overrides a failed
generation/fidelity gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jobs.eval import eval_checkpoint_generation as semantic_eval
from jobs.eval import eval_chk3_native_three_model as native_eval
from jobs.eval import score_chk3_native_checkpoint_sweep_semantic as shared
from open_r1.validator.loo_generation_spec import seal_manifest, validate_manifest_integrity


SCHEMA_VERSION = "chk3-external-holdout-three-model-semantic-v1"
ROW_SCHEMA_VERSION = "chk3-external-holdout-semantic-row-v1"
RUN_ORDER = ("chk0", "chk1", "chk3")
EXPECTED_CASES = 12


class ExternalSemanticError(RuntimeError):
    """A sealed-input or scoring invariant failed."""


def _binding(path: Path, *, sealed: bool = False) -> dict[str, Any]:
    try:
        return shared._file_binding(path, sealed=sealed)
    except Exception as exc:  # pragma: no cover - normalized boundary
        raise ExternalSemanticError(str(exc)) from exc


def _row_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("sample_id"),
        row.get("length_bucket"),
        row.get("source_prompt_sha256"),
        row.get("source_analysis_sha256"),
        row.get("reference_minutes_sha256"),
        bool(row.get("normalized_identity")),
    )


def _load_runs(
    *, manifests: Mapping[str, Path], sample_manifest: Path, sample_sha256: str
) -> dict[str, dict[str, Any]]:
    if tuple(manifests) != RUN_ORDER:
        raise ExternalSemanticError("run order must be chk0, chk1, chk3")
    runs: dict[str, dict[str, Any]] = {}
    for stage in RUN_ORDER:
        try:
            run = native_eval.load_and_validate_run(
                manifests[stage],
                expected_stage_id=stage,
                sample_manifest_path=sample_manifest,
                sample_manifest_sha256=sample_sha256,
            )
        except Exception as exc:
            raise ExternalSemanticError(f"{stage} deep validation failed: {exc}") from exc
        runs[stage] = dict(run)

    reference_rows = runs["chk0"].get("results")
    if not isinstance(reference_rows, list) or len(reference_rows) != EXPECTED_CASES:
        raise ExternalSemanticError("chk0 must contain exactly 12 rows")
    identities = [_row_identity(row) for row in reference_rows]
    if any(item[-1] for item in identities):
        raise ExternalSemanticError("external smoke must contain zero normalized identities")
    contract = runs["chk0"].get("generation_contract")
    for stage, run in runs.items():
        rows = run.get("results")
        if not isinstance(rows, list) or len(rows) != EXPECTED_CASES:
            raise ExternalSemanticError(f"{stage} must contain exactly 12 rows")
        if [_row_identity(row) for row in rows] != identities:
            raise ExternalSemanticError(f"paired sample drift in {stage}")
        if run.get("generation_contract") != contract:
            raise ExternalSemanticError(f"generation contract drift in {stage}")
        if run.get("manifest", {}).get("adapter") is not None:
            raise ExternalSemanticError(f"{stage} must be a merged-model run")
    label = str(runs["chk3"].get("model_label", ""))
    if "cp318" not in label or "exact-merged" not in label:
        raise ExternalSemanticError("chk3 label must bind exact-merged cp318")
    return runs


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if bool(row["core_valid"])]
    return {
        "cases": len(rows),
        "core_valid_cases": len(eligible),
        "zero_penalty_cases": len(rows) - len(eligible),
        "bertscore_f1": {
            "raw_all_mean": _mean([float(row["bertscore_f1_raw"]) for row in rows]),
            "hard_gate_zero_mean": _mean(
                [float(row["bertscore_f1_zero_penalized"]) for row in rows]
            ),
            "valid_only_mean": _mean(
                [float(row["bertscore_f1_raw"]) for row in eligible]
            ),
            "valid_only_n": len(eligible),
        },
        "mpnet_cosine": {
            "raw_all_mean": _mean([float(row["mpnet_cosine_raw"]) for row in rows]),
            "hard_gate_zero_mean": _mean(
                [float(row["mpnet_cosine_zero_penalized"]) for row in rows]
            ),
            "valid_only_mean": _mean(
                [float(row["mpnet_cosine_raw"]) for row in eligible]
            ),
            "valid_only_n": len(eligible),
        },
    }


def score(
    *,
    manifests: Mapping[str, Path],
    sample_manifest: Path,
    sample_sha256: str,
    semantic_manifest: Path,
    output: Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    output = output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ExternalSemanticError(f"refusing to overwrite: {output}")
    if device.startswith("cuda"):
        visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
        if device != "cuda:0" or visible != ["0"]:
            raise ExternalSemanticError("CUDA scoring requires CUDA_VISIBLE_DEVICES=0 and cuda:0")

    runs = _load_runs(
        manifests=manifests,
        sample_manifest=sample_manifest,
        sample_sha256=sample_sha256,
    )
    sample_binding = _binding(sample_manifest, sealed=True)
    if sample_binding["sha256"] != sample_sha256:
        raise ExternalSemanticError("sample manifest hash differs from CLI binding")
    try:
        bert, mpnet, semantic_provenance = shared.load_formal_semantic_backends(
            semantic_manifest_path=semantic_manifest,
            batch_size=batch_size,
            device=device,
        )
    except Exception as exc:
        raise ExternalSemanticError(str(exc)) from exc

    work: list[tuple[str, Mapping[str, Any]]] = []
    gates: dict[tuple[str, str], dict[str, Any]] = {}
    for stage in RUN_ORDER:
        for row in runs[stage]["results"]:
            sample_id = str(row["sample_id"])
            try:
                gate = shared._recompute_core_hard_gate(row)
            except Exception as exc:
                raise ExternalSemanticError(f"gate recomputation failed: {exc}") from exc
            gates[(stage, sample_id)] = gate
            work.append((stage, row))

    try:
        semantic_values, backend_audit = shared._score_semantic_pairs(
            work_items=work, bert=bert, mpnet=mpnet
        )
    except Exception as exc:
        raise ExternalSemanticError(str(exc)) from exc
    if len(semantic_values) != EXPECTED_CASES * len(RUN_ORDER):
        raise ExternalSemanticError("semantic result cardinality drift")

    output_rows: list[dict[str, Any]] = []
    by_stage: dict[str, list[dict[str, Any]]] = {stage: [] for stage in RUN_ORDER}
    for (stage, row), values in zip(work, semantic_values, strict=True):
        sample_id = str(row["sample_id"])
        gate = gates[(stage, sample_id)]
        bert_f1 = float(values["bertscore_f1"])
        cosine = float(values["mpnet_cosine"])
        if not math.isfinite(bert_f1) or not math.isfinite(cosine):
            raise ExternalSemanticError("non-finite semantic value")
        core_valid = bool(gate["preregistered_core_valid"])
        item = {
            "schema_version": ROW_SCHEMA_VERSION,
            "stage_id": stage,
            "model_label": runs[stage]["model_label"],
            "sample_id": sample_id,
            "length_bucket": row["length_bucket"],
            "normalized_identity": bool(row["normalized_identity"]),
            "answer_sha256": row["answer_sha256"],
            "reference_minutes_sha256": row["reference_minutes_sha256"],
            "core_valid": core_valid,
            "core_failures": list(gate["preregistered_core_failures"]),
            "bertscore_f1_raw": bert_f1,
            "mpnet_cosine_raw": cosine,
            "bertscore_f1_zero_penalized": bert_f1 if core_valid else 0.0,
            "mpnet_cosine_zero_penalized": cosine if core_valid else 0.0,
        }
        output_rows.append(item)
        by_stage[stage].append(item)

    sources = {
        "scorer": _binding(Path(__file__)),
        "semantic_backend": _binding(Path(semantic_eval.__file__)),
        "shared_contract": _binding(Path(shared.__file__)),
        "native_validator": _binding(Path(native_eval.__file__)),
    }
    result = seal_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "task_contract_id": native_eval.TASK_CONTRACT_ID,
            "sample_manifest": sample_binding,
            "run_order": list(RUN_ORDER),
            "scoring_contract": {
                "candidate": "answer",
                "reference": "reference_minutes",
                "cases_per_model": EXPECTED_CASES,
                "identity_cases": 0,
                "raw_all": "all 12 sealed rows scored",
                "hard_gate_zero": "core-invalid rows fixed to zero",
                "valid_only": "diagnostic; denominator varies by model",
                "statistical_inference_authorized": False,
            },
            "semantic_models": semantic_provenance,
            "semantic_backend_audit": backend_audit,
            "execution": {
                "device": device,
                "batch_size": batch_size,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "python_executable": sys.executable,
                "network": False,
            },
            "sources": sources,
            "run_manifests": {
                stage: dict(runs[stage]["manifest_binding"]) for stage in RUN_ORDER
            },
            "row_scores": output_rows,
            "summaries": {stage: _summary(by_stage[stage]) for stage in RUN_ORDER},
            "limitations": [
                "N12 is a deterministic smoke sample, not a population inference set.",
                "References are deterministic templated Minutes-style targets, not official Minutes.",
                "BERTScore and MPNet measure reference similarity, not factual correctness.",
                "Raw semantic scores must be read alongside the hard generation/fidelity gates.",
            ],
        }
    )
    for label, binding in sources.items():
        shared._assert_file_binding_unchanged(binding, label=label)
    shared._assert_run_sources_unchanged(runs)
    shared._assert_file_binding_unchanged(
        semantic_provenance["binding"], label="semantic manifest"
    )
    if _binding(sample_manifest, sealed=True) != sample_binding:
        raise ExternalSemanticError("sample manifest changed during scoring")
    try:
        shared._write_new_sealed_json(output, result)
    except Exception as exc:
        raise ExternalSemanticError(str(exc)) from exc
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chk0-manifest", required=True, type=Path)
    parser.add_argument("--chk1-manifest", required=True, type=Path)
    parser.add_argument("--chk3-manifest", required=True, type=Path)
    parser.add_argument("--sample-manifest", required=True, type=Path)
    parser.add_argument("--sample-manifest-sha256", required=True)
    parser.add_argument("--semantic-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = score(
            manifests={
                "chk0": args.chk0_manifest,
                "chk1": args.chk1_manifest,
                "chk3": args.chk3_manifest,
            },
            sample_manifest=args.sample_manifest,
            sample_sha256=args.sample_manifest_sha256,
            semantic_manifest=args.semantic_manifest,
            output=args.output,
            device=args.device,
            batch_size=args.batch_size,
        )
    except ExternalSemanticError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    validate_manifest_integrity(result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.expanduser().resolve()),
                "rows": len(result["row_scores"]),
                "payload_sha256": result["integrity"]["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
