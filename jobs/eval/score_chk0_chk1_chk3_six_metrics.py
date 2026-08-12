"""Seal six native ``analysis -> Minutes`` metrics for chk0/chk1/chk3-cp318.

This command never generates text.  It deep-validates three already-sealed N12
generation runs, independently recomputes the preregistered generation gate,
and scores eligible ``answer -> reference_minutes`` pairs with the pinned local
BERTScore and MPNet backends.  A row failing any generation component receives
zero for both semantic metrics.  No weighted or composite score is produced.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.eval import eval_checkpoint_generation as semantic_eval
from jobs.eval import eval_chk3_native_three_model as native_eval
from jobs.eval import score_chk3_native_checkpoint_sweep_semantic as shared
from open_r1.validator.loo_generation_spec import seal_manifest, validate_manifest_integrity


SCHEMA_VERSION = "chk0-chk1-chk3-six-metrics-v1"
ROW_SCHEMA_VERSION = "chk0-chk1-chk3-six-metrics-row-v1"
EVALUATION_ID = "chk3-native-analysis-to-minutes-chk0-chk1-chk3-cp318-n12-v1"
EXPECTED_CASES = 12
EXPECTED_NON_IDENTITY_CASES = 11
RUN_ORDER = ("chk0", "chk1", "chk3")
STAGE_BY_RUN = {run_id: run_id for run_id in RUN_ORDER}
COHORTS = ("all_n12", "non_identity_n11")
SIX_METRICS = (
    "structure_delivery",
    "numeric_fidelity",
    "date_fidelity",
    "degeneration_free",
    "bertscore_f1",
    "mpnet_cosine",
)
_CP_RE = re.compile(r"(?:^|[-_])cp(\d+)(?=$|[-_])")


class SixMetricError(RuntimeError):
    """A sealed input or scoring invariant failed."""


def _file_binding(path: Path, *, sealed: bool = False) -> dict[str, Any]:
    try:
        return shared._file_binding(path, sealed=sealed)
    except Exception as exc:
        raise SixMetricError(str(exc)) from exc


def _model_binding(run: Mapping[str, Any]) -> dict[str, Any]:
    model = run.get("manifest", {}).get("model")
    if not isinstance(model, Mapping):
        raise SixMetricError("run has no sealed model binding")
    path = model.get("path")
    files = model.get("files")
    if not isinstance(path, str) or not path or not isinstance(files, Mapping) or not files:
        raise SixMetricError("sealed model binding is incomplete")
    normalized: dict[str, str] = {}
    for name, digest in files.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise SixMetricError("sealed model file hash inventory is malformed")
        normalized[name] = digest
    canonical = native_eval.native_probe.common_probe.canonical_json(normalized)
    return {
        "path": path,
        "files": normalized,
        "file_count": len(normalized),
        "file_hash_inventory_sha256": native_eval.native_probe.common_probe.sha256_text(
            canonical
        ),
        "hash_binding_role": "historical model fingerprint sealed by generation run",
    }


def _row_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("sample_id"),
        row.get("length_bucket"),
        row.get("source_prompt_sha256"),
        row.get("source_analysis_sha256"),
        row.get("reference_minutes_sha256"),
        bool(row.get("normalized_identity")),
    )


def load_runs(
    *,
    chk0_manifest: Path,
    chk1_manifest: Path,
    chk3_manifest: Path,
    sample_manifest: Path,
    sample_manifest_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Deep-validate exactly chk0, chk1, and exact-merged chk3 checkpoint 318."""

    manifest_paths = {
        "chk0": chk0_manifest,
        "chk1": chk1_manifest,
        "chk3": chk3_manifest,
    }
    resolved = [path.expanduser().resolve() for path in manifest_paths.values()]
    if len(set(resolved)) != len(RUN_ORDER):
        raise SixMetricError("the three run manifests must be distinct")

    runs: dict[str, dict[str, Any]] = {}
    for run_id in RUN_ORDER:
        try:
            run = native_eval.load_and_validate_run(
                manifest_paths[run_id],
                expected_stage_id=STAGE_BY_RUN[run_id],
                sample_manifest_path=sample_manifest,
                sample_manifest_sha256=sample_manifest_sha256,
            )
        except Exception as exc:
            raise SixMetricError(f"{run_id} deep validation failed: {exc}") from exc
        run = dict(run)
        run["run_id"] = run_id
        runs[run_id] = run

    reference = runs["chk0"]
    reference_rows = reference.get("results")
    reference_contract = reference.get("generation_contract")
    if not isinstance(reference_rows, list) or len(reference_rows) != EXPECTED_CASES:
        raise SixMetricError("formal scorecard requires a complete chk0 N12 run")
    reference_identity = [_row_identity(row) for row in reference_rows]
    if sum(bool(row.get("normalized_identity")) for row in reference_rows) != 1:
        raise SixMetricError("formal N12 must contain exactly one identity row")

    for run_id, run in runs.items():
        rows = run.get("results")
        if not isinstance(rows, list) or len(rows) != EXPECTED_CASES:
            raise SixMetricError(f"{run_id} is not a complete N12 run")
        if run.get("generation_contract") != reference_contract:
            raise SixMetricError(f"generation contract drift in {run_id}")
        if [_row_identity(row) for row in rows] != reference_identity:
            raise SixMetricError(f"sample/reference drift in {run_id}")
        if run.get("manifest", {}).get("adapter") is not None:
            raise SixMetricError(f"{run_id} must be evaluated as a merged model")

    chk3 = runs["chk3"]
    label = chk3.get("model_label")
    if not isinstance(label, str) or "exact-merged" not in label:
        raise SixMetricError("chk3 must be the exact-merged checkpoint-318 run")
    steps = {int(value) for value in _CP_RE.findall(label)}
    if steps != {318}:
        raise SixMetricError("chk3 model label must bind exactly checkpoint 318")
    return runs


def _gate(row: Mapping[str, Any]) -> dict[str, Any]:
    """Independently recompute the four generation metrics from sealed text/tokens."""

    try:
        audit = shared._recompute_core_hard_gate(row)
    except Exception as exc:
        raise SixMetricError(f"generation gate recomputation failed: {exc}") from exc
    metrics = {
        "structure_delivery": bool(audit["delivery_valid"])
        and bool(audit["native_structure_valid"]),
        "numeric_fidelity": bool(audit["numeric_multiset_preserved"]),
        "date_fidelity": bool(audit["date_set_preserved"]),
        "degeneration_free": bool(audit["degeneration_free"]),
    }
    preregistered = all(metrics.values())
    if preregistered is not bool(audit["preregistered_core_valid"]):
        raise SixMetricError("four generation metrics disagree with preregistered core")
    return {
        "metrics": metrics,
        "preregistered_core_valid": preregistered,
        "preregistered_core_failures": list(audit["preregistered_core_failures"]),
        "delivery_valid": bool(audit["delivery_valid"]),
        "native_structure_valid": bool(audit["native_structure_valid"]),
    }


def _semantic_scores(
    *,
    work_items: Sequence[tuple[str, Mapping[str, Any]]],
    bert: Any,
    mpnet: Any,
) -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, Any]]:
    try:
        values, audit = shared._score_semantic_pairs(
            work_items=work_items, bert=bert, mpnet=mpnet
        )
    except Exception as exc:
        raise SixMetricError(str(exc)) from exc
    scores: dict[tuple[str, str], dict[str, float]] = {}
    for (run_id, row), item in zip(work_items, values, strict=True):
        bert_f1 = float(item["bertscore_f1"])
        cosine = float(item["mpnet_cosine"])
        if not math.isfinite(bert_f1) or not math.isfinite(cosine):
            raise SixMetricError("semantic scorer returned a non-finite value")
        scores[(run_id, str(row["sample_id"]))] = {
            "bertscore_f1": bert_f1,
            "mpnet_cosine": cosine,
        }
    return scores, audit


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise SixMetricError("cannot summarize an empty metric cohort")
    return sum(values) / len(values)


def _summary(rows: Sequence[Mapping[str, Any]], *, cohort: str) -> dict[str, Any]:
    selected = list(rows)
    if cohort == "non_identity_n11":
        selected = [row for row in selected if not bool(row["normalized_identity"])]
    expected = EXPECTED_CASES if cohort == "all_n12" else EXPECTED_NON_IDENTITY_CASES
    if len(selected) != expected:
        raise SixMetricError(f"{cohort} cohort size drift")
    failure_counts: Counter[str] = Counter()
    for row in selected:
        failure_counts.update(row["preregistered_core_failures"])
    metric_values = {
        metric: [float(row["six_metrics"][metric]) for row in selected]
        for metric in SIX_METRICS
    }
    return {
        "cases": len(selected),
        "core_valid_cases": sum(bool(row["preregistered_core_valid"]) for row in selected),
        "semantic_zero_penalty_cases": sum(
            not bool(row["preregistered_core_valid"]) for row in selected
        ),
        "sample_ids": [row["sample_id"] for row in selected],
        "core_failure_counts": dict(sorted(failure_counts.items())),
        "six_metrics": {
            metric: {
                "score": _mean(values),
                "numerator": sum(values),
                "denominator": len(values),
                "unit": (
                    "case_pass_rate"
                    if metric in SIX_METRICS[:4]
                    else "invalid_zero_penalized_mean_similarity"
                ),
            }
            for metric, values in metric_values.items()
        },
    }


def build_scorecard(
    *,
    runs: Mapping[str, Mapping[str, Any]],
    bert: Any,
    mpnet: Any,
    semantic_provenance: Mapping[str, Any],
    sample_manifest_binding: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
    execution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if tuple(runs) != RUN_ORDER:
        raise SixMetricError(f"run inventory/order must be exactly {list(RUN_ORDER)}")

    gate_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    work_items: list[tuple[str, Mapping[str, Any]]] = []
    for run_id in RUN_ORDER:
        rows = runs[run_id].get("results")
        if not isinstance(rows, list) or len(rows) != EXPECTED_CASES:
            raise SixMetricError(f"{run_id} does not contain N12")
        for row in rows:
            sample_id = str(row.get("sample_id"))
            gate = _gate(row)
            gate_by_key[(run_id, sample_id)] = gate
            if gate["preregistered_core_valid"]:
                work_items.append((run_id, row))

    semantic, backend_audit = _semantic_scores(
        work_items=work_items, bert=bert, mpnet=mpnet
    )
    backend_audit["pair_order"] = [
        {
            "run_id": run_id,
            "sample_id": row["sample_id"],
            "answer_sha256": row["answer_sha256"],
            "reference_minutes_sha256": row["reference_minutes_sha256"],
        }
        for run_id, row in work_items
    ]

    row_scores: list[dict[str, Any]] = []
    by_run: dict[str, list[dict[str, Any]]] = {run_id: [] for run_id in RUN_ORDER}
    for run_id in RUN_ORDER:
        run = runs[run_id]
        for row_number, row in enumerate(run["results"], start=1):
            sample_id = str(row["sample_id"])
            gate = gate_by_key[(run_id, sample_id)]
            raw = semantic.get((run_id, sample_id))
            eligible = bool(gate["preregistered_core_valid"])
            if eligible is not (raw is not None):
                raise SixMetricError("semantic eligibility/result mapping drift")
            semantic_values = raw or {"bertscore_f1": 0.0, "mpnet_cosine": 0.0}
            six = {
                **{name: float(value) for name, value in gate["metrics"].items()},
                **semantic_values,
            }
            output_row = {
                "schema_version": ROW_SCHEMA_VERSION,
                "run_id": run_id,
                "stage_id": run["stage_id"],
                "model_label": run["model_label"],
                "source_generation_row": row_number,
                "sample_id": sample_id,
                "length_bucket": row["length_bucket"],
                "normalized_identity": bool(row["normalized_identity"]),
                "answer_sha256": row["answer_sha256"],
                "reference_minutes_sha256": row["reference_minutes_sha256"],
                "generation_metrics": gate["metrics"],
                "delivery_valid": gate["delivery_valid"],
                "native_structure_valid": gate["native_structure_valid"],
                "preregistered_core_valid": eligible,
                "preregistered_core_failures": gate[
                    "preregistered_core_failures"
                ],
                "semantic_eligible": eligible,
                "semantic_zero_penalty_applied": not eligible,
                "raw_semantic_metrics": raw,
                "six_metrics": six,
            }
            row_scores.append(output_row)
            by_run[run_id].append(output_row)

    summaries = {
        run_id: {
            "stage_id": runs[run_id]["stage_id"],
            "model_label": runs[run_id]["model_label"],
            "cohorts": {
                cohort: _summary(by_run[run_id], cohort=cohort)
                for cohort in COHORTS
            },
        }
        for run_id in RUN_ORDER
    }
    run_inputs = {
        run_id: {
            "run_manifest": dict(runs[run_id]["manifest_binding"]),
            "generations": dict(
                runs[run_id]["manifest"]["artifacts"]["generations"]
            ),
            "model": _model_binding(runs[run_id]),
            "adapter": runs[run_id]["manifest"].get("adapter"),
        }
        for run_id in RUN_ORDER
    }
    return seal_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "evaluation_id": EVALUATION_ID,
            "task_contract_id": native_eval.TASK_CONTRACT_ID,
            "operation": "six_metric_scoring_only_no_generation",
            "selected_checkpoint": {
                "stage_id": "chk3",
                "checkpoint_step": 318,
                "representation": "exact_merged",
            },
            "run_order": list(RUN_ORDER),
            "sample_manifest": dict(sample_manifest_binding),
            "scoring_contract": {
                "candidate_text": "answer",
                "reference_text": "reference_minutes",
                "metric_order": list(SIX_METRICS),
                "generation_metrics": list(SIX_METRICS[:4]),
                "semantic_metrics": list(SIX_METRICS[4:]),
                "semantic_eligibility": "independently_recomputed_preregistered_core_valid",
                "invalid_semantic_policy": "bertscore_f1_and_mpnet_cosine_fixed_to_zero",
                "cohorts": {
                    "all_n12": EXPECTED_CASES,
                    "non_identity_n11": EXPECTED_NON_IDENTITY_CASES,
                },
                "weighted_composite_score_authorized": False,
                "statistical_inference_authorized": False,
            },
            "semantic_models": dict(semantic_provenance),
            "semantic_backend_audit": backend_audit,
            "execution": dict(execution or {}),
            "sources": dict(source_bindings),
            "run_inputs": run_inputs,
            "row_scores": row_scores,
            "summaries": summaries,
            "limitations": [
                "N12 is a deterministic diagnostic sample, not a population inference set.",
                "References are synthetic Minutes-style teacher targets, not official Minutes.",
                "The non-identity N11 view excludes the single normalized-identity row.",
                "BERTScore and MPNet measure reference similarity, not factual correctness.",
                "No weighted composite score is calculated or authorized.",
            ],
        }
    )


def score(
    *,
    chk0_manifest: Path,
    chk1_manifest: Path,
    chk3_manifest: Path,
    sample_manifest: Path,
    sample_manifest_sha256: str,
    semantic_manifest: Path,
    output: Path,
    semantic_device: str,
    semantic_batch_size: int,
) -> dict[str, Any]:
    output = output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise SixMetricError(f"refusing to overwrite scorecard: {output}")
    if semantic_device.startswith("cuda"):
        visible = [
            value.strip()
            for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if value.strip()
        ]
        if semantic_device != "cuda:0" or visible != ["0"]:
            raise SixMetricError(
                "CUDA scoring requires CUDA_VISIBLE_DEVICES=0 and --semantic-device cuda:0"
            )

    runs = load_runs(
        chk0_manifest=chk0_manifest,
        chk1_manifest=chk1_manifest,
        chk3_manifest=chk3_manifest,
        sample_manifest=sample_manifest,
        sample_manifest_sha256=sample_manifest_sha256,
    )
    source_bindings = {
        "six_metric_scorer": _file_binding(Path(__file__)),
        "shared_gate_and_semantic_contract": _file_binding(Path(shared.__file__)),
        "native_run_validator": _file_binding(Path(native_eval.__file__)),
        "semantic_backend_implementation": _file_binding(Path(semantic_eval.__file__)),
    }
    try:
        bert, mpnet, semantic_provenance = shared.load_formal_semantic_backends(
            semantic_manifest_path=semantic_manifest,
            batch_size=semantic_batch_size,
            device=semantic_device,
        )
    except Exception as exc:
        raise SixMetricError(str(exc)) from exc

    sample_binding = _file_binding(sample_manifest, sealed=True)
    if sample_binding["sha256"] != sample_manifest_sha256:
        raise SixMetricError("sample manifest SHA-256 differs from CLI binding")
    result = build_scorecard(
        runs=runs,
        bert=bert,
        mpnet=mpnet,
        semantic_provenance=semantic_provenance,
        sample_manifest_binding=sample_binding,
        source_bindings=source_bindings,
        execution={
            "semantic_device": semantic_device,
            "semantic_batch_size": semantic_batch_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "python_executable": sys.executable,
            "network_at_scoring_time": False,
        },
    )
    for label, binding in source_bindings.items():
        shared._assert_file_binding_unchanged(binding, label=label)
    shared._assert_run_sources_unchanged(runs)
    shared._assert_file_binding_unchanged(
        semantic_provenance["binding"], label="semantic model manifest"
    )
    if _file_binding(sample_manifest, sealed=True) != sample_binding:
        raise SixMetricError("sample manifest changed during scoring")
    try:
        shared._write_new_sealed_json(output, result)
    except Exception as exc:
        raise SixMetricError(str(exc)) from exc
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
    parser.add_argument("--semantic-device", default="cuda:0")
    parser.add_argument("--semantic-batch-size", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = score(
            chk0_manifest=args.chk0_manifest,
            chk1_manifest=args.chk1_manifest,
            chk3_manifest=args.chk3_manifest,
            sample_manifest=args.sample_manifest,
            sample_manifest_sha256=args.sample_manifest_sha256,
            semantic_manifest=args.semantic_manifest,
            output=args.output,
            semantic_device=args.semantic_device,
            semantic_batch_size=args.semantic_batch_size,
        )
    except SixMetricError as exc:
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
