#!/usr/bin/env python3
"""Render the sealed paper-chk2 Tadle-form LM K=10 diagnostic report."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from jobs.eval import paper_chk2_tadle_lm_k10_statistics as statistics_core


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_ROOT = ROOT / (
    "output/evaluation/retrain_v2/"
    "paper_chk2_cp50_tadle_form_lm_ff_futures_k10_v1_20260902"
)
SCHEMA = "paper-chk2-tadle-form-lm-k10-report-v1"
EVALUATION_ID = "paper-chk2-cp50-tadle-form-lm-ff-futures-k10-v1"
STATISTICS_SCHEMA = "paper-chk2-tadle-lm-k10-statistics-v1"
PREPARATION_SCHEMA = "paper-chk2-tadle-lm-k10-preparation-v1"
OVERLAP_SCHEMA = "paper-chk2-tadle-lm-k10-training-overlap-audit-v1"
GENERATION_SCHEMA = "paper-chk2-tadle-lm-k10-all-model-validation-v1"
MODEL_GENERATION_SCHEMA = "paper-chk2-tadle-lm-k10-model-validation-v1"
BACKEND_ID = "wgan_historical_lm_polarity_v1"
CONVENTION = "calendar_month_offset"
SPECIFICATION = "tadle_post2011_interaction"
EXPECTED_NOBS = {"FF1": 2_613, "FF3": 2_612, "FF6": 2_612, "FF12": 2_338}
EXPECTED_MEETINGS = 84
EXPECTED_STANDARDIZATION_MEETINGS = 83
EXPECTED_ESTIMABLE_EVENTS = 82
EXPECTED_TOPICS = 8
EXPECTED_MODELS = 3
REPLICATE_SEEDS = (
    20_260_811,
    21_260_811,
    22_260_811,
    23_260_811,
    24_260_811,
    25_260_811,
    26_260_811,
    27_260_811,
    28_260_811,
    29_260_811,
)
ARMS = ("official", "chk0", "chk1", "paper_chk2_cp50")
GENERATED_ARMS = ("chk0", "chk1", "paper_chk2_cp50")
LABELS = {
    "official": "Official FOMC Minutes",
    "chk0": "Model chk-0",
    "chk1": "Model chk-1 cp200",
    "paper_chk2_cp50": "Model chk-2 cp50",
}
HORIZONS = ("FF1", "FF3", "FF6", "FF12")


class ReportError(RuntimeError):
    """A sealed statistics input or report replay failed validation."""


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ReportError(f"missing/unsafe JSON input: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise ReportError(f"missing/unsafe JSON input: {resolved}")
    try:
        value = json.loads(
            resolved.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ReportError(f"invalid JSON input: {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportError(f"JSON object required: {resolved}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink():
        raise ReportError(f"missing/unsafe JSONL input: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise ReportError(f"missing/unsafe JSONL input: {resolved}")
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ReportError(f"blank JSONL row: {resolved}:{line_number}")
            try:
                value = json.loads(raw, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ReportError(
                    f"invalid JSONL input: {resolved}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ReportError(f"JSONL object required: {resolved}:{line_number}")
            rows.append(value)
    return rows


def binding(
    path: Path, *, relative_to: Path, rows: int | None = None
) -> dict[str, Any]:
    if path.is_symlink():
        raise ReportError(f"cannot bind missing/unsafe artifact: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise ReportError(f"cannot bind missing/unsafe artifact: {resolved}")
    result: dict[str, Any] = {
        "path": str(resolved.relative_to(relative_to.resolve())),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _jsonl_row_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ReportError(f"blank JSONL row: {path}:{line_number}")
            count += 1
    return count


def _verify_record(
    root: Path,
    record: Mapping[str, Any],
    *,
    relative_path: str,
    rows: int | None = None,
) -> Path:
    if not isinstance(record, Mapping) or record.get("path") != relative_path:
        raise ReportError(f"artifact path drift: {relative_path}")
    path = root / relative_path
    observed = binding(path, relative_to=root)
    for key in ("path", "bytes", "sha256"):
        if observed[key] != record.get(key):
            raise ReportError(f"artifact binding drift: {relative_path}/{key}")
    if rows is not None:
        if record.get("rows") != rows or _jsonl_row_count(path) != rows:
            raise ReportError(f"artifact row-count drift: {relative_path}")
    elif "rows" in record:
        raise ReportError(f"unexpected row metadata: {relative_path}")
    return path


def _verify_absolute_input(
    record: Mapping[str, Any], path: Path, *, rows: int | None = None
) -> None:
    if path.is_symlink():
        raise ReportError(f"statistics input symlink is forbidden: {path}")
    resolved = path.resolve()
    if not isinstance(record, Mapping) or record.get("path") != str(resolved):
        raise ReportError(f"statistics input path drift: {resolved}")
    if resolved.stat().st_size != record.get("bytes") or sha256_file(
        resolved
    ) != record.get("sha256"):
        raise ReportError(f"statistics input binding drift: {resolved}")
    if rows is not None and (
        record.get("rows") != rows or _jsonl_row_count(resolved) != rows
    ):
        raise ReportError(f"statistics input row-count drift: {resolved}")


def _require_all_true(value: Any, label: str) -> None:
    if (
        not isinstance(value, Mapping)
        or not value
        or not all(item is True for item in value.values())
    ):
        raise ReportError(f"{label} gates are incomplete or failed")


def _validate_manifests(root: Path) -> dict[str, Any]:
    preparation_path = root / "preparation/evaluation_manifest.json"
    overlap_path = root / "preparation/training_overlap_audit.json"
    generation_path = root / "generation/validation.json"
    sentiment_path = root / "sentiment/manifest.json"
    estimate_path = root / "estimation/estimate_receipt.json"
    statistics_path = root / "estimation/statistics_manifest.json"
    preparation = read_json(preparation_path)
    overlap = read_json(overlap_path)
    generation = read_json(generation_path)
    try:
        sentiment_manifest = statistics_core.verify_committed_phase(
            root, sentiment_path, phase="sentiment"
        )
    except (OSError, ValueError, KeyError, statistics_core.StatisticsError) as exc:
        raise ReportError(f"sentiment phase validation failed: {exc}") from exc
    estimate = read_json(estimate_path)
    statistics_manifest = read_json(statistics_path)

    if (
        preparation.get("schema_version") != PREPARATION_SCHEMA
        or preparation.get("status") != "prepared"
        or preparation.get("immutable") is not True
        or preparation.get("evaluation_id") != EVALUATION_ID
    ):
        raise ReportError("preparation manifest contract drift")
    population = preparation.get("population") or {}
    expected_population = {
        "meeting_union": EXPECTED_MEETINGS,
        "estimable_current_events": EXPECTED_ESTIMABLE_EVENTS,
        "standardization_meetings": EXPECTED_STANDARDIZATION_MEETINGS,
        "topics_per_meeting": EXPECTED_TOPICS,
        "atomic_prompts": EXPECTED_MEETINGS * EXPECTED_TOPICS,
    }
    if any(population.get(key) != value for key, value in expected_population.items()):
        raise ReportError("preparation population drift")
    generation_contract = preparation.get("generation") or {}
    replicate_seeds = generation_contract.get("replicate_seeds")
    if tuple(replicate_seeds or ()) != REPLICATE_SEEDS:
        raise ReportError("preparation replicate schedule is missing")
    replicate_count = len(replicate_seeds)
    expected_atomic = (
        EXPECTED_MEETINGS * EXPECTED_TOPICS * replicate_count * EXPECTED_MODELS
    )
    expected_documents = EXPECTED_MEETINGS * replicate_count * EXPECTED_MODELS
    if (
        replicate_count != 10
        or expected_atomic != 20_160
        or expected_documents != 2_520
        or generation_contract.get("replicates") != replicate_count
        or tuple(generation_contract.get("models") or ()) != GENERATED_ARMS
        or generation_contract.get("total_generation_rows") != expected_atomic
        or generation_contract.get("generated_documents") != expected_documents
    ):
        raise ReportError("preparation generation coverage drift")
    if (
        sentiment_manifest.get("schema_version")
        != f"{STATISTICS_SCHEMA}:sentiment-manifest"
        or sentiment_manifest.get("status") != "complete"
        or sentiment_manifest.get("evaluation_id") != EVALUATION_ID
        or sentiment_manifest.get("backend_id") != BACKEND_ID
        or sentiment_manifest.get("replicates") != replicate_count
        or sentiment_manifest.get("atomic_generation_rows") != expected_atomic
        or sentiment_manifest.get("generation_meetings") != EXPECTED_MEETINGS
        or sentiment_manifest.get("estimable_current_meeting_ids")
        != EXPECTED_ESTIMABLE_EVENTS
        or len(sentiment_manifest.get("lag_only_meeting_ids") or ()) != 2
        or sentiment_manifest.get("broader_standardization_roster")
        != EXPECTED_STANDARDIZATION_MEETINGS
        or sentiment_manifest.get("generated_document_rows") != expected_documents
        or sentiment_manifest.get("document_score_rows") != 2_688
        or sentiment_manifest.get("statement_removal_rows") != EXPECTED_MEETINGS
        or sentiment_manifest.get("k_stability_rows") != 30
        or sentiment_manifest.get("formal_documents_lineage_verified") is not True
        or sentiment_manifest.get("unsealed_custom_generated_documents") is not False
    ):
        raise ReportError("sentiment manifest contract drift")
    if (
        estimate.get("schema_version") != f"{STATISTICS_SCHEMA}:estimate-manifest"
        or estimate.get("status") != "complete"
        or estimate.get("evaluation_id") != EVALUATION_ID
        or estimate.get("backend_id") != BACKEND_ID
        or estimate.get("convention") != CONVENTION
        or estimate.get("specification") != SPECIFICATION
        or estimate.get("replicates") != replicate_count
        or estimate.get("atomic_generation_rows") != expected_atomic
        or estimate.get("generated_documents") != expected_documents
        or estimate.get("generation_meetings") != EXPECTED_MEETINGS
        or estimate.get("estimable_current_meeting_ids") != EXPECTED_ESTIMABLE_EVENTS
        or len(estimate.get("lag_only_meeting_ids") or ()) != 2
        or estimate.get("broader_standardization_roster")
        != EXPECTED_STANDARDIZATION_MEETINGS
        or estimate.get("analysis_panel_rows") != 40_700
        or estimate.get("coefficient_rows") != 16
        or estimate.get("leave_one_event_out_rows") != 1_272
        or estimate.get("leave_one_year_out_rows") != 172
    ):
        raise ReportError("estimate receipt contract drift")
    expected_estimate_artifacts = {
        "standardization_scales": ("estimation/standardization_scales.json", None),
        "analysis_panel": ("estimation/analysis_panel.jsonl", 40_700),
        "coefficient_estimates": ("estimation/coefficient_estimates.jsonl", 16),
        "leave_one_event_out": ("estimation/leave_one_event_out.jsonl", 1_272),
        "leave_one_year_out": ("estimation/leave_one_year_out.jsonl", 172),
        "point_diagnostics": ("estimation/point_diagnostics.json", None),
    }
    estimate_artifacts = estimate.get("artifacts") or {}
    if set(estimate_artifacts) != set(expected_estimate_artifacts):
        raise ReportError("estimate artifact inventory drift")
    for name, (relative_path, rows) in expected_estimate_artifacts.items():
        _verify_record(
            root, estimate_artifacts[name], relative_path=relative_path, rows=rows
        )
    prep_inputs = preparation.get("inputs") or {}
    ledger_record = prep_inputs.get("atomic_prompt_ledger") or {}
    overlap_record = prep_inputs.get("training_overlap_audit") or {}
    _verify_absolute_input(
        ledger_record,
        root / "preparation/atomic_prompt_ledger.jsonl",
        rows=EXPECTED_MEETINGS * EXPECTED_TOPICS,
    )
    _verify_absolute_input(overlap_record, overlap_path)

    if (
        overlap.get("schema_version") != OVERLAP_SCHEMA
        or overlap.get("status") != "complete"
    ):
        raise ReportError("training-overlap schema/status drift")
    counts = overlap.get("counts") or {}
    expected_overlap = {
        "train": {"union_84": 47, "standardization_83": 47, "estimable_current_82": 46},
        "validation": {
            "union_84": 0,
            "standardization_83": 0,
            "estimable_current_82": 0,
        },
        "test": {"union_84": 0, "standardization_83": 0, "estimable_current_82": 0},
    }
    if counts != expected_overlap:
        raise ReportError("training-overlap count drift")
    train_meetings = set(
        (overlap.get("overlap_by_split") or {}).get("train", {}).get("union_84") or ()
    )
    if len(train_meetings) != 47:
        raise ReportError("training-overlap meeting inventory drift")

    if (
        generation.get("schema_version") != GENERATION_SCHEMA
        or generation.get("status") != "complete"
        or generation.get("mode") != "formal"
        or generation.get("evaluation_id") != EVALUATION_ID
        or generation.get("generation_rows") != expected_atomic
        or generation.get("rows_per_model") != expected_atomic // EXPECTED_MODELS
        or generation.get("paired_tuples") != expected_atomic // EXPECTED_MODELS
    ):
        raise ReportError("generation validation contract drift")
    models = generation.get("models") or {}
    if set(models) != set(GENERATED_ARMS):
        raise ReportError("generation model inventory drift")
    for arm, record in models.items():
        if (
            record.get("schema_version") != MODEL_GENERATION_SCHEMA
            or record.get("status") != "complete"
            or record.get("mode") != "formal"
            or record.get("model_id") != arm
            or record.get("generation_rows") != expected_atomic // EXPECTED_MODELS
            or record.get("meetings") != EXPECTED_MEETINGS
            or record.get("replicates") != replicate_count
        ):
            raise ReportError(f"generation model validation drift: {arm}")
        _require_all_true(record.get("gates"), f"generation/{arm}")
    _require_all_true(generation.get("gates"), "generation")
    if not isinstance(generation.get("delivery_funnel"), Mapping):
        raise ReportError("generation delivery-funnel contract drift")
    _verify_absolute_input(
        generation.get("combined_generation_rows") or {},
        root / "generation/generation_rows.jsonl",
        rows=expected_atomic,
    )

    if (
        statistics_manifest.get("schema_version")
        != f"{STATISTICS_SCHEMA}:statistics-manifest"
        or statistics_manifest.get("status") != "complete"
        or statistics_manifest.get("evaluation_id") != EVALUATION_ID
        or statistics_manifest.get("backend_id") != BACKEND_ID
        or statistics_manifest.get("convention") != CONVENTION
        or statistics_manifest.get("specification") != SPECIFICATION
        or statistics_manifest.get("replicates") != replicate_count
        or statistics_manifest.get("atomic_generation_rows") != expected_atomic
        or statistics_manifest.get("generated_documents") != expected_documents
        or statistics_manifest.get("bootstrap_draws") != 10_000
        or statistics_manifest.get("bootstrap_seed") != 20_260_902
        or statistics_manifest.get("coefficient_rows") != 16
        or statistics_manifest.get("contrast_rows") != 36
        or statistics_manifest.get("absolute_distance_rows") != 12
        or statistics_manifest.get("distance_gain_rows") != 8
        or statistics_manifest.get("generation_meetings") != EXPECTED_MEETINGS
        or statistics_manifest.get("estimable_current_meeting_ids")
        != EXPECTED_ESTIMABLE_EVENTS
        or statistics_manifest.get("broader_standardization_roster")
        != EXPECTED_STANDARDIZATION_MEETINGS
        or len(statistics_manifest.get("lag_only_meeting_ids") or ()) != 2
        or statistics_manifest.get("exact_tadle_regression_form") is not True
        or statistics_manifest.get("exact_tadle_2022_replication") is not False
        or statistics_manifest.get("interpretation")
        != "mixed-scope historical document-source association diagnostic"
        or not isinstance(statistics_manifest.get("limitations"), list)
        or not statistics_manifest.get("limitations")
    ):
        raise ReportError("statistics manifest contract drift")
    bootstrap_draws = int(statistics_manifest["bootstrap_draws"])
    statistics_inputs = statistics_manifest.get("inputs") or {}
    _verify_absolute_input(
        statistics_inputs.get("estimate_receipt") or {}, estimate_path
    )
    _verify_absolute_input(
        statistics_inputs.get("analysis_panel") or {},
        root / "estimation/analysis_panel.jsonl",
        rows=40_700,
    )
    _verify_absolute_input(
        statistics_inputs.get("coefficient_estimates") or {},
        root / "estimation/coefficient_estimates.jsonl",
        rows=16,
    )
    expected_artifacts = {
        "bootstrap_plan": ("estimation/bootstrap_plan.json", None),
        "bootstrap_plan_arrays": ("estimation/bootstrap_plan_arrays.npz", None),
        "bootstrap_draws": ("estimation/bootstrap_draws.jsonl", bootstrap_draws),
        "model_minus_official_contrasts": (
            "estimation/model_minus_official_contrasts.jsonl",
            36,
        ),
        "absolute_distances": ("estimation/absolute_distances.jsonl", 12),
        "distance_gains": ("estimation/distance_gains.jsonl", 8),
        "validation_report": ("estimation/validation_report.json", None),
    }
    artifacts = statistics_manifest.get("artifacts") or {}
    if set(artifacts) != set(expected_artifacts):
        raise ReportError("statistics artifact inventory drift")
    for name, (relative_path, rows) in expected_artifacts.items():
        _verify_record(root, artifacts[name], relative_path=relative_path, rows=rows)
    plan = read_json(root / "estimation/bootstrap_plan.json")
    if (
        plan.get("schema_version") != f"{STATISTICS_SCHEMA}:bootstrap-plan"
        or plan.get("draws") != bootstrap_draws
        or plan.get("replicate_count") != replicate_count
        or plan.get("plan_sha256") != statistics_manifest.get("plan_sha256")
    ):
        raise ReportError("bootstrap-plan contract drift")
    validation = read_json(root / "estimation/validation_report.json")
    if (
        validation.get("schema_version") != f"{STATISTICS_SCHEMA}:validation-report"
        or validation.get("status") != "passed"
    ):
        raise ReportError("statistics validation report drift")
    _require_all_true(validation.get("checks"), "statistics validation")

    panel = read_jsonl(root / "estimation/analysis_panel.jsonl")
    post_events = {
        (str(row.get("event_meeting_id")), str(row.get("lag_meeting_id")))
        for row in panel
        if row.get("arm") == "official"
        and row.get("horizon_label") == "FF1"
        and row.get("is_minutes_release_event") is True
        and int(row.get("post_2011", 0)) == 1
    }
    if len(post_events) != 30 or any(
        current not in train_meetings and lag not in train_meetings
        for current, lag in post_events
    ):
        raise ReportError("30/30 post-period training-overlap claim drift")
    return {
        "preparation": preparation,
        "overlap": overlap,
        "generation": generation,
        "sentiment": sentiment_manifest,
        "statistics": statistics_manifest,
        "replicate_count": replicate_count,
        "atomic_generation_rows": expected_atomic,
        "generated_documents": expected_documents,
        "bootstrap_draws": bootstrap_draws,
    }


def _fmt(value: float, digits: int = 4, *, signed: bool = True) -> str:
    prefix = "+" if signed else ""
    return f"{float(value):{prefix}.{digits}f}"


def _stars(value: float) -> str:
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return ""


def _lookup(
    rows: Sequence[Mapping[str, Any]], *keys: str
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    output: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        if key in output:
            raise ReportError(f"duplicate report coordinate: {key}")
        output[key] = row
    return output


def _validate_inputs(
    points: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    distances: Sequence[Mapping[str, Any]],
    gains: Sequence[Mapping[str, Any]],
) -> None:
    if (len(points), len(contrasts), len(distances), len(gains)) != (16, 36, 12, 8):
        raise ReportError("statistics row-count contract drift")
    if {(row["horizon_label"], row["arm"]) for row in points} != {
        (horizon, arm) for horizon in HORIZONS for arm in ARMS
    }:
        raise ReportError("coefficient Cartesian product drift")
    for row in points:
        if (
            row.get("schema_version") != f"{STATISTICS_SCHEMA}:coefficient-row"
            or row.get("backend_id") != BACKEND_ID
            or row.get("convention") != CONVENTION
            or row.get("specification") != SPECIFICATION
            or row.get("paper_label") != LABELS.get(str(row.get("arm")))
            or int(row.get("nobs", -1)) != EXPECTED_NOBS[str(row.get("horizon_label"))]
        ):
            raise ReportError("coefficient row contract drift")
        numeric = (
            "beta_pre_basis_points",
            "hc1_se_pre_basis_points",
            "beta_interaction_basis_points",
            "hc1_se_interaction_basis_points",
            "beta_post_basis_points",
            "hc1_se_post_basis_points",
            "r_squared",
            "condition_number",
        )
        if not all(math.isfinite(float(row[field])) for field in numeric):
            raise ReportError("non-finite coefficient row")
        if not math.isclose(
            float(row["beta_post_basis_points"]),
            float(row["beta_pre_basis_points"])
            + float(row["beta_interaction_basis_points"]),
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ReportError("coefficient post identity drift")
    if {(row["estimand"], row["horizon_label"], row["arm"]) for row in contrasts} != {
        (estimand, horizon, arm)
        for estimand in ("pre", "interaction", "post")
        for horizon in HORIZONS
        for arm in GENERATED_ARMS
    }:
        raise ReportError("contrast Cartesian product drift")
    interval_fields = (
        "estimate_basis_points",
        "ci_95_low_basis_points",
        "ci_95_high_basis_points",
    )
    for row in contrasts:
        expected_family = (
            "derived_post_marginal_12"
            if row.get("estimand") == "post"
            else "primitive_pre_and_interaction_24"
        )
        if (
            row.get("schema_version") != f"{STATISTICS_SCHEMA}:model-minus-official-row"
            or row.get("backend_id") != BACKEND_ID
            or row.get("convention") != CONVENTION
            or row.get("specification") != SPECIFICATION
            or row.get("family") != expected_family
            or row.get("paper_label") != LABELS.get(str(row.get("arm")))
            or int(row.get("bootstrap_draws", -1)) != 10_000
        ):
            raise ReportError("model-minus-official row contract drift")
    expected_distance_coordinates = {
        (horizon, arm) for horizon in HORIZONS for arm in GENERATED_ARMS
    }
    if {
        (row.get("horizon_label"), row.get("arm")) for row in distances
    } != expected_distance_coordinates:
        raise ReportError("distance Cartesian product drift")
    for row in distances:
        if (
            row.get("schema_version") != f"{STATISTICS_SCHEMA}:absolute-distance-row"
            or row.get("backend_id") != BACKEND_ID
            or row.get("convention") != CONVENTION
            or row.get("specification") != SPECIFICATION
            or row.get("estimand") != "post"
            or row.get("paper_label") != LABELS.get(str(row.get("arm")))
            or int(row.get("bootstrap_draws", -1)) != 10_000
        ):
            raise ReportError("absolute-distance row contract drift")
    expected_gain_coordinates = {
        (horizon, comparator) for horizon in HORIZONS for comparator in ("chk0", "chk1")
    }
    if {
        (row.get("horizon_label"), row.get("comparator_arm")) for row in gains
    } != expected_gain_coordinates:
        raise ReportError("distance-gain Cartesian product drift")
    for row in gains:
        comparator = str(row.get("comparator_arm"))
        if (
            row.get("schema_version") != f"{STATISTICS_SCHEMA}:distance-gain-row"
            or row.get("backend_id") != BACKEND_ID
            or row.get("convention") != CONVENTION
            or row.get("specification") != SPECIFICATION
            or row.get("family") != "chk2_distance_gains_8"
            or row.get("estimand") != "post"
            or row.get("comparator_label") != LABELS.get(comparator)
            or row.get("target_arm") != "paper_chk2_cp50"
            or row.get("target_label") != LABELS["paper_chk2_cp50"]
            or row.get("positive_means_chk2_closer") is not True
            or int(row.get("bootstrap_draws", -1)) != 10_000
        ):
            raise ReportError("distance-gain row contract drift")
    for row in [*contrasts, *distances, *gains]:
        if not all(math.isfinite(float(row[field])) for field in interval_fields):
            raise ReportError("non-finite interval row")
        low = float(row["ci_95_low_basis_points"])
        high = float(row["ci_95_high_basis_points"])
        if low > high:
            raise ReportError("reversed confidence interval")
    if any(
        float(row["estimate_basis_points"]) < 0
        or float(row["ci_95_low_basis_points"]) < 0
        or float(row["ci_95_high_basis_points"]) < 0
        for row in distances
    ):
        raise ReportError("absolute distance or its interval is negative")
    for row in [*contrasts, *gains]:
        if not all(
            math.isfinite(float(row[field])) and 0.0 <= float(row[field]) <= 1.0
            for field in ("bootstrap_p_raw", "holm_p")
        ):
            raise ReportError("bootstrap/Holm probability outside [0,1]")


def build_post_beta_table(
    points: Sequence[Mapping[str, Any]], contrasts: Sequence[Mapping[str, Any]]
) -> str:
    point = _lookup(points, "horizon_label", "arm")
    post = _lookup(
        [row for row in contrasts if row["estimand"] == "post"],
        "horizon_label",
        "arm",
    )
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \footnotesize",
        r"    \setlength{\tabcolsep}{5pt}",
        r"    \renewcommand{\arraystretch}{1.12}",
        r"    \caption{Post-2011 Marginal Federal Funds Futures Price-Return Coefficients}",
        r"    \label{tab:ch2:tadle_post_beta}",
        r"    \begin{threeparttable}",
        r"    \begin{tabularx}{\textwidth}{@{}l>{\centering\arraybackslash}p{0.09\textwidth}*{4}{>{\centering\arraybackslash}X}@{}}",
        r"        \toprule",
        r"        \textbf{Horizon} & \(\boldsymbol{N}\) & \makecell{\textbf{Official}\\\textbf{Minutes}} & \textit{Model chk-0} & \makecell{\textit{Model chk-1}\\cp200} & \makecell{\textit{Model chk-2}\\cp50} \\",
        r"        \midrule",
    ]
    for horizon in HORIZONS:
        cells: list[str] = []
        for arm in ARMS:
            row = point[(horizon, arm)]
            suffix = ""
            if arm != "official":
                suffix = _stars(float(post[(horizon, arm)]["holm_p"]))
            estimate = _fmt(float(row["beta_post_basis_points"]))
            if suffix:
                estimate += rf"^{{{suffix}}}"
            cells.append(
                rf"\makecell{{$ {estimate} $\\$({float(row['hc1_se_post_basis_points']):.4f})$}}"
            )
        nobs = int(point[(horizon, "official")]["nobs"])
        lines.append(f"        {horizon} & {nobs:,} & " + " & ".join(cells) + r" \\")
        if horizon != HORIZONS[-1]:
            lines.append(r"        \addlinespace")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabularx}",
            r"    \begin{tablenotes}[flushleft]",
            r"        \scriptsize",
            r"        \item \textit{Notes:} Each cell reports $\widehat\beta_{\mathrm{post}}=\widehat\beta_{\mathrm{pre}}+\widehat\beta_{\mathrm{interaction}}$, scaled as basis points of the Federal Funds futures price log return per one-unit constructed news shock. Here one unit refers to the constructed news-shock regressor formed from Minutes and Statement sentiment components standardized separately by source; the constructed regressor is not itself re-standardized to unit variance. This is not a policy-rate change. HC1 standard errors are in parentheses. Stars on generated-text coefficients refer only to paired model-minus-official null-centred bootstrap tests, Holm-adjusted across the 12 post-period comparisons: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.",
            r"    \end{tablenotes}",
            r"    \end{threeparttable}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def build_distance_gain_table(
    distances: Sequence[Mapping[str, Any]], gains: Sequence[Mapping[str, Any]]
) -> str:
    distance = _lookup(distances, "horizon_label", "arm")
    gain = _lookup(gains, "horizon_label", "comparator_arm")

    def distance_cell(row: Mapping[str, Any]) -> str:
        return (
            rf"\makecell{{$ {float(row['estimate_basis_points']):.4f} $\\"
            rf"$[{float(row['ci_95_low_basis_points']):.4f},\,{float(row['ci_95_high_basis_points']):.4f}]$}}"
        )

    def gain_cell(row: Mapping[str, Any]) -> str:
        stars = _stars(float(row["holm_p"]))
        estimate = _fmt(float(row["estimate_basis_points"]))
        if stars:
            estimate += rf"^{{{stars}}}"
        return (
            rf"\makecell{{$ {estimate} $\\"
            rf"$[{_fmt(float(row['ci_95_low_basis_points']))},\,{_fmt(float(row['ci_95_high_basis_points']))}]$}}"
        )

    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \scriptsize",
        r"    \setlength{\tabcolsep}{3pt}",
        r"    \renewcommand{\arraystretch}{1.14}",
        r"    \caption{Post-2011 Coefficient Distances and Direct Model chk-2 Distance Gains}",
        r"    \label{tab:ch2:tadle_beta_distance}",
        r"    \begin{threeparttable}",
        r"    \begin{tabularx}{\textwidth}{@{}l*{5}{>{\centering\arraybackslash}X}@{}}",
        r"        \toprule",
        r"        \textbf{Horizon} & \makecell{$D_{0,O}$\\chk-0} & \makecell{$D_{1,O}$\\chk-1} & \makecell{$D_{2,O}$\\chk-2} & \makecell{$G_{0}$\\chk-2 vs. chk-0} & \makecell{$G_{1}$\\chk-2 vs. chk-1} \\",
        r"        \midrule",
    ]
    for horizon in HORIZONS:
        cells = [distance_cell(distance[(horizon, arm)]) for arm in GENERATED_ARMS]
        cells.extend(
            gain_cell(gain[(horizon, comparator)]) for comparator in ("chk0", "chk1")
        )
        lines.append(f"        {horizon} & " + " & ".join(cells) + r" \\")
        if horizon != HORIZONS[-1]:
            lines.append(r"        \addlinespace")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabularx}",
            r"    \begin{tablenotes}[flushleft]",
            r"        \scriptsize",
            r"        \item \textit{Notes:} $D_{m,O}=|\widehat\beta_{m}^{\mathrm{post}}-\widehat\beta_{O}^{\mathrm{post}}|$. The direct gain is $G_j=D_{j,O}-D_{2,O}$, so a positive value indicates that Model chk-2 is closer to the official-Minutes coefficient than comparator $j$. Brackets are paired percentile-bootstrap intervals computed directly on each distance or gain. Stars appear only on gains and use a separate eight-test Holm family; no signed interval is transformed into a distance interval.",
            r"    \end{tablenotes}",
            r"    \end{threeparttable}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def build_methods_tex(
    *, replicate_count: int, atomic_rows: int, documents: int, draws: int
) -> str:
    template = r"""This diagnostic applies a Tadle-form interaction regression to
historical Federal Funds futures price returns using sentiment extracted from
official and synthetic FOMC Minutes documents.  The generation roster contains
84 current-or-lag meetings required to form 82 estimable news-shock events.
For each meeting, the eight Core8 analyses are rewritten independently and
joined in a frozen order.  Each of Model chk-0, Model chk-1 checkpoint 200, and
Model chk-2 checkpoint 50 generates @@REPLICATES@@ stochastic replicates under a common
prompt contract and paired seed schedule, producing @@ATOMIC_ROWS@@ atomic completions and @@DOCUMENTS@@
eight-section documents.  The previously generated Core8 leave-one-out texts
are not reused.

Document sentiment is the frozen Loughran--McDonald polarity
$\left(P-N\right)/\max\left(1,P+N\right)$.  Official Minutes are scored after removing the passage
that repeats the corresponding Statement.  For source $m$, relative sentiment
is $RZ_{m,t}=Z^M_{m,t}-Z^S_t$, and the news shock is
$NS_{m,t}=RZ_{m,t}-0.368RZ_{m,t-1}$.  The interaction regression is estimated
separately for FF1, FF3, FF6, and FF12 with contemporaneous VIX log change,
the post-August-8-2011 indicator, and calendar-year fixed effects.  Point
estimates use OLS with HC1 standard errors.

Uncertainty is evaluated with @@DRAWS@@ paired, regime-stratified calendar-year
block-bootstrap draws.  Pre- and post-2011 years are sampled separately, 2011
is retained once, and @@REPLICATES@@ within-meeting replicates are resampled with indices
shared across all generated models.  The standardization scales are fixed at
their point-sample values.  Null-centred two-sided bootstrap probabilities are
Holm-adjusted in three prespecified families: 24 pre/interaction
model-minus-official contrasts, 12 post-period contrasts, and eight direct
Model chk-2 distance-gain contrasts.

This is a mixed-scope historical document-source association diagnostic, not
an exact replication of Tadle (2022), a causal market-impact design, an
equivalence test, or a leakage-safe external evaluation.  In particular, all
30 post-period shocks contain a current or lag meeting overlapping the
task-specific Model chk-2 training-meeting roster."""
    return (
        template.replace("@@REPLICATES@@", str(replicate_count))
        .replace("@@ATOMIC_ROWS@@", f"{atomic_rows:,}")
        .replace("@@DOCUMENTS@@", f"{documents:,}")
        .replace("@@DRAWS@@", f"{draws:,}")
        + "\n"
    )


def _markdown_tables(
    points: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    distances: Sequence[Mapping[str, Any]],
    gains: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    point = _lookup(points, "horizon_label", "arm")
    post = _lookup(
        [row for row in contrasts if row["estimand"] == "post"], "horizon_label", "arm"
    )
    distance = _lookup(distances, "horizon_label", "arm")
    gain = _lookup(gains, "horizon_label", "comparator_arm")
    beta = [
        "| Horizon | N | Official | chk0 | chk1 cp200 | chk2 cp50 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    dg = [
        "| Horizon | D0 | D1 | D2 | G0 (vs chk0) | G1 (vs chk1) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS:
        cells = []
        for arm in ARMS:
            row = point[(horizon, arm)]
            suffix = (
                ""
                if arm == "official"
                else _stars(float(post[(horizon, arm)]["holm_p"]))
            )
            cells.append(
                f"{_fmt(row['beta_post_basis_points'])}{suffix} ({float(row['hc1_se_post_basis_points']):.4f})"
            )
        beta.append(
            f"| {horizon} | {int(point[(horizon, 'official')]['nobs']):,} | "
            + " | ".join(cells)
            + " |"
        )
        d_cells = [
            f"{float(distance[(horizon, arm)]['estimate_basis_points']):.4f}"
            for arm in GENERATED_ARMS
        ]
        g_cells = [
            f"{_fmt(gain[(horizon, arm)]['estimate_basis_points'])}{_stars(float(gain[(horizon, arm)]['holm_p']))}"
            for arm in ("chk0", "chk1")
        ]
        dg.append(f"| {horizon} | " + " | ".join([*d_cells, *g_cells]) + " |")
    return "\n".join(beta), "\n".join(dg)


def build_results_tex(
    distances: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    gains: Sequence[Mapping[str, Any]],
) -> str:
    means = {
        arm: statistics.fmean(
            float(row["estimate_basis_points"])
            for row in distances
            if row["arm"] == arm
        )
        for arm in GENERATED_ARMS
    }
    closest = min(means, key=means.__getitem__)
    per_horizon = {
        horizon: min(
            (row for row in distances if row["horizon_label"] == horizon),
            key=lambda row: float(row["estimate_basis_points"]),
        )["arm"]
        for horizon in HORIZONS
    }
    significant_post = sum(
        row["estimand"] == "post" and float(row["holm_p"]) < 0.05 for row in contrasts
    )
    significant_gains = sum(float(row["holm_p"]) < 0.05 for row in gains)
    horizon_text = ", ".join(
        f"{horizon} ({LABELS[str(arm)]})" for horizon, arm in per_horizon.items()
    )
    return (
        "The post-period coefficient comparison is a historical association diagnostic. "
        f"Across the four horizons, the unweighted mean absolute distances are "
        f"{means['chk0']:.4f} for Model chk-0, {means['chk1']:.4f} for Model chk-1, "
        f"and {means['paper_chk2_cp50']:.4f} for Model chk-2; the smallest descriptive mean is "
        f"therefore obtained by {LABELS[closest]}. The closest generated source by horizon is {horizon_text}. "
        f"Of the 12 post-period model-minus-official contrasts, {significant_post} survive their "
        f"prespecified Holm family, while {significant_gains} of the eight direct Model chk-2 "
        "distance-gain contrasts survive the separate gain family. These tests concern coefficient "
        "differences or proximity, not equivalence or causal market effects. Moreover, the post-period "
        "result cannot be interpreted as leakage-safe generalization because all 30 post-period shocks "
        "contain a current or lag meeting represented in the task-specific training roster.\n"
    )


def _build_payloads(root: Path, *, created_at: str) -> dict[str, bytes]:
    verified = _validate_manifests(root)
    statistics_manifest_path = root / "estimation/statistics_manifest.json"
    points = read_jsonl(root / "estimation/coefficient_estimates.jsonl")
    contrasts = read_jsonl(root / "estimation/model_minus_official_contrasts.jsonl")
    distances = read_jsonl(root / "estimation/absolute_distances.jsonl")
    gains = read_jsonl(root / "estimation/distance_gains.jsonl")
    loeo = read_jsonl(root / "estimation/leave_one_event_out.jsonl")
    loyo = read_jsonl(root / "estimation/leave_one_year_out.jsonl")
    point_diagnostics = read_json(root / "estimation/point_diagnostics.json")
    k_stability = read_jsonl(root / "sentiment/k_stability.jsonl")
    generation = verified["generation"]
    _validate_inputs(points, contrasts, distances, gains)

    condition_numbers = [float(row["condition_number"]) for row in points]
    if not all(math.isfinite(value) and value > 0.0 for value in condition_numbers):
        raise ReportError("invalid point condition numbers")
    loo_maxima: dict[str, dict[str, float]] = {}
    for label, rows, expected_schema in (
        ("event", loeo, f"{STATISTICS_SCHEMA}:leave-one-event-out-row"),
        ("year", loyo, f"{STATISTICS_SCHEMA}:leave-one-year-out-row"),
    ):
        if any(row.get("schema_version") != expected_schema for row in rows):
            raise ReportError(f"{label} leave-one-out schema drift")
        loo_maxima[label] = {}
        for estimand in ("pre", "interaction", "post"):
            values = [abs(float(row[f"delta_{estimand}_basis_points"])) for row in rows]
            if not all(math.isfinite(value) for value in values):
                raise ReportError(f"non-finite {label} leave-one-out diagnostic")
            loo_maxima[label][estimand] = max(values)
    if (
        point_diagnostics.get("schema_version")
        != f"{STATISTICS_SCHEMA}:point-diagnostics"
        or float(point_diagnostics.get("point_max_condition_number", math.nan))
        != max(condition_numbers)
        or point_diagnostics.get("max_abs_leave_one_event_delta_basis_points")
        != loo_maxima["event"]
        or point_diagnostics.get("max_abs_leave_one_year_delta_basis_points")
        != loo_maxima["year"]
    ):
        raise ReportError("point diagnostics summary drift")
    if len(k_stability) != len(GENERATED_ARMS) * int(verified["replicate_count"]):
        raise ReportError("K-stability row-count drift")
    stability_summary: dict[str, dict[str, float]] = {}
    for arm in GENERATED_ARMS:
        rows = sorted(
            (row for row in k_stability if row.get("arm") == arm),
            key=lambda row: int(row["k"]),
        )
        if [int(row["k"]) for row in rows] != list(range(1, 11)) or any(
            row.get("schema_version") != f"{STATISTICS_SCHEMA}:k-stability-row"
            for row in rows
        ):
            raise ReportError(f"K-stability inventory drift: {arm}")
        correlations = [float(row["pearson_correlation_with_kmax"]) for row in rows]
        drifts = [float(row["mean_absolute_drift_from_kmax"]) for row in rows]
        if not all(math.isfinite(value) for value in correlations + drifts):
            raise ReportError(f"non-finite K-stability diagnostic: {arm}")
        stability_summary[arm] = {
            "k1_correlation": correlations[0],
            "minimum_correlation": min(correlations),
            "k1_mean_absolute_drift": drifts[0],
            "k10_mean_absolute_drift": drifts[-1],
        }

    beta_md, distance_md = _markdown_tables(points, contrasts, distances, gains)
    methods = build_methods_tex(
        replicate_count=int(verified["replicate_count"]),
        atomic_rows=int(verified["atomic_generation_rows"]),
        documents=int(verified["generated_documents"]),
        draws=int(verified["bootstrap_draws"]),
    )
    results = build_results_tex(distances, contrasts, gains)
    significant = {
        "primitive_pre_interaction": sum(
            row["family"] == "primitive_pre_and_interaction_24"
            and float(row["holm_p"]) < 0.05
            for row in contrasts
        ),
        "post": sum(
            row["family"] == "derived_post_marginal_12" and float(row["holm_p"]) < 0.05
            for row in contrasts
        ),
        "distance_gain": sum(float(row["holm_p"]) < 0.05 for row in gains),
    }
    replicate_count = int(verified["replicate_count"])
    atomic_rows = int(verified["atomic_generation_rows"])
    document_count = int(verified["generated_documents"])
    bootstrap_draws = int(verified["bootstrap_draws"])
    markdown = f"""# Paper chk-2 cp50: Historical Federal Funds Futures Document-Source Diagnostic

Status: complete  
Created: {created_at}

## Design

- 84 current-or-lag generation meetings, 82 estimable events, eight atomic Core8 topics and K={replicate_count}.
- {atomic_rows:,} fresh atomic completions and {document_count:,} assembled documents across chk0, chk1 cp200 and chk2 cp50.
- The Core8 leave-one-out generations were not reused.
- Sentiment uses the frozen Loughran--McDonald polarity dictionary.
- A one-unit news shock means one unit of the constructed NS regressor formed from separately source-standardized Minutes and Statement components; NS itself is not re-standardized to unit variance.
- Inference uses {bootstrap_draws:,} paired, 2011-anchored calendar-year block-bootstrap draws.

## Post-2011 coefficients

{beta_md}

## Distances and direct chk2 gains

{distance_md}

Holm-significant results: {significant["primitive_pre_interaction"]}/24 primitive contrasts, {significant["post"]}/12 post-period contrasts, and {significant["distance_gain"]}/8 direct distance gains.

## Numerical and sensitivity diagnostics

- Point-estimate condition numbers range from {min(condition_numbers):.4f} to {max(condition_numbers):.4f}.
- Maximum absolute leave-one-event deltas (pre / interaction / post, basis points): {loo_maxima["event"]["pre"]:.4f} / {loo_maxima["event"]["interaction"]:.4f} / {loo_maxima["event"]["post"]:.4f}.
- Maximum absolute leave-one-year deltas (pre / interaction / post, basis points): {loo_maxima["year"]["pre"]:.4f} / {loo_maxima["year"]["interaction"]:.4f} / {loo_maxima["year"]["post"]:.4f}. The anchored 2011 bridge year is not omitted.
- K=1--10 sentiment stability versus K=10: {json.dumps(stability_summary, sort_keys=True)}.

## Interpretation

{results}

## Limitations

- The generated documents were never released to market participants; coefficients are not causal market effects.
- This is a Tadle-form adaptation, not an exact replication. The dictionary, Core8 document scope, and futures continuation rule are project-specific.
- Source-specific standardization and the anchored 2011 bridge-year bootstrap limit cross-source and population interpretations.
- The task-specific training roster overlaps 47 of the 83 standardization meetings and 46 of the 82 estimable current meetings; all 30 post-period shocks have current-or-lag overlap.
- Failure to reject a difference is not evidence of equivalence.
- Generation delivery, numeric/date fidelity and repetition gates are diagnostics only and do not filter the prespecified sample.

Generation delivery funnel: `{json.dumps(generation["delivery_funnel"], sort_keys=True)}`.
"""
    artifact = {
        "schema_version": f"{SCHEMA}:artifact",
        "status": "complete",
        "created_at_utc": created_at,
        "evaluation_id": EVALUATION_ID,
        "coverage": {
            "generation_meetings": EXPECTED_MEETINGS,
            "estimable_events": EXPECTED_ESTIMABLE_EVENTS,
            "atomic_generation_rows": int(verified["atomic_generation_rows"]),
            "generated_documents": int(verified["generated_documents"]),
            "bootstrap_draws": int(verified["bootstrap_draws"]),
        },
        "significant_holm_counts": significant,
        "diagnostics": {
            "condition_number_range": [min(condition_numbers), max(condition_numbers)],
            "leave_one_out_max_abs_delta_basis_points": loo_maxima,
            "k_stability_k1_to_k10": stability_summary,
        },
        "interpretation": {
            "mixed_scope_historical_diagnostic": True,
            "causal_market_effect": False,
            "equivalence_test": False,
            "exact_tadle_replication": False,
            "leakage_safe_external_evaluation": False,
        },
        "sources": {
            "statistics_manifest": binding(statistics_manifest_path, relative_to=root),
            "sentiment_manifest": binding(
                root / "sentiment/manifest.json", relative_to=root
            ),
            "estimate_receipt": binding(
                root / "estimation/estimate_receipt.json", relative_to=root
            ),
            "point_diagnostics": binding(
                root / "estimation/point_diagnostics.json", relative_to=root
            ),
            "leave_one_event_out": binding(
                root / "estimation/leave_one_event_out.jsonl",
                relative_to=root,
                rows=1_272,
            ),
            "leave_one_year_out": binding(
                root / "estimation/leave_one_year_out.jsonl", relative_to=root, rows=172
            ),
            "k_stability": binding(
                root / "sentiment/k_stability.jsonl", relative_to=root, rows=30
            ),
            "coefficient_estimates": binding(
                root / "estimation/coefficient_estimates.jsonl",
                relative_to=root,
                rows=16,
            ),
            "contrasts": binding(
                root / "estimation/model_minus_official_contrasts.jsonl",
                relative_to=root,
                rows=36,
            ),
            "distances": binding(
                root / "estimation/absolute_distances.jsonl", relative_to=root, rows=12
            ),
            "distance_gains": binding(
                root / "estimation/distance_gains.jsonl", relative_to=root, rows=8
            ),
            "training_overlap": binding(
                root / "preparation/training_overlap_audit.json", relative_to=root
            ),
            "preparation_manifest": binding(
                root / "preparation/evaluation_manifest.json", relative_to=root
            ),
            "generation_validation": binding(
                root / "generation/validation.json", relative_to=root
            ),
        },
    }
    post_table = build_post_beta_table(points, contrasts)
    distance_table = build_distance_gain_table(distances, gains)
    html_report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Paper chk-2 Tadle-form LM K=10</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;line-height:1.45}}pre{{white-space:pre-wrap;background:#f6f8fa;padding:1rem;border-radius:6px}}</style>
</head><body><h1>Paper chk-2 cp50: Historical Federal Funds Futures Document-Source Diagnostic</h1>
<pre>{html.escape(markdown)}</pre></body></html>
"""
    return {
        "artifact.json": (
            json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "report.md": markdown.encode("utf-8"),
        "report.html": html_report.encode("utf-8"),
        "methods.tex": methods.encode("utf-8"),
        "post_beta_table.tex": post_table.encode("utf-8"),
        "distance_and_gain_table.tex": distance_table.encode("utf-8"),
        "results.tex": results.encode("utf-8"),
    }


def _manifest_payload(report_dir: Path, root: Path, created_at: str) -> dict[str, Any]:
    names = (
        "artifact.json",
        "report.md",
        "report.html",
        "methods.tex",
        "post_beta_table.tex",
        "distance_and_gain_table.tex",
        "results.tex",
    )
    artifact_bindings: dict[str, dict[str, Any]] = {}
    for name in names:
        record = binding(report_dir / name, relative_to=root)
        # During publication ``report_dir`` is a staging directory.  Bind the
        # intended immutable location rather than the temporary path name.
        record["path"] = str(Path("report") / name)
        artifact_bindings[name] = record
    return {
        "schema_version": f"{SCHEMA}:manifest",
        "status": "complete",
        "immutable": True,
        "created_at_utc": created_at,
        "artifacts": artifact_bindings,
        "statistics_manifest": binding(
            root / "estimation/statistics_manifest.json", relative_to=root
        ),
        "sentiment_manifest": binding(
            root / "sentiment/manifest.json", relative_to=root
        ),
        "estimate_receipt": binding(
            root / "estimation/estimate_receipt.json", relative_to=root
        ),
        "mutation_policy": "create_once_then_full_verify_only",
    }


def _deterministic_created_at(root: Path) -> str:
    value = read_json(root / "estimation/statistics_manifest.json").get(
        "created_at_utc"
    )
    if not isinstance(value, str) or not value:
        raise ReportError("statistics manifest created_at_utc contract drift")
    return value


def _recover_partial_report(root: Path, report_dir: Path) -> dict[str, Any]:
    created_at = _deterministic_created_at(root)
    payloads = _build_payloads(root, created_at=created_at)
    allowed = set(payloads) | {"report_manifest.json"}
    observed = {entry.name for entry in report_dir.iterdir()}
    unexpected = observed - allowed
    if unexpected:
        raise ReportError(f"unexpected partial report entries: {sorted(unexpected)}")
    for name, payload in payloads.items():
        path = report_dir / name
        if path.exists() or path.is_symlink():
            if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
                raise ReportError(f"partial report drift: {name}")
        else:
            _write_new_bytes(path, payload)
    manifest = _manifest_payload(report_dir, root, created_at)
    _write_new_bytes(
        report_dir / "report_manifest.json",
        (
            json.dumps(
                manifest, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
            )
            + "\n"
        ).encode("utf-8"),
    )
    _fsync_directory(report_dir)
    _fsync_directory(root)
    return read_json(report_dir / "report_manifest.json")


def render(
    evaluation_root: Path = DEFAULT_EVALUATION_ROOT, *, resume: bool = False
) -> dict[str, Any]:
    root = evaluation_root.expanduser().resolve()
    report_dir = root / "report"
    manifest_path = report_dir / "report_manifest.json"
    if report_dir.exists() or report_dir.is_symlink():
        if not resume:
            raise FileExistsError(f"create-only report exists: {report_dir}")
        if not report_dir.is_dir() or report_dir.is_symlink():
            raise ReportError(f"unsafe report directory: {report_dir}")
        if not manifest_path.exists() and not manifest_path.is_symlink():
            return _recover_partial_report(root, report_dir)
        manifest = read_json(manifest_path)
        created_at = str(manifest.get("created_at_utc", ""))
        payloads = _build_payloads(root, created_at=created_at)
        for name, payload in payloads.items():
            path = report_dir / name
            if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
                raise ReportError(f"report replay drift: {name}")
        expected = _manifest_payload(report_dir, root, created_at)
        if manifest != expected:
            raise ReportError("report manifest replay drift")
        return manifest

    root.mkdir(parents=True, exist_ok=True)
    created_at = _deterministic_created_at(root)
    payloads = _build_payloads(root, created_at=created_at)
    # Reserving the final directory itself is the create-once publication
    # primitive.  Concurrent publishers cannot replace an existing report,
    # even when it is empty or only partially written after a failed attempt.
    report_dir.mkdir(mode=0o700, exist_ok=False)
    _fsync_directory(root)
    for name, payload in payloads.items():
        _write_new_bytes(report_dir / name, payload)
    manifest = _manifest_payload(report_dir, root, created_at)
    _write_new_bytes(
        manifest_path,
        (
            json.dumps(
                manifest, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
            )
            + "\n"
        ).encode("utf-8"),
    )
    _fsync_directory(report_dir)
    _fsync_directory(root)
    return read_json(manifest_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = render(args.evaluation_root, resume=args.resume)
    print(canonical({"status": "complete", "manifest": manifest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
