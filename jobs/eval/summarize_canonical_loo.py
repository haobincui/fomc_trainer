"""Canonical meeting-level report for the scoped six-indicator LOO experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import stats

from jobs.eval.eval_leave_one_out import (
    DELETION_STRATEGY,
    NEUTRAL_STRATEGY,
    _holm_adjust,
    _load_scoped_experiment,
    validate_scoped_reference_manifest,
    validate_scoped_release_binding,
    validate_scoped_score_matrix,
)
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


REPORT_SCHEMA_VERSION = "loo-six-indicator-report-v1"
ARM_SPECS = {
    "deletion_primary": ("primary", DELETION_STRATEGY),
    "neutral_primary": ("primary", NEUTRAL_STRATEGY),
    "deletion_stochastic": ("stochastic", DELETION_STRATEGY),
    "neutral_stochastic": ("stochastic", NEUTRAL_STRATEGY),
}
ESTIMANDS = ("deletion", "neutral", "delete_minus_neutral")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label} JSON {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {resolved}")
    return payload


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {label} {resolved}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected JSON object in {label} {resolved}:{line_number}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{label} is empty: {resolved}")
    return rows


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _csv_data_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _row in csv.reader(handle)) - 1, 0)


def validate_report_manifest(
    manifest_file: Path,
    *,
    scope_manifest_file: Path,
    population_id: str,
) -> dict[str, Any]:
    """Validate a sealed report and re-hash every input and output artifact."""

    path = manifest_file.expanduser().resolve()
    manifest = _read_json(path, label="six-indicator report manifest")
    if manifest.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported six-indicator report schema: {path}")
    payload_sha256 = validate_manifest_integrity(manifest)
    scope = _load_scoped_experiment(scope_manifest_file)
    if (
        manifest.get("status") != "complete"
        or manifest.get("population_id") != population_id
        or manifest.get("experiment_id") != scope["experiment_id"]
        or manifest.get("scope_manifest", {}).get("sha256") != scope["_sha256"]
    ):
        raise ValueError("Report manifest is not bound to the requested scope/population")

    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(ARM_SPECS):
        raise ValueError("Report manifest does not bind exactly four scoring arms")
    verified_inputs: dict[str, dict[str, str]] = {}
    for arm, record in inputs.items():
        if not isinstance(record, dict):
            raise ValueError(f"Invalid report input record for {arm}")
        rows_path = Path(str(record.get("rows_file") or "")).expanduser().resolve()
        audit_path = Path(str(record.get("audit_file") or "")).expanduser().resolve()
        if (
            not rows_path.is_file()
            or not audit_path.is_file()
            or sha256_file(rows_path) != record.get("rows_sha256")
            or sha256_file(audit_path) != record.get("audit_sha256")
        ):
            raise ValueError(f"Report scoring input changed or is missing for {arm}")
        scoring_audit = _read_json(audit_path, label=f"{arm} scoring audit")
        scoped_audit = scoring_audit.get("scoped_experiment")
        generation_audit = scoring_audit.get("generation_manifest")
        compatibility = record.get("compatibility")
        if (
            not isinstance(scoped_audit, dict)
            or not isinstance(generation_audit, dict)
            or not isinstance(compatibility, dict)
        ):
            raise ValueError(f"Report scoring dependency audit is incomplete for {arm}")

        reference_path = Path(
            str(scoring_audit.get("reference_file") or "")
        ).expanduser().resolve()
        reference_binding = scoped_audit.get("reference_manifest")
        if not isinstance(reference_binding, dict):
            raise ValueError(f"Report lacks a reference-manifest binding for {arm}")
        reference_manifest_path = Path(
            str(reference_binding.get("path") or "")
        ).expanduser().resolve()
        reference_validation = validate_scoped_reference_manifest(
            reference_file=reference_path,
            reference_manifest_file=reference_manifest_path,
            scope=scope,
            population_id=population_id,
        )
        if (
            reference_validation["sha256"] != reference_binding.get("sha256")
            or reference_validation["sha256"]
            != compatibility.get("reference_manifest_sha256")
            or reference_validation["reference_sha256"]
            != scoring_audit.get("reference_file_sha256")
            or reference_validation["reference_sha256"]
            != compatibility.get("reference_file_sha256")
        ):
            raise ValueError(
                f"Actual-Minutes reference dependency changed for {arm}"
            )

        release_binding = scoped_audit.get("generation_release")
        if not isinstance(release_binding, dict):
            raise ValueError(f"Report lacks a scoped-release binding for {arm}")
        release_path = Path(
            str(release_binding.get("path") or "")
        ).expanduser().resolve()
        if (
            not release_path.is_file()
            or sha256_file(release_path) != release_binding.get("sha256")
            or release_binding.get("sha256")
            != compatibility.get("generation_release_sha256")
        ):
            raise ValueError(f"Scoped generation release dependency changed for {arm}")
        generation_manifest_path = Path(
            str(generation_audit.get("path") or "")
        ).expanduser().resolve()
        if (
            not generation_manifest_path.is_file()
            or sha256_file(generation_manifest_path) != generation_audit.get("sha256")
        ):
            raise ValueError(f"Generation manifest dependency changed for {arm}")
        generation_manifest = _read_json(
            generation_manifest_path,
            label=f"{arm} generation manifest",
        )
        release_validation = validate_scoped_release_binding(
            release_manifest_file=release_path,
            input_folder=generation_manifest_path.parent,
            generation_manifest=generation_manifest,
            scope=scope,
            population_id=population_id,
            regime=str(scoped_audit.get("regime") or ""),
            intervention_strategy=str(
                scoped_audit.get("intervention_strategy") or ""
            ),
        )
        if (
            release_validation["sha256"] != release_binding.get("sha256")
            or release_validation["sha256"]
            != compatibility.get("generation_release_sha256")
            or release_validation["payload_sha256"]
            != release_binding.get("payload_sha256")
            or release_validation["payload_sha256"]
            != compatibility.get("generation_release_payload_sha256")
            or release_validation["run_id"] != compatibility.get("generation_run_id")
            or release_validation["minutes_model_sha256"]
            != compatibility.get("generation_model_sha256")
            or release_validation["minutes_tokenizer_sha256"]
            != compatibility.get("generation_tokenizer_sha256")
        ):
            raise ValueError(f"Scoped generation release dependency changed for {arm}")
        verified_inputs[arm] = {
            "rows_sha256": str(record["rows_sha256"]),
            "audit_sha256": str(record["audit_sha256"]),
            "reference_sha256": str(reference_validation["reference_sha256"]),
            "reference_manifest_sha256": str(reference_validation["sha256"]),
            "generation_manifest_sha256": str(generation_audit["sha256"]),
            "generation_release_sha256": str(release_validation["sha256"]),
            "generation_release_payload_sha256": str(
                release_validation["payload_sha256"]
            ),
        }

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {
        "overall_csv",
        "meeting_level_csv",
        "report_markdown",
    }:
        raise ValueError("Report manifest has an invalid output inventory")
    verified_outputs: dict[str, str] = {}
    for name, record in outputs.items():
        if not isinstance(record, dict):
            raise ValueError(f"Invalid report output record for {name}")
        artifact_path = Path(str(record.get("path") or "")).expanduser().resolve()
        if (
            not artifact_path.is_file()
            or sha256_file(artifact_path) != record.get("sha256")
        ):
            raise ValueError(f"Report output changed or is missing: {name}")
        if name.endswith("csv") and _csv_data_row_count(artifact_path) != record.get(
            "row_count"
        ):
            raise ValueError(f"Report CSV row count changed: {name}")
        verified_outputs[name] = str(record["sha256"])
    return {
        "status": "validated",
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "payload_sha256": payload_sha256,
        "population_id": population_id,
        "inputs": verified_inputs,
        "outputs": verified_outputs,
    }


def _safe_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _safe_number(value)
    return value


def _validate_scoring_arm(
    *,
    arm: str,
    rows_file: Path,
    audit_file: Path,
    scope: dict[str, Any],
    population_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    regime, strategy = ARM_SPECS[arm]
    rows_path = rows_file.expanduser().resolve()
    audit_path = audit_file.expanduser().resolve()
    rows = _read_jsonl(rows_path, label=f"{arm} scored rows")
    audit = _read_json(audit_path, label=f"{arm} scoring audit")
    scoped = audit.get("scoped_experiment")
    generation = audit.get("generation_manifest")
    if (
        audit.get("target_mode") != "actual-minutes"
        or audit.get("excluded_rows") != 0
        or audit.get("scored_pairs") != len(rows)
        or Path(str(audit.get("row_output_file") or "")).expanduser().resolve()
        != rows_path
        or not isinstance(scoped, dict)
        or scoped.get("sha256") != scope["_sha256"]
        or scoped.get("experiment_id") != scope["experiment_id"]
        or scoped.get("population_id") != population_id
        or scoped.get("regime") != regime
        or scoped.get("intervention_strategy") != strategy
        or not isinstance(generation, dict)
        or generation.get("masking_strategy") != strategy
    ):
        raise ValueError(f"{arm} scoring audit is not bound to the requested scope/arm")
    matrix_audit = validate_scoped_score_matrix(
        rows,
        scope=scope,
        population_id=population_id,
        regime=regime,
    )
    for index, row in enumerate(rows):
        if row.get("target_mode") != "actual-minutes":
            raise ValueError(f"{arm} row {index} does not use the Actual-Minutes target")
        if row.get("masking_strategy") != strategy:
            raise ValueError(f"{arm} row {index} has the wrong intervention strategy")
        try:
            full = float(row["similarity_full"])
            intervention = float(row["similarity_masked"])
            delta = float(row["delta"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{arm} row {index} has invalid LOO metrics") from exc
        if not all(math.isfinite(value) for value in (full, intervention, delta)):
            raise ValueError(f"{arm} row {index} has non-finite LOO metrics")
        if not math.isclose(delta, full - intervention, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(
                f"{arm} row {index} violates delta = similarity_full - "
                "similarity_intervention"
            )
    embedding_artifact = audit.get("embedding_model_artifact")
    reference_binding = scoped.get("reference_manifest")
    release_binding = scoped.get("generation_release")
    generation_model = generation.get("generation_model_artifact")
    generation_tokenizer = generation.get("generation_tokenizer_artifact")
    if not all(
        isinstance(value, dict)
        for value in (
            embedding_artifact,
            reference_binding,
            release_binding,
            generation_model,
            generation_tokenizer,
        )
    ):
        raise ValueError(f"{arm} audit lacks complete provenance bindings")
    compatibility = {
        "reference_file_sha256": audit.get("reference_file_sha256"),
        "reference_manifest_sha256": reference_binding.get("sha256"),
        "reference_key_fields": audit.get("reference_key_fields"),
        "reference_text_field": audit.get("reference_text_field"),
        "reference_duplicate_policy": audit.get("reference_duplicate_policy"),
        "embedding_model_path": audit.get("embedding_model_path"),
        "embedding_model_sha256": embedding_artifact.get("sha256"),
        "embedding_model_kind": embedding_artifact.get("kind"),
        "embedding_max_tokens": audit.get("embedding_max_tokens"),
        "embedding_long_text_policy": audit.get("embedding_long_text_policy"),
        "baseline_indicator": audit.get("baseline_indicator"),
        "generation_release_sha256": release_binding.get("sha256"),
        "generation_release_payload_sha256": release_binding.get("payload_sha256"),
        "generation_run_id": release_binding.get("run_id"),
        "generation_model_sha256": generation_model.get("sha256"),
        "generation_tokenizer_sha256": generation_tokenizer.get("sha256"),
    }
    if any(value is None for value in compatibility.values()):
        raise ValueError(f"{arm} audit has an incomplete cross-arm compatibility record")
    return rows, {
        "rows_file": str(rows_path),
        "rows_sha256": sha256_file(rows_path),
        "audit_file": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "regime": regime,
        "intervention_strategy": strategy,
        "matrix": matrix_audit,
        "compatibility": compatibility,
    }


def aggregate_sections_replicates_meetings(
    rows: list[dict[str, Any]],
    *,
    scope: Mapping[str, Any],
    population_id: str,
    regime: str,
) -> pd.DataFrame:
    """Apply the frozen sections -> meeting-replicate -> meeting order."""

    frame = pd.DataFrame(rows)
    frame["replicate_id"] = frame["replicate_id"].astype(str)
    meeting_replicate = (
        frame.groupby(
            ["indicator", "meeting_date", "replicate_id"],
            sort=False,
            as_index=False,
        )
        .agg(delta=("delta", "mean"), section_count=("section_name", "nunique"))
    )
    if not meeting_replicate["section_count"].eq(len(scope["section_names"])).all():
        raise ValueError(f"{regime} aggregation encountered an incomplete section group")
    meeting = (
        meeting_replicate.groupby(
            ["indicator", "meeting_date"],
            sort=False,
            as_index=False,
        )
        .agg(
            delta=("delta", "mean"),
            replicate_count=("replicate_id", "nunique"),
        )
    )
    expected_replicates = len(scope["decoding"][regime]["replicate_seeds"])
    if not meeting["replicate_count"].eq(expected_replicates).all():
        raise ValueError(f"{regime} aggregation encountered an incomplete replicate group")
    expected_rows = (
        len(scope["intervention_indicators"])
        * len(scope["populations"][population_id]["meeting_dates"])
    )
    if len(meeting) != expected_rows:
        raise ValueError(
            f"{regime} meeting aggregation expected {expected_rows} rows, "
            f"observed {len(meeting)}"
        )
    return meeting


def _ordered_meeting_values(
    meeting: pd.DataFrame,
    *,
    scope: Mapping[str, Any],
    population_id: str,
) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    dates = scope["populations"][population_id]["meeting_dates"]
    for indicator in scope["intervention_indicators"]:
        subset = meeting.loc[meeting["indicator"] == indicator].set_index("meeting_date")
        if set(subset.index) != set(dates):
            raise ValueError(f"Meeting-level values are incomplete for {indicator!r}")
        values[indicator] = subset.loc[dates, "delta"].to_numpy(dtype=float)
    return values


def _t_test(
    values: np.ndarray,
) -> tuple[float | None, float | None, str]:
    if len(values) < 2:
        return None, None, "insufficient_meetings"
    standard_deviation = float(np.std(values, ddof=1))
    if standard_deviation == 0.0:
        mean = float(np.mean(values))
        if mean == 0.0:
            return 0.0, 1.0, "zero_variance_zero_mean"
        return (
            None,
            0.0,
            "zero_variance_nonzero_constant_infinite_t_limit",
        )
    result = stats.ttest_1samp(values, popmean=0.0)
    return (
        _safe_number(result.statistic),
        _safe_number(result.pvalue),
        "two_sided_one_sample_t",
    )


def _holm_six_deletion_family(p_values: list[float | None]) -> list[float]:
    """Apply Holm only when the entire preregistered six-test family exists."""

    if len(p_values) != 6 or any(
        value is None or not math.isfinite(float(value)) for value in p_values
    ):
        raise ValueError(
            "The preregistered Holm family requires six finite deletion p-values; "
            "refusing to shrink the family"
        )
    return [
        float(value)
        for value in _holm_adjust([float(value) for value in p_values])
    ]


def _descriptive_metrics(
    values: np.ndarray,
    *,
    ci_lower: float,
    ci_upper: float,
) -> dict[str, Any]:
    leave_one_out = np.asarray(
        [np.mean(np.delete(values, index)) for index in range(len(values))],
        dtype=float,
    )
    t_stat, p_value, p_value_status = _t_test(values)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "sd_meeting": float(np.std(values, ddof=1)),
        "positive_meeting_share": float(np.mean(values > 0.0)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "lomo_min": float(np.min(leave_one_out)),
        "lomo_max": float(np.max(leave_one_out)),
        "lomo_sign_stable": bool(
            np.all(leave_one_out > 0.0) or np.all(leave_one_out < 0.0)
        ),
        "t_stat": t_stat,
        "p_value": p_value,
        "p_value_status": p_value_status,
    }


def _joint_bootstrap_metrics(
    series: dict[tuple[str, str], np.ndarray],
    *,
    draws: int,
    seed: int,
) -> dict[tuple[str, str], tuple[float, float]]:
    if draws < 1:
        raise ValueError("bootstrap_draws must be positive")
    ordered_keys = list(series)
    matrix = np.vstack([series[key] for key in ordered_keys])
    n_meetings = matrix.shape[1]
    rng = np.random.default_rng(seed)
    meeting_indices = rng.integers(0, n_meetings, size=(draws, n_meetings))
    bootstrapped = matrix[:, meeting_indices].mean(axis=2)
    return {
        key: (
            float(np.quantile(bootstrapped[index], 0.025)),
            float(np.quantile(bootstrapped[index], 0.975)),
        )
        for index, key in enumerate(ordered_keys)
    }


def _correlation_summary(
    left: list[float],
    right: list[float],
    *,
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if np.std(left_values) == 0.0 or np.std(right_values) == 0.0:
        pearson_r = pearson_p = spearman_rho = spearman_p = None
    else:
        pearson = stats.pearsonr(left_values, right_values)
        spearman = stats.spearmanr(left_values, right_values)
        pearson_r = _safe_number(pearson.statistic)
        pearson_p = _safe_number(pearson.pvalue)
        spearman_rho = _safe_number(spearman.statistic)
        spearman_p = _safe_number(spearman.pvalue)
    return {
        "left": left_label,
        "right": right_label,
        "n_indicators": len(left_values),
        "pearson_r": pearson_r,
        "pearson_p_value": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p_value": spearman_p,
        "sign_agreement": float(np.mean(np.sign(left_values) == np.sign(right_values))),
    }


def _fmt(value: object, digits: int = 6) -> str:
    number = _safe_number(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def _report_markdown(
    *,
    scope: Mapping[str, Any],
    population_id: str,
    overall_rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> str:
    phase = scope["populations"][population_id]["phase"]
    release_label = scope["release_policy"][phase]
    phase_title = (
        "Pilot — exploratory/protocol validation"
        if phase == "pilot"
        else "Formal — final six-indicator inference"
    )
    lines = [
        f"# Six-Indicator Canonical LOO — {phase_title}",
        "",
        f"- Phase: `{phase}` (`{release_label}`)",
        "- Target: frozen Actual-Minutes meeting section",
        "- Estimand: `cos(full, target) - cos(intervention, target)`",
        "- Aggregation: sections within meeting-replicate, replicates within meeting, then equal-weight meetings",
        f"- Inference: meeting-level; {bootstrap_draws:,} joint meeting bootstrap draws (seed `{bootstrap_seed}`)",
        "- Raw p-values: two-sided meeting-level one-sample t-tests against zero",
        "- Zero-variance rule: all-zero meeting effects use p=1; constant nonzero effects use the infinite-|t| limit p=0",
        "- Multiplicity: Holm correction over the six deletion raw p-values, separately for Primary and Stochastic",
        "",
    ]
    for regime in ("primary", "stochastic"):
        lines.extend(
            [
                f"## {regime.title()}",
                "",
                f"### {regime.title()} deletion estimand",
                "",
                "| Indicator | Mean Δ | Median | Meeting SD | Positive | 95% CI | LOMO range | Raw p | Holm p |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        regime_rows = [row for row in overall_rows if row["regime"] == regime]
        for row in regime_rows:
            lines.append(
                "| {indicator} | {mean} | {median} | {sd} | {positive:.1%} | "
                "[{lo}, {hi}] | [{lomo_lo}, {lomo_hi}] | {raw_p} | {holm} |".format(
                    indicator=row["indicator"],
                    mean=_fmt(row["deletion_mean"]),
                    median=_fmt(row["deletion_median"]),
                    sd=_fmt(row["deletion_sd_meeting"]),
                    lo=_fmt(row["deletion_ci_lower"]),
                    hi=_fmt(row["deletion_ci_upper"]),
                    raw_p=_fmt(row["deletion_p_value"]),
                    holm=_fmt(row["deletion_p_value_holm"]),
                    positive=float(row["deletion_positive_meeting_share"]),
                    lomo_lo=_fmt(row["deletion_lomo_min"]),
                    lomo_hi=_fmt(row["deletion_lomo_max"]),
                )
            )
        lines.extend(
            [
                "",
                "### Auxiliary neutral and deletion−neutral estimands",
                "",
                "| Estimand | Indicator | Mean Δ | Median | Meeting SD | Positive | 95% CI | LOMO range | Raw p |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in regime_rows:
            for estimand, label in (
                ("neutral", "Neutral"),
                ("delete_minus_neutral", "Delete−Neutral"),
            ):
                lines.append(
                    "| {label} | {indicator} | {mean} | {median} | {sd} | "
                    "{positive:.1%} | [{lo}, {hi}] | [{lomo_lo}, {lomo_hi}] | "
                    "{raw_p} |".format(
                        label=label,
                        indicator=row["indicator"],
                        mean=_fmt(row[f"{estimand}_mean"]),
                        median=_fmt(row[f"{estimand}_median"]),
                        sd=_fmt(row[f"{estimand}_sd_meeting"]),
                        positive=float(row[f"{estimand}_positive_meeting_share"]),
                        lo=_fmt(row[f"{estimand}_ci_lower"]),
                        hi=_fmt(row[f"{estimand}_ci_upper"]),
                        lomo_lo=_fmt(row[f"{estimand}_lomo_min"]),
                        lomo_hi=_fmt(row[f"{estimand}_lomo_max"]),
                        raw_p=_fmt(row[f"{estimand}_p_value"]),
                    )
                )
        consistency = diagnostics["deletion_neutral_consistency"][regime]
        lines.extend(
            [
                "",
                "Deletion–neutral consistency across the six indicator means: "
                f"Pearson r={_fmt(consistency['pearson_r'], 4)}, "
                f"Spearman ρ={_fmt(consistency['spearman_rho'], 4)}, "
                f"sign agreement={consistency['sign_agreement']:.1%}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Cross-regime robustness",
            "",
            "| Estimand | Pearson r | Spearman ρ | Sign agreement |",
            "|---|---:|---:|---:|",
        ]
    )
    for estimand, label in (
        ("deletion", "Deletion"),
        ("neutral", "Neutral"),
        ("delete_minus_neutral", "Delete−Neutral"),
    ):
        cross = diagnostics["primary_stochastic_consistency"][estimand]
        lines.append(
            f"| {label} | {_fmt(cross['pearson_r'], 4)} | "
            f"{_fmt(cross['spearman_rho'], 4)} | "
            f"{cross['sign_agreement']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            str(scope["claim_boundary"]),
            "",
            "Pilot results are exploratory protocol validation. Formal results are "
            "the frozen six-indicator inference. Legacy 7-indicator and prior "
            "remaining-20 values are not pooled into any estimate or test in this report.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_canonical_loo(
    *,
    scope_manifest_file: Path,
    population_id: str,
    arm_rows: Mapping[str, Path],
    arm_audits: Mapping[str, Path],
    output_dir: Path,
    bootstrap_draws: int = 10_000,
    bootstrap_seed: int = 20260728,
) -> dict[str, Any]:
    """Validate four scoring arms and write one population-specific release."""

    if set(arm_rows) != set(ARM_SPECS) or set(arm_audits) != set(ARM_SPECS):
        raise ValueError(f"Exactly four named scoring arms are required: {sorted(ARM_SPECS)}")
    if bootstrap_draws != 10_000:
        raise ValueError("Canonical six-indicator reports require exactly 10,000 bootstrap draws")
    scope = _load_scoped_experiment(scope_manifest_file)
    if population_id not in scope["populations"]:
        raise ValueError(f"Population {population_id!r} is outside the scoped experiment")

    validated_rows: dict[str, list[dict[str, Any]]] = {}
    input_audits: dict[str, dict[str, Any]] = {}
    for arm in ARM_SPECS:
        validated_rows[arm], input_audits[arm] = _validate_scoring_arm(
            arm=arm,
            rows_file=arm_rows[arm],
            audit_file=arm_audits[arm],
            scope=scope,
            population_id=population_id,
        )
    first_arm = next(iter(ARM_SPECS))
    frozen_compatibility = input_audits[first_arm]["compatibility"]
    incompatible = {
        arm: {
            field: {
                "expected": expected,
                "observed": input_audits[arm]["compatibility"].get(field),
            }
            for field, expected in frozen_compatibility.items()
            if input_audits[arm]["compatibility"].get(field) != expected
        }
        for arm in ARM_SPECS
        if input_audits[arm]["compatibility"] != frozen_compatibility
    }
    if incompatible:
        raise ValueError(
            f"Scoring arms have incompatible target/model/release provenance: "
            f"{incompatible}"
        )

    arm_meeting: dict[str, dict[str, np.ndarray]] = {}
    for arm, rows in validated_rows.items():
        regime, _strategy = ARM_SPECS[arm]
        meeting_frame = aggregate_sections_replicates_meetings(
            rows,
            scope=scope,
            population_id=population_id,
            regime=regime,
        )
        arm_meeting[arm] = _ordered_meeting_values(
            meeting_frame,
            scope=scope,
            population_id=population_id,
        )

    values_by_regime: dict[str, dict[tuple[str, str], np.ndarray]] = {}
    metric_by_regime: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for regime in ("primary", "stochastic"):
        deletion = arm_meeting[f"deletion_{regime}"]
        neutral = arm_meeting[f"neutral_{regime}"]
        joint_series: dict[tuple[str, str], np.ndarray] = {}
        for indicator in scope["intervention_indicators"]:
            joint_series[(indicator, "deletion")] = deletion[indicator]
            joint_series[(indicator, "neutral")] = neutral[indicator]
            joint_series[(indicator, "delete_minus_neutral")] = (
                deletion[indicator] - neutral[indicator]
            )
        intervals = _joint_bootstrap_metrics(
            joint_series,
            draws=bootstrap_draws,
            seed=bootstrap_seed,
        )
        values_by_regime[regime] = joint_series
        metric_by_regime[regime] = {
            key: _descriptive_metrics(
                values,
                ci_lower=intervals[key][0],
                ci_upper=intervals[key][1],
            )
            for key, values in joint_series.items()
        }

    for regime in ("primary", "stochastic"):
        deletion_p_values = [
            metric_by_regime[regime][(indicator, "deletion")]["p_value"]
            for indicator in scope["intervention_indicators"]
        ]
        adjusted = _holm_six_deletion_family(deletion_p_values)
        for indicator, adjusted_p in zip(
            scope["intervention_indicators"], adjusted, strict=True
        ):
            metric_by_regime[regime][(indicator, "deletion")][
                "p_value_holm"
            ] = _safe_number(adjusted_p)

    overall_rows: list[dict[str, Any]] = []
    meeting_rows: list[dict[str, Any]] = []
    meeting_dates = scope["populations"][population_id]["meeting_dates"]
    for regime in ("primary", "stochastic"):
        for indicator in scope["intervention_indicators"]:
            row: dict[str, Any] = {
                "schema_version": REPORT_SCHEMA_VERSION,
                "experiment_id": scope["experiment_id"],
                "population_id": population_id,
                "phase": scope["populations"][population_id]["phase"],
                "regime": regime,
                "indicator": indicator,
                "n_meetings": len(meeting_dates),
                "n_sections_per_meeting_replicate": len(scope["section_names"]),
                "n_replicates_per_meeting": len(
                    scope["decoding"][regime]["replicate_seeds"]
                ),
                "bootstrap_draws": bootstrap_draws,
                "bootstrap_seed": bootstrap_seed,
                "holm_family": f"{population_id}:{regime}:six_deletion_estimands",
                "holm_family_size": 6,
            }
            for estimand in ESTIMANDS:
                metrics = metric_by_regime[regime][(indicator, estimand)]
                for field, value in metrics.items():
                    row[f"{estimand}_{field}"] = value
            overall_rows.append(row)
            for meeting_index, meeting_date in enumerate(meeting_dates):
                meeting_rows.append(
                    {
                        "schema_version": REPORT_SCHEMA_VERSION,
                        "experiment_id": scope["experiment_id"],
                        "population_id": population_id,
                        "phase": scope["populations"][population_id]["phase"],
                        "regime": regime,
                        "indicator": indicator,
                        "meeting_date": meeting_date,
                        "deletion_delta": values_by_regime[regime][
                            (indicator, "deletion")
                        ][meeting_index],
                        "neutral_delta": values_by_regime[regime][
                            (indicator, "neutral")
                        ][meeting_index],
                        "delete_minus_neutral_delta": values_by_regime[regime][
                            (indicator, "delete_minus_neutral")
                        ][meeting_index],
                    }
                )

    diagnostics: dict[str, Any] = {
        "deletion_neutral_consistency": {},
        "primary_stochastic_consistency": {},
    }
    for regime in ("primary", "stochastic"):
        diagnostics["deletion_neutral_consistency"][regime] = _correlation_summary(
            [
                metric_by_regime[regime][(indicator, "deletion")]["mean"]
                for indicator in scope["intervention_indicators"]
            ],
            [
                metric_by_regime[regime][(indicator, "neutral")]["mean"]
                for indicator in scope["intervention_indicators"]
            ],
            left_label=f"{regime}_deletion",
            right_label=f"{regime}_neutral",
        )
    for estimand in ESTIMANDS:
        diagnostics["primary_stochastic_consistency"][estimand] = _correlation_summary(
            [
                metric_by_regime["primary"][(indicator, estimand)]["mean"]
                for indicator in scope["intervention_indicators"]
            ],
            [
                metric_by_regime["stochastic"][(indicator, estimand)]["mean"]
                for indicator in scope["intervention_indicators"]
            ],
            left_label=f"primary_{estimand}",
            right_label=f"stochastic_{estimand}",
        )

    output_root = output_dir.expanduser().resolve()
    prefix = population_id
    overall_path = output_root / f"{prefix}_loo_six_indicator_overall.csv"
    meeting_path = output_root / f"{prefix}_loo_six_indicator_meeting_level.csv"
    report_path = output_root / f"{prefix}_loo_six_indicator_report.md"
    manifest_path = output_root / f"{prefix}_loo_six_indicator_manifest.json"
    safe_overall = [_json_safe(row) for row in overall_rows]
    safe_meeting = [_json_safe(row) for row in meeting_rows]
    _atomic_write_csv(overall_path, safe_overall)  # type: ignore[arg-type]
    _atomic_write_csv(meeting_path, safe_meeting)  # type: ignore[arg-type]
    _atomic_write_text(
        report_path,
        _report_markdown(
            scope=scope,
            population_id=population_id,
            overall_rows=overall_rows,
            diagnostics=diagnostics,
            bootstrap_draws=bootstrap_draws,
            bootstrap_seed=bootstrap_seed,
        ),
    )

    manifest: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete",
        "experiment_id": scope["experiment_id"],
        "population_id": population_id,
        "phase": scope["populations"][population_id]["phase"],
        "release_label": scope["release_policy"][
            scope["populations"][population_id]["phase"]
        ],
        "scope_manifest": {
            "path": scope["_path"],
            "sha256": scope["_sha256"],
        },
        "claim_boundary": scope["claim_boundary"],
        "legacy_or_remaining20_pooling": "forbidden",
        "estimand": "delta = cos(full_output, actual_minutes) - cos(intervention_output, actual_minutes)",
        "aggregation_order": [
            "equal-weight sections within meeting-replicate",
            "equal-weight replicates within meeting",
            "equal-weight meetings",
        ],
        "inference_unit": "meeting",
        "bootstrap": {
            "kind": "joint_meeting_cluster_percentile",
            "draws": bootstrap_draws,
            "seed": bootstrap_seed,
            "confidence": 0.95,
            "same_meeting_resample_across_all_indicators_and_estimands": True,
        },
        "multiplicity": {
            "method": "Holm",
            "raw_test": (
                "two-sided meeting-level one-sample t-test against zero; "
                "zero variance maps all-zero to p=1 and constant nonzero to "
                "the infinite-|t| limit p=0"
            ),
            "family": "six overall deletion estimands separately within primary and stochastic",
            "family_size": 6,
            "neutral_and_contrast_p_values": "descriptive_unadjusted_auxiliary",
        },
        "inputs": input_audits,
        "diagnostics": diagnostics,
        "outputs": {
            "overall_csv": {
                "path": str(overall_path),
                "sha256": sha256_file(overall_path),
                "row_count": len(overall_rows),
            },
            "meeting_level_csv": {
                "path": str(meeting_path),
                "sha256": sha256_file(meeting_path),
                "row_count": len(meeting_rows),
            },
            "report_markdown": {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
            },
        },
    }
    sealed_manifest = seal_manifest(manifest)
    _atomic_write_text(
        manifest_path,
        json.dumps(_json_safe(sealed_manifest), ensure_ascii=False, indent=2) + "\n",
    )
    return sealed_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize strict four-arm six-indicator LOO scores."
    )
    parser.add_argument("--scope-manifest", type=Path, required=True)
    parser.add_argument(
        "--population-id",
        choices=["pilot_eval_13", "formal_test_13"],
        required=True,
    )
    parser.add_argument(
        "--validate-report-manifest",
        type=Path,
        help="Validate a sealed prior report and every bound input/output hash.",
    )
    for arm in ARM_SPECS:
        option = arm.replace("_", "-")
        parser.add_argument(f"--{option}-rows", type=Path)
        parser.add_argument(f"--{option}-audit", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.validate_report_manifest is not None:
        validation = validate_report_manifest(
            args.validate_report_manifest,
            scope_manifest_file=args.scope_manifest,
            population_id=args.population_id,
        )
        print("Six-indicator LOO report manifest valid")
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return
    arm_rows = {
        arm: getattr(args, f"{arm}_rows")
        for arm in ARM_SPECS
    }
    arm_audits = {
        arm: getattr(args, f"{arm}_audit")
        for arm in ARM_SPECS
    }
    missing = sorted(
        [name for name, value in arm_rows.items() if value is None]
        + [f"{name}-audit" for name, value in arm_audits.items() if value is None]
    )
    if args.output_dir is None:
        missing.append("output-dir")
    if missing:
        raise SystemExit(
            "Report generation requires all four row/audit inputs and --output-dir; "
            f"missing: {missing}"
        )
    manifest = summarize_canonical_loo(
        scope_manifest_file=args.scope_manifest,
        population_id=args.population_id,
        arm_rows=arm_rows,
        arm_audits=arm_audits,
        output_dir=args.output_dir,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    print("Six-indicator LOO report complete")
    print(json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
