"""Render the paper-CHK2 text-similarity score bundle as portable HTML.

This module is deliberately presentation-only.  The scoring module remains the
authoritative implementation of semantic scoring, bootstrap resampling, and
paired contrasts.  Before rendering, this adapter deep-replays that bundle and
then projects the sealed CSV/JSON rows into the canonical Data Analytics report
artifact.  It never generates text, recomputes embeddings, or changes a score.

The resulting report is explicitly an in-sample training-release
reconstruction diagnostic.  It is not a leakage-safe evaluation of held-out
generalization.
"""

from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from open_r1.validator.loo_generation_spec import seal_manifest, validate_manifest_integrity


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCORE_DIR = ROOT / (
    "output/evaluation/retrain_v2/"
    "paper_chk2_text_similarity_all391_chk0_chk1_chk2cp50_"
    "vllm_t06_p09_k10_b1000_v1_20260901/scores"
)
DEFAULT_REPORT_DIR = DEFAULT_SCORE_DIR.parent / "report"
DEFAULT_PLUGIN_ROOT = Path(
    "/home/haobin_cui/.codex/plugins/cache/openai-curated-remote/"
    "data-analytics/0.2.9-13ceeea1f599"
)
PACKAGER_RELATIVE = Path("skills/build-report/scripts/deliver_portable_artifact.mjs")
DEFAULT_NODE_CANDIDATES = (
    ROOT / ".cache/report_node/bin/node",
    ROOT / ".cache/node_official/node-v22.12.0-linux-x64/bin/node",
    Path("/home/haobin_cui/.conda/envs/fomc_trainer/bin/node"),
)

REPORT_ID = "paper-chk2-text-similarity-technical-report-v1"
REPORT_MANIFEST_SCHEMA = "paper-chk2-text-similarity-report-manifest-v1"
REPORT_ARTIFACT_CONTRACT = "paper-chk2-text-similarity-report-artifact-v1"
REQUIRED_INPUTS = (
    "model_summary.csv",
    "per_k_results.csv",
    "split_results.csv",
    "pairwise_contrasts.csv",
    "results.json",
    "score_manifest.json",
)
MODEL_ORDER = ("chk0", "chk1", "chk2")
METRIC_ORDER = ("mpnet_cosine", "bertscore_f1")
SPLIT_ORDER = ("train", "validation", "test")
POLICY_ORDER = ("raw_best_effort", "delivery_penalized")
PRIMARY_SCHEME = "meeting_cluster_primary"
PRIMARY_POLICY = "raw_best_effort"
EXPECTED_ROWS = {
    "model_summary.csv": 24,
    "per_k_results.csv": 240,
    "split_results.csv": 36,
    "pairwise_contrasts.csv": 264,
}

MODEL_LABELS = {"chk0": "Model chk-0", "chk1": "Model chk-1", "chk2": "Model chk-2"}
METRIC_LABELS = {
    "mpnet_cosine": "MPNet cosine similarity",
    "bertscore_f1": "BERTScore F1",
}
POLICY_LABELS = {
    "raw_best_effort": "Raw best effort",
    "delivery_penalized": "Delivery penalized",
}

SCOPE_DISCLOSURE = (
    "This is an in-sample training-release reconstruction diagnostic; it is "
    "not a leakage-safe evaluation of held-out generalization."
)
REFERENCE_DISCLOSURE = (
    "The reference is the synthetic teacher rewrite extracted after </think>, "
    "not an official FOMC Minutes paragraph."
)
METRIC_DISCLOSURE = (
    "MPNet cosine similarity and BERTScore F1 measure textual-semantic "
    "resemblance to that synthetic reference, not factual correctness."
)


class PaperChk2SimilarityReportError(RuntimeError):
    """The score-to-report contract or create-only publication failed."""


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise PaperChk2SimilarityReportError(f"non-canonical JSON value: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, *, label: str) -> Path:
    unresolved = path.expanduser()
    resolved = unresolved.resolve()
    if unresolved.is_symlink() or resolved.is_symlink() or not resolved.is_file():
        raise PaperChk2SimilarityReportError(
            f"{label} must be a regular non-symlink file: {unresolved}"
        )
    return resolved


def _binding(path: Path, *, payload_sha256: str | None = None) -> dict[str, Any]:
    resolved = _regular_file(path, label="bound artifact")
    result: dict[str, Any] = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }
    if payload_sha256 is not None:
        result["payload_sha256"] = payload_sha256
    return result


def _read_json(path: Path, *, label: str, sealed: bool = False) -> dict[str, Any]:
    resolved = _regular_file(path, label=label)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PaperChk2SimilarityReportError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise PaperChk2SimilarityReportError(f"{label} must be a JSON object")
    if sealed or "integrity" in value:
        try:
            validate_manifest_integrity(value)
        except Exception as exc:
            raise PaperChk2SimilarityReportError(f"{label} integrity failed: {exc}") from exc
    return value


def _read_csv(path: Path, *, expected_fields: Sequence[str], label: str) -> list[dict[str, str]]:
    resolved = _regular_file(path, label=label)
    try:
        with resolved.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(expected_fields):
                raise PaperChk2SimilarityReportError(
                    f"{label} header drift: expected {list(expected_fields)}, "
                    f"observed {reader.fieldnames}"
                )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise PaperChk2SimilarityReportError(f"cannot read {label}: {exc}") from exc
    if any(set(row) != set(expected_fields) for row in rows):
        raise PaperChk2SimilarityReportError(f"{label} row schema drift")
    return rows


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise PaperChk2SimilarityReportError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PaperChk2SimilarityReportError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise PaperChk2SimilarityReportError(f"{label} must be finite")
    return result


def _strict_int(value: Any, *, label: str) -> int:
    numeric = _finite_float(value, label=label)
    result = int(numeric)
    if numeric != result:
        raise PaperChk2SimilarityReportError(f"{label} must be integral")
    return result


def _ci(row: Mapping[str, Any]) -> str:
    return f"[{_finite_float(row['ci_lower'], label='ci_lower'):.4f}, {_finite_float(row['ci_upper'], label='ci_upper'):.4f}]"


def _score(value: Any) -> str:
    return f"{_finite_float(value, label='score'):.4f}"


def _p(value: Any) -> str:
    numeric = _finite_float(value, label="p value")
    return "<0.0001" if numeric < 0.0001 else f"{numeric:.4f}"


def _source(
    *, source_id: str, label: str, filename: str, binding: Mapping[str, Any], executed_at: str
) -> dict[str, Any]:
    # ``path`` is intentionally omitted: a score root may be outside the repository,
    # while absolute machine-local paths are not portable.  The exact immutable
    # file identity remains visible in the source query description.
    return {
        "id": source_id,
        "label": label,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": f"SELECT * FROM read_csv_auto('{filename}')" if filename.endswith(".csv") else f"SELECT * FROM read_json_auto('{filename}')",
            "description": (
                f"Validated projection of {filename}; SHA-256 "
                f"{binding['sha256']}, {binding['bytes']} bytes."
            ),
            "executed_at": executed_at,
            "tables_used": [filename],
        },
    }


def _deep_validate_score_bundle(score_dir: Path) -> dict[str, Any]:
    try:
        from jobs.eval import paper_chk2_text_similarity_scoring as scoring
    except ImportError as exc:  # pragma: no cover - deployment guard
        raise PaperChk2SimilarityReportError("authoritative scoring module unavailable") from exc
    try:
        receipt = scoring.validate_command(output_dir=score_dir)
    except Exception as exc:
        raise PaperChk2SimilarityReportError(
            f"authoritative score-bundle replay failed: {exc}"
        ) from exc
    if not isinstance(receipt, Mapping) or receipt.get("status") != "complete_validated":
        raise PaperChk2SimilarityReportError("score validator returned no completion receipt")
    return copy.deepcopy(dict(receipt))


def _validate_manifest_bindings(score_dir: Path, manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise PaperChk2SimilarityReportError("score manifest artifact inventory missing")
    bindings: dict[str, dict[str, Any]] = {}
    for filename in REQUIRED_INPUTS[:-1]:
        expected = artifacts.get(filename)
        if not isinstance(expected, Mapping):
            raise PaperChk2SimilarityReportError(f"score manifest does not bind {filename}")
        path = _regular_file(score_dir / filename, label=filename)
        resolved_expected = Path(str(expected.get("path", ""))).expanduser().resolve()
        if resolved_expected != path:
            raise PaperChk2SimilarityReportError(f"{filename} manifest path drift")
        observed = _binding(path)
        for key in ("bytes", "sha256"):
            if observed[key] != expected.get(key):
                raise PaperChk2SimilarityReportError(f"{filename} binding drift: {key}")
        if filename.endswith(".csv") and expected.get("rows") != EXPECTED_ROWS[filename]:
            raise PaperChk2SimilarityReportError(f"{filename} manifest row-count drift")
        bindings[filename] = observed
    payload_sha = manifest.get("integrity", {}).get("payload_sha256")
    bindings["score_manifest.json"] = _binding(
        score_dir / "score_manifest.json",
        payload_sha256=str(payload_sha) if isinstance(payload_sha, str) else None,
    )
    return bindings


def _project_summary(row: Mapping[str, str]) -> dict[str, Any]:
    model = row["model_id"]
    metric = row["metric"]
    if model not in MODEL_ORDER or metric not in METRIC_ORDER:
        raise PaperChk2SimilarityReportError("model-summary identity drift")
    estimate = _finite_float(row["estimate"], label="summary estimate")
    lower = _finite_float(row["ci_lower"], label="summary ci lower")
    upper = _finite_float(row["ci_upper"], label="summary ci upper")
    if not 0 <= lower <= estimate <= upper <= 1:
        raise PaperChk2SimilarityReportError("summary similarity interval is invalid")
    return {
        "bootstrap_scheme": row["bootstrap_scheme"],
        "scoring_policy": row["scoring_policy"],
        "policy_label": POLICY_LABELS.get(row["scoring_policy"], row["scoring_policy"]),
        "model_id": model,
        "model": MODEL_LABELS[model],
        "metric": metric,
        "metric_label": METRIC_LABELS[metric],
        "estimate": estimate,
        "estimate_display": _score(estimate),
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_95": _ci(row),
        "samples": _strict_int(row["samples"], label="summary samples"),
        "meetings": _strict_int(row["meetings"], label="summary meetings"),
        "replicates": _strict_int(row["replicates"], label="summary replicates"),
        "bootstrap_draws": _strict_int(row["bootstrap_draws"], label="summary draws"),
    }


def _project_per_k(row: Mapping[str, str]) -> dict[str, Any]:
    base = _project_common(row, kind="per-k")
    replicate = _strict_int(row["replicate_id"], label="replicate_id")
    if replicate not in range(10):
        raise PaperChk2SimilarityReportError("replicate identity drift")
    base.update({"replicate_id": replicate, "k": replicate + 1, "k_label": f"k={replicate + 1}"})
    return base


def _project_split(row: Mapping[str, str]) -> dict[str, Any]:
    base = _project_common(row, kind="split")
    split = row["split"]
    if split not in SPLIT_ORDER:
        raise PaperChk2SimilarityReportError("split identity drift")
    base.update(
        {
            "split": split,
            "split_label": "Validation" if split == "validation" else split.title(),
            "replicates": _strict_int(row["replicates"], label="split replicates"),
        }
    )
    return base


def _project_common(row: Mapping[str, str], *, kind: str) -> dict[str, Any]:
    model = row["model_id"]
    metric = row["metric"]
    if model not in MODEL_ORDER or metric not in METRIC_ORDER:
        raise PaperChk2SimilarityReportError(f"{kind} identity drift")
    estimate = _finite_float(row["estimate"], label=f"{kind} estimate")
    lower = _finite_float(row["ci_lower"], label=f"{kind} ci lower")
    upper = _finite_float(row["ci_upper"], label=f"{kind} ci upper")
    if not 0 <= lower <= estimate <= upper <= 1:
        raise PaperChk2SimilarityReportError(f"{kind} similarity interval invalid")
    return {
        "bootstrap_scheme": row["bootstrap_scheme"],
        "scoring_policy": row["scoring_policy"],
        "model_id": model,
        "model": MODEL_LABELS[model],
        "metric": metric,
        "metric_label": METRIC_LABELS[metric],
        "estimate": estimate,
        "estimate_display": _score(estimate),
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_95": _ci(row),
        "samples": _strict_int(row["samples"], label=f"{kind} samples"),
        "meetings": _strict_int(row["meetings"], label=f"{kind} meetings"),
        "bootstrap_draws": _strict_int(row["bootstrap_draws"], label=f"{kind} draws"),
    }


def _project_contrast(row: Mapping[str, str]) -> dict[str, Any]:
    metric = row["metric"]
    before = row["before_model"]
    after = row["after_model"]
    if metric not in METRIC_ORDER or before not in MODEL_ORDER or after not in MODEL_ORDER:
        raise PaperChk2SimilarityReportError("contrast identity drift")
    estimate = _finite_float(row["estimate"], label="contrast estimate")
    lower = _finite_float(row["ci_lower"], label="contrast ci lower")
    upper = _finite_float(row["ci_upper"], label="contrast ci upper")
    if not lower <= estimate <= upper:
        raise PaperChk2SimilarityReportError("contrast interval invalid")
    inferential = row["p_value"] != "" or row["holm_adjusted_p"] != ""
    if inferential:
        p_value = _finite_float(row["p_value"], label="contrast p")
        holm = _finite_float(row["holm_adjusted_p"], label="contrast Holm p")
        if not 0 <= p_value <= holm <= 1:
            raise PaperChk2SimilarityReportError("contrast probability invalid")
    else:
        p_value = None
        holm = None
    replicate_raw = row["replicate_id"]
    replicate: int | None = None if replicate_raw == "" else _strict_int(replicate_raw, label="contrast replicate")
    return {
        "bootstrap_scheme": row["bootstrap_scheme"],
        "scoring_policy": row["scoring_policy"],
        "aggregate_scope": row["aggregate_scope"],
        "replicate_id": replicate,
        "contrast_id": row["contrast_id"],
        "contrast": f"{MODEL_LABELS[after]} − {MODEL_LABELS[before]}",
        "metric": metric,
        "metric_label": METRIC_LABELS[metric],
        "series_label": f"{MODEL_LABELS[after]} − {MODEL_LABELS[before]} · {METRIC_LABELS[metric]}",
        "estimate": estimate,
        "estimate_display": f"{estimate:+.4f}",
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_95": f"[{lower:+.4f}, {upper:+.4f}]",
        "p_value": p_value,
        "p_value_display": "not tested" if p_value is None else _p(p_value),
        "holm_adjusted_p": holm,
        "holm_adjusted_p_display": "not adjusted" if holm is None else _p(holm),
        "holm_reject_005": False if holm is None else holm < 0.05,
        "p_value_method": row["p_value_method"],
        "bootstrap_draws": _strict_int(row["bootstrap_draws"], label="contrast draws"),
    }


def load_report_model(score_dir: Path) -> dict[str, Any]:
    """Deep-validate and project one complete sealed score bundle."""

    score_root = score_dir.expanduser().resolve()
    if score_root.is_symlink() or not score_root.is_dir():
        raise PaperChk2SimilarityReportError(f"score directory is missing: {score_root}")
    _deep_validate_score_bundle(score_root)
    manifest = _read_json(score_root / "score_manifest.json", label="score manifest", sealed=True)
    if (
        manifest.get("schema_version") != "paper-chk2-text-similarity-score-manifest-v1"
        or manifest.get("status") != "complete"
        or manifest.get("evaluation_role") != "in_sample_training_release_reconstruction_diagnostic"
        or manifest.get("training_release_evaluation_eligible") is not False
    ):
        raise PaperChk2SimilarityReportError("score manifest report-scope contract drift")
    bindings = _validate_manifest_bindings(score_root, manifest)
    results = _read_json(score_root / "results.json", label="bootstrap results", sealed=True)
    if (
        results.get("schema_version") != "paper-chk2-text-similarity-bootstrap-results-v1"
        or results.get("status") != "complete"
        or results.get("evaluation_role") != "in_sample_training_release_reconstruction_diagnostic"
        or results.get("models") != list(MODEL_ORDER)
        or results.get("splits") != list(SPLIT_ORDER)
        or results.get("policies") != list(POLICY_ORDER)
        or results.get("metrics") != list(METRIC_ORDER)
    ):
        raise PaperChk2SimilarityReportError("results report-scope contract drift")

    from jobs.eval import paper_chk2_text_similarity_scoring as scoring

    summary_raw = _read_csv(
        score_root / "model_summary.csv", expected_fields=scoring.MODEL_SUMMARY_FIELDS,
        label="model summary",
    )
    per_k_raw = _read_csv(
        score_root / "per_k_results.csv", expected_fields=scoring.PER_K_FIELDS,
        label="per-k results",
    )
    split_raw = _read_csv(
        score_root / "split_results.csv", expected_fields=scoring.SPLIT_FIELDS,
        label="split results",
    )
    contrasts_raw = _read_csv(
        score_root / "pairwise_contrasts.csv", expected_fields=scoring.CONTRAST_FIELDS,
        label="pairwise contrasts",
    )
    for filename, rows in (
        ("model_summary.csv", summary_raw),
        ("per_k_results.csv", per_k_raw),
        ("split_results.csv", split_raw),
        ("pairwise_contrasts.csv", contrasts_raw),
    ):
        if len(rows) != EXPECTED_ROWS[filename]:
            raise PaperChk2SimilarityReportError(
                f"{filename} row-count drift: {len(rows)}"
            )

    summaries = [_project_summary(row) for row in summary_raw]
    per_k = [_project_per_k(row) for row in per_k_raw]
    splits = [_project_split(row) for row in split_raw]
    contrasts = [_project_contrast(row) for row in contrasts_raw]

    primary_summary = [
        row for row in summaries
        if row["bootstrap_scheme"] == PRIMARY_SCHEME
        and row["scoring_policy"] == PRIMARY_POLICY
    ]
    if len(primary_summary) != 6:
        raise PaperChk2SimilarityReportError("primary pooled summary must contain six rows")
    primary_per_k = [
        row for row in per_k
        if row["bootstrap_scheme"] == PRIMARY_SCHEME
        and row["scoring_policy"] == PRIMARY_POLICY
    ]
    if len(primary_per_k) != 60:
        raise PaperChk2SimilarityReportError("primary per-k inventory must contain 60 rows")
    primary_splits = [
        row for row in splits
        if row["bootstrap_scheme"] == PRIMARY_SCHEME
        and row["scoring_policy"] == PRIMARY_POLICY
    ]
    if len(primary_splits) != 18:
        raise PaperChk2SimilarityReportError("primary split inventory must contain 18 rows")
    policy_summary = [row for row in summaries if row["bootstrap_scheme"] == PRIMARY_SCHEME]
    if len(policy_summary) != 12:
        raise PaperChk2SimilarityReportError("policy comparison must contain 12 rows")
    primary_contrasts = [
        row for row in contrasts
        if row["bootstrap_scheme"] == PRIMARY_SCHEME
        and row["scoring_policy"] == PRIMARY_POLICY
        and row["aggregate_scope"] == "pooled_k10"
        and row["replicate_id"] is None
    ]
    if len(primary_contrasts) != 6:
        raise PaperChk2SimilarityReportError("primary Holm family must contain six contrasts")
    if any(
        row["p_value"] is None
        or row["holm_adjusted_p"] is None
        or row["p_value_method"] == ""
        for row in primary_contrasts
    ):
        raise PaperChk2SimilarityReportError(
            "primary Holm family is missing inferential diagnostics"
        )

    coverage = results.get("coverage")
    diagnostics = results.get("delivery_diagnostics")
    limitations = results.get("limitations")
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("samples") != 391
        or coverage.get("meetings") != 124
        or coverage.get("replicates") != 10
        or coverage.get("generation_rows") != 11_730
        or not isinstance(diagnostics, Mapping)
        or set(diagnostics) != set(MODEL_ORDER)
        or not isinstance(limitations, list)
        or not limitations
    ):
        raise PaperChk2SimilarityReportError("coverage, delivery, or limitation contract drift")

    delivery: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        diag = diagnostics[model]
        if not isinstance(diag, Mapping):
            raise PaperChk2SimilarityReportError("delivery diagnostic is malformed")
        total = _strict_int(diag.get("generations"), label="delivery generations")
        invalid = _strict_int(diag.get("delivery_invalid"), label="delivery invalid")
        empty = _strict_int(diag.get("recovered_text_empty"), label="recovered empty")
        if total != 3_910 or not (0 <= invalid <= total) or not (0 <= empty <= total):
            raise PaperChk2SimilarityReportError("delivery diagnostic count drift")
        delivery.append(
            {
                "model_id": model,
                "model": MODEL_LABELS[model],
                "generations": total,
                "delivery_valid": total - invalid,
                "delivery_invalid": invalid,
                "recovered_nonempty": total - empty,
                "recovered_empty": empty,
                "delivery_valid_rate": (total - invalid) / total,
                "delivery_valid_rate_display": f"{100 * (total - invalid) / total:.2f}%",
                "recovered_nonempty_rate_display": f"{100 * (total - empty) / total:.2f}%",
                "failure_counts": json.dumps(diag.get("delivery_failure_counts", {}), sort_keys=True),
            }
        )

    inputs = manifest.get("inputs")
    created_at = None
    if isinstance(inputs, Mapping) and isinstance(inputs.get("evaluation_manifest"), Mapping):
        evaluation_path = Path(str(inputs["evaluation_manifest"].get("path", "")))
        # The completed evaluation manifest is SHA-bound by the sealed score
        # manifest but is not itself necessarily sealed with an ``integrity``
        # block.  ``_deep_validate_score_bundle`` has already replayed that
        # binding, so read it without requiring a second seal here.
        evaluation = _read_json(evaluation_path, label="evaluation manifest", sealed=False)
        created_at = evaluation.get("created_at_utc")
    if not isinstance(created_at, str) or not created_at:
        raise PaperChk2SimilarityReportError("stable evaluation timestamp unavailable")

    return {
        "generated_at_utc": created_at,
        "score_root": str(score_root),
        "source_bindings": bindings,
        "coverage": copy.deepcopy(dict(coverage)),
        "bootstrap_contract": copy.deepcopy(results.get("bootstrap_contract")),
        "multiple_testing": copy.deepcopy(results.get("multiple_testing")),
        "primary_summary": primary_summary,
        "primary_per_k": primary_per_k,
        "primary_splits": primary_splits,
        "policy_summary": policy_summary,
        "primary_contrasts": primary_contrasts,
        "delivery": delivery,
        "limitations": [str(item) for item in limitations],
    }


def _chart(
    *, chart_id: str, title: str, subtitle: str, chart_type: str, dataset: str,
    source_id: str, x_field: str, x_type: str, x_label: str, y_field: str,
    y_label: str, color_field: str | None = None, color_label: str | None = None,
    tooltip: Sequence[Mapping[str, str]] = (), reference_zero: bool = False,
) -> dict[str, Any]:
    encodings: dict[str, Any] = {
        "x": {"field": x_field, "type": x_type, "label": x_label},
        "y": {"field": y_field, "type": "quantitative", "label": y_label},
        "tooltip": [dict(item) for item in tooltip],
    }
    if color_field:
        encodings["color"] = {
            "field": color_field,
            "type": "nominal",
            "label": color_label or color_field,
        }
    result: dict[str, Any] = {
        "id": chart_id,
        "title": title,
        "subtitle": subtitle,
        "intent": "trend" if chart_type == "line" else "comparison",
        "question": title,
        "rationale": "A source-backed native chart makes the prespecified comparison visible; exact values remain in the adjacent table.",
        "type": chart_type,
        "dataset": dataset,
        "sourceId": source_id,
        "encodings": encodings,
        "palette": {"kind": "categorical", "name": "blue"},
        "labels": {"values": "auto"},
        "valueFormat": "number",
        "layout": "full",
    }
    if reference_zero:
        result["referenceLines"] = [
            {"axis": "y", "value": 0, "label": "Zero", "color": "neutral", "lineStyle": "dashed"}
        ]
    return result


def _table(
    *, table_id: str, title: str, subtitle: str, dataset: str, source_id: str,
    columns: Sequence[Mapping[str, Any]], density: str = "spacious",
) -> dict[str, Any]:
    return {
        "id": table_id,
        "title": title,
        "subtitle": subtitle,
        "dataset": dataset,
        "sourceId": source_id,
        "density": density,
        "layout": "full",
        "columns": [dict(column) for column in columns],
    }


def build_canonical_artifact(model: Mapping[str, Any]) -> dict[str, Any]:
    """Create the canonical native-block report artifact."""

    primary = list(model["primary_summary"])
    per_k = list(model["primary_per_k"])
    splits = list(model["primary_splits"])
    policies = list(model["policy_summary"])
    contrasts = list(model["primary_contrasts"])
    delivery = list(model["delivery"])
    source_bindings = model["source_bindings"]
    generated_at = str(model["generated_at_utc"])

    metric_sets = {
        metric: [row for row in primary if row["metric"] == metric]
        for metric in METRIC_ORDER
    }
    winners = {
        metric: max(rows, key=lambda row: float(row["estimate"]))
        for metric, rows in metric_sets.items()
    }
    same_winner = winners[METRIC_ORDER[0]]["model_id"] == winners[METRIC_ORDER[1]]["model_id"]
    if same_winner:
        winner_text = (
            f"{winners[METRIC_ORDER[0]]['model']} has the largest pooled raw mean "
            "on both reported similarity metrics."
        )
    else:
        winner_text = (
            f"The descriptive rankings differ: {winners['mpnet_cosine']['model']} "
            "has the largest MPNet cosine mean, while "
            f"{winners['bertscore_f1']['model']} has the largest BERTScore F1 mean."
        )
    chk2_contrasts = {
        row["metric"]: row for row in contrasts if row["contrast_id"] == "chk2_minus_chk1"
    }
    if set(chk2_contrasts) != set(METRIC_ORDER):
        raise PaperChk2SimilarityReportError("CHK2-minus-CHK1 primary contrast missing")
    contrast_text = " ".join(
        (
            f"For {METRIC_LABELS[metric]}, chk-2 minus chk-1 is "
            f"{row['estimate_display']} (95% CI {row['ci_95']}; Holm-adjusted "
            f"p={row['holm_adjusted_p_display']})."
        )
        for metric, row in chk2_contrasts.items()
    )

    sources = []
    labels = {
        "model_summary.csv": "Pooled model similarity summary",
        "per_k_results.csv": "Per-replicate similarity estimates",
        "split_results.csv": "Split-specific similarity estimates",
        "pairwise_contrasts.csv": "Paired model contrasts and Holm adjustment",
        "results.json": "Sealed bootstrap results and delivery diagnostics",
        "score_manifest.json": "Sealed score-bundle manifest",
    }
    source_ids = {
        "model_summary.csv": "model_summary_source",
        "per_k_results.csv": "per_k_source",
        "split_results.csv": "split_source",
        "pairwise_contrasts.csv": "contrast_source",
        "results.json": "results_source",
        "score_manifest.json": "score_manifest_source",
    }
    for filename in REQUIRED_INPUTS:
        sources.append(
            _source(
                source_id=source_ids[filename],
                label=labels[filename],
                filename=filename,
                binding=source_bindings[filename],
                executed_at=generated_at,
            )
        )

    score_columns = [
        {"field": "model", "label": "Model", "type": "text"},
        {"field": "metric_label", "label": "Metric", "type": "text"},
        {"field": "estimate_display", "label": "Mean", "type": "text"},
        {"field": "ci_95", "label": "95% CI", "type": "text"},
        {"field": "samples", "label": "Samples", "format": "number"},
        {"field": "meetings", "label": "Meetings", "format": "number"},
        {"field": "replicates", "label": "k", "format": "number"},
    ]
    split_columns = [
        {"field": "split_label", "label": "Split", "type": "text"},
        *score_columns,
    ]
    policy_columns = [
        {"field": "policy_label", "label": "Scoring policy", "type": "text"},
        *score_columns,
    ]
    contrast_columns = [
        {"field": "contrast", "label": "Contrast", "type": "text"},
        {"field": "metric_label", "label": "Metric", "type": "text"},
        {"field": "estimate_display", "label": "Difference", "type": "text"},
        {"field": "ci_95", "label": "95% CI", "type": "text"},
        {"field": "p_value_display", "label": "Paired sign-flip p", "type": "text"},
        {"field": "holm_adjusted_p_display", "label": "Holm-adjusted p", "type": "text"},
        {"field": "holm_reject_005", "label": "Holm p < .05", "type": "boolean"},
    ]
    delivery_columns = [
        {"field": "model", "label": "Model", "type": "text"},
        {"field": "generations", "label": "Requested", "format": "number"},
        {"field": "delivery_valid", "label": "Delivery valid", "format": "number"},
        {"field": "delivery_invalid", "label": "Delivery invalid", "format": "number"},
        {"field": "recovered_nonempty", "label": "Nonempty recovered text", "format": "number"},
        {"field": "recovered_empty", "label": "Empty recovered text", "format": "number"},
        {"field": "delivery_valid_rate_display", "label": "Valid rate", "type": "text"},
        {"field": "failure_counts", "label": "Failure-code counts", "type": "text"},
    ]

    charts: list[dict[str, Any]] = []
    for metric in METRIC_ORDER:
        metric_slug = "mpnet" if metric == "mpnet_cosine" else "bert"
        charts.append(
            _chart(
                chart_id=f"pooled_{metric_slug}_chart",
                title=f"Pooled raw {METRIC_LABELS[metric]} mean and 95% CI",
                subtitle="Bars show means; exact meeting-cluster bootstrap 95% CIs appear in tooltips and the adjacent table.",
                chart_type="bar", dataset=f"pooled_{metric_slug}",
                source_id="model_summary_source", x_field="model", x_type="nominal",
                x_label="Model", y_field="estimate", y_label=METRIC_LABELS[metric],
                tooltip=(
                    {"field": "estimate_display", "type": "nominal", "label": "Mean"},
                    {"field": "ci_95", "type": "nominal", "label": "95% CI"},
                ),
            )
        )
        charts.append(
            _chart(
                chart_id=f"per_k_{metric_slug}_chart",
                title=f"{METRIC_LABELS[metric]} across k=1–10",
                subtitle="Each line follows one model across the ten stochastic generations; intervals remain available in the underlying table data.",
                chart_type="line", dataset=f"per_k_{metric_slug}",
                source_id="per_k_source", x_field="k", x_type="quantitative",
                x_label="Generation replicate k", y_field="estimate",
                y_label=METRIC_LABELS[metric], color_field="model", color_label="Model",
                tooltip=(
                    {"field": "k_label", "type": "nominal", "label": "Replicate"},
                    {"field": "estimate_display", "type": "nominal", "label": "Mean"},
                    {"field": "ci_95", "type": "nominal", "label": "95% CI"},
                ),
            )
        )
        charts.append(
            _chart(
                chart_id=f"split_{metric_slug}_chart",
                title=f"{METRIC_LABELS[metric]} by release split",
                subtitle="Train, validation, and test are original release labels, not leakage-safe evaluation partitions in this reconstruction diagnostic.",
                chart_type="bar", dataset=f"split_{metric_slug}",
                source_id="split_source", x_field="split_label", x_type="nominal",
                x_label="Release split", y_field="estimate", y_label=METRIC_LABELS[metric],
                color_field="model", color_label="Model",
                tooltip=(
                    {"field": "estimate_display", "type": "nominal", "label": "Mean"},
                    {"field": "ci_95", "type": "nominal", "label": "95% CI"},
                ),
            )
        )
        charts.append(
            _chart(
                chart_id=f"policy_{metric_slug}_chart",
                title=f"Raw versus delivery-penalized {METRIC_LABELS[metric]}",
                subtitle="Delivery-penalized scoring assigns zero to outputs that fail the delivery contract; it is a prespecified diagnostic, not the primary estimand.",
                chart_type="bar", dataset=f"policy_{metric_slug}",
                source_id="model_summary_source", x_field="model", x_type="nominal",
                x_label="Model", y_field="estimate", y_label=METRIC_LABELS[metric],
                color_field="policy_label", color_label="Policy",
                tooltip=(
                    {"field": "estimate_display", "type": "nominal", "label": "Mean"},
                    {"field": "ci_95", "type": "nominal", "label": "95% CI"},
                ),
            )
        )
    charts.extend(
        [
            _chart(
                chart_id="delivery_funnel_chart",
                title="Delivery funnel: valid output rate by model",
                subtitle="Every model has 3,910 requested generations; the table preserves valid, invalid, nonempty, and empty counts.",
                chart_type="bar", dataset="delivery", source_id="results_source",
                x_field="model", x_type="nominal", x_label="Model",
                y_field="delivery_valid_rate", y_label="Delivery-valid share",
                tooltip=(
                    {"field": "delivery_valid_rate_display", "type": "nominal", "label": "Valid rate"},
                    {"field": "delivery_valid", "type": "quantitative", "label": "Valid"},
                    {"field": "delivery_invalid", "type": "quantitative", "label": "Invalid"},
                ),
            ),
            _chart(
                chart_id="primary_contrast_chart",
                title="Primary pooled paired model contrasts",
                subtitle="Raw K10 point differences; exact 95% CIs and Holm-adjusted p-values are in the adjacent table.",
                chart_type="bar", dataset="primary_contrasts", source_id="contrast_source",
                x_field="series_label", x_type="nominal", x_label="Contrast and metric",
                y_field="estimate", y_label="Similarity difference",
                tooltip=(
                    {"field": "estimate_display", "type": "nominal", "label": "Difference"},
                    {"field": "ci_95", "type": "nominal", "label": "95% CI"},
                    {"field": "holm_adjusted_p_display", "type": "nominal", "label": "Holm-adjusted p"},
                ), reference_zero=True,
            ),
        ]
    )

    tables = [
        _table(table_id="pooled_table", title="Pooled raw means and 95% intervals",
               subtitle="Primary split-stratified meeting-cluster bootstrap; K10 pooled.",
               dataset="pooled_all", source_id="model_summary_source", columns=score_columns),
        _table(table_id="split_table", title="Release-split comparison",
               subtitle="Descriptive split-specific estimates; labels do not confer held-out status.",
               dataset="split_all", source_id="split_source", columns=split_columns, density="dense"),
        _table(table_id="policy_table", title="Raw and delivery-penalized estimates",
               subtitle="Both policies use the same generation matrix.", dataset="policy_all",
               source_id="model_summary_source", columns=policy_columns, density="dense"),
        _table(table_id="delivery_table", title="Delivery funnel counts",
               subtitle="Requested generations, recovered text, and delivery-contract outcomes.",
               dataset="delivery", source_id="results_source", columns=delivery_columns),
        _table(table_id="contrast_table", title="Primary six-test contrast family",
               subtitle="Three paired model contrasts × two metrics; family-wise adjustment is Holm.",
               dataset="primary_contrasts", source_id="contrast_source", columns=contrast_columns),
    ]

    cards = [
        {
            "id": "coverage_card", "description": "Training-release reconstruction coverage.",
            "dataset": "coverage", "sourceId": "results_source",
            "metrics": [
                {"label": "Samples", "field": "samples", "format": "number"},
                {"label": "Meetings", "field": "meetings", "format": "number"},
            ],
        },
        {
            "id": "generation_card", "description": "Three models × 391 samples × k=10.",
            "dataset": "coverage", "sourceId": "results_source",
            "metrics": [
                {"label": "Generation rows", "field": "generation_rows", "format": "number"},
                {"label": "Replicates", "field": "replicates", "format": "number"},
            ],
        },
    ]

    limitations = list(dict.fromkeys([SCOPE_DISCLOSURE, REFERENCE_DISCLOSURE, METRIC_DISCLOSURE, *model["limitations"]]))
    blocks: list[dict[str, Any]] = [
        {"id": "title", "type": "markdown", "body": "# Textual Similarity of Minutes-Style Paragraph Rewriting"},
        {"id": "scope", "type": "markdown", "sourceId": "score_manifest_source",
         "body": f"## Scope and interpretation\n\n**{SCOPE_DISCLOSURE}** {REFERENCE_DISCLOSURE}"},
        {"id": "summary", "type": "markdown", "sourceId": "model_summary_source",
         "body": f"## Technical summary\n\n{winner_text} {contrast_text} These are descriptive reconstruction results, not evidence of out-of-sample performance."},
        {"id": "coverage", "type": "metric-strip", "cardIds": ["coverage_card", "generation_card"]},
        {"id": "pooled_intro", "type": "markdown", "sourceId": "model_summary_source",
         "body": "## Pooled raw similarity\n\nThe primary view averages the ten stochastic generations within each meeting and then gives meetings equal weight. Bars display raw best-effort means. Because the native portable chart contract has no error-bar mark, exact 95% intervals are carried in tooltips and the immediately adjacent table."},
        {"id": "pooled_mpnet", "type": "chart", "chartId": "pooled_mpnet_chart", "layout": "full"},
        {"id": "pooled_bert", "type": "chart", "chartId": "pooled_bert_chart", "layout": "full"},
        {"id": "pooled_table_block", "type": "table", "tableId": "pooled_table", "layout": "full"},
        {"id": "k_intro", "type": "markdown", "sourceId": "per_k_source",
         "body": "## Stochastic stability across k=1–10\n\nThe two curves retain each generation replicate rather than hiding variability in the pooled K10 mean. The ten replicates are repeated generations of the same 391 samples, not independent datasets."},
        {"id": "k_mpnet", "type": "chart", "chartId": "per_k_mpnet_chart", "layout": "full"},
        {"id": "k_bert", "type": "chart", "chartId": "per_k_bert_chart", "layout": "full"},
        {"id": "split_intro", "type": "markdown", "sourceId": "split_source",
         "body": "## Release-split comparison\n\nSplit estimates diagnose heterogeneity across the original train, validation, and test labels. All 391 rows belong to the training-only release used for this reconstruction exercise, so none of these panels is a leakage-safe generalization test."},
        {"id": "split_mpnet", "type": "chart", "chartId": "split_mpnet_chart", "layout": "full"},
        {"id": "split_bert", "type": "chart", "chartId": "split_bert_chart", "layout": "full"},
        {"id": "split_table_block", "type": "table", "tableId": "split_table", "layout": "full"},
        {"id": "policy_intro", "type": "markdown", "sourceId": "model_summary_source",
         "body": "## Raw versus delivery-penalized scoring\n\nRaw best effort scores recovered final text and assigns zero only to empty recovery. The delivery-penalized sensitivity assigns zero whenever the generation fails the delivery contract, exposing whether model comparisons depend on format compliance."},
        {"id": "policy_mpnet", "type": "chart", "chartId": "policy_mpnet_chart", "layout": "full"},
        {"id": "policy_bert", "type": "chart", "chartId": "policy_bert_chart", "layout": "full"},
        {"id": "policy_table_block", "type": "table", "tableId": "policy_table", "layout": "full"},
        {"id": "delivery_intro", "type": "markdown", "sourceId": "results_source",
         "body": "## Delivery funnel\n\nThe funnel separates requested generations, recoverable text, and delivery-valid outputs. Counts matter because a similarity score alone can obscure malformed or empty model responses."},
        {"id": "delivery_chart", "type": "chart", "chartId": "delivery_funnel_chart", "layout": "full"},
        {"id": "delivery_table_block", "type": "table", "tableId": "delivery_table", "layout": "full"},
        {"id": "contrast_intro", "type": "markdown", "sourceId": "contrast_source",
         "body": "## Paired contrasts and multiplicity\n\nThe primary family contains all three model contrasts for both metrics. Shared bootstrap indices preserve pairing, and six descriptive sign-flip probabilities are adjusted together with Holm's method. Read direction, interval, and adjusted probability jointly."},
        {"id": "contrast_chart", "type": "chart", "chartId": "primary_contrast_chart", "layout": "full"},
        {"id": "contrast_table_block", "type": "table", "tableId": "contrast_table", "layout": "full"},
        {"id": "methods", "type": "markdown", "sourceId": "results_source",
         "body": "## Methods\n\nFor each model and sample, ten generations were drawn at temperature 0.6 and top-p 0.9. The primary analysis uses raw best-effort answer text, equal meeting weighting, and a split-stratified meeting-cluster shared-index paired percentile bootstrap with 1,000 draws. MPNet cosine and BERTScore F1 are computed against the synthetic teacher rewrite. A split-stratified row bootstrap is retained only as a dependence-ignoring sensitivity analysis."},
        {"id": "limitations", "type": "markdown", "sourceId": "score_manifest_source",
         "body": "## Limitations\n\n" + "\n".join(f"- {item}" for item in limitations)},
        {"id": "next_steps", "type": "markdown", "sourceId": "score_manifest_source",
         "body": "## What this diagnostic can and cannot support\n\nUse these results to compare reconstruction resemblance and delivery behavior on the frozen 391-row release. Do not use them to claim leakage-safe generalization, factual superiority, or similarity to official Minutes. A separate held-out source set, factual-entailment audit, and official-style assessment are required for those claims."},
        {"id": "provenance", "type": "markdown", "sourceId": "score_manifest_source",
         "body": "## Provenance\n\nEvery displayed statistic is projected from the deeply replayed, sealed score bundle. This renderer does not perform generation, embedding inference, bootstrap resampling, or score modification."},
    ]

    # Portable widget cells accept scalars only.  The authoritative coverage
    # object also contains nested split_rows/split_meetings mappings for audit
    # purposes, but the metric cards consume only these four scalar totals.
    # Project the card dataset explicitly instead of leaking the nested audit
    # mappings into a widget row.
    coverage_row = {
        field: copy.deepcopy(model["coverage"][field])
        for field in ("samples", "meetings", "replicates", "generation_rows")
    }
    datasets = {
        "coverage": [coverage_row],
        "pooled_all": copy.deepcopy(primary),
        "pooled_mpnet": [row for row in copy.deepcopy(primary) if row["metric"] == "mpnet_cosine"],
        "pooled_bert": [row for row in copy.deepcopy(primary) if row["metric"] == "bertscore_f1"],
        "per_k_mpnet": [row for row in copy.deepcopy(per_k) if row["metric"] == "mpnet_cosine"],
        "per_k_bert": [row for row in copy.deepcopy(per_k) if row["metric"] == "bertscore_f1"],
        "split_all": copy.deepcopy(splits),
        "split_mpnet": [row for row in copy.deepcopy(splits) if row["metric"] == "mpnet_cosine"],
        "split_bert": [row for row in copy.deepcopy(splits) if row["metric"] == "bertscore_f1"],
        "policy_all": copy.deepcopy(policies),
        "policy_mpnet": [row for row in copy.deepcopy(policies) if row["metric"] == "mpnet_cosine"],
        "policy_bert": [row for row in copy.deepcopy(policies) if row["metric"] == "bertscore_f1"],
        "delivery": copy.deepcopy(delivery),
        "primary_contrasts": copy.deepcopy(contrasts),
    }
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Textual Similarity of Minutes-Style Paragraph Rewriting",
            "description": "Technical in-sample reconstruction diagnostic for chk-0, chk-1, and chk-2.",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {"version": 1, "generatedAt": generated_at, "status": "ready", "datasets": datasets},
        "sources": copy.deepcopy(sources),
    }


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _resolve_node(node_bin: Path | None) -> Path:
    candidates = (node_bin,) if node_bin is not None else DEFAULT_NODE_CANDIDATES
    for candidate in candidates:
        if candidate is None:
            continue
        path = candidate.expanduser().resolve()
        if path.is_file() and not path.is_symlink() and os.access(path, os.X_OK):
            return path
    system = shutil.which("node")
    if node_bin is None and system:
        return _regular_file(Path(system), label="node executable")
    raise PaperChk2SimilarityReportError("no executable Node.js runtime found")


def _resolve_packager(plugin_root: Path | None) -> tuple[Path, Path]:
    root = (plugin_root or DEFAULT_PLUGIN_ROOT).expanduser().resolve()
    script = root / PACKAGER_RELATIVE
    package = root / "package.json"
    _regular_file(script, label="portable report builder")
    _regular_file(package, label="Data Analytics plugin package")
    return root, script


def _package_report(
    *, artifact_path: Path, report_path: Path, node_bin: Path | None,
    plugin_root: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    node = _resolve_node(node_bin)
    plugin, script = _resolve_packager(plugin_root)
    try:
        completed = subprocess.run(
            [str(node), str(script), "--input", str(artifact_path), "--output", str(report_path), "--timeout-ms", "30000"],
            cwd=plugin, check=False, capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PaperChk2SimilarityReportError(f"portable renderer failed: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        raise PaperChk2SimilarityReportError(
            f"portable renderer failed: {detail[-1] if detail else completed.returncode}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise PaperChk2SimilarityReportError("portable renderer emitted ambiguous receipt")
    try:
        receipt = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise PaperChk2SimilarityReportError("portable renderer receipt is invalid") from exc
    stages = receipt.get("stages") if isinstance(receipt, Mapping) else None
    if (
        receipt.get("ok") is not True
        or not isinstance(stages, Mapping)
        or stages.get("validation") != "passed"
        or stages.get("package") != "passed"
        or stages.get("verification") not in {"passed", "structural_only"}
        or not report_path.is_file()
    ):
        raise PaperChk2SimilarityReportError("portable renderer did not complete")
    return copy.deepcopy(dict(receipt)), {
        "node": _binding(node),
        "packager_script": _binding(script),
        "plugin_package": _binding(plugin / "package.json"),
    }


def _intended_binding(staged: Path, final: Path) -> dict[str, Any]:
    result = _binding(staged)
    result["path"] = str(final.expanduser().resolve())
    return result


def _rename_noreplace(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise PaperChk2SimilarityReportError(f"report directory already exists: {destination}")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError):
        renameat2 = None
    if renameat2 is not None:
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise PaperChk2SimilarityReportError(f"report directory already exists: {destination}")
        if error_number not in {errno.ENOSYS, errno.EINVAL}:
            raise OSError(error_number, os.strerror(error_number), str(destination))
    if destination.exists() or destination.is_symlink():
        raise PaperChk2SimilarityReportError(f"report directory already exists: {destination}")
    os.rename(source, destination)


def _seal_permissions(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise PaperChk2SimilarityReportError(f"report output contains symlink: {path}")
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def validate_existing_report(*, score_dir: Path, report_dir: Path) -> dict[str, Any]:
    """Deep-validate a create-only report for ``--resume`` no-op reuse."""

    output = report_dir.expanduser().resolve()
    if report_dir.expanduser().is_symlink() or output.is_symlink() or not output.is_dir():
        raise PaperChk2SimilarityReportError(
            f"resume report directory must be a regular non-symlink directory: {output}"
        )
    allowed = {"artifact.json", "report.html", "manifest.json"}
    observed_names = {path.name for path in output.iterdir()}
    if observed_names != allowed:
        raise PaperChk2SimilarityReportError(
            f"resume report inventory drift: expected {sorted(allowed)}, "
            f"observed {sorted(observed_names)}"
        )
    manifest = _read_json(output / "manifest.json", label="report manifest", sealed=True)
    if (
        manifest.get("schema_version") != REPORT_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("immutable") is not True
        or manifest.get("report_id") != REPORT_ID
        or manifest.get("operation")
        != "render_only_no_generation_scoring_or_bootstrap_recomputation"
    ):
        raise PaperChk2SimilarityReportError("resume report manifest contract drift")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise PaperChk2SimilarityReportError("resume report artifact inventory missing")
    expected_paths = {
        "canonical_artifact": output / "artifact.json",
        "portable_report": output / "report.html",
    }
    for artifact_id, path in expected_paths.items():
        expected = artifacts.get(artifact_id)
        if not isinstance(expected, Mapping):
            raise PaperChk2SimilarityReportError(
                f"resume report does not bind {artifact_id}"
            )
        observed = _binding(path)
        for field in ("path", "bytes", "sha256"):
            if expected.get(field) != observed[field]:
                raise PaperChk2SimilarityReportError(
                    f"resume {artifact_id} binding drift: {field}"
                )

    model = load_report_model(score_dir)
    if manifest.get("score_manifest") != model["source_bindings"]["score_manifest.json"]:
        raise PaperChk2SimilarityReportError("resume score-manifest binding drift")
    artifact = _read_json(output / "artifact.json", label="canonical report artifact")
    expected_artifact = build_canonical_artifact(model)
    if artifact != expected_artifact:
        raise PaperChk2SimilarityReportError(
            "resume canonical artifact differs from current validated score bundle"
        )
    artifact_text = _canonical(artifact)
    try:
        html_text = (output / "report.html").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PaperChk2SimilarityReportError(f"cannot read portable report: {exc}") from exc
    for disclosure in (SCOPE_DISCLOSURE, REFERENCE_DISCLOSURE, METRIC_DISCLOSURE):
        if disclosure not in artifact_text:
            raise PaperChk2SimilarityReportError(
                "resume canonical artifact lost a required disclosure"
            )
    # The portable renderer HTML-escapes the literal reasoning boundary.
    html_disclosures = (
        SCOPE_DISCLOSURE,
        REFERENCE_DISCLOSURE.replace("</think>", "&lt;/think&gt;"),
        METRIC_DISCLOSURE,
    )
    for disclosure in html_disclosures:
        if disclosure not in html_text:
            raise PaperChk2SimilarityReportError(
                "resume portable report lost a required disclosure"
            )
    return copy.deepcopy(manifest)


def render_and_seal_report(
    *, score_dir: Path, report_dir: Path, node_bin: Path | None = None,
    plugin_root: Path | None = None, resume: bool = False,
) -> dict[str, Any]:
    output = report_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        if resume and output.exists() and not output.is_symlink():
            return validate_existing_report(score_dir=score_dir, report_dir=output)
        raise PaperChk2SimilarityReportError(f"report directory already exists: {output}")
    model = load_report_model(score_dir)
    artifact = build_canonical_artifact(model)
    artifact_text = _canonical(artifact)
    for disclosure in (SCOPE_DISCLOSURE, REFERENCE_DISCLOSURE, METRIC_DISCLOSURE):
        if disclosure not in artifact_text:
            raise PaperChk2SimilarityReportError("canonical artifact lost required disclosure")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)).resolve()
    published = False
    try:
        artifact_path = staging / "artifact.json"
        report_path = staging / "report.html"
        _write_new_json(artifact_path, artifact)
        receipt, builder_bindings = _package_report(
            artifact_path=artifact_path, report_path=report_path,
            node_bin=node_bin, plugin_root=plugin_root,
        )
        refreshed = load_report_model(score_dir)
        if refreshed != model or build_canonical_artifact(refreshed) != artifact:
            raise PaperChk2SimilarityReportError("score bundle changed during rendering")
        final_artifact = output / "artifact.json"
        final_report = output / "report.html"
        manifest = seal_manifest(
            {
                "schema_version": REPORT_MANIFEST_SCHEMA,
                "status": "complete",
                "immutable": True,
                "report_id": REPORT_ID,
                "created_at_utc": model["generated_at_utc"],
                "operation": "render_only_no_generation_scoring_or_bootstrap_recomputation",
                "score_manifest": copy.deepcopy(model["source_bindings"]["score_manifest.json"]),
                "artifacts": {
                    "canonical_artifact": {
                        **_intended_binding(artifact_path, final_artifact),
                        "media_type": "application/json",
                        "contract": REPORT_ARTIFACT_CONTRACT,
                    },
                    "portable_report": {
                        **_intended_binding(report_path, final_report),
                        "media_type": "text/html; charset=utf-8",
                        "self_contained": True,
                        "network_dependencies": False,
                    },
                },
                "report_builder": builder_bindings,
                "delivery_receipt": receipt,
                "renderer": _binding(Path(__file__)),
                "report_contract": {
                    "audience": "technical",
                    "delivery_mode": "portable_html",
                    "evaluation_role": "in_sample_training_release_reconstruction_diagnostic",
                    "leakage_safe": False,
                    "native_chart_and_table_blocks_only": True,
                    "pooled_raw_means_and_95pct_ci_displayed": True,
                    "k1_to_k10_curves_displayed": True,
                    "split_comparison_displayed": True,
                    "raw_vs_penalized_displayed": True,
                    "delivery_funnel_displayed": True,
                    "paired_contrasts_and_holm_displayed": True,
                    "chart_contract": {
                        "mean_marks": "native categorical bars",
                        "confidence_intervals": "exact values in tooltips and adjacent tables",
                        "interval_mark_omission_reason": "The canonical portable chart contract has no native error-bar mark.",
                    },
                },
                "limitations": list(dict.fromkeys([SCOPE_DISCLOSURE, REFERENCE_DISCLOSURE, METRIC_DISCLOSURE, *model["limitations"]])),
            }
        )
        _write_new_json(staging / "manifest.json", manifest)
        validate_manifest_integrity(manifest)
        _seal_permissions(staging)
        _fsync_directory(staging)
        _rename_noreplace(staging, output)
        published = True
        _fsync_directory(output.parent)
        return copy.deepcopy(manifest)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-dir", "--score-root", dest="score_dir", type=Path,
        default=DEFAULT_SCORE_DIR,
    )
    parser.add_argument(
        "--report-dir", "--report-root", dest="report_dir", type=Path,
        default=DEFAULT_REPORT_DIR,
    )
    parser.add_argument("--node-bin", type=Path)
    parser.add_argument("--plugin-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    existed = args.report_dir.expanduser().resolve().exists()
    manifest = render_and_seal_report(
        score_dir=args.score_dir,
        report_dir=args.report_dir,
        node_bin=args.node_bin,
        plugin_root=args.plugin_root,
        resume=args.resume,
    )
    status = "already_complete_validated" if existed and args.resume else "complete"
    print(_canonical({"status": status, "report_dir": str(args.report_dir.resolve()), "manifest_payload_sha256": manifest["integrity"]["payload_sha256"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
