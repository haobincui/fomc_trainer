#!/usr/bin/env python3
"""Prepare, estimate, and score the reduced-predictor van den Hauwe model.

This job is deliberately independent of the decision-comparison evaluator.  It
implements the publication boundary for an architecture-faithful, K=3 nested
version of van den Hauwe et al. (2013); it must not be described as the
published K=33 replication.  The estimator lives behind the single
generic sampler and recursive-importance APIs in
``open_r1.utils.van_den_hauwe_2013``.

Inputs are accepted only from a SHA-bound source release containing real-time
vintages.  Missing values, current-vintage fallbacks, unbound source files, or
information dated after the monthly forecast origin fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION = Path(__file__).resolve()
DEFAULT_CONFIG = ROOT / "configs/main/van_den_hauwe_2013_reduced_v1.json"
CONFIG_SCHEMA = "van-den-hauwe-2013-reduced-config-v1"
SOURCE_SCHEMA = "van-den-hauwe-2013-reduced-source-v1"
PANEL_SCHEMA = "van-den-hauwe-2013-reduced-monthly-panel-v1"
PREDICTION_SCHEMA = "van-den-hauwe-2013-reduced-prediction-v1"
MANIFEST_SCHEMA = "van-den-hauwe-2013-reduced-run-v1"
SUMMARY_SCHEMA = "van-den-hauwe-2013-reduced-summary-v2"
STATUS_SCHEMA = "van-den-hauwe-2013-reduced-status-v1"
MODEL_ID = "van_den_hauwe_2013_reduced_predictor"
CONTRACTS = ("matched_h19_posterior_imputation", "paper_recursive_oos")
DIRECTIONS = ("cut", "hold", "hike")
MIN_RECURSIVE_PRE_UPDATE_ESS = 100.0
RECURSIVE_ESS_UNAVAILABLE_REASON = (
    "recursive_importance_pre_update_ess_below_100"
)
FROZEN_PREDICTORS = (
    ("6TFF", "av", 1, "historical_market_nonrevised"),
    ("IP", "gr", 1, "real_time_vintage"),
    ("INF", "gr", 1, "real_time_vintage"),
)
HEX_DIGITS = frozenset("0123456789abcdef")


class VanDenHauweReducedError(RuntimeError):
    """A frozen input, vintage, estimation, or publication contract failed."""


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    monthly_panel: Path
    predictions: Path
    summary: Path
    manifest: Path
    status: Path


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and set(text) <= HEX_DIGITS


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VanDenHauweReducedError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VanDenHauweReducedError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise VanDenHauweReducedError(f"cannot read JSONL {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VanDenHauweReducedError(
                f"malformed JSONL {path}:{number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise VanDenHauweReducedError(f"non-object JSONL row {path}:{number}")
        rows.append(value)
    return rows


def _atomic_write(path: Path, content: bytes, *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise VanDenHauweReducedError(f"create-only output already exists: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            try:
                os.link(temporary_path, path)
            except FileExistsError as exc:
                raise VanDenHauweReducedError(
                    f"create-only output already exists: {path}"
                ) from exc
            temporary_path.unlink()
        else:
            os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_json(path: Path, value: Mapping[str, Any], *, resume: bool = False) -> None:
    content = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    if resume and path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
            raise VanDenHauweReducedError(f"resume output drift: {path}")
        return
    _atomic_write(path, content, exclusive=True)


def write_jsonl(
    path: Path, rows: Iterable[Mapping[str, Any]], *, resume: bool = False
) -> int:
    values = [dict(row) for row in rows]
    content = "".join(canonical_json(row) + "\n" for row in values).encode("utf-8")
    if resume and path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
            raise VanDenHauweReducedError(f"resume output drift: {path}")
        return len(values)
    _atomic_write(path, content, exclusive=True)
    return len(values)


def write_status(
    path: Path, *, phase: str, state: str, detail: Mapping[str, Any] | None = None
) -> None:
    payload = {
        "schema_version": STATUS_SCHEMA,
        "updated_at_utc": utc_now(),
        "phase": phase,
        "state": state,
        "detail": dict(detail or {}),
    }
    _atomic_write(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        exclusive=False,
    )


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _month_index(month: str) -> int:
    try:
        parsed = date.fromisoformat(f"{month}-01")
    except ValueError as exc:
        raise VanDenHauweReducedError(f"invalid canonical month: {month!r}") from exc
    if parsed.strftime("%Y-%m") != month:
        raise VanDenHauweReducedError(f"invalid canonical month: {month!r}")
    return parsed.year * 12 + parsed.month - 1


def _month_sequence(start: str, end: str) -> list[str]:
    first = _month_index(start)
    last = _month_index(end)
    if last < first:
        raise VanDenHauweReducedError("sample end precedes sample start")
    return [f"{value // 12:04d}-{value % 12 + 1:02d}" for value in range(first, last + 1)]


def _canonical_date(value: Any, *, label: str) -> str:
    text = str(value or "")
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise VanDenHauweReducedError(f"invalid {label}: {value!r}") from exc
    if parsed.isoformat() != text:
        raise VanDenHauweReducedError(f"non-canonical {label}: {value!r}")
    return text


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise VanDenHauweReducedError("unsupported reduced VDH config schema")
    if config.get("model_id") != MODEL_ID:
        raise VanDenHauweReducedError("frozen model id drift")
    sample = config.get("sample")
    if not isinstance(sample, dict) or sample != {
        "start_month": "1990-01",
        "end_month": "2008-06",
        "expected_months": 222,
        "expected_observed_decision_months": 157,
        "frequency": "monthly",
    }:
        raise VanDenHauweReducedError("frozen 1990-01--2008-06 sample drift")
    predictors = config.get("predictors")
    if not isinstance(predictors, list):
        raise VanDenHauweReducedError("predictor specification missing")
    observed = tuple(
        (
            row.get("paper_id"),
            row.get("transform"),
            row.get("m"),
            row.get("source_policy"),
        )
        for row in predictors
        if isinstance(row, dict)
    )
    if observed != FROZEN_PREDICTORS:
        raise VanDenHauweReducedError(
            "reduced predictor contract must be 6TFF/IP/INF with av/gr/gr, m=1, and frozen source policies"
        )
    source = config.get("source_release")
    if (
        not isinstance(source, dict)
        or source.get("required_manifest_schema") != SOURCE_SCHEMA
        or source.get("real_time_vintages_required_for") != ["IP", "INF"]
        or source.get("historical_market_nonrevised_allowed_for") != ["6TFF"]
        or source.get("current_vintage_fallback_allowed") is not False
    ):
        raise VanDenHauweReducedError("real-time source-release contract drift")
    contracts = config.get("contracts")
    if not isinstance(contracts, dict) or tuple(contracts) != CONTRACTS:
        raise VanDenHauweReducedError("frozen evaluation contracts drift")
    model = config.get("model")
    if (
        not isinstance(model, dict)
        or model.get("implementation_module") != "open_r1.utils.van_den_hauwe_2013"
        or model.get("ordered_classes") != list(DIRECTIONS)
        or model.get("latent_error") != "AR(1)"
        or model.get("variable_selection") != "Kuo-Mallick"
        or model.get("coefficient_prior_variance") != 16.0
        or model.get("simulation_protocol")
        != "discard the 50,000-run initialization/convergence period documented in van den Hauwe's thesis Chapter 2 Appendix B, then retain the following 100,000 consecutive draws"
        or model.get("simulation_protocol_source")
        != "https://repub.eur.nl/pub/80126/vanderHauwe.pdf"
        or model.get("inclusion_probability_prior")
        != {"distribution": "Beta", "alpha": 1.0, "beta": 1.0}
        or model.get("mcmc")
        != {
            "draws": 100000,
            "burn_in": 50000,
            "thin": 1,
            "phi_proposal_sd": 0.12,
            "standardize_predictors": True,
        }
    ):
        raise VanDenHauweReducedError("model implementation contract drift")
    scoring = config.get("scoring")
    if (
        not isinstance(scoring, dict)
        or scoring.get("direction_order") != list(DIRECTIONS)
        or scoring.get("pooled_contracts_allowed") is not False
    ):
        raise VanDenHauweReducedError("scoring contract drift")
    return config


def runtime_paths(config: Mapping[str, Any], output_root: Path | None) -> RuntimePaths:
    root = resolve_path(output_root or str(config["default_output_root"]))
    return RuntimePaths(
        root=root,
        monthly_panel=root / "monthly_panel.jsonl",
        predictions=root / "predictions.jsonl",
        summary=root / "summary.json",
        manifest=root / "manifest.json",
        status=root / "status.json",
    )


def _safe_release_file(root: Path, relative: Any, *, label: str) -> Path:
    text = str(relative or "")
    candidate = Path(text)
    if not text or candidate.is_absolute() or ".." in candidate.parts:
        raise VanDenHauweReducedError(f"unsafe {label} path: {relative!r}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise VanDenHauweReducedError(f"{label} escapes source root: {relative!r}") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise VanDenHauweReducedError(f"missing or unsafe {label}: {resolved}")
    return resolved


def _frozen_config_binding(config_path: Path) -> tuple[str, str]:
    """Return the repository-relative path and SHA256 for the active config."""

    candidate = config_path.expanduser()
    resolved = candidate.resolve()
    if candidate.is_symlink() or not resolved.is_file():
        raise VanDenHauweReducedError(f"missing or unsafe active config: {resolved}")
    try:
        relative = resolved.relative_to(ROOT)
    except ValueError as exc:
        raise VanDenHauweReducedError(
            f"active config must be frozen inside the repository: {resolved}"
        ) from exc
    return relative.as_posix(), sha256_file(resolved)


def validate_source_release(
    config: Mapping[str, Any],
    source_root: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], Path, dict[str, list[dict[str, Any]]]]:
    source_spec = config["source_release"]
    manifest_path = _safe_release_file(
        source_root, source_spec["manifest"], label="source manifest"
    )
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != SOURCE_SCHEMA:
        raise VanDenHauweReducedError("unsupported source-release manifest schema")
    expected_config_path, expected_config_sha256 = _frozen_config_binding(config_path)
    if manifest.get("config") != expected_config_path:
        raise VanDenHauweReducedError(
            "source-release config path does not bind the active frozen config"
        )
    if manifest.get("config_sha256") != expected_config_sha256:
        raise VanDenHauweReducedError(
            "source-release config SHA256 does not bind the active frozen config"
        )
    if manifest.get("current_vintage_fallback_allowed") is not False:
        raise VanDenHauweReducedError("current-vintage fallback is forbidden")
    manifest_predictors = manifest.get("predictors")
    expected_predictors = [row[0] for row in FROZEN_PREDICTORS]
    if not isinstance(manifest_predictors, list) or [
        row.get("paper_id") for row in manifest_predictors if isinstance(row, dict)
    ] != expected_predictors:
        raise VanDenHauweReducedError("source-release predictor ids/order drift")
    expected_source_specs = {
        "6TFF": ("av", 1, "historical_market_nonrevised"),
        "IP": ("gr", 1, "alfred_end_of_prior_month_vintage"),
        "INF": ("gr", 1, "alfred_end_of_prior_month_vintage"),
    }
    for raw in manifest_predictors:
        paper_id = str(raw["paper_id"])
        observed = (raw.get("transform"), raw.get("m"), raw.get("source_policy"))
        if observed != expected_source_specs[paper_id]:
            raise VanDenHauweReducedError(
                f"source-release transformation/time contract drift: {paper_id}"
            )
    sample = manifest.get("sample")
    if (
        not isinstance(sample, dict)
        or sample.get("start_month") != "1990-01"
        or sample.get("end_month") != "2008-06"
        or sample.get("months") != 222
        or sample.get("decision_months") != 157
        or sample.get("direction_counts") != {"cut": 40, "hold": 86, "hike": 31}
    ):
        raise VanDenHauweReducedError("source-release sample closure drift")
    if not str(manifest.get("market_history_caveat") or "").strip():
        raise VanDenHauweReducedError("6TFF market-history limitation is undisclosed")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise VanDenHauweReducedError("source-release files ledger missing")
    ledger: dict[str, list[dict[str, Any]]] = {}
    path_seen: set[str] = set()
    for index, raw in enumerate(files):
        if not isinstance(raw, dict):
            raise VanDenHauweReducedError(f"non-object source file ledger row {index}")
        series_id = str(raw.get("series_id") or "")
        relative = str(raw.get("path") or "")
        expected_sha = str(raw.get("sha256") or "")
        if not series_id:
            raise VanDenHauweReducedError(f"source series id missing in ledger row {index}")
        if not _is_sha256(expected_sha):
            raise VanDenHauweReducedError(f"invalid source SHA256 ledger row {index}")
        if relative in path_seen:
            raise VanDenHauweReducedError(f"duplicate source file path: {relative}")
        file_path = _safe_release_file(source_root, relative, label="source artifact")
        observed_sha = sha256_file(file_path)
        if observed_sha != expected_sha:
            raise VanDenHauweReducedError(f"source artifact SHA256 mismatch: {relative}")
        record = dict(raw)
        ledger.setdefault(series_id, []).append(record)
        path_seen.add(relative)
    monthly_entries = ledger.get("monthly_source", [])
    if len(monthly_entries) != 1 or monthly_entries[0]["path"] != source_spec["monthly_source"]:
        raise VanDenHauweReducedError("monthly source is not bound at the frozen path")
    required_singletons = ("DFEDTAR", "TB6MS", "FEDFUNDS")
    if any(len(ledger.get(series_id, [])) != 1 for series_id in required_singletons):
        raise VanDenHauweReducedError("target/market source ledger closure failed")
    expected_vintages = {
        _previous_month_end(month)
        for month in _month_sequence("1990-01", "2008-06")
    }
    for series_id in ("INDPRO", "CPIAUCSL"):
        records = ledger.get(series_id, [])
        covered: list[str] = []
        for record in records:
            vintages = record.get("vintage_dates")
            if not isinstance(vintages, list) or not vintages:
                raise VanDenHauweReducedError(
                    f"ALFRED vintage ledger missing dates: {series_id}"
                )
            covered.extend(str(value) for value in vintages)
        if set(covered) != expected_vintages or len(covered) != len(expected_vintages):
            raise VanDenHauweReducedError(
                f"ALFRED vintage ledger coverage drift: {series_id}"
            )
    monthly_path = _safe_release_file(
        source_root, monthly_entries[0]["path"], label="monthly source"
    )
    if manifest.get("monthly_source_sha256") != monthly_entries[0]["sha256"]:
        raise VanDenHauweReducedError("monthly source top-level SHA256 drift")
    return manifest, monthly_path, ledger


def _previous_month_end(month: str) -> str:
    first = date.fromisoformat(f"{month}-01")
    return (first - timedelta(days=1)).isoformat()


def _ledger_hash_for_vintage(
    ledger: Mapping[str, Sequence[Mapping[str, Any]]], series_id: str, vintage: str
) -> str:
    matches = [
        str(record["sha256"])
        for record in ledger.get(series_id, [])
        if vintage in record.get("vintage_dates", [])
    ]
    if len(matches) != 1:
        raise VanDenHauweReducedError(
            f"vintage is not uniquely manifest-bound: {series_id}/{vintage}"
        )
    return matches[0]


def _h19_by_month(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    path = resolve_path(config["historical_h19"])
    payload = read_json(path)
    meetings = payload.get("meetings")
    if not isinstance(meetings, list) or len(meetings) != 19:
        raise VanDenHauweReducedError("historical H19 roster must contain 19 meetings")
    result: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for index, raw in enumerate(meetings):
        if not isinstance(raw, dict):
            raise VanDenHauweReducedError(f"non-object H19 row {index}")
        start = _canonical_date(raw.get("meeting_start_date"), label="H19 start")
        cutoff = _canonical_date(raw.get("evidence_cutoff"), label="H19 cutoff")
        direction = str(raw.get("direction") or "")
        month = start[:7]
        if month in result:
            raise VanDenHauweReducedError(f"multiple H19 meetings in month {month}")
        if direction not in DIRECTIONS or not cutoff < start:
            raise VanDenHauweReducedError(f"invalid H19 target row: {month}")
        result[month] = {
            "meeting_id": str(raw.get("meeting_id") or ""),
            "meeting_start_date": start,
            "evidence_cutoff": cutoff,
            "target_direction": direction,
            "h19_source_path": str(path.relative_to(ROOT)),
            "h19_source_sha256": sha256_file(path),
        }
        counts[direction] += 1
    if counts != Counter({"cut": 4, "hold": 10, "hike": 5}):
        raise VanDenHauweReducedError(f"H19 direction counts drift: {dict(counts)}")
    return result


def prepare_monthly_panel(
    config: Mapping[str, Any],
    source_root: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_manifest, monthly_path, ledger = validate_source_release(
        config, source_root, config_path=config_path
    )
    raw_rows = read_jsonl(monthly_path)
    expected_months = _month_sequence(
        config["sample"]["start_month"], config["sample"]["end_month"]
    )
    if len(raw_rows) != len(expected_months) or len(raw_rows) != 222:
        raise VanDenHauweReducedError(
            f"monthly source row count drift: {len(raw_rows)} != 222"
        )
    h19 = _h19_by_month(config)
    predictor_specs = {row["paper_id"]: row for row in config["predictors"]}
    panel: list[dict[str, Any]] = []
    for index, (raw, expected_month) in enumerate(zip(raw_rows, expected_months, strict=True)):
        month = str(raw.get("month") or "")
        if month != expected_month:
            raise VanDenHauweReducedError(
                f"monthly source sequence drift at row {index}: {month} != {expected_month}"
            )
        if raw.get("schema_version") != "van-den-hauwe-2013-reduced-month-v1":
            raise VanDenHauweReducedError(f"monthly source schema drift in {month}")
        expected_source_row_hash = sha256_bytes(
            canonical_json(
                {key: value for key, value in raw.items() if key != "row_sha256"}
            ).encode("utf-8")
        )
        if raw.get("row_sha256") != expected_source_row_hash:
            raise VanDenHauweReducedError(f"monthly source row SHA256 drift in {month}")
        forecast_origin = _canonical_date(
            raw.get("information_cutoff"), label=f"information cutoff {month}"
        )
        if forecast_origin != _previous_month_end(month):
            raise VanDenHauweReducedError(
                f"forecast origin must equal end(t-1) in {month}"
            )
        decision_observed = raw.get("decision_month")
        if decision_observed not in {True, False}:
            raise VanDenHauweReducedError(
                f"decision_month flag missing in {month}"
            )
        raw_direction = raw.get("direction")
        target_direction = str(raw_direction) if raw_direction is not None else None
        if decision_observed and target_direction not in DIRECTIONS:
            raise VanDenHauweReducedError(f"invalid observed target direction in {month}")
        if not decision_observed and raw_direction is not None:
            raise VanDenHauweReducedError(
                f"meeting-free month must have a null target, not Hold: {month}"
            )
        target_ledger = ledger["DFEDTAR"][0]
        if raw.get("target_source_sha256") != target_ledger["sha256"]:
            raise VanDenHauweReducedError(f"unbound target source SHA256 in {month}")
        try:
            target_rate = float(raw.get("target_rate_end_month_pct"))
            target_previous = float(raw.get("previous_target_rate_pct"))
        except (TypeError, ValueError) as exc:
            raise VanDenHauweReducedError(f"missing target-rate state in {month}") from exc
        if not math.isfinite(target_rate) or not math.isfinite(target_previous):
            raise VanDenHauweReducedError(f"non-finite target-rate state in {month}")
        signed_change = target_rate - target_previous
        implied_direction = (
            "cut"
            if signed_change < -1e-10
            else "hike"
            if signed_change > 1e-10
            else "hold"
        )
        if decision_observed and implied_direction != target_direction:
            raise VanDenHauweReducedError(
                f"target-rate change disagrees with decision direction in {month}"
            )
        if not decision_observed and implied_direction != "hold":
            raise VanDenHauweReducedError(
                f"meeting-free month contains a target-rate change in {month}"
            )
        target_observation = _canonical_date(
            raw.get("target_rate_observation_date"),
            label=f"target observation {month}",
        )
        previous_target_observation = _canonical_date(
            raw.get("previous_target_rate_observation_date"),
            label=f"previous target observation {month}",
        )
        if target_observation[:7] != month or previous_target_observation > forecast_origin:
            raise VanDenHauweReducedError(f"target-rate observation timing drift in {month}")
        predictors = raw.get("predictors")
        if not isinstance(predictors, dict) or set(predictors) != set(predictor_specs):
            raise VanDenHauweReducedError(
                f"{month} must contain exactly 6TFF/IP/INF predictors"
            )
        normalized_predictors: dict[str, dict[str, Any]] = {}
        for paper_id, spec in predictor_specs.items():
            value = predictors[paper_id]
            if not isinstance(value, dict):
                raise VanDenHauweReducedError(f"non-object {paper_id} value in {month}")
            if value.get("current_vintage_fallback") not in {None, False}:
                raise VanDenHauweReducedError(
                    f"current-vintage fallback for {paper_id} is forbidden in {month}"
                )
            if value.get("transform") != spec["transform"] or value.get(
                "averaging_months"
            ) != spec["m"]:
                raise VanDenHauweReducedError(
                    f"paper transformation/m drift for {paper_id} in {month}"
                )
            try:
                numeric = float(value.get("value"))
            except (TypeError, ValueError) as exc:
                raise VanDenHauweReducedError(
                    f"missing/non-numeric {paper_id} value in {month}"
                ) from exc
            if not math.isfinite(numeric):
                raise VanDenHauweReducedError(f"non-finite {paper_id} value in {month}")
            observation_period = str(value.get("observation_period") or "")
            if _month_index(observation_period) > _month_index(month) - 1:
                raise VanDenHauweReducedError(
                    f"future/insufficiently lagged {paper_id} observation in {month}"
                )
            raw_observation_dates = value.get("observation_dates")
            if not isinstance(raw_observation_dates, list) or not raw_observation_dates:
                raise VanDenHauweReducedError(
                    f"observation-date ledger missing for {paper_id} in {month}"
                )
            observation_dates = [
                _canonical_date(item, label=f"{paper_id} observation date {month}")
                for item in raw_observation_dates
            ]
            if any(item > forecast_origin for item in observation_dates):
                raise VanDenHauweReducedError(
                    f"future {paper_id} observation in {month}"
                )
            availability_date = _canonical_date(
                value.get("availability_date"),
                label=f"{paper_id} availability date {month}",
            )
            source_policy = str(spec["source_policy"])
            if source_policy == "real_time_vintage":
                if (
                    value.get("source_policy")
                    != "alfred_end_of_prior_month_vintage"
                    or value.get("real_time_vintage") is not True
                ):
                    raise VanDenHauweReducedError(
                        f"non-vintage {paper_id} input is forbidden in {month}"
                    )
                vintage_date = _canonical_date(
                    value.get("vintage_date"), label=f"{paper_id} vintage date {month}"
                )
                series_id = "INDPRO" if paper_id == "IP" else "CPIAUCSL"
                expected_source_sha = _ledger_hash_for_vintage(
                    ledger, series_id, vintage_date
                )
                if value.get("source_sha256") != expected_source_sha:
                    raise VanDenHauweReducedError(
                        f"unbound ALFRED source SHA256 for {paper_id} in {month}"
                    )
                source_paths = [
                    str(record["path"])
                    for record in ledger[series_id]
                    if vintage_date in record.get("vintage_dates", [])
                ]
                source_policy_output = "real_time_vintage"
            elif source_policy == "historical_market_nonrevised":
                if (
                    paper_id != "6TFF"
                    or value.get("source_policy") != "historical_market_nonrevised"
                    or value.get("real_time_vintage") is not False
                ):
                    raise VanDenHauweReducedError(
                        f"6TFF must use certified non-revised market history in {month}"
                    )
                vintage_date = None
                expected_source_sha = sha256_bytes(
                    canonical_json(
                        {
                            "TB6MS": ledger["TB6MS"][0]["sha256"],
                            "FEDFUNDS": ledger["FEDFUNDS"][0]["sha256"],
                        }
                    ).encode("utf-8")
                )
                if value.get("source_sha256") != expected_source_sha:
                    raise VanDenHauweReducedError(
                        f"unbound market-history source SHA256 for 6TFF in {month}"
                    )
                source_paths = [
                    str(ledger["TB6MS"][0]["path"]),
                    str(ledger["FEDFUNDS"][0]["path"]),
                ]
                source_policy_output = "historical_market_nonrevised"
            else:  # pragma: no cover - frozen config validation makes this unreachable.
                raise VanDenHauweReducedError(
                    f"unsupported source policy for {paper_id}: {source_policy}"
                )
            if (vintage_date is not None and vintage_date != forecast_origin) or availability_date > forecast_origin:
                raise VanDenHauweReducedError(
                    f"future {paper_id} vintage/availability in {month}"
                )
            normalized_predictors[paper_id] = {
                "value": numeric,
                "transform": spec["transform"],
                "m": int(spec["m"]),
                "observation_period": observation_period,
                "observation_dates": observation_dates,
                "vintage_date": vintage_date,
                "availability_date": availability_date,
                "source_policy": source_policy_output,
                "real_time_vintage": source_policy == "real_time_vintage",
                "current_vintage_fallback": False,
                "source_paths": source_paths,
                "source_sha256": value["source_sha256"],
            }
        h19_target = h19.get(month)
        if h19_target:
            if not decision_observed or h19_target["target_direction"] != target_direction:
                raise VanDenHauweReducedError(
                    f"monthly target disagrees with H19 meeting label in {month}"
                )
        normalized = {
            "schema_version": PANEL_SCHEMA,
            "month": month,
            "forecast_origin_date": forecast_origin,
            "decision_observed": decision_observed,
            "target_direction": target_direction,
            "target_rate_end_month_pct": target_rate,
            "target_rate_observation_date": target_observation,
            "previous_target_rate_pct": target_previous,
            "previous_target_rate_observation_date": previous_target_observation,
            "target_source_path": target_ledger["path"],
            "target_source_sha256": target_ledger["sha256"],
            "predictors": normalized_predictors,
            "matched_h19": h19_target,
            "paper_recursive_oos": decision_observed and "2001-01" <= month <= "2008-06",
            "source_row_sha256": raw["row_sha256"],
        }
        normalized["row_sha256"] = sha256_bytes(
            canonical_json(normalized).encode("utf-8")
        )
        panel.append(normalized)
    matched = [row for row in panel if row["matched_h19"] is not None]
    recursive = [row for row in panel if row["paper_recursive_oos"]]
    observed_decisions = [row for row in panel if row["decision_observed"]]
    if len(observed_decisions) != 157:
        raise VanDenHauweReducedError(
            f"observed decision-month count drift: {len(observed_decisions)} != 157"
        )
    if Counter(row["target_direction"] for row in observed_decisions) != Counter(
        {"cut": 40, "hold": 86, "hike": 31}
    ):
        raise VanDenHauweReducedError("observed decision-direction count drift")
    if len(matched) != 19 or len(recursive) != 62:
        raise VanDenHauweReducedError(
            f"evaluation population drift: H19={len(matched)}, recursive={len(recursive)}"
        )
    metadata = {
        "source_manifest": source_manifest,
        "source_manifest_path": str((source_root / config["source_release"]["manifest"]).resolve()),
        "source_manifest_sha256": sha256_file(
            (source_root / config["source_release"]["manifest"]).resolve()
        ),
        "monthly_source_path": str(monthly_path),
        "monthly_source_sha256": sha256_file(monthly_path),
        "source_files": list(source_manifest["files"]),
        "h19_sha256": matched[0]["matched_h19"]["h19_source_sha256"],
    }
    return panel, metadata


def _manifest_payload(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    source_root: Path,
    paths: RuntimePaths,
    panel: Sequence[Mapping[str, Any]],
    source_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "model_id": MODEL_ID,
        "replication_claim": config["replication_claim"],
        "created_at_utc": utc_now(),
        "create_only": True,
        "resume_requires_exact_hash_match": True,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "implementation_path": str(IMPLEMENTATION),
        "implementation_sha256": sha256_file(IMPLEMENTATION),
        "source_root": str(source_root),
        "source_manifest_path": source_metadata["source_manifest_path"],
        "source_manifest_sha256": source_metadata["source_manifest_sha256"],
        "monthly_source_path": source_metadata["monthly_source_path"],
        "monthly_source_sha256": source_metadata["monthly_source_sha256"],
        "source_files": source_metadata["source_files"],
        "historical_h19_path": str(resolve_path(config["historical_h19"])),
        "historical_h19_sha256": source_metadata["h19_sha256"],
        "monthly_panel_path": str(paths.monthly_panel),
        "monthly_panel_sha256": sha256_file(paths.monthly_panel),
        "monthly_panel_rows": len(panel),
        "sample": dict(config["sample"]),
        "predictors": [dict(row) for row in config["predictors"]],
        "contracts": dict(config["contracts"]),
        "recursive_importance_coverage_gate": {
            "statistic": "pre_update_effective_particle_count_at_forecast_origin",
            "minimum": MIN_RECURSIVE_PRE_UPDATE_ESS,
            "failure_policy": "prediction row retained with probabilities null and coverage_status unavailable",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
        },
    }


def run_prepare(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    source_root: Path,
    paths: RuntimePaths,
    resume: bool,
) -> dict[str, Any]:
    panel, source_metadata = prepare_monthly_panel(
        config, source_root, config_path=config_path
    )
    write_jsonl(paths.monthly_panel, panel, resume=resume)
    expected = _manifest_payload(
        config_path=config_path,
        config=config,
        source_root=source_root,
        paths=paths,
        panel=panel,
        source_metadata=source_metadata,
    )
    if resume and paths.manifest.exists():
        # Creation time is intentionally stable across a byte-identical resume.
        observed = read_json(paths.manifest)
        expected["created_at_utc"] = observed.get("created_at_utc")
    write_json(paths.manifest, expected, resume=resume)
    return expected


def validate_prepared_run(
    *, config_path: Path, config: Mapping[str, Any], source_root: Path, paths: RuntimePaths
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not paths.monthly_panel.is_file() or not paths.manifest.is_file():
        raise VanDenHauweReducedError("prepare phase has not completed")
    observed_manifest = read_json(paths.manifest)
    if observed_manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise VanDenHauweReducedError("prepared manifest schema drift")
    if observed_manifest.get("config_sha256") != sha256_file(config_path):
        raise VanDenHauweReducedError("prepared config SHA256 drift")
    if observed_manifest.get("implementation_sha256") != sha256_file(IMPLEMENTATION):
        raise VanDenHauweReducedError("prepared implementation SHA256 drift")
    if observed_manifest.get("source_root") != str(source_root):
        raise VanDenHauweReducedError("prepared source root drift")
    if observed_manifest.get("monthly_panel_sha256") != sha256_file(paths.monthly_panel):
        raise VanDenHauweReducedError("prepared monthly panel SHA256 drift")
    source_manifest_path = Path(str(observed_manifest.get("source_manifest_path") or ""))
    monthly_source_path = Path(str(observed_manifest.get("monthly_source_path") or ""))
    for artifact, key in (
        (source_manifest_path, "source_manifest_sha256"),
        (monthly_source_path, "monthly_source_sha256"),
        (resolve_path(config["historical_h19"]), "historical_h19_sha256"),
    ):
        if not artifact.is_file() or sha256_file(artifact) != observed_manifest.get(key):
            raise VanDenHauweReducedError(f"prepared input SHA256 drift: {artifact}")
    panel, metadata = prepare_monthly_panel(
        config, source_root, config_path=config_path
    )
    if canonical_json(panel) != canonical_json(read_jsonl(paths.monthly_panel)):
        raise VanDenHauweReducedError("prepared panel differs from current source release")
    if metadata["source_manifest_sha256"] != observed_manifest["source_manifest_sha256"]:
        raise VanDenHauweReducedError("prepared source manifest drift")
    return panel, observed_manifest


def _selected_contracts(value: str) -> tuple[str, ...]:
    return CONTRACTS if value == "both" else (value,)


def _evaluation_rows(
    panel: Sequence[Mapping[str, Any]], contract: str
) -> list[Mapping[str, Any]]:
    if contract == "matched_h19_posterior_imputation":
        return [row for row in panel if row.get("matched_h19") is not None]
    if contract == "paper_recursive_oos":
        return [row for row in panel if row.get("paper_recursive_oos") is True]
    raise VanDenHauweReducedError(f"unknown contract: {contract}")


def _load_model_module(config: Mapping[str, Any]) -> tuple[Any, Path]:
    module_name = str(config["model"]["implementation_module"])
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise VanDenHauweReducedError(
            f"cannot import reduced VDH estimator module {module_name}: {exc}"
        ) from exc
    required = (
        "MCMCConfig",
        "ImportanceSamplingConfig",
        "fit_van_den_hauwe_2013",
        "recursive_importance_forecast",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise VanDenHauweReducedError(f"estimator API is incomplete: {missing}")
    module_file = Path(str(getattr(module, "__file__", ""))).resolve()
    if not module_file.is_file():
        raise VanDenHauweReducedError("estimator module has no hashable source file")
    return module, module_file


def _panel_arrays(
    panel: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(
        [
            [float(row["predictors"][paper_id]["value"]) for paper_id in ("6TFF", "IP", "INF")]
            for row in panel
        ],
        dtype=float,
    )
    class_index = {direction: index for index, direction in enumerate(DIRECTIONS)}
    y = np.asarray(
        [
            float(class_index[str(row["target_direction"])])
            if row["decision_observed"]
            else np.nan
            for row in panel
        ],
        dtype=float,
    )
    target_previous = np.asarray(
        [float(row["previous_target_rate_pct"]) for row in panel], dtype=float
    )
    if x.shape != (222, 3) or y.shape != (222,) or target_previous.shape != (222,):
        raise VanDenHauweReducedError("prepared model array shape drift")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(target_previous)):
        raise VanDenHauweReducedError("prepared model arrays contain non-finite values")
    return x, y, target_previous


def _fit_contract_predictions(
    module: Any,
    *,
    panel: Sequence[Mapping[str, Any]],
    contract: str,
    config: Mapping[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    x, y, target_previous = _panel_arrays(panel)
    mcmc = config["model"]["mcmc"]
    sampler_config = module.MCMCConfig(
        draws=int(mcmc["draws"]),
        burn_in=int(mcmc["burn_in"]),
        thin=int(mcmc["thin"]),
        seed=int(seed),
        phi_proposal_sd=float(mcmc["phi_proposal_sd"]),
        standardize_predictors=bool(mcmc["standardize_predictors"]),
    )
    if contract == "matched_h19_posterior_imputation":
        evaluation_indices = [
            index for index, row in enumerate(panel) if row["matched_h19"] is not None
        ]
        training_y = y.copy()
        training_y[evaluation_indices] = np.nan
        fitted = module.fit_van_den_hauwe_2013(
            x,
            training_y,
            target_previous,
            config=sampler_config,
        )
        probabilities = fitted.smoothed_class_probabilities()
        sampler_diagnostics = fitted.diagnostics.to_dict()
        posterior_summary = fitted.posterior_summary()
        training_count = int(np.sum(~np.isnan(training_y)))
        return [
            {
                "month": panel[index]["month"],
                "coverage_status": "available",
                "unavailable_reason": None,
                "probabilities": {
                    direction: float(probabilities[index, direction_index])
                    for direction_index, direction in enumerate(DIRECTIONS)
                },
                "training_count": training_count,
                "posterior_diagnostics": {
                    "prediction_method": "posterior_smoothed_missing-H19_direction",
                    "realized_target_rate_state_retained": True,
                    "retrospective_not_chronological_forecast": True,
                    "sampler": sampler_diagnostics,
                    "posterior": posterior_summary,
                },
            }
            for index in evaluation_indices
        ]
    if contract == "paper_recursive_oos":
        split = next(index for index, row in enumerate(panel) if row["month"] == "2001-01")
        fitted = module.fit_van_den_hauwe_2013(
            x[:split],
            y[:split],
            target_previous[:split],
            config=sampler_config,
        )
        recursive = module.recursive_importance_forecast(
            fitted,
            x[split:],
            y[split:],
            target_previous[split:],
            config=module.ImportanceSamplingConfig(seed=int(seed)),
        )
        sampler_diagnostics = fitted.diagnostics.to_dict()
        posterior_summary = fitted.posterior_summary()
        output: list[dict[str, Any]] = []
        observed_training = int(np.sum(~np.isnan(y[:split])))
        for future_index, panel_index in enumerate(range(split, len(panel))):
            if not panel[panel_index]["paper_recursive_oos"]:
                continue
            pre_update_ess = float(recursive.pre_update_ess[future_index])
            is_available = pre_update_ess >= MIN_RECURSIVE_PRE_UPDATE_ESS
            output.append(
                {
                    "month": panel[panel_index]["month"],
                    "coverage_status": "available" if is_available else "unavailable",
                    "unavailable_reason": (
                        None if is_available else RECURSIVE_ESS_UNAVAILABLE_REASON
                    ),
                    "probabilities": (
                        {
                            direction: float(
                                recursive.probabilities[future_index, direction_index]
                            )
                            for direction_index, direction in enumerate(DIRECTIONS)
                        }
                        if is_available
                        else None
                    ),
                    "training_count": observed_training,
                    "posterior_diagnostics": {
                        "prediction_method": "recursive_importance_sampling",
                        "sampler": sampler_diagnostics,
                        "posterior": posterior_summary,
                        "pre_update_ess": pre_update_ess,
                        "minimum_required_pre_update_ess": MIN_RECURSIVE_PRE_UPDATE_ESS,
                        "ess_coverage_gate_passed": is_available,
                        "post_update_ess": float(recursive.post_update_ess[future_index]),
                        "pre_update_ess_ratio": float(
                            recursive.pre_update_ess_ratio[future_index]
                        ),
                        "post_update_ess_ratio": float(
                            recursive.post_update_ess_ratio[future_index]
                        ),
                    },
                }
            )
            observed_training += 1
        return output
    raise VanDenHauweReducedError(f"unknown contract: {contract}")


def _normalize_model_predictions(
    raw_predictions: Any,
    *,
    panel: Sequence[Mapping[str, Any]],
    contract: str,
    seed: int,
    run_binding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(raw_predictions, list):
        raise VanDenHauweReducedError(
            f"estimator must return a list for contract {contract}"
        )
    targets = {str(row["month"]): row for row in _evaluation_rows(panel, contract)}
    if len(raw_predictions) != len(targets):
        raise VanDenHauweReducedError(
            f"prediction count drift for {contract}: {len(raw_predictions)} != {len(targets)}"
        )
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    tolerance = 1e-8
    for index, raw in enumerate(raw_predictions):
        if not isinstance(raw, dict):
            raise VanDenHauweReducedError(f"non-object prediction {contract}:{index}")
        month = str(raw.get("month") or "")
        if month in seen or month not in targets:
            raise VanDenHauweReducedError(
                f"duplicate/unexpected prediction month {contract}:{month}"
            )
        seen.add(month)
        target = targets[month]
        status = str(raw.get("coverage_status") or "")
        if status not in {"available", "unavailable"}:
            raise VanDenHauweReducedError(
                f"invalid coverage status {contract}:{month}: {status!r}"
            )
        probabilities: dict[str, float] | None = None
        unavailable_reason: str | None = None
        if status == "available":
            raw_probabilities = raw.get("probabilities")
            if not isinstance(raw_probabilities, dict) or set(raw_probabilities) != set(
                DIRECTIONS
            ):
                raise VanDenHauweReducedError(
                    f"available prediction lacks three probabilities: {contract}:{month}"
                )
            probabilities = {}
            for direction in DIRECTIONS:
                try:
                    probability = float(raw_probabilities[direction])
                except (TypeError, ValueError) as exc:
                    raise VanDenHauweReducedError(
                        f"non-numeric probability {contract}:{month}:{direction}"
                    ) from exc
                if not math.isfinite(probability) or probability < -tolerance:
                    raise VanDenHauweReducedError(
                        f"invalid probability {contract}:{month}:{direction}"
                    )
                probabilities[direction] = max(0.0, probability)
            total = sum(probabilities.values())
            if not math.isclose(total, 1.0, abs_tol=tolerance):
                raise VanDenHauweReducedError(
                    f"probabilities do not sum to one {contract}:{month}: {total}"
                )
            probabilities = {key: value / total for key, value in probabilities.items()}
            if raw.get("unavailable_reason") not in {None, ""}:
                raise VanDenHauweReducedError(
                    f"available prediction has unavailable reason: {contract}:{month}"
                )
        else:
            unavailable_reason = str(raw.get("unavailable_reason") or "")
            if not unavailable_reason:
                raise VanDenHauweReducedError(
                    f"unavailable prediction lacks reason: {contract}:{month}"
                )
            if raw.get("probabilities") is not None:
                raise VanDenHauweReducedError(
                    f"unavailable prediction carries probabilities: {contract}:{month}"
                )
        h19 = target.get("matched_h19")
        row = {
            "schema_version": PREDICTION_SCHEMA,
            "model": MODEL_ID,
            "contract": contract,
            "month": month,
            "forecast_origin_date": target["forecast_origin_date"],
            "meeting_id": h19["meeting_id"] if h19 else None,
            "meeting_start_date": h19["meeting_start_date"] if h19 else None,
            "evidence_cutoff": h19["evidence_cutoff"] if h19 else None,
            "target_direction": target["target_direction"],
            "coverage_status": status,
            "unavailable_reason": unavailable_reason,
            "probabilities": probabilities,
            "training_count": raw.get("training_count"),
            "posterior_diagnostics": raw.get("posterior_diagnostics"),
            "seed": int(seed),
            "monthly_panel_row_sha256": target["row_sha256"],
            "prediction_run_binding": dict(run_binding),
        }
        row["row_sha256"] = sha256_bytes(canonical_json(row).encode("utf-8"))
        normalized.append(row)
    normalized.sort(key=lambda row: (CONTRACTS.index(row["contract"]), row["month"]))
    return normalized


def run_predict(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    source_root: Path,
    paths: RuntimePaths,
    contracts: Sequence[str],
    seed: int,
    resume: bool,
) -> list[dict[str, Any]]:
    panel, manifest = validate_prepared_run(
        config_path=config_path, config=config, source_root=source_root, paths=paths
    )
    module, module_file = _load_model_module(config)
    run_binding = {
        "manifest_sha256": sha256_file(paths.manifest),
        "monthly_panel_sha256": sha256_file(paths.monthly_panel),
        "model_module_path": str(module_file),
        "model_module_sha256": sha256_file(module_file),
        "contracts": list(contracts),
        "seed": int(seed),
    }
    if resume and paths.predictions.exists():
        rows = read_jsonl(paths.predictions)
        observed_bindings = {canonical_json(row.get("prediction_run_binding")) for row in rows}
        if observed_bindings != {canonical_json(run_binding)}:
            raise VanDenHauweReducedError("resume prediction binding drift")
        # Re-use only a fully validated artifact; never re-run MCMC on resume.
        by_contract = Counter(str(row.get("contract")) for row in rows)
        expected = {
            contract: len(_evaluation_rows(panel, contract)) for contract in contracts
        }
        if by_contract != Counter(expected):
            raise VanDenHauweReducedError("resume prediction population drift")
        return rows
    all_predictions: list[dict[str, Any]] = []
    for contract in contracts:
        try:
            raw = _fit_contract_predictions(
                module,
                panel=panel,
                contract=contract,
                config=config,
                seed=int(seed),
            )
        except Exception as exc:
            raise VanDenHauweReducedError(
                f"reduced VDH estimation failed for {contract}: {exc}"
            ) from exc
        all_predictions.extend(
            _normalize_model_predictions(
                raw,
                panel=panel,
                contract=contract,
                seed=seed,
                run_binding=run_binding,
            )
        )
    write_jsonl(paths.predictions, all_predictions, resume=False)
    return all_predictions


def _argmax_direction(probabilities: Mapping[str, float]) -> str | None:
    maximum = max(float(probabilities[value]) for value in DIRECTIONS)
    winners = [
        value
        for value in DIRECTIONS
        if math.isclose(float(probabilities[value]), maximum, abs_tol=1e-15)
    ]
    return winners[0] if len(winners) == 1 else None


def _score_contract(rows: Sequence[Mapping[str, Any]], *, epsilon: float) -> dict[str, Any]:
    available = [row for row in rows if row.get("coverage_status") == "available"]
    unavailable = [row for row in rows if row.get("coverage_status") == "unavailable"]
    by_direction: dict[str, Any] = {}
    recalls: list[float] = []
    for direction in DIRECTIONS:
        subset = [row for row in available if row["target_direction"] == direction]
        correct = sum(
            _argmax_direction(row["probabilities"]) == direction for row in subset
        )
        recalls.append(correct / len(subset)) if subset else None
        by_direction[direction] = {
            "available_n": len(subset),
            "mean_true_direction_probability": (
                sum(float(row["probabilities"][direction]) for row in subset)
                / len(subset)
                if subset
                else None
            ),
            "argmax_recall": correct / len(subset) if subset else None,
        }
    if not available:
        metrics = {
            "mean_true_direction_probability": None,
            "argmax_accuracy": None,
            "supported_class_balanced_accuracy": None,
            "multiclass_brier": None,
            "clipped_log_loss": None,
        }
    else:
        true_probabilities = [
            float(row["probabilities"][row["target_direction"]]) for row in available
        ]
        accuracy = sum(
            _argmax_direction(row["probabilities"]) == row["target_direction"]
            for row in available
        ) / len(available)
        brier = sum(
            sum(
                (
                    float(row["probabilities"][direction])
                    - (1.0 if direction == row["target_direction"] else 0.0)
                )
                ** 2
                for direction in DIRECTIONS
            )
            for row in available
        ) / len(available)
        log_loss = -sum(math.log(max(epsilon, value)) for value in true_probabilities) / len(
            available
        )
        metrics = {
            "mean_true_direction_probability": sum(true_probabilities) / len(available),
            "argmax_accuracy": accuracy,
            "supported_class_balanced_accuracy": sum(recalls) / len(recalls)
            if recalls
            else None,
            "multiclass_brier": brier,
            "clipped_log_loss": log_loss,
        }
    return {
        "total_n": len(rows),
        "available_n": len(available),
        "unavailable_n": len(unavailable),
        "coverage_rate": len(available) / len(rows) if rows else None,
        "target_support": dict(Counter(str(row["target_direction"]) for row in rows)),
        "unavailable_reasons": dict(
            Counter(str(row["unavailable_reason"]) for row in unavailable)
        ),
        "metrics": metrics,
        "by_direction": by_direction,
    }


def _score_paper_recursive_oos(
    rows: Sequence[Mapping[str, Any]], *, epsilon: float
) -> dict[str, Any]:
    """Score the full paper OOS and its H19 overlap without pooling either set."""

    if len(rows) != 62:
        raise VanDenHauweReducedError(
            f"paper-recursive full OOS population drift: {len(rows)} != 62"
        )
    overlap = [row for row in rows if row.get("meeting_id") is not None]
    overlap_support = Counter(str(row.get("target_direction")) for row in overlap)
    expected_support = Counter({"cut": 3, "hold": 2, "hike": 3})
    if len(overlap) != 8 or overlap_support != expected_support:
        raise VanDenHauweReducedError(
            "paper-recursive H19-overlap population drift: "
            f"N={len(overlap)}, support={dict(overlap_support)}"
        )

    full_result = _score_contract(rows, epsilon=epsilon)
    full_result["population_id"] = "paper_recursive_oos_full_n62"
    overlap_result = _score_contract(overlap, epsilon=epsilon)
    overlap_result.update(
        {
            "population_id": "paper_recursive_oos_h19_overlap_n8",
            "is_subset_of": "paper_recursive_oos_full_n62",
            "pooled_with_full_oos": False,
        }
    )
    full_result["h19_overlap"] = overlap_result
    return full_result


def validate_prediction_artifact(
    rows: Sequence[Mapping[str, Any]],
    *,
    panel: Sequence[Mapping[str, Any]],
    contracts: Sequence[str],
    paths: RuntimePaths,
) -> None:
    expected_binding_manifest = sha256_file(paths.manifest)
    expected: dict[tuple[str, str], Mapping[str, Any]] = {}
    for contract in contracts:
        for target in _evaluation_rows(panel, contract):
            expected[(contract, str(target["month"]))] = target
    if len(rows) != len(expected):
        raise VanDenHauweReducedError("prediction artifact population count drift")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("contract")), str(row.get("month")))
        target = expected.get(key)
        if (
            target is None
            or key in seen
            or row.get("schema_version") != PREDICTION_SCHEMA
            or row.get("model") != MODEL_ID
            or row.get("target_direction") != target["target_direction"]
            or row.get("monthly_panel_row_sha256") != target["row_sha256"]
            or row.get("prediction_run_binding", {}).get("manifest_sha256")
            != expected_binding_manifest
        ):
            raise VanDenHauweReducedError(f"prediction artifact binding drift: {key}")
        seen.add(key)
        expected_row_hash = sha256_bytes(
            canonical_json({k: v for k, v in row.items() if k != "row_sha256"}).encode(
                "utf-8"
            )
        )
        if row.get("row_sha256") != expected_row_hash:
            raise VanDenHauweReducedError(f"prediction row SHA256 drift: {key}")
        status = row.get("coverage_status")
        if status == "available":
            probabilities = row.get("probabilities")
            if not isinstance(probabilities, dict) or set(probabilities) != set(DIRECTIONS):
                raise VanDenHauweReducedError(f"prediction probability schema drift: {key}")
            if any(
                not math.isfinite(float(value)) or float(value) < 0
                for value in probabilities.values()
            ) or not math.isclose(
                sum(float(value) for value in probabilities.values()), 1.0, abs_tol=1e-8
            ):
                raise VanDenHauweReducedError(f"invalid prediction probabilities: {key}")
        elif status == "unavailable":
            if row.get("probabilities") is not None or not row.get("unavailable_reason"):
                raise VanDenHauweReducedError(f"invalid unavailable prediction: {key}")
        else:
            raise VanDenHauweReducedError(f"invalid prediction coverage status: {key}")


def run_score(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    source_root: Path,
    paths: RuntimePaths,
    contracts: Sequence[str],
    resume: bool,
) -> dict[str, Any]:
    panel, _ = validate_prepared_run(
        config_path=config_path, config=config, source_root=source_root, paths=paths
    )
    if not paths.predictions.is_file():
        raise VanDenHauweReducedError("predict phase has not completed")
    predictions = read_jsonl(paths.predictions)
    validate_prediction_artifact(
        predictions, panel=panel, contracts=contracts, paths=paths
    )
    epsilon = float(config["scoring"]["log_loss_epsilon"])
    contract_results: dict[str, Any] = {}
    for contract in contracts:
        rows = [row for row in predictions if row["contract"] == contract]
        if contract == "paper_recursive_oos":
            contract_results[contract] = _score_paper_recursive_oos(
                rows, epsilon=epsilon
            )
        else:
            result = _score_contract(rows, epsilon=epsilon)
            result["population_id"] = "matched_h19_posterior_imputation_n19"
            contract_results[contract] = result
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "model": MODEL_ID,
        "replication_claim": config["replication_claim"],
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(paths.manifest),
        "monthly_panel_sha256": sha256_file(paths.monthly_panel),
        "predictions_sha256": sha256_file(paths.predictions),
        "contracts": contract_results,
        "pooled_result": None,
        "pooled_result_reason": (
            "retrospective H19 posterior imputation, the full paper-recursive monthly OOS, and its explicitly "
            "nested H19 overlap are distinct estimands and are never pooled"
        ),
    }
    write_json(paths.summary, summary, resume=resume)
    return summary


def status_payload(paths: RuntimePaths) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for name, path in (
        ("manifest", paths.manifest),
        ("monthly_panel", paths.monthly_panel),
        ("predictions", paths.predictions),
        ("summary", paths.summary),
        ("status", paths.status),
    ):
        artifacts[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None,
        }
    return {"schema_version": STATUS_SCHEMA, "output_root": str(paths.root), "artifacts": artifacts}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("prepare", "predict", "score", "all", "status"),
        default="all",
    )
    parser.add_argument(
        "--contract",
        choices=(*CONTRACTS, "both"),
        default="both",
        help="evaluation contract(s); contracts are never pooled",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    paths = runtime_paths(config, args.output_root)
    source_root = resolve_path(
        args.source_root or str(config["source_release"]["default_root"])
    )
    seed = int(args.seed if args.seed is not None else config["model"]["seed"])
    contracts = _selected_contracts(args.contract)
    if args.phase == "status":
        print(json.dumps(status_payload(paths), indent=2, sort_keys=True))
        return 0
    paths.root.mkdir(parents=True, exist_ok=True)
    write_status(paths.status, phase=args.phase, state="running")
    try:
        if args.phase in {"prepare", "all"}:
            run_prepare(
                config_path=config_path,
                config=config,
                source_root=source_root,
                paths=paths,
                resume=args.resume,
            )
        if args.phase in {"predict", "all"}:
            run_predict(
                config_path=config_path,
                config=config,
                source_root=source_root,
                paths=paths,
                contracts=contracts,
                seed=seed,
                resume=args.resume,
            )
        if args.phase in {"score", "all"}:
            run_score(
                config_path=config_path,
                config=config,
                source_root=source_root,
                paths=paths,
                contracts=contracts,
                resume=args.resume,
            )
    except Exception as exc:
        write_status(
            paths.status,
            phase=args.phase,
            state="failed",
            detail={"error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
    write_status(
        paths.status,
        phase=args.phase,
        state="complete",
        detail={"contracts": list(contracts), "seed": seed},
    )
    print(json.dumps(status_payload(paths), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
