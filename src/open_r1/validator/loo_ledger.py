"""Build and validate canonical D-1 indicator-input ledgers.

The canonical information set is the latest historical vintage available on
the calendar day immediately before an FOMC decision date.  Observation-date
filtering alone is not sufficient: every raw snapshot must also be explicitly
requested with ``vintage_dates == meeting_date - 1 day``.

The module consumes:

* ``loo-indicator-source-registry-v1`` source registries;
* sealed ``loo-source-snapshot-manifest-v1`` manifests; and
* the exact raw CSV bytes bound by that snapshot manifest.

It produces one ``canonical-loo-indicator-input-v2`` row per
meeting-indicator pair plus independently re-playable evidence, exclusion, and
coverage artifacts.  Validation reconstructs all outputs from the raw bytes;
it does not trust hashes merely because they have the right shape.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any
from zoneinfo import ZoneInfo

from open_r1.provenance import sha256_file, sha256_text, validate_sha256
from open_r1.validator.loo_generation_spec import (
    ManifestIntegrityError,
    seal_manifest,
    validate_manifest_integrity,
)


REGISTRY_SCHEMA_VERSION = "loo-indicator-source-registry-v1"
SNAPSHOT_MANIFEST_SCHEMA_VERSION = "loo-source-snapshot-manifest-v1"
LEDGER_ROW_SCHEMA_VERSION = "canonical-loo-indicator-input-v2"
LEDGER_MANIFEST_SCHEMA_VERSION = "loo-indicator-ledger-manifest-v1"
SOURCE_EVIDENCE_SCHEMA_VERSION = "loo-indicator-source-evidence-v1"
COVERAGE_SCHEMA_VERSION = "loo-indicator-ledger-coverage-v1"
EXCLUSION_SCHEMA_VERSION = "loo-indicator-ledger-exclusion-v1"

EXPECTED_MEETING_COUNT = 13
EXPECTED_INDICATOR_COUNT = 26
EXPECTED_LEDGER_ROW_COUNT = EXPECTED_MEETING_COUNT * EXPECTED_INDICATOR_COUNT

INFORMATION_CUTOFF_POLICY = (
    "latest historical vintage available on meeting_date minus one calendar day; "
    "observation_date must also be on or before that date"
)
AVAILABILITY_EVIDENCE_TYPE = "alfred_vintage_snapshot"
SOURCE_INTERFACE = "alfred-graph-csv-v1"
SAMPLING_POLICY_VERSION = "d1-frequency-aware-v1"

_FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "actual_minutes",
        "actual_output",
        "answer",
        "gold",
        "gold_text",
        "ground_truth",
        "minutes",
        "reference",
        "reference_text",
        "target",
        "target_text",
    }
)
_DATE_COLUMNS = ("observation_date", "date", "DATE")
_FREQUENCY_ALIASES = {
    "d": "daily",
    "daily": "daily",
    "business daily": "daily",
    "w": "weekly",
    "weekly": "weekly",
    "bw": "biweekly",
    "biweekly": "biweekly",
    "m": "monthly",
    "monthly": "monthly",
    "q": "quarterly",
    "quarterly": "quarterly",
    "sa": "semiannual",
    "semiannual": "semiannual",
    "a": "annual",
    "annual": "annual",
    "yearly": "annual",
}
_LOOKBACK_PATTERN = re.compile(
    r"^\s*(?P<value>[1-9][0-9]*)\s*[-_ ]*"
    r"(?P<unit>observations?|days?|weeks?|months?|quarters?|years?)\s*$",
    flags=re.IGNORECASE,
)


class LooLedgerError(ValueError):
    """Raised when a source ledger cannot be proven D-1 safe."""


class LooLedgerIntegrityError(LooLedgerError):
    """Raised when a bound input or output artifact changed."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise LooLedgerError(f"Value is not finite canonical JSON: {exc}") from exc


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LooLedgerError(f"Unable to read {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LooLedgerError(f"{label} must be a JSON object: {path}")
    return payload


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise LooLedgerError(
                        f"{label} row {line_number} must be a JSON object"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise LooLedgerError(f"Unable to read {label} {path}: {exc}") from exc
    return rows


def _parse_iso_date(value: Any, *, label: str) -> date:
    text = str(value or "").strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise LooLedgerError(f"{label} must be YYYY-MM-DD, got {text!r}") from exc
    if parsed.isoformat() != text:
        raise LooLedgerError(f"{label} must be canonical YYYY-MM-DD, got {text!r}")
    return parsed


def _parse_timestamp(value: Any, *, label: str) -> datetime:
    text = str(value or "").strip()
    candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise LooLedgerError(f"{label} must be ISO-8601, got {text!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LooLedgerError(f"{label} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def decision_identity_timestamp(meeting_date: date) -> str:
    """Return the DST-aware pre-statement identity timestamp for a meeting."""

    local = datetime.combine(
        meeting_date,
        time(13, 59, 59),
        tzinfo=ZoneInfo("America/New_York"),
    )
    return _format_timestamp(local)


def information_as_of_date(meeting_date: date) -> date:
    """Return the previous calendar day used for the canonical information set."""

    return meeting_date - timedelta(days=1)


def compute_request_id(request: Mapping[str, Any]) -> str:
    """Compute the frozen request identifier shared with the snapshot fetcher."""

    descriptor = {
        field: request.get(field)
        for field in (
            "source_key",
            "series_id",
            "vintage_dates",
            "cosd",
            "coed",
            "canonical_url",
        )
    }
    return sha256_text(_canonical_json(descriptor))


def _load_roster(path: Path) -> list[str]:
    payload = _load_json(path, label="indicator roster")
    values = payload.get("indicators")
    if not isinstance(values, list):
        raise LooLedgerError("Indicator roster requires an 'indicators' list")
    roster = [str(value or "").strip() for value in values]
    if (
        len(roster) != EXPECTED_INDICATOR_COUNT
        or any(not value for value in roster)
        or len(set(roster)) != len(roster)
    ):
        raise LooLedgerError(
            f"Canonical roster must contain exactly {EXPECTED_INDICATOR_COUNT} "
            "unique non-empty indicators"
        )
    return roster


def _load_population(path: Path) -> tuple[str, list[date]]:
    payload = _load_json(path, label="population")
    population_id = str(payload.get("population_id") or "").strip()
    raw_dates = payload.get("meeting_dates")
    if not population_id or not isinstance(raw_dates, list):
        raise LooLedgerError(
            "Population requires population_id and meeting_dates"
        )
    dates = [
        _parse_iso_date(value, label=f"population meeting_dates[{index}]")
        for index, value in enumerate(raw_dates)
    ]
    if (
        len(dates) != EXPECTED_MEETING_COUNT
        or dates != sorted(set(dates))
    ):
        raise LooLedgerError(
            f"Canonical population must contain exactly {EXPECTED_MEETING_COUNT} "
            "unique ascending meeting dates"
        )
    return population_id, dates


def _normalise_frequency(value: Any, *, source_key: str) -> str:
    text = str(value or "").strip().lower()
    normalised = _FREQUENCY_ALIASES.get(text)
    if normalised is None:
        raise LooLedgerError(
            f"{source_key}: unsupported frequency {value!r}; use a canonical "
            "daily/weekly/monthly/quarterly/annual label"
        )
    return normalised


def _normalise_lookback(value: Any, *, source_key: str) -> dict[str, Any]:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return {"mode": "observations", "value": value}

    if isinstance(value, Mapping):
        if "observations" in value or "count" in value:
            raw = value.get("observations", value.get("count"))
            if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
                return {"mode": "observations", "value": raw}
        for unit in ("days", "weeks", "months", "quarters", "years"):
            raw = value.get(unit)
            if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
                return {"mode": "calendar", "unit": unit, "value": raw}
        raw_value = value.get("value")
        raw_unit = str(value.get("unit") or "").strip()
        if isinstance(raw_value, int) and not isinstance(raw_value, bool):
            value = f"{raw_value} {raw_unit}"

    if isinstance(value, str):
        compact_aliases = {
            "2y": "2 years",
            "24m": "24 months",
            "104w": "104 weeks",
            "730d": "730 days",
        }
        text = compact_aliases.get(value.strip().lower(), value)
        match = _LOOKBACK_PATTERN.match(text)
        if match:
            amount = int(match.group("value"))
            unit = match.group("unit").lower().rstrip("s")
            if unit == "observation":
                return {"mode": "observations", "value": amount}
            return {
                "mode": "calendar",
                "unit": f"{unit}s",
                "value": amount,
            }

    raise LooLedgerError(
        f"{source_key}: lookback must be a positive count or calendar duration"
    )


def _reject_forbidden_keys(value: Any, *, prefix: str = "source_payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).strip().lower()
            path = f"{prefix}.{key}"
            if key_text in _FORBIDDEN_PAYLOAD_KEYS:
                raise LooLedgerError(
                    f"Canonical source payload contains prohibited field {path}"
                )
            _reject_forbidden_keys(child, prefix=path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, prefix=f"{prefix}[{index}]")


def _load_registry(
    path: Path,
    *,
    roster: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    payload = _load_json(path, label="source registry")
    if payload.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise LooLedgerError(
            f"Source registry schema must be {REGISTRY_SCHEMA_VERSION!r}"
        )
    policy = payload.get("policy")
    series_defaults = payload.get("series_defaults", {})
    sources = payload.get("sources")
    indicators = payload.get("indicators")
    if not isinstance(policy, Mapping):
        raise LooLedgerError("Source registry requires an object-valued policy")
    required_policy = {
        "vintage_lag_calendar_days": 1,
        "same_meeting_day_data": "excluded",
        "unknown_availability": "fail_closed",
        "runtime_fallback": "forbidden",
        "local_legacy_values": "mapping_only",
        "synthetic_text_values": "mapping_only",
        "access_interface": SOURCE_INTERFACE,
        "raw_response_required": True,
        "raw_response_sha256_required": True,
    }
    mismatched_policy = {
        key: {"expected": expected, "observed": policy.get(key)}
        for key, expected in required_policy.items()
        if policy.get(key) != expected
    }
    if mismatched_policy:
        raise LooLedgerError(
            "Source registry policy is not canonical D-1: "
            f"{mismatched_policy}"
        )
    if not isinstance(series_defaults, Mapping):
        raise LooLedgerError("Source registry series_defaults must be an object")
    if not isinstance(sources, Mapping) or not sources:
        raise LooLedgerError("Source registry requires non-empty sources")
    if not isinstance(indicators, Mapping):
        raise LooLedgerError("Source registry requires keyed indicators")
    if set(indicators) != set(roster):
        raise LooLedgerError(
            "Source registry indicators must exactly match the frozen roster: "
            f"missing={sorted(set(roster) - set(indicators))}, "
            f"extra={sorted(set(indicators) - set(roster))}"
        )

    normalised_sources: dict[str, dict[str, Any]] = {}
    required_fields = (
        "series_id",
        "provider",
        "access_interface",
        "frequency",
        "units",
        "seasonal_adjustment",
        "lookback",
        "transformation",
        "redistribution_allowed",
        "enabled",
    )
    for raw_source_key, raw_source in sources.items():
        source_key = str(raw_source_key or "").strip()
        if not source_key or not isinstance(raw_source, Mapping):
            raise LooLedgerError("Every source registry entry must be a keyed object")
        merged_source = {
            **deepcopy(dict(series_defaults)),
            **deepcopy(dict(raw_source)),
        }
        if "license" not in merged_source and "license_class" in merged_source:
            merged_source["license"] = merged_source["license_class"]
        missing = [field for field in required_fields if field not in merged_source]
        if "license" not in merged_source:
            missing.append("license")
        if missing:
            raise LooLedgerError(f"{source_key}: missing registry fields {missing}")
        enabled = merged_source["enabled"]
        redistribution_allowed = merged_source["redistribution_allowed"]
        if not isinstance(enabled, bool) or not isinstance(
            redistribution_allowed, bool
        ):
            raise LooLedgerError(
                f"{source_key}: enabled and redistribution_allowed must be booleans"
            )
        transformation = deepcopy(merged_source["transformation"])
        if isinstance(transformation, str):
            transformation = transformation.strip()
        elif isinstance(transformation, Mapping):
            transformation = deepcopy(dict(transformation))
        if not transformation:
            raise LooLedgerError(
                f"{source_key}: transformation must be a non-empty string or object"
            )
        _reject_forbidden_keys(
            transformation,
            prefix=f"sources.{source_key}.transformation",
        )
        selection = deepcopy(
            merged_source.get(
                "selection",
                {
                    "method": "period_end_plus_latest",
                    "period": "month",
                    "max_points": 25,
                },
            )
        )
        if not isinstance(selection, Mapping):
            raise LooLedgerError(f"{source_key}: selection must be an object")
        selection = deepcopy(dict(selection))
        if selection.get("method") != "period_end_plus_latest":
            raise LooLedgerError(
                f"{source_key}: selection.method must be "
                "'period_end_plus_latest'"
            )
        if selection.get("period") != "month":
            raise LooLedgerError(
                f"{source_key}: selection.period must be 'month'"
            )
        max_points = selection.get("max_points")
        if (
            not isinstance(max_points, int)
            or isinstance(max_points, bool)
            or max_points <= 0
        ):
            raise LooLedgerError(
                f"{source_key}: selection.max_points must be positive"
            )
        source = {
            **merged_source,
            "source_key": source_key,
            "series_id": str(merged_source["series_id"] or "").strip(),
            "provider": str(merged_source["provider"] or "").strip(),
            "access_interface": str(
                merged_source["access_interface"] or ""
            ).strip(),
            "frequency": _normalise_frequency(
                merged_source["frequency"], source_key=source_key
            ),
            "units": str(merged_source["units"] or "").strip(),
            "seasonal_adjustment": str(
                merged_source["seasonal_adjustment"] or ""
            ).strip(),
            "transformation": transformation,
            "selection": selection,
            "license": str(merged_source["license"] or "").strip(),
            "lookback_policy": _normalise_lookback(
                merged_source["lookback"], source_key=source_key
            ),
        }
        empty = [
            field
            for field in (
                "series_id",
                "provider",
                "access_interface",
                "units",
                "seasonal_adjustment",
                "license",
            )
            if not source[field]
        ]
        if empty:
            raise LooLedgerError(f"{source_key}: empty registry fields {empty}")
        if source["access_interface"] != SOURCE_INTERFACE:
            raise LooLedgerError(
                f"{source_key}: canonical access_interface must be "
                f"{SOURCE_INTERFACE!r}"
            )
        if source["provider"].lower() not in {
            "alfred",
            "alfred_graph_csv",
        }:
            raise LooLedgerError(
                f"{source_key}: canonical provider must be ALFRED"
            )
        if source.get("date_column") != "observation_date":
            raise LooLedgerError(
                f"{source_key}: date_column must be 'observation_date'"
            )
        if source.get("value_column_template") != (
            "{series_id}_{vintage_date_yyyymmdd}"
        ):
            raise LooLedgerError(
                f"{source_key}: value_column_template is not canonical"
            )
        transformation_method = (
            source["transformation"].get("method")
            if isinstance(source["transformation"], Mapping)
            else source["transformation"]
        )
        if str(transformation_method).strip().lower() not in {
            "identity",
            "level",
            "none",
            "raw",
        }:
            raise LooLedgerError(
                f"{source_key}: v1 only supports identity transformations"
            )
        normalised_sources[source_key] = source

    normalised_indicators: dict[str, list[str]] = {}
    for indicator in roster:
        raw_keys = indicators[indicator]
        if isinstance(raw_keys, Mapping):
            raw_keys = raw_keys.get("source_keys")
        if not isinstance(raw_keys, list) or not raw_keys:
            raise LooLedgerError(
                f"{indicator}: indicator registry entry must be a non-empty list"
            )
        keys = [str(key or "").strip() for key in raw_keys]
        if any(not key for key in keys) or len(keys) != len(set(keys)):
            raise LooLedgerError(
                f"{indicator}: source keys must be non-empty and unique"
            )
        missing = sorted(set(keys) - set(normalised_sources))
        if missing:
            raise LooLedgerError(
                f"{indicator}: unknown source keys in registry: {missing}"
            )
        enabled_keys = [
            key for key in keys if normalised_sources[key]["enabled"]
        ]
        if not enabled_keys:
            raise LooLedgerError(
                f"{indicator}: no enabled canonical source remains"
            )
        normalised_indicators[indicator] = enabled_keys

    return normalised_sources, normalised_indicators, deepcopy(dict(policy))


def _ensure_safe_relative_file(base: Path, relative: Any, *, label: str) -> Path:
    text = str(relative or "").strip()
    pure = PurePosixPath(text)
    if not text or pure.is_absolute() or ".." in pure.parts:
        raise LooLedgerIntegrityError(
            f"{label} must be a safe relative POSIX path, got {text!r}"
        )
    candidate = base.joinpath(*pure.parts)
    resolved_base = base.resolve()
    resolved = candidate.resolve()
    if not resolved.is_relative_to(resolved_base):
        raise LooLedgerIntegrityError(f"{label} escapes its manifest directory")
    current = resolved_base
    for part in resolved.relative_to(resolved_base).parts:
        current = current / part
        if current.is_symlink():
            raise LooLedgerIntegrityError(f"{label} traverses a symlink: {current}")
    if not resolved.is_file():
        raise LooLedgerIntegrityError(f"{label} is not a regular file: {resolved}")
    return resolved


def _normalise_request_array(value: Any, *, label: str) -> list[str]:
    """Normalise scalar and batched fetcher fields without changing request IDs."""

    if isinstance(value, list):
        values = [str(item or "").strip() for item in value]
    elif isinstance(value, str) and "," in value:
        values = [item.strip() for item in value.split(",")]
    else:
        values = [str(value or "").strip()]
    if not values or any(not item for item in values):
        raise LooLedgerError(f"{label} must contain non-empty values")
    if len(values) > 12:
        raise LooLedgerError(f"{label} may contain at most 12 batched values")
    return values


def _validate_snapshot_manifest(
    path: Path,
    *,
    registry_path: Path,
    registry_sha256: str,
    population_id: str,
    meeting_dates: Sequence[date],
    sources: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    str,
    dict[str, Any],
]:
    manifest = _load_json(path, label="snapshot manifest")
    if manifest.get("schema_version") != SNAPSHOT_MANIFEST_SCHEMA_VERSION:
        raise LooLedgerError(
            f"Snapshot schema must be {SNAPSHOT_MANIFEST_SCHEMA_VERSION!r}"
        )
    if manifest.get("status") != "complete":
        raise LooLedgerError("Snapshot manifest status must be complete")
    try:
        payload_sha256 = validate_manifest_integrity(manifest)
    except ManifestIntegrityError as exc:
        raise LooLedgerIntegrityError(str(exc)) from exc

    registry_binding = manifest.get("registry")
    if not isinstance(registry_binding, Mapping):
        raise LooLedgerError("Snapshot manifest requires a registry binding")
    try:
        bound_registry_sha = validate_sha256(
            registry_binding.get("sha256"),
            label="snapshot registry.sha256",
        )
    except ValueError as exc:
        raise LooLedgerIntegrityError(str(exc)) from exc
    if bound_registry_sha != registry_sha256:
        raise LooLedgerIntegrityError(
            "Snapshot manifest was built from a different source registry"
        )

    raw_meetings = manifest.get("meetings")
    if not isinstance(raw_meetings, list):
        raise LooLedgerError("Snapshot manifest requires a meetings list")
    meeting_index: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw_meetings):
        if not isinstance(item, Mapping):
            raise LooLedgerError(f"Snapshot meetings[{index}] must be an object")
        meeting = _parse_iso_date(
            item.get("meeting_date"),
            label=f"snapshot meetings[{index}].meeting_date",
        )
        as_of = _parse_iso_date(
            item.get("information_as_of_date"),
            label=f"snapshot meetings[{index}].information_as_of_date",
        )
        if as_of != information_as_of_date(meeting):
            raise LooLedgerError(
                f"{meeting}: information_as_of_date must be the previous "
                "calendar day"
            )
        population_ids = item.get("population_ids")
        if (
            not isinstance(population_ids, list)
            or any(not isinstance(value, str) for value in population_ids)
        ):
            raise LooLedgerError(
                f"{meeting}: population_ids must be a list of strings"
            )
        meeting_text = meeting.isoformat()
        if meeting_text in meeting_index:
            raise LooLedgerError(
                f"Duplicate snapshot meeting record: {meeting_text}"
            )
        meeting_index[meeting_text] = dict(item)

    expected_meetings = {value.isoformat() for value in meeting_dates}
    for meeting_text in sorted(expected_meetings):
        item = meeting_index.get(meeting_text)
        if item is None or population_id not in item["population_ids"]:
            raise LooLedgerError(
                f"Snapshot manifest does not bind {population_id!r} to "
                f"meeting {meeting_text}"
            )

    raw_requests = manifest.get("requests")
    if not isinstance(raw_requests, list):
        raise LooLedgerError("Snapshot manifest requires a requests list")
    requests: dict[tuple[str, str], dict[str, Any]] = {}
    required_request_fields = (
        "request_id",
        "source_key",
        "series_id",
        "vintage_dates",
        "cosd",
        "coed",
        "canonical_url",
        "raw_relative_path",
        "raw_sha256",
        "byte_count",
        "content_type",
        "retrieved_at_utc",
        "status_code",
    )
    for index, raw_request in enumerate(raw_requests):
        if not isinstance(raw_request, Mapping):
            raise LooLedgerError(f"Snapshot requests[{index}] must be an object")
        missing = [
            field for field in required_request_fields if field not in raw_request
        ]
        if missing:
            raise LooLedgerError(
                f"Snapshot requests[{index}] is missing fields {missing}"
            )
        request = deepcopy(dict(raw_request))
        source_key = str(request["source_key"] or "").strip()
        if source_key not in sources:
            raise LooLedgerError(
                f"Snapshot request references unknown source {source_key!r}"
            )
        source = sources[source_key]
        if not source["enabled"]:
            continue
        if str(request["series_id"] or "").strip() != source["series_id"]:
            raise LooLedgerError(
                f"{source_key}: snapshot series_id differs from registry"
            )
        vintage_values = _normalise_request_array(
            request["vintage_dates"],
            label=f"{source_key} vintage_dates",
        )
        cosd_values = _normalise_request_array(
            request["cosd"],
            label=f"{source_key} cosd",
        )
        coed_values = _normalise_request_array(
            request["coed"],
            label=f"{source_key} coed",
        )
        if not (
            len(vintage_values) == len(cosd_values) == len(coed_values)
        ):
            raise LooLedgerError(
                f"{source_key}: vintage_dates, cosd, and coed batches must "
                "have equal lengths"
            )
        canonical_url = str(request["canonical_url"] or "").strip()
        if not canonical_url.startswith("https://"):
            raise LooLedgerError(
                f"{source_key}: canonical_url must use HTTPS"
            )
        if "api_key" in canonical_url.lower():
            raise LooLedgerError(
                f"{source_key}: canonical_url leaks an API key"
            )
        observed_request_id = compute_request_id(request)
        try:
            recorded_request_id = validate_sha256(
                request["request_id"],
                label=f"{source_key} request_id",
            )
        except ValueError as exc:
            raise LooLedgerIntegrityError(str(exc)) from exc
        if observed_request_id != recorded_request_id:
            raise LooLedgerIntegrityError(
                f"{source_key}: request_id does not bind the "
                "canonical request descriptor"
            )
        if request["status_code"] != 200:
            raise LooLedgerError(
                f"{source_key}: snapshot status_code must be 200"
            )
        if not isinstance(request["byte_count"], int) or request["byte_count"] <= 0:
            raise LooLedgerError(
                f"{source_key}: byte_count must be positive"
            )
        content_type = str(request["content_type"] or "").lower()
        if not any(
            marker in content_type
            for marker in ("csv", "text/plain", "octet-stream")
        ):
            raise LooLedgerError(
                f"{source_key}: unexpected CSV content_type "
                f"{request['content_type']!r}"
            )
        retrieved = _parse_timestamp(
            request["retrieved_at_utc"],
            label=f"{source_key} retrieved_at_utc",
        )
        raw_path = _ensure_safe_relative_file(
            path.parent,
            request["raw_relative_path"],
            label=f"{source_key} raw_relative_path",
        )
        if raw_path.stat().st_size != request["byte_count"]:
            raise LooLedgerIntegrityError(
                f"{source_key}: raw byte count changed"
            )
        try:
            recorded_raw_sha = validate_sha256(
                request["raw_sha256"],
                label=f"{source_key} raw_sha256",
            )
        except ValueError as exc:
            raise LooLedgerIntegrityError(str(exc)) from exc
        observed_raw_sha = sha256_file(raw_path)
        if observed_raw_sha != recorded_raw_sha:
            raise LooLedgerIntegrityError(
                f"{source_key}: raw CSV SHA-256 mismatch"
            )
        for batch_index, (
            raw_vintage,
            raw_cosd,
            raw_coed,
        ) in enumerate(zip(vintage_values, cosd_values, coed_values)):
            vintage = _parse_iso_date(
                raw_vintage,
                label=f"{source_key} vintage_dates[{batch_index}]",
            )
            coed = _parse_iso_date(
                raw_coed,
                label=f"{source_key} coed[{batch_index}]",
            )
            cosd = _parse_iso_date(
                raw_cosd,
                label=f"{source_key} cosd[{batch_index}]",
            )
            meeting = vintage + timedelta(days=1)
            meeting_text = meeting.isoformat()
            if meeting_text not in meeting_index:
                continue
            if vintage != information_as_of_date(meeting) or coed != vintage:
                raise LooLedgerError(
                    f"{source_key}/{meeting_text}: request must use D-1 for "
                    "both vintage_dates and coed"
                )
            if cosd > coed:
                raise LooLedgerError(
                    f"{source_key}/{meeting_text}: cosd is after coed"
                )
            key = (meeting_text, source_key)
            if key in requests:
                raise LooLedgerError(
                    f"Duplicate snapshot request for {source_key}/{meeting_text}"
                )
            expanded = deepcopy(request)
            expanded.update(
                {
                    "_raw_path": raw_path,
                    "_retrieved": retrieved,
                    "_cosd": cosd,
                    "_coed": coed,
                    "_vintage": vintage,
                    "_batch_index": batch_index,
                }
            )
            requests[key] = expanded

    return requests, payload_sha256, manifest


def _subtract_months(value: date, months: int) -> date:
    absolute_month = value.year * 12 + (value.month - 1) - months
    year, zero_based_month = divmod(absolute_month, 12)
    month = zero_based_month + 1
    month_lengths = (
        31,
        29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    )
    return date(year, month, min(value.day, month_lengths[month - 1]))


def _calendar_lookback_start(as_of: date, policy: Mapping[str, Any]) -> date | None:
    if policy["mode"] != "calendar":
        return None
    amount = int(policy["value"])
    unit = policy["unit"]
    if unit == "days":
        return as_of - timedelta(days=amount)
    if unit == "weeks":
        return as_of - timedelta(weeks=amount)
    if unit == "months":
        return _subtract_months(as_of, amount)
    if unit == "quarters":
        return _subtract_months(as_of, amount * 3)
    if unit == "years":
        return _subtract_months(as_of, amount * 12)
    raise AssertionError(f"Unhandled lookback unit {unit!r}")


def _parse_csv_observations(
    raw_bytes: bytes,
    *,
    source: Mapping[str, Any],
    request: Mapping[str, Any],
    meeting_date: date,
    indicator: str,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    source_key = str(source["source_key"])
    sample_id = f"{meeting_date.isoformat()}::{indicator}"
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise LooLedgerError(f"{source_key}: raw CSV is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames
    if not fieldnames or len(set(fieldnames)) != len(fieldnames):
        raise LooLedgerError(
            f"{source_key}: raw CSV needs unique non-empty headers"
        )
    date_column = next(
        (candidate for candidate in _DATE_COLUMNS if candidate in fieldnames),
        None,
    )
    if date_column is None:
        raise LooLedgerError(
            f"{source_key}: raw CSV lacks observation_date/date column"
        )
    configured_value_column = str(source.get("value_column") or "").strip()
    value_column_template = str(
        source.get("value_column_template") or ""
    ).strip()
    if value_column_template:
        vintage_text = request["_vintage"].isoformat()
        try:
            configured_value_column = value_column_template.format(
                series_id=source["series_id"],
                vintage_date=vintage_text,
                vintage_date_yyyymmdd=vintage_text.replace("-", ""),
            )
        except (KeyError, ValueError) as exc:
            raise LooLedgerError(
                f"{source_key}: invalid value_column_template "
                f"{value_column_template!r}"
            ) from exc
    if configured_value_column:
        if configured_value_column not in fieldnames:
            raise LooLedgerError(
                f"{source_key}: configured value_column "
                f"{configured_value_column!r} is absent"
            )
        value_column = configured_value_column
    elif source["series_id"] in fieldnames:
        value_column = str(source["series_id"])
    else:
        candidates = [name for name in fieldnames if name != date_column]
        if len(candidates) != 1:
            raise LooLedgerError(
                f"{source_key}: cannot infer one value column from {fieldnames}"
            )
        value_column = candidates[0]

    as_of = request["_vintage"]
    cosd = request["_cosd"]
    accepted: list[dict[str, str]] = []
    excluded: list[dict[str, Any]] = []
    seen_dates: set[str] = set()
    for row_number, row in enumerate(reader, 2):
        raw_record_sha256 = sha256_text(_canonical_json(row))
        raw_date = str(row.get(date_column) or "").strip()
        try:
            observation_date = _parse_iso_date(
                raw_date,
                label=f"{source_key} CSV row {row_number} date",
            )
        except LooLedgerError:
            excluded.append(
                {
                    "schema_version": EXCLUSION_SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "source_key": source_key,
                    "row_number": row_number,
                    "reason": "invalid_observation_date",
                    "raw_record_sha256": raw_record_sha256,
                }
            )
            continue
        if raw_date in seen_dates:
            raise LooLedgerError(
                f"{source_key}: duplicate observation date {raw_date}"
            )
        seen_dates.add(raw_date)
        if observation_date > as_of:
            excluded.append(
                {
                    "schema_version": EXCLUSION_SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "source_key": source_key,
                    "observation_date": raw_date,
                    "reason": "observation_after_information_as_of_date",
                    "raw_record_sha256": raw_record_sha256,
                }
            )
            continue
        if observation_date < cosd:
            excluded.append(
                {
                    "schema_version": EXCLUSION_SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "source_key": source_key,
                    "observation_date": raw_date,
                    "reason": "observation_before_request_window",
                    "raw_record_sha256": raw_record_sha256,
                }
            )
            continue
        raw_value = str(row.get(value_column) or "").strip()
        if raw_value in ("", ".", "NA", "N/A", "NaN", "nan", "null"):
            excluded.append(
                {
                    "schema_version": EXCLUSION_SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "source_key": source_key,
                    "observation_date": raw_date,
                    "reason": "missing_observation_value",
                    "raw_record_sha256": raw_record_sha256,
                }
            )
            continue
        try:
            decimal_value = Decimal(raw_value)
        except InvalidOperation:
            decimal_value = Decimal("NaN")
        if not decimal_value.is_finite():
            excluded.append(
                {
                    "schema_version": EXCLUSION_SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "source_key": source_key,
                    "observation_date": raw_date,
                    "reason": "non_finite_observation_value",
                    "raw_record_sha256": raw_record_sha256,
                }
            )
            continue
        accepted.append({"date": raw_date, "value": raw_value})

    accepted.sort(key=lambda item: item["date"])
    return accepted, excluded


def _sample_observations(
    observations: Sequence[dict[str, str]],
    *,
    source: Mapping[str, Any],
    sample_id: str,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    as_of = _parse_iso_date(observations[-1]["date"], label="latest observation")
    selected = list(observations)
    frequency = source["frequency"]
    true_as_of = source["_information_as_of_date"]
    if frequency in {"daily", "weekly", "biweekly"}:
        # The primary design fixes a calendar window for high-frequency series,
        # independent of how a registry request is over-provisioned.
        lower = _subtract_months(true_as_of, 24)
        selected = [
            item
            for item in selected
            if _parse_iso_date(item["date"], label="observation date") >= lower
        ]
    # Monthly and lower-frequency series are count-based in the approved
    # design.  Apply their frozen tail caps below directly to the safe,
    # request-bounded observations.  Pre-applying a generic calendar window
    # would incorrectly discard annual period labels near the boundary (for
    # example 2020-01-01 for a 2022-01-25 information date).

    if frequency in {"daily", "weekly", "biweekly"} and selected:
        by_month: dict[tuple[int, int], dict[str, str]] = {}
        for item in selected:
            parsed = _parse_iso_date(item["date"], label="observation date")
            by_month[(parsed.year, parsed.month)] = item
        selected = [by_month[key] for key in sorted(by_month)]
        selected = selected[-25:]
    else:
        frequency_caps = {
            "monthly": 24,
            "quarterly": 8,
            "semiannual": 6,
            "annual": 5,
        }
        cap = frequency_caps[frequency]
        selected = selected[-cap:]

    selected_dates = {item["date"] for item in selected}
    excluded = [
        {
            "schema_version": EXCLUSION_SCHEMA_VERSION,
            "sample_id": sample_id,
            "source_key": source["source_key"],
            "observation_date": item["date"],
            "reason": "outside_frozen_sampling_policy",
            "raw_record_sha256": sha256_text(_canonical_json(item)),
        }
        for item in observations
        if item["date"] not in selected_dates
    ]
    if not selected:
        latest = as_of.isoformat() if observations else "none"
        raise LooLedgerError(
            f"{sample_id}/{source['source_key']}: no observation survives the "
            f"frozen sampling policy (latest candidate={latest})"
        )
    return selected, excluded


def _serialise_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(_canonical_json(row) + "\n" for row in rows)


def _serialise_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _relative_output_record(
    *,
    relative_path: str,
    content: str,
    row_count: int | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": relative_path,
        "sha256": sha256_text(content),
    }
    if row_count is not None:
        record["row_count"] = row_count
    return record


def _build_artifacts(
    *,
    registry_path: Path,
    snapshot_manifest_path: Path,
    population_path: Path,
    roster_path: Path,
) -> dict[str, Any]:
    roster = _load_roster(roster_path)
    population_id, meeting_dates = _load_population(population_path)
    registry_sha256 = sha256_file(registry_path)
    sources, indicator_sources, registry_policy = _load_registry(
        registry_path,
        roster=roster,
    )
    requests, snapshot_payload_sha256, snapshot_manifest = (
        _validate_snapshot_manifest(
            snapshot_manifest_path,
            registry_path=registry_path,
            registry_sha256=registry_sha256,
            population_id=population_id,
            meeting_dates=meeting_dates,
            sources=sources,
        )
    )

    ledger_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for meeting in meeting_dates:
        meeting_text = meeting.isoformat()
        as_of = information_as_of_date(meeting)
        as_of_text = as_of.isoformat()
        for indicator in roster:
            sample_id = f"{meeting_text}::{indicator}"
            series_payloads: list[dict[str, Any]] = []
            request_evidence: list[dict[str, Any]] = []
            retrieval_timestamps: list[datetime] = []
            all_observation_dates: list[str] = []

            for source_key in indicator_sources[indicator]:
                source = deepcopy(sources[source_key])
                source["_information_as_of_date"] = as_of
                request = requests.get((meeting_text, source_key))
                if request is None:
                    raise LooLedgerError(
                        f"Missing D-1 snapshot request for "
                        f"{source_key}/{meeting_text}"
                    )
                raw_path = request["_raw_path"]
                raw_bytes = raw_path.read_bytes()
                observations, parse_exclusions = _parse_csv_observations(
                    raw_bytes,
                    source=source,
                    request=request,
                    meeting_date=meeting,
                    indicator=indicator,
                )
                exclusion_rows.extend(parse_exclusions)
                if not observations:
                    raise LooLedgerError(
                        f"{sample_id}/{source_key}: raw snapshot has no safe "
                        "numeric observations"
                    )
                sampled, sample_exclusions = _sample_observations(
                    observations,
                    source=source,
                    sample_id=sample_id,
                )
                exclusion_rows.extend(sample_exclusions)
                all_observation_dates.extend(item["date"] for item in sampled)
                retrieval_timestamps.append(request["_retrieved"])
                title = str(
                    source.get("title")
                    or source.get("label")
                    or source["series_id"]
                ).strip()
                series_payloads.append(
                    {
                        "source_key": source_key,
                        "series_id": source["series_id"],
                        "title": title,
                        "frequency": source["frequency"],
                        "units": source["units"],
                        "seasonal_adjustment": source["seasonal_adjustment"],
                        "transformation": source["transformation"],
                        "availability_as_of_date": as_of_text,
                        "requested_vintage_date": as_of_text,
                        "observations": sampled,
                    }
                )
                request_evidence.append(
                    {
                        "source_key": source_key,
                        "series_id": source["series_id"],
                        "license": source["license"],
                        "redistribution_allowed": source[
                            "redistribution_allowed"
                        ],
                        "request_id": request["request_id"],
                        "raw_relative_path": request["raw_relative_path"],
                        "raw_sha256": request["raw_sha256"],
                        "byte_count": request["byte_count"],
                        "retrieved_at_utc": _format_timestamp(
                            request["_retrieved"]
                        ),
                        "request_batch": {
                            "vintage_dates": request["vintage_dates"],
                            "cosd": request["cosd"],
                            "coed": request["coed"],
                        },
                        "selected_batch_index": request["_batch_index"],
                        "requested_vintage_date": request[
                            "_vintage"
                        ].isoformat(),
                        "observation_start": request["_cosd"].isoformat(),
                        "observation_end": request["_coed"].isoformat(),
                        "candidate_observation_count": len(observations),
                        "selected_observation_count": len(sampled),
                        "selected_observation_sha256": sha256_text(
                            _canonical_json(sampled)
                        ),
                    }
                )

            source_payload = {
                "sampling_policy": {
                    "version": SAMPLING_POLICY_VERSION,
                    "daily_weekly": (
                        "trailing 24 calendar months; last available observation "
                        "per month plus latest; maximum 25"
                    ),
                    "monthly": "latest 24 safe observations",
                    "quarterly": "latest 8 safe observations",
                    "semiannual": "latest 6 safe observations",
                    "annual": "latest 5 safe observations",
                },
                "series": series_payloads,
            }
            _reject_forbidden_keys(source_payload)
            source_payload_sha256 = sha256_text(_canonical_json(source_payload))
            source_id = (
                f"canonical-loo-d1:{population_id}:{meeting_text}:{indicator}"
            )
            evidence_binding = {
                "schema_version": SOURCE_EVIDENCE_SCHEMA_VERSION,
                "population_id": population_id,
                "sample_id": sample_id,
                "source_id": source_id,
                "information_as_of_date": as_of_text,
                "registry_sha256": registry_sha256,
                "snapshot_manifest_payload_sha256": snapshot_payload_sha256,
                "source_payload_sha256": source_payload_sha256,
                "requests": request_evidence,
            }
            source_sha256 = sha256_text(_canonical_json(evidence_binding))
            evidence_row = {
                **evidence_binding,
                "source_sha256": source_sha256,
            }
            source_timestamp = _format_timestamp(max(retrieval_timestamps))
            observation_date = max(all_observation_dates)
            ledger_row = {
                "schema_version": LEDGER_ROW_SCHEMA_VERSION,
                "meeting_id": meeting_text,
                "sample_id": sample_id,
                "meeting_timestamp": decision_identity_timestamp(meeting),
                "meeting_date": meeting_text,
                "indicator": indicator,
                "source_id": source_id,
                "source_sha256": source_sha256,
                "source_timestamp": source_timestamp,
                "information_as_of_date": as_of_text,
                "requested_vintage_date": as_of_text,
                "availability_as_of_date": as_of_text,
                "availability_evidence_type": AVAILABILITY_EVIDENCE_TYPE,
                "source_interface": SOURCE_INTERFACE,
                "observation_date": observation_date,
                "source_payload": source_payload,
            }
            ledger_rows.append(ledger_row)
            evidence_rows.append(evidence_row)
            coverage_rows.append(
                {
                    "sample_id": sample_id,
                    "meeting_date": meeting_text,
                    "information_as_of_date": as_of_text,
                    "indicator": indicator,
                    "source_count": len(series_payloads),
                    "observation_count": len(all_observation_dates),
                    "source_keys": [
                        item["source_key"] for item in series_payloads
                    ],
                    "latest_observation_date": observation_date,
                }
            )

    if len(ledger_rows) != EXPECTED_LEDGER_ROW_COUNT:
        raise LooLedgerError(
            f"Canonical ledger must contain {EXPECTED_LEDGER_ROW_COUNT} rows, "
            f"constructed {len(ledger_rows)}"
        )
    if len({row["sample_id"] for row in ledger_rows}) != len(ledger_rows):
        raise LooLedgerError("Canonical ledger contains duplicate sample IDs")

    exclusion_rows.sort(
        key=lambda row: (
            str(row.get("sample_id") or ""),
            str(row.get("source_key") or ""),
            str(row.get("observation_date") or ""),
            str(row.get("row_number") or ""),
            str(row.get("reason") or ""),
        )
    )
    coverage = {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "population_id": population_id,
        "meeting_count": len(meeting_dates),
        "indicator_count": len(roster),
        "row_count": len(ledger_rows),
        "source_series_count": sum(
            row["source_count"] for row in coverage_rows
        ),
        "observation_count": sum(
            row["observation_count"] for row in coverage_rows
        ),
        "excluded_record_count": len(exclusion_rows),
        "information_as_of_dates": {
            meeting.isoformat(): information_as_of_date(meeting).isoformat()
            for meeting in meeting_dates
        },
        "rows": coverage_rows,
    }
    contents = {
        "indicator_inputs.jsonl": _serialise_jsonl(ledger_rows),
        "source_evidence.jsonl": _serialise_jsonl(evidence_rows),
        "excluded_records.jsonl": _serialise_jsonl(exclusion_rows),
        "coverage.json": _serialise_json(coverage),
    }
    snapshot_sha256 = sha256_file(snapshot_manifest_path)
    active_source_keys = list(
        dict.fromkeys(
            source_key
            for indicator in roster
            for source_key in indicator_sources[indicator]
        )
    )
    license_records = [
        {
            "source_key": source_key,
            "license": sources[source_key]["license"],
            "redistribution_allowed": sources[source_key][
                "redistribution_allowed"
            ],
        }
        for source_key in active_source_keys
    ]
    manifest_payload = {
        "schema_version": LEDGER_MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "population_id": population_id,
        "information_cutoff_policy": INFORMATION_CUTOFF_POLICY,
        "license_summary": {
            "local_research_use_only": any(
                not record["redistribution_allowed"]
                for record in license_records
            ),
            "redistribution_allowed_for_all_sources": all(
                record["redistribution_allowed"]
                for record in license_records
            ),
            "sources": license_records,
        },
        "inputs": {
            "registry": {
                "path": str(registry_path),
                "sha256": registry_sha256,
            },
            "snapshot_manifest": {
                "path": str(snapshot_manifest_path),
                "sha256": snapshot_sha256,
                "payload_sha256": snapshot_payload_sha256,
            },
            "population": {
                "path": str(population_path),
                "sha256": sha256_file(population_path),
            },
            "roster": {
                "path": str(roster_path),
                "sha256": sha256_file(roster_path),
            },
        },
        "outputs": {
            "indicator_inputs": _relative_output_record(
                relative_path="indicator_inputs.jsonl",
                content=contents["indicator_inputs.jsonl"],
                row_count=len(ledger_rows),
            ),
            "source_evidence": _relative_output_record(
                relative_path="source_evidence.jsonl",
                content=contents["source_evidence.jsonl"],
                row_count=len(evidence_rows),
            ),
            "excluded_records": _relative_output_record(
                relative_path="excluded_records.jsonl",
                content=contents["excluded_records.jsonl"],
                row_count=len(exclusion_rows),
            ),
            "coverage": _relative_output_record(
                relative_path="coverage.json",
                content=contents["coverage.json"],
            ),
        },
    }
    manifest = seal_manifest(manifest_payload)
    contents["ledger_manifest.json"] = _serialise_json(manifest)
    return {
        "manifest": manifest,
        "contents": contents,
        "ledger_rows": ledger_rows,
        "evidence_rows": evidence_rows,
        "exclusion_rows": exclusion_rows,
        "coverage": coverage,
        "registry_policy": registry_policy,
        "snapshot_manifest": snapshot_manifest,
    }


def _write_content_immutably(path: Path, content: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != content:
            raise LooLedgerIntegrityError(
                f"Refusing to overwrite incompatible frozen artifact {path}"
            )
        return
    if path.exists():
        raise LooLedgerIntegrityError(
            f"Artifact destination exists but is not a regular file: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_loo_indicator_ledger(
    *,
    registry_file: str | Path,
    snapshot_manifest_file: str | Path,
    population_file: str | Path,
    roster_file: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build one immutable, fully re-playable D-1 population ledger."""

    registry_path = Path(registry_file).expanduser().resolve()
    snapshot_path = Path(snapshot_manifest_file).expanduser().resolve()
    population_path = Path(population_file).expanduser().resolve()
    roster_path = Path(roster_file).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    for path, label in (
        (registry_path, "registry"),
        (snapshot_path, "snapshot manifest"),
        (population_path, "population"),
        (roster_path, "roster"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    artifacts = _build_artifacts(
        registry_path=registry_path,
        snapshot_manifest_path=snapshot_path,
        population_path=population_path,
        roster_path=roster_path,
    )
    output_path.mkdir(parents=True, exist_ok=True)
    for name in (
        "indicator_inputs.jsonl",
        "source_evidence.jsonl",
        "excluded_records.jsonl",
        "coverage.json",
    ):
        _write_content_immutably(output_path / name, artifacts["contents"][name])
    # The sealed manifest is the commit marker and is written last.
    _write_content_immutably(
        output_path / "ledger_manifest.json",
        artifacts["contents"]["ledger_manifest.json"],
    )
    validate_loo_indicator_ledger(
        ledger_manifest_file=output_path / "ledger_manifest.json",
        registry_file=registry_path,
        snapshot_manifest_file=snapshot_path,
        population_file=population_path,
        roster_file=roster_path,
    )
    return artifacts["manifest"]


def _validate_output_record(
    *,
    manifest_dir: Path,
    record: Any,
    label: str,
    expected_row_count: int | None = None,
) -> Path:
    if not isinstance(record, Mapping):
        raise LooLedgerIntegrityError(f"Manifest output {label} must be an object")
    path = _ensure_safe_relative_file(
        manifest_dir,
        record.get("path"),
        label=f"outputs.{label}.path",
    )
    try:
        expected_sha = validate_sha256(
            record.get("sha256"),
            label=f"outputs.{label}.sha256",
        )
    except ValueError as exc:
        raise LooLedgerIntegrityError(str(exc)) from exc
    if sha256_file(path) != expected_sha:
        raise LooLedgerIntegrityError(f"Output hash mismatch for {label}")
    if expected_row_count is not None and record.get("row_count") != expected_row_count:
        raise LooLedgerIntegrityError(
            f"Output row count mismatch for {label}: "
            f"{record.get('row_count')} != {expected_row_count}"
        )
    return path


def validate_loo_indicator_ledger(
    *,
    ledger_manifest_file: str | Path,
    registry_file: str | Path,
    snapshot_manifest_file: str | Path,
    population_file: str | Path,
    roster_file: str | Path,
    expected_manifest_payload_sha256: str | None = None,
) -> dict[str, Any]:
    """Rebuild a D-1 ledger from raw bytes and verify every frozen artifact."""

    manifest_path = Path(ledger_manifest_file).expanduser().resolve()
    registry_path = Path(registry_file).expanduser().resolve()
    snapshot_path = Path(snapshot_manifest_file).expanduser().resolve()
    population_path = Path(population_file).expanduser().resolve()
    roster_path = Path(roster_file).expanduser().resolve()
    manifest = _load_json(manifest_path, label="ledger manifest")
    if manifest.get("schema_version") != LEDGER_MANIFEST_SCHEMA_VERSION:
        raise LooLedgerIntegrityError(
            f"Ledger manifest schema must be {LEDGER_MANIFEST_SCHEMA_VERSION!r}"
        )
    if manifest.get("status") != "complete":
        raise LooLedgerIntegrityError("Ledger manifest status must be complete")
    if manifest.get("information_cutoff_policy") != INFORMATION_CUTOFF_POLICY:
        raise LooLedgerIntegrityError("Ledger cutoff policy is not canonical D-1")
    try:
        payload_sha256 = validate_manifest_integrity(
            manifest,
            expected_payload_sha256=expected_manifest_payload_sha256,
        )
    except ManifestIntegrityError as exc:
        raise LooLedgerIntegrityError(str(exc)) from exc

    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise LooLedgerIntegrityError("Ledger manifest lacks inputs")
    expected_inputs = {
        "registry": (registry_path, None),
        "snapshot_manifest": (snapshot_path, "payload_sha256"),
        "population": (population_path, None),
        "roster": (roster_path, None),
    }
    for label, (path, _) in expected_inputs.items():
        record = inputs.get(label)
        if not isinstance(record, Mapping):
            raise LooLedgerIntegrityError(f"Ledger input {label} is missing")
        try:
            expected_sha = validate_sha256(
                record.get("sha256"),
                label=f"inputs.{label}.sha256",
            )
        except ValueError as exc:
            raise LooLedgerIntegrityError(str(exc)) from exc
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise LooLedgerIntegrityError(f"Ledger input changed: {label}")

    expected = _build_artifacts(
        registry_path=registry_path,
        snapshot_manifest_path=snapshot_path,
        population_path=population_path,
        roster_path=roster_path,
    )
    if manifest != expected["manifest"]:
        raise LooLedgerIntegrityError(
            "Ledger manifest does not match the replayed raw evidence"
        )

    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise LooLedgerIntegrityError("Ledger manifest lacks outputs")
    paths = {
        "indicator_inputs": _validate_output_record(
            manifest_dir=manifest_path.parent,
            record=outputs.get("indicator_inputs"),
            label="indicator_inputs",
            expected_row_count=EXPECTED_LEDGER_ROW_COUNT,
        ),
        "source_evidence": _validate_output_record(
            manifest_dir=manifest_path.parent,
            record=outputs.get("source_evidence"),
            label="source_evidence",
            expected_row_count=EXPECTED_LEDGER_ROW_COUNT,
        ),
        "excluded_records": _validate_output_record(
            manifest_dir=manifest_path.parent,
            record=outputs.get("excluded_records"),
            label="excluded_records",
            expected_row_count=len(expected["exclusion_rows"]),
        ),
        "coverage": _validate_output_record(
            manifest_dir=manifest_path.parent,
            record=outputs.get("coverage"),
            label="coverage",
        ),
    }
    observed_ledger = _read_jsonl(
        paths["indicator_inputs"], label="indicator inputs"
    )
    observed_evidence = _read_jsonl(
        paths["source_evidence"], label="source evidence"
    )
    observed_exclusions = _read_jsonl(
        paths["excluded_records"], label="excluded records"
    )
    observed_coverage = _load_json(paths["coverage"], label="coverage")
    if observed_ledger != expected["ledger_rows"]:
        raise LooLedgerIntegrityError(
            "indicator_inputs.jsonl cannot be reconstructed from raw evidence"
        )
    if observed_evidence != expected["evidence_rows"]:
        raise LooLedgerIntegrityError(
            "source_evidence.jsonl cannot be reconstructed from raw evidence"
        )
    if observed_exclusions != expected["exclusion_rows"]:
        raise LooLedgerIntegrityError(
            "excluded_records.jsonl cannot be reconstructed from raw evidence"
        )
    if observed_coverage != expected["coverage"]:
        raise LooLedgerIntegrityError(
            "coverage.json cannot be reconstructed from raw evidence"
        )

    for row in observed_ledger:
        meeting = _parse_iso_date(row["meeting_date"], label="meeting_date")
        as_of = information_as_of_date(meeting).isoformat()
        if any(
            row[field] != as_of
            for field in (
                "information_as_of_date",
                "requested_vintage_date",
                "availability_as_of_date",
            )
        ):
            raise LooLedgerIntegrityError(
                f"{row['sample_id']}: row is not frozen at D-1"
            )
        if row["observation_date"] > as_of:
            raise LooLedgerIntegrityError(
                f"{row['sample_id']}: top-level observation date is post-cutoff"
            )
        _reject_forbidden_keys(row["source_payload"])

    return {
        "status": "valid",
        "schema_version": LEDGER_MANIFEST_SCHEMA_VERSION,
        "population_id": manifest["population_id"],
        "row_count": EXPECTED_LEDGER_ROW_COUNT,
        "manifest_payload_sha256": payload_sha256,
        "ledger_sha256": manifest["outputs"]["indicator_inputs"]["sha256"],
        "excluded_record_count": len(observed_exclusions),
    }


__all__ = [
    "AVAILABILITY_EVIDENCE_TYPE",
    "COVERAGE_SCHEMA_VERSION",
    "EXPECTED_LEDGER_ROW_COUNT",
    "INFORMATION_CUTOFF_POLICY",
    "LEDGER_MANIFEST_SCHEMA_VERSION",
    "LEDGER_ROW_SCHEMA_VERSION",
    "LooLedgerError",
    "LooLedgerIntegrityError",
    "SAMPLING_POLICY_VERSION",
    "SOURCE_INTERFACE",
    "build_loo_indicator_ledger",
    "compute_request_id",
    "decision_identity_timestamp",
    "information_as_of_date",
    "validate_loo_indicator_ledger",
]
