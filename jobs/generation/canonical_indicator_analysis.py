"""Canonical, generation-only construction of meeting-time indicator analyses.

This stage deliberately has no training or checkpoint-writing interface.  It
validates a complete meeting-by-indicator panel, builds deterministic prompts,
invokes the project's existing generation helper, and binds the resulting
JSONL to an immutable-style manifest.

Canonical input JSONL schema (one row per meeting and roster indicator):

* ``schema_version``: ``canonical-loo-indicator-input-v2``.
* ``meeting_id``: stable meeting identifier.
* ``sample_id``: exactly ``<meeting_id>::<indicator>``.
* ``meeting_timestamp``: decision timestamp retained as meeting metadata.
* ``information_as_of_date``: exactly the calendar day before the meeting.
* ``requested_vintage_date`` and ``availability_as_of_date``: D-1 evidence.
* ``indicator``: an entry in the supplied roster.
* ``source_id`` and ``source_sha256``: stable source identifier and fingerprint.
* ``source_timestamp``: timezone-aware source-snapshot timestamp.
* ``observation_date``: latest included observation date, never later than D-1.
* ``source_payload``: JSON value containing only data available for the prompt.

The D-1 ALFRED snapshot is date-granular evidence, so the canonical schema does
not manufacture an exact release timestamp.  The archival retrieval timestamp
is recorded but can legitimately be later than the meeting.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from open_r1.provenance import (
    fingerprint_artifact_path,
    sha256_file,
    sha256_text,
    validate_sha256,
)
from open_r1.validator.loo_generation_spec import (
    GenerationSafetyError,
    validate_manifest_integrity,
    validate_generation_completion,
)


SCHEMA_VERSION = "indicator-analysis-generation-v2"
INPUT_SCHEMA_VERSION = "canonical-loo-indicator-input-v2"
PROMPT_TEMPLATE_VERSION = "indicator-analysis-prompt-d1-v1"
SYSTEM_PROMPT_VERSION = "indicator-analysis-system-prompt-v1"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_NEW_TOKENS = 8192
DEFAULT_REQUESTED_MAX_OUTPUT_TOKENS = 4096
DEFAULT_MAX_MODEL_LEN = 16384
DEFAULT_SEED = 20260728
DEFAULT_BATCH_SIZE = 20
DEFAULT_MAX_TOKEN_LIMIT_ERRORS = 0

_PROMPT_TEMPLATE = """Task: produce a meeting-time analysis of one economic indicator.
Prompt template: {prompt_template_version}
FOMC meeting: {meeting_id} ({meeting_timestamp})
Indicator: {indicator}
Information available through: {information_as_of_date}
Requested ALFRED vintage: {requested_vintage_date}
Availability evidence: {availability_evidence_type} as of {availability_as_of_date}
Latest included observation date: {observation_date}
Source ID: {source_id}
Source data: {source_payload}

Use only the source data above. Explain what it showed as of the meeting and its plausible relevance to policymakers. Do not add facts released after the meeting, reconstruct the historical Minutes, or discuss other indicators. Return only the indicator analysis."""

_SYSTEM_PROMPT_TEMPLATE = """You are an economic-analysis assistant preparing one concise input block for an FOMC Minutes generation experiment.

Return only the final indicator analysis. Do not output chain-of-thought, scratch work, planning text, XML reasoning wrappers, headings, or preambles.
Use compact, evidence-focused prose. Prioritize the indicator's direction, magnitude, timing, and plausible policy relevance using only the supplied meeting-time data.
The complete response must not exceed {requested_max_output_tokens} generated tokens. Do not pad the response or attempt to fill the token budget."""

_REQUIRED_FIELDS = (
    "schema_version",
    "meeting_id",
    "sample_id",
    "meeting_timestamp",
    "meeting_date",
    "indicator",
    "source_id",
    "source_sha256",
    "source_timestamp",
    "information_as_of_date",
    "requested_vintage_date",
    "availability_as_of_date",
    "availability_evidence_type",
    "source_interface",
    "observation_date",
    "source_payload",
)


@dataclass(frozen=True)
class IndicatorAnalysisDecoding:
    """Frozen decoding configuration for the upstream analysis generator."""

    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    requested_max_output_tokens: int = DEFAULT_REQUESTED_MAX_OUTPUT_TOKENS
    seed: int = DEFAULT_SEED
    batch_size: int = DEFAULT_BATCH_SIZE
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    max_token_limit_errors: int = DEFAULT_MAX_TOKEN_LIMIT_ERRORS

    def validate(self) -> None:
        if self.temperature != 0.0:
            raise ValueError("Canonical indicator analysis requires temperature=0")
        if self.top_p != 1.0:
            raise ValueError("Canonical indicator analysis requires top_p=1")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if (
            isinstance(self.requested_max_output_tokens, bool)
            or not isinstance(self.requested_max_output_tokens, int)
            or self.requested_max_output_tokens <= 0
        ):
            raise ValueError("requested_max_output_tokens must be a positive integer")
        if self.requested_max_output_tokens > self.max_new_tokens:
            raise ValueError(
                "requested_max_output_tokens must not exceed max_new_tokens"
            )
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_model_len <= self.max_new_tokens:
            raise ValueError("max_model_len must exceed max_new_tokens")
        if (
            isinstance(self.max_token_limit_errors, bool)
            or not isinstance(self.max_token_limit_errors, int)
            or not 0 <= self.max_token_limit_errors <= 2
        ):
            raise ValueError(
                "max_token_limit_errors must be an integer from 0 through 2"
            )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_indicator_analysis_system_prompt(
    requested_max_output_tokens: int,
) -> str:
    """Build the stage-specific concise-output system instruction."""

    if (
        isinstance(requested_max_output_tokens, bool)
        or not isinstance(requested_max_output_tokens, int)
        or requested_max_output_tokens <= 0
    ):
        raise ValueError("requested_max_output_tokens must be a positive integer")
    return _SYSTEM_PROMPT_TEMPLATE.format(
        requested_max_output_tokens=requested_max_output_tokens,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} in {path} must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"Input JSONL contains no rows: {path}")
    return rows


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {label} JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _resolve_manifest_artifact(
    manifest_path: Path,
    value: Any,
    *,
    label: str,
) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} has no path")
    raw = Path(text).expanduser()
    candidate = raw if raw.is_absolute() else manifest_path.parent / raw
    resolved = candidate.resolve()
    root = manifest_path.parent.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} escapes the manifest directory: {resolved}")
    return resolved


def validate_ledger_provenance(
    *,
    input_path: Path,
    ledger_manifest_path: Path,
    snapshot_manifest_path: Path,
    source_registry_path: Path,
    roster_path: Path,
    population_path: Path,
) -> dict[str, str]:
    """Validate the sealed ledger chain before loading any prompt data."""

    ledger = _read_json_object(ledger_manifest_path, label="ledger manifest")
    snapshot = _read_json_object(
        snapshot_manifest_path,
        label="snapshot manifest",
    )
    validate_manifest_integrity(ledger)
    validate_manifest_integrity(snapshot)
    if ledger.get("schema_version") != "loo-indicator-ledger-manifest-v1":
        raise ValueError(
            f"Unsupported ledger manifest schema: {ledger.get('schema_version')!r}"
        )
    if snapshot.get("schema_version") != "loo-source-snapshot-manifest-v1":
        raise ValueError(
            f"Unsupported snapshot manifest schema: {snapshot.get('schema_version')!r}"
        )
    outputs = ledger.get("outputs")
    inputs = ledger.get("inputs")
    if not isinstance(outputs, Mapping) or not isinstance(inputs, Mapping):
        raise ValueError("Ledger manifest must contain inputs and outputs")
    indicator_input = outputs.get("indicator_inputs")
    if not isinstance(indicator_input, Mapping):
        raise ValueError("Ledger manifest does not bind indicator_inputs")
    bound_input_path = _resolve_manifest_artifact(
        ledger_manifest_path,
        indicator_input.get("path"),
        label="ledger indicator_inputs",
    )
    if bound_input_path != input_path:
        raise ValueError("Ledger manifest indicator_inputs path differs from --input")
    expected_input_sha = validate_sha256(
        indicator_input.get("sha256"),
        label="ledger outputs.indicator_inputs.sha256",
    )
    if sha256_file(input_path) != expected_input_sha:
        raise ValueError("Ledger-bound indicator_inputs hash mismatch")

    registry_binding = inputs.get("registry")
    snapshot_binding = inputs.get("snapshot_manifest")
    roster_binding = inputs.get("roster")
    population_binding = inputs.get("population")
    if any(
        not isinstance(binding, Mapping)
        for binding in (
            registry_binding,
            snapshot_binding,
            roster_binding,
            population_binding,
        )
    ):
        raise ValueError(
            "Ledger manifest must bind registry, snapshot_manifest, roster, "
            "and population"
        )
    assert isinstance(registry_binding, Mapping)
    assert isinstance(snapshot_binding, Mapping)
    assert isinstance(roster_binding, Mapping)
    assert isinstance(population_binding, Mapping)
    expected_registry_sha = validate_sha256(
        registry_binding.get("sha256"),
        label="ledger inputs.registry.sha256",
    )
    expected_snapshot_sha = validate_sha256(
        snapshot_binding.get("sha256"),
        label="ledger inputs.snapshot_manifest.sha256",
    )
    expected_roster_sha = validate_sha256(
        roster_binding.get("sha256"),
        label="ledger inputs.roster.sha256",
    )
    expected_population_sha = validate_sha256(
        population_binding.get("sha256"),
        label="ledger inputs.population.sha256",
    )
    if sha256_file(source_registry_path) != expected_registry_sha:
        raise ValueError("Ledger-bound source registry hash mismatch")
    if sha256_file(snapshot_manifest_path) != expected_snapshot_sha:
        raise ValueError("Ledger-bound snapshot manifest hash mismatch")
    if sha256_file(roster_path) != expected_roster_sha:
        raise ValueError("Ledger-bound indicator roster hash mismatch")
    if sha256_file(population_path) != expected_population_sha:
        raise ValueError("Ledger-bound population hash mismatch")

    snapshot_registry = snapshot.get("registry")
    if not isinstance(snapshot_registry, Mapping):
        raise ValueError("Snapshot manifest does not bind its registry")
    if snapshot_registry.get("sha256") != expected_registry_sha:
        raise ValueError(
            "Snapshot and ledger manifests bind different source registries"
        )
    return {
        "indicator_input_sha256": expected_input_sha,
        "ledger_manifest_sha256": sha256_file(ledger_manifest_path),
        "snapshot_manifest_sha256": expected_snapshot_sha,
        "source_registry_sha256": expected_registry_sha,
        "roster_sha256": expected_roster_sha,
        "population_sha256": expected_population_sha,
    }


def load_indicator_roster(path: str | Path) -> list[str]:
    """Load an ordered indicator roster from a JSON list or roster object."""

    roster_path = Path(path).expanduser().resolve()
    with roster_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    indicators = payload.get("indicators") if isinstance(payload, dict) else payload
    if not isinstance(indicators, list) or not indicators:
        raise ValueError("Indicator roster must contain a non-empty 'indicators' list")
    if not all(
        isinstance(indicator, str) and indicator.strip() for indicator in indicators
    ):
        raise ValueError("Every roster indicator must be a non-empty string")
    if len(set(indicators)) != len(indicators):
        raise ValueError("Indicator roster contains duplicate entries")
    return indicators


def load_population_dates(path: str | Path) -> tuple[str, list[str]]:
    """Load a frozen population ID and ordered ISO meeting-date roster."""

    population_path = Path(path).expanduser().resolve()
    with population_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("Population roster must be a JSON object")
    population_id = str(payload.get("population_id") or "").strip()
    raw_dates = payload.get("meeting_dates", payload.get("dates"))
    if not population_id or not isinstance(raw_dates, list) or not raw_dates:
        raise ValueError("Population roster requires population_id and meeting_dates")
    meeting_dates: list[str] = []
    for index, raw_date in enumerate(raw_dates):
        text = str(raw_date or "").strip()
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as error:
            raise ValueError(
                f"Population meeting date {index} is not ISO-8601: {text!r}"
            ) from error
        if parsed.date().isoformat() != text:
            raise ValueError(
                f"Population meeting date {index} must use YYYY-MM-DD: {text!r}"
            )
        meeting_dates.append(text)
    if meeting_dates != sorted(set(meeting_dates)):
        raise ValueError(
            "Population meeting dates must be unique and in ascending order"
        )
    return population_id, meeting_dates


def _parse_timestamp(value: Any, *, field: str, sample_id: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{sample_id}: {field} must be a non-empty ISO-8601 timestamp")
    candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as error:
        raise ValueError(
            f"{sample_id}: {field} is not a valid ISO-8601 timestamp: {text!r}"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{sample_id}: {field} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_date(value: Any, *, field: str, sample_id: str) -> date:
    text = str(value or "").strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError as error:
        raise ValueError(
            f"{sample_id}: {field} is not a valid YYYY-MM-DD date: {text!r}"
        ) from error
    if parsed.isoformat() != text:
        raise ValueError(f"{sample_id}: {field} must use canonical YYYY-MM-DD form")
    return parsed


def _require_non_empty_text(row: Mapping[str, Any], field: str, sample_id: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{sample_id}: {field} must be a non-empty string")
    return value.strip()


def _validate_source_payload(
    payload: Any,
    *,
    sample_id: str,
    information_as_of_date: date,
    requested_vintage_date: date,
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{sample_id}: source_payload must be a JSON object")
    series = payload.get("series")
    if not isinstance(series, list) or not series:
        raise ValueError(f"{sample_id}: source_payload.series must be a non-empty list")
    seen_source_keys: set[str] = set()
    for series_index, record in enumerate(series):
        label = f"{sample_id}: source_payload.series[{series_index}]"
        if not isinstance(record, Mapping):
            raise ValueError(f"{label} must be an object")
        source_key = _require_non_empty_text(record, "source_key", label)
        _require_non_empty_text(record, "series_id", label)
        if source_key in seen_source_keys:
            raise ValueError(f"{label} duplicates source_key {source_key!r}")
        seen_source_keys.add(source_key)
        record_vintage = _parse_date(
            record.get("requested_vintage_date"),
            field="requested_vintage_date",
            sample_id=label,
        )
        record_availability = _parse_date(
            record.get("availability_as_of_date"),
            field="availability_as_of_date",
            sample_id=label,
        )
        if record_vintage != requested_vintage_date:
            raise ValueError(f"{label}: requested_vintage_date differs from the row")
        if record_availability > information_as_of_date:
            raise ValueError(f"{label}: availability_as_of_date is after D-1")
        observations = record.get("observations")
        if not isinstance(observations, list) or not observations:
            raise ValueError(f"{label}.observations must be a non-empty list")
        previous_date: date | None = None
        for observation_index, observation in enumerate(observations):
            observation_label = f"{label}.observations[{observation_index}]"
            if not isinstance(observation, Mapping):
                raise ValueError(f"{observation_label} must be an object")
            observation_date = _parse_date(
                observation.get("date"),
                field="date",
                sample_id=observation_label,
            )
            if observation_date > information_as_of_date:
                raise ValueError(f"{observation_label}: observation date is after D-1")
            if previous_date is not None and observation_date <= previous_date:
                raise ValueError(
                    f"{label}.observations must be strictly date-ascending"
                )
            previous_date = observation_date
            value = observation.get("value")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{observation_label}.value must be a non-empty string"
                )


def build_indicator_analysis_prompt(row: Mapping[str, Any]) -> str:
    """Render the versioned upstream prompt from a validated normalized row."""

    return _PROMPT_TEMPLATE.format(
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        meeting_id=row["meeting_id"],
        meeting_timestamp=row["meeting_timestamp"],
        indicator=row["indicator"],
        information_as_of_date=row["information_as_of_date"],
        requested_vintage_date=row["requested_vintage_date"],
        availability_evidence_type=row["availability_evidence_type"],
        availability_as_of_date=row["availability_as_of_date"],
        observation_date=row["observation_date"],
        source_id=row["source_id"],
        source_payload=_canonical_json(row["source_payload"]),
    )


def validate_and_prepare_rows(
    rows: Iterable[Mapping[str, Any]],
    indicators: Sequence[str],
) -> list[dict[str, Any]]:
    """Validate temporal safety and complete roster coverage, then add prompts."""

    roster = list(indicators)
    if not roster or len(roster) != len(set(roster)):
        raise ValueError("indicators must be a non-empty unique ordered roster")
    roster_set = set(roster)
    roster_position = {indicator: index for index, indicator in enumerate(roster)}

    prepared: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    meeting_timestamps: dict[str, str] = {}
    meeting_indicators: dict[str, set[str]] = {}

    for row_number, original in enumerate(rows, 1):
        if not isinstance(original, Mapping):
            raise ValueError(f"Input row {row_number} must be an object")
        missing = [field for field in _REQUIRED_FIELDS if field not in original]
        if missing:
            raise ValueError(
                f"Input row {row_number} is missing required fields: {missing}"
            )

        sample_id = _require_non_empty_text(original, "sample_id", f"row {row_number}")
        if original.get("schema_version") != INPUT_SCHEMA_VERSION:
            raise ValueError(
                f"{sample_id}: canonical analysis requires "
                f"schema_version={INPUT_SCHEMA_VERSION!r}"
            )
        meeting_id = _require_non_empty_text(original, "meeting_id", sample_id)
        meeting_date_text = _require_non_empty_text(
            original,
            "meeting_date",
            sample_id,
        )
        indicator = _require_non_empty_text(original, "indicator", sample_id)
        source_id = _require_non_empty_text(original, "source_id", sample_id)
        source_interface = _require_non_empty_text(
            original,
            "source_interface",
            sample_id,
        )
        availability_evidence_type = _require_non_empty_text(
            original,
            "availability_evidence_type",
            sample_id,
        )
        if source_interface != "alfred-graph-csv-v1":
            raise ValueError(
                f"{sample_id}: unsupported source_interface {source_interface!r}"
            )
        if availability_evidence_type != "alfred_vintage_snapshot":
            raise ValueError(
                f"{sample_id}: unsupported availability evidence "
                f"{availability_evidence_type!r}"
            )
        expected_sample_id = f"{meeting_id}::{indicator}"
        if sample_id != expected_sample_id:
            raise ValueError(
                f"{sample_id}: stable sample_id must equal {expected_sample_id!r}"
            )
        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        if indicator not in roster_set:
            raise ValueError(f"{sample_id}: indicator is not in the frozen roster")
        pair = (meeting_id, indicator)
        if pair in seen_pairs:
            raise ValueError(f"Duplicate meeting-indicator pair: {pair}")

        meeting_timestamp = _parse_timestamp(
            original["meeting_timestamp"],
            field="meeting_timestamp",
            sample_id=sample_id,
        )
        source_timestamp = _parse_timestamp(
            original["source_timestamp"],
            field="source_timestamp",
            sample_id=sample_id,
        )
        parsed_meeting_date = _parse_date(
            meeting_date_text,
            field="meeting_date",
            sample_id=sample_id,
        )
        information_as_of_date = _parse_date(
            original["information_as_of_date"],
            field="information_as_of_date",
            sample_id=sample_id,
        )
        requested_vintage_date = _parse_date(
            original["requested_vintage_date"],
            field="requested_vintage_date",
            sample_id=sample_id,
        )
        availability_as_of_date = _parse_date(
            original["availability_as_of_date"],
            field="availability_as_of_date",
            sample_id=sample_id,
        )
        observation_date = _parse_date(
            original["observation_date"],
            field="observation_date",
            sample_id=sample_id,
        )
        expected_information_date = parsed_meeting_date - timedelta(days=1)
        if information_as_of_date != expected_information_date:
            raise ValueError(
                f"{sample_id}: information_as_of_date must equal meeting_date - 1 day"
            )
        if requested_vintage_date != information_as_of_date:
            raise ValueError(f"{sample_id}: requested_vintage_date must equal D-1")
        if availability_as_of_date > information_as_of_date:
            raise ValueError(f"{sample_id}: availability_as_of_date is after D-1")
        if observation_date > information_as_of_date:
            raise ValueError(f"{sample_id}: observation_date is after D-1")

        normalized_meeting_timestamp = _format_timestamp(meeting_timestamp)
        if meeting_id != normalized_meeting_timestamp[:10]:
            raise ValueError(
                f"{sample_id}: meeting_id must equal the meeting date "
                f"{normalized_meeting_timestamp[:10]!r}"
            )
        if meeting_id != parsed_meeting_date.isoformat():
            raise ValueError(f"{sample_id}: meeting_id and meeting_date must match")
        previous_timestamp = meeting_timestamps.setdefault(
            meeting_id, normalized_meeting_timestamp
        )
        if previous_timestamp != normalized_meeting_timestamp:
            raise ValueError(
                f"{meeting_id}: meeting_timestamp is inconsistent across indicators"
            )

        source_sha256 = validate_sha256(
            original["source_sha256"],
            label=f"{sample_id} source_sha256",
        )
        source_payload = original["source_payload"]
        if (
            source_payload is None
            or source_payload == ""
            or source_payload == []
            or source_payload == {}
        ):
            raise ValueError(f"{sample_id}: source_payload must not be empty")
        _validate_source_payload(
            source_payload,
            sample_id=sample_id,
            information_as_of_date=information_as_of_date,
            requested_vintage_date=requested_vintage_date,
        )
        payload_sha256 = sha256_text(_canonical_json(source_payload))

        normalized = {
            "schema_version": SCHEMA_VERSION,
            "meeting_id": meeting_id,
            "sample_id": sample_id,
            "meeting_timestamp": normalized_meeting_timestamp,
            "meeting_date": normalized_meeting_timestamp[:10],
            "indicator": indicator,
            "source_id": source_id,
            "source_sha256": source_sha256,
            "source_payload_sha256": payload_sha256,
            "source_timestamp": _format_timestamp(source_timestamp),
            "information_as_of_date": information_as_of_date.isoformat(),
            "requested_vintage_date": requested_vintage_date.isoformat(),
            "availability_as_of_date": availability_as_of_date.isoformat(),
            "availability_evidence_type": availability_evidence_type,
            "source_interface": source_interface,
            "observation_date": observation_date.isoformat(),
            "source_payload": source_payload,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        }
        prompt = build_indicator_analysis_prompt(normalized)
        normalized["prompt"] = prompt
        normalized["prompt_sha256"] = sha256_text(prompt)
        prepared.append(normalized)

        seen_sample_ids.add(sample_id)
        seen_pairs.add(pair)
        meeting_indicators.setdefault(meeting_id, set()).add(indicator)

    if not prepared:
        raise ValueError("No indicator-analysis rows were supplied")

    coverage_errors: list[str] = []
    for meeting_id, observed in sorted(meeting_indicators.items()):
        missing = sorted(roster_set - observed)
        extra = sorted(observed - roster_set)
        if missing or extra:
            coverage_errors.append(
                f"{meeting_id}: missing={missing or []}, extra={extra or []}"
            )
    if coverage_errors:
        raise ValueError(
            "Incomplete indicator roster coverage: " + "; ".join(coverage_errors)
        )

    return sorted(
        prepared,
        key=lambda row: (
            row["meeting_timestamp"],
            row["meeting_id"],
            roster_position[row["indicator"]],
        ),
    )


def generate_indicator_analysis_rows(
    prepared_rows: Sequence[Mapping[str, Any]],
    *,
    model_path: str | Path,
    tokenizer_path: str | Path | None = None,
    decoding: IndicatorAnalysisDecoding | None = None,
    generation_fn: Callable[..., list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Generate and strictly reconcile one output for every prepared input row."""

    config = decoding or IndicatorAnalysisDecoding()
    config.validate()
    if not prepared_rows:
        raise ValueError("prepared_rows must not be empty")

    require_runtime_metadata = generation_fn is None
    if generation_fn is None:
        from generate_new_response import generate_new_response

        generation_fn = generate_new_response

    prompt_rows = [dict(row) for row in prepared_rows]
    system_prompt = build_indicator_analysis_system_prompt(
        config.requested_max_output_tokens
    )
    generated_rows = generation_fn(
        prompt_rows,
        str(Path(model_path).expanduser().resolve()),
        batch_size=config.batch_size,
        seed=config.seed,
        replicate_id="0",
        temperature=config.temperature,
        top_p=config.top_p,
        max_new_tokens=config.max_new_tokens,
        max_model_len=config.max_model_len,
        tokenizer_path=(
            None
            if tokenizer_path is None
            else str(Path(tokenizer_path).expanduser().resolve())
        ),
        system_prompt=system_prompt,
        seed_policy="sample-id-sha256-v1",
        generation_metadata={
            "generation_stage": "canonical_indicator_analysis",
            "generation_schema_version": SCHEMA_VERSION,
            "generation_system_prompt_version": SYSTEM_PROMPT_VERSION,
            "generation_system_prompt_sha256": sha256_text(system_prompt),
            "generation_requested_max_output_tokens": (
                config.requested_max_output_tokens
            ),
        },
    )
    if not isinstance(generated_rows, list):
        raise ValueError("Generation helper must return a list of output rows")

    generated_by_id: dict[str, dict[str, Any]] = {}
    token_limit_errors: list[dict[str, Any]] = []
    for row in generated_rows:
        if not isinstance(row, Mapping):
            raise ValueError("Generation helper returned a non-object row")
        sample_id = str(row.get("sample_id") or "")
        generated = row.get("generated")
        if sample_id in generated_by_id:
            raise ValueError(
                f"Generation helper returned duplicate sample_id: {sample_id}"
            )
        if not isinstance(generated, str) or not generated.strip():
            raise ValueError(
                f"Generation failed or returned empty text for {sample_id!r}"
            )
        generation_record = dict(row)
        if require_runtime_metadata:
            preflight_token_count = row.get("prompt_preflight_token_count")
            completion_arguments = {
                "input_token_count": preflight_token_count,
                "output_token_count": row.get("output_token_count"),
                "max_new_tokens": config.max_new_tokens,
                "context_limit": config.max_model_len,
                "finish_reason": str(row.get("generation_finish_reason") or ""),
                "input_was_truncated": row.get("input_was_truncated"),
                "consumed_input_token_count": row.get("prompt_token_count"),
            }
            try:
                validate_generation_completion(**completion_arguments)
            except GenerationSafetyError as error:
                finish_reason = (
                    str(completion_arguments["finish_reason"]).strip().lower()
                )
                if finish_reason not in {"length", "max_length", "max_tokens"}:
                    raise
                # Revalidate all context and token accounting with a normal
                # finish reason. This ensures the tolerance applies only to
                # an otherwise valid token-limit completion.
                validate_generation_completion(
                    **{
                        **completion_arguments,
                        "finish_reason": "stop",
                    }
                )
                error_record = {
                    "sample_id": sample_id,
                    "error_type": "token_limit_finish",
                    "message": str(error),
                    "finish_reason": str(completion_arguments["finish_reason"]),
                    "input_token_count": preflight_token_count,
                    "output_token_count": row.get("output_token_count"),
                    "max_new_tokens": config.max_new_tokens,
                }
                token_limit_errors.append(error_record)
                generation_record["generation_validation_status"] = (
                    "accepted_token_limit_error"
                )
                generation_record["generation_validation_error"] = error_record
            else:
                generation_record["generation_validation_status"] = "passed"
        generated_by_id[sample_id] = generation_record

    if len(token_limit_errors) > config.max_token_limit_errors:
        sample_ids = [record["sample_id"] for record in token_limit_errors]
        raise GenerationSafetyError(
            "Indicator analysis produced "
            f"{len(token_limit_errors)} token-limit completion errors, "
            f"exceeding the allowed maximum of "
            f"{config.max_token_limit_errors}; sample_ids={sample_ids}"
        )

    expected_ids = [str(row["sample_id"]) for row in prepared_rows]
    expected_set = set(expected_ids)
    actual_set = set(generated_by_id)
    if actual_set != expected_set:
        raise ValueError(
            "Generation output inventory mismatch: "
            f"missing={sorted(expected_set - actual_set)}, "
            f"unexpected={sorted(actual_set - expected_set)}"
        )

    results: list[dict[str, Any]] = []
    for row in prepared_rows:
        generation_record = generated_by_id[str(row["sample_id"])]
        generated = str(generation_record["generated"])
        result = dict(row)
        for key, value in generation_record.items():
            if key.startswith(("generation_", "decoding_")) or key in {
                "generated",
                "generated_sha256",
                "replicate_id",
                "max_new_tokens",
                "max_model_len",
                "prompt_token_count",
                "prompt_preflight_token_count",
                "output_token_count",
                "input_was_truncated",
            }:
                result[key] = value
        result["generated"] = generated
        result["generated_sha256"] = sha256_text(generated)
        result.setdefault("replicate_id", "0")
        result.setdefault("generation_seed", config.seed)
        result.setdefault(
            "generation_seed_policy",
            "sample-id-sha256-v1",
        )
        result.setdefault("decoding_temperature", config.temperature)
        result.setdefault("decoding_top_p", config.top_p)
        result.setdefault("max_new_tokens", config.max_new_tokens)
        result.setdefault("max_model_len", config.max_model_len)
        results.append(result)
    return results


def _same_fingerprint(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    return (
        before.get("kind"),
        before.get("sha256"),
        before.get("file_count"),
        before.get("total_bytes"),
    ) == (
        after.get("kind"),
        after.get("sha256"),
        after.get("file_count"),
        after.get("total_bytes"),
    )


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def run_indicator_analysis_generation(
    *,
    input_jsonl: str | Path,
    roster_json: str | Path,
    model_path: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    population_json: str | Path | None = None,
    ledger_manifest: str | Path | None = None,
    snapshot_manifest: str | Path | None = None,
    source_registry: str | Path | None = None,
    decoding: IndicatorAnalysisDecoding | None = None,
    generation_fn: Callable[..., list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Run the immutable generation stage and return its manifest."""

    config = decoding or IndicatorAnalysisDecoding()
    config.validate()
    input_path = Path(input_jsonl).expanduser().resolve()
    roster_path = Path(roster_json).expanduser().resolve()
    resolved_model_path = Path(model_path).expanduser().resolve()
    resolved_tokenizer_path = Path(tokenizer_path).expanduser().resolve()
    population_path = (
        None
        if population_json is None
        else Path(population_json).expanduser().resolve()
    )
    provenance_values = {
        "ledger_manifest": ledger_manifest,
        "snapshot_manifest": snapshot_manifest,
        "source_registry": source_registry,
    }
    supplied_provenance = {
        label: value for label, value in provenance_values.items() if value is not None
    }
    if supplied_provenance and len(supplied_provenance) != len(provenance_values):
        raise ValueError(
            "ledger_manifest, snapshot_manifest, and source_registry must be "
            "supplied together"
        )
    if generation_fn is None and not supplied_provenance:
        raise ValueError(
            "Canonical runtime requires ledger, snapshot, and registry provenance"
        )
    provenance_paths = {
        label: Path(value).expanduser().resolve()
        for label, value in supplied_provenance.items()
    }
    destination = Path(output_dir).expanduser().resolve()
    output_path = destination / "indicator_analysis.jsonl"
    manifest_path = destination / "analysis_manifest.json"
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Canonical generation artifacts are immutable; choose a new output_dir"
        )

    fingerprints_before = {
        "input_jsonl": fingerprint_artifact_path(input_path),
        "roster": fingerprint_artifact_path(roster_path),
        "model": fingerprint_artifact_path(resolved_model_path),
        "tokenizer": fingerprint_artifact_path(resolved_tokenizer_path),
    }
    if population_path is not None:
        fingerprints_before["population"] = fingerprint_artifact_path(population_path)
    for label, path in provenance_paths.items():
        fingerprints_before[label] = fingerprint_artifact_path(path)
    provenance_binding: dict[str, str] | None = None
    if provenance_paths:
        if population_path is None:
            raise ValueError("Canonical ledger provenance requires a frozen population")
        provenance_binding = validate_ledger_provenance(
            input_path=input_path,
            ledger_manifest_path=provenance_paths["ledger_manifest"],
            snapshot_manifest_path=provenance_paths["snapshot_manifest"],
            source_registry_path=provenance_paths["source_registry"],
            roster_path=roster_path,
            population_path=population_path,
        )
    roster = load_indicator_roster(roster_path)
    prepared_rows = validate_and_prepare_rows(_read_jsonl(input_path), roster)
    population_id: str | None = None
    population_dates: list[str] | None = None
    if population_path is not None:
        population_id, population_dates = load_population_dates(population_path)
        observed_dates = sorted({str(row["meeting_date"]) for row in prepared_rows})
        if observed_dates != population_dates:
            raise ValueError(
                "Indicator-analysis meetings do not match the frozen "
                f"population: missing={sorted(set(population_dates) - set(observed_dates))}, "
                f"extra={sorted(set(observed_dates) - set(population_dates))}"
            )
    results = generate_indicator_analysis_rows(
        prepared_rows,
        model_path=resolved_model_path,
        tokenizer_path=resolved_tokenizer_path,
        decoding=config,
        generation_fn=generation_fn,
    )

    fingerprints_after = {
        "input_jsonl": fingerprint_artifact_path(input_path),
        "roster": fingerprint_artifact_path(roster_path),
        "model": fingerprint_artifact_path(resolved_model_path),
        "tokenizer": fingerprint_artifact_path(resolved_tokenizer_path),
    }
    if population_path is not None:
        fingerprints_after["population"] = fingerprint_artifact_path(population_path)
    for label, path in provenance_paths.items():
        fingerprints_after[label] = fingerprint_artifact_path(path)
    changed = [
        label
        for label in fingerprints_before
        if not _same_fingerprint(
            fingerprints_before[label],
            fingerprints_after[label],
        )
    ]
    if changed:
        raise RuntimeError(
            "Generation inputs/checkpoints changed during the run: "
            + ", ".join(changed)
        )

    run_binding = {
        "schema_version": SCHEMA_VERSION,
        "input_sha256": fingerprints_before["input_jsonl"]["sha256"],
        "roster_sha256": fingerprints_before["roster"]["sha256"],
        "model_sha256": fingerprints_before["model"]["sha256"],
        "tokenizer_sha256": fingerprints_before["tokenizer"]["sha256"],
        "population_id": population_id,
        "population_dates": population_dates,
        "ledger_provenance": provenance_binding,
        "prompt_template_sha256": sha256_text(_PROMPT_TEMPLATE),
        "system_prompt_sha256": sha256_text(
            build_indicator_analysis_system_prompt(config.requested_max_output_tokens)
        ),
        "decoding": asdict(config),
    }
    run_id = f"indicator-analysis-{sha256_text(_canonical_json(run_binding))[:16]}"

    bound_results: list[dict[str, Any]] = []
    for row in results:
        bound = dict(row)
        bound.update(
            {
                "run_id": run_id,
                "generation_model_sha256": fingerprints_before["model"]["sha256"],
                "generation_tokenizer_sha256": fingerprints_before["tokenizer"][
                    "sha256"
                ],
            }
        )
        bound_results.append(bound)

    _write_jsonl_exclusive(output_path, bound_results)
    output_fingerprint = {
        "path": str(output_path),
        "kind": "file",
        "sha256": sha256_file(output_path),
        "file_count": 1,
        "total_bytes": output_path.stat().st_size,
        "algorithm": "sha256(file_bytes)",
    }
    meeting_ids = sorted({str(row["meeting_id"]) for row in bound_results})
    source_inventory = [
        {
            "sample_id": row["sample_id"],
            "source_id": row["source_id"],
            "source_sha256": row["source_sha256"],
            "source_payload_sha256": row["source_payload_sha256"],
            "source_timestamp": row["source_timestamp"],
            "information_as_of_date": row["information_as_of_date"],
            "requested_vintage_date": row["requested_vintage_date"],
            "availability_as_of_date": row["availability_as_of_date"],
            "availability_evidence_type": row["availability_evidence_type"],
            "source_interface": row["source_interface"],
            "observation_date": row["observation_date"],
        }
        for row in bound_results
    ]
    artifact_inventory = [
        {
            "sample_id": row["sample_id"],
            "meeting_id": row["meeting_id"],
            "indicator": row["indicator"],
            "prompt_sha256": row["prompt_sha256"],
            "generated_sha256": row["generated_sha256"],
            "generation_validation_status": row.get(
                "generation_validation_status",
                "not_evaluated",
            ),
        }
        for row in bound_results
    ]
    completion_errors = [
        dict(row["generation_validation_error"])
        for row in bound_results
        if row.get("generation_validation_status") == "accepted_token_limit_error"
    ]
    passed_completion_count = sum(
        row.get("generation_validation_status") == "passed" for row in bound_results
    )
    not_evaluated_count = sum(
        "generation_validation_status" not in row for row in bound_results
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "run_id": run_id,
        "generation_only": True,
        "training_performed": False,
        "execution_environment": {
            "physical_gpu_index": os.environ.get(
                "LOO_CANONICAL_PHYSICAL_GPU_INDEX"
            ),
            "physical_gpu_uuid": os.environ.get(
                "LOO_CANONICAL_PHYSICAL_GPU_UUID"
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "declared_visible_device_count": os.environ.get(
                "LOO_CANONICAL_VISIBLE_DEVICE_COUNT"
            ),
            "tensor_parallel_size": 1,
        },
        "prompt_template": {
            "version": PROMPT_TEMPLATE_VERSION,
            "template_sha256": sha256_text(_PROMPT_TEMPLATE),
        },
        "system_prompt": {
            "version": SYSTEM_PROMPT_VERSION,
            "sha256": sha256_text(
                build_indicator_analysis_system_prompt(
                    config.requested_max_output_tokens
                )
            ),
            "requested_max_output_tokens": (config.requested_max_output_tokens),
            "hard_max_new_tokens": config.max_new_tokens,
        },
        "decoding": asdict(config),
        "inputs": fingerprints_before,
        "ledger_provenance": provenance_binding,
        "completion_validation": {
            "policy": "bounded-token-limit-errors-v1",
            "max_token_limit_errors": config.max_token_limit_errors,
            "observed_token_limit_error_count": len(completion_errors),
            "passed_count": passed_completion_count,
            "not_evaluated_count": not_evaluated_count,
            "errors": completion_errors,
        },
        "inventory": {
            "row_count": len(bound_results),
            "meeting_count": len(meeting_ids),
            "indicator_count": len(roster),
            "meeting_ids": meeting_ids,
            "indicator_roster": roster,
            "population_id": population_id,
            "population_dates": population_dates,
            "source_inventory_sha256": sha256_text(_canonical_json(source_inventory)),
            "rows": artifact_inventory,
        },
        "output": output_fingerprint,
    }
    _write_json_exclusive(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate canonical meeting-time indicator analyses without "
            "training or modifying a model."
        )
    )
    parser.add_argument("--input", required=True, help="Meeting×indicator JSONL")
    parser.add_argument("--roster", required=True, help="Ordered roster JSON")
    parser.add_argument(
        "--population",
        required=True,
        help="Frozen population JSON with population_id and meeting_dates",
    )
    parser.add_argument(
        "--ledger-manifest",
        required=True,
        help="Sealed D-1 ledger manifest that binds --input",
    )
    parser.add_argument(
        "--snapshot-manifest",
        required=True,
        help="Sealed keyless ALFRED snapshot manifest",
    )
    parser.add_argument(
        "--source-registry",
        required=True,
        help="Frozen indicator source registry",
    )
    parser.add_argument("--model", required=True, help="Frozen generation model")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Frozen tokenizer directory (may equal --model)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New immutable run directory",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=(
            "Technical generation stop limit. The system prompt requests a "
            "smaller response limit to retain overflow tolerance."
        ),
    )
    parser.add_argument(
        "--requested-max-output-tokens",
        type=int,
        default=DEFAULT_REQUESTED_MAX_OUTPUT_TOKENS,
        help=("Maximum response length requested in the analysis system prompt."),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
    )
    parser.add_argument(
        "--max-token-limit-errors",
        type=int,
        default=DEFAULT_MAX_TOKEN_LIMIT_ERRORS,
        help=(
            "Allow at most this many otherwise-valid analysis rows whose "
            "finish reason is a token limit; maximum supported value is 2."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = run_indicator_analysis_generation(
        input_jsonl=args.input,
        roster_json=args.roster,
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        population_json=args.population,
        ledger_manifest=args.ledger_manifest,
        snapshot_manifest=args.snapshot_manifest,
        source_registry=args.source_registry,
        decoding=IndicatorAnalysisDecoding(
            seed=args.seed,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            requested_max_output_tokens=(args.requested_max_output_tokens),
            max_model_len=args.max_model_len,
            max_token_limit_errors=args.max_token_limit_errors,
        ),
    )
    print(
        f"Generated {manifest['inventory']['row_count']} immutable indicator "
        f"analyses in run {manifest['run_id']}"
    )


if __name__ == "__main__":
    main()
