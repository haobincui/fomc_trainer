"""Build the clean common-sample dataset for the Chapter 2 checkpoint audit.

The builder consumes the already validated D-1 indicator ledger.  It performs
no model inference: every evidence sentence is rendered mechanically from the
latest, previous, and year-comparable observations.  Official Minutes are
downloaded into a separate reference artifact and never interpolated into the
generation prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from open_r1.provenance import sha256_file, sha256_text
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)
from open_r1.validator.text_leakage import (
    validate_no_prompt_reference_token_overlap,
)


SCHEMA_VERSION = "checkpoint-eval-test-manifest-v1"
PROMPT_SCHEMA_VERSION = "checkpoint-eval-prompt-v1"
REFERENCE_SCHEMA_VERSION = "checkpoint-eval-reference-v1"
DEFAULT_USER_AGENT = "fomc-trainer-checkpoint-evaluation/1"
FED_MINUTES_URL = (
    "https://www.federalreserve.gov/monetarypolicy/"
    "fomcminutes{meeting_date_compact}.htm"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label} JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid {label} JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"{label} row must be an object at {path}:{line_number}"
                )
            rows.append(value)
    if not rows:
        raise ValueError(f"{label} is empty: {path}")
    return rows


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Immutable evaluation artifact exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    content = "".join(f"{_canonical_json(row)}\n" for row in rows)
    _atomic_write(path, content)


def _normalise_space(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _canonical_decimal(value: object, *, label: str) -> Decimal:
    text = str(value or "").strip()
    try:
        parsed = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"{label} is not a finite decimal: {value!r}") from exc
    if not parsed.is_finite():
        raise ValueError(f"{label} is not finite: {value!r}")
    return parsed


def _display_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _change_unit(units: str) -> str:
    return "percentage points" if "percent" in units.lower() else units


def _year_comparable(
    observations: list[dict[str, str]],
    *,
    latest_date: date,
) -> dict[str, str] | None:
    threshold = latest_date.replace(year=latest_date.year - 1)
    eligible = [
        observation
        for observation in observations[:-1]
        if date.fromisoformat(observation["date"]) <= threshold
    ]
    return eligible[-1] if eligible else None


def _series_evidence(
    series: dict[str, Any],
    *,
    cutoff: str,
    indicator: str,
) -> tuple[str, list[dict[str, Any]]]:
    series_id = _normalise_space(series.get("series_id"))
    title = _normalise_space(series.get("title"))
    units = _normalise_space(series.get("units"))
    observations_raw = series.get("observations")
    if not series_id or not title or not units:
        raise ValueError(f"{indicator}: series metadata is incomplete")
    if not isinstance(observations_raw, list) or not observations_raw:
        raise ValueError(f"{indicator}/{series_id}: observations are empty")

    observations: list[dict[str, str]] = []
    previous_date: str | None = None
    for index, observation in enumerate(observations_raw):
        if not isinstance(observation, dict):
            raise ValueError(f"{indicator}/{series_id}: invalid observation")
        observation_date = str(observation.get("date") or "").strip()
        try:
            date.fromisoformat(observation_date)
        except ValueError as exc:
            raise ValueError(
                f"{indicator}/{series_id}: invalid observation date"
            ) from exc
        if observation_date > cutoff:
            raise ValueError(
                f"{indicator}/{series_id}: {observation_date} exceeds {cutoff}"
            )
        if previous_date is not None and observation_date <= previous_date:
            raise ValueError(
                f"{indicator}/{series_id}: observations are not ascending"
            )
        value = _canonical_decimal(
            observation.get("value"),
            label=f"{indicator}/{series_id} observation {index}",
        )
        observations.append(
            {"date": observation_date, "value": _display_decimal(value)}
        )
        previous_date = observation_date

    latest = observations[-1]
    previous = observations[-2] if len(observations) > 1 else None
    comparable = _year_comparable(
        observations,
        latest_date=date.fromisoformat(latest["date"]),
    )
    latest_value = Decimal(latest["value"])
    clauses = [
        f"{series_id} ({title}; {units})",
        f"latest {latest['date']} = {latest['value']}",
    ]
    facts: list[dict[str, Any]] = [
        {
            "indicator": indicator,
            "series_id": series_id,
            "kind": "observed_value",
            "date": latest["date"],
            "value": latest["value"],
            "unit": units,
        }
    ]

    if previous is not None:
        previous_value = Decimal(previous["value"])
        absolute_change = latest_value - previous_value
        clauses.extend(
            [
                f"previous {previous['date']} = {previous['value']}",
                (
                    "latest-minus-previous = "
                    f"{_display_decimal(absolute_change)} "
                    f"{_change_unit(units)}"
                ),
            ]
        )
        facts.extend(
            [
                {
                    "indicator": indicator,
                    "series_id": series_id,
                    "kind": "observed_value",
                    "date": previous["date"],
                    "value": previous["value"],
                    "unit": units,
                },
                {
                    "indicator": indicator,
                    "series_id": series_id,
                    "kind": "derived_absolute_change",
                    "from_date": previous["date"],
                    "to_date": latest["date"],
                    "value": _display_decimal(absolute_change),
                    "unit": _change_unit(units),
                },
            ]
        )
        if previous_value != 0:
            percentage_change = (
                (latest_value - previous_value) / abs(previous_value)
            ) * Decimal("100")
            clauses.append(
                "latest percent change = "
                f"{_display_decimal(percentage_change)} percent"
            )
            facts.append(
                {
                    "indicator": indicator,
                    "series_id": series_id,
                    "kind": "derived_percent_change",
                    "from_date": previous["date"],
                    "to_date": latest["date"],
                    "value": _display_decimal(percentage_change),
                    "unit": "Percent",
                }
            )

    if comparable is not None:
        comparable_value = Decimal(comparable["value"])
        year_change = latest_value - comparable_value
        clauses.extend(
            [
                (
                    f"year-comparable {comparable['date']} = "
                    f"{comparable['value']}"
                ),
                (
                    "latest-minus-year-comparable = "
                    f"{_display_decimal(year_change)} {_change_unit(units)}"
                ),
            ]
        )
        facts.extend(
            [
                {
                    "indicator": indicator,
                    "series_id": series_id,
                    "kind": "observed_value",
                    "date": comparable["date"],
                    "value": comparable["value"],
                    "unit": units,
                },
                {
                    "indicator": indicator,
                    "series_id": series_id,
                    "kind": "derived_year_absolute_change",
                    "from_date": comparable["date"],
                    "to_date": latest["date"],
                    "value": _display_decimal(year_change),
                    "unit": _change_unit(units),
                },
            ]
        )
        if comparable_value != 0:
            year_percentage_change = (
                (latest_value - comparable_value) / abs(comparable_value)
            ) * Decimal("100")
            clauses.append(
                "year-comparable percent change = "
                f"{_display_decimal(year_percentage_change)} percent"
            )
            facts.append(
                {
                    "indicator": indicator,
                    "series_id": series_id,
                    "kind": "derived_year_percent_change",
                    "from_date": comparable["date"],
                    "to_date": latest["date"],
                    "value": _display_decimal(year_percentage_change),
                    "unit": "Percent",
                }
            )

    return "; ".join(clauses) + ".", facts


def render_meeting_evidence(
    rows: list[dict[str, Any]],
    *,
    meeting_date: str,
) -> tuple[str, list[dict[str, Any]], str]:
    expected_cutoff = (
        date.fromisoformat(meeting_date).fromordinal(
            date.fromisoformat(meeting_date).toordinal() - 1
        )
    ).isoformat()
    indicators: list[str] = []
    rendered: list[str] = []
    facts: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("meeting_date") or "") != meeting_date:
            raise ValueError("Meeting evidence rows contain another meeting")
        cutoff = str(row.get("information_as_of_date") or "")
        if cutoff != expected_cutoff:
            raise ValueError(
                f"{meeting_date}: expected D-1 cutoff {expected_cutoff}, got {cutoff}"
            )
        indicator = _normalise_space(row.get("indicator"))
        if not indicator or indicator in indicators:
            raise ValueError(f"{meeting_date}: duplicate/empty indicator")
        source_payload = row.get("source_payload")
        series_rows = (
            source_payload.get("series")
            if isinstance(source_payload, dict)
            else None
        )
        if not isinstance(series_rows, list) or not series_rows:
            raise ValueError(f"{meeting_date}/{indicator}: missing source series")
        indicators.append(indicator)
        rendered.append(f"### {indicator}")
        for series in series_rows:
            if not isinstance(series, dict):
                raise ValueError(f"{meeting_date}/{indicator}: invalid series")
            line, series_facts = _series_evidence(
                series,
                cutoff=cutoff,
                indicator=indicator,
            )
            rendered.append(f"- {line}")
            facts.extend(series_facts)
    return "\n".join(rendered), facts, expected_cutoff


def _section_heading_element(
    soup: BeautifulSoup,
    section_name: str,
):
    expected = _normalise_space(section_name)
    for element in soup.find_all(["strong", "b"]):
        if _normalise_space(element.get_text(" ", strip=True)) == expected:
            return element
    raise ValueError(f"Official Minutes HTML lacks section {section_name!r}")


def extract_minutes_section(html: str, section_name: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    heading = _section_heading_element(soup, section_name)
    paragraph = heading.find_parent("p")
    if paragraph is None:
        raise ValueError(f"Section heading {section_name!r} is not in a paragraph")

    paragraphs: list[str] = []
    first = _normalise_space(paragraph.get_text(" ", strip=True))
    heading_text = _normalise_space(heading.get_text(" ", strip=True))
    if first.startswith(heading_text):
        first = first[len(heading_text) :].strip(" :-\n")
    if first:
        paragraphs.append(first)

    for sibling in paragraph.find_next_siblings():
        if getattr(sibling, "name", None) != "p":
            continue
        if sibling.find(["strong", "b"]) is not None:
            break
        text = _normalise_space(sibling.get_text(" ", strip=True))
        if text:
            paragraphs.append(text)
    if not paragraphs:
        raise ValueError(f"Section {section_name!r} has no body paragraphs")
    return "\n\n".join(paragraphs)


def extract_last_update(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find(id="lastUpdate")
    text = _normalise_space(
        element.get_text(" ", strip=True) if element is not None else ""
    )
    match = re.search(
        r"([A-Z][a-z]+)\s+([0-9]{1,2}),\s+([0-9]{4})",
        text,
    )
    if not match:
        raise ValueError("Official Minutes HTML lacks a parseable Last Update")
    return datetime.strptime(match.group(0), "%B %d, %Y").date().isoformat()


def _download_minutes_html(
    meeting_date: str,
    *,
    raw_dir: Path,
    offline: bool,
) -> tuple[str, str, Path]:
    compact = meeting_date.replace("-", "")
    url = FED_MINUTES_URL.format(meeting_date_compact=compact)
    destination = raw_dir / f"fomcminutes{compact}.html"
    if destination.is_file():
        return destination.read_text(encoding="utf-8"), url, destination
    if offline:
        raise FileNotFoundError(
            f"Offline mode requires cached official Minutes HTML: {destination}"
        )
    request = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    with urlopen(request, timeout=60) as response:
        body = response.read()
        status = getattr(response, "status", 200)
        final_url = str(response.geturl())
    if status != 200 or final_url != url:
        raise ValueError(
            f"Unexpected official Minutes response for {meeting_date}: "
            f"status={status}, url={final_url!r}"
        )
    html = body.decode("utf-8")
    if meeting_date not in html and compact not in html:
        raise ValueError(
            f"Official Minutes response does not identify {meeting_date}"
        )
    _atomic_write(destination, html)
    return html, url, destination


def _prompt(
    *,
    meeting_date: str,
    cutoff: str,
    section_name: str,
    instruction: str,
    evidence: str,
) -> str:
    return (
        f"You are drafting one section of historical FOMC Minutes for the "
        f"meeting on {meeting_date}.\n"
        f"Required section: {section_name}\n"
        f"Task: {instruction}\n"
        f"Information cutoff: {cutoff} (meeting date minus one calendar day).\n\n"
        "Use only the mechanically rendered evidence packet below. Do not "
        "invent values, dates, policy decisions, or participant views. "
        "Do not quote this instruction and do not add another Minutes section.\n\n"
        "<<<BEGIN-D1-EVIDENCE>>>\n"
        f"{evidence}\n"
        "<<<END-D1-EVIDENCE>>>"
    )


def _validate_existing_dataset(
    *,
    manifest_path: Path,
    config_path: Path,
    indicator_path: Path,
    ledger_manifest_path: Path,
    destination: Path,
) -> dict[str, Any]:
    """Verify and reuse an already committed common-test dataset."""

    manifest = _read_json_object(manifest_path, label="test manifest")
    validate_manifest_integrity(manifest)
    config = _read_json_object(config_path, label="checkpoint evaluation config")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Existing test manifest uses another schema")
    if manifest.get("evaluation_id") != config.get("evaluation_id"):
        raise ValueError("Existing test manifest uses another evaluation config")

    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Existing test manifest lacks input bindings")
    for label, path in (
        ("config", config_path),
        ("indicator_inputs", indicator_path),
        ("ledger_manifest", ledger_manifest_path),
    ):
        record = inputs.get(label)
        if not isinstance(record, dict) or record.get("sha256") != sha256_file(path):
            raise ValueError(f"Existing test manifest input changed: {label}")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Existing test manifest lacks output bindings")
    prompt_path = destination / "prompts.jsonl"
    reference_path = destination / "references.jsonl"
    prompts = _read_jsonl(prompt_path, label="evaluation prompts")
    references = _read_jsonl(reference_path, label="evaluation references")
    for label, path, rows in (
        ("prompts", prompt_path, prompts),
        ("references", reference_path, references),
    ):
        record = outputs.get(label)
        if (
            not isinstance(record, dict)
            or Path(str(record.get("path") or "")).resolve() != path
            or record.get("sha256") != sha256_file(path)
            or record.get("row_count") != len(rows)
        ):
            raise ValueError(f"Existing test manifest output changed: {label}")

    population = config.get("population")
    sections = config.get("sections")
    if not isinstance(population, dict) or not isinstance(sections, list):
        raise ValueError("Evaluation config requires population and sections")
    expected_ids = [
        f"{meeting_date}::{section['section_id']}"
        for meeting_date in population.get("meeting_dates", [])
        for section in sections
        if isinstance(section, dict) and section.get("section_id")
    ]
    prompt_ids = [str(row.get("sample_id") or "") for row in prompts]
    reference_ids = [str(row.get("sample_id") or "") for row in references]
    if (
        prompt_ids != expected_ids
        or reference_ids != expected_ids
        or manifest.get("sample_count") != len(expected_ids)
    ):
        raise ValueError("Existing common-test sample matrix changed")
    for prompt in prompts:
        text = str(prompt.get("prompt") or "")
        if (
            prompt.get("prompt_sha256") != sha256_text(text)
            or prompt.get("reference_used_in_prompt") is not False
            or "reference" in prompt
        ):
            raise ValueError(
                f"{prompt.get('sample_id')}: prompt provenance is invalid"
            )
    for reference in references:
        text = str(reference.get("reference") or "")
        if reference.get("reference_sha256") != sha256_text(text):
            raise ValueError(
                f"{reference.get('sample_id')}: reference hash changed"
            )
    for prompt, reference in zip(prompts, references, strict=True):
        sample_id = str(prompt.get("sample_id") or "")
        prompt_text = str(prompt.get("prompt") or "")
        reference_text = str(reference.get("reference") or "")
        if reference_text in prompt_text:
            raise ValueError(f"Reference was copied into prompt for {sample_id}")
        validate_no_prompt_reference_token_overlap(
            prompt_text,
            reference_text,
            sample_id=sample_id,
        )
    for source in manifest.get("reference_sources", []):
        if not isinstance(source, dict):
            raise ValueError("Existing reference source record is invalid")
        raw_path = Path(str(source.get("raw_html_path") or "")).resolve()
        if (
            not raw_path.is_file()
            or source.get("raw_html_sha256") != sha256_file(raw_path)
        ):
            raise ValueError("Existing official Minutes HTML changed")
    return manifest


def build_dataset(
    *,
    config_file: str | Path,
    indicator_inputs_file: str | Path,
    ledger_manifest_file: str | Path,
    output_dir: str | Path,
    offline: bool = False,
) -> tuple[Path, dict[str, Any]]:
    config_path = Path(config_file).expanduser().resolve()
    indicator_path = Path(indicator_inputs_file).expanduser().resolve()
    ledger_manifest_path = Path(ledger_manifest_file).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    manifest_path = destination / "test_manifest.json"
    if manifest_path.exists():
        return manifest_path, _validate_existing_dataset(
            manifest_path=manifest_path,
            config_path=config_path,
            indicator_path=indicator_path,
            ledger_manifest_path=ledger_manifest_path,
            destination=destination,
        )
    for incomplete in (
        destination / "prompts.jsonl",
        destination / "references.jsonl",
    ):
        if incomplete.exists():
            raise FileExistsError(
                "Evaluation output exists without the commit manifest; "
                f"use a new versioned output directory: {incomplete}"
            )
    config = _read_json_object(config_path, label="checkpoint evaluation config")
    ledger_manifest = _read_json_object(
        ledger_manifest_path,
        label="D-1 ledger manifest",
    )
    validate_manifest_integrity(ledger_manifest)
    ledger_rows = _read_jsonl(indicator_path, label="D-1 indicator inputs")

    population = config.get("population")
    sections = config.get("sections")
    if not isinstance(population, dict) or not isinstance(sections, list):
        raise ValueError("Evaluation config requires population and sections")
    if ledger_manifest.get("population_id") != population.get("population_id"):
        raise ValueError(
            "D-1 ledger population differs from evaluation population"
        )
    ledger_output = (
        ledger_manifest.get("outputs", {}).get("indicator_inputs")
        if isinstance(ledger_manifest.get("outputs"), dict)
        else None
    )
    if not isinstance(ledger_output, dict):
        raise ValueError("D-1 ledger manifest does not bind indicator_inputs")
    if ledger_output.get("sha256") != sha256_file(indicator_path):
        raise ValueError("D-1 indicator_inputs hash differs from ledger manifest")
    if ledger_output.get("row_count") != len(ledger_rows):
        raise ValueError("D-1 indicator_inputs row count differs from manifest")
    meeting_dates = population.get("meeting_dates")
    if not isinstance(meeting_dates, list) or meeting_dates != sorted(
        set(meeting_dates)
    ):
        raise ValueError("population.meeting_dates must be unique and ascending")
    if not sections:
        raise ValueError("Evaluation config sections are empty")

    rows_by_meeting: dict[str, list[dict[str, Any]]] = {
        str(meeting): [] for meeting in meeting_dates
    }
    for row in ledger_rows:
        meeting = str(row.get("meeting_date") or "")
        if meeting not in rows_by_meeting:
            raise ValueError(f"Indicator ledger contains out-of-population {meeting}")
        rows_by_meeting[meeting].append(row)
    indicator_counts = {len(rows) for rows in rows_by_meeting.values()}
    if len(indicator_counts) != 1 or 0 in indicator_counts:
        raise ValueError(
            f"Every meeting must have the same non-zero indicator count: "
            f"{sorted(indicator_counts)}"
        )

    prompts: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    reference_sources: list[dict[str, Any]] = []
    raw_dir = destination / "official_minutes_html"
    for meeting_date in meeting_dates:
        meeting_evidence, evidence_facts, cutoff = render_meeting_evidence(
            rows_by_meeting[meeting_date],
            meeting_date=meeting_date,
        )
        html, source_url, raw_path = _download_minutes_html(
            meeting_date,
            raw_dir=raw_dir,
            offline=offline,
        )
        release_date = extract_last_update(html)
        if release_date <= meeting_date:
            raise ValueError(
                f"Minutes release date must follow meeting date: {meeting_date}"
            )
        reference_sources.append(
            {
                "meeting_date": meeting_date,
                "source_url": source_url,
                "release_date": release_date,
                "raw_html_path": str(raw_path),
                "raw_html_sha256": sha256_file(raw_path),
            }
        )
        for section in sections:
            if not isinstance(section, dict):
                raise ValueError("Each section config must be an object")
            section_id = _normalise_space(section.get("section_id"))
            section_name = _normalise_space(section.get("section_name"))
            instruction = _normalise_space(section.get("instruction"))
            if not section_id or not section_name or not instruction:
                raise ValueError("Section config is incomplete")
            sample_id = f"{meeting_date}::{section_id}"
            reference = extract_minutes_section(html, section_name)
            prompt = _prompt(
                meeting_date=meeting_date,
                cutoff=cutoff,
                section_name=section_name,
                instruction=instruction,
                evidence=meeting_evidence,
            )
            if reference in prompt:
                raise ValueError(f"Reference was copied into prompt for {sample_id}")
            validate_no_prompt_reference_token_overlap(
                prompt,
                reference,
                sample_id=sample_id,
            )
            prompts.append(
                {
                    "schema_version": PROMPT_SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "meeting_date": meeting_date,
                    "section_id": section_id,
                    "section_name": section_name,
                    "information_as_of_date": cutoff,
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                    "evidence_sha256": sha256_text(meeting_evidence),
                    "evidence_facts": evidence_facts,
                    "reference_used_in_prompt": False,
                    "response": "",
                }
            )
            references.append(
                {
                    "schema_version": REFERENCE_SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "meeting_date": meeting_date,
                    "section_id": section_id,
                    "section_name": section_name,
                    "release_date": release_date,
                    "reference": reference,
                    "reference_sha256": sha256_text(reference),
                    "source_url": source_url,
                    "source_html_sha256": sha256_file(raw_path),
                }
            )

    expected_rows = len(meeting_dates) * len(sections)
    if len(prompts) != expected_rows or len(references) != expected_rows:
        raise AssertionError("Common-test matrix construction is incomplete")
    if len({row["sample_id"] for row in prompts}) != expected_rows:
        raise ValueError("Prompt sample IDs are not unique")

    prompts_path = destination / "prompts.jsonl"
    references_path = destination / "references.jsonl"
    _write_jsonl(prompts_path, prompts)
    _write_jsonl(references_path, references)
    manifest = seal_manifest({
        "schema_version": SCHEMA_VERSION,
        "evaluation_id": config.get("evaluation_id"),
        "population_id": population.get("population_id"),
        "meeting_dates": meeting_dates,
        "prospective_only_meeting_dates": population.get(
            "prospective_only_meeting_dates", []
        ),
        "section_count": len(sections),
        "sample_count": expected_rows,
        "indicator_count_per_meeting": next(iter(indicator_counts)),
        "information_cutoff_policy": "previous-calendar-day-v1",
        "reference_in_prompt": False,
        "secondary_llm_summary": False,
        "inputs": {
            "config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            "indicator_inputs": {
                "path": str(indicator_path),
                "sha256": sha256_file(indicator_path),
                "row_count": len(ledger_rows),
            },
            "ledger_manifest": {
                "path": str(ledger_manifest_path),
                "sha256": sha256_file(ledger_manifest_path),
                "payload_sha256": ledger_manifest["integrity"][
                    "payload_sha256"
                ],
            },
        },
        "outputs": {
            "prompts": {
                "path": str(prompts_path),
                "sha256": sha256_file(prompts_path),
                "row_count": len(prompts),
            },
            "references": {
                "path": str(references_path),
                "sha256": sha256_file(references_path),
                "row_count": len(references),
            },
        },
        "reference_sources": reference_sources,
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    })
    _atomic_write(
        manifest_path,
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return manifest_path, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the clean D-1 common-test prompts and separate official "
            "Minutes references for the four-artifact Chapter 2 comparison."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--indicator-inputs", required=True)
    parser.add_argument("--ledger-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--offline", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path, manifest = build_dataset(
        config_file=args.config,
        indicator_inputs_file=args.indicator_inputs,
        ledger_manifest_file=args.ledger_manifest,
        output_dir=args.output_dir,
        offline=args.offline,
    )
    print(f"test_manifest={path}")
    print(f"sample_count={manifest['sample_count']}")
    print(
        "indicator_count_per_meeting="
        f"{manifest['indicator_count_per_meeting']}"
    )


if __name__ == "__main__":
    main()
