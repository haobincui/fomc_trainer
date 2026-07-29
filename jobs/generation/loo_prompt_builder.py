"""Build frozen, auditable prompts for canonical leave-one-out generation.

The builder deliberately stops at prompt construction.  It does not load a
generative model, generate outputs, score outputs, or inspect target Minutes.

Input contracts
---------------
``analysis_blocks`` is JSONL with one row per meeting and indicator.  Each row
must contain ``meeting_date``, ``indicator``, and the configured analysis text
field (``generated`` by default).

``population`` is JSON containing either a list of ISO dates or an object with
``population_id`` and ``dates``/``meeting_dates``.

``section_roster`` is JSON containing either a list of section-family strings
or an object with ``sections``/``section_families``.  Entries may be strings or
objects with ``section_family`` (or ``name``) and an optional ``instruction``.

The exact-delete export is directly consumable by ``mask_generation.py`` using
the emitted ``run_intervention_roster.json``.  Neutral replacements live in a
separate directory because they are replacements, not exact deletions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from open_r1.provenance import fingerprint_artifact_path, sha256_file, sha256_text
from open_r1.validator.intervention import (
    has_line_block_boundaries,
    match_indicator_marker,
    normalise_indicator_text,
    single_contiguous_deletion,
)


PROMPT_ROW_SCHEMA_VERSION = "loo-prompt-row-v1"
PROMPT_MANIFEST_SCHEMA_VERSION = "loo-prompt-manifest-v1"
INTERVENTION_MANIFEST_SCHEMA_VERSION = "loo-intervention-v2"
NEUTRAL_MANIFEST_SCHEMA_VERSION = "loo-neutral-intervention-v1"
SOURCE_ROSTER_SCHEMA_VERSION = "loo-intervention-roster-v1"
RUN_ROSTER_SCHEMA_VERSION = SOURCE_ROSTER_SCHEMA_VERSION

BASELINE_INDICATOR = "None"
EXPECTED_INDICATOR_COUNT = 26

DEFAULT_PROMPT_TEMPLATE = """\
You are drafting one section of FOMC Minutes for the meeting on {meeting_date}.
Required section family: {section_family}
{section_instruction}

Use the following indicator-conditioned analysis blocks as evidence. Do not
quote the blocks verbatim and do not write any other Minutes section.

<<<BEGIN-LOO-ANALYSIS-BLOCKS>>>
{analysis_blocks}<<<END-LOO-ANALYSIS-BLOCKS>>>
"""

FORBIDDEN_TARGET_FIELDS = frozenset(
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

RESERVED_ANALYSIS_MARKERS = (
    "LOO-INDICATOR-BLOCK",
    "LOO-BLOCK-SLOT",
    "END-LOO-BLOCK-SLOTS",
    "BEGIN-LOO-ANALYSIS-BLOCKS",
    "END-LOO-ANALYSIS-BLOCKS",
)


class TokenizerLike(Protocol):
    """The minimal tokenizer interface used by the prompt builder."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ...


@dataclass(frozen=True)
class SectionSpec:
    family: str
    instruction: str


@dataclass(frozen=True)
class RenderedBlock:
    indicator: str
    text: str
    analysis_sha256: str
    local_start: int
    local_end: int


def _json_text(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _jsonl_text(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    )


def _read_json(path: Path, *, label: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label} JSON in {path}: {exc}") from exc


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid {label} JSON in {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"{label} row must be an object in {path}:{line_number}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{label} is empty: {path}")
    return rows


def _normalise_field_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def _find_forbidden_fields(value: Any, *, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            field = _normalise_field_name(key)
            path = f"{prefix}.{key}" if prefix else str(key)
            if field in FORBIDDEN_TARGET_FIELDS:
                found.append(path)
            found.extend(_find_forbidden_fields(child, prefix=path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_forbidden_fields(child, prefix=f"{prefix}[{index}]"))
    return found


def _reject_forbidden_fields(value: Any, *, label: str) -> None:
    forbidden = _find_forbidden_fields(value)
    if forbidden:
        raise ValueError(
            f"{label} contains prohibited target/reference text fields: "
            f"{forbidden[:10]}"
        )


def _normalise_iso_date(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO YYYY-MM-DD date, got {value!r}") from exc
    if parsed.isoformat() != text:
        raise ValueError(f"{label} must use canonical YYYY-MM-DD form, got {value!r}")
    return text


def _safe_context(population_id: str) -> str:
    context = re.sub(r"[^A-Za-z0-9.-]+", "_", population_id).strip("._-")
    if not context:
        raise ValueError("population_id does not contain a filename-safe character")
    return context


def _validate_artifact_filename_component(value: str, *, label: str) -> None:
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "_masked_" in value
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError(f"Unsafe {label} for prompt artifact filename: {value!r}")


def _load_population(
    path: Path,
    *,
    population_id_override: str | None,
) -> tuple[str, list[str]]:
    payload = _read_json(path, label="population roster")
    _reject_forbidden_fields(payload, label="Population roster")

    if isinstance(payload, list):
        population_id = str(population_id_override or "").strip()
        raw_dates = payload
    elif isinstance(payload, dict):
        declared_id = str(payload.get("population_id") or "").strip()
        population_id = str(population_id_override or declared_id).strip()
        if population_id_override and declared_id and population_id_override != declared_id:
            raise ValueError(
                "population_id override conflicts with the ID declared in the "
                f"population roster: {population_id_override!r} != {declared_id!r}"
            )
        raw_dates = payload.get("dates", payload.get("meeting_dates"))
    else:
        raise ValueError("Population roster must be a JSON list or object")

    if not population_id:
        raise ValueError(
            "population_id is required in the population roster or via "
            "--population-id"
        )
    if "::" in population_id or "\n" in population_id or "\r" in population_id:
        raise ValueError(f"Invalid population_id: {population_id!r}")
    if not isinstance(raw_dates, list) or not raw_dates:
        raise ValueError("Population roster requires a non-empty dates list")

    dates = [
        _normalise_iso_date(value, label=f"population date {index}")
        for index, value in enumerate(raw_dates)
    ]
    if len(dates) != len(set(dates)):
        raise ValueError("Population roster contains duplicate meeting dates")
    if dates != sorted(dates):
        raise ValueError("Population meeting dates must be in ascending order")
    return population_id, dates


def _load_sections(path: Path) -> tuple[str, list[SectionSpec]]:
    payload = _read_json(path, label="section roster")
    _reject_forbidden_fields(payload, label="Section roster")
    if isinstance(payload, list):
        roster_id = path.stem
        entries = payload
    elif isinstance(payload, dict):
        roster_id = str(payload.get("roster_id") or path.stem).strip()
        entries = payload.get("sections", payload.get("section_families"))
    else:
        raise ValueError("Section roster must be a JSON list or object")
    if not roster_id:
        raise ValueError("Section roster requires a non-empty roster_id")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Section roster requires a non-empty sections list")

    sections: list[SectionSpec] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            family = entry.strip()
            instruction = f"Write only the section belonging to {family!r}."
        elif isinstance(entry, dict):
            family = str(
                entry.get("section_family", entry.get("name", ""))
            ).strip()
            instruction = str(entry.get("instruction") or "").strip()
            if not instruction and family:
                instruction = f"Write only the section belonging to {family!r}."
        else:
            raise ValueError(
                f"Section roster entry {index} must be a string or object"
            )
        if not family:
            raise ValueError(f"Section roster entry {index} has no section family")
        if "::" in family or "\n" in family or "\r" in family:
            raise ValueError(
                f"Section family cannot contain '::' or a newline: {family!r}"
            )
        sections.append(SectionSpec(family=family, instruction=instruction))

    families = [section.family for section in sections]
    if len(families) != len(set(families)):
        raise ValueError("Section roster contains duplicate section families")
    return roster_id, sections


def _load_indicator_roster(path: Path) -> dict[str, Any]:
    payload = _read_json(path, label="indicator roster")
    _reject_forbidden_fields(payload, label="Indicator roster")
    if not isinstance(payload, dict):
        raise ValueError("Indicator roster must be a JSON object")
    if payload.get("schema_version") != SOURCE_ROSTER_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported indicator roster schema: {payload.get('schema_version')!r}"
        )

    baseline = str(payload.get("baseline_indicator") or "").strip()
    indicators = [
        str(value).strip()
        for value in payload.get("indicators", [])
        if str(value).strip()
    ]
    if baseline != BASELINE_INDICATOR:
        raise ValueError(
            f"Canonical prompt construction requires baseline_indicator "
            f"{BASELINE_INDICATOR!r}, got {baseline!r}"
        )
    if len(indicators) != EXPECTED_INDICATOR_COUNT:
        raise ValueError(
            f"Canonical prompt construction requires exactly "
            f"{EXPECTED_INDICATOR_COUNT} indicators, found {len(indicators)}"
        )
    if len(indicators) != len(set(indicators)):
        raise ValueError("Indicator roster contains duplicate indicators")
    for indicator in indicators:
        _validate_artifact_filename_component(indicator, label="indicator")

    raw_markers = payload.get("indicator_markers")
    if not isinstance(raw_markers, dict) or set(raw_markers) != set(indicators):
        raise ValueError(
            "Indicator roster requires marker lists for exactly the declared "
            "indicator universe"
        )
    markers: dict[str, list[str]] = {}
    for indicator in indicators:
        values = raw_markers[indicator]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Indicator {indicator!r} has no declared markers")
        normalised = [str(value).strip() for value in values]
        if any(not normalise_indicator_text(value) for value in normalised):
            raise ValueError(f"Indicator {indicator!r} has an invalid marker")
        markers[indicator] = normalised

    return {
        "schema_version": SOURCE_ROSTER_SCHEMA_VERSION,
        "roster_id": str(payload.get("roster_id") or path.stem),
        "baseline_indicator": baseline,
        "indicators": indicators,
        "indicator_markers": markers,
        "source_contexts": payload.get("contexts", []),
    }


def _load_analysis_blocks(
    path: Path,
    *,
    analysis_text_field: str,
    dates: list[str],
    indicators: list[str],
) -> dict[tuple[str, str], str]:
    if not analysis_text_field.strip():
        raise ValueError("analysis_text_field cannot be blank")
    rows = _read_jsonl(path, label="analysis blocks")
    expected = {(meeting_date, indicator) for meeting_date in dates for indicator in indicators}
    blocks: dict[tuple[str, str], str] = {}

    for line_number, row in enumerate(rows, 1):
        _reject_forbidden_fields(
            row,
            label=f"Analysis block row {line_number}",
        )
        meeting_date = _normalise_iso_date(
            row.get("meeting_date"),
            label=f"analysis block meeting_date at row {line_number}",
        )
        indicator = str(row.get("indicator") or "").strip()
        key = (meeting_date, indicator)
        if key not in expected:
            raise ValueError(
                f"Analysis block row {line_number} is outside the frozen "
                f"meeting×indicator universe: {key!r}"
            )
        if key in blocks:
            raise ValueError(f"Duplicate analysis block coverage for {key!r}")
        if analysis_text_field not in row:
            raise ValueError(
                f"Analysis block row {line_number} lacks configured text field "
                f"{analysis_text_field!r}"
            )
        text = row[analysis_text_field]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Analysis block text is empty for {key!r}")
        if any(marker in text for marker in RESERVED_ANALYSIS_MARKERS):
            raise ValueError(
                f"Analysis block {key!r} contains a reserved LOO delimiter"
            )
        blocks[key] = text

    observed = set(blocks)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            "Analysis blocks do not provide complete frozen meeting×indicator "
            f"coverage: missing={missing[:10]}, extra={extra[:10]}"
        )
    return blocks


def _validate_prompt_template(template: str) -> None:
    formatter = string.Formatter()
    fields = [
        field_name
        for _, field_name, _, _ in formatter.parse(template)
        if field_name is not None
    ]
    required = {
        "analysis_blocks",
        "meeting_date",
        "section_family",
        "section_instruction",
    }
    unknown = set(fields) - required
    missing = required - set(fields)
    repeated = sorted(field for field in required if fields.count(field) != 1)
    if unknown or missing or repeated:
        raise ValueError(
            "Prompt template must contain each supported placeholder exactly "
            f"once: missing={sorted(missing)}, unknown={sorted(unknown)}, "
            f"not_once={repeated}"
        )


def _render_block(indicator: str, marker: str, analysis_text: str) -> str:
    return (
        f"<<<LOO-INDICATOR-BLOCK:{indicator} | {marker}>>>\n"
        f"Indicator: {marker}\n"
        "Analysis:\n"
        f"{analysis_text}\n"
        f"<<<END-LOO-INDICATOR-BLOCK:{indicator}>>>\n"
    )


def _render_analysis_region(
    *,
    meeting_date: str,
    indicators: list[str],
    markers: dict[str, list[str]],
    analysis_blocks: dict[tuple[str, str], str],
) -> tuple[str, list[RenderedBlock]]:
    parts: list[str] = []
    rendered: list[RenderedBlock] = []
    cursor = 0
    for position, indicator in enumerate(indicators):
        slot = f"--- LOO-BLOCK-SLOT {position:03d} ---\n"
        parts.append(slot)
        cursor += len(slot)
        analysis = analysis_blocks[(meeting_date, indicator)]
        block = _render_block(indicator, markers[indicator][0], analysis)
        start = cursor
        end = start + len(block)
        parts.append(block)
        rendered.append(
            RenderedBlock(
                indicator=indicator,
                text=block,
                analysis_sha256=sha256_text(analysis),
                local_start=start,
                local_end=end,
            )
        )
        cursor = end
    terminator = "--- END-LOO-BLOCK-SLOTS ---\n"
    parts.append(terminator)
    return "".join(parts), rendered


def _token_count(tokenizer: TokenizerLike, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(token_ids, (list, tuple)):
        raise TypeError("Tokenizer.encode must return a list or tuple of token IDs")
    return len(token_ids)


def _candidate_k_values(
    count_for_k: Any,
    *,
    target: int,
    maximum: int,
) -> list[int]:
    """Return likely repetition counts, then a complete deterministic fallback."""

    cache: dict[int, int] = {}

    def observed(k: int) -> int:
        if k not in cache:
            cache[k] = int(count_for_k(k))
        return cache[k]

    likely: list[int] = []
    base = observed(0)
    likely.append(max(0, min(maximum, target - base)))

    low, high = 0, maximum
    while low < high:
        middle = (low + high) // 2
        if observed(middle) < target:
            low = middle + 1
        else:
            high = middle
    likely.append(low)

    expanded: list[int] = []
    for centre in likely:
        for distance in range(0, 33):
            for candidate in (centre - distance, centre + distance):
                if 0 <= candidate <= maximum and candidate not in expanded:
                    expanded.append(candidate)
    expanded.extend(value for value in range(maximum + 1) if value not in expanded)
    return expanded


def _make_token_matched_neutral_block(
    *,
    tokenizer: TokenizerLike,
    indicator: str,
    marker: str,
    source_block: str,
    full_prefix: str,
    full_suffix: str,
    source_full_prompt_token_count: int,
) -> tuple[str, int]:
    target_block_tokens = _token_count(tokenizer, source_block)
    stems = (
        "Analysis withheld.",
        "Withheld.",
        "N/A",
        "[WITHHELD]",
        "Neutral information withheld.",
    )
    padding_units = (" neutral", " withheld", " data")
    maximum = max(target_block_tokens * 2 + 128, 256)

    for stem in stems:
        for padding_unit in padding_units:
            def candidate(k: int) -> str:
                body = stem + (padding_unit * k)
                return _render_block(indicator, marker, body)

            for repetitions in _candidate_k_values(
                lambda k: _token_count(tokenizer, candidate(k)),
                target=target_block_tokens,
                maximum=maximum,
            ):
                neutral_block = candidate(repetitions)
                if neutral_block == source_block:
                    continue
                if _token_count(tokenizer, neutral_block) != target_block_tokens:
                    continue
                neutral_prompt = full_prefix + neutral_block + full_suffix
                if (
                    _token_count(tokenizer, neutral_prompt)
                    != source_full_prompt_token_count
                ):
                    continue
                return neutral_block, target_block_tokens

    raise ValueError(
        f"Could not construct a content-neutral, exact token-length replacement "
        f"for indicator {indicator!r} with {target_block_tokens} tokens"
    )


def _make_prompt_row(
    *,
    sample_id: str,
    meeting_date: str,
    section_family: str,
    population_id: str,
    population_context: str,
    source_index: int,
    arm: str,
    indicator: str,
    prompt: str,
    tokenizer: TokenizerLike,
) -> dict[str, Any]:
    return {
        "schema_version": PROMPT_ROW_SCHEMA_VERSION,
        "sample_id": sample_id,
        "meeting_date": meeting_date,
        "section_name": section_family,
        "section_family": section_family,
        "population_id": population_id,
        "evaluation_context": population_context,
        "source_index": source_index,
        "arm": arm,
        "indicator": indicator,
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "prompt_token_count_no_special_tokens": _token_count(tokenizer, prompt),
    }


def _validate_tokenizer_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        raise ValueError("tokenizer_artifact must be a fingerprint object")
    digest = str(artifact.get("sha256") or "").strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(
            "tokenizer_artifact.sha256 must be a lowercase 64-character digest"
        )
    return dict(artifact)


def _build_run_roster(
    source_roster: dict[str, Any],
    *,
    population_context: str,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_ROSTER_SCHEMA_VERSION,
        "roster_id": (
            f"{source_roster['roster_id']}::{population_context}"
        ),
        "source_roster_id": source_roster["roster_id"],
        "baseline_indicator": source_roster["baseline_indicator"],
        "contexts": [population_context],
        "indicators": source_roster["indicators"],
        "indicator_markers": source_roster["indicator_markers"],
    }


def _write_frozen_artifacts(
    output_dir: Path,
    artifacts: dict[str, str],
) -> None:
    resolved_root = output_dir.resolve()
    targets: list[tuple[Path, str]] = []
    for relative_path, content in artifacts.items():
        target = (output_dir / relative_path).resolve()
        if not target.is_relative_to(resolved_root):
            raise ValueError(f"Artifact path escapes output directory: {relative_path}")
        if target.is_file() and target.read_text(encoding="utf-8") != content:
            raise ValueError(
                f"Refusing to overwrite incompatible frozen artifact {target}; "
                "use a new run-scoped output directory"
            )
        if target.exists() and not target.is_file():
            raise ValueError(f"Artifact destination is not a file: {target}")
        targets.append((target, content))

    for target, content in targets:
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)


def build_canonical_loo_prompts(
    *,
    analysis_blocks_file: str | Path,
    section_roster_file: str | Path,
    indicator_roster_file: str | Path,
    population_file: str | Path,
    tokenizer: TokenizerLike,
    tokenizer_artifact: dict[str, Any],
    output_dir: str | Path,
    analysis_text_field: str = "generated",
    population_id: str | None = None,
    prompt_template_file: str | Path | None = None,
) -> dict[str, Any]:
    """Build and freeze exact-delete and token-matched neutral prompt artifacts."""

    analysis_path = Path(analysis_blocks_file).expanduser().resolve()
    section_path = Path(section_roster_file).expanduser().resolve()
    indicator_path = Path(indicator_roster_file).expanduser().resolve()
    population_path = Path(population_file).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    tokenizer_fingerprint = _validate_tokenizer_artifact(tokenizer_artifact)

    population_id_value, meeting_dates = _load_population(
        population_path,
        population_id_override=population_id,
    )
    population_context = _safe_context(population_id_value)
    _validate_artifact_filename_component(
        population_context,
        label="population context",
    )
    section_roster_id, sections = _load_sections(section_path)
    source_roster = _load_indicator_roster(indicator_path)
    indicators = list(source_roster["indicators"])
    markers = dict(source_roster["indicator_markers"])
    analysis_blocks = _load_analysis_blocks(
        analysis_path,
        analysis_text_field=analysis_text_field,
        dates=meeting_dates,
        indicators=indicators,
    )

    if prompt_template_file is None:
        template = DEFAULT_PROMPT_TEMPLATE
        template_source = "builtin:loo_prompt_builder.DEFAULT_PROMPT_TEMPLATE"
    else:
        template_path = Path(prompt_template_file).expanduser().resolve()
        if not template_path.is_file():
            raise FileNotFoundError(f"Prompt template does not exist: {template_path}")
        template = template_path.read_text(encoding="utf-8")
        template_source = str(template_path)
    _validate_prompt_template(template)

    run_roster = _build_run_roster(
        source_roster,
        population_context=population_context,
    )
    run_roster_text = _json_text(run_roster)
    run_roster_sha256 = sha256_text(run_roster_text)

    exact_rows: dict[str, list[dict[str, Any]]] = {
        BASELINE_INDICATOR: []
    }
    exact_rows.update({indicator: [] for indicator in indicators})
    neutral_rows: dict[str, list[dict[str, Any]]] = {
        BASELINE_INDICATOR: []
    }
    neutral_rows.update({indicator: [] for indicator in indicators})
    deletion_proofs: dict[tuple[str, str], dict[str, Any]] = {}
    neutral_proofs: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    block_inventory: list[dict[str, Any]] = []

    unit_index = 0
    for meeting_date in meeting_dates:
        analysis_region, rendered_blocks = _render_analysis_region(
            meeting_date=meeting_date,
            indicators=indicators,
            markers=markers,
            analysis_blocks=analysis_blocks,
        )
        for rendered in rendered_blocks:
            block_inventory.append(
                {
                    "meeting_date": meeting_date,
                    "indicator": rendered.indicator,
                    "analysis_text_sha256": rendered.analysis_sha256,
                    "rendered_block_sha256": sha256_text(rendered.text),
                    "rendered_block_token_count_no_special_tokens": _token_count(
                        tokenizer,
                        rendered.text,
                    ),
                }
            )

        for section in sections:
            sample_id = f"{meeting_date}::{section.family}"
            sentinel = f"__LOO_ANALYSIS_BLOCK_SENTINEL_{hashlib.sha256(sample_id.encode()).hexdigest()}__"
            rendered_template = template.format(
                meeting_date=meeting_date,
                section_family=section.family,
                section_instruction=section.instruction,
                analysis_blocks=sentinel,
            )
            if rendered_template.count(sentinel) != 1:
                raise ValueError(
                    f"Prompt template did not preserve one analysis placeholder "
                    f"for {sample_id}"
                )
            prompt_prefix, prompt_suffix = rendered_template.split(sentinel)
            full_prompt = prompt_prefix + analysis_region + prompt_suffix
            full_prompt_token_count = _token_count(tokenizer, full_prompt)
            full_row = _make_prompt_row(
                sample_id=sample_id,
                meeting_date=meeting_date,
                section_family=section.family,
                population_id=population_id_value,
                population_context=population_context,
                source_index=unit_index,
                arm="full",
                indicator=BASELINE_INDICATOR,
                prompt=full_prompt,
                tokenizer=tokenizer,
            )
            exact_rows[BASELINE_INDICATOR].append(full_row)
            neutral_rows[BASELINE_INDICATOR].append(dict(full_row))
            ledger_rows.append(
                {
                    "sample_id": sample_id,
                    "meeting_date": meeting_date,
                    "section_family": section.family,
                    "arm": "full",
                    "indicator": BASELINE_INDICATOR,
                    "prompt_sha256": full_row["prompt_sha256"],
                    "prompt_token_count_no_special_tokens": full_row[
                        "prompt_token_count_no_special_tokens"
                    ],
                }
            )

            analysis_offset = len(prompt_prefix)
            for rendered in rendered_blocks:
                deletion_start = analysis_offset + rendered.local_start
                deletion_end = analysis_offset + rendered.local_end
                source_block = full_prompt[deletion_start:deletion_end]
                if source_block != rendered.text:
                    raise AssertionError(
                        f"Internal block-offset mismatch for "
                        f"{sample_id}/{rendered.indicator}"
                    )
                delete_prompt = (
                    full_prompt[:deletion_start] + full_prompt[deletion_end:]
                )
                observed_start, observed_end, removed = single_contiguous_deletion(
                    full_prompt,
                    delete_prompt,
                )
                if (
                    observed_start != deletion_start
                    or observed_end != deletion_end
                    or removed != source_block
                ):
                    raise ValueError(
                        f"Ambiguous deletion span for "
                        f"{sample_id}/{rendered.indicator}"
                    )
                if not has_line_block_boundaries(
                    full_prompt,
                    deletion_start,
                    deletion_end,
                ):
                    raise ValueError(
                        f"Deletion is not line/block bounded for "
                        f"{sample_id}/{rendered.indicator}"
                    )
                matched_marker = match_indicator_marker(
                    removed,
                    markers[rendered.indicator],
                )
                if matched_marker is None:
                    raise ValueError(
                        f"Removed block does not prove indicator identity for "
                        f"{sample_id}/{rendered.indicator}"
                    )
                delete_row = _make_prompt_row(
                    sample_id=sample_id,
                    meeting_date=meeting_date,
                    section_family=section.family,
                    population_id=population_id_value,
                    population_context=population_context,
                    source_index=unit_index,
                    arm="delete",
                    indicator=rendered.indicator,
                    prompt=delete_prompt,
                    tokenizer=tokenizer,
                )
                exact_rows[rendered.indicator].append(delete_row)
                deletion_proofs[(rendered.indicator, sample_id)] = {
                    "indicator": rendered.indicator,
                    "context": population_context,
                    "row_key": ["sample_id", sample_id],
                    "sample_id": sample_id,
                    "meeting_date": meeting_date,
                    "section_name": section.family,
                    "full_prompt_sha256": full_row["prompt_sha256"],
                    "masked_prompt_sha256": delete_row["prompt_sha256"],
                    "removed_block_sha256": sha256_text(removed),
                    "matched_indicator_marker": matched_marker,
                    "deletion_start": deletion_start,
                    "deletion_end": deletion_end,
                    "removed_character_count": len(removed),
                    "line_block_boundaries_validated": True,
                }
                ledger_rows.append(
                    {
                        "sample_id": sample_id,
                        "meeting_date": meeting_date,
                        "section_family": section.family,
                        "arm": "delete",
                        "indicator": rendered.indicator,
                        "prompt_sha256": delete_row["prompt_sha256"],
                        "prompt_token_count_no_special_tokens": delete_row[
                            "prompt_token_count_no_special_tokens"
                        ],
                        "source_full_prompt_sha256": full_row["prompt_sha256"],
                        "intervention_start": deletion_start,
                        "intervention_end": deletion_end,
                    }
                )

                block_prefix = full_prompt[:deletion_start]
                block_suffix = full_prompt[deletion_end:]
                neutral_block, block_token_count = _make_token_matched_neutral_block(
                    tokenizer=tokenizer,
                    indicator=rendered.indicator,
                    marker=markers[rendered.indicator][0],
                    source_block=source_block,
                    full_prefix=block_prefix,
                    full_suffix=block_suffix,
                    source_full_prompt_token_count=full_prompt_token_count,
                )
                neutral_prompt = block_prefix + neutral_block + block_suffix
                neutral_row = _make_prompt_row(
                    sample_id=sample_id,
                    meeting_date=meeting_date,
                    section_family=section.family,
                    population_id=population_id_value,
                    population_context=population_context,
                    source_index=unit_index,
                    arm="neutral",
                    indicator=rendered.indicator,
                    prompt=neutral_prompt,
                    tokenizer=tokenizer,
                )
                neutral_rows[rendered.indicator].append(neutral_row)
                neutral_proof = {
                    "sample_id": sample_id,
                    "meeting_date": meeting_date,
                    "section_family": section.family,
                    "indicator": rendered.indicator,
                    "full_prompt_sha256": full_row["prompt_sha256"],
                    "neutral_prompt_sha256": neutral_row["prompt_sha256"],
                    "source_block_sha256": sha256_text(source_block),
                    "neutral_block_sha256": sha256_text(neutral_block),
                    "replacement_start": deletion_start,
                    "source_replacement_end": deletion_end,
                    "neutral_replacement_end": deletion_start + len(neutral_block),
                    "prefix_sha256": sha256_text(block_prefix),
                    "suffix_sha256": sha256_text(block_suffix),
                    "source_block_token_count_no_special_tokens": block_token_count,
                    "neutral_block_token_count_no_special_tokens": _token_count(
                        tokenizer,
                        neutral_block,
                    ),
                    "source_full_prompt_token_count_no_special_tokens": (
                        full_prompt_token_count
                    ),
                    "neutral_full_prompt_token_count_no_special_tokens": (
                        neutral_row["prompt_token_count_no_special_tokens"]
                    ),
                    "label_and_delimiters_preserved": (
                        neutral_block.startswith(
                            source_block.split("Analysis:\n", 1)[0] + "Analysis:\n"
                        )
                        and neutral_block.endswith(
                            f"<<<END-LOO-INDICATOR-BLOCK:"
                            f"{rendered.indicator}>>>\n"
                        )
                    ),
                    "prefix_suffix_unchanged": True,
                    "token_length_matched": True,
                }
                if not neutral_proof["label_and_delimiters_preserved"]:
                    raise AssertionError(
                        f"Neutral block lost its label/delimiters for "
                        f"{sample_id}/{rendered.indicator}"
                    )
                neutral_proofs.append(neutral_proof)
                ledger_rows.append(
                    {
                        "sample_id": sample_id,
                        "meeting_date": meeting_date,
                        "section_family": section.family,
                        "arm": "neutral",
                        "indicator": rendered.indicator,
                        "prompt_sha256": neutral_row["prompt_sha256"],
                        "prompt_token_count_no_special_tokens": neutral_row[
                            "prompt_token_count_no_special_tokens"
                        ],
                        "source_full_prompt_sha256": full_row["prompt_sha256"],
                        "intervention_start": deletion_start,
                        "source_intervention_end": deletion_end,
                        "neutral_intervention_end": (
                            deletion_start + len(neutral_block)
                        ),
                    }
                )
            unit_index += 1

    expected_unit_count = len(meeting_dates) * len(sections)
    if unit_index != expected_unit_count:
        raise AssertionError("Internal unit-count mismatch")

    exact_texts: dict[str, str] = {}
    neutral_texts: dict[str, str] = {}
    artifact_inventory: list[dict[str, Any]] = []
    for indicator in [BASELINE_INDICATOR, *indicators]:
        filename = f"{indicator}_masked_{population_context}.jsonl"
        exact_relative = f"exact_delete/{filename}"
        neutral_relative = f"neutral/{filename}"
        exact_text = _jsonl_text(exact_rows[indicator])
        neutral_text = _jsonl_text(neutral_rows[indicator])
        exact_texts[exact_relative] = exact_text
        neutral_texts[neutral_relative] = neutral_text
        artifact_inventory.extend(
            [
                {
                    "relative_path": exact_relative,
                    "arm": "full" if indicator == BASELINE_INDICATOR else "delete",
                    "indicator": indicator,
                    "context": population_context,
                    "row_count": len(exact_rows[indicator]),
                    "sha256": sha256_text(exact_text),
                },
                {
                    "relative_path": neutral_relative,
                    "arm": (
                        "full_export"
                        if indicator == BASELINE_INDICATOR
                        else "neutral"
                    ),
                    "indicator": indicator,
                    "context": population_context,
                    "row_count": len(neutral_rows[indicator]),
                    "sha256": sha256_text(neutral_text),
                },
            ]
        )

    exact_input_artifacts = [
        {
            "relative_path": f"{indicator}_masked_{population_context}.jsonl",
            "indicator": indicator,
            "context": population_context,
            "source_row_count": len(exact_rows[indicator]),
            "source_file_sha256": sha256_text(
                exact_texts[
                    f"exact_delete/{indicator}_masked_{population_context}.jsonl"
                ]
            ),
        }
        for indicator in [BASELINE_INDICATOR, *indicators]
    ]
    interventions = [
        deletion_proofs[(indicator, row["sample_id"])]
        for indicator in indicators
        for row in exact_rows[BASELINE_INDICATOR]
    ]
    intervention_manifest = {
        "schema_version": INTERVENTION_MANIFEST_SCHEMA_VERSION,
        "population_id": population_id_value,
        "population_context": population_context,
        "roster_id": run_roster["roster_id"],
        "roster_file_sha256": run_roster_sha256,
        "baseline_indicator": BASELINE_INDICATOR,
        "indicators": indicators,
        "indicator_markers": markers,
        "contexts": [population_context],
        "validation": "exact_single_contiguous_prompt_deletion",
        "indicator_identity_validation": "declared_marker_in_removed_block_header",
        "block_boundary_validation": "complete_line_or_block_boundaries",
        "indicator_block_span_validation": "pairwise_non_overlapping_per_sample",
        "distinct_masked_prompt_per_indicator": True,
        "input_artifacts": exact_input_artifacts,
        "interventions": interventions,
    }
    intervention_manifest_text = _json_text(intervention_manifest)

    neutral_manifest = {
        "schema_version": NEUTRAL_MANIFEST_SCHEMA_VERSION,
        "population_id": population_id_value,
        "population_context": population_context,
        "roster_id": run_roster["roster_id"],
        "roster_file_sha256": run_roster_sha256,
        "baseline_indicator": BASELINE_INDICATOR,
        "indicators": indicators,
        "contexts": [population_context],
        "validation": (
            "single_block_replacement_with_preserved_label_delimiters_and_"
            "exact_token_length"
        ),
        "token_count_policy": "tokenizer.encode(add_special_tokens=False)",
        "tokenizer_artifact": tokenizer_fingerprint,
        "replacements": neutral_proofs,
    }
    neutral_manifest_text = _json_text(neutral_manifest)
    ledger_text = _jsonl_text(ledger_rows)

    prompt_manifest = {
        "schema_version": PROMPT_MANIFEST_SCHEMA_VERSION,
        "population_id": population_id_value,
        "population_context": population_context,
        "population_dates": meeting_dates,
        "population_dates_sha256": sha256_text(
            json.dumps(meeting_dates, ensure_ascii=False, separators=(",", ":"))
        ),
        "stable_sample_id_schema": "meeting_date::section_family",
        "section_roster_id": section_roster_id,
        "section_families": [section.family for section in sections],
        "source_indicator_roster_id": source_roster["roster_id"],
        "run_intervention_roster": {
            "relative_path": "run_intervention_roster.json",
            "sha256": run_roster_sha256,
        },
        "baseline_indicator": BASELINE_INDICATOR,
        "indicators": indicators,
        "block_order": indicators,
        "analysis_text_field": analysis_text_field,
        "prompt_template": {
            "source": template_source,
            "sha256": sha256_text(template),
            "text": template,
        },
        "token_count_policy": "tokenizer.encode(add_special_tokens=False)",
        "tokenizer_artifact": tokenizer_fingerprint,
        "source_artifacts": [
            {
                "role": "analysis_blocks",
                "path": str(analysis_path),
                "sha256": sha256_file(analysis_path),
            },
            {
                "role": "section_roster",
                "path": str(section_path),
                "sha256": sha256_file(section_path),
            },
            {
                "role": "indicator_roster",
                "path": str(indicator_path),
                "sha256": sha256_file(indicator_path),
            },
            {
                "role": "population",
                "path": str(population_path),
                "sha256": sha256_file(population_path),
            },
        ],
        "forbidden_target_reference_fields": sorted(FORBIDDEN_TARGET_FIELDS),
        "counts": {
            "meeting_count": len(meeting_dates),
            "section_count": len(sections),
            "unit_count": expected_unit_count,
            "indicator_count": len(indicators),
            "analysis_block_count": len(analysis_blocks),
            "full_prompt_count": expected_unit_count,
            "delete_prompt_count": expected_unit_count * len(indicators),
            "neutral_prompt_count": expected_unit_count * len(indicators),
        },
        "artifacts": artifact_inventory,
        "analysis_block_inventory": block_inventory,
        "prompt_ledger": {
            "relative_path": "prompt_ledger.jsonl",
            "row_count": len(ledger_rows),
            "sha256": sha256_text(ledger_text),
        },
        "intervention_manifest": {
            "relative_path": "intervention_manifest.json",
            "schema_version": INTERVENTION_MANIFEST_SCHEMA_VERSION,
            "sha256": sha256_text(intervention_manifest_text),
        },
        "neutral_intervention_manifest": {
            "relative_path": "neutral_intervention_manifest.json",
            "schema_version": NEUTRAL_MANIFEST_SCHEMA_VERSION,
            "sha256": sha256_text(neutral_manifest_text),
        },
        "mask_generation_interface": {
            "input_folder": "exact_delete",
            "roster_file": "run_intervention_roster.json",
            "filename_schema": "<indicator>_masked_<population_context>.jsonl",
        },
    }
    prompt_manifest_text = _json_text(prompt_manifest)

    artifacts_to_write = {
        **exact_texts,
        **neutral_texts,
        "run_intervention_roster.json": run_roster_text,
        "prompt_ledger.jsonl": ledger_text,
        "intervention_manifest.json": intervention_manifest_text,
        "neutral_intervention_manifest.json": neutral_manifest_text,
        "prompt_manifest.json": prompt_manifest_text,
    }
    _write_frozen_artifacts(destination, artifacts_to_write)
    return prompt_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build frozen full, exact-delete, and token-length-matched neutral "
            "LOO prompts. This command does not generate model outputs."
        )
    )
    parser.add_argument("--analysis-blocks", required=True)
    parser.add_argument("--section-roster", required=True)
    parser.add_argument(
        "--indicator-roster",
        default="configs/main/leave_one_out_roster.json",
    )
    parser.add_argument("--population", required=True)
    parser.add_argument(
        "--population-id",
        help="Required only when the population JSON is a bare list.",
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-field", default="generated")
    parser.add_argument("--prompt-template")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tokenizer_path = Path(args.tokenizer).expanduser().resolve()
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer must be a frozen local artifact: {tokenizer_path}"
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    manifest = build_canonical_loo_prompts(
        analysis_blocks_file=args.analysis_blocks,
        section_roster_file=args.section_roster,
        indicator_roster_file=args.indicator_roster,
        population_file=args.population,
        tokenizer=tokenizer,
        tokenizer_artifact=fingerprint_artifact_path(tokenizer_path),
        output_dir=args.output_dir,
        analysis_text_field=args.analysis_field,
        population_id=args.population_id,
        prompt_template_file=args.prompt_template,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "schema_version": manifest["schema_version"],
                "population_id": manifest["population_id"],
                "counts": manifest["counts"],
                "output_dir": str(Path(args.output_dir).expanduser().resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
