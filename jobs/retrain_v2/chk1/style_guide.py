"""Build train-only, fact-free style guides from official Minutes excerpts.

This module intentionally does not produce sanitized example prose.  Even a
heavily redacted excerpt can preserve a meeting-specific conclusion through its
syntax.  Instead, source sentences are reduced to a small, controlled set of
style atoms and each section guide is rendered exclusively from fixed text.
Consequently, no arbitrary source token can enter ``guide_text``.

The accepted row schema matches the existing Minutes target artifacts:

``sample_id, split, meeting_date, section_name, source_row_index,
reference_excerpt``.

Every row must be labelled ``train``.  Eval/test rows, mixed splits, missing
split labels, duplicate sample identifiers, and non-Minutes text fields fail
closed.  The returned artifact contains only opaque source hashes, deterministic
style statistics, and fixed guide prose; it never contains an excerpt, meeting
date, sample identifier, or section name in clear text.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STYLE_GUIDE_SCHEMA_VERSION = "chk1-section-style-guide-v1"
ABSTRACTION_VERSION = "controlled-style-atoms-v1"
REQUIRED_SPLIT = "train"
SOURCE_TEXT_FIELD = "reference_excerpt"
REQUIRED_ROW_FIELDS = (
    "sample_id",
    "split",
    "meeting_date",
    "section_name",
    "source_row_index",
    SOURCE_TEXT_FIELD,
)
STYLE_ID_PREFIX = "section-style-v1-"


class StyleGuideError(ValueError):
    """Raised when a style corpus or artifact violates the safety contract."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MONTH_NAMES = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_MONTH_RE = re.compile(r"\b(?:" + "|".join(_MONTH_NAMES) + r")\b", re.IGNORECASE)
_NUMERIC_RE = re.compile(
    r"(?<![A-Za-z])(?:[+$\-]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)"
    r"(?:\s*(?:percent|percentage\s+points?|basis\s+points?|bps?|"
    r"million|billion|trillion))?",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|"
    r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b|"
    r"\b(?:q[1-4]|first|second|third|fourth)\s+quarter\b",
    re.IGNORECASE,
)
_PERSON_REFERENCE_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Mses|Messrs|Dr)\.\s*[A-Z][A-Za-z'\-]+|"
    r"\b(?:Chair(?:man|woman)?|Vice\s+Chair(?:man|woman)?|Governor|President)"
    r"\s+[A-Z][A-Za-z'\-]+",
)
_MEETING_FACT_RE = re.compile(
    r"\bpresent\s*:|"
    r"\bsecretary(?:'s)?\s+note\b|"
    r"\b(?:meeting|conference\s+call)\s+(?:convened|began|adjourned|was\s+held)\b|"
    r"\bminutes\s+of\s+the\s+(?:previous|federal\s+open\s+market)\b|"
    r"\b(?:agenda|attendance|oaths?\s+of\s+office|officers?\s+were\s+selected)\b",
    re.IGNORECASE,
)
_POLICY_OUTCOME_RE = re.compile(
    r"\bby\s+unanimous\s+vote\b|"
    r"\b(?:the\s+)?(?:committee|board|members?|participants?)\s+"
    r"(?:voted|decided|approved|authorized|directed|reaffirmed|adopted)\b|"
    r"\b(?:voted|decided|approved|authorized|directed)\s+to\b|"
    r"\b(?:target(?:\s+range)?|federal\s+funds\s+rate)\b.{0,80}"
    r"\b(?:raised|lowered|maintained|kept|left|reduced|increased)\b",
    re.IGNORECASE,
)
_POLICY_PREFERENCE_RE = re.compile(
    r"\bparticipants?\b.{0,160}\b(?:agreed|supported|preferred|favou?red)\b"
    r".{0,160}\b(?:policy|rate|purchase|facility|program|balance\s+sheet)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


# Only these source-derived lexical items may appear in a style signature or
# rendered guide.  They are discourse/hedging markers, not names or facts.
_LEXICAL_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("although", re.compile(r"\balthough\b", re.IGNORECASE)),
    ("however", re.compile(r"\bhowever\b", re.IGNORECASE)),
    ("while", re.compile(r"\bwhile\b", re.IGNORECASE)),
    ("on balance", re.compile(r"\bon\s+balance\b", re.IGNORECASE)),
    ("appeared", re.compile(r"\bappeared\b", re.IGNORECASE)),
    ("seemed", re.compile(r"\bseemed\b", re.IGNORECASE)),
    ("likely", re.compile(r"\blikely\b", re.IGNORECASE)),
    ("could", re.compile(r"\bcould\b", re.IGNORECASE)),
    ("might", re.compile(r"\bmight\b", re.IGNORECASE)),
    ("reportedly", re.compile(r"\breportedly\b", re.IGNORECASE)),
    ("estimated", re.compile(r"\bestimated\b", re.IGNORECASE)),
    ("suggested", re.compile(r"\bsuggested\b", re.IGNORECASE)),
    ("continued", re.compile(r"\bcontinued\b", re.IGNORECASE)),
    ("remained", re.compile(r"\bremained\b", re.IGNORECASE)),
    ("edged", re.compile(r"\bedged\b", re.IGNORECASE)),
    ("most", re.compile(r"\bmost\b", re.IGNORECASE)),
    ("many", re.compile(r"\bmany\b", re.IGNORECASE)),
    ("several", re.compile(r"\bseveral\b", re.IGNORECASE)),
    ("some", re.compile(r"\bsome\b", re.IGNORECASE)),
    ("few", re.compile(r"\bfew\b", re.IGNORECASE)),
)

_STYLE_MOVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "contrast",
        re.compile(r"\b(?:although|however|while|in\s+contrast|on\s+the\s+other\s+hand)\b", re.IGNORECASE),
    ),
    (
        "causal_link",
        re.compile(r"\b(?:reflecting|because|attributable\s+to|in\s+response\s+to|as\s+a\s+result)\b", re.IGNORECASE),
    ),
    (
        "uncertainty",
        re.compile(r"\b(?:appeared|seemed|likely|could|might|reportedly|estimated|suggested|uncertain)\b", re.IGNORECASE),
    ),
    (
        "temporal_comparison",
        re.compile(r"\b(?:continued|remained|edged|recent|earlier|over\s+the\s+period|on\s+balance)\b", re.IGNORECASE),
    ),
    (
        "source_attribution",
        re.compile(r"\b(?:staff|participants?|members?|market\s+participants|respondents)\b", re.IGNORECASE),
    ),
    (
        "group_quantification",
        re.compile(r"\b(?:most|many|several|some|few)\b", re.IGNORECASE),
    ),
    (
        "risk_balance",
        re.compile(r"\b(?:risk|risks|uncertainty|uncertainties|upside|downside)\b", re.IGNORECASE),
    ),
)

_OPENING_ATTRIBUTION_RE = re.compile(
    r"^(?:the\s+)?(?:staff|participants?|members?|market\s+participants|respondents)\b",
    re.IGNORECASE,
)
_OPENING_CONTRAST_RE = re.compile(
    r"^(?:although|however|while|in\s+contrast|on\s+balance)\b",
    re.IGNORECASE,
)

_OPENING_ORDER = ("evidence_first", "attribution_first", "contrast_first")
_SENTENCE_SHAPE_ORDER = ("balanced", "compact", "extended")
_STYLE_MOVE_ORDER = tuple(name for name, _ in _STYLE_MOVE_PATTERNS)
_LEXICAL_MARKER_ORDER = tuple(name for name, _ in _LEXICAL_MARKERS)


# These aliases cover the section variants already present in the repository.
# They are applied to an internal canonical key; raw section names never enter
# the returned artifact.
_SECTION_ALIASES = {
    "participants view on current conditions and the economic outlook": (
        "participants views on current conditions and the economic outlook"
    ),
    "participants views on current economic conditions and the economic outlook": (
        "participants views on current conditions and the economic outlook"
    ),
    "participants views on current conditions and economic outlook": (
        "participants views on current conditions and the economic outlook"
    ),
    "participants views of current conditions and the economic outlook": (
        "participants views on current conditions and the economic outlook"
    ),
    "staff review of financial situation": "staff review of the financial situation",
    "staff review of economic situation": "staff review of the economic situation",
}

_FORBIDDEN_GUIDE_RE = re.compile(
    r"\d|"
    r"\b(?:" + "|".join(_MONTH_NAMES) + r")\b|"
    r"\b(?:Mr|Mrs|Ms|Mses|Messrs|Dr)\.|"
    r"\b(?:voted|decided|approved|authorized|reaffirmed|adopted)\b|"
    r"\bunanimous\s+vote\b",
    re.IGNORECASE,
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _validate_sha256(value: str, *, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise StyleGuideError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _normalize_unicode(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    replacements = {
        "\u00a0": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "â": "-",
        "â": "'",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return re.sub(r"\s+", " ", normalized).strip()


def _canonical_section_key(section_name: str) -> str:
    if not isinstance(section_name, str) or not section_name.strip():
        raise StyleGuideError("section_name must be a non-empty string")
    normalized = _normalize_unicode(section_name)
    normalized = _PERSON_REFERENCE_RE.sub(" named person ", normalized)
    lowered = normalized.casefold().replace("&", " and ")
    lowered = _MONTH_RE.sub(" date ", lowered)
    lowered = _DATE_RE.sub(" date ", lowered)
    lowered = _NUMERIC_RE.sub(" value ", lowered)
    key = re.sub(r"[^a-z0-9]+", " ", lowered)
    key = re.sub(r"\s+", " ", key).strip()
    if not key:
        # The current processed Minutes schema contains a small number of
        # punctuation-only labels (notably "," and "-").  They carry no safe
        # section identity, so map them to one generic bucket rather than
        # embedding their source spelling or inventing a per-row identifier.
        key = "unclassified section"

    # Meeting/date headings are administrative containers, not distinct style
    # identities.  Collapsing them prevents an opaque ID from becoming a
    # one-meeting proxy.
    if (
        "videoconference meeting" in key
        or key.startswith("date ")
        or key in {"minutes of the federal open market committee", "conference call"}
    ):
        key = "meeting administration"
    return _SECTION_ALIASES.get(key, key)


def stable_section_style_id(section_name: str) -> str:
    """Return a deterministic opaque ID for a canonical Minutes section.

    The clear-text section name is deliberately not embedded in the identifier.
    Normalization is Unicode-, case-, punctuation-, and whitespace-stable and
    includes aliases already used by the repository's Minutes schema.
    """

    key = _canonical_section_key(section_name)
    digest = _sha256_text(f"{ABSTRACTION_VERSION}\0{key}")[:20]
    return f"{STYLE_ID_PREFIX}{digest}"


# Friendly aliases for downstream builders.
section_style_id = stable_section_style_id
get_section_style_id = stable_section_style_id


def _source_sentences(text: str) -> list[str]:
    normalized = _normalize_unicode(text)
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(normalized) if part.strip()]


def _unsafe_sentence_reason(sentence: str) -> str | None:
    if _POLICY_OUTCOME_RE.search(sentence) or _POLICY_PREFERENCE_RE.search(sentence):
        return "policy_outcome"
    if _MEETING_FACT_RE.search(sentence):
        return "meeting_fact"
    if _PERSON_REFERENCE_RE.search(sentence):
        return "person_reference"
    return None


def _sentence_shape(sentences: Sequence[str]) -> str:
    lengths: list[int] = []
    for sentence in sentences:
        scrubbed = _MONTH_RE.sub(" time ", sentence)
        scrubbed = _DATE_RE.sub(" time ", scrubbed)
        scrubbed = _NUMERIC_RE.sub(" value ", scrubbed)
        scrubbed = _PERSON_REFERENCE_RE.sub(" source ", scrubbed)
        lengths.append(len(_WORD_RE.findall(scrubbed)))
    if not lengths:
        return "balanced"
    ordered = sorted(lengths)
    middle = len(ordered) // 2
    median_length = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    if median_length <= 16:
        return "compact"
    if median_length <= 32:
        return "balanced"
    return "extended"


def _opening_style(sentences: Sequence[str]) -> str:
    if not sentences:
        return "evidence_first"
    opening = sentences[0].lstrip(" '\"([{-")
    if _OPENING_ATTRIBUTION_RE.search(opening):
        return "attribution_first"
    if _OPENING_CONTRAST_RE.search(opening):
        return "contrast_first"
    return "evidence_first"


def abstract_minutes_style(text: str) -> dict[str, Any]:
    """Reduce Minutes prose to controlled style atoms.

    No substring taken from ``text`` is returned.  Source sentences containing
    people, administrative meeting facts, or policy outcomes are excluded before
    style classification.  Numbers and dates are replaced before sentence-shape
    measurement.  The remaining output values all come from module constants.
    """

    if not isinstance(text, str) or not text.strip():
        raise StyleGuideError("reference_excerpt must be a non-empty string")

    sentences = _source_sentences(text)
    excluded: Counter[str] = Counter()
    redacted: Counter[str] = Counter()
    usable: list[str] = []
    for sentence in sentences:
        reason = _unsafe_sentence_reason(sentence)
        if reason is not None:
            excluded[reason] += 1
            continue
        if _NUMERIC_RE.search(sentence):
            redacted["numeric"] += 1
        if _DATE_RE.search(sentence) or _MONTH_RE.search(sentence):
            redacted["date"] += 1
        usable.append(sentence)

    searchable = " ".join(usable)
    moves = [name for name, pattern in _STYLE_MOVE_PATTERNS if pattern.search(searchable)]
    markers = [name for name, pattern in _LEXICAL_MARKERS if pattern.search(searchable)]
    opening = _opening_style(usable)
    shape = _sentence_shape(usable)

    atoms = [f"opening_{opening}", f"sentence_shape_{shape}"]
    atoms.extend(f"move_{name}" for name in moves)
    if not usable:
        atoms.append("fallback_neutral")
    abstracted_text = "; ".join(atoms)
    if re.search(r"\d", abstracted_text):
        raise AssertionError("controlled style abstraction unexpectedly contains a digit")

    return {
        "abstracted_text": abstracted_text,
        "style_atoms": atoms,
        "opening": opening,
        "sentence_shape": shape,
        "moves": moves,
        "lexical_markers": markers,
        "source_sentence_count": len(sentences),
        "usable_sentence_count": len(usable),
        "excluded_sentence_counts": dict(sorted(excluded.items())),
        "redacted_sentence_counts": dict(sorted(redacted.items())),
    }


def abstract_minutes_text(text: str) -> str:
    """Return only the controlled, fact-free abstraction string for ``text``."""

    return str(abstract_minutes_style(text)["abstracted_text"])


def _choose_majority(values: Iterable[str], order: Sequence[str]) -> str:
    counts = Counter(values)
    if not counts:
        return order[0]
    rank = {value: index for index, value in enumerate(order)}
    return max(order, key=lambda value: (counts[value], -rank[value]))


def _prevalent_values(
    signatures: Sequence[Mapping[str, Any]],
    *,
    field: str,
    order: Sequence[str],
) -> list[str]:
    usable = [item for item in signatures if int(item["usable_sentence_count"]) > 0]
    if not usable:
        return []
    hits: Counter[str] = Counter()
    for signature in usable:
        hits.update(set(str(value) for value in signature[field]))
    # A marker/move must occur in at least one third of usable excerpts.  The
    # integer comparison avoids floating-point/version-dependent boundaries.
    return [value for value in order if hits[value] * 3 >= len(usable)]


def _join_words(values: Sequence[str]) -> str:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _render_guide(style_signature: Mapping[str, Any]) -> str:
    sentences = ["Use neutral third-person prose in one coherent paragraph."]
    opening = style_signature["opening"]
    if opening == "attribution_first":
        sentences.append(
            "Open with attribution to a generic source group, then state the supporting evidence."
        )
    elif opening == "contrast_first":
        sentences.append(
            "Open with the principal contrast, then organize the supporting evidence."
        )
    else:
        sentences.append(
            "Open with an aggregate evidence-based observation before supporting detail."
        )

    shape = style_signature["sentence_shape"]
    if shape == "compact":
        sentences.append("Prefer compact declarative sentences.")
    elif shape == "extended":
        sentences.append(
            "Use measured multi-clause sentences while keeping each evidentiary link explicit."
        )
    else:
        sentences.append("Use balanced sentences with limited subordinate clauses.")

    moves = set(style_signature["moves"])
    if "contrast" in moves:
        sentences.append("Use restrained contrast to separate competing signals.")
    if "causal_link" in moves:
        sentences.append(
            "Use causal phrasing only when the fact-card evidence explicitly states the "
            "relationship; otherwise describe association or sequence without causation."
        )
    if "temporal_comparison" in moves:
        sentences.append("Order observations from recent movement to broader comparison.")
    if "uncertainty" in moves or "risk_balance" in moves:
        sentences.append("Qualify uncertainty and risks with calibrated modal language.")
    if "group_quantification" in moves:
        sentences.append("Use generic group quantifiers instead of naming individuals.")
    if "source_attribution" in moves:
        sentences.append("Keep source attribution generic and institutionally neutral.")

    markers = list(style_signature["lexical_markers"])
    if markers:
        sentences.append(
            "Preferred generic discourse markers are " + _join_words(markers) + "."
        )

    sentences.append("This style guidance supplies no facts.")
    sentences.append(
        "In generated prose, use numerical values and dates only when they appear in "
        "fact-card evidence."
    )
    sentences.append(
        "Do not introduce named individuals, meeting-specific events, or action outcomes."
    )
    guide = " ".join(sentences)
    _validate_guide_text(guide)
    return guide


def _validate_guide_text(guide_text: str) -> None:
    if _FORBIDDEN_GUIDE_RE.search(guide_text):
        raise StyleGuideError("rendered guide contains a prohibited factual surface form")


def _require_row(row: Mapping[str, Any], *, row_number: int) -> dict[str, Any]:
    missing = [field for field in REQUIRED_ROW_FIELDS if field not in row]
    if missing:
        raise StyleGuideError(
            f"Minutes row {row_number} is missing required fields: {', '.join(missing)}"
        )

    split = row["split"]
    if split != REQUIRED_SPLIT:
        raise StyleGuideError(
            f"style corpus only accepts split='train'; row {row_number} has {split!r}"
        )

    sample_id = row["sample_id"]
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise StyleGuideError(f"Minutes row {row_number} has an invalid sample_id")

    meeting_date = row["meeting_date"]
    if not isinstance(meeting_date, str) or not meeting_date.strip():
        raise StyleGuideError(f"Minutes row {row_number} has an invalid meeting_date")
    try:
        parsed_date = date.fromisoformat(meeting_date)
    except ValueError as exc:
        raise StyleGuideError(
            f"Minutes row {row_number} meeting_date must be ISO YYYY-MM-DD"
        ) from exc
    if parsed_date.isoformat() != meeting_date:
        raise StyleGuideError(
            f"Minutes row {row_number} meeting_date must be canonical ISO YYYY-MM-DD"
        )

    section_name = row["section_name"]
    if not isinstance(section_name, str) or not section_name.strip():
        raise StyleGuideError(f"Minutes row {row_number} has an invalid section_name")

    source_row_index = row["source_row_index"]
    if isinstance(source_row_index, bool) or not isinstance(source_row_index, int):
        raise StyleGuideError(
            f"Minutes row {row_number} source_row_index must be an integer"
        )
    if source_row_index < 0:
        raise StyleGuideError(
            f"Minutes row {row_number} source_row_index must be non-negative"
        )

    excerpt = row[SOURCE_TEXT_FIELD]
    if not isinstance(excerpt, str) or not excerpt.strip():
        raise StyleGuideError(
            f"Minutes row {row_number} {SOURCE_TEXT_FIELD} must be non-empty"
        )

    return {
        "sample_id": sample_id.strip(),
        "split": split,
        "meeting_date": meeting_date,
        "section_name": section_name,
        "source_row_index": source_row_index,
        SOURCE_TEXT_FIELD: excerpt,
    }


def _aggregate_section(
    *,
    style_id: str,
    section_key_sha256: str,
    signatures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    opening = _choose_majority(
        (str(item["opening"]) for item in signatures), _OPENING_ORDER
    )
    shape = _choose_majority(
        (str(item["sentence_shape"]) for item in signatures), _SENTENCE_SHAPE_ORDER
    )
    moves = _prevalent_values(signatures, field="moves", order=_STYLE_MOVE_ORDER)
    markers = _prevalent_values(
        signatures,
        field="lexical_markers",
        order=_LEXICAL_MARKER_ORDER,
    )
    excluded: Counter[str] = Counter()
    redacted: Counter[str] = Counter()
    for signature in signatures:
        excluded.update(signature["excluded_sentence_counts"])
        redacted.update(signature["redacted_sentence_counts"])

    style_signature = {
        "opening": opening,
        "sentence_shape": shape,
        "moves": moves,
        "lexical_markers": markers,
    }
    guide_text = _render_guide(style_signature)
    entry: dict[str, Any] = {
        "section_style_id": style_id,
        "section_key_sha256": section_key_sha256,
        "corpus_row_count": len(signatures),
        "usable_row_count": sum(
            int(item["usable_sentence_count"]) > 0 for item in signatures
        ),
        "excluded_sentence_counts": dict(sorted(excluded.items())),
        "redacted_sentence_counts": dict(sorted(redacted.items())),
        "style_signature": style_signature,
        "guide_text": guide_text,
        "guide_text_sha256": _sha256_text(guide_text),
    }
    entry["style_entry_sha256"] = _sha256_json(entry)
    return entry


def compute_style_guide_sha256(artifact: Mapping[str, Any]) -> str:
    """Compute the artifact hash, excluding its self-referential hash field."""

    return _sha256_json(
        {key: value for key, value in artifact.items() if key != "style_guide_sha256"}
    )


def build_style_guide(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_artifact_sha256: str | None = None,
    source_artifact_bytes: int | None = None,
    source_id: str = "in-memory-train-minutes",
) -> dict[str, Any]:
    """Build deterministic section guides from train Minutes rows only.

    ``source_artifact_sha256`` is optional for in-memory callers.  File callers
    should use :func:`build_style_guide_from_jsonl`, which records both the raw
    byte hash and an order-independent canonical corpus hash.
    """

    materialized = list(rows)
    if not materialized:
        raise StyleGuideError("style corpus must contain at least one train row")
    if not isinstance(source_id, str) or not source_id.strip():
        raise StyleGuideError("source_id must be a non-empty stable identifier")
    if source_artifact_sha256 is not None:
        source_artifact_sha256 = _validate_sha256(
            source_artifact_sha256, label="source_artifact_sha256"
        )
    if source_artifact_bytes is not None:
        if isinstance(source_artifact_bytes, bool) or not isinstance(
            source_artifact_bytes, int
        ) or source_artifact_bytes < 0:
            raise StyleGuideError("source_artifact_bytes must be a non-negative integer")

    seen_sample_ids: set[str] = set()
    seen_row_keys: set[str] = set()
    section_keys_by_id: dict[str, str] = {}
    section_signatures: dict[str, list[dict[str, Any]]] = defaultdict(list)
    corpus_rows: list[dict[str, Any]] = []

    for row_number, raw_row in enumerate(materialized, 1):
        if not isinstance(raw_row, Mapping):
            raise StyleGuideError(f"Minutes row {row_number} must be a mapping")
        row = _require_row(raw_row, row_number=row_number)
        sample_id = row["sample_id"]
        if sample_id in seen_sample_ids:
            raise StyleGuideError(f"duplicate sample_id in style corpus: {sample_id!r}")
        seen_sample_ids.add(sample_id)

        section_key = _canonical_section_key(row["section_name"])
        style_id = stable_section_style_id(row["section_name"])
        previous_key = section_keys_by_id.setdefault(style_id, section_key)
        if previous_key != section_key:
            raise StyleGuideError(
                f"section_style_id collision detected for {style_id}; refusing corpus"
            )

        signature = abstract_minutes_style(row[SOURCE_TEXT_FIELD])
        section_signatures[style_id].append(signature)
        row_identity = {
            "sample_id": sample_id,
            "meeting_date": row["meeting_date"],
            "source_row_index": row["source_row_index"],
        }
        row_key_sha256 = _sha256_json(row_identity)
        if row_key_sha256 in seen_row_keys:
            raise StyleGuideError("duplicate canonical Minutes row key in style corpus")
        seen_row_keys.add(row_key_sha256)
        corpus_rows.append(
            {
                "row_key_sha256": row_key_sha256,
                "sample_id_sha256": _sha256_text(sample_id),
                "meeting_date_sha256": _sha256_text(row["meeting_date"]),
                "source_row_index_sha256": _sha256_text(
                    str(row["source_row_index"])
                ),
                "section_key_sha256": _sha256_text(section_key),
                "section_style_id": style_id,
                "source_text_sha256": _sha256_text(row[SOURCE_TEXT_FIELD]),
                "source_text_bytes": len(row[SOURCE_TEXT_FIELD].encode("utf-8")),
                "style_abstraction_sha256": _sha256_text(
                    signature["abstracted_text"]
                ),
                "usable_sentence_count": signature["usable_sentence_count"],
            }
        )

    corpus_rows.sort(key=lambda item: item["row_key_sha256"])
    styles = [
        _aggregate_section(
            style_id=style_id,
            section_key_sha256=_sha256_text(section_keys_by_id[style_id]),
            signatures=section_signatures[style_id],
        )
        for style_id in sorted(section_signatures)
    ]
    provenance: dict[str, Any] = {
        "required_split": REQUIRED_SPLIT,
        "input_schema_fields": list(REQUIRED_ROW_FIELDS),
        "source_text_field": SOURCE_TEXT_FIELD,
        "source_id_sha256": _sha256_text(source_id.strip()),
        "source_artifact_sha256": source_artifact_sha256,
        "source_artifact_bytes": source_artifact_bytes,
        "corpus_row_count": len(corpus_rows),
        "section_style_count": len(styles),
        "corpus_sha256": _sha256_json(corpus_rows),
        "corpus_rows": corpus_rows,
    }
    provenance["provenance_sha256"] = _sha256_json(provenance)
    artifact: dict[str, Any] = {
        "schema_version": STYLE_GUIDE_SCHEMA_VERSION,
        "abstraction_version": ABSTRACTION_VERSION,
        "required_split": REQUIRED_SPLIT,
        "provenance": provenance,
        "styles": styles,
    }
    artifact["style_guide_sha256"] = compute_style_guide_sha256(artifact)
    verify_style_guide_artifact(artifact)
    return artifact


def _path_declares_held_out_split(path: Path) -> bool:
    tokens = [token for token in re.split(r"[^a-z0-9]+", path.stem.casefold()) if token]
    return "eval" in tokens or "test" in tokens


def build_style_guide_from_jsonl(
    path: str | Path,
    *,
    expected_source_sha256: str | None = None,
    source_id: str = "minutes-train-jsonl",
) -> dict[str, Any]:
    """Read an existing train Minutes JSONL without writing any data files."""

    source_path = Path(path)
    if _path_declares_held_out_split(source_path):
        raise StyleGuideError(
            f"held-out eval/test source path is forbidden for style extraction: {source_path.name}"
        )
    if not source_path.is_file():
        raise FileNotFoundError(f"Minutes style source does not exist: {source_path}")
    source_bytes = source_path.read_bytes()
    source_sha256 = _sha256_bytes(source_bytes)
    if expected_source_sha256 is not None:
        expected = _validate_sha256(
            expected_source_sha256, label="expected_source_sha256"
        )
        if expected != source_sha256:
            raise StyleGuideError(
                "Minutes style source SHA-256 mismatch: "
                f"expected={expected}, observed={source_sha256}"
            )
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StyleGuideError("Minutes style source must be valid UTF-8") from exc

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source_text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StyleGuideError(
                f"invalid JSON in Minutes style source at line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise StyleGuideError(
                f"Minutes style source line {line_number} must be a JSON object"
            )
        rows.append(row)

    return build_style_guide(
        rows,
        source_artifact_sha256=source_sha256,
        source_artifact_bytes=len(source_bytes),
        source_id=source_id,
    )


def verify_style_guide_artifact(artifact: Mapping[str, Any]) -> bool:
    """Fail closed if a guide or any provenance hash has been altered."""

    if not isinstance(artifact, Mapping):
        raise StyleGuideError("style guide artifact must be a mapping")
    if artifact.get("schema_version") != STYLE_GUIDE_SCHEMA_VERSION:
        raise StyleGuideError("unexpected style guide schema_version")
    if artifact.get("abstraction_version") != ABSTRACTION_VERSION:
        raise StyleGuideError("unexpected style guide abstraction_version")
    if artifact.get("required_split") != REQUIRED_SPLIT:
        raise StyleGuideError("style guide artifact is not train-only")

    provenance = artifact.get("provenance")
    if not isinstance(provenance, Mapping):
        raise StyleGuideError("style guide provenance must be a mapping")
    corpus_rows = provenance.get("corpus_rows")
    if not isinstance(corpus_rows, list) or not corpus_rows:
        raise StyleGuideError("style guide corpus_rows must be a non-empty list")
    if provenance.get("corpus_sha256") != _sha256_json(corpus_rows):
        raise StyleGuideError("style guide corpus_sha256 mismatch")
    provenance_without_hash = {
        key: value for key, value in provenance.items() if key != "provenance_sha256"
    }
    if provenance.get("provenance_sha256") != _sha256_json(provenance_without_hash):
        raise StyleGuideError("style guide provenance_sha256 mismatch")

    styles = artifact.get("styles")
    if not isinstance(styles, list) or not styles:
        raise StyleGuideError("style guide styles must be a non-empty list")
    observed_ids: list[str] = []
    for entry in styles:
        if not isinstance(entry, Mapping):
            raise StyleGuideError("each style guide entry must be a mapping")
        style_id = entry.get("section_style_id")
        if not isinstance(style_id, str) or not style_id.startswith(STYLE_ID_PREFIX):
            raise StyleGuideError("invalid section_style_id in style guide artifact")
        observed_ids.append(style_id)
        guide_text = entry.get("guide_text")
        if not isinstance(guide_text, str):
            raise StyleGuideError("style guide entry is missing guide_text")
        _validate_guide_text(guide_text)
        if entry.get("guide_text_sha256") != _sha256_text(guide_text):
            raise StyleGuideError("guide_text_sha256 mismatch")
        entry_without_hash = {
            key: value for key, value in entry.items() if key != "style_entry_sha256"
        }
        if entry.get("style_entry_sha256") != _sha256_json(entry_without_hash):
            raise StyleGuideError("style_entry_sha256 mismatch")
    if observed_ids != sorted(set(observed_ids)):
        raise StyleGuideError("section style entries must be unique and sorted")

    if artifact.get("style_guide_sha256") != compute_style_guide_sha256(artifact):
        raise StyleGuideError("style_guide_sha256 mismatch")
    return True


# Plural aliases make the multi-section nature explicit for callers that prefer
# that vocabulary.
build_style_guides = build_style_guide
build_style_guides_from_jsonl = build_style_guide_from_jsonl
build_section_style_guides = build_style_guide


__all__ = [
    "ABSTRACTION_VERSION",
    "REQUIRED_ROW_FIELDS",
    "REQUIRED_SPLIT",
    "SOURCE_TEXT_FIELD",
    "STYLE_GUIDE_SCHEMA_VERSION",
    "StyleGuideError",
    "abstract_minutes_style",
    "abstract_minutes_text",
    "build_section_style_guides",
    "build_style_guide",
    "build_style_guide_from_jsonl",
    "build_style_guides",
    "build_style_guides_from_jsonl",
    "compute_style_guide_sha256",
    "get_section_style_id",
    "section_style_id",
    "stable_section_style_id",
    "verify_style_guide_artifact",
]
