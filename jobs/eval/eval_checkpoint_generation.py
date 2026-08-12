"""Provenance-aware evaluation of checkpoint generations.

The evaluator is deliberately generation-inference-free.  It consumes completed
generation, reference, and prompt/evidence-fact JSONL artifacts and writes auditable
row-level and aggregate results.  Optional semantic scorers are dependency
injected; the bundled BERTScore and MPNet adapters require an already-downloaded
local model with a pinned checksum.

Canonical generation rows are keyed by ``(artifact_id, sample_id)``.  Reference
and evidence rows are keyed by ``sample_id``.  The default policy requires four
artifacts and an identical sample universe for every input, so a missing row can
never disappear through an inner join.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from open_r1.structured_response import extract_length_tolerant_candidate
from open_r1.validator.loo_generation_spec import (
    derive_row_seed,
    seal_manifest,
    validate_manifest_integrity,
)
from open_r1.validator.text_leakage import (
    validate_no_prompt_reference_token_overlap,
)


SCHEMA_VERSION = "checkpoint-generation-evaluation-v1"
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_260_729
LONG_TEXT_POLICY = "sentence-boundary-chunk-weighted-v1"
FORMAL_ROW_SEED_POLICY = "sample-id-sha256-v1"
SEMANTIC_MANIFEST_SCHEMA_VERSION = "checkpoint-eval-semantic-model-manifest-v1"
LINEAGE_EVIDENCE_SCHEMA_VERSION = "lora-merge-lineage-evidence-v1"
FORMAL_TEST_SAMPLE_COUNT = 33
FORMAL_PROSPECTIVE_SAMPLE_COUNT = 27
FORMAL_MEETING_COUNT = 11
FORMAL_PROSPECTIVE_MEETING_COUNT = 9
FORMAL_SECTION_COUNT = 3
STRICT_SCORING_POLICY = "strict-final-answer-v2"
LENGTH_TOLERANT_SCORING_POLICY = "length-tolerant-open-tags-v1"
SCORING_POLICIES = (
    STRICT_SCORING_POLICY,
    LENGTH_TOLERANT_SCORING_POLICY,
)

REQUIRED_PROVENANCE_FIELDS = (
    "checkpoint_sha256",
    "test_set_sha256",
    "prompt_template_sha256",
    "decoding_config_sha256",
)
COMMON_PROVENANCE_FIELDS = (
    "test_set_sha256",
    "prompt_template_sha256",
    "decoding_config_sha256",
)
PER_SAMPLE_COMPARABILITY_FIELDS = (
    "source_prompt_sha256",
    "generation_seed",
    "temperature",
    "top_p",
    "max_new_tokens",
)

METRIC_DIRECTIONS: dict[str, str] = {
    "valid_output": "higher",
    "rouge_l_f1": "higher",
    "generated_token_count": "descriptive",
    "generated_character_count": "descriptive",
    "generated_sentence_count": "descriptive",
    "length_ratio": "descriptive",
    "repetition_rate": "lower",
    "format_compliance": "higher",
    "numeric_value_accuracy": "higher",
    "evidence_value_coverage": "higher",
    "unit_accuracy": "higher",
    "time_accuracy": "higher",
    "novel_number_rate": "lower",
    "rule_covered_unsupported_rate": "lower",
    "direction_consistency": "higher",
    "direction_coverage": "higher",
    "policy_stance_coverage": "higher",
    "policy_stance_consistency": "higher",
}

TEXT_FIELD_ALIASES = {
    "generation": ("final_answer", "generated", "generation", "output", "text"),
    "reference": ("reference", "target", "expected", "text"),
}

NUMBER_RE = re.compile(r"(?<!\w)[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w)")
TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|[-+]?\d+(?:\.\d+)?")
SENTENCE_RE = re.compile(r"[^.!?;\n]+(?:[.!?;]|$)")
YEAR_RANGE_RE = re.compile(
    r"\b(?:19|20)\d{2}\s*(?:-|–|—|to|through)\s*(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
ISO_DATE_RANGE_RE = re.compile(
    r"\b(?:19|20)\d{2}-\d{2}-\d{2}\s*(?:to|through|–|—)\s*"
    r"(?:19|20)\d{2}-\d{2}-\d{2}\b",
    re.IGNORECASE,
)
ISO_DATE_RE = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")
QUARTER_RE = re.compile(
    r"\b(?:Q[1-4]\s*(?:19|20)\d{2}|(?:19|20)\d{2}\s*Q[1-4])\b",
    re.IGNORECASE,
)
MONTH_YEAR_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
RELATIVE_TIME_RE = re.compile(
    r"\b(?:over|during|in)\s+the\s+(?:past|previous|prior|last)\s+"
    r"(?:month|quarter|year|twelve months)\b|\b(?:month|quarter|year)[ -]over[ -]"
    r"(?:month|quarter|year)\b",
    re.IGNORECASE,
)
FORMAL_BODY_ONLY_FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "reasoning_or_answer_tag",
        re.compile(r"<\s*/?\s*(?:think|answer)\s*>", re.IGNORECASE),
    ),
    (
        "d1_evidence_delimiter",
        re.compile(
            r"<<<\s*(?:BEGIN|END)\s*-\s*D1\s*-\s*EVIDENCE\s*>>>",
            re.IGNORECASE,
        ),
    ),
    (
        "participants_views_section_heading",
        re.compile(
            r"participants[’']?\s+views\s+on\s+current\s+conditions\s+and\s+"
            r"the\s+economic\s+outlook",
            re.IGNORECASE,
        ),
    ),
    (
        "economic_situation_section_heading",
        re.compile(r"staff\s+review\s+of\s+the\s+economic\s+situation", re.IGNORECASE),
    ),
    (
        "financial_situation_section_heading",
        re.compile(r"staff\s+review\s+of\s+the\s+financial\s+situation", re.IGNORECASE),
    ),
)

UNIT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "percentage_point",
        re.compile(
            r"^\s*(?:percentage\s+points?|percent\s+points?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "percent",
        re.compile(r"^\s*(?:%(?=\s|$)|percent(?:age)?\b)", re.IGNORECASE),
    ),
    ("basis_point", re.compile(r"^\s*(?:bp|bps|basis\s+points?)\b", re.IGNORECASE)),
    (
        "yuan_per_dollar",
        re.compile(
            r"^\s*(?:Chinese\s+)?yuan\s+per\s+(?:U\.?S\.?\s+)?dollars?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "yen_per_dollar",
        re.compile(
            r"^\s*(?:Japanese\s+)?yen\s+per\s+(?:U\.?S\.?\s+)?dollars?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "dollar_per_euro",
        re.compile(
            r"^\s*(?:U\.?S\.?\s+)?dollars?\s+per\s+euros?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "dollar_per_pound",
        re.compile(
            r"^\s*(?:U\.?S\.?\s+)?dollars?\s+per\s+pounds?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "dollar_per_barrel",
        re.compile(r"^\s*dollars?\s+per\s+barrels?\b", re.IGNORECASE),
    ),
    ("trillion", re.compile(r"^\s*trillion\b", re.IGNORECASE)),
    ("billion", re.compile(r"^\s*billion\b", re.IGNORECASE)),
    ("million", re.compile(r"^\s*million\b", re.IGNORECASE)),
    ("thousand", re.compile(r"^\s*thousand\b", re.IGNORECASE)),
    ("dollar", re.compile(r"^\s*dollars?\b", re.IGNORECASE)),
    ("index_point", re.compile(r"^\s*(?:index\s+)?points?\b", re.IGNORECASE)),
)

TOPIC_ALIASES: dict[str, tuple[str, ...]] = {
    "inflation": (
        "inflation",
        "price pressure",
        "price pressures",
        "consumer prices",
        "pce prices",
        "cpi",
    ),
    "growth": (
        "growth",
        "economic activity",
        "real activity",
        "output",
        "gdp",
        "economy",
    ),
    "employment": (
        "employment",
        "labor market",
        "labour market",
        "payrolls",
        "job gains",
        "jobs",
    ),
    "unemployment": ("unemployment", "jobless rate"),
}
CORE_DIRECTION_TOPICS = frozenset({"inflation", "growth", "employment", "unemployment"})

DIRECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "up": re.compile(
        r"\b(?:rise|rises|rose|risen|rising|increase|increases|increased|"
        r"increasing|accelerate|accelerates|accelerated|accelerating|strengthen|"
        r"strengthens|strengthened|strengthening|expand|expands|expanded|"
        r"expanding|improve|improves|improved|improving|higher|upward)\b",
        re.IGNORECASE,
    ),
    "down": re.compile(
        r"\b(?:fall|falls|fell|fallen|falling|decline|declines|declined|"
        r"declining|decrease|decreases|decreased|decreasing|decelerate|"
        r"decelerates|decelerated|decelerating|slow|slows|slowed|slowing|"
        r"weaken|weakens|weakened|weakening|contract|contracts|contracted|"
        r"contracting|deteriorate|deteriorates|deteriorated|deteriorating|"
        r"lower|downward)\b",
        re.IGNORECASE,
    ),
    "flat": re.compile(
        r"\b(?:stable|stabilized|steady|unchanged|flat|held constant|"
        r"little changed|no change|plateaued)\b",
        re.IGNORECASE,
    ),
}

STANCE_PATTERNS: dict[str, re.Pattern[str]] = {
    "hawkish": re.compile(
        r"\b(?:hawkish|tighten|tightening|restrictive|raise rates?|rate hikes?|"
        r"hike rates?|higher policy rate|additional firming)\b",
        re.IGNORECASE,
    ),
    "dovish": re.compile(
        r"\b(?:dovish|ease|easing|accommodative|cut rates?|rate cuts?|"
        r"lower policy rate|policy accommodation)\b",
        re.IGNORECASE,
    ),
    "neutral": re.compile(
        r"\b(?:neutral stance|balanced stance|hold rates?|keep rates? unchanged|"
        r"maintain the (?:current )?(?:target )?rate|wait-and-see)\b",
        re.IGNORECASE,
    ),
}


class SemanticScorer(Protocol):
    """Minimal protocol for an injected, batched semantic scorer."""

    scorer_id: str
    model_id: str
    model_sha256: str

    def score(
        self,
        candidates: Sequence[str],
        references: Sequence[str],
    ) -> Mapping[str, Sequence[float]] | Sequence[float]:
        """Return one or more metric vectors in input order."""


@dataclass(frozen=True)
class NumberMention:
    raw: str
    value: Decimal
    unit: str | None
    start: int
    end: int
    is_time: bool


@dataclass(frozen=True)
class TimeMention:
    raw: str
    normalized: str
    start: int
    end: int


@dataclass(frozen=True)
class EvidenceFact:
    fact_id: str
    value: Decimal | None
    unit: str | None
    time_range: str | None
    topic: str | None
    direction: str | None
    kind: str | None
    series_id: str | None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    """Hash a file or a directory tree deterministically."""

    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise ValueError(f"Frozen model path does not exist: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Frozen model directory contains no files: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{field} must be a 64-character hexadecimal SHA-256")
    return normalized


def _read_jsonl(
    path: Path, *, kind: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"{kind} JSONL does not exist: {path}")
    rows: list[dict[str, Any]] = []
    blank_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                blank_lines += 1
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid {kind} JSON on line {line_number} of {path}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"{kind} line {line_number} of {path} must be a JSON object"
                )
            row = dict(row)
            row["_source_path"] = str(path)
            row["_source_line"] = line_number
            rows.append(row)
    if not rows:
        raise ValueError(f"{kind} JSONL contains no records: {path}")
    return rows, {
        "path": str(path),
        "sha256": _sha256_file(path),
        "row_count": len(rows),
        "blank_line_count": blank_lines,
    }


def _read_json_object(
    path: Path,
    *,
    kind: str,
    require_integrity: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"{kind} JSON does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {kind} JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{kind} must be a JSON object: {path}")
    payload_sha256 = None
    if require_integrity:
        try:
            payload_sha256 = validate_manifest_integrity(payload)
        except Exception as exc:
            raise ValueError(
                f"{kind} integrity validation failed: {path}: {exc}"
            ) from exc
    return payload, {
        "path": str(path),
        "sha256": _sha256_file(path),
        "payload_sha256": payload_sha256,
    }


def _nonblank(value: Any) -> str:
    return str(value or "").strip()


def _sample_id(row: Mapping[str, Any], *, kind: str) -> str:
    value = _nonblank(row.get("sample_id"))
    if not value:
        raise ValueError(
            f"{kind} row {row.get('_source_path')}:{row.get('_source_line')} "
            "has no non-empty sample_id"
        )
    return value


def _artifact_id(row: Mapping[str, Any]) -> str:
    nested = row.get("provenance")
    provenance = nested if isinstance(nested, Mapping) else {}
    values = [
        row.get("artifact_id"),
        row.get("checkpoint_id"),
        provenance.get("artifact_id"),
        provenance.get("checkpoint_id"),
    ]
    distinct = {_nonblank(value) for value in values if _nonblank(value)}
    if not distinct:
        raise ValueError(
            f"Generation row {row.get('_source_path')}:{row.get('_source_line')} "
            "has no artifact_id"
        )
    if len(distinct) != 1:
        raise ValueError(
            f"Generation row has conflicting artifact identifiers: {sorted(distinct)}"
        )
    return distinct.pop()


def _select_text(row: Mapping[str, Any], *, kind: str) -> str:
    aliases = TEXT_FIELD_ALIASES[kind]
    if kind == "generation" and row.get("final_answer") is not None:
        # The canonical generator preserves both the raw completion and the
        # parsed final answer.  Evaluation is intentionally final-answer-only.
        return str(row["final_answer"])
    present = [(field, row.get(field)) for field in aliases if field in row]
    non_null = [(field, str(value)) for field, value in present if value is not None]
    if not non_null:
        raise ValueError(
            f"{kind.title()} row {row.get('_source_path')}:{row.get('_source_line')} "
            f"must contain one of {list(aliases)}"
        )
    distinct = {value for _, value in non_null}
    if len(distinct) > 1:
        raise ValueError(
            f"{kind.title()} row has conflicting text aliases: "
            f"{[field for field, _ in non_null]}"
        )
    return non_null[0][1]


def _select_raw_generation_completion(row: Mapping[str, Any]) -> str:
    """Select the model completion before any strict final-answer parsing."""

    for field in ("generated", "generation", "output", "text", "final_answer"):
        if field in row and row.get(field) is not None:
            return str(row[field])
    raise ValueError(
        f"Generation row {row.get('_source_path')}:{row.get('_source_line')} "
        "has no raw completion or final_answer"
    )


def _index_unique(
    rows: Sequence[dict[str, Any]],
    *,
    kind: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        key = _sample_id(row, kind=kind)
        if key in indexed:
            duplicates[key].extend(
                [
                    f"{indexed[key].get('_source_path')}:{indexed[key].get('_source_line')}",
                    f"{row.get('_source_path')}:{row.get('_source_line')}",
                ]
            )
        else:
            indexed[key] = row
    if duplicates:
        details = {key: sorted(set(locations)) for key, locations in duplicates.items()}
        raise ValueError(f"Duplicate {kind} sample_id values: {details}")
    return indexed


def _index_generations(
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, set[str]]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    by_artifact: dict[str, set[str]] = defaultdict(set)
    duplicates: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        artifact_id = _artifact_id(row)
        sample_id = _sample_id(row, kind="generation")
        key = (artifact_id, sample_id)
        if key in indexed:
            duplicates[key].extend(
                [
                    f"{indexed[key].get('_source_path')}:{indexed[key].get('_source_line')}",
                    f"{row.get('_source_path')}:{row.get('_source_line')}",
                ]
            )
            continue
        indexed[key] = row
        by_artifact[artifact_id].add(sample_id)
    if duplicates:
        details = {
            f"{artifact_id}::{sample_id}": sorted(set(locations))
            for (artifact_id, sample_id), locations in duplicates.items()
        }
        raise ValueError(
            f"Duplicate generation (artifact_id, sample_id) keys: {details}"
        )
    return indexed, dict(by_artifact)


def _provenance_value(row: Mapping[str, Any], field: str) -> Any:
    nested = row.get("provenance")
    provenance = nested if isinstance(nested, Mapping) else {}
    aliases: dict[str, tuple[str, ...]] = {
        "checkpoint_sha256": (
            "checkpoint_sha256",
            "artifact_sha256",
            "model_sha256",
            "generation_model_sha256",
        ),
        "parent_artifact_id": (
            "parent_artifact_id",
            "parent_checkpoint_id",
            "verified_parent_artifact_id",
            "parent",
        ),
        "test_set_sha256": ("test_set_sha256", "data_manifest_sha256"),
        "prompt_template_sha256": (
            "prompt_template_sha256",
            "prompt_config_sha256",
            "evaluation_config_sha256",
        ),
        "decoding_config_sha256": (
            "decoding_config_sha256",
            "generation_config_sha256",
            "evaluation_config_sha256",
        ),
        "reference_sha256": ("reference_sha256",),
        "evidence_sha256": ("evidence_sha256",),
        "source_prompt_sha256": ("source_prompt_sha256", "prompt_sha256"),
    }
    names = aliases.get(field, (field,))
    values: list[Any] = []
    for name in names:
        if name in row and row[name] is not None:
            values.append(row[name])
        if name in provenance and provenance[name] is not None:
            values.append(provenance[name])
    normalized = {_canonical_scalar(value) for value in values}
    if len(normalized) > 1:
        raise ValueError(f"Generation row has conflicting {field} values: {values}")
    return values[0] if values else None


def _canonical_scalar(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    return str(value).strip()


def _collect_provenance(
    generation_index: Mapping[tuple[str, str], Mapping[str, Any]],
    artifact_ids: Sequence[str],
    *,
    require_provenance: bool,
    reference_sha256: str,
    evidence_sha256: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    stable_fields = (
        *REQUIRED_PROVENANCE_FIELDS,
        "parent_artifact_id",
    )
    for artifact_id in artifact_ids:
        rows = [
            row
            for (row_artifact_id, _), row in generation_index.items()
            if row_artifact_id == artifact_id
        ]
        provenance: dict[str, Any] = {}
        missing: list[str] = []
        for field in stable_fields:
            values = {
                _canonical_scalar(value)
                for row in rows
                if (value := _provenance_value(row, field)) not in (None, "")
            }
            if len(values) > 1:
                raise ValueError(
                    f"Artifact {artifact_id!r} has non-constant provenance field "
                    f"{field!r}: {sorted(values)}"
                )
            provenance[field] = next(iter(values)) if values else None
            if (
                require_provenance
                and field in REQUIRED_PROVENANCE_FIELDS
                and not values
            ):
                missing.append(field)
        if missing:
            raise ValueError(
                f"Artifact {artifact_id!r} is missing required provenance fields: {missing}"
            )
        if provenance.get("checkpoint_sha256"):
            provenance["checkpoint_sha256"] = _validate_sha256(
                provenance["checkpoint_sha256"],
                field=f"{artifact_id}.checkpoint_sha256",
            )
        for field in COMMON_PROVENANCE_FIELDS:
            if provenance.get(field):
                provenance[field] = _validate_sha256(
                    provenance[field],
                    field=f"{artifact_id}.{field}",
                )
        # Reference and prompt/evidence-file hashes are bound by the evaluator
        # (and, in formal mode, sealed manifests), rather than inferred from
        # per-sample ``reference_sha256``/``evidence_sha256`` fields.
        provenance["evaluation_reference_file_sha256"] = reference_sha256
        provenance["evaluation_prompts_file_sha256"] = evidence_sha256
        result[artifact_id] = provenance

    for field in COMMON_PROVENANCE_FIELDS:
        values = {
            row[field] for row in result.values() if row.get(field) not in (None, "")
        }
        if len(values) > 1:
            raise ValueError(
                f"Artifacts are not comparable: {field} differs across artifacts: "
                f"{sorted(values)}"
            )
        if require_provenance and len(values) != 1:
            raise ValueError(f"No common {field} was declared by all artifacts")

    checkpoint_hash_owners: dict[str, list[str]] = defaultdict(list)
    for artifact_id, row in result.items():
        checkpoint_hash = _nonblank(row.get("checkpoint_sha256"))
        if checkpoint_hash:
            checkpoint_hash_owners[checkpoint_hash].append(artifact_id)
    duplicate_hashes = {
        checkpoint_hash: sorted(owners)
        for checkpoint_hash, owners in checkpoint_hash_owners.items()
        if len(owners) > 1
    }
    if duplicate_hashes:
        raise ValueError(
            "Distinct artifact_id values resolve to identical checkpoint_sha256 "
            f"values: {duplicate_hashes}"
        )
    return result


def _meeting_id(
    reference_row: Mapping[str, Any] | None,
    evidence_row: Mapping[str, Any] | None,
    generation_row: Mapping[str, Any] | None,
    *,
    sample_id: str,
) -> str:
    values: list[tuple[str, str]] = []
    for label, row in (
        ("reference", reference_row),
        ("evidence", evidence_row),
        ("generation", generation_row),
    ):
        if not row:
            continue
        value = _nonblank(row.get("meeting_id") or row.get("meeting_date"))
        if value:
            values.append((label, value))
    distinct = {value for _, value in values}
    if len(distinct) > 1:
        raise ValueError(
            f"sample_id {sample_id!r} has inconsistent meeting identifiers: {values}"
        )
    if not distinct:
        raise ValueError(
            f"sample_id {sample_id!r} has no meeting_id or meeting_date in any input"
        )
    return distinct.pop()


def align_artifacts(
    generation_rows: Sequence[dict[str, Any]],
    reference_rows: Sequence[dict[str, Any]],
    evidence_rows: Sequence[dict[str, Any]],
    *,
    expected_artifact_count: int | None = 4,
    required_artifact_ids: Sequence[str] | None = None,
    missing_policy: str = "error",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Outer-align all inputs and make every missing key explicit.

    ``missing_policy="error"`` is the canonical mode.  ``"record"`` retains
    missing cells as rows with an explicit status for forensic audits.
    """

    if missing_policy not in {"error", "record"}:
        raise ValueError("missing_policy must be 'error' or 'record'")
    generation_index, generation_sets = _index_generations(generation_rows)
    reference_index = _index_unique(reference_rows, kind="reference")
    evidence_index = _index_unique(evidence_rows, kind="evidence")
    artifact_ids = sorted(generation_sets)
    if (
        expected_artifact_count is not None
        and len(artifact_ids) != expected_artifact_count
    ):
        raise ValueError(
            f"Expected exactly {expected_artifact_count} generation artifacts, "
            f"found {len(artifact_ids)}: {artifact_ids}"
        )
    if required_artifact_ids is not None:
        required = sorted({_nonblank(value) for value in required_artifact_ids})
        if artifact_ids != required:
            raise ValueError(
                f"Generation artifact IDs differ from the required inventory; "
                f"expected {required}, found {artifact_ids}"
            )

    sample_universe = set(reference_index) | set(evidence_index)
    for values in generation_sets.values():
        sample_universe.update(values)
    if not sample_universe:
        raise ValueError("The outer-join sample universe is empty")

    missing_reference = sorted(sample_universe - set(reference_index))
    missing_evidence = sorted(sample_universe - set(evidence_index))
    extra_reference = sorted(set(reference_index) - set(evidence_index))
    extra_evidence = sorted(set(evidence_index) - set(reference_index))
    missing_generation = {
        artifact_id: sorted(sample_universe - generation_sets.get(artifact_id, set()))
        for artifact_id in artifact_ids
    }
    mismatch = bool(
        missing_reference
        or missing_evidence
        or any(missing_generation.values())
        or set(reference_index) != set(evidence_index)
    )
    alignment_audit = {
        "join": "full_outer_by_sample_id",
        "missing_policy": missing_policy,
        "artifact_ids": artifact_ids,
        "sample_universe_count": len(sample_universe),
        "reference_sample_count": len(reference_index),
        "evidence_sample_count": len(evidence_index),
        "generation_sample_count_by_artifact": {
            artifact_id: len(generation_sets[artifact_id])
            for artifact_id in artifact_ids
        },
        "missing_reference_sample_ids": missing_reference,
        "missing_evidence_sample_ids": missing_evidence,
        "reference_without_evidence_sample_ids": extra_reference,
        "evidence_without_reference_sample_ids": extra_evidence,
        "missing_generation_sample_ids_by_artifact": missing_generation,
        "complete": not mismatch,
    }
    if mismatch and missing_policy == "error":
        raise ValueError(
            "Outer alignment found missing rows; refusing a silent inner join: "
            + json.dumps(alignment_audit, ensure_ascii=False, sort_keys=True)
        )

    aligned: list[dict[str, Any]] = []
    for artifact_id in artifact_ids:
        for sample_id in sorted(sample_universe):
            generation_row = generation_index.get((artifact_id, sample_id))
            reference_row = reference_index.get(sample_id)
            evidence_row = evidence_index.get(sample_id)
            missing = []
            if generation_row is None:
                missing.append("generation")
            if reference_row is None:
                missing.append("reference")
            if evidence_row is None:
                missing.append("evidence")
            meeting_id = _meeting_id(
                reference_row,
                evidence_row,
                generation_row,
                sample_id=sample_id,
            )
            aligned.append(
                {
                    "artifact_id": artifact_id,
                    "sample_id": sample_id,
                    "meeting_id": meeting_id,
                    "generation_row": generation_row,
                    "reference_row": reference_row,
                    "evidence_row": evidence_row,
                    "alignment_status": (
                        "complete" if not missing else "missing_" + "_".join(missing)
                    ),
                    "missing_inputs": missing,
                }
            )
    return aligned, alignment_audit


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_RE.finditer(text)]


def rouge_l_f1(candidate: str, reference: str) -> float:
    """Compute token-level ROUGE-L F1 without an external package."""

    candidate_tokens = _tokens(candidate)
    reference_tokens = _tokens(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0

    # Exact bit-parallel LCS.  ``state`` encodes the positions at which the
    # current dynamic-programming row increases.  Python's arbitrary-precision
    # integers update the full reference row at once, avoiding quadratic Python
    # loops for token-limit completions while preserving the same LCS value.
    reference_masks: dict[str, int] = {}
    for index, reference_token in enumerate(reference_tokens):
        reference_masks[reference_token] = reference_masks.get(
            reference_token, 0
        ) | (1 << index)
    state = 0
    for candidate_token in candidate_tokens:
        matches_or_state = state | reference_masks.get(candidate_token, 0)
        state = matches_or_state & ~(
            matches_or_state - ((state << 1) | 1)
        )
    lcs = state.bit_count()
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall) if lcs else 0.0


def repetition_rate(text: str, *, ngram_size: int = 3) -> float:
    tokens = _tokens(text)
    if len(tokens) < ngram_size:
        return 0.0
    ngrams = [
        tuple(tokens[index : index + ngram_size])
        for index in range(len(tokens) - ngram_size + 1)
    ]
    return (len(ngrams) - len(set(ngrams))) / len(ngrams)


def _format_requirements(
    reference_row: Mapping[str, Any],
    evidence_row: Mapping[str, Any],
) -> dict[str, Any]:
    requirements: dict[str, Any] = {}
    for row in (reference_row, evidence_row):
        value = row.get("format_requirements") or row.get("format_rules")
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise ValueError("format_requirements must be a JSON object")
        for key, item in value.items():
            if key in requirements and requirements[key] != item:
                raise ValueError(f"Conflicting format requirement {key!r}")
            requirements[key] = item
    return requirements


def check_format(
    text: str,
    generation_row: Mapping[str, Any],
    reference_row: Mapping[str, Any],
    evidence_row: Mapping[str, Any],
) -> tuple[float, list[str]]:
    """Check nonfatal output-format requirements.

    Generation failures are classified separately by
    ``_generation_fatal_reasons``.  A non-empty, normally completed answer can
    therefore remain scoreable while receiving ``format_compliance=0``.
    """

    violations: list[str] = []

    for label, pattern in FORMAL_BODY_ONLY_FORBIDDEN_PATTERNS:
        if pattern.search(text):
            violations.append(f"formal_body_only_violation:{label}")

    requirements = _format_requirements(reference_row, evidence_row)
    required_patterns = requirements.get("required_patterns", [])
    forbidden_patterns = requirements.get("forbidden_patterns", [])
    required_tags = requirements.get("required_tags", [])
    required_sections = requirements.get("required_sections", [])
    for label, values in (
        ("required_patterns", required_patterns),
        ("forbidden_patterns", forbidden_patterns),
        ("required_tags", required_tags),
        ("required_sections", required_sections),
    ):
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, Sequence):
            raise ValueError(f"format_requirements.{label} must be a list or string")
        if label == "required_patterns":
            for pattern in values:
                if re.search(str(pattern), text, flags=re.MULTILINE) is None:
                    violations.append(f"missing_required_pattern:{pattern}")
        elif label == "forbidden_patterns":
            for pattern in values:
                if re.search(str(pattern), text, flags=re.MULTILINE) is not None:
                    violations.append(f"forbidden_pattern:{pattern}")
        else:
            for literal in values:
                if str(literal) not in text:
                    violations.append(f"missing_{label[:-1]}:{literal}")
    min_chars = requirements.get("min_chars")
    max_chars = requirements.get("max_chars")
    if min_chars is not None and len(text) < int(min_chars):
        violations.append(f"below_min_chars:{int(min_chars)}")
    if max_chars is not None and len(text) > int(max_chars):
        violations.append(f"above_max_chars:{int(max_chars)}")

    known_pairs = (("<think>", "</think>"), ("<answer>", "</answer>"))
    for opening, closing in known_pairs:
        if opening in text or closing in text:
            if text.count(opening) != text.count(closing):
                violations.append(f"unbalanced_tag:{opening}")
            elif opening in text and text.find(opening) > text.find(closing):
                violations.append(f"misordered_tag:{opening}")
    return (0.0 if violations else 1.0), violations


def _generation_fatal_reasons(
    text: str,
    generation_row: Mapping[str, Any],
) -> list[str]:
    """Return only failures that make the generated answer unscoreable."""

    reasons: list[str] = []
    if not text.strip():
        reasons.append("empty_output")
    if generation_row.get("valid_generation") is False:
        upstream_reasons = generation_row.get("invalid_reasons", [])
        if isinstance(upstream_reasons, str):
            upstream_reasons = [upstream_reasons]
        if not isinstance(upstream_reasons, list):
            raise ValueError("generation.invalid_reasons must be a list or string")
        normalized = [_nonblank(reason) for reason in upstream_reasons]
        reasons.extend(
            f"upstream_invalid:{reason}" for reason in normalized if reason
        )
        if not any(normalized):
            reasons.append("upstream_valid_generation_false")
    validation_status = _nonblank(generation_row.get("generation_validation_status"))
    if validation_status and validation_status.casefold() not in {
        "passed",
        "valid",
        "validated",
        "ok",
        "success",
    }:
        reasons.append(f"generation_validation_status:{validation_status}")
    finish_reason = _nonblank(generation_row.get("generation_finish_reason"))
    if finish_reason.casefold() in {"length", "max_tokens", "token_limit"}:
        reasons.append(f"truncated:{finish_reason}")
    if generation_row.get("input_was_truncated") is True:
        reasons.append("input_was_truncated")
    for field in ("response_parseable", "parseable_final_answer"):
        if generation_row.get(field) is False:
            reasons.append(f"{field}:false")
    return list(dict.fromkeys(reasons))


def _length_tolerant_fatal_reasons(
    text: str,
    generation_row: Mapping[str, Any],
) -> list[str]:
    """Fatal conditions for the post-hoc length-tolerant robustness analysis."""

    reasons: list[str] = []
    if not text.strip():
        reasons.append("empty_extracted_candidate")
    if generation_row.get("input_was_truncated") is True:
        reasons.append("input_was_truncated")
    return reasons


def _scoring_policy_spec(scoring_policy: str) -> dict[str, Any]:
    if scoring_policy == STRICT_SCORING_POLICY:
        return {
            "policy_id": STRICT_SCORING_POLICY,
            "analysis_status": "primary_strict",
            "source_text": "strict parsed final_answer",
            "token_limit_finish_is_fatal": True,
            "upstream_valid_generation_enforced": True,
        }
    if scoring_policy == LENGTH_TOLERANT_SCORING_POLICY:
        return {
            "policy_id": LENGTH_TOLERANT_SCORING_POLICY,
            "analysis_status": "post_hoc_robustness",
            "primary_result": False,
            "source_text": "raw model completion",
            "extraction": {
                "closing_think_required": False,
                "first_case_insensitive_answer_opening_tag_is_boundary": True,
                "optional_trailing_answer_closing_tag_removed": True,
                "answer_opening_tag_absent": "score_full_nonempty_completion",
                "think_opening_tag": "audit_only",
            },
            "token_limit_finish_is_fatal": False,
            "upstream_valid_generation_enforced": False,
            "fatal_conditions": [
                "empty extracted candidate",
                "input truncation",
            ],
        }
    raise ValueError(
        f"scoring_policy must be one of {list(SCORING_POLICIES)}, "
        f"got {scoring_policy!r}"
    )


def _normalize_unit(value: Any) -> str | None:
    text = _nonblank(value).casefold().replace("_", " ")
    if not text:
        return None
    aliases = {
        "%": "percent",
        "percentage": "percent",
        "percentage point": "percentage_point",
        "percentage points": "percentage_point",
        "percent point": "percentage_point",
        "percent points": "percentage_point",
        "percent": "percent",
        "bp": "basis_point",
        "bps": "basis_point",
        "basis point": "basis_point",
        "basis points": "basis_point",
        "$": "dollar",
        "dollars": "dollar",
        "usd": "dollar",
        "index points": "index_point",
        "points": "index_point",
    }
    if "basis point" in text or text in {"bp", "bps"}:
        return "basis_point"
    if "percentage point" in text or "percent point" in text:
        return "percentage_point"
    if "percent" in text or "percentage" in text:
        return "percent"
    if "yuan per" in text and "dollar" in text:
        return "yuan_per_dollar"
    if "yen per" in text and "dollar" in text:
        return "yen_per_dollar"
    if "dollar" in text and "per euro" in text:
        return "dollar_per_euro"
    if "dollar" in text and "per pound" in text:
        return "dollar_per_pound"
    if "dollar" in text and "per barrel" in text:
        return "dollar_per_barrel"
    if "trillion" in text:
        return "trillion"
    if "billion" in text:
        return "billion"
    if "million" in text:
        return "million"
    if "thousand" in text:
        return "thousand"
    if "index" in text:
        return "index_point"
    if "dollar" in text or text == "usd":
        return "dollar"
    return aliases.get(text, text.replace(" ", "_"))


def _decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip().replace(",", "")
    text = text.removesuffix("%").strip()
    if not text:
        return None
    try:
        return Decimal(text).normalize()
    except InvalidOperation:
        return None


def _normalize_time(value: Any) -> str | None:
    text = _nonblank(value)
    if not text:
        return None
    return re.sub(r"\s+", " ", text.replace("–", "-").replace("—", "-")).casefold()


def _extract_times(text: str) -> list[TimeMention]:
    matches: dict[tuple[int, int], TimeMention] = {}
    for pattern in (
        ISO_DATE_RANGE_RE,
        ISO_DATE_RE,
        YEAR_RANGE_RE,
        QUARTER_RE,
        MONTH_YEAR_RE,
        RELATIVE_TIME_RE,
        YEAR_RE,
    ):
        for match in pattern.finditer(text):
            span = match.span()
            if any(
                other_start <= span[0] and span[1] <= other_end
                for other_start, other_end in matches
            ):
                continue
            matches[span] = TimeMention(
                raw=match.group(0),
                normalized=_normalize_time(match.group(0)) or "",
                start=span[0],
                end=span[1],
            )
    return [matches[key] for key in sorted(matches)]


def _unit_after_number(text: str, start: int, end: int) -> str | None:
    if start > 0 and text[start - 1] == "$":
        return "dollar"
    tail = text[end : end + 32]
    for unit, pattern in UNIT_PATTERNS:
        if pattern.search(tail):
            return unit
    return None


def _extract_numbers(text: str) -> list[NumberMention]:
    time_spans = [(item.start, item.end) for item in _extract_times(text)]
    result: list[NumberMention] = []
    for match in NUMBER_RE.finditer(text):
        value = _decimal(match.group(0))
        if value is None:
            continue
        start, end = match.span()
        result.append(
            NumberMention(
                raw=match.group(0),
                value=value,
                unit=_unit_after_number(text, start, end),
                start=start,
                end=end,
                is_time=any(
                    left <= start and end <= right for left, right in time_spans
                ),
            )
        )
    return result


def _display_decimal_places(raw_number: str) -> int:
    normalized = raw_number.strip().replace(",", "")
    if "." not in normalized:
        return 0
    return len(normalized.rsplit(".", maxsplit=1)[1])


def _matches_display_precision(
    evidence_value: Decimal,
    mention: NumberMention,
) -> bool:
    """Match only when evidence rounds to the number as displayed.

    This is a decimal display-precision rule, not a relative fuzzy tolerance and
    not a unit conversion.  ROUND_HALF_UP is fixed here so ties and negative
    values are deterministic across Python/decimal contexts.
    """

    decimal_places = _display_decimal_places(mention.raw)
    quantum = Decimal(1).scaleb(-decimal_places)
    with localcontext() as context:
        context.prec = max(
            28,
            len(evidence_value.as_tuple().digits) + decimal_places + 8,
            len(mention.value.as_tuple().digits) + decimal_places + 8,
        )
        try:
            rounded = evidence_value.quantize(quantum, rounding=ROUND_HALF_UP)
        except InvalidOperation:
            return False
    return rounded == mention.value


def _normalize_direction(value: Any) -> str | None:
    text = _nonblank(value).casefold()
    aliases = {
        "increase": "up",
        "increasing": "up",
        "rise": "up",
        "rising": "up",
        "higher": "up",
        "positive": "up",
        "strengthen": "up",
        "improve": "up",
        "decrease": "down",
        "decreasing": "down",
        "fall": "down",
        "falling": "down",
        "lower": "down",
        "negative": "down",
        "weaken": "down",
        "deteriorate": "down",
        "unchanged": "flat",
        "stable": "flat",
        "steady": "flat",
        "neutral": "flat",
        "mixed": "mixed",
        "up": "up",
        "down": "down",
        "flat": "flat",
    }
    return aliases.get(text)


def _normalize_topic(value: Any) -> str | None:
    text = _nonblank(value).casefold().replace("_", " ").replace("-", " ")
    if not text:
        return None
    if "unemployment" in text or "jobless" in text:
        return "unemployment"
    if any(term in text for term in ("inflation", "price", "cpi", "pce")):
        return "inflation"
    if any(
        term in text for term in ("employment", "payroll", "labor", "labour", "jobs")
    ):
        return "employment"
    if any(term in text for term in ("gdp", "growth", "economic activity", "output")):
        return "growth"
    for topic, aliases in TOPIC_ALIASES.items():
        if text == topic or text in aliases:
            return topic
    return text.replace(" ", "_")


def _normalize_stance(value: Any) -> str | None:
    text = _nonblank(value).casefold()
    aliases = {
        "tightening": "hawkish",
        "restrictive": "hawkish",
        "hike": "hawkish",
        "raise": "hawkish",
        "hawkish": "hawkish",
        "easing": "dovish",
        "accommodative": "dovish",
        "cut": "dovish",
        "lower": "dovish",
        "dovish": "dovish",
        "hold": "neutral",
        "unchanged": "neutral",
        "balanced": "neutral",
        "neutral": "neutral",
    }
    return aliases.get(text)


def _fact_rows(evidence_row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for field in ("evidence_facts", "facts", "numeric_facts", "claims"):
        value = evidence_row.get(field)
        if value is None:
            continue
        if not isinstance(value, list) or any(
            not isinstance(item, Mapping) for item in value
        ):
            raise ValueError(f"evidence.{field} must be a list of JSON objects")
        return list(value)
    return []


def _parse_evidence_facts(evidence_row: Mapping[str, Any]) -> list[EvidenceFact]:
    facts: list[EvidenceFact] = []
    for index, row in enumerate(_fact_rows(evidence_row)):
        value = row.get("value")
        if value is None:
            value = row.get("numeric_value")
        time_range = (
            row.get("time_range")
            or row.get("period")
            or row.get("date")
            or row.get("reference_period")
        )
        if not time_range and row.get("from_date") and row.get("to_date"):
            time_range = f"{row['from_date']} to {row['to_date']}"
        numeric_value = _decimal(value)
        direction = _normalize_direction(row.get("direction") or row.get("trend"))
        kind = _nonblank(row.get("kind")).casefold()
        if (
            direction is None
            and numeric_value is not None
            and kind.startswith("derived_")
        ):
            direction = (
                "up" if numeric_value > 0 else ("down" if numeric_value < 0 else "flat")
            )
        derived_fact_id = "::".join(
            filter(
                None,
                (
                    _nonblank(row.get("indicator")),
                    _nonblank(row.get("series_id")),
                    _nonblank(row.get("kind")),
                    _nonblank(row.get("date") or row.get("to_date")),
                ),
            )
        )
        facts.append(
            EvidenceFact(
                fact_id=_nonblank(row.get("fact_id") or row.get("claim_id"))
                or derived_fact_id
                or f"fact-{index + 1}",
                value=numeric_value,
                unit=_normalize_unit(row.get("unit")),
                time_range=_normalize_time(time_range),
                topic=_normalize_topic(
                    row.get("topic") or row.get("indicator") or row.get("metric")
                ),
                direction=direction,
                kind=kind or None,
                series_id=_nonblank(row.get("series_id")) or None,
            )
        )
    return facts


def _evidence_text(evidence_row: Mapping[str, Any]) -> str:
    values = []
    for field in ("evidence_text", "input_text", "source_text", "prompt", "context"):
        value = evidence_row.get(field)
        if value not in (None, ""):
            values.append(str(value))
    return "\n".join(values)


def _allowed_numbers(
    evidence_row: Mapping[str, Any],
    facts: Sequence[EvidenceFact],
) -> set[Decimal]:
    allowed = {fact.value for fact in facts if fact.value is not None}
    for fact in facts:
        if fact.time_range:
            allowed.update(
                mention.value for mention in _extract_numbers(fact.time_range)
            )
    for field in ("allowed_numbers", "numbers", "numeric_values"):
        values = evidence_row.get(field, [])
        if values is None:
            continue
        if not isinstance(values, list):
            values = [values]
        allowed.update(
            value for item in values if (value := _decimal(item)) is not None
        )
    for field in ("allowed_times", "time_ranges", "periods", "dates"):
        values = evidence_row.get(field, [])
        if values is None:
            continue
        if not isinstance(values, list):
            values = [values]
        for item in values:
            allowed.update(mention.value for mention in _extract_numbers(str(item)))
    allowed.update(
        item.value for item in _extract_numbers(_evidence_text(evidence_row))
    )
    return allowed


def _allowed_times(
    evidence_row: Mapping[str, Any],
    facts: Sequence[EvidenceFact],
) -> set[str]:
    allowed = {fact.time_range for fact in facts if fact.time_range is not None}
    for field in ("allowed_times", "time_ranges", "periods", "dates"):
        values = evidence_row.get(field, [])
        if values is None:
            continue
        if not isinstance(values, list):
            values = [values]
        allowed.update(
            value for item in values if (value := _normalize_time(item)) is not None
        )
    allowed.update(
        item.normalized for item in _extract_times(_evidence_text(evidence_row))
    )
    return allowed


def _expected_directions(
    evidence_row: Mapping[str, Any],
    facts: Sequence[EvidenceFact],
) -> dict[str, str]:
    result: dict[str, str] = {}
    ambiguous_topics: set[str] = set()
    raw = (
        evidence_row.get("directions") or evidence_row.get("expected_directions") or {}
    )
    if raw and not isinstance(raw, Mapping):
        raise ValueError("evidence.directions must be a JSON object")
    for topic, direction in dict(raw).items():
        normalized_topic = _normalize_topic(topic)
        normalized_direction = _normalize_direction(direction)
        if (
            normalized_topic in CORE_DIRECTION_TOPICS
            and normalized_direction == "mixed"
        ):
            ambiguous_topics.add(normalized_topic)
        if normalized_topic in CORE_DIRECTION_TOPICS and normalized_direction in {
            "up",
            "down",
            "flat",
        }:
            result[normalized_topic] = normalized_direction
    explicit_topics = set(result) | ambiguous_topics
    direction_facts: dict[str, list[EvidenceFact]] = defaultdict(list)
    for fact in facts:
        if (
            fact.topic in CORE_DIRECTION_TOPICS
            and fact.direction
            and fact.topic not in explicit_topics
        ):
            direction_facts[fact.topic].append(fact)
    for topic, topic_facts in direction_facts.items():
        short_run = [
            fact for fact in topic_facts if fact.kind == "derived_absolute_change"
        ]
        year_run = [
            fact for fact in topic_facts if fact.kind == "derived_year_absolute_change"
        ]
        selected = short_run or year_run or topic_facts
        directions = {fact.direction for fact in selected if fact.direction}
        if len(directions) == 1:
            result[topic] = directions.pop()
    return result


def _extract_direction_claims(
    text: str,
    expected: Mapping[str, str],
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for sentence_match in SENTENCE_RE.finditer(text):
        sentence = sentence_match.group(0)
        lower = sentence.casefold()
        direction_matches = [
            (direction, match)
            for direction, pattern in DIRECTION_PATTERNS.items()
            for match in pattern.finditer(sentence)
        ]
        if not direction_matches:
            continue
        for topic in sorted(set(expected) | set(CORE_DIRECTION_TOPICS)):
            expected_direction = expected.get(topic)
            aliases = TOPIC_ALIASES.get(topic, (topic.replace("_", " "),))
            topic_matches = [
                match
                for alias in aliases
                for match in re.finditer(rf"\b{re.escape(alias)}\b", lower)
            ]
            if not topic_matches:
                continue
            for topic_match in topic_matches:
                topic_center = (topic_match.start() + topic_match.end()) / 2.0
                direction, match = min(
                    direction_matches,
                    key=lambda item: abs(
                        ((item[1].start() + item[1].end()) / 2.0) - topic_center
                    ),
                )
                start = sentence_match.start() + match.start()
                end = sentence_match.start() + match.end()
                key = (topic, start, end)
                if key in seen:
                    continue
                seen.add(key)
                claims.append(
                    {
                        "rule_kind": "direction",
                        "claim_text": sentence.strip(),
                        "normalized_claim": direction,
                        "expected": expected_direction,
                        "supported": (
                            expected_direction is not None
                            and direction == expected_direction
                        ),
                        "reason": (
                            "direction_evidence_unavailable"
                            if expected_direction is None
                            else (
                                "direction_matches_evidence"
                                if direction == expected_direction
                                else "direction_conflicts_with_evidence"
                            )
                        ),
                        "char_start": start,
                        "char_end": end,
                        "topic": topic,
                        "counts_toward_unsupported": True,
                    }
                )
    return claims


def _extract_stance_claims(
    text: str,
    expected_stance: str | None,
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for stance, pattern in STANCE_PATTERNS.items():
        for match in pattern.finditer(text):
            claims.append(
                {
                    "rule_kind": "policy_stance",
                    "claim_text": match.group(0),
                    "normalized_claim": stance,
                    "expected": expected_stance,
                    "supported": (
                        expected_stance is not None and stance == expected_stance
                    ),
                    "reason": (
                        "stance_evidence_unavailable"
                        if expected_stance is None
                        else (
                            "stance_matches_evidence"
                            if stance == expected_stance
                            else "stance_conflicts_with_evidence"
                        )
                    ),
                    "char_start": match.start(),
                    "char_end": match.end(),
                    "topic": "policy_stance",
                    "counts_toward_unsupported": True,
                }
            )
    return claims


def _expected_policy_stance(
    evidence_row: Mapping[str, Any],
    facts: Sequence[EvidenceFact],
) -> str | None:
    explicit = _normalize_stance(
        evidence_row.get("policy_stance") or evidence_row.get("expected_policy_stance")
    )
    if explicit is not None:
        return explicit
    rate_facts = [
        fact
        for fact in facts
        if fact.direction
        and fact.topic
        and any(
            marker in fact.topic
            for marker in (
                "federal_funds",
                "fed_funds",
                "policy_rate",
                "target_rate",
            )
        )
    ]
    short_run = [fact for fact in rate_facts if fact.kind == "derived_absolute_change"]
    selected = short_run or rate_facts
    directions = {fact.direction for fact in selected if fact.direction}
    if len(directions) != 1:
        return None
    return {
        "up": "hawkish",
        "down": "dovish",
        "flat": "neutral",
    }.get(directions.pop())


def _custom_claim_checks(
    text: str,
    evidence_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rules = evidence_row.get("claim_rules", [])
    unsupported_patterns = evidence_row.get("unsupported_patterns", [])
    if isinstance(unsupported_patterns, str):
        unsupported_patterns = [unsupported_patterns]
    if unsupported_patterns:
        if not isinstance(unsupported_patterns, list):
            raise ValueError("evidence.unsupported_patterns must be a list or string")
        rules = list(rules or []) + [
            {
                "claim_id": f"unsupported-pattern-{index + 1}",
                "pattern": pattern,
                "supported": False,
            }
            for index, pattern in enumerate(unsupported_patterns)
        ]
    if not rules:
        return []
    if not isinstance(rules, list) or any(
        not isinstance(item, Mapping) for item in rules
    ):
        raise ValueError("evidence.claim_rules must be a list of JSON objects")
    checks: list[dict[str, Any]] = []
    for index, rule in enumerate(rules):
        patterns = rule.get("patterns", rule.get("pattern"))
        if isinstance(patterns, str):
            patterns = [patterns]
        if not patterns:
            continue
        if not isinstance(patterns, list):
            raise ValueError("claim_rule.patterns must be a list or string")
        supported = bool(rule.get("supported", True))
        for pattern in patterns:
            for match in re.finditer(str(pattern), text, flags=re.IGNORECASE):
                checks.append(
                    {
                        "rule_kind": "custom",
                        "claim_text": match.group(0),
                        "normalized_claim": _nonblank(
                            rule.get("claim_id") or f"custom-{index + 1}"
                        ),
                        "expected": "supported" if supported else "absent",
                        "supported": supported,
                        "reason": (
                            "matched_supported_claim_rule"
                            if supported
                            else "matched_explicit_unsupported_rule"
                        ),
                        "char_start": match.start(),
                        "char_end": match.end(),
                        "topic": _normalize_topic(rule.get("topic")),
                        "counts_toward_unsupported": True,
                    }
                )
    return checks


def evaluate_factual_rules(
    text: str,
    evidence_row: Mapping[str, Any],
) -> tuple[dict[str, float | int | None], list[dict[str, Any]]]:
    """Evaluate rule-covered numerical, temporal, directional, and stance claims."""

    facts = _parse_evidence_facts(evidence_row)
    if not any(fact.value is not None for fact in facts):
        for index, mention in enumerate(_extract_numbers(_evidence_text(evidence_row))):
            if mention.is_time:
                continue
            facts.append(
                EvidenceFact(
                    fact_id=f"evidence-text-number-{index + 1}",
                    value=mention.value,
                    unit=mention.unit,
                    time_range=None,
                    topic=None,
                    direction=None,
                    kind="evidence_text_number",
                    series_id=None,
                )
            )
    allowed_numbers = _allowed_numbers(evidence_row, facts)
    allowed_times = _allowed_times(evidence_row, facts)
    allowed_number_values = sorted(str(value) for value in allowed_numbers)
    allowed_number_set_sha256 = hashlib.sha256(
        json.dumps(
            allowed_number_values,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    allowed_time_values = sorted(allowed_times)
    allowed_time_set_sha256 = hashlib.sha256(
        json.dumps(
            allowed_time_values,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    expected_directions = _expected_directions(evidence_row, facts)
    expected_stance = _expected_policy_stance(evidence_row, facts)
    number_mentions = _extract_numbers(text)
    time_mentions = _extract_times(text)
    checks: list[dict[str, Any]] = []

    non_time_mentions = [mention for mention in number_mentions if not mention.is_time]
    supported_numeric_count = 0
    novel_count = 0
    unit_results: list[bool] = []
    for mention in number_mentions:
        matching_facts = [
            fact
            for fact in facts
            if fact.value is not None
            and _matches_display_precision(fact.value, mention)
        ]
        matching_allowed_numbers = [
            value
            for value in allowed_numbers
            if _matches_display_precision(value, mention)
        ]
        number_supported = bool(matching_allowed_numbers)
        expected_units = (
            {fact.unit for fact in matching_facts if fact.unit}
            if not mention.is_time
            else set()
        )
        unit_supported: bool | None = None
        if expected_units:
            unit_supported = mention.unit in expected_units
            unit_results.append(unit_supported)
            checks.append(
                {
                    "rule_kind": "unit",
                    "claim_text": text[
                        mention.start : min(len(text), mention.end + 24)
                    ].strip(),
                    "normalized_claim": mention.unit,
                    "expected": sorted(expected_units),
                    "supported": unit_supported,
                    "reason": (
                        "unit_matches_evidence"
                        if unit_supported
                        else "unit_missing_or_conflicts_with_evidence"
                    ),
                    "char_start": mention.start,
                    "char_end": mention.end,
                    "topic": None,
                    "counts_toward_unsupported": False,
                }
            )
        claim_supported = number_supported and unit_supported is not False
        if not mention.is_time:
            supported_numeric_count += int(number_supported)
        novel_count += int(not number_supported)
        checks.append(
            {
                "rule_kind": "numeric",
                "claim_text": mention.raw,
                "normalized_claim": str(mention.value),
                "expected": str(mention.value) if number_supported else None,
                "matched_fact_ids": sorted(fact.fact_id for fact in matching_facts),
                "matched_evidence_values": sorted(
                    {str(value) for value in matching_allowed_numbers}
                ),
                "display_decimal_places": _display_decimal_places(mention.raw),
                "numeric_match_policy": (
                    "decimal-display-precision-round-half-up-v1"
                ),
                "evidence_allowed_value_count": len(allowed_number_values),
                "evidence_allowed_value_set_sha256": allowed_number_set_sha256,
                "supported": claim_supported,
                "reason": (
                    "number_and_unit_supported"
                    if claim_supported
                    else (
                        "number_absent_from_evidence"
                        if not number_supported
                        else "unit_conflicts_with_evidence"
                    )
                ),
                "char_start": mention.start,
                "char_end": mention.end,
                "topic": None,
                "counts_toward_unsupported": not mention.is_time,
            }
        )

    time_results: list[bool] = []
    for mention in time_mentions:
        supported = mention.normalized in allowed_times
        time_results.append(supported)
        checks.append(
            {
                "rule_kind": "time",
                "claim_text": mention.raw,
                "normalized_claim": mention.normalized,
                "expected": mention.normalized if supported else None,
                "evidence_allowed_time_count": len(allowed_time_values),
                "evidence_allowed_time_set_sha256": allowed_time_set_sha256,
                "supported": supported,
                "reason": (
                    "time_matches_evidence"
                    if supported
                    else "time_absent_from_evidence"
                ),
                "char_start": mention.start,
                "char_end": mention.end,
                "topic": None,
                "counts_toward_unsupported": True,
            }
        )

    direction_checks = _extract_direction_claims(text, expected_directions)
    stance_checks = _extract_stance_claims(text, expected_stance)
    coverable_stance_checks = [
        check for check in stance_checks if check.get("expected") is not None
    ]
    checks.extend(direction_checks)
    checks.extend(stance_checks)
    checks.extend(_custom_claim_checks(text, evidence_row))

    covered_checks = [
        check for check in checks if check.get("counts_toward_unsupported")
    ]
    unsupported_count = sum(not bool(check["supported"]) for check in covered_checks)
    coverable_direction_checks = [
        check for check in direction_checks if check.get("expected") is not None
    ]
    direction_supported = sum(
        bool(check["supported"]) for check in coverable_direction_checks
    )
    mentioned_direction_topics = {
        str(check["topic"]) for check in direction_checks if check.get("topic")
    }
    covered_fact_values = {
        fact.fact_id
        for fact in facts
        if fact.value is not None
        and any(
            _matches_display_precision(fact.value, mention)
            and (fact.unit is None or mention.unit == fact.unit)
            for mention in non_time_mentions
        )
    }
    facts_with_values = [fact for fact in facts if fact.value is not None]

    metrics: dict[str, float | int | None] = {
        "numeric_claim_count": len(non_time_mentions),
        "numeric_value_accuracy": (
            supported_numeric_count / len(non_time_mentions)
            if non_time_mentions
            else None
        ),
        "evidence_numeric_fact_count": len(facts_with_values),
        "evidence_unit_fact_count": sum(fact.unit is not None for fact in facts),
        "evidence_time_fact_count": len(allowed_times),
        "evidence_value_coverage": (
            len(covered_fact_values) / len(facts_with_values)
            if facts_with_values
            else None
        ),
        "unit_check_count": len(unit_results),
        "unit_accuracy": (
            sum(unit_results) / len(unit_results) if unit_results else None
        ),
        "time_check_count": len(time_results),
        "time_accuracy": (
            sum(time_results) / len(time_results) if time_results else None
        ),
        "number_mention_count": len(number_mentions),
        "novel_number_count": novel_count,
        "novel_number_rate": (
            novel_count / len(number_mentions) if number_mentions else None
        ),
        "rule_covered_claim_count": len(covered_checks),
        "rule_covered_unsupported_count": unsupported_count,
        "rule_covered_unsupported_rate": (
            unsupported_count / len(covered_checks) if covered_checks else None
        ),
        "direction_claim_count": len(direction_checks),
        "direction_coverable_claim_count": len(coverable_direction_checks),
        "direction_consistency": (
            direction_supported / len(coverable_direction_checks)
            if coverable_direction_checks
            else None
        ),
        "expected_direction_topic_count": len(expected_directions),
        "direction_coverage": (
            len(mentioned_direction_topics & set(expected_directions))
            / len(expected_directions)
            if expected_directions
            else None
        ),
        "policy_stance_claim_count": len(stance_checks),
        "policy_stance_coverable_claim_count": len(coverable_stance_checks),
        "expected_policy_stance_available": int(expected_stance is not None),
        "policy_stance_coverage": (
            float(bool(stance_checks)) if expected_stance is not None else None
        ),
        "policy_stance_consistency": (
            sum(bool(check["supported"]) for check in coverable_stance_checks)
            / len(coverable_stance_checks)
            if coverable_stance_checks
            else None
        ),
    }
    return metrics, checks


@dataclass(frozen=True)
class TextChunk:
    """One tokenizer-bounded unit produced without dropping input tokens."""

    text: str
    token_count: int


_LONG_TEXT_SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[.!?;。！？；])(?:\s+|(?=[^\s]))|\n+"
)
_UNBOUNDED_TOKENIZER_LIMIT = 1_000_000


def _token_ids(tokenizer: Any, text: str) -> list[Any]:
    """Tokenize without special tokens or truncation.

    Hugging Face tokenizers expose ``encode`` directly.  Keeping this helper
    deliberately small also makes the chunking contract testable with a fake
    tokenizer, without importing a model package.
    """

    try:
        raw_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
    except TypeError:
        raw_ids = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(raw_ids, "tolist"):
        raw_ids = raw_ids.tolist()
    if not isinstance(raw_ids, (list, tuple)):
        raise ValueError("Tokenizer.encode must return a token-id sequence")
    if raw_ids and isinstance(raw_ids[0], (list, tuple)):
        raise ValueError("Tokenizer.encode returned batched IDs for one text")
    return list(raw_ids)


def _decode_token_ids(tokenizer: Any, token_ids: Sequence[Any]) -> str:
    try:
        decoded = tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        decoded = tokenizer.decode(list(token_ids))
    if not isinstance(decoded, str) or (token_ids and decoded == ""):
        raise ValueError("Tokenizer.decode did not preserve a non-empty token slice")
    return decoded


def _tokenizer_limit(tokenizer: Any, configured_max_length: int | None) -> int:
    limits: list[int] = []
    if configured_max_length is not None:
        if configured_max_length <= 0:
            raise ValueError("max_length must be positive")
        limits.append(int(configured_max_length))
    model_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_limit, int) and 0 < model_limit < _UNBOUNDED_TOKENIZER_LIMIT:
        limits.append(model_limit)
    if not limits:
        raise ValueError(
            "Tokenizer has no finite model_max_length; pin max_length explicitly"
        )
    return min(limits)


def _special_token_reserve(tokenizer: Any) -> int:
    special_count_method = getattr(tokenizer, "num_special_tokens_to_add", None)
    if callable(special_count_method):
        try:
            count = special_count_method(pair=False)
        except TypeError:
            count = special_count_method(False)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Tokenizer returned an invalid special-token count")
        return count
    with_special = len(tokenizer.encode("", add_special_tokens=True, truncation=False))
    without_special = len(_token_ids(tokenizer, ""))
    reserve = with_special - without_special
    if reserve < 0:
        raise ValueError("Tokenizer special-token reserve cannot be negative")
    return reserve


def effective_text_token_budget(
    tokenizer: Any,
    *,
    max_length: int | None = None,
) -> tuple[int, int, int]:
    """Return ``(effective_budget, total_limit, special_token_reserve)``."""

    total_limit = _tokenizer_limit(tokenizer, max_length)
    special_reserve = _special_token_reserve(tokenizer)
    budget = total_limit - special_reserve
    if budget <= 0:
        raise ValueError(
            "Tokenizer max length leaves no content-token budget after special tokens"
        )
    return budget, total_limit, special_reserve


def _sentence_segments(text: str) -> list[str]:
    return [
        segment.strip()
        for segment in _LONG_TEXT_SENTENCE_BOUNDARY_RE.split(text.strip())
        if segment.strip()
    ]


def _safe_token_slices(
    tokenizer: Any,
    token_ids: Sequence[Any],
    *,
    token_budget: int,
) -> list[TextChunk]:
    """Decode every token in an overlong sentence into bounded text pieces."""

    pending = [
        list(token_ids[start : start + token_budget])
        for start in range(0, len(token_ids), token_budget)
    ]
    chunks: list[TextChunk] = []
    while pending:
        ids = pending.pop(0)
        text = _decode_token_ids(tokenizer, ids)
        encoded_count = len(_token_ids(tokenizer, text))
        if encoded_count <= token_budget:
            chunks.append(TextChunk(text=text, token_count=encoded_count))
            continue
        if len(ids) <= 1:
            raise ValueError(
                "A decoded tokenizer token exceeds the effective token budget"
            )
        midpoint = len(ids) // 2
        pending[0:0] = [ids[:midpoint], ids[midpoint:]]
    return chunks


def _join_chunk_text(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if left[-1].isspace() or right[0].isspace():
        return left + right
    return f"{left} {right}"


def chunk_text_sentence_boundary(
    text: str,
    tokenizer: Any,
    *,
    max_length: int | None = None,
) -> list[TextChunk]:
    """Greedily chunk text at sentence boundaries under a tokenizer budget.

    A sentence longer than the effective budget is deterministically split by
    its complete token-id sequence.  Every returned chunk is re-tokenized and
    checked against the budget; truncation is never used.
    """

    token_budget, _, _ = effective_text_token_budget(
        tokenizer,
        max_length=max_length,
    )
    if not text.strip():
        return []

    atomic_units: list[TextChunk] = []
    for sentence in _sentence_segments(text):
        sentence_ids = _token_ids(tokenizer, sentence)
        if not sentence_ids:
            continue
        if len(sentence_ids) <= token_budget:
            atomic_units.append(TextChunk(text=sentence, token_count=len(sentence_ids)))
        else:
            atomic_units.extend(
                _safe_token_slices(
                    tokenizer,
                    sentence_ids,
                    token_budget=token_budget,
                )
            )

    chunks: list[TextChunk] = []
    current_text = ""
    current_count = 0
    for unit in atomic_units:
        combined = _join_chunk_text(current_text, unit.text)
        combined_count = len(_token_ids(tokenizer, combined))
        if current_text and combined_count > token_budget:
            chunks.append(TextChunk(current_text, current_count))
            current_text = unit.text
            current_count = unit.token_count
        else:
            current_text = combined
            current_count = combined_count
    if current_text:
        chunks.append(TextChunk(current_text, current_count))

    if any(
        chunk.token_count <= 0 or chunk.token_count > token_budget for chunk in chunks
    ):
        raise AssertionError("Long-text chunker emitted an invalid token count")
    return chunks


def _normalise_chunk_scores(
    raw: Mapping[str, Sequence[float]],
    *,
    expected_metrics: Sequence[str],
    expected_length: int,
) -> dict[str, list[float]]:
    missing = set(expected_metrics) - set(raw)
    if missing:
        raise ValueError(f"Chunk scorer omitted metrics: {sorted(missing)}")
    result: dict[str, list[float]] = {}
    for metric_name in expected_metrics:
        values = [float(value) for value in raw[metric_name]]
        if len(values) != expected_length:
            raise ValueError(
                f"Chunk scorer returned {len(values)} {metric_name} values for "
                f"{expected_length} non-empty chunk pairs"
            )
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"Chunk scorer returned non-finite {metric_name}")
        result[metric_name] = values
    return result


def _score_chunked_text_pairs(
    candidates: Sequence[str],
    references: Sequence[str],
    *,
    tokenizer: Any,
    max_length: int | None,
    expected_metrics: Sequence[str],
    chunk_scorer: Callable[
        [Sequence[str], Sequence[str]], Mapping[str, Sequence[float]]
    ],
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Score ordinal chunk pairs and aggregate with the frozen weighting rule."""

    if len(candidates) != len(references):
        raise ValueError("Semantic candidate/reference lengths differ")
    token_budget, total_limit, special_reserve = effective_text_token_budget(
        tokenizer,
        max_length=max_length,
    )
    candidate_chunks = [
        chunk_text_sentence_boundary(text, tokenizer, max_length=total_limit)
        for text in candidates
    ]
    reference_chunks = [
        chunk_text_sentence_boundary(text, tokenizer, max_length=total_limit)
        for text in references
    ]

    score_candidates: list[str] = []
    score_references: list[str] = []
    score_locations: list[tuple[int, int]] = []
    document_weights = [0] * len(candidates)
    document_pair_counts = [0] * len(candidates)
    missing_side_pairs = 0
    for document_index, (candidate_doc, reference_doc) in enumerate(
        zip(candidate_chunks, reference_chunks, strict=True)
    ):
        for candidate_chunk, reference_chunk in zip_longest(
            candidate_doc,
            reference_doc,
        ):
            candidate_tokens = candidate_chunk.token_count if candidate_chunk else 0
            reference_tokens = reference_chunk.token_count if reference_chunk else 0
            weight = max(candidate_tokens, reference_tokens)
            if weight <= 0:
                continue
            document_weights[document_index] += weight
            document_pair_counts[document_index] += 1
            if candidate_chunk is None or reference_chunk is None:
                missing_side_pairs += 1
                continue
            score_locations.append((document_index, weight))
            score_candidates.append(candidate_chunk.text)
            score_references.append(reference_chunk.text)

    if score_candidates:
        raw_scores = chunk_scorer(score_candidates, score_references)
        chunk_scores = _normalise_chunk_scores(
            raw_scores,
            expected_metrics=expected_metrics,
            expected_length=len(score_candidates),
        )
    else:
        chunk_scores = {metric_name: [] for metric_name in expected_metrics}

    aggregate = {
        metric_name: [0.0] * len(candidates) for metric_name in expected_metrics
    }
    for metric_name, values in chunk_scores.items():
        for (document_index, weight), value in zip(
            score_locations,
            values,
            strict=True,
        ):
            aggregate[metric_name][document_index] += weight * value
        for document_index, denominator in enumerate(document_weights):
            if denominator:
                aggregate[metric_name][document_index] /= denominator

    audit = {
        "long_text_policy": LONG_TEXT_POLICY,
        "pairing": "ordinal_zip_longest",
        "chunk_weight": "max(candidate_tokens,reference_tokens)",
        "silent_truncation": False,
        "tokenizer_total_limit": total_limit,
        "special_token_reserve": special_reserve,
        "effective_content_token_budget": token_budget,
        "document_count": len(candidates),
        "candidate_chunk_counts": [len(chunks) for chunks in candidate_chunks],
        "reference_chunk_counts": [len(chunks) for chunks in reference_chunks],
        "ordinal_pair_counts": document_pair_counts,
        "document_weight_tokens": document_weights,
        "scored_nonempty_chunk_pairs": len(score_candidates),
        "missing_side_zero_chunk_pairs": missing_side_pairs,
    }
    return aggregate, audit


def _semantic_backend_metadata(scorer: SemanticScorer) -> dict[str, Any]:
    scorer_id = _nonblank(getattr(scorer, "scorer_id", scorer.__class__.__name__))
    model_id = _nonblank(getattr(scorer, "model_id", "injected"))
    model_sha256 = _nonblank(getattr(scorer, "model_sha256", ""))
    metadata = {
        "scorer_id": scorer_id,
        "model_id": model_id,
        "model_sha256": model_sha256 or None,
        "backend_class": scorer.__class__.__name__,
        "injected": not isinstance(scorer, (BERTScoreBackend, MPNetCosineBackend)),
    }
    backend_metadata = getattr(scorer, "semantic_metadata", None)
    if callable(backend_metadata):
        extra = backend_metadata()
        if not isinstance(extra, Mapping):
            raise ValueError("semantic_metadata() must return a mapping")
        metadata.update(extra)
    return metadata


def _call_semantic_scorer(
    scorer: SemanticScorer,
    candidates: Sequence[str],
    references: Sequence[str],
) -> dict[str, list[float]]:
    raw = scorer.score(candidates, references)
    scorer_id = _nonblank(getattr(scorer, "scorer_id", scorer.__class__.__name__))
    if isinstance(raw, Mapping):
        metrics = {str(name): list(values) for name, values in raw.items()}
    else:
        metric_name = _nonblank(getattr(scorer, "metric_name", scorer_id))
        metrics = {metric_name: list(raw)}
    result: dict[str, list[float]] = {}
    for name, values in metrics.items():
        metric_name = re.sub(r"[^a-z0-9_]+", "_", name.casefold()).strip("_")
        if not metric_name:
            raise ValueError(
                f"Semantic scorer {scorer_id!r} returned an empty metric name"
            )
        if len(values) != len(candidates):
            raise ValueError(
                f"Semantic scorer {scorer_id!r} returned {len(values)} {metric_name} "
                f"scores for {len(candidates)} rows"
            )
        numeric = [float(value) for value in values]
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError(
                f"Semantic scorer {scorer_id!r} returned a non-finite {metric_name}"
            )
        result[metric_name] = numeric
    return result


class BERTScoreBackend:
    """BERTScore adapter pinned to a local model tree.

    No network identifier is accepted and ``bert_score`` is imported only when
    ``score`` is called.
    """

    scorer_id = "bertscore"

    def __init__(
        self,
        model_path: str | Path,
        model_sha256: str,
        *,
        num_layers: int | None = None,
        batch_size: int = 16,
        device: str = "cpu",
        max_length: int | None = None,
        verify_checksum: bool = True,
        tokenizer: Any | None = None,
        chunk_scorer: Callable[
            [Sequence[str], Sequence[str]], Mapping[str, Sequence[float]]
        ]
        | None = None,
    ) -> None:
        self.path = Path(model_path).expanduser().resolve()
        if not self.path.exists():
            raise ValueError(
                f"BERTScore requires an existing local model path: {self.path}"
            )
        self.model_id = str(self.path)
        self.model_sha256 = _validate_sha256(
            model_sha256,
            field="BERTScore model_sha256",
        )
        if verify_checksum:
            actual = _sha256_path(self.path)
            if actual != self.model_sha256:
                raise ValueError(
                    f"BERTScore model checksum mismatch: expected {self.model_sha256}, "
                    f"found {actual}"
                )
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self._tokenizer = tokenizer
        self._chunk_scorer = chunk_scorer
        self._chunk_audit: dict[str, Any] | None = None

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.path),
                local_files_only=True,
            )
        return self._tokenizer

    def _score_nonempty_chunks(
        self,
        candidates: Sequence[str],
        references: Sequence[str],
    ) -> Mapping[str, Sequence[float]]:
        if self._chunk_scorer is not None:
            return self._chunk_scorer(candidates, references)

        from bert_score import score as bert_score

        kwargs: dict[str, Any] = {
            "model_type": str(self.path),
            "batch_size": self.batch_size,
            "device": self.device,
            "verbose": False,
            "idf": False,
            "rescale_with_baseline": False,
        }
        if self.num_layers is not None:
            kwargs["num_layers"] = self.num_layers
        precision, recall, f1 = bert_score(list(candidates), list(references), **kwargs)
        return {
            "bertscore_precision": precision.detach().cpu().tolist(),
            "bertscore_recall": recall.detach().cpu().tolist(),
            "bertscore_f1": f1.detach().cpu().tolist(),
        }

    def score(
        self,
        candidates: Sequence[str],
        references: Sequence[str],
    ) -> Mapping[str, Sequence[float]]:
        scores, self._chunk_audit = _score_chunked_text_pairs(
            candidates,
            references,
            tokenizer=self._get_tokenizer(),
            max_length=self.max_length,
            expected_metrics=(
                "bertscore_precision",
                "bertscore_recall",
                "bertscore_f1",
            ),
            chunk_scorer=self._score_nonempty_chunks,
        )
        return scores

    def semantic_metadata(self) -> Mapping[str, Any]:
        return {
            "long_text_policy": LONG_TEXT_POLICY,
            "chunk_audit": self._chunk_audit,
        }


class MPNetCosineBackend:
    """Mean-pooled cosine scorer for a pinned, local MPNet encoder."""

    scorer_id = "mpnet"
    metric_name = "mpnet_cosine"

    def __init__(
        self,
        model_path: str | Path,
        model_sha256: str,
        *,
        batch_size: int = 16,
        device: str = "cpu",
        max_length: int = 512,
        verify_checksum: bool = True,
        tokenizer: Any | None = None,
        chunk_scorer: Callable[[Sequence[str], Sequence[str]], Sequence[float]]
        | None = None,
    ) -> None:
        self.path = Path(model_path).expanduser().resolve()
        if not self.path.exists():
            raise ValueError(
                f"MPNet requires an existing local model path: {self.path}"
            )
        self.model_id = str(self.path)
        self.model_sha256 = _validate_sha256(
            model_sha256,
            field="MPNet model_sha256",
        )
        if verify_checksum:
            actual = _sha256_path(self.path)
            if actual != self.model_sha256:
                raise ValueError(
                    f"MPNet model checksum mismatch: expected {self.model_sha256}, "
                    f"found {actual}"
                )
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self._tokenizer = tokenizer
        self._chunk_scorer = chunk_scorer
        self._chunk_audit: dict[str, Any] | None = None

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.path),
                local_files_only=True,
            )
        return self._tokenizer

    def _score_nonempty_chunks(
        self,
        candidates: Sequence[str],
        references: Sequence[str],
    ) -> Mapping[str, Sequence[float]]:
        if self._chunk_scorer is not None:
            return {self.metric_name: self._chunk_scorer(candidates, references)}

        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel

        tokenizer = self._get_tokenizer()
        model = AutoModel.from_pretrained(
            str(self.path),
            local_files_only=True,
        ).to(self.device)
        model.eval()

        def encode(texts: Sequence[str]) -> Any:
            chunks = []
            for start in range(0, len(texts), self.batch_size):
                encoded = tokenizer(
                    list(texts[start : start + self.batch_size]),
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                with torch.no_grad():
                    hidden = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                chunks.append(functional.normalize(pooled, p=2, dim=1).cpu())
            return torch.cat(chunks, dim=0)

        candidate_embeddings = encode(candidates)
        reference_embeddings = encode(references)
        return {
            self.metric_name: (candidate_embeddings * reference_embeddings)
            .sum(dim=1)
            .tolist()
        }

    def score(
        self,
        candidates: Sequence[str],
        references: Sequence[str],
    ) -> Sequence[float]:
        scores, self._chunk_audit = _score_chunked_text_pairs(
            candidates,
            references,
            tokenizer=self._get_tokenizer(),
            max_length=self.max_length,
            expected_metrics=(self.metric_name,),
            chunk_scorer=self._score_nonempty_chunks,
        )
        return scores[self.metric_name]

    def semantic_metadata(self) -> Mapping[str, Any]:
        return {
            "long_text_policy": LONG_TEXT_POLICY,
            "chunk_audit": self._chunk_audit,
        }


def score_rows(
    aligned_rows: Sequence[dict[str, Any]],
    *,
    semantic_scorers: Sequence[SemanticScorer] = (),
    scoring_policy: str = STRICT_SCORING_POLICY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    _scoring_policy_spec(scoring_policy)
    row_scores: list[dict[str, Any]] = []
    claim_checks: list[dict[str, Any]] = []
    semantic_positions: list[int] = []
    semantic_candidates: list[str] = []
    semantic_references: list[str] = []
    invalid_positions: list[int] = []

    for aligned in aligned_rows:
        generation_row = aligned["generation_row"]
        reference_row = aligned["reference_row"]
        evidence_row = aligned["evidence_row"]
        base = {
            "schema_version": SCHEMA_VERSION,
            "artifact_id": aligned["artifact_id"],
            "sample_id": aligned["sample_id"],
            "meeting_id": aligned["meeting_id"],
            "section_name": _nonblank(
                (reference_row or {}).get("section_name")
                or (evidence_row or {}).get("section_name")
                or (generation_row or {}).get("section_name")
            )
            or None,
            "alignment_status": aligned["alignment_status"],
            "missing_inputs": list(aligned["missing_inputs"]),
            "scoring_policy": scoring_policy,
        }
        if generation_row is None or reference_row is None or evidence_row is None:
            row_scores.append(
                {
                    **base,
                    "status": "unscored_missing_input",
                    "invalid_reasons": list(aligned["missing_inputs"]),
                    "generated_text": None,
                    "reference_text": (
                        _select_text(reference_row, kind="reference")
                        if reference_row is not None
                        else None
                    ),
                }
            )
            continue

        extraction_audit: dict[str, Any] = {}
        if scoring_policy == LENGTH_TOLERANT_SCORING_POLICY:
            raw_completion = _select_raw_generation_completion(generation_row)
            extracted = extract_length_tolerant_candidate(raw_completion)
            generated = extracted.text
            finish_reason = _nonblank(
                generation_row.get("generation_finish_reason")
            )
            upstream_invalid_reasons = generation_row.get("invalid_reasons", [])
            if isinstance(upstream_invalid_reasons, str):
                upstream_invalid_reasons = [upstream_invalid_reasons]
            extraction_audit = {
                "candidate_extraction_mode": extracted.extraction_mode,
                "has_leading_think_opening_tag": (
                    extracted.has_leading_think_tag
                ),
                "has_answer_opening_tag": extracted.has_answer_opening_tag,
                "has_trailing_answer_closing_tag": (
                    extracted.has_trailing_answer_closing_tag
                ),
                "source_generation_text_sha256": hashlib.sha256(
                    raw_completion.encode("utf-8")
                ).hexdigest(),
                "source_generation_character_count": len(raw_completion),
                "source_generation_finish_reason": finish_reason or None,
                "source_generation_hit_token_limit": finish_reason.casefold()
                in {"length", "max_tokens", "token_limit"},
                "upstream_valid_generation": generation_row.get(
                    "valid_generation"
                ),
                "ignored_upstream_invalid_reasons": list(
                    upstream_invalid_reasons
                ),
            }
            fatal_reasons = _length_tolerant_fatal_reasons(
                generated,
                generation_row,
            )
        else:
            generated = _select_text(generation_row, kind="generation")
            fatal_reasons = _generation_fatal_reasons(
                generated,
                generation_row,
            )
        reference = _select_text(reference_row, kind="reference")
        format_score, format_violations = check_format(
            generated,
            generation_row,
            reference_row,
            evidence_row,
        )
        factual_metrics, checks = evaluate_factual_rules(generated, evidence_row)
        generated_tokens = _tokens(generated)
        reference_tokens = _tokens(reference)
        sentence_count = len(
            [
                match.group(0)
                for match in SENTENCE_RE.finditer(generated)
                if match.group(0).strip()
            ]
        )
        invalid = bool(fatal_reasons)
        row: dict[str, Any] = {
            **base,
            "status": "invalid_output" if invalid else "scored",
            "invalid_reasons": fatal_reasons,
            "fatal_generation_invalid": invalid,
            "format_violations": format_violations,
            "valid_output": 0.0 if invalid else 1.0,
            "generated_text": generated,
            "reference_text": reference,
            "generated_text_sha256": hashlib.sha256(
                generated.encode("utf-8")
            ).hexdigest(),
            "reference_text_sha256": hashlib.sha256(
                reference.encode("utf-8")
            ).hexdigest(),
            # Invalid rows receive the predeclared finite worst case below.
            # Avoid the quadratic ROUGE-L dynamic program for long token-limit
            # completions whose score cannot affect the result.
            "rouge_l_f1": 0.0 if invalid else rouge_l_f1(generated, reference),
            "generated_token_count": len(generated_tokens),
            "generated_character_count": len(generated),
            "generated_sentence_count": sentence_count,
            "reference_token_count": len(reference_tokens),
            "length_ratio": (
                len(generated_tokens) / len(reference_tokens)
                if reference_tokens
                else None
            ),
            "repetition_rate": 1.0 if invalid else repetition_rate(generated),
            "format_compliance": format_score,
            **extraction_audit,
            **factual_metrics,
        }
        if invalid:
            # Invalid generations are part of the predeclared matrix.  Giving
            # headline metrics finite worst-case values prevents an apparently
            # stronger artifact from improving merely by failing to emit rows.
            row.update(
                {
                    "rouge_l_f1": 0.0,
                    "repetition_rate": 1.0,
                    "format_compliance": 0.0,
                    "numeric_value_accuracy": (
                        0.0 if factual_metrics["evidence_numeric_fact_count"] else None
                    ),
                    "evidence_value_coverage": (
                        0.0 if factual_metrics["evidence_numeric_fact_count"] else None
                    ),
                    "unit_accuracy": (
                        0.0 if factual_metrics["evidence_unit_fact_count"] else None
                    ),
                    "time_accuracy": (
                        0.0 if factual_metrics["evidence_time_fact_count"] else None
                    ),
                    "novel_number_rate": 1.0,
                    "rule_covered_unsupported_rate": 1.0,
                    "direction_consistency": (
                        0.0
                        if factual_metrics["expected_direction_topic_count"]
                        or factual_metrics["direction_coverable_claim_count"]
                        else None
                    ),
                    "direction_coverage": (
                        0.0
                        if factual_metrics["expected_direction_topic_count"]
                        else None
                    ),
                    "policy_stance_consistency": (
                        0.0
                        if factual_metrics["expected_policy_stance_available"]
                        or factual_metrics["policy_stance_coverable_claim_count"]
                        else None
                    ),
                    "policy_stance_coverage": (
                        0.0
                        if factual_metrics["expected_policy_stance_available"]
                        else None
                    ),
                }
            )
        for field in (
            "checkpoint_sha256",
            "parent_artifact_id",
            "test_set_sha256",
            "prompt_template_sha256",
            "decoding_config_sha256",
            *PER_SAMPLE_COMPARABILITY_FIELDS,
        ):
            value = _provenance_value(generation_row, field)
            if value is not None:
                row[field] = value
        position = len(row_scores)
        row_scores.append(row)
        if invalid:
            invalid_positions.append(position)
        else:
            semantic_positions.append(position)
            semantic_candidates.append(generated)
            semantic_references.append(reference)
        for check_index, check in enumerate(checks, start=1):
            claim_checks.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "artifact_id": aligned["artifact_id"],
                    "sample_id": aligned["sample_id"],
                    "meeting_id": aligned["meeting_id"],
                    "claim_check_id": (
                        f"{aligned['artifact_id']}::{aligned['sample_id']}::{check_index}"
                    ),
                    **check,
                }
            )

    semantic_metadata = []
    metric_names: set[str] = set()
    for scorer in semantic_scorers:
        if not semantic_candidates:
            expected_metrics: tuple[str, ...] = ()
            if isinstance(scorer, BERTScoreBackend):
                expected_metrics = (
                    "bertscore_precision",
                    "bertscore_recall",
                    "bertscore_f1",
                )
            elif isinstance(scorer, MPNetCosineBackend):
                expected_metrics = (scorer.metric_name,)
            for metric_name in expected_metrics:
                METRIC_DIRECTIONS.setdefault(metric_name, "higher")
                for position in invalid_positions:
                    row_scores[position][metric_name] = 0.0
            semantic_metadata.append(
                {
                    **_semantic_backend_metadata(scorer),
                    "metrics": list(expected_metrics),
                    "inference_skipped": "no_nonfatal_generation_rows",
                }
            )
            continue
        scores = _call_semantic_scorer(
            scorer,
            semantic_candidates,
            semantic_references,
        )
        metadata = _semantic_backend_metadata(scorer)
        overlap = metric_names & set(scores)
        if overlap:
            raise ValueError(
                f"Semantic metric names collide across scorers: {sorted(overlap)}"
            )
        metric_names.update(scores)
        semantic_metadata.append({**metadata, "metrics": sorted(scores)})
        for metric_name, values in scores.items():
            METRIC_DIRECTIONS.setdefault(metric_name, "higher")
            for position, value in zip(semantic_positions, values, strict=True):
                row_scores[position][metric_name] = value
            for position in invalid_positions:
                row_scores[position][metric_name] = 0.0
    return row_scores, claim_checks, semantic_metadata


def _finite_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> list[tuple[str, float]]:
    result = []
    for row in rows:
        value = row.get(metric)
        if value is None or isinstance(value, bool):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            result.append((str(row["meeting_id"]), numeric))
    return result


def _meeting_means(values: Sequence[tuple[str, float]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for meeting_id, value in values:
        grouped[meeting_id].append(value)
    return {
        meeting_id: float(np.mean(meeting_values))
        for meeting_id, meeting_values in sorted(grouped.items())
    }


def cluster_bootstrap_interval(
    meeting_values: Mapping[str, float],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> tuple[float | None, float | None]:
    """Percentile CI obtained by resampling meeting clusters."""

    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    values = np.asarray(list(meeting_values.values()), dtype=float)
    if len(values) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    alpha = 1.0 - confidence
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def summarise_rows(
    row_scores: Sequence[dict[str, Any]],
    *,
    subsets: Mapping[str, set[str] | frozenset[str] | None] | None = None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    if not row_scores:
        raise ValueError("No row scores are available to summarise")
    artifact_ids = sorted({str(row["artifact_id"]) for row in row_scores})
    excluded_fields = {
        "schema_version",
        "artifact_id",
        "sample_id",
        "meeting_id",
        "section_name",
        "alignment_status",
        "missing_inputs",
        "status",
        "invalid_reasons",
        "fatal_generation_invalid",
        "format_violations",
        "generated_text",
        "reference_text",
        "generated_text_sha256",
        "reference_text_sha256",
        "is_prospective_only",
        "checkpoint_sha256",
        "parent_artifact_id",
        "test_set_sha256",
        "prompt_template_sha256",
        "decoding_config_sha256",
        *PER_SAMPLE_COMPARABILITY_FIELDS,
    }
    metric_names = sorted(
        {
            key
            for row in row_scores
            for key, value in row.items()
            if key not in excluded_fields
            and (
                value is None
                or (
                    isinstance(value, (int, float, np.integer, np.floating))
                    and not isinstance(value, bool)
                )
            )
        }
    )
    subset_plan: Mapping[str, set[str] | frozenset[str] | None] = (
        subsets if subsets is not None else {"all": None}
    )
    if not subset_plan or any(not _nonblank(name) for name in subset_plan):
        raise ValueError("At least one named evaluation subset is required")
    summaries: list[dict[str, Any]] = []
    for subset_name, meeting_filter in subset_plan.items():
        for artifact_id in artifact_ids:
            rows = [
                row
                for row in row_scores
                if row["artifact_id"] == artifact_id
                and (meeting_filter is None or str(row["meeting_id"]) in meeting_filter)
            ]
            if not rows:
                raise ValueError(
                    f"Evaluation subset {subset_name!r} has no rows for "
                    f"artifact {artifact_id!r}"
                )
            status_counts = Counter(str(row["status"]) for row in rows)
            for metric in metric_names:
                finite = _finite_metric_rows(rows, metric)
                meeting_values = _meeting_means(finite)
                ci_lower, ci_upper = cluster_bootstrap_interval(
                    meeting_values,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed,
                )
                values = list(meeting_values.values())
                summaries.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "evaluation_subset": str(subset_name),
                        "artifact_id": artifact_id,
                        "metric": metric,
                        "metric_direction": METRIC_DIRECTIONS.get(
                            metric, "descriptive"
                        ),
                        "aggregation": "meeting_equal_weight",
                        "n_rows_total": len(rows),
                        "n_rows_eligible": len(finite),
                        "n_rows_ineligible": len(rows) - len(finite),
                        "n_meetings_total": len(
                            {str(row["meeting_id"]) for row in rows}
                        ),
                        "n_meetings": len(meeting_values),
                        "n_missing_input": status_counts["unscored_missing_input"],
                        "n_invalid_output": status_counts["invalid_output"],
                        "n_valid_output": status_counts["scored"],
                        "mean": float(np.mean(values)) if values else None,
                        "median": float(np.median(values)) if values else None,
                        "std_between_meetings": (
                            float(np.std(values, ddof=1)) if len(values) > 1 else None
                        ),
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "confidence": 0.95,
                        "bootstrap_unit": "meeting_id",
                        "bootstrap_samples": bootstrap_samples,
                        "bootstrap_seed": bootstrap_seed,
                    }
                )
    return summaries


def holm_adjust(p_values: Sequence[float | None]) -> list[float | None]:
    adjusted: list[float | None] = [None] * len(p_values)
    valid = [
        (index, float(value))
        for index, value in enumerate(p_values)
        if value is not None and math.isfinite(float(value))
    ]
    valid.sort(key=lambda item: item[1])
    running_max = 0.0
    total = len(valid)
    for rank, (original_index, p_value) in enumerate(valid):
        candidate = min(1.0, (total - rank) * p_value)
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted


def _sign_flip_p_value(
    differences: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float | None, str]:
    values = np.asarray(differences, dtype=float)
    if len(values) < 2:
        return None, "insufficient_meeting_clusters"
    observed = abs(float(values.mean()))
    if np.all(values == 0):
        return 1.0, "exact_zero_difference"
    if len(values) <= 16:
        combinations = 1 << len(values)
        indexes = np.arange(combinations, dtype=np.uint64)[:, None]
        bits = (indexes >> np.arange(len(values), dtype=np.uint64)) & 1
        signs = np.where(bits == 1, 1.0, -1.0)
        null_means = np.abs((signs * values).mean(axis=1))
        return float(np.mean(null_means >= observed - 1e-15)), "exact_sign_flip"
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(samples, len(values)))
    null_means = np.abs((signs * values).mean(axis=1))
    extreme = int(np.sum(null_means >= observed - 1e-15))
    return (extreme + 1) / (samples + 1), "monte_carlo_sign_flip"


def _load_lineage(path: str | Path | None) -> dict[str, str | None]:
    if path is None:
        return {}
    lineage_path = Path(path)
    try:
        payload = json.loads(lineage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load lineage manifest {lineage_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Lineage manifest must be a JSON object")
    entries: Any = payload.get("checkpoints", payload)
    if isinstance(entries, list):
        entries = {
            _nonblank(item.get("artifact_id") or item.get("checkpoint_id")): item
            for item in entries
            if isinstance(item, Mapping)
        }
    if not isinstance(entries, Mapping):
        raise ValueError("Lineage manifest checkpoints must be an object or list")
    result: dict[str, str | None] = {}
    for artifact_id, item in entries.items():
        identifier = _nonblank(artifact_id)
        if not identifier or not isinstance(item, Mapping):
            continue
        parent = _nonblank(
            item.get("parent_artifact_id")
            or item.get("parent_checkpoint_id")
            or item.get("parent")
        )
        result[identifier] = parent or None
    return result


def _normalize_contrast_plan(
    contrasts: Sequence[Mapping[str, Any] | Sequence[str]] | None,
    *,
    artifact_ids: Sequence[str],
    parents: Mapping[str, str | None],
) -> list[dict[str, Any]]:
    if contrasts is None:
        return [
            {
                "contrast_id": f"{candidate}_minus_{baseline}",
                "baseline_artifact_id": baseline,
                "candidate_artifact_id": candidate,
                "kind": "parent_child",
            }
            for candidate, baseline in sorted(parents.items())
            if baseline in artifact_ids and candidate in artifact_ids
        ]
    normalized = []
    for index, item in enumerate(contrasts):
        if isinstance(item, Mapping):
            row = dict(item)
            baseline = _nonblank(row.get("baseline_artifact_id") or row.get("baseline"))
            candidate = _nonblank(
                row.get("candidate_artifact_id") or row.get("candidate")
            )
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            values = list(item)
            if len(values) != 2:
                raise ValueError(
                    "Tuple/list contrasts must contain baseline and candidate"
                )
            baseline, candidate = map(_nonblank, values)
            row = {}
        else:
            raise ValueError("Each contrast must be an object or a two-item sequence")
        if not baseline or not candidate or baseline == candidate:
            raise ValueError(f"Invalid contrast at position {index}: {item}")
        normalized.append(
            {
                **row,
                "contrast_id": _nonblank(row.get("contrast_id"))
                or f"{candidate}_minus_{baseline}",
                "baseline_artifact_id": baseline,
                "candidate_artifact_id": candidate,
                "kind": _nonblank(row.get("kind")) or "parent_child",
            }
        )
    identifiers = [row["contrast_id"] for row in normalized]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Duplicate contrast_id values: {identifiers}")
    return normalized


def _validate_contrast(
    contrast: Mapping[str, Any],
    *,
    artifact_ids: set[str],
    parents: Mapping[str, str | None],
    provenance: Mapping[str, Mapping[str, Any]],
    rows_by_artifact: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    baseline = str(contrast["baseline_artifact_id"])
    candidate = str(contrast["candidate_artifact_id"])
    unknown = {baseline, candidate} - artifact_ids
    if unknown:
        raise ValueError(
            f"Contrast {contrast['contrast_id']!r} references unknown artifacts: "
            f"{sorted(unknown)}"
        )
    kind = str(contrast["kind"])
    if kind == "parent_child":
        if parents.get(candidate) != baseline:
            raise ValueError(
                f"Illegal parent_child contrast {contrast['contrast_id']!r}: "
                f"{candidate!r} has parent {parents.get(candidate)!r}, not {baseline!r}"
            )
    elif kind == "sibling":
        if not bool(contrast.get("pre_registered")):
            raise ValueError("Sibling contrasts must set pre_registered=true")
        if not parents.get(candidate) or parents.get(candidate) != parents.get(
            baseline
        ):
            raise ValueError(
                f"Illegal sibling contrast {contrast['contrast_id']!r}: parents differ"
            )
    elif kind == "planned":
        if not bool(contrast.get("pre_registered")) or not _nonblank(
            contrast.get("rationale")
        ):
            raise ValueError(
                "A planned non-lineage contrast requires pre_registered=true and rationale"
            )
    else:
        raise ValueError(
            f"Contrast {contrast['contrast_id']!r} has unsupported kind {kind!r}"
        )

    baseline_samples = set(rows_by_artifact[baseline])
    candidate_samples = set(rows_by_artifact[candidate])
    if baseline_samples != candidate_samples:
        raise ValueError(
            f"Contrast {contrast['contrast_id']!r} has unequal sample sets: "
            f"baseline-only={sorted(baseline_samples - candidate_samples)}, "
            f"candidate-only={sorted(candidate_samples - baseline_samples)}"
        )
    for field in COMMON_PROVENANCE_FIELDS:
        baseline_value = provenance[baseline].get(field)
        candidate_value = provenance[candidate].get(field)
        if baseline_value != candidate_value:
            raise ValueError(
                f"Contrast {contrast['contrast_id']!r} is not comparable: "
                f"{field} differs ({baseline_value!r} vs {candidate_value!r})"
            )
    for sample_id in sorted(baseline_samples):
        baseline_row = rows_by_artifact[baseline][sample_id]
        candidate_row = rows_by_artifact[candidate][sample_id]
        for field in PER_SAMPLE_COMPARABILITY_FIELDS:
            baseline_value = baseline_row.get(field)
            candidate_value = candidate_row.get(field)
            if (
                baseline_value is not None
                and candidate_value is not None
                and _canonical_scalar(baseline_value)
                != _canonical_scalar(candidate_value)
            ):
                raise ValueError(
                    f"Contrast {contrast['contrast_id']!r}, sample {sample_id!r} "
                    f"has different {field}: {baseline_value!r} vs {candidate_value!r}"
                )


def contrast_rows(
    row_scores: Sequence[dict[str, Any]],
    *,
    provenance: Mapping[str, Mapping[str, Any]],
    lineage: Mapping[str, str | None],
    contrasts: Sequence[Mapping[str, Any] | Sequence[str]] | None = None,
    metrics: Sequence[str] | None = None,
    subsets: Mapping[str, set[str] | frozenset[str] | None] | None = None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    artifact_ids = sorted({str(row["artifact_id"]) for row in row_scores})
    parents = dict(lineage)
    for artifact_id in artifact_ids:
        declared = _nonblank(provenance.get(artifact_id, {}).get("parent_artifact_id"))
        if declared:
            if artifact_id in parents and parents[artifact_id] != declared:
                raise ValueError(
                    f"Lineage conflict for {artifact_id!r}: manifest says "
                    f"{parents[artifact_id]!r}, generation provenance says {declared!r}"
                )
            parents[artifact_id] = declared
    for artifact_id in artifact_ids:
        if parents.get(artifact_id) == artifact_id:
            raise ValueError(f"Lineage self-cycle at artifact {artifact_id!r}")
        visited: set[str] = set()
        current: str | None = artifact_id
        while current is not None and current in parents:
            if current in visited:
                raise ValueError(
                    f"Lineage cycle detected while traversing artifact {artifact_id!r}"
                )
            visited.add(current)
            current = parents.get(current)
    plan = _normalize_contrast_plan(
        contrasts,
        artifact_ids=artifact_ids,
        parents=parents,
    )
    rows_by_artifact: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in row_scores:
        rows_by_artifact[str(row["artifact_id"])][str(row["sample_id"])] = row
    for contrast in plan:
        _validate_contrast(
            contrast,
            artifact_ids=set(artifact_ids),
            parents=parents,
            provenance=provenance,
            rows_by_artifact=rows_by_artifact,
        )

    if metrics is None:
        excluded = {
            "numeric_claim_count",
            "evidence_numeric_fact_count",
            "evidence_unit_fact_count",
            "evidence_time_fact_count",
            "unit_check_count",
            "time_check_count",
            "number_mention_count",
            "novel_number_count",
            "rule_covered_claim_count",
            "rule_covered_unsupported_count",
            "direction_claim_count",
            "direction_coverable_claim_count",
            "expected_direction_topic_count",
            "policy_stance_claim_count",
            "policy_stance_coverable_claim_count",
            "expected_policy_stance_available",
            "reference_token_count",
        }
        metrics = sorted(
            metric
            for metric in {
                key for row in row_scores for key in row if key in METRIC_DIRECTIONS
            }
            if metric not in excluded
        )

    subset_plan: Mapping[str, set[str] | frozenset[str] | None] = (
        subsets if subsets is not None else {"all": None}
    )
    if not subset_plan:
        raise ValueError("At least one named evaluation subset is required")
    results: list[dict[str, Any]] = []
    for subset_name, meeting_filter in subset_plan.items():
        family_name = (
            "all_prespecified_contrast_metric_tests"
            if subsets is None
            else (f"{subset_name}::all_prespecified_contrast_metric_tests")
        )
        family_rows: list[dict[str, Any]] = []
        for contrast in plan:
            baseline_id = str(contrast["baseline_artifact_id"])
            candidate_id = str(contrast["candidate_artifact_id"])
            shared_samples = sorted(
                sample_id
                for sample_id in (
                    set(rows_by_artifact[baseline_id])
                    & set(rows_by_artifact[candidate_id])
                )
                if (
                    meeting_filter is None
                    or str(rows_by_artifact[baseline_id][sample_id]["meeting_id"])
                    in meeting_filter
                )
            )
            if not shared_samples:
                raise ValueError(
                    f"Evaluation subset {subset_name!r} has no common samples for "
                    f"contrast {contrast['contrast_id']!r}"
                )
            for metric in metrics:
                paired: list[tuple[str, float, float]] = []
                for sample_id in shared_samples:
                    baseline_row = rows_by_artifact[baseline_id][sample_id]
                    candidate_row = rows_by_artifact[candidate_id][sample_id]
                    baseline_value = baseline_row.get(metric)
                    candidate_value = candidate_row.get(metric)
                    try:
                        baseline_numeric = float(baseline_value)
                        candidate_numeric = float(candidate_value)
                    except (TypeError, ValueError):
                        continue
                    if not (
                        math.isfinite(baseline_numeric)
                        and math.isfinite(candidate_numeric)
                    ):
                        continue
                    if baseline_row["meeting_id"] != candidate_row["meeting_id"]:
                        raise ValueError(
                            f"Contrast {contrast['contrast_id']!r}, sample "
                            f"{sample_id!r} has inconsistent meeting_id"
                        )
                    paired.append(
                        (
                            str(baseline_row["meeting_id"]),
                            baseline_numeric,
                            candidate_numeric,
                        )
                    )
                by_meeting: dict[str, list[tuple[float, float]]] = defaultdict(list)
                for meeting_id, baseline_value, candidate_value in paired:
                    by_meeting[meeting_id].append((baseline_value, candidate_value))
                baseline_meeting = {
                    meeting_id: float(np.mean([value[0] for value in values]))
                    for meeting_id, values in sorted(by_meeting.items())
                }
                candidate_meeting = {
                    meeting_id: float(np.mean([value[1] for value in values]))
                    for meeting_id, values in sorted(by_meeting.items())
                }
                differences = {
                    meeting_id: (
                        candidate_meeting[meeting_id] - baseline_meeting[meeting_id]
                    )
                    for meeting_id in baseline_meeting
                }
                difference_values = list(differences.values())
                ci_lower, ci_upper = cluster_bootstrap_interval(
                    differences,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed,
                )
                p_value, p_method = _sign_flip_p_value(
                    difference_values,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed,
                )
                std = (
                    float(np.std(difference_values, ddof=1))
                    if len(difference_values) > 1
                    else None
                )
                effect_size = (
                    float(np.mean(difference_values)) / std
                    if std not in (None, 0.0)
                    else None
                )
                direction = METRIC_DIRECTIONS.get(metric, "descriptive")
                mean_difference = (
                    float(np.mean(difference_values)) if difference_values else None
                )
                family_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "evaluation_subset": str(subset_name),
                        "contrast_id": contrast["contrast_id"],
                        "contrast_kind": contrast["kind"],
                        "baseline_artifact_id": baseline_id,
                        "candidate_artifact_id": candidate_id,
                        "metric": metric,
                        "metric_direction": direction,
                        "aggregation": (
                            "paired_within_sample_then_meeting_equal_weight"
                        ),
                        "n_common_samples_total": len(shared_samples),
                        "n_pairs_eligible": len(paired),
                        "n_pairs_ineligible": len(shared_samples) - len(paired),
                        "n_meetings_total": len(
                            {
                                str(
                                    rows_by_artifact[baseline_id][sample_id][
                                        "meeting_id"
                                    ]
                                )
                                for sample_id in shared_samples
                            }
                        ),
                        "n_meetings": len(differences),
                        "mean_baseline": (
                            float(np.mean(list(baseline_meeting.values())))
                            if baseline_meeting
                            else None
                        ),
                        "mean_candidate": (
                            float(np.mean(list(candidate_meeting.values())))
                            if candidate_meeting
                            else None
                        ),
                        "paired_mean_difference": mean_difference,
                        "benefit_difference": (
                            -mean_difference
                            if mean_difference is not None and direction == "lower"
                            else mean_difference
                        ),
                        "benefit_ci_lower": (
                            -ci_upper
                            if ci_upper is not None and direction == "lower"
                            else ci_lower
                        ),
                        "benefit_ci_upper": (
                            -ci_lower
                            if ci_lower is not None and direction == "lower"
                            else ci_upper
                        ),
                        "effect_size_paired_dz": effect_size,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "confidence": 0.95,
                        "bootstrap_unit": "meeting_id",
                        "bootstrap_samples": bootstrap_samples,
                        "bootstrap_seed": bootstrap_seed,
                        "p_value": p_value,
                        "p_value_method": p_method,
                        "p_value_holm": None,
                        "holm_family": family_name,
                        "holm_family_size": None,
                        "inference_status": (
                            "available"
                            if p_value is not None and ci_lower is not None
                            else "insufficient_meeting_clusters"
                        ),
                    }
                )
        adjusted = holm_adjust([row["p_value"] for row in family_rows])
        family_size = sum(value is not None for value in adjusted)
        for row, value in zip(family_rows, adjusted, strict=True):
            row["p_value_holm"] = value
            row["holm_family_size"] = family_size
        results.extend(family_rows)
    return results, plan


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def _json_payload_sha256(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    _json_safe(row),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                _json_safe(payload),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


def _read_output_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid existing evaluation output at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Existing evaluation output row is not an object at "
                    f"{path}:{line_number}"
                )
            rows.append(row)
    return rows


def _semantic_backend_identity(scorer: SemanticScorer) -> dict[str, Any]:
    identity = {
        "scorer_id": _nonblank(getattr(scorer, "scorer_id", scorer.__class__.__name__)),
        "backend_class": (
            f"{scorer.__class__.__module__}.{scorer.__class__.__qualname__}"
        ),
        "model_id": _nonblank(getattr(scorer, "model_id", "injected")),
        "model_sha256": _nonblank(getattr(scorer, "model_sha256", "")) or None,
        "long_text_policy": (
            LONG_TEXT_POLICY
            if isinstance(scorer, (BERTScoreBackend, MPNetCosineBackend))
            else None
        ),
    }
    for field in (
        "num_layers",
        "batch_size",
        "device",
        "max_length",
        "metric_name",
    ):
        if hasattr(scorer, field):
            identity[field] = getattr(scorer, field)
    extra_identity = getattr(scorer, "evaluation_identity", None)
    if callable(extra_identity):
        extra = extra_identity()
        if not isinstance(extra, Mapping):
            raise ValueError("evaluation_identity() must return a mapping")
        identity["injected_identity"] = dict(extra)
    return identity


def _reuse_existing_evaluation(
    paths: Mapping[str, Path],
    *,
    run_spec_sha256: str,
) -> dict[str, Any] | None:
    existence = {name: path.exists() for name, path in paths.items()}
    if not any(existence.values()):
        return None
    if not all(existence.values()):
        raise FileExistsError(
            "Refusing to overwrite a partial immutable evaluation output set: "
            f"{existence}"
        )

    audit, _ = _read_json_object(
        paths["audit"],
        kind="existing evaluation audit",
        require_integrity=True,
    )
    if audit.get("status") != "validated":
        raise ValueError("Existing evaluation audit is not validated")
    stored_run_spec = audit.get("run_spec")
    if not isinstance(stored_run_spec, Mapping) or audit.get(
        "run_spec_sha256"
    ) != _json_payload_sha256(stored_run_spec):
        raise ValueError("Existing evaluation audit has an invalid run-spec digest")
    if audit.get("run_spec_sha256") != run_spec_sha256:
        raise FileExistsError(
            "Immutable evaluation outputs already exist for a different run "
            "specification; choose a new --output-dir"
        )
    output_records = audit.get("output_artifacts")
    if not isinstance(output_records, Mapping):
        raise ValueError("Existing evaluation audit lacks output_artifacts")

    loaded: dict[str, list[dict[str, Any]]] = {}
    for name in ("row_scores", "claim_checks", "summary", "contrasts"):
        path = paths[name]
        record = output_records.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"Existing evaluation audit lacks {name!r} binding")
        rows = _read_output_jsonl(path)
        if record.get("sha256") != _sha256_file(path) or record.get("row_count") != len(
            rows
        ):
            raise ValueError(f"Existing immutable evaluation output changed: {name}")
        loaded[name] = rows
    return {
        **loaded,
        "audit": audit,
        "paths": {key: str(path) for key, path in paths.items()},
        "reused_existing": True,
    }


def _load_contrast_plan(path: str | Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    contrast_path = Path(path)
    try:
        payload = json.loads(contrast_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load contrast plan {contrast_path}: {exc}") from exc
    if isinstance(payload, Mapping):
        payload = payload.get("contrasts")
    if not isinstance(payload, list):
        raise ValueError("Contrast plan must be a JSON list or {'contrasts': [...]}")
    if any(not isinstance(item, Mapping) for item in payload):
        raise ValueError("Every contrast-plan entry must be a JSON object")
    return [dict(item) for item in payload]


def _mapping_field(
    row: Mapping[str, Any],
    field: str,
    *,
    label: str,
) -> Mapping[str, Any]:
    value = row.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}.{field} must be a JSON object")
    return value


def _string_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{label} must be a list of non-empty strings")
    result = [_nonblank(item) for item in value]
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicates")
    return result


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer")
    return result


def _assert_file_binding(
    record: Any,
    actual: Mapping[str, Any],
    *,
    label: str,
    expected_row_count: int | None = None,
) -> None:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} binding must be a JSON object")
    declared_sha256 = _validate_sha256(record.get("sha256"), field=f"{label}.sha256")
    if declared_sha256 != actual["sha256"]:
        raise ValueError(
            f"{label} binding changed: manifest={declared_sha256}, "
            f"supplied={actual['sha256']}"
        )
    if expected_row_count is not None and record.get("row_count") != expected_row_count:
        raise ValueError(
            f"{label}.row_count must be {expected_row_count}, "
            f"found {record.get('row_count')!r}"
        )


def _resolve_semantic_model_path(
    value: Any,
    *,
    manifest_path: Path,
    label: str,
) -> Path:
    raw_text = _nonblank(value)
    if not raw_text:
        raise ValueError(f"{label}.local_path must be non-empty")
    raw = Path(raw_text).expanduser()
    candidates = (
        [raw]
        if raw.is_absolute()
        else [Path.cwd() / raw, manifest_path.parent / raw]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved
    raise ValueError(f"{label}.local_path does not resolve to a directory: {raw_text}")


def _validate_formal_semantic_backends(
    *,
    semantic_manifest_path: str | Path,
    semantic_scorers: Sequence[SemanticScorer],
    evaluation_id: Any,
) -> dict[str, Any]:
    """Bind formal scoring to exactly the two frozen, independent encoders."""

    manifest_path = Path(semantic_manifest_path).expanduser().resolve()
    manifest, manifest_input = _read_json_object(
        manifest_path,
        kind="semantic model manifest",
        require_integrity=True,
    )
    if manifest.get("schema_version") != SEMANTIC_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Semantic model manifest has an unsupported schema_version")
    if manifest.get("created_for_evaluation_id") != evaluation_id:
        raise ValueError(
            "Semantic model manifest created_for_evaluation_id differs from config"
        )
    if manifest.get("network_at_scoring_time") is not False:
        raise ValueError(
            "Formal semantic model manifest must declare network_at_scoring_time=false"
        )
    models = _mapping_field(manifest, "models", label="semantic model manifest")
    if set(models) != {"bertscore", "embedding_cosine"}:
        raise ValueError(
            "Formal semantic manifest must contain exactly bertscore and "
            "embedding_cosine models"
        )
    scorer_by_id: dict[str, SemanticScorer] = {}
    for scorer in semantic_scorers:
        scorer_id = _nonblank(getattr(scorer, "scorer_id", ""))
        if not scorer_id or scorer_id in scorer_by_id:
            raise ValueError(
                "Formal semantic scorers require unique non-empty scorer_id values"
            )
        scorer_by_id[scorer_id] = scorer
    if set(scorer_by_id) != {"bertscore", "mpnet"}:
        raise ValueError(
            "Formal evaluation requires exactly one frozen BERTScore scorer and "
            "one independent MPNet scorer"
        )
    if not isinstance(scorer_by_id["bertscore"], BERTScoreBackend):
        raise ValueError("Formal scorer_id='bertscore' must use BERTScoreBackend")
    if not isinstance(scorer_by_id["mpnet"], MPNetCosineBackend):
        raise ValueError("Formal scorer_id='mpnet' must use MPNetCosineBackend")

    checked_models: dict[str, Any] = {}
    specifications = (
        ("bertscore", "bertscore", BERTScoreBackend),
        ("embedding_cosine", "mpnet", MPNetCosineBackend),
    )
    for manifest_key, scorer_id, _backend_type in specifications:
        record = _mapping_field(
            models,
            manifest_key,
            label="semantic model manifest.models",
        )
        expected_hash = _validate_sha256(
            record.get("directory_sha256"),
            field=(
                "semantic model manifest.models."
                f"{manifest_key}.directory_sha256"
            ),
        )
        model_path = _resolve_semantic_model_path(
            record.get("local_path"),
            manifest_path=manifest_path,
            label=f"semantic model manifest.models.{manifest_key}",
        )
        observed_hash = _sha256_path(model_path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"Frozen semantic model directory changed for {manifest_key}: "
                f"expected={expected_hash}, observed={observed_hash}"
            )
        scorer = scorer_by_id[scorer_id]
        scorer_path = Path(_nonblank(getattr(scorer, "model_id", ""))).resolve()
        if scorer_path != model_path:
            raise ValueError(
                f"Formal {scorer_id} scorer model path differs from semantic manifest"
            )
        scorer_hash = _validate_sha256(
            getattr(scorer, "model_sha256", None),
            field=f"semantic scorer[{scorer_id}].model_sha256",
        )
        if scorer_hash != expected_hash:
            raise ValueError(
                f"Formal {scorer_id} scorer hash differs from semantic manifest"
            )
        checked = {
            "scorer_id": scorer_id,
            "model_path": str(model_path),
            "directory_sha256": observed_hash,
            "repo_id": _nonblank(record.get("repo_id")),
            "resolved_revision": _nonblank(record.get("resolved_revision")),
        }
        if manifest_key == "bertscore":
            num_layers = _integer(
                record.get("num_layers"),
                label="semantic model manifest.models.bertscore.num_layers",
            )
            if num_layers < 1:
                raise ValueError("BERTScore num_layers must be positive")
            if getattr(scorer, "num_layers", None) != num_layers:
                raise ValueError(
                    "BERTScore scorer num_layers differs from semantic manifest"
                )
            checked["num_layers"] = num_layers
        else:
            if record.get("independent_from_training_reward") is not True:
                raise ValueError(
                    "Formal MPNet scorer must be declared independent_from_training_reward"
                )
            checked["independent_from_training_reward"] = True
        checked_models[manifest_key] = checked

    return {
        "manifest": manifest_input,
        "manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
        "models": checked_models,
        "required_scorer_ids": ["bertscore", "mpnet"],
        "complete": True,
    }


def _validate_formal_lineage_evidence(
    *,
    lineage_evidence_path: str | Path,
    checkpoint_manifest_path: str | Path,
    checkpoint_manifest: Mapping[str, Any],
    checkpoint_input: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the precomputed exact LoRA-merge evidence without tensor I/O."""

    evidence_path = Path(lineage_evidence_path).expanduser().resolve()
    evidence, evidence_input = _read_json_object(
        evidence_path,
        kind="exact-merge lineage evidence",
        require_integrity=True,
    )
    if evidence.get("schema_version") != LINEAGE_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("Exact-merge lineage evidence has unsupported schema_version")
    if evidence.get("conclusion") != "exact_base_plus_adapter_merge_verified":
        raise ValueError("Exact-merge lineage evidence conclusion is not verified")
    if evidence.get("subject_artifact_id") != "eval-analysis-sft":
        raise ValueError(
            "Exact-merge lineage evidence subject must be eval-analysis-sft"
        )
    if evidence.get("algorithm_version") != "peft-lora-fp32-exact-v1":
        raise ValueError("Exact-merge lineage evidence algorithm_version changed")

    bindings = _mapping_field(
        evidence,
        "bindings",
        label="exact-merge lineage evidence",
    )
    binding = _mapping_field(
        bindings,
        "checkpoint_manifest",
        label="exact-merge lineage evidence.bindings",
    )
    expected_checkpoint_path = Path(checkpoint_manifest_path).expanduser().resolve()
    if Path(_nonblank(binding.get("path"))).expanduser().resolve() != (
        expected_checkpoint_path
    ):
        raise ValueError(
            "Exact-merge lineage evidence binds a different checkpoint manifest path"
        )
    if (
        _validate_sha256(
            binding.get("sha256"),
            field="lineage evidence checkpoint manifest sha256",
        )
        != checkpoint_input["sha256"]
        or _validate_sha256(
            binding.get("payload_sha256"),
            field="lineage evidence checkpoint manifest payload_sha256",
        )
        != checkpoint_input["payload_sha256"]
    ):
        raise ValueError(
            "Exact-merge lineage evidence checkpoint manifest hash binding changed"
        )
    base_id = _nonblank(binding.get("base_artifact_id"))
    merged_id = _nonblank(binding.get("merged_artifact_id"))
    if (base_id, merged_id) != ("eval-base", "eval-analysis-sft"):
        raise ValueError(
            "Exact-merge lineage evidence base/merged artifact IDs changed"
        )
    assertions = _mapping_field(
        binding,
        "assertions",
        label="exact-merge lineage evidence checkpoint binding",
    )
    required_assertions = {
        "base_fingerprint_matches",
        "base_path_matches",
        "merged_fingerprint_matches",
        "merged_path_matches",
        "merged_verified_parent_matches_base",
    }
    if set(assertions) != required_assertions or any(
        value is not True for value in assertions.values()
    ):
        raise ValueError(
            "Exact-merge lineage evidence checkpoint assertions did not all pass"
        )

    records_raw = checkpoint_manifest.get("artifacts")
    if not isinstance(records_raw, list):
        raise ValueError("Checkpoint manifest artifacts must be a list")
    records = {
        _nonblank(record.get("artifact_id")): record
        for record in records_raw
        if isinstance(record, Mapping)
    }
    if base_id not in records or merged_id not in records:
        raise ValueError(
            "Exact-merge lineage evidence artifacts are absent from checkpoint manifest"
        )
    base_record = records[base_id]
    merged_record = records[merged_id]
    if (
        base_record.get("usable_for_evaluation") is not True
        or merged_record.get("usable_for_evaluation") is not True
        or _nonblank(merged_record.get("verified_parent_artifact_id")) != base_id
    ):
        raise ValueError(
            "Exact-merge lineage evidence conflicts with checkpoint parent/usable state"
        )
    sources = _mapping_field(
        evidence,
        "sources",
        label="exact-merge lineage evidence",
    )
    base_source = _mapping_field(
        sources,
        "base_model",
        label="exact-merge lineage evidence.sources",
    )
    merged_source = _mapping_field(
        sources,
        "merged_model",
        label="exact-merge lineage evidence.sources",
    )
    for label, source, record in (
        ("base", base_source, base_record),
        ("merged", merged_source, merged_record),
    ):
        source_sha256 = _validate_sha256(
            source.get("sha256"),
            field=f"lineage evidence {label} model sha256",
        )
        record_sha256 = _validate_sha256(
            record.get("model_sha256"),
            field=f"checkpoint {label} model_sha256",
        )
        tokenizer_sha256 = _validate_sha256(
            record.get("tokenizer_sha256"),
            field=f"checkpoint {label} tokenizer_sha256",
        )
        if source_sha256 != record_sha256 or source_sha256 != tokenizer_sha256:
            raise ValueError(
                f"Exact-merge lineage evidence {label} fingerprint differs "
                "from checkpoint manifest"
            )
        source_path = Path(_nonblank(source.get("path"))).expanduser().resolve()
        if (
            source_path
            != Path(_nonblank(record.get("model_path"))).expanduser().resolve()
            or source_path
            != Path(_nonblank(record.get("tokenizer_path"))).expanduser().resolve()
        ):
            raise ValueError(
                f"Exact-merge lineage evidence {label} model path differs "
                "from checkpoint manifest"
            )

    metadata = _mapping_field(
        evidence,
        "metadata_evidence",
        label="exact-merge lineage evidence",
    )
    adapter_metadata = _mapping_field(
        metadata,
        "adapter",
        label="exact-merge lineage evidence.metadata_evidence",
    )
    training_metadata = _mapping_field(
        metadata,
        "training_config",
        label="exact-merge lineage evidence.metadata_evidence",
    )
    metadata_assertions: dict[str, Any] = {
        "adapter.base_path_matches": adapter_metadata.get("base_path_matches"),
    }
    for group in ("lora_metadata_checks", "path_checks"):
        checks = _mapping_field(
            training_metadata,
            group,
            label="exact-merge lineage evidence.metadata_evidence.training_config",
        )
        metadata_assertions.update(
            {f"training_config.{group}.{key}": value for key, value in checks.items()}
        )
    if not metadata_assertions or any(
        value is not True for value in metadata_assertions.values()
    ):
        raise ValueError(
            "Exact-merge lineage evidence metadata assertions did not all pass"
        )

    tensor = _mapping_field(
        evidence,
        "tensor_verification",
        label="exact-merge lineage evidence",
    )
    adapted = _integer(
        tensor.get("adapted_model_tensor_count"),
        label="lineage evidence adapted_model_tensor_count",
    )
    exact_adapted = _integer(
        tensor.get("exact_adapted_model_tensor_count"),
        label="lineage evidence exact_adapted_model_tensor_count",
    )
    unchanged = _integer(
        tensor.get("unchanged_model_tensor_count"),
        label="lineage evidence unchanged_model_tensor_count",
    )
    exact_unchanged = _integer(
        tensor.get("exact_unchanged_model_tensor_count"),
        label="lineage evidence exact_unchanged_model_tensor_count",
    )
    model_tensor_count = _integer(
        tensor.get("model_tensor_count"),
        label="lineage evidence model_tensor_count",
    )
    mismatch_count = _integer(
        tensor.get("mismatch_count"),
        label="lineage evidence mismatch_count",
    )
    adapted_tensors = tensor.get("adapted_tensors")
    if not isinstance(adapted_tensors, list) or any(
        not isinstance(item, Mapping) for item in adapted_tensors
    ):
        raise ValueError("Exact-merge adapted_tensors must be a list of objects")
    tensor_names = [_nonblank(item.get("tensor_name")) for item in adapted_tensors]
    total_adapted_elements = _integer(
        tensor.get("total_adapted_elements"),
        label="lineage evidence total_adapted_elements",
    )
    total_unchanged_elements = _integer(
        tensor.get("total_unchanged_elements"),
        label="lineage evidence total_unchanged_elements",
    )
    changed_adapted_elements = _integer(
        tensor.get("changed_adapted_elements_vs_base"),
        label="lineage evidence changed_adapted_elements_vs_base",
    )
    if (
        adapted < 1
        or unchanged < 1
        or exact_adapted != adapted
        or exact_unchanged != unchanged
        or model_tensor_count != adapted + unchanged
        or mismatch_count != 0
        or len(adapted_tensors) != adapted
        or any(not name for name in tensor_names)
        or len(tensor_names) != len(set(tensor_names))
        or any(item.get("exact_reconstruction") is not True for item in adapted_tensors)
        or _integer(
            tensor.get("adapter_tensor_count"),
            label="lineage evidence adapter_tensor_count",
        )
        != 2 * adapted
        or total_adapted_elements < 1
        or total_unchanged_elements < 1
        or changed_adapted_elements < 1
        or changed_adapted_elements > total_adapted_elements
        or tensor.get("comparison") != "torch.equal (exact stored tensor values)"
        or not _nonblank(tensor.get("merge_expression"))
    ):
        raise ValueError(
            "Exact-merge lineage evidence tensor summary did not fully pass"
        )

    return {
        "evidence": evidence_input,
        "schema_version": evidence["schema_version"],
        "algorithm_version": evidence["algorithm_version"],
        "conclusion": evidence["conclusion"],
        "subject_artifact_id": evidence["subject_artifact_id"],
        "checkpoint_manifest_binding": {
            "path": str(expected_checkpoint_path),
            "sha256": checkpoint_input["sha256"],
            "payload_sha256": checkpoint_input["payload_sha256"],
            "base_artifact_id": base_id,
            "merged_artifact_id": merged_id,
            "base_model_sha256": base_source["sha256"],
            "merged_model_sha256": merged_source["sha256"],
            "assertions": dict(assertions),
        },
        "tensor_summary": {
            "model_tensor_count": model_tensor_count,
            "adapted_model_tensor_count": adapted,
            "exact_adapted_model_tensor_count": exact_adapted,
            "unchanged_model_tensor_count": unchanged,
            "exact_unchanged_model_tensor_count": exact_unchanged,
            "mismatch_count": mismatch_count,
            "all_adapted_tensors_exact": True,
        },
        "tensor_comparison_recomputed": False,
        "complete": True,
    }


def _validate_formal_input_matrix(
    *,
    generation_batches: Sequence[tuple[list[dict[str, Any]], Mapping[str, Any]]],
    generation_manifest_paths: Sequence[str | Path],
    prompt_rows: Sequence[dict[str, Any]],
    prompt_input: Mapping[str, Any],
    reference_rows: Sequence[dict[str, Any]],
    reference_input: Mapping[str, Any],
    evaluation_config_path: str | Path,
    test_manifest_path: str | Path,
    checkpoint_manifest_path: str | Path,
    semantic_manifest_path: str | Path,
    lineage_evidence_path: str | Path,
    semantic_scorers: Sequence[SemanticScorer],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, str | None], dict[str, set[str]]]:
    """Validate the immutable 4 x 33 upstream matrix before any scoring.

    This is intentionally stricter than the generic row aligner.  It binds every
    supplied byte artifact to a sealed manifest, then verifies each generation
    row against the canonical prompt/config/checkpoint records.  Invalid
    completions remain valid *matrix rows* and therefore are never dropped.
    """

    config_path = Path(evaluation_config_path).expanduser().resolve()
    test_path = Path(test_manifest_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_manifest_path).expanduser().resolve()
    config, config_input = _read_json_object(
        config_path,
        kind="evaluation config",
    )
    test_manifest, test_input = _read_json_object(
        test_path,
        kind="test manifest",
        require_integrity=True,
    )
    checkpoint_manifest, checkpoint_input = _read_json_object(
        checkpoint_path,
        kind="checkpoint manifest",
        require_integrity=True,
    )
    semantic_validation = _validate_formal_semantic_backends(
        semantic_manifest_path=semantic_manifest_path,
        semantic_scorers=semantic_scorers,
        evaluation_id=config.get("evaluation_id"),
    )

    population = _mapping_field(config, "population", label="evaluation config")
    meeting_dates = _string_list(
        population.get("meeting_dates"),
        label="evaluation config.population.meeting_dates",
    )
    prospective_dates = _string_list(
        population.get("prospective_only_meeting_dates"),
        label="evaluation config.population.prospective_only_meeting_dates",
    )
    sections_raw = config.get("sections")
    if not isinstance(sections_raw, list) or any(
        not isinstance(section, Mapping) for section in sections_raw
    ):
        raise ValueError("evaluation config.sections must be a list of objects")
    section_ids = [_nonblank(section.get("section_id")) for section in sections_raw]
    if any(not section_id for section_id in section_ids):
        raise ValueError("Every evaluation section requires a section_id")
    if len(section_ids) != len(set(section_ids)):
        raise ValueError("evaluation config.sections contains duplicate section_id")
    if (
        len(meeting_dates) != FORMAL_MEETING_COUNT
        or len(prospective_dates) != FORMAL_PROSPECTIVE_MEETING_COUNT
        or len(section_ids) != FORMAL_SECTION_COUNT
    ):
        raise ValueError(
            "The formal checkpoint evaluation requires exactly "
            f"{FORMAL_MEETING_COUNT} meetings, "
            f"{FORMAL_PROSPECTIVE_MEETING_COUNT} prospective-only meetings, "
            f"and {FORMAL_SECTION_COUNT} sections"
        )
    if not set(prospective_dates) < set(meeting_dates):
        raise ValueError(
            "prospective_only_meeting_dates must be a strict subset of meeting_dates"
        )
    expected_sample_ids = [
        f"{meeting_date}::{section_id}"
        for meeting_date in meeting_dates
        for section_id in section_ids
    ]
    prospective_sample_ids = {
        f"{meeting_date}::{section_id}"
        for meeting_date in prospective_dates
        for section_id in section_ids
    }
    if (
        len(expected_sample_ids) != FORMAL_TEST_SAMPLE_COUNT
        or len(prospective_sample_ids) != FORMAL_PROSPECTIVE_SAMPLE_COUNT
    ):
        raise AssertionError(
            "The frozen 33/27 sample matrix is internally inconsistent"
        )

    generation_config = _mapping_field(
        config,
        "generation",
        label="evaluation config",
    )
    base_seed = _integer(
        generation_config.get("base_seed"),
        label="evaluation config.generation.base_seed",
    )
    if base_seed < 0:
        raise ValueError("Formal generation base_seed must be non-negative")
    if generation_config.get("seed_policy") != FORMAL_ROW_SEED_POLICY:
        raise ValueError(
            f"Formal generation seed_policy must equal {FORMAL_ROW_SEED_POLICY!r}"
        )
    if (
        _finite_float(
            generation_config.get("temperature"),
            label="evaluation config.generation.temperature",
        )
        != 0.0
    ):
        raise ValueError("Formal generation temperature must equal 0.0")
    if (
        _finite_float(
            generation_config.get("top_p"),
            label="evaluation config.generation.top_p",
        )
        != 1.0
    ):
        raise ValueError("Formal generation top_p must equal 1.0")
    evaluation_config = _mapping_field(
        config,
        "evaluation",
        label="evaluation config",
    )
    if evaluation_config.get("score_final_answer_only") is not True:
        raise ValueError("Formal evaluation must score final_answer only")
    if evaluation_config.get("long_text_policy") != LONG_TEXT_POLICY:
        raise ValueError(f"Formal long_text_policy must equal {LONG_TEXT_POLICY!r}")
    if (
        _integer(
            evaluation_config.get("bootstrap_iterations"),
            label="evaluation config.evaluation.bootstrap_iterations",
        )
        != bootstrap_samples
        or _integer(
            evaluation_config.get("bootstrap_seed"),
            label="evaluation config.evaluation.bootstrap_seed",
        )
        != bootstrap_seed
    ):
        raise ValueError(
            "Runtime bootstrap settings differ from the frozen evaluation config"
        )

    if test_manifest.get("evaluation_id") != config.get("evaluation_id"):
        raise ValueError("Test manifest evaluation_id differs from the config")
    if test_manifest.get("meeting_dates") != meeting_dates:
        raise ValueError("Test manifest meeting_dates differ from the config")
    if test_manifest.get("prospective_only_meeting_dates") != prospective_dates:
        raise ValueError(
            "Test manifest prospective_only_meeting_dates differ from the config"
        )
    if (
        test_manifest.get("section_count") != FORMAL_SECTION_COUNT
        or test_manifest.get("sample_count") != FORMAL_TEST_SAMPLE_COUNT
        or test_manifest.get("reference_in_prompt") is not False
    ):
        raise ValueError("Test manifest does not declare the frozen 3 x 11 matrix")
    test_inputs = _mapping_field(test_manifest, "inputs", label="test manifest")
    _assert_file_binding(
        test_inputs.get("config"),
        config_input,
        label="test manifest.inputs.config",
    )
    test_outputs = _mapping_field(test_manifest, "outputs", label="test manifest")
    _assert_file_binding(
        test_outputs.get("prompts"),
        prompt_input,
        label="test manifest.outputs.prompts",
        expected_row_count=FORMAL_TEST_SAMPLE_COUNT,
    )
    _assert_file_binding(
        test_outputs.get("references"),
        reference_input,
        label="test manifest.outputs.references",
        expected_row_count=FORMAL_TEST_SAMPLE_COUNT,
    )

    prompt_ids = [_sample_id(row, kind="prompt") for row in prompt_rows]
    reference_ids = [_sample_id(row, kind="reference") for row in reference_rows]
    if prompt_ids != expected_sample_ids or reference_ids != expected_sample_ids:
        raise ValueError(
            "Prompts/references do not preserve the exact ordered 33-row matrix"
        )
    prompt_by_id = {str(row["sample_id"]): row for row in prompt_rows}
    for sample_id, prompt_row, reference_row in zip(
        expected_sample_ids,
        prompt_rows,
        reference_rows,
        strict=True,
    ):
        expected_meeting, expected_section = sample_id.split("::", maxsplit=1)
        for label, row in (("prompt", prompt_row), ("reference", reference_row)):
            if (
                _nonblank(row.get("meeting_date")) != expected_meeting
                or _nonblank(row.get("section_id")) != expected_section
            ):
                raise ValueError(
                    f"{label} row {sample_id!r} has inconsistent meeting/section keys"
                )
        prompt_text = str(prompt_row.get("prompt") or "")
        if (
            not prompt_text
            or prompt_row.get("prompt_sha256")
            != hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
            or prompt_row.get("reference_used_in_prompt") is not False
            or not isinstance(prompt_row.get("evidence_facts"), list)
            or not prompt_row.get("evidence_facts")
        ):
            raise ValueError(
                f"Prompt row {sample_id!r} failed content/provenance checks"
            )
        _validate_sha256(
            prompt_row.get("evidence_sha256"),
            field=f"prompt[{sample_id}].evidence_sha256",
        )
        reference_text = _select_text(reference_row, kind="reference")
        if (
            not reference_text
            or reference_row.get("reference_sha256")
            != hashlib.sha256(reference_text.encode("utf-8")).hexdigest()
        ):
            raise ValueError(f"Reference row {sample_id!r} failed checksum validation")
        if reference_text in prompt_text:
            raise ValueError(f"Reference leakage detected in prompt {sample_id!r}")
        validate_no_prompt_reference_token_overlap(
            prompt_text,
            reference_text,
            sample_id=sample_id,
        )

    checkpoint_records_raw = checkpoint_manifest.get("artifacts")
    if not isinstance(checkpoint_records_raw, list) or any(
        not isinstance(record, Mapping) for record in checkpoint_records_raw
    ):
        raise ValueError("Checkpoint manifest.artifacts must be a list of objects")
    checkpoint_records: dict[str, Mapping[str, Any]] = {}
    for record in checkpoint_records_raw:
        artifact_id = _nonblank(record.get("artifact_id"))
        if not artifact_id:
            raise ValueError("Checkpoint artifact record has no artifact_id")
        if artifact_id in checkpoint_records:
            raise ValueError(f"Duplicate checkpoint artifact_id {artifact_id!r}")
        checkpoint_records[artifact_id] = record
    lineage_evidence_validation = _validate_formal_lineage_evidence(
        lineage_evidence_path=lineage_evidence_path,
        checkpoint_manifest_path=checkpoint_path,
        checkpoint_manifest=checkpoint_manifest,
        checkpoint_input=checkpoint_input,
    )

    if len(generation_batches) != 4 or len(generation_manifest_paths) != 4:
        raise ValueError(
            "Formal evaluation requires four generation JSONLs and four manifests"
        )
    generation_inputs_by_sha: dict[
        str, tuple[list[dict[str, Any]], Mapping[str, Any]]
    ] = {}
    for rows, metadata in generation_batches:
        digest = str(metadata["sha256"])
        if digest in generation_inputs_by_sha:
            raise ValueError("Two supplied generation JSONLs have identical bytes")
        generation_inputs_by_sha[digest] = (rows, metadata)

    manifest_inputs: list[dict[str, Any]] = []
    generation_artifact_ids: list[str] = []
    lineage: dict[str, str | None] = {}
    bound_generation_shas: set[str] = set()
    checkpoint_manifest_sha256 = str(checkpoint_input["sha256"])
    config_sha256 = str(config_input["sha256"])
    prompts_sha256 = str(prompt_input["sha256"])
    for manifest_path_value in generation_manifest_paths:
        manifest_path = Path(manifest_path_value).expanduser().resolve()
        manifest, manifest_input = _read_json_object(
            manifest_path,
            kind="generation manifest",
            require_integrity=True,
        )
        artifact_id = _nonblank(manifest.get("artifact_id"))
        if not artifact_id:
            raise ValueError(f"Generation manifest has no artifact_id: {manifest_path}")
        if artifact_id in generation_artifact_ids:
            raise ValueError(f"Duplicate generation manifest for {artifact_id!r}")
        generation_artifact_ids.append(artifact_id)
        checkpoint_record = checkpoint_records.get(artifact_id)
        if checkpoint_record is None:
            raise ValueError(
                f"Generation manifest references unknown checkpoint {artifact_id!r}"
            )
        if checkpoint_record.get("usable_for_evaluation") is not True:
            raise ValueError(f"Checkpoint {artifact_id!r} is not usable_for_evaluation")

        _assert_file_binding(
            manifest.get("checkpoint_manifest"),
            checkpoint_input,
            label=f"generation manifest[{artifact_id}].checkpoint_manifest",
        )
        _assert_file_binding(
            manifest.get("evaluation_config"),
            config_input,
            label=f"generation manifest[{artifact_id}].evaluation_config",
        )
        _assert_file_binding(
            manifest.get("prompts"),
            prompt_input,
            label=f"generation manifest[{artifact_id}].prompts",
            expected_row_count=FORMAL_TEST_SAMPLE_COUNT,
        )
        output_record = _mapping_field(
            manifest,
            "output",
            label=f"generation manifest[{artifact_id}]",
        )
        output_sha256 = _validate_sha256(
            output_record.get("sha256"),
            field=f"generation manifest[{artifact_id}].output.sha256",
        )
        batch = generation_inputs_by_sha.get(output_sha256)
        if batch is None:
            raise ValueError(
                f"Generation manifest {artifact_id!r} output is not among supplied JSONLs"
            )
        rows, generation_input = batch
        if output_sha256 in bound_generation_shas:
            raise ValueError("A generation JSONL is bound by more than one manifest")
        bound_generation_shas.add(output_sha256)
        if output_record.get("row_count") != FORMAL_TEST_SAMPLE_COUNT:
            raise ValueError(
                f"Generation manifest {artifact_id!r} output row_count is not 33"
            )
        if manifest.get("sample_count") != FORMAL_TEST_SAMPLE_COUNT:
            raise ValueError(
                f"Generation manifest {artifact_id!r} sample_count is not 33"
            )
        observed_ids = [_sample_id(row, kind="generation") for row in rows]
        observed_artifact_ids = {_artifact_id(row) for row in rows}
        if observed_ids != expected_sample_ids or observed_artifact_ids != {
            artifact_id
        }:
            raise ValueError(
                f"Generation artifact {artifact_id!r} does not preserve its exact "
                "ordered 33-row matrix"
            )
        if any(not isinstance(row.get("valid_generation"), bool) for row in rows):
            raise ValueError(
                f"Generation artifact {artifact_id!r} lacks Boolean valid_generation"
            )
        valid_count = sum(bool(row["valid_generation"]) for row in rows)
        if (
            manifest.get("valid_count") != valid_count
            or manifest.get("invalid_count") != FORMAL_TEST_SAMPLE_COUNT - valid_count
        ):
            raise ValueError(
                f"Generation manifest {artifact_id!r} validity counts changed"
            )
        inference_environment = manifest.get("inference_environment")
        inference_environment_sha256 = manifest.get(
            "inference_environment_sha256"
        )
        if inference_environment is None and inference_environment_sha256 is None:
            environment_audit = {
                "status": "not_recorded_legacy_compatible",
                "required_for_formal_acceptance": False,
            }
        else:
            if not isinstance(inference_environment, Mapping):
                raise ValueError(
                    f"Generation manifest {artifact_id!r} inference_environment "
                    "must be an object"
                )
            observed_environment_sha256 = _json_payload_sha256(
                inference_environment
            )
            declared_environment_sha256 = _validate_sha256(
                inference_environment_sha256,
                field=(
                    f"generation manifest[{artifact_id}]."
                    "inference_environment_sha256"
                ),
            )
            if declared_environment_sha256 != observed_environment_sha256:
                raise ValueError(
                    f"Generation manifest {artifact_id!r} inference environment "
                    "digest changed"
                )
            for row in rows:
                if (
                    row.get("inference_environment") != inference_environment
                    or row.get("inference_environment_sha256")
                    != declared_environment_sha256
                ):
                    raise ValueError(
                        f"Generation row {artifact_id}::{row.get('sample_id')} "
                        "has inconsistent inference environment metadata"
                    )
            environment_audit = {
                "status": "recorded_and_verified",
                "required_for_formal_acceptance": False,
                "sha256": declared_environment_sha256,
                "environment": dict(inference_environment),
            }

        model_record = _mapping_field(
            manifest,
            "model_artifact",
            label=f"generation manifest[{artifact_id}]",
        )
        tokenizer_record = _mapping_field(
            manifest,
            "tokenizer_artifact",
            label=f"generation manifest[{artifact_id}]",
        )
        model_sha256 = _validate_sha256(
            model_record.get("sha256"),
            field=f"generation manifest[{artifact_id}].model_artifact.sha256",
        )
        tokenizer_sha256 = _validate_sha256(
            tokenizer_record.get("sha256"),
            field=f"generation manifest[{artifact_id}].tokenizer_artifact.sha256",
        )
        if (
            _validate_sha256(
                checkpoint_record.get("model_sha256"),
                field=f"checkpoint[{artifact_id}].model_sha256",
            )
            != model_sha256
            or _validate_sha256(
                checkpoint_record.get("tokenizer_sha256"),
                field=f"checkpoint[{artifact_id}].tokenizer_sha256",
            )
            != tokenizer_sha256
        ):
            raise ValueError(
                f"Generation manifest {artifact_id!r} model/tokenizer differs "
                "from checkpoint provenance"
            )
        verified_parent = _nonblank(
            checkpoint_record.get("verified_parent_artifact_id")
        )
        lineage[artifact_id] = verified_parent or None

        for row in rows:
            sample_id = str(row["sample_id"])
            prompt_row = prompt_by_id[sample_id]
            row_parent = _nonblank(row.get("verified_parent_artifact_id"))
            final_answer = row.get("final_answer")
            expected_generation_seed = derive_row_seed(base_seed, sample_id)
            if row.get("generation_seed_policy") != FORMAL_ROW_SEED_POLICY:
                raise ValueError(
                    f"Generation row {artifact_id}::{sample_id} has invalid "
                    "generation_seed_policy"
                )
            if (
                _integer(
                    row.get("generation_seed"),
                    label=f"generation[{artifact_id}::{sample_id}].generation_seed",
                )
                != expected_generation_seed
            ):
                raise ValueError(
                    f"Generation row {artifact_id}::{sample_id} generation_seed "
                    "does not match base_seed + sample_id derivation"
                )
            if (
                row.get("prompt_sha256") != prompt_row.get("prompt_sha256")
                or row.get("evidence_sha256") != prompt_row.get("evidence_sha256")
                or row.get("prompt") != prompt_row.get("prompt")
                or row.get("evidence_facts") != prompt_row.get("evidence_facts")
                or row.get("meeting_date") != prompt_row.get("meeting_date")
                or row.get("section_id") != prompt_row.get("section_id")
                or row.get("section_name") != prompt_row.get("section_name")
                or row.get("reference_used_in_prompt") is not False
                or row.get("generation_model_sha256") != model_sha256
                or row.get("generation_tokenizer_sha256") != tokenizer_sha256
                or row.get("test_set_sha256") != prompts_sha256
                or row.get("prompt_template_sha256") != config_sha256
                or row.get("decoding_config_sha256") != config_sha256
                or row_parent != verified_parent
                or _finite_float(
                    row.get("temperature"),
                    label=f"generation[{artifact_id}::{sample_id}].temperature",
                )
                != _finite_float(
                    generation_config.get("temperature"),
                    label="evaluation config.generation.temperature",
                )
                or _finite_float(
                    row.get("top_p"),
                    label=f"generation[{artifact_id}::{sample_id}].top_p",
                )
                != _finite_float(
                    generation_config.get("top_p"),
                    label="evaluation config.generation.top_p",
                )
                or _integer(
                    row.get("max_new_tokens"),
                    label=f"generation[{artifact_id}::{sample_id}].max_new_tokens",
                )
                != _integer(
                    generation_config.get("max_new_tokens"),
                    label="evaluation config.generation.max_new_tokens",
                )
                or _integer(
                    row.get("max_model_len"),
                    label=f"generation[{artifact_id}::{sample_id}].max_model_len",
                )
                != _integer(
                    generation_config.get("max_model_len"),
                    label="evaluation config.generation.max_model_len",
                )
                or not isinstance(final_answer, str)
                or row.get("final_answer_sha256")
                != hashlib.sha256(str(final_answer).encode("utf-8")).hexdigest()
                or (
                    row.get("valid_generation") is True
                    and not str(final_answer).strip()
                )
            ):
                raise ValueError(
                    f"Generation row {artifact_id}::{sample_id} failed frozen "
                    "prompt/model/decoding provenance validation"
                )
            declared_checkpoint_manifest = row.get("checkpoint_manifest_sha256")
            if (
                declared_checkpoint_manifest not in (None, "")
                and declared_checkpoint_manifest != checkpoint_manifest_sha256
            ):
                raise ValueError(
                    f"Generation row {artifact_id}::{sample_id} binds another "
                    "checkpoint manifest"
                )
            declared_config = row.get("evaluation_config_sha256")
            if declared_config not in (None, "") and declared_config != config_sha256:
                raise ValueError(
                    f"Generation row {artifact_id}::{sample_id} binds another config"
                )
        manifest_inputs.append(
            {
                **manifest_input,
                "artifact_id": artifact_id,
                "generation_jsonl": dict(generation_input),
                "valid_count": valid_count,
                "invalid_count": FORMAL_TEST_SAMPLE_COUNT - valid_count,
                "model_sha256": model_sha256,
                "tokenizer_sha256": tokenizer_sha256,
                "design_checkpoint_id": checkpoint_record.get("design_checkpoint_id"),
                "intended_parent_id": checkpoint_record.get("intended_parent_id"),
                "verified_parent_artifact_id": lineage[artifact_id],
                "lineage_status": checkpoint_record.get("lineage_status"),
                "inference_environment_audit": environment_audit,
            }
        )

    if len(bound_generation_shas) != len(generation_batches):
        raise ValueError(
            "At least one supplied generation JSONL has no manifest binding"
        )
    if len(set(generation_artifact_ids)) != 4:
        raise ValueError("Formal evaluation requires four distinct artifact IDs")

    subsets = {
        "all_11_meetings": set(meeting_dates),
        "prospective_only_9_meetings": set(prospective_dates),
    }
    audit = {
        "mode": "formal_manifest_bound",
        "config": config_input,
        "test_manifest": test_input,
        "checkpoint_manifest": checkpoint_input,
        "semantic_models": semantic_validation,
        "lineage_evidence": lineage_evidence_validation,
        "generation_manifests": sorted(
            manifest_inputs,
            key=lambda item: str(item["artifact_id"]),
        ),
        "expected_matrix": {
            "meeting_count": FORMAL_MEETING_COUNT,
            "prospective_only_meeting_count": FORMAL_PROSPECTIVE_MEETING_COUNT,
            "section_count": FORMAL_SECTION_COUNT,
            "sample_count_per_artifact": FORMAL_TEST_SAMPLE_COUNT,
            "prospective_only_sample_count_per_artifact": (
                FORMAL_PROSPECTIVE_SAMPLE_COUNT
            ),
            "artifact_count": 4,
            "total_generation_rows": 4 * FORMAL_TEST_SAMPLE_COUNT,
            "meeting_dates": meeting_dates,
            "prospective_only_meeting_dates": prospective_dates,
            "sample_ids_sha256": hashlib.sha256(
                "\n".join(expected_sample_ids).encode("utf-8")
            ).hexdigest(),
        },
        "long_text_policy": LONG_TEXT_POLICY,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "invalid_rows_preserved": True,
        "comparison_claim_boundary": (
            "Only verified-parent direct edges (or explicitly pre-registered "
            "non-lineage contrasts) are inferentially eligible; artifact labels "
            "must not be interpreted as an intended chk-0->chk-1->chk-2->chk-3 "
            "incremental chain."
        ),
        "complete": True,
    }
    return audit, lineage, subsets


def run_checkpoint_generation_evaluation(
    generation_paths: Sequence[str | Path],
    reference_path: str | Path,
    prompts_path: str | Path,
    output_dir: str | Path,
    *,
    generation_manifest_paths: Sequence[str | Path] = (),
    evaluation_config_path: str | Path | None = None,
    test_manifest_path: str | Path | None = None,
    checkpoint_manifest_path: str | Path | None = None,
    semantic_manifest_path: str | Path | None = None,
    lineage_evidence_path: str | Path | None = None,
    require_formal_manifests: bool = True,
    semantic_scorers: Sequence[SemanticScorer] = (),
    contrasts: Sequence[Mapping[str, Any] | Sequence[str]] | None = None,
    lineage_manifest_path: str | Path | None = None,
    expected_artifact_count: int | None = 4,
    required_artifact_ids: Sequence[str] | None = None,
    missing_policy: str = "error",
    require_provenance: bool = True,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    scoring_policy: str = STRICT_SCORING_POLICY,
    evaluation_subsets: Mapping[
        str, Sequence[str] | set[str] | frozenset[str] | None
    ]
    | None = None,
) -> dict[str, Any]:
    """Run the complete evaluator and write five result artifacts plus audit."""

    if not generation_paths:
        raise ValueError("At least one generations JSONL path is required")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    scoring_policy_spec = _scoring_policy_spec(scoring_policy)
    generation_rows: list[dict[str, Any]] = []
    generation_inputs = []
    generation_batches: list[tuple[list[dict[str, Any]], Mapping[str, Any]]] = []
    for path_value in generation_paths:
        rows, metadata = _read_jsonl(Path(path_value), kind="generations")
        generation_rows.extend(rows)
        generation_inputs.append(metadata)
        generation_batches.append((rows, metadata))
    reference_rows, reference_input = _read_jsonl(
        Path(reference_path),
        kind="reference",
    )
    prompt_rows, prompt_input = _read_jsonl(
        Path(prompts_path),
        kind="prompts",
    )
    formal_audit: dict[str, Any] | None = None
    subsets: Mapping[str, set[str] | frozenset[str] | None] | None = None
    if require_formal_manifests:
        if evaluation_subsets is not None:
            raise ValueError(
                "evaluation_subsets cannot override the sealed formal test subsets"
            )
        missing_formal = [
            label
            for label, value in (
                ("generation_manifest_paths", generation_manifest_paths),
                ("evaluation_config_path", evaluation_config_path),
                ("test_manifest_path", test_manifest_path),
                ("checkpoint_manifest_path", checkpoint_manifest_path),
                ("semantic_manifest_path", semantic_manifest_path),
                ("lineage_evidence_path", lineage_evidence_path),
            )
            if not value
        ]
        if missing_formal:
            raise ValueError(
                f"Formal evaluation requires sealed upstream bindings: {missing_formal}"
            )
        formal_audit, formal_lineage, subsets = _validate_formal_input_matrix(
            generation_batches=generation_batches,
            generation_manifest_paths=generation_manifest_paths,
            prompt_rows=prompt_rows,
            prompt_input=prompt_input,
            reference_rows=reference_rows,
            reference_input=reference_input,
            evaluation_config_path=str(evaluation_config_path),
            test_manifest_path=str(test_manifest_path),
            checkpoint_manifest_path=str(checkpoint_manifest_path),
            semantic_manifest_path=str(semantic_manifest_path),
            lineage_evidence_path=str(lineage_evidence_path),
            semantic_scorers=semantic_scorers,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        formal_artifact_ids = sorted(
            str(item["artifact_id"]) for item in formal_audit["generation_manifests"]
        )
        if (
            required_artifact_ids is not None
            and sorted(map(str, required_artifact_ids)) != formal_artifact_ids
        ):
            raise ValueError(
                "required_artifact_ids differs from the four sealed "
                "generation manifests"
            )
        required_artifact_ids = formal_artifact_ids
        expected_artifact_count = 4
        missing_policy = "error"
        require_provenance = True
        lineage = formal_lineage
    else:
        lineage = _load_lineage(lineage_manifest_path)
        if evaluation_subsets is not None:
            if not isinstance(evaluation_subsets, Mapping) or not evaluation_subsets:
                raise ValueError("evaluation_subsets must be a non-empty mapping")
            available_meetings = {
                _nonblank(row.get("meeting_id") or row.get("meeting_date"))
                for row in prompt_rows
            }
            available_meetings.discard("")
            normalized_subsets: dict[str, set[str] | None] = {}
            for raw_name, raw_meetings in evaluation_subsets.items():
                name = _nonblank(raw_name)
                if not name:
                    raise ValueError("evaluation_subsets names must be non-empty")
                if name in normalized_subsets:
                    raise ValueError(f"Duplicate evaluation subset name: {name!r}")
                if raw_meetings is None:
                    normalized_subsets[name] = None
                    continue
                if isinstance(raw_meetings, (str, bytes)):
                    raise ValueError(
                        f"evaluation_subsets[{name!r}] must be a meeting sequence"
                    )
                meetings = {_nonblank(value) for value in raw_meetings}
                if "" in meetings or not meetings:
                    raise ValueError(
                        f"evaluation_subsets[{name!r}] must contain non-empty meetings"
                    )
                unknown = sorted(meetings - available_meetings)
                if unknown:
                    raise ValueError(
                        f"evaluation_subsets[{name!r}] contains unknown meetings: {unknown}"
                    )
                normalized_subsets[name] = meetings
            subsets = normalized_subsets
    aligned_rows, alignment_audit = align_artifacts(
        generation_rows,
        reference_rows,
        prompt_rows,
        expected_artifact_count=expected_artifact_count,
        required_artifact_ids=required_artifact_ids,
        missing_policy=missing_policy,
    )
    generation_index, _ = _index_generations(generation_rows)
    artifact_ids = alignment_audit["artifact_ids"]
    provenance = _collect_provenance(
        generation_index,
        artifact_ids,
        require_provenance=require_provenance,
        reference_sha256=reference_input["sha256"],
        evidence_sha256=prompt_input["sha256"],
    )

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "row_scores": destination / "row_scores.jsonl",
        "claim_checks": destination / "claim_checks.jsonl",
        "summary": destination / "summary.jsonl",
        "contrasts": destination / "contrasts.jsonl",
        "audit": destination / "audit.json",
    }
    subset_spec = {
        str(name): None if meetings is None else sorted(meetings)
        for name, meetings in (
            subsets.items() if subsets is not None else {"all": None}.items()
        )
    }
    run_spec = {
        "schema_version": SCHEMA_VERSION,
        "evaluator_source_sha256": _sha256_file(Path(__file__).resolve()),
        "formal_manifest_validation_required": require_formal_manifests,
        "formal_validation": formal_audit,
        "generation_inputs": sorted(
            (
                {
                    "sha256": item["sha256"],
                    "row_count": item["row_count"],
                    "blank_line_count": item["blank_line_count"],
                }
                for item in generation_inputs
            ),
            key=lambda item: str(item["sha256"]),
        ),
        "reference_input": {
            "sha256": reference_input["sha256"],
            "row_count": reference_input["row_count"],
            "blank_line_count": reference_input["blank_line_count"],
        },
        "prompts_input": {
            "sha256": prompt_input["sha256"],
            "row_count": prompt_input["row_count"],
            "blank_line_count": prompt_input["blank_line_count"],
        },
        "artifact_ids": artifact_ids,
        "provenance": provenance,
        "verified_lineage": dict(sorted(lineage.items())),
        "contrasts_requested": _json_safe(contrasts),
        "evaluation_subsets": subset_spec,
        "semantic_backends": [
            _semantic_backend_identity(scorer) for scorer in semantic_scorers
        ],
        "semantic_manifest": (
            formal_audit["semantic_models"]["manifest"]
            if formal_audit is not None
            else None
        ),
        "lineage_evidence": (
            formal_audit["lineage_evidence"]["evidence"]
            if formal_audit is not None
            else None
        ),
        "expected_artifact_count": expected_artifact_count,
        "required_artifact_ids": (
            sorted(map(str, required_artifact_ids))
            if required_artifact_ids is not None
            else None
        ),
        "missing_policy": missing_policy,
        "require_provenance": require_provenance,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "scoring_policy": scoring_policy_spec,
        "invalid_output_policy_version": (
            "length-tolerant-nonempty-completion-v1"
            if scoring_policy == LENGTH_TOLERANT_SCORING_POLICY
            else "fatal-only-finite-worst-case-v2"
        ),
        "format_policy_version": "nonfatal-formal-section-body-only-v2",
        "long_text_policy": LONG_TEXT_POLICY,
    }
    run_spec_sha256 = _json_payload_sha256(run_spec)
    reused = _reuse_existing_evaluation(
        paths,
        run_spec_sha256=run_spec_sha256,
    )
    if reused is not None:
        return reused

    row_scores, claim_checks, semantic_metadata = score_rows(
        aligned_rows,
        semantic_scorers=semantic_scorers,
        scoring_policy=scoring_policy,
    )
    prospective_meetings = (
        set(subsets.get("prospective_only_9_meetings") or set())
        if subsets is not None
        else set()
    )
    for row in row_scores:
        row["is_prospective_only"] = (
            str(row["meeting_id"]) in prospective_meetings
            if subsets is not None
            else None
        )
    for row in claim_checks:
        row["is_prospective_only"] = (
            str(row["meeting_id"]) in prospective_meetings
            if subsets is not None
            else None
        )
    summary = summarise_rows(
        row_scores,
        subsets=subsets,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    contrast_results, normalized_contrasts = contrast_rows(
        row_scores,
        provenance=provenance,
        lineage=lineage,
        contrasts=contrasts,
        subsets=subsets,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    if require_formal_manifests:
        if len(row_scores) != 4 * FORMAL_TEST_SAMPLE_COUNT:
            raise AssertionError("Formal row_scores matrix is not 4 x 33")
        formal_subset_counts = Counter(
            (
                str(row["artifact_id"]),
                bool(row["is_prospective_only"]),
            )
            for row in row_scores
        )
        for artifact_id in artifact_ids:
            if (
                formal_subset_counts[(artifact_id, True)]
                != FORMAL_PROSPECTIVE_SAMPLE_COUNT
                or formal_subset_counts[(artifact_id, False)]
                != FORMAL_TEST_SAMPLE_COUNT - FORMAL_PROSPECTIVE_SAMPLE_COUNT
            ):
                raise AssertionError(
                    f"Formal prospective subset is not 27 rows for {artifact_id!r}"
                )

    _write_jsonl(paths["row_scores"], row_scores)
    _write_jsonl(paths["claim_checks"], claim_checks)
    _write_jsonl(paths["summary"], summary)
    _write_jsonl(paths["contrasts"], contrast_results)
    scoring_policy_row_audit: dict[str, Any] | None = None
    if scoring_policy == LENGTH_TOLERANT_SCORING_POLICY:
        by_artifact: dict[str, Any] = {}
        for artifact_id in artifact_ids:
            artifact_rows = [
                row for row in row_scores if row["artifact_id"] == artifact_id
            ]
            by_artifact[artifact_id] = {
                "row_count": len(artifact_rows),
                "valid_output_count": sum(
                    row.get("valid_output") == 1.0 for row in artifact_rows
                ),
                "source_generation_hit_token_limit_count": sum(
                    row.get("source_generation_hit_token_limit") is True
                    for row in artifact_rows
                ),
                "upstream_valid_generation_false_count": sum(
                    row.get("upstream_valid_generation") is False
                    for row in artifact_rows
                ),
                "input_was_truncated_count": sum(
                    (
                        aligned["generation_row"] or {}
                    ).get("input_was_truncated")
                    is True
                    for aligned in aligned_rows
                    if aligned["artifact_id"] == artifact_id
                ),
                "candidate_extraction_mode_counts": dict(
                    sorted(
                        Counter(
                            str(row.get("candidate_extraction_mode"))
                            for row in artifact_rows
                        ).items()
                    )
                ),
                "think_opening_tag_count": sum(
                    row.get("has_leading_think_opening_tag") is True
                    for row in artifact_rows
                ),
                "answer_opening_tag_count": sum(
                    row.get("has_answer_opening_tag") is True
                    for row in artifact_rows
                ),
            }
        scoring_policy_row_audit = {
            "all_declared_rows_valid": all(
                row.get("valid_output") == 1.0 for row in row_scores
            ),
            "by_artifact": by_artifact,
        }
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "validated",
        "immutable": True,
        "reuse_policy": "validate-all-input-and-output-hashes-or-refuse",
        "run_spec": run_spec,
        "run_spec_sha256": run_spec_sha256,
        "inference_performed": any(
            isinstance(scorer, (BERTScoreBackend, MPNetCosineBackend))
            for scorer in semantic_scorers
        ),
        "generation_inference_performed": False,
        "semantic_encoder_inference_performed": any(
            isinstance(scorer, (BERTScoreBackend, MPNetCosineBackend))
            for scorer in semantic_scorers
        ),
        "network_required": False,
        "input_artifacts": {
            "generations": generation_inputs,
            "reference": reference_input,
            "prompts_with_evidence_facts": prompt_input,
        },
        "formal_validation": formal_audit,
        "artifact_count": len(artifact_ids),
        "artifact_ids": artifact_ids,
        "provenance": provenance,
        "alignment": alignment_audit,
        "semantic_backends": semantic_metadata,
        "rule_scope": {
            "unsupported_claim_metric": "rule_covered_claims_only",
            "unsupported_claim_kinds": [
                "numeric",
                "direction",
                "policy_stance",
                "custom",
            ],
            "not_an_nli_or_exhaustive_factuality_metric": True,
        },
        "scoring_policy": scoring_policy_spec,
        "scoring_policy_row_audit": scoring_policy_row_audit,
        "invalid_output_policy": (
            {
                "rows_retained_in_declared_matrix": True,
                "fatal_only": True,
                "fatal_conditions": [
                    "empty extracted candidate",
                    "input truncation",
                ],
                "token_limit_finish": "audited_but_scored_as_valid",
                "upstream_valid_generation_false": (
                    "audited_but_not_fatal_in_this_robustness_analysis"
                ),
                "strict_parse_failure": (
                    "ignored; extraction uses answer opening tag or full completion"
                ),
                "valid_output": 1.0,
                "metrics": "computed_on_extracted_candidate",
            }
            if scoring_policy == LENGTH_TOLERANT_SCORING_POLICY
            else {
                "rows_retained_in_declared_matrix": True,
                "fatal_only": True,
                "fatal_conditions": [
                    "empty final answer",
                    "upstream valid_generation=false",
                    "bad generation_validation_status",
                    "token-limit finish",
                    "input truncation",
                    "explicit parse failure",
                ],
                "valid_output": 0.0,
                "semantic_metrics": 0.0,
                "rouge_l_f1": 0.0,
                "format_compliance": 0.0,
                "factual_headline_metrics": (
                    "finite worst-case values when their evidence rule is coverable; "
                    "otherwise NA"
                ),
            }
        ),
        "format_noncompliance_policy": {
            "fatal": False,
            "status": "scored",
            "valid_output": 1.0,
            "semantic_surface_and_factual_metrics_scored_normally": True,
            "format_compliance": 0.0,
            "violations_recorded_in": "format_violations",
        },
        "aggregation": {
            "unit": "meeting_id",
            "within_meeting": "arithmetic_mean_over_samples",
            "across_meetings": "equal_weight_arithmetic_mean",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "confidence": 0.95,
            "evaluation_subsets": {
                name: (
                    "all meetings" if meeting_filter is None else sorted(meeting_filter)
                )
                for name, meeting_filter in (
                    subsets.items() if subsets is not None else {"all": None}.items()
                )
            },
        },
        "contrast_plan": normalized_contrasts,
        "multiple_testing": {
            "method": "Holm",
            "families": {
                family: sum(
                    row["p_value"] is not None
                    for row in contrast_results
                    if row["holm_family"] == family
                )
                for family in sorted(
                    {str(row["holm_family"]) for row in contrast_results}
                )
            },
        },
        "row_counts": {
            "row_scores": len(row_scores),
            "claim_checks": len(claim_checks),
            "summary": len(summary),
            "contrasts": len(contrast_results),
        },
        "output_artifacts": {
            key: {
                "path": str(path),
                "sha256": _sha256_file(path),
                "row_count": (
                    len(row_scores)
                    if key == "row_scores"
                    else (
                        len(claim_checks)
                        if key == "claim_checks"
                        else (
                            len(summary) if key == "summary" else len(contrast_results)
                        )
                    )
                ),
            }
            for key, path in paths.items()
            if key != "audit"
        },
    }
    audit = seal_manifest(audit)
    _write_json(paths["audit"], audit)
    return {
        "row_scores": row_scores,
        "claim_checks": claim_checks,
        "summary": summary,
        "contrasts": contrast_results,
        "audit": audit,
        "paths": {key: str(path) for key, path in paths.items()},
        "reused_existing": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate four completed checkpoint-generation artifacts against a "
            "manifest-bound common prompts/reference set. No generation "
            "inference is run."
        )
    )
    parser.add_argument(
        "--generations",
        action="append",
        required=True,
        help=("One generation JSONL. Repeat exactly four times."),
    )
    parser.add_argument(
        "--generation-manifest",
        action="append",
        required=True,
        help="Sealed manifest for a generation JSONL. Repeat exactly four times.",
    )
    parser.add_argument(
        "--references",
        required=True,
        help="Shared official-Minutes references JSONL.",
    )
    parser.add_argument(
        "--prompts",
        required=True,
        help="Shared prompts JSONL carrying evidence_facts.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument(
        "--semantic-manifest",
        required=True,
        help="Sealed manifest binding the frozen BERTScore and MPNet directories.",
    )
    parser.add_argument(
        "--lineage-evidence",
        required=True,
        help="Sealed exact-merge lineage evidence for eval-analysis-sft.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--contrast-plan",
        help="Optional pre-registered contrast-plan JSON; otherwise direct lineage edges are used.",
    )
    parser.add_argument(
        "--required-artifact-id",
        action="append",
        default=None,
        help="Repeat to pin the exact artifact inventory.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    parser.add_argument(
        "--bertscore-model", help="Existing local BERTScore encoder path."
    )
    parser.add_argument("--bertscore-model-sha256")
    parser.add_argument("--bertscore-num-layers", type=int)
    parser.add_argument("--mpnet-model", help="Existing local independent MPNet path.")
    parser.add_argument("--mpnet-model-sha256")
    parser.add_argument("--semantic-device", default="cpu")
    parser.add_argument("--semantic-batch-size", type=int, default=16)
    parser.add_argument(
        "--scoring-policy",
        choices=SCORING_POLICIES,
        default=STRICT_SCORING_POLICY,
        help=(
            "Strict primary scoring, or the separately audited post-hoc "
            "length-tolerant robustness policy."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    semantic_scorers: list[SemanticScorer] = []
    if bool(args.bertscore_model) != bool(args.bertscore_model_sha256):
        raise ValueError(
            "--bertscore-model and --bertscore-model-sha256 must be provided together"
        )
    if bool(args.mpnet_model) != bool(args.mpnet_model_sha256):
        raise ValueError(
            "--mpnet-model and --mpnet-model-sha256 must be provided together"
        )
    if args.bertscore_model:
        semantic_scorers.append(
            BERTScoreBackend(
                args.bertscore_model,
                args.bertscore_model_sha256,
                num_layers=args.bertscore_num_layers,
                batch_size=args.semantic_batch_size,
                device=args.semantic_device,
            )
        )
    if args.mpnet_model:
        semantic_scorers.append(
            MPNetCosineBackend(
                args.mpnet_model,
                args.mpnet_model_sha256,
                batch_size=args.semantic_batch_size,
                device=args.semantic_device,
            )
        )
    run_checkpoint_generation_evaluation(
        args.generations,
        args.references,
        args.prompts,
        args.output_dir,
        generation_manifest_paths=args.generation_manifest,
        evaluation_config_path=args.config,
        test_manifest_path=args.test_manifest,
        checkpoint_manifest_path=args.checkpoint_manifest,
        semantic_manifest_path=args.semantic_manifest,
        lineage_evidence_path=args.lineage_evidence,
        require_formal_manifests=True,
        semantic_scorers=semantic_scorers,
        contrasts=_load_contrast_plan(args.contrast_plan),
        required_artifact_ids=args.required_artifact_id,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        scoring_policy=args.scoring_policy,
    )


if __name__ == "__main__":
    main()
