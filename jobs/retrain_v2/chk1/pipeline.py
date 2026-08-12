"""Prepare immutable, reference-free chk1 generation inputs.

This module is deliberately limited to the data-preparation boundary.  It
combines the already-audited source-data, style-guide, source-handoff, and
prompt-contract APIs; it does not load a model, generate a target, verify a
candidate, or publish an SFT release.

The preparation boundary has two representations of a fact card:

* ``fact_card`` is the generator/student projection.  It contains only the
  compact evidence payload and never contains ``evidence_lineage``.
* ``evidence_lineage`` is retained beside the projection for downstream audit
  and release manifests.  It is never rendered into either model prompt.

Same-sample Minutes are read only to build the train-only style abstraction
and to bind an opaque ``minutes_reference_sha256``.  Raw Minutes text is not
written to any prepared row or exclusion artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from jobs.main.checkpoint_provenance import fingerprint_tokenizer_payload

from .contracts import (
    GENERATOR_SYSTEM_PROMPT,
    canonical_json,
    render_generator_payload,
    render_student_prompt,
    sha256_text,
)
from .local_models import CHK0_MODEL_RELATIVE_PATH, resolve_local_model_path
from .source_data import (
    PROMPT_TOKEN_LIMIT,
    PromptBudgetError,
    build_canonical_samples,
    build_fact_card,
    build_file_inventory,
    evidence_from_loo_ledger,
    stable_sample_id,
    validate_prompt_token_budget,
)
from .sparse_source import (
    LEDGER_MANIFEST_SCHEMA_VERSION as SPARSE_LEDGER_MANIFEST_SCHEMA_VERSION,
    SAMPLE_EXCLUSION_REASON as SPARSE_SAMPLE_EXCLUSION_REASON,
    SAMPLE_EXCLUSION_SCHEMA_VERSION as SPARSE_SAMPLE_EXCLUSION_SCHEMA_VERSION,
    SNAPSHOT_MANIFEST_SCHEMA_VERSION as SPARSE_SNAPSHOT_MANIFEST_SCHEMA_VERSION,
    SparseSourceError,
    validate_sparse_loo_ledger,
    validate_sparse_snapshot_manifest,
)
from .source_pipeline import (
    SOURCE_HANDOFF_SCHEMA_VERSION,
    SOURCE_PLAN_SCHEMA_VERSION,
)
from .style_guide import (
    build_style_guide_from_jsonl,
    verify_style_guide_artifact,
)
from .topic_styles import (
    ATOMIC_TOPICS,
    canonical_atomic_topic,
    ledger_indicator_for_topic,
    topic_key,
    topic_style_map,
)


PREPARED_SAMPLE_SCHEMA_VERSION = "chk1-prepared-sample-v1"
PREPARE_HANDOFF_SCHEMA_VERSION = "chk1-prepare-handoff-v1"
SAMPLE_EXCLUSION_SCHEMA_VERSION = "chk1-prepare-sample-exclusion-v1"
EVIDENCE_EXCLUSION_SCHEMA_VERSION = "chk1-prepare-evidence-exclusion-v1"
PRECANONICAL_EXCLUSION_SCHEMA_VERSION = "chk1-precanonical-exclusion-v1"
MINUTES_REFERENCE_HASH_POLICY = "sha256-sorted-unique-member-sha256-v1"
FACT_CARD_SELECTION_SCHEMA_VERSION = "chk1-fact-card-selection-v1"
FACT_CARD_SELECTION_POLICY = (
    "adaptive-3-2-1-recent-observations-per-series-no-truncation-v1"
)

_SPLITS = ("train", "eval", "test")
_DENSE_LEDGER_MANIFEST_SCHEMA_VERSION = "loo-indicator-ledger-manifest-v1"
_LEDGER_ROW_SCHEMA_VERSION = "canonical-loo-indicator-input-v2"
_SOURCE_EVIDENCE_SCHEMA_VERSION = "loo-indicator-source-evidence-v1"
_COVERAGE_SCHEMA_VERSION = "loo-indicator-ledger-coverage-v1"
_POPULATION_SCHEMA_VERSION = "loo-population-v1"
_INTEGRITY_ALGORITHM = "sha256(canonical-json-without-integrity)"
_DENSE_LEDGER_OUTPUTS = (
    "indicator_inputs",
    "source_evidence",
    "excluded_records",
    "coverage",
)
_SPARSE_LEDGER_OUTPUTS = (*_DENSE_LEDGER_OUTPUTS[:-1], "sample_exclusions", "coverage")
_FACT_CARD_RECENT_LIMITS = (3, 2, 1)
_GENERATOR_FACT_CARD_FIELDS = (
    "schema_version",
    "sample_id",
    "canonical_key",
    "meeting_date",
    "atomic_topic",
    "cutoff_ts",
    "evidence",
)
_GENERATOR_EVIDENCE_FIELDS = (
    "evidence_id",
    "series_id",
    "metric",
    "fact_kind",
    "value",
    "units",
    "observation_date",
    "release_ts",
    "availability_upper_bound_ts",
    "availability_basis",
    "cutoff_ts",
    "source_sha256",
    "formula",
    "operand_evidence_ids",
)
_FORBIDDEN_PREPARED_KEYS = frozenset(
    {
        "reference_excerpt",
        "archived_response",
        "teacher_response",
        "rate_change",
        "current_rate",
        "decision_label",
    }
)
_SAFE_STYLE_ENTRY_FIELDS = frozenset(
    {
        "section_style_id",
        "section_key_sha256",
        "corpus_row_count",
        "usable_row_count",
        "excluded_sentence_counts",
        "redacted_sentence_counts",
        "style_signature",
        "guide_text",
        "guide_text_sha256",
        "style_entry_sha256",
    }
)
_FORBIDDEN_STYLE_KEYS = frozenset(
    {
        "reference_excerpt",
        "source_text",
        "minutes",
        "meeting_date",
        "sample_id",
        "source_row_index",
        "evidence_lineage",
        "fact_card",
    }
)
_SAFE_STYLE_SIGNATURE_FIELDS = frozenset(
    {"opening", "sentence_shape", "moves", "lexical_markers"}
)
_SAFE_STYLE_OPENINGS = frozenset(
    {"evidence_first", "attribution_first", "contrast_first"}
)
_SAFE_STYLE_SENTENCE_SHAPES = frozenset({"balanced", "compact", "extended"})
_SAFE_STYLE_MOVES = frozenset(
    {
        "contrast",
        "causal_link",
        "uncertainty",
        "temporal_comparison",
        "source_attribution",
        "group_quantification",
        "risk_balance",
    }
)
_SAFE_STYLE_LEXICAL_MARKERS = frozenset(
    {
        "although",
        "however",
        "while",
        "on balance",
        "appeared",
        "seemed",
        "likely",
        "could",
        "might",
        "reportedly",
        "estimated",
        "suggested",
        "continued",
        "remained",
        "edged",
        "most",
        "many",
        "several",
        "some",
        "few",
    }
)
_SAFE_STYLE_EXCLUSION_REASONS = frozenset(
    {"policy_outcome", "meeting_fact", "person_reference"}
)
_SAFE_STYLE_REDACTION_REASONS = frozenset({"numeric", "date"})


class PreparationError(ValueError):
    """Raised when preparation cannot prove a complete immutable input set."""


MinutesResolver = Callable[[Mapping[str, str]], tuple[str, ...]]


@dataclass(frozen=True)
class LocalTokenCounterBundle:
    """Exact no-truncation counters and tokenizer fingerprints for chk1."""

    generator_token_counter: Callable[[str], int]
    student_token_counter: Callable[[str], int]
    generator_tokenizer_sha256: str
    student_tokenizer_sha256: str


def build_local_token_counters(
    *,
    repo_root: str | Path,
    tokenizer_loader: Callable[[Path], Any] | None = None,
) -> LocalTokenCounterBundle:
    """Use the chk0 tokenizer for both DeepSeek-teacher and student budgets.

    The hosted DeepSeek teacher does not expose a versioned local tokenizer.
    The repository-local chk0 DeepSeek tokenizer is therefore the immutable,
    conservative budget proxy and the exact student tokenizer.  Qwen is not
    loaded here because it is reserved for the chk2 GRPO reward service.
    Neither branch truncates.
    """

    repository = Path(repo_root).expanduser().resolve()
    student_model = resolve_local_model_path(
        repo_root=repository,
        model_path=CHK0_MODEL_RELATIVE_PATH,
        expected_relative_path=CHK0_MODEL_RELATIVE_PATH,
    )
    if tokenizer_loader is None:
        from transformers import AutoTokenizer

        def tokenizer_loader(path: Path) -> Any:
            return AutoTokenizer.from_pretrained(
                str(path),
                local_files_only=True,
                trust_remote_code=True,
            )

    student_tokenizer = tokenizer_loader(student_model)
    generator_tokenizer = student_tokenizer

    def token_length(tokenizer: Any, text: str) -> int:
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence):
            raise PreparationError("tokenizer.encode must return a token sequence")
        return len(token_ids)

    def count_generator(prompt: str) -> int:
        rendered = generator_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str) or not rendered:
            raise PreparationError("DeepSeek chat template did not return text")
        return token_length(generator_tokenizer, rendered)

    def count_student(prompt: str) -> int:
        return token_length(student_tokenizer, prompt)

    return LocalTokenCounterBundle(
        generator_token_counter=count_generator,
        student_token_counter=count_student,
        generator_tokenizer_sha256=fingerprint_tokenizer_payload(student_model)[
            "sha256"
        ],
        student_tokenizer_sha256=fingerprint_tokenizer_payload(student_model)[
            "sha256"
        ],
    )


def normalize_minutes_reference_members(
    excerpts: Iterable[str] | str,
) -> tuple[str, ...]:
    """Return exact Minutes excerpts deduplicated and ordered by content hash.

    This function is intentionally pure.  It neither reads nor writes files,
    and it preserves the exact source bytes represented by each Python string.
    Downstream leakage verification can therefore use the returned tuple while
    reproducing :func:`compute_minutes_reference_sha256` exactly.
    """

    raw_members: Iterable[str] = (excerpts,) if isinstance(excerpts, str) else excerpts
    by_sha256: dict[str, str] = {}
    for index, excerpt in enumerate(raw_members):
        if not isinstance(excerpt, str) or not excerpt.strip():
            raise PreparationError(
                f"Minutes reference member {index} must be non-empty text"
            )
        if "\x00" in excerpt or "\ufffd" in excerpt:
            raise PreparationError(
                f"Minutes reference member {index} contains invalid encoding"
            )
        digest = sha256_text(excerpt)
        prior = by_sha256.setdefault(digest, excerpt)
        if prior != excerpt:
            raise PreparationError("Minutes reference SHA-256 collision detected")
    if not by_sha256:
        raise PreparationError("Minutes reference members must not be empty")
    return tuple(by_sha256[digest] for digest in sorted(by_sha256))


def compute_minutes_reference_sha256(excerpts: Iterable[str] | str) -> str:
    """Hash one or more raw Minutes excerpts using the frozen public policy."""

    members = normalize_minutes_reference_members(excerpts)
    member_hashes = [sha256_text(excerpt) for excerpt in members]
    if len(member_hashes) == 1:
        return member_hashes[0]
    return sha256_text(
        canonical_json({"member_minutes_reference_sha256": member_hashes})
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value or "")
    if not _is_sha256(text):
        raise PreparationError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PreparationError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationError(f"Unable to read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PreparationError(f"{label} must contain one JSON object: {path}")
    return value


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PreparationError(f"Missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PreparationError(
                        f"{label} line {line_number} must be a JSON object"
                    )
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationError(f"Unable to read {label}: {path}") from exc
    return rows


def _canonical_date(value: object, *, label: str) -> str:
    text = str(value or "")
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise PreparationError(f"{label} must be canonical YYYY-MM-DD") from exc
    if parsed.isoformat() != text:
        raise PreparationError(f"{label} must be canonical YYYY-MM-DD")
    return text


def _verify_payload_hash(
    value: Mapping[str, Any],
    *,
    hash_field: str,
    label: str,
) -> str:
    expected = _require_sha256(value.get(hash_field), label=f"{label}.{hash_field}")
    payload = {key: item for key, item in value.items() if key != hash_field}
    observed = sha256_text(canonical_json(payload))
    if observed != expected:
        raise PreparationError(f"{label} payload SHA-256 mismatch")
    return expected


def _resolve_path(raw_path: object, *, base: Path, label: str) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        raise PreparationError(f"{label} path is empty")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise PreparationError(f"Missing {label}: {resolved}")
    return resolved


def _resolve_output_path(raw_path: object, *, ledger_dir: Path, label: str) -> Path:
    text = str(raw_path or "").strip()
    if not text or Path(text).is_absolute():
        raise PreparationError(f"{label} must be a relative ledger output path")
    root = ledger_dir.resolve()
    resolved = (root / text).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PreparationError(f"{label} escapes its ledger directory") from exc
    if not resolved.is_file():
        raise PreparationError(f"Missing {label}: {resolved}")
    return resolved


def _snapshot_inventory_paths(snapshot_manifest: Path) -> list[Path]:
    """Enumerate the sealed manifest and every local cache file beneath it."""

    root = snapshot_manifest.parent.resolve()
    paths: list[Path] = []
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise PreparationError(
                f"Snapshot cache file escapes its sealed directory: {candidate}"
            ) from exc
        paths.append(resolved)
    if snapshot_manifest.resolve() not in paths:
        raise PreparationError(
            f"Snapshot manifest is absent from its cache inventory: {snapshot_manifest}"
        )
    return paths


def _verify_file_hash(path: Path, expected: object, *, label: str) -> str:
    expected_sha = _require_sha256(expected, label=f"{label}.sha256")
    observed = _sha256_file(path)
    if observed != expected_sha:
        raise PreparationError(f"{label} SHA-256 mismatch: {path}")
    return observed


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreparationError(f"{label} must be an object")
    return value


def _require_list(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PreparationError(f"{label} must be a list")
    return value


def _validate_population_record(
    record: Mapping[str, Any],
    *,
    plan_dir: Path,
    seen_meetings: dict[str, str],
) -> tuple[dict[str, Any], Path]:
    population_id = str(record.get("population_id") or "").strip()
    split = str(record.get("split") or "").strip()
    if not population_id:
        raise PreparationError("source plan population_id is empty")
    if split not in _SPLITS:
        raise PreparationError(f"{population_id}: invalid source-plan split {split!r}")
    raw_meetings = _require_list(
        record.get("meeting_dates"), label=f"{population_id}.meeting_dates"
    )
    meetings = [
        _canonical_date(value, label=f"{population_id}.meeting_dates")
        for value in raw_meetings
    ]
    if not meetings or meetings != sorted(set(meetings)):
        raise PreparationError(
            f"{population_id}: meeting_dates must be non-empty, sorted, and unique"
        )
    for meeting in meetings:
        prior = seen_meetings.setdefault(meeting, split)
        if prior != split:
            raise PreparationError(
                f"Meeting {meeting} occurs in source-plan splits {prior!r} and {split!r}"
            )

    population_path = _resolve_path(
        record.get("path"), base=plan_dir, label=f"{population_id} population"
    )
    _verify_file_hash(
        population_path,
        record.get("sha256"),
        label=f"{population_id} population",
    )
    payload = _load_json(population_path, label=f"{population_id} population")
    if payload.get("schema_version") != _POPULATION_SCHEMA_VERSION:
        raise PreparationError(f"{population_id}: unsupported population schema")
    if payload.get("population_id") != population_id:
        raise PreparationError(f"{population_id}: population file ID mismatch")
    if payload.get("meeting_dates") != meetings:
        raise PreparationError(f"{population_id}: population meeting coverage mismatch")
    if payload.get("split_label") not in (None, split):
        raise PreparationError(f"{population_id}: population split mismatch")
    normalized = {
        "population_id": population_id,
        "split": split,
        "meeting_dates": meetings,
        "path": population_path,
        "sha256": str(record["sha256"]),
    }
    return normalized, population_path


def _validate_source_plan(
    handoff: Mapping[str, Any],
    *,
    handoff_dir: Path,
) -> tuple[dict[str, Any], Path, dict[str, dict[str, Any]], list[Path]]:
    plan_binding = _require_mapping(handoff.get("source_plan"), label="source_plan")
    plan_path = _resolve_path(
        plan_binding.get("path"), base=handoff_dir, label="source plan"
    )
    _verify_file_hash(plan_path, plan_binding.get("sha256"), label="source plan")
    plan = _load_json(plan_path, label="source plan")
    if plan.get("schema_version") != SOURCE_PLAN_SCHEMA_VERSION:
        raise PreparationError("Unsupported source plan schema_version")
    plan_payload_sha = _verify_payload_hash(
        plan, hash_field="payload_sha256", label="source plan"
    )
    if plan_binding.get("payload_sha256") != plan_payload_sha:
        raise PreparationError("source_handoff/source_plan payload hash mismatch")

    populations_raw = _require_list(plan.get("populations"), label="plan.populations")
    populations: dict[str, dict[str, Any]] = {}
    seen_meetings: dict[str, str] = {}
    consumed = [plan_path]
    for raw in populations_raw:
        record = _require_mapping(raw, label="source plan population")
        normalized, population_path = _validate_population_record(
            record,
            plan_dir=plan_path.parent,
            seen_meetings=seen_meetings,
        )
        population_id = normalized["population_id"]
        if population_id in populations:
            raise PreparationError(f"Duplicate source-plan population: {population_id}")
        populations[population_id] = normalized
        consumed.append(population_path)

    if not populations:
        raise PreparationError("Source plan contains no populations")
    expected_counts = {
        split: sum(
            len(record["meeting_dates"])
            for record in populations.values()
            if record["split"] == split
        )
        for split in _SPLITS
    }
    if plan.get("meeting_counts") != expected_counts:
        raise PreparationError("Source plan meeting_counts do not match populations")
    if any(count <= 0 for count in expected_counts.values()):
        raise PreparationError("Source plan must cover train, eval, and test meetings")
    return plan, plan_path, populations, consumed


def _validate_manifest_integrity(manifest: Mapping[str, Any], *, label: str) -> str:
    integrity = _require_mapping(manifest.get("integrity"), label=f"{label}.integrity")
    if integrity.get("algorithm") != _INTEGRITY_ALGORITHM:
        raise PreparationError(f"{label}: unsupported integrity algorithm")
    expected = _require_sha256(
        integrity.get("payload_sha256"), label=f"{label}.integrity.payload_sha256"
    )
    payload = {key: value for key, value in manifest.items() if key != "integrity"}
    if sha256_text(canonical_json(payload)) != expected:
        raise PreparationError(f"{label}: manifest integrity hash mismatch")
    return expected


def _validate_output_record(
    record: object,
    *,
    name: str,
    ledger_dir: Path,
) -> tuple[Path, int | None]:
    binding = _require_mapping(record, label=f"ledger.outputs.{name}")
    path = _resolve_output_path(
        binding.get("path"), ledger_dir=ledger_dir, label=f"ledger output {name}"
    )
    _verify_file_hash(path, binding.get("sha256"), label=f"ledger output {name}")
    row_count: int | None = None
    if name != "coverage":
        raw_count = binding.get("row_count")
        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count < 0
        ):
            raise PreparationError(f"ledger.outputs.{name}.row_count is invalid")
        row_count = raw_count
    elif "row_count" in binding:
        raise PreparationError("ledger coverage output must not declare row_count")
    return path, row_count


def _validate_coverage(
    coverage: Mapping[str, Any],
    *,
    population_id: str,
    meetings: Sequence[str],
    ledger_rows: Sequence[Mapping[str, Any]],
    excluded_count: int,
) -> None:
    expected_row_count = len(meetings) * len(ATOMIC_TOPICS)
    if coverage.get("schema_version") != _COVERAGE_SCHEMA_VERSION:
        raise PreparationError(f"{population_id}: unsupported ledger coverage schema")
    scalar_expectations = {
        "population_id": population_id,
        "meeting_count": len(meetings),
        "indicator_count": len(ATOMIC_TOPICS),
        "row_count": expected_row_count,
        "excluded_record_count": excluded_count,
    }
    for field, expected in scalar_expectations.items():
        if coverage.get(field) != expected:
            raise PreparationError(f"{population_id}: coverage {field} mismatch")
    coverage_rows = _require_list(
        coverage.get("rows"), label=f"{population_id}.coverage.rows"
    )
    if len(coverage_rows) != expected_row_count:
        raise PreparationError(f"{population_id}: coverage row count mismatch")
    ledger_keys = {
        (
            str(row.get("sample_id") or ""),
            str(row.get("meeting_date") or ""),
            str(row.get("indicator") or ""),
        )
        for row in ledger_rows
    }
    coverage_keys: set[tuple[str, str, str]] = set()
    for raw in coverage_rows:
        row = _require_mapping(raw, label=f"{population_id} coverage row")
        key = (
            str(row.get("sample_id") or ""),
            str(row.get("meeting_date") or ""),
            str(row.get("indicator") or ""),
        )
        if key in coverage_keys:
            raise PreparationError(
                f"{population_id}: duplicate coverage row {key[0]!r}"
            )
        coverage_keys.add(key)
    if coverage_keys != ledger_keys:
        raise PreparationError(
            f"{population_id}: coverage rows do not bind ledger rows"
        )


def _validate_ledger_rows(
    *,
    population: Mapping[str, Any],
    ledger_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    registry_sha256: str,
    snapshot_payload_sha256: str,
    manifest_sha256: str,
    require_dense_coverage: bool = True,
) -> dict[tuple[str, str], dict[str, Any]]:
    population_id = str(population["population_id"])
    split = str(population["split"])
    meetings = list(population["meeting_dates"])
    expected_count = len(meetings) * len(ATOMIC_TOPICS)
    if len(evidence_rows) != len(ledger_rows) or (
        require_dense_coverage and len(ledger_rows) != expected_count
    ):
        raise PreparationError(
            f"{population_id}: ledger/evidence row coverage is inconsistent"
        )
    if len(ledger_rows) > expected_count:
        raise PreparationError(f"{population_id}: ledger exceeds source-plan coverage")

    evidence_by_sample: dict[str, Mapping[str, Any]] = {}
    for row in evidence_rows:
        if row.get("schema_version") != _SOURCE_EVIDENCE_SCHEMA_VERSION:
            raise PreparationError(
                f"{population_id}: unsupported source-evidence schema"
            )
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in evidence_by_sample:
            raise PreparationError(
                f"{population_id}: duplicate/empty source-evidence sample_id"
            )
        if row.get("population_id") != population_id:
            raise PreparationError(f"{sample_id}: source-evidence population mismatch")
        if row.get("registry_sha256") != registry_sha256:
            raise PreparationError(
                f"{sample_id}: source-evidence registry hash mismatch"
            )
        if row.get("snapshot_manifest_payload_sha256") != snapshot_payload_sha256:
            raise PreparationError(f"{sample_id}: snapshot payload hash mismatch")
        evidence_by_sample[sample_id] = row

    expected_topic_keys = {topic_key(topic) for topic in ATOMIC_TOPICS}
    topics_by_meeting: dict[str, set[str]] = defaultdict(set)
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    seen_sample_ids: set[str] = set()
    for row in ledger_rows:
        if row.get("schema_version") != _LEDGER_ROW_SCHEMA_VERSION:
            raise PreparationError(
                f"{population_id}: unsupported indicator ledger schema"
            )
        meeting = _canonical_date(
            row.get("meeting_date"), label=f"{population_id}.ledger.meeting_date"
        )
        if meeting not in meetings or row.get("meeting_id") != meeting:
            raise PreparationError(f"{population_id}: ledger meeting coverage mismatch")
        try:
            topic = canonical_atomic_topic(row.get("indicator"))
        except ValueError as exc:
            raise PreparationError(
                f"{population_id}: unknown ledger indicator {row.get('indicator')!r}"
            ) from exc
        indicator = ledger_indicator_for_topic(topic)
        if row.get("indicator") != indicator:
            raise PreparationError(f"{population_id}: noncanonical ledger indicator")
        sample_id = f"{meeting}::{indicator}"
        if row.get("sample_id") != sample_id or sample_id in seen_sample_ids:
            raise PreparationError(
                f"{population_id}: invalid/duplicate ledger sample_id"
            )
        seen_sample_ids.add(sample_id)
        topics_by_meeting[meeting].add(topic_key(topic))

        evidence_row = evidence_by_sample.get(sample_id)
        if evidence_row is None:
            raise PreparationError(
                f"{population_id}: missing source evidence for {sample_id}"
            )
        if row.get("source_id") != evidence_row.get("source_id"):
            raise PreparationError(f"{sample_id}: source_id binding mismatch")
        if row.get("source_sha256") != evidence_row.get("source_sha256"):
            raise PreparationError(f"{sample_id}: source SHA binding mismatch")
        try:
            candidates = evidence_from_loo_ledger(
                row,
                evidence_row,
                atomic_topic=topic,
                cutoff_ts=str(row.get("meeting_timestamp") or ""),
            )
        except ValueError as exc:
            raise PreparationError(
                f"{sample_id}: invalid sealed ledger evidence"
            ) from exc
        if not candidates:
            raise PreparationError(
                f"{sample_id}: ledger produced no evidence candidates"
            )
        lookup[(meeting, topic_key(topic))] = {
            "population_id": population_id,
            "split": split,
            "coverage_mode": "dense" if require_dense_coverage else "sparse",
            "manifest_sha256": manifest_sha256,
            "ledger_row": dict(row),
            "evidence_row": dict(evidence_row),
            "evidence_candidates": candidates,
        }

    if set(evidence_by_sample) != seen_sample_ids:
        raise PreparationError(f"{population_id}: orphan source-evidence rows")
    if require_dense_coverage:
        if set(topics_by_meeting) != set(meetings):
            raise PreparationError(f"{population_id}: ledger meeting set mismatch")
        for meeting in meetings:
            if topics_by_meeting[meeting] != expected_topic_keys:
                raise PreparationError(
                    f"{population_id}: {meeting} does not cover the frozen "
                    "26-topic roster"
                )
    return lookup


def _validate_one_dense_ledger(
    record: Mapping[str, Any],
    *,
    population: Mapping[str, Any],
    handoff_dir: Path,
    registry_sha256: str,
    roster_sha256: str,
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    list[Path],
]:
    population_id = str(population["population_id"])
    if record.get("coverage_mode", "dense") != "dense":
        raise PreparationError(f"{population_id}: expected dense ledger coverage")
    if record.get("population_id") != population_id:
        raise PreparationError(f"{population_id}: source_handoff population mismatch")
    if record.get("split") != population["split"]:
        raise PreparationError(f"{population_id}: source_handoff split mismatch")
    if record.get("meeting_count") != len(population["meeting_dates"]):
        raise PreparationError(
            f"{population_id}: source_handoff meeting_count mismatch"
        )

    ledger_dir_text = str(record.get("ledger_dir") or "").strip()
    if not ledger_dir_text:
        raise PreparationError(f"{population_id}: ledger_dir is empty")
    ledger_dir = Path(ledger_dir_text).expanduser()
    if not ledger_dir.is_absolute():
        ledger_dir = handoff_dir / ledger_dir
    ledger_dir = ledger_dir.resolve()
    if not ledger_dir.is_dir():
        raise PreparationError(
            f"Missing ledger directory for {population_id}: {ledger_dir}"
        )
    manifest_path = ledger_dir / "ledger_manifest.json"
    if not manifest_path.is_file():
        raise PreparationError(f"Missing ledger manifest for {population_id}")
    manifest_sha = _verify_file_hash(
        manifest_path,
        record.get("ledger_manifest_sha256"),
        label=f"{population_id} ledger manifest",
    )
    manifest = _load_json(manifest_path, label=f"{population_id} ledger manifest")
    if manifest.get("schema_version") != _DENSE_LEDGER_MANIFEST_SCHEMA_VERSION:
        raise PreparationError(f"{population_id}: unsupported ledger manifest schema")
    if (
        manifest.get("status") != "complete"
        or manifest.get("population_id") != population_id
    ):
        raise PreparationError(
            f"{population_id}: ledger manifest is not complete/bound"
        )
    manifest_payload_sha = _validate_manifest_integrity(
        manifest, label=f"{population_id} ledger manifest"
    )
    if record.get("ledger_payload_sha256") != manifest_payload_sha:
        raise PreparationError(f"{population_id}: handoff/ledger payload hash mismatch")

    inputs = _require_mapping(manifest.get("inputs"), label=f"{population_id}.inputs")
    consumed: list[Path] = [manifest_path]
    input_paths: dict[str, Path] = {}
    for name in ("registry", "snapshot_manifest", "population", "roster"):
        binding = _require_mapping(
            inputs.get(name), label=f"{population_id}.inputs.{name}"
        )
        path = _resolve_path(
            binding.get("path"),
            base=manifest_path.parent,
            label=f"{population_id} {name}",
        )
        _verify_file_hash(path, binding.get("sha256"), label=f"{population_id} {name}")
        input_paths[name] = path
        consumed.append(path)
    if inputs["registry"].get("sha256") != registry_sha256:
        raise PreparationError(f"{population_id}: registry hash differs from handoff")
    if inputs["roster"].get("sha256") != roster_sha256:
        raise PreparationError(f"{population_id}: roster hash differs from handoff")
    if input_paths["population"] != Path(population["path"]):
        raise PreparationError(f"{population_id}: ledger population path mismatch")
    if inputs["population"].get("sha256") != population["sha256"]:
        raise PreparationError(f"{population_id}: ledger population hash mismatch")

    snapshot_path = _resolve_path(
        record.get("snapshot_manifest"),
        base=handoff_dir,
        label=f"{population_id} handoff snapshot manifest",
    )
    snapshot_sha = _verify_file_hash(
        snapshot_path,
        record.get("snapshot_manifest_sha256"),
        label=f"{population_id} handoff snapshot manifest",
    )
    if snapshot_path != input_paths["snapshot_manifest"]:
        raise PreparationError(f"{population_id}: snapshot manifest path mismatch")
    if inputs["snapshot_manifest"].get("sha256") != snapshot_sha:
        raise PreparationError(f"{population_id}: snapshot manifest hash mismatch")
    snapshot_payload_sha = _require_sha256(
        inputs["snapshot_manifest"].get("payload_sha256"),
        label=f"{population_id}.snapshot_manifest.payload_sha256",
    )
    snapshot_manifest = _load_json(
        snapshot_path, label=f"{population_id} snapshot manifest"
    )
    if (
        snapshot_manifest.get("schema_version") != "loo-source-snapshot-manifest-v1"
        or snapshot_manifest.get("status") != "complete"
    ):
        raise PreparationError(f"{population_id}: snapshot manifest is not complete")
    observed_snapshot_payload_sha = _validate_manifest_integrity(
        snapshot_manifest, label=f"{population_id} snapshot manifest"
    )
    if observed_snapshot_payload_sha != snapshot_payload_sha:
        raise PreparationError(f"{population_id}: snapshot payload hash mismatch")
    consumed.extend(_snapshot_inventory_paths(snapshot_path))

    outputs = _require_mapping(
        manifest.get("outputs"), label=f"{population_id}.outputs"
    )
    if set(outputs) != set(_DENSE_LEDGER_OUTPUTS):
        raise PreparationError(f"{population_id}: ledger output set is incomplete")
    output_paths: dict[str, Path] = {}
    output_counts: dict[str, int | None] = {}
    for name in _DENSE_LEDGER_OUTPUTS:
        output_path, row_count = _validate_output_record(
            outputs.get(name), name=name, ledger_dir=ledger_dir
        )
        output_paths[name] = output_path
        output_counts[name] = row_count
        consumed.append(output_path)

    ledger_rows = _load_jsonl(
        output_paths["indicator_inputs"], label=f"{population_id} indicator inputs"
    )
    evidence_rows = _load_jsonl(
        output_paths["source_evidence"], label=f"{population_id} source evidence"
    )
    excluded_rows = _load_jsonl(
        output_paths["excluded_records"], label=f"{population_id} excluded records"
    )
    if output_counts["indicator_inputs"] != len(ledger_rows):
        raise PreparationError(f"{population_id}: indicator_inputs row_count mismatch")
    if output_counts["source_evidence"] != len(evidence_rows):
        raise PreparationError(f"{population_id}: source_evidence row_count mismatch")
    if output_counts["excluded_records"] != len(excluded_rows):
        raise PreparationError(f"{population_id}: excluded_records row_count mismatch")
    if record.get("row_count") != len(ledger_rows):
        raise PreparationError(f"{population_id}: handoff ledger row_count mismatch")

    lookup = _validate_ledger_rows(
        population=population,
        ledger_rows=ledger_rows,
        evidence_rows=evidence_rows,
        registry_sha256=registry_sha256,
        snapshot_payload_sha256=snapshot_payload_sha,
        manifest_sha256=manifest_sha,
    )
    coverage = _load_json(output_paths["coverage"], label=f"{population_id} coverage")
    _validate_coverage(
        coverage,
        population_id=population_id,
        meetings=population["meeting_dates"],
        ledger_rows=ledger_rows,
        excluded_count=len(excluded_rows),
    )
    return lookup, {}, consumed


def _validate_sparse_sample_exclusions(
    rows: Sequence[Mapping[str, Any]],
    *,
    population: Mapping[str, Any],
    registry_sha256: str,
    roster_sha256: str,
    snapshot_payload_sha256: str,
    manifest_sha256: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    population_id = str(population["population_id"])
    split = str(population["split"])
    meetings = set(population["meeting_dates"])
    declared: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw in enumerate(rows, 1):
        row = _require_mapping(
            raw, label=f"{population_id}.sample_exclusions[{index}]"
        )
        if row.get("schema_version") != SPARSE_SAMPLE_EXCLUSION_SCHEMA_VERSION:
            raise PreparationError(
                f"{population_id}: unsupported sparse sample-exclusion schema"
            )
        meeting = _canonical_date(
            row.get("meeting_date"),
            label=f"{population_id}.sample_exclusions[{index}].meeting_date",
        )
        if meeting not in meetings or row.get("split") != split:
            raise PreparationError(
                f"{population_id}: sparse sample-exclusion population mismatch"
            )
        try:
            topic = canonical_atomic_topic(row.get("atomic_topic"))
        except ValueError as exc:
            raise PreparationError(
                f"{population_id}: unknown sparse exclusion topic"
            ) from exc
        indicator = ledger_indicator_for_topic(topic)
        if row.get("ledger_indicator") != indicator:
            raise PreparationError(
                f"{population_id}: sparse exclusion indicator/topic mismatch"
            )
        expected_source_sample_id = stable_sample_id(meeting, topic)
        if row.get("sample_id") != expected_source_sample_id:
            raise PreparationError(
                f"{population_id}: sparse exclusion sample_id mismatch"
            )
        if row.get("reason_code") != SPARSE_SAMPLE_EXCLUSION_REASON:
            raise PreparationError(
                f"{population_id}: unsupported sparse exclusion reason"
            )
        if (
            row.get("registry_sha256") != registry_sha256
            or row.get("roster_sha256") != roster_sha256
            or row.get("snapshot_manifest_payload_sha256")
            != snapshot_payload_sha256
        ):
            raise PreparationError(
                f"{population_id}: sparse exclusion provenance mismatch"
            )
        configured_sources = _require_list(
            row.get("configured_source_keys"),
            label=f"{population_id}.sample_exclusions[{index}].configured_source_keys",
        )
        unusable_sources = _require_list(
            row.get("unusable_sources"),
            label=f"{population_id}.sample_exclusions[{index}].unusable_sources",
        )
        if not configured_sources or len(unusable_sources) != len(configured_sources):
            raise PreparationError(
                f"{population_id}: sparse exclusion source proof is incomplete"
            )
        unusable_reasons = sorted(
            {
                str(
                    _require_mapping(
                        item,
                        label=(
                            f"{population_id}.sample_exclusions[{index}]"
                            ".unusable_sources"
                        ),
                    ).get("reason_code")
                    or ""
                )
                for item in unusable_sources
            }
        )
        if not unusable_reasons or "" in unusable_reasons:
            raise PreparationError(
                f"{population_id}: sparse exclusion lacks sealed reason codes"
            )
        key = (meeting, topic_key(topic))
        if key in declared:
            raise PreparationError(
                f"{population_id}: duplicate sparse exclusion for {meeting}/{topic}"
            )
        declared[key] = {
            "population_id": population_id,
            "split": split,
            "coverage_mode": "sparse",
            "manifest_sha256": manifest_sha256,
            "reason_code": SPARSE_SAMPLE_EXCLUSION_REASON,
            "source_exclusion_sample_id": expected_source_sample_id,
            "source_exclusion_sha256": sha256_text(canonical_json(dict(row))),
            "configured_source_count": len(configured_sources),
            "unusable_source_reason_codes": unusable_reasons,
        }
    return declared


def _validate_one_sparse_ledger(
    record: Mapping[str, Any],
    *,
    population: Mapping[str, Any],
    handoff_dir: Path,
    registry_sha256: str,
    roster_sha256: str,
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    list[Path],
]:
    population_id = str(population["population_id"])
    if record.get("coverage_mode") != "sparse":
        raise PreparationError(f"{population_id}: expected sparse ledger coverage")
    if population.get("split") != "train" or record.get("split") != "train":
        raise PreparationError(
            f"{population_id}: sparse source coverage is restricted to train"
        )
    if record.get("population_id") != population_id:
        raise PreparationError(f"{population_id}: source_handoff population mismatch")
    if record.get("meeting_count") != len(population["meeting_dates"]):
        raise PreparationError(
            f"{population_id}: source_handoff meeting_count mismatch"
        )

    ledger_dir_text = str(record.get("ledger_dir") or "").strip()
    if not ledger_dir_text:
        raise PreparationError(f"{population_id}: ledger_dir is empty")
    ledger_dir = Path(ledger_dir_text).expanduser()
    if not ledger_dir.is_absolute():
        ledger_dir = handoff_dir / ledger_dir
    ledger_dir = ledger_dir.resolve()
    if not ledger_dir.is_dir():
        raise PreparationError(
            f"Missing sparse ledger directory for {population_id}: {ledger_dir}"
        )
    manifest_path = ledger_dir / "ledger_manifest.json"
    manifest_sha = _verify_file_hash(
        manifest_path,
        record.get("ledger_manifest_sha256"),
        label=f"{population_id} sparse ledger manifest",
    )
    manifest = _load_json(
        manifest_path, label=f"{population_id} sparse ledger manifest"
    )
    if (
        manifest.get("schema_version") != SPARSE_LEDGER_MANIFEST_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("population_id") != population_id
    ):
        raise PreparationError(
            f"{population_id}: sparse ledger manifest is not complete/bound"
        )
    manifest_payload_sha = _validate_manifest_integrity(
        manifest, label=f"{population_id} sparse ledger manifest"
    )
    if record.get("ledger_payload_sha256") != manifest_payload_sha:
        raise PreparationError(
            f"{population_id}: handoff/sparse-ledger payload hash mismatch"
        )

    inputs = _require_mapping(manifest.get("inputs"), label=f"{population_id}.inputs")
    if set(inputs) != {"registry", "roster", "snapshot_manifest"}:
        raise PreparationError(f"{population_id}: sparse ledger input set changed")
    consumed: list[Path] = [manifest_path]
    input_paths: dict[str, Path] = {}
    for name in ("registry", "roster", "snapshot_manifest"):
        binding = _require_mapping(
            inputs.get(name), label=f"{population_id}.inputs.{name}"
        )
        input_path = _resolve_path(
            binding.get("path"),
            base=manifest_path.parent,
            label=f"{population_id} sparse {name}",
        )
        _verify_file_hash(
            input_path,
            binding.get("sha256"),
            label=f"{population_id} sparse {name}",
        )
        input_paths[name] = input_path
        consumed.append(input_path)
    if inputs["registry"].get("sha256") != registry_sha256:
        raise PreparationError(f"{population_id}: registry hash differs from handoff")
    if inputs["roster"].get("sha256") != roster_sha256:
        raise PreparationError(f"{population_id}: roster hash differs from handoff")

    snapshot_path = _resolve_path(
        record.get("snapshot_manifest"),
        base=handoff_dir,
        label=f"{population_id} sparse handoff snapshot manifest",
    )
    snapshot_sha = _verify_file_hash(
        snapshot_path,
        record.get("snapshot_manifest_sha256"),
        label=f"{population_id} sparse handoff snapshot manifest",
    )
    if (
        snapshot_path != input_paths["snapshot_manifest"]
        or inputs["snapshot_manifest"].get("sha256") != snapshot_sha
    ):
        raise PreparationError(f"{population_id}: sparse snapshot binding mismatch")
    snapshot_manifest = _load_json(
        snapshot_path, label=f"{population_id} sparse snapshot manifest"
    )
    if (
        snapshot_manifest.get("schema_version")
        != SPARSE_SNAPSHOT_MANIFEST_SCHEMA_VERSION
        or snapshot_manifest.get("status") != "complete"
        or snapshot_manifest.get("population_id") != population_id
    ):
        raise PreparationError(
            f"{population_id}: sparse snapshot manifest is not complete/bound"
        )
    snapshot_payload_sha = _require_sha256(
        inputs["snapshot_manifest"].get("payload_sha256"),
        label=f"{population_id}.snapshot_manifest.payload_sha256",
    )
    if record.get("snapshot_payload_sha256") != snapshot_payload_sha:
        raise PreparationError(
            f"{population_id}: handoff/sparse-snapshot payload hash mismatch"
        )

    try:
        snapshot_validation = validate_sparse_snapshot_manifest(
            snapshot_path,
            registry_file=input_paths["registry"],
            roster_file=input_paths["roster"],
            expected_meeting_dates=population["meeting_dates"],
            expected_population_id=population_id,
        )
        ledger_validation = validate_sparse_loo_ledger(
            manifest_path,
            snapshot_manifest_file=snapshot_path,
            registry_file=input_paths["registry"],
            roster_file=input_paths["roster"],
        )
    except (SparseSourceError, OSError) as exc:
        raise PreparationError(
            f"{population_id}: sparse source replay failed: {exc}"
        ) from exc
    if (
        snapshot_validation.get("status") != "valid"
        or snapshot_validation.get("population_id") != population_id
        or snapshot_validation.get("manifest_payload_sha256")
        != snapshot_payload_sha
    ):
        raise PreparationError(f"{population_id}: sparse snapshot replay differs")
    consumed.extend(_snapshot_inventory_paths(snapshot_path))

    expected_count = len(population["meeting_dates"]) * len(ATOMIC_TOPICS)
    ready_count = ledger_validation.get("ready_sample_count")
    excluded_count = ledger_validation.get("excluded_sample_count")
    replay_expected_count = ledger_validation.get("expected_sample_count")
    counts = (ready_count, excluded_count, replay_expected_count)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
        raise PreparationError(f"{population_id}: sparse replay counts are invalid")
    if (
        ledger_validation.get("status") != "valid"
        or ledger_validation.get("population_id") != population_id
        or ledger_validation.get("manifest_payload_sha256") != manifest_payload_sha
        or replay_expected_count != expected_count
        or ready_count + excluded_count != expected_count
        or record.get("row_count") != ready_count
        or record.get("excluded_sample_count") != excluded_count
        or record.get("expected_sample_count") != expected_count
    ):
        raise PreparationError(f"{population_id}: sparse ledger replay differs")

    outputs = _require_mapping(
        manifest.get("outputs"), label=f"{population_id}.outputs"
    )
    if set(outputs) != set(_SPARSE_LEDGER_OUTPUTS):
        raise PreparationError(f"{population_id}: sparse ledger output set changed")
    output_paths: dict[str, Path] = {}
    output_counts: dict[str, int | None] = {}
    for name in _SPARSE_LEDGER_OUTPUTS:
        output_path, row_count = _validate_output_record(
            outputs.get(name), name=name, ledger_dir=ledger_dir
        )
        output_paths[name] = output_path
        output_counts[name] = row_count
        consumed.append(output_path)

    ledger_rows = _load_jsonl(
        output_paths["indicator_inputs"], label=f"{population_id} indicator inputs"
    )
    evidence_rows = _load_jsonl(
        output_paths["source_evidence"], label=f"{population_id} source evidence"
    )
    excluded_rows = _load_jsonl(
        output_paths["excluded_records"], label=f"{population_id} excluded records"
    )
    source_exclusions = _load_jsonl(
        output_paths["sample_exclusions"],
        label=f"{population_id} sparse sample exclusions",
    )
    _load_json(output_paths["coverage"], label=f"{population_id} sparse coverage")
    observed_counts = {
        "indicator_inputs": len(ledger_rows),
        "source_evidence": len(evidence_rows),
        "excluded_records": len(excluded_rows),
        "sample_exclusions": len(source_exclusions),
    }
    for name, observed in observed_counts.items():
        if output_counts[name] != observed:
            raise PreparationError(f"{population_id}: {name} row_count mismatch")
    if len(ledger_rows) != ready_count or len(source_exclusions) != excluded_count:
        raise PreparationError(f"{population_id}: sparse handoff counts changed")

    exclusion_binding = _require_mapping(
        record.get("sample_exclusions"),
        label=f"{population_id}.handoff.sample_exclusions",
    )
    bound_exclusion_path = _resolve_path(
        exclusion_binding.get("path"),
        base=handoff_dir,
        label=f"{population_id} handoff sample exclusions",
    )
    if bound_exclusion_path != output_paths["sample_exclusions"]:
        raise PreparationError(
            f"{population_id}: handoff sparse exclusion path mismatch"
        )
    _verify_file_hash(
        bound_exclusion_path,
        exclusion_binding.get("sha256"),
        label=f"{population_id} handoff sample exclusions",
    )
    if exclusion_binding.get("row_count") != excluded_count:
        raise PreparationError(
            f"{population_id}: handoff sparse exclusion count mismatch"
        )

    lookup = _validate_ledger_rows(
        population=population,
        ledger_rows=ledger_rows,
        evidence_rows=evidence_rows,
        registry_sha256=registry_sha256,
        snapshot_payload_sha256=snapshot_payload_sha,
        manifest_sha256=manifest_sha,
        require_dense_coverage=False,
    )
    declared = _validate_sparse_sample_exclusions(
        source_exclusions,
        population=population,
        registry_sha256=registry_sha256,
        roster_sha256=roster_sha256,
        snapshot_payload_sha256=snapshot_payload_sha,
        manifest_sha256=manifest_sha,
    )
    expected_keys = {
        (meeting, topic_key(topic))
        for meeting in population["meeting_dates"]
        for topic in ATOMIC_TOPICS
    }
    if set(lookup) & set(declared):
        raise PreparationError(f"{population_id}: sparse ready/excluded overlap")
    if set(lookup) | set(declared) != expected_keys:
        raise PreparationError(
            f"{population_id}: sparse ready/excluded rows do not close source plan"
        )
    return lookup, declared, consumed


def _validate_one_ledger(
    record: Mapping[str, Any],
    *,
    population: Mapping[str, Any],
    handoff_dir: Path,
    registry_sha256: str,
    roster_sha256: str,
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    list[Path],
]:
    coverage_mode = record.get("coverage_mode", "dense")
    if coverage_mode == "dense":
        return _validate_one_dense_ledger(
            record,
            population=population,
            handoff_dir=handoff_dir,
            registry_sha256=registry_sha256,
            roster_sha256=roster_sha256,
        )
    if coverage_mode == "sparse":
        return _validate_one_sparse_ledger(
            record,
            population=population,
            handoff_dir=handoff_dir,
            registry_sha256=registry_sha256,
            roster_sha256=roster_sha256,
        )
    raise PreparationError(
        f"{population['population_id']}: unsupported coverage_mode {coverage_mode!r}"
    )


def _load_source_handoff(path: Path) -> dict[str, Any]:
    handoff_path = path.expanduser().resolve()
    handoff = _load_json(handoff_path, label="source handoff")
    if handoff.get("schema_version") != SOURCE_HANDOFF_SCHEMA_VERSION:
        raise PreparationError("Unsupported source handoff schema_version")
    handoff_payload_sha = _verify_payload_hash(
        handoff, hash_field="payload_sha256", label="source handoff"
    )
    registry_sha = _require_sha256(
        handoff.get("registry_sha256"), label="source_handoff.registry_sha256"
    )
    roster_sha = _require_sha256(
        handoff.get("roster_sha256"), label="source_handoff.roster_sha256"
    )
    plan, plan_path, populations, consumed = _validate_source_plan(
        handoff, handoff_dir=handoff_path.parent
    )

    ledgers_raw = _require_list(handoff.get("ledgers"), label="source_handoff.ledgers")
    ledger_records: dict[str, Mapping[str, Any]] = {}
    for raw in ledgers_raw:
        record = _require_mapping(raw, label="source_handoff ledger")
        population_id = str(record.get("population_id") or "")
        if not population_id or population_id in ledger_records:
            raise PreparationError("source_handoff has duplicate/empty ledger IDs")
        ledger_records[population_id] = record
    if set(ledger_records) != set(populations):
        missing = sorted(set(populations) - set(ledger_records))
        extra = sorted(set(ledger_records) - set(populations))
        raise PreparationError(
            f"source_handoff ledger coverage mismatch; missing={missing}, extra={extra}"
        )

    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    declared_unavailable: dict[tuple[str, str], dict[str, Any]] = {}
    for population_id in sorted(populations):
        population_lookup, population_unavailable, ledger_paths = _validate_one_ledger(
            ledger_records[population_id],
            population=populations[population_id],
            handoff_dir=handoff_path.parent,
            registry_sha256=registry_sha,
            roster_sha256=roster_sha,
        )
        population_keys = set(population_lookup) | set(population_unavailable)
        overlap = sorted((set(lookup) | set(declared_unavailable)) & population_keys)
        if overlap:
            raise PreparationError(
                f"Duplicate source-plan coverage for {overlap[0]}"
            )
        lookup.update(population_lookup)
        declared_unavailable.update(population_unavailable)
        consumed.extend(ledger_paths)

    expected_lookup = {
        (meeting, topic_key(topic))
        for population in populations.values()
        for meeting in population["meeting_dates"]
        for topic in ATOMIC_TOPICS
    }
    if set(lookup) & set(declared_unavailable):
        raise PreparationError("Combined source ledgers have ready/excluded overlap")
    if set(lookup) | set(declared_unavailable) != expected_lookup:
        raise PreparationError(
            "Combined ready/excluded source ledgers do not exactly cover source plan"
        )
    return {
        "handoff": handoff,
        "handoff_path": handoff_path,
        "handoff_payload_sha256": handoff_payload_sha,
        "plan": plan,
        "plan_path": plan_path,
        "populations": populations,
        "lookup": lookup,
        "declared_unavailable": declared_unavailable,
        "consumed_paths": [handoff_path, *consumed],
    }


def _load_student_population(
    root: Path,
) -> tuple[list[dict[str, Any]], list[Path], dict[str, set[str]]]:
    rows: list[dict[str, Any]] = []
    paths: list[Path] = []
    meetings_by_split: dict[str, set[str]] = {split: set() for split in _SPLITS}
    owner: dict[str, str] = {}
    for split in _SPLITS:
        path = root / f"{split}.jsonl"
        split_rows = _load_jsonl(path, label=f"{split} student prompts")
        if not split_rows:
            raise PreparationError(f"{split} student prompt split is empty")
        for index, row in enumerate(split_rows, 1):
            if row.get("split") != split:
                raise PreparationError(f"{path}:{index} split mismatch")
            meeting = _canonical_date(
                row.get("meeting_date"), label=f"{path}:{index}.meeting_date"
            )
            prior = owner.setdefault(meeting, split)
            if prior != split:
                raise PreparationError(
                    f"Meeting {meeting} occurs in student splits {prior!r} and {split!r}"
                )
            meetings_by_split[split].add(meeting)
        rows.extend(split_rows)
        paths.append(path.resolve())
    return rows, paths, meetings_by_split


def _load_minutes_references(
    root: Path,
) -> tuple[
    dict[str, tuple[str, str, str]],
    list[Path],
]:
    references: dict[str, tuple[str, str, str]] = {}
    paths: list[Path] = []
    for split in _SPLITS:
        path = root / f"{split}.jsonl"
        rows = _load_jsonl(path, label=f"{split} Minutes references")
        if not rows:
            raise PreparationError(f"{split} Minutes reference split is empty")
        for index, row in enumerate(rows, 1):
            if row.get("split") != split:
                raise PreparationError(f"{path}:{index} Minutes split mismatch")
            sample_id = str(row.get("sample_id") or "").strip()
            excerpt = row.get("reference_excerpt")
            if not sample_id or not isinstance(excerpt, str) or not excerpt.strip():
                raise PreparationError(
                    f"{path}:{index} Minutes reference lacks sample_id/reference_excerpt"
                )
            binding = (split, sha256_text(excerpt), excerpt)
            prior = references.setdefault(sample_id, binding)
            if prior != binding:
                raise PreparationError(f"Conflicting Minutes reference for {sample_id}")
        paths.append(path.resolve())
    return references, paths


def _minutes_reference_for_sample(
    sample: Mapping[str, Any],
    references: Mapping[str, tuple[str, str, str]],
) -> tuple[str, tuple[str, ...]] | None:
    raw_references: list[str] = []
    split = str(sample["split"])
    for legacy_sample_id in sample.get("legacy_sample_ids", []):
        binding = references.get(str(legacy_sample_id))
        if binding is None:
            continue
        reference_split, reference_sha, raw_reference = binding
        if reference_split != split:
            raise PreparationError(
                f"{sample['sample_id']}: Minutes reference crosses split boundary"
            )
        if reference_sha != sha256_text(raw_reference):
            raise PreparationError(
                f"{sample['sample_id']}: Minutes reference digest mismatch"
            )
        raw_references.append(raw_reference)
    if not raw_references:
        return None
    members = normalize_minutes_reference_members(raw_references)
    return compute_minutes_reference_sha256(members), members


def build_minutes_resolver(
    *,
    student_prompt_dir: str | Path,
    minutes_reference_dir: str | Path,
) -> MinutesResolver:
    """Build an in-memory, read-only same-sample Minutes resolver.

    The resolver rebuilds the canonical sample mapping from reference-free
    student prompts.  It returns exact excerpts as a tuple ordered by member
    SHA-256, so generation can verify the prepared aggregate digest before
    using the text for leakage rejection.  Neither this factory nor the
    returned callable writes Minutes text to disk.
    """

    student_rows, _, _ = _load_student_population(
        Path(student_prompt_dir).expanduser().resolve()
    )
    references, _ = _load_minutes_references(
        Path(minutes_reference_dir).expanduser().resolve()
    )
    try:
        canonical_population = build_canonical_samples(
            student_rows,
            section_style_by_topic=topic_style_map(),
        )
    except (TypeError, ValueError) as exc:
        raise PreparationError("Unable to rebuild canonical Minutes lookup") from exc

    index: dict[str, dict[str, Any]] = {}
    for sample in canonical_population["samples"]:
        binding = _minutes_reference_for_sample(sample, references)
        if binding is None:
            continue
        expected_sha256, members = binding
        sample_id = str(sample["sample_id"])
        index[sample_id] = {
            "split": str(sample["split"]),
            "meeting_date": str(sample["meeting_date"]),
            "atomic_topic": str(sample["atomic_topic"]),
            "minutes_reference_sha256": expected_sha256,
            "members": members,
        }

    def resolve(sample: Mapping[str, str]) -> tuple[str, ...]:
        if not isinstance(sample, Mapping):
            raise PreparationError("Minutes resolver lookup must be an object")
        sample_id = str(sample.get("sample_id") or "").strip()
        record = index.get(sample_id)
        if record is None:
            raise PreparationError(
                f"No Minutes reference for canonical sample {sample_id!r}"
            )
        for field in ("split", "meeting_date", "atomic_topic"):
            if sample.get(field) != record[field]:
                raise PreparationError(
                    f"{sample_id}: Minutes resolver {field} binding mismatch"
                )
        members = tuple(record["members"])
        if (
            compute_minutes_reference_sha256(members)
            != record["minutes_reference_sha256"]
        ):
            raise PreparationError(f"{sample_id}: Minutes resolver digest mismatch")
        return members

    return resolve


def _validate_meeting_coverage(
    *,
    meetings_by_split: Mapping[str, set[str]],
    populations: Mapping[str, Mapping[str, Any]],
) -> None:
    expected = {
        split: {
            meeting
            for population in populations.values()
            if population["split"] == split
            for meeting in population["meeting_dates"]
        }
        for split in _SPLITS
    }
    for split in _SPLITS:
        if meetings_by_split[split] != expected[split]:
            missing = sorted(meetings_by_split[split] - expected[split])
            extra = sorted(expected[split] - meetings_by_split[split])
            raise PreparationError(
                f"{split} source-plan/student meeting coverage mismatch; "
                f"missing_from_plan={missing}, missing_from_students={extra}"
            )


def _project_generator_fact_card(fact_card: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in _GENERATOR_FACT_CARD_FIELDS if field not in fact_card]
    if missing:
        raise PreparationError(f"fact card lacks generator fields: {missing}")
    evidence = _require_list(fact_card.get("evidence"), label="fact_card.evidence")
    lineage = _require_list(
        fact_card.get("evidence_lineage"), label="fact_card.evidence_lineage"
    )
    lineage_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(lineage):
        item = _require_mapping(raw, label=f"fact_card.evidence_lineage[{index}]")
        evidence_id = str(item.get("evidence_id") or "").strip()
        if not evidence_id or evidence_id in lineage_by_id:
            raise PreparationError("fact-card lineage has duplicate/empty evidence_id")
        lineage_by_id[evidence_id] = item

    projected_evidence: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(evidence):
        item = _require_mapping(raw, label=f"fact_card.evidence[{index}]")
        evidence_id = str(item.get("evidence_id") or "").strip()
        if not evidence_id or evidence_id in seen_ids:
            raise PreparationError("fact card has duplicate/empty evidence_id")
        seen_ids.add(evidence_id)
        lineage_item = lineage_by_id.get(evidence_id)
        if lineage_item is None:
            raise PreparationError(
                f"fact-card evidence {evidence_id!r} lacks lineage"
            )
        source_id = str(lineage_item.get("source_id") or "").strip()
        if not source_id:
            raise PreparationError(
                f"fact-card evidence {evidence_id!r} lacks source_id"
            )
        if item.get("source_sha256") != lineage_item.get("source_sha256"):
            raise PreparationError(
                f"fact-card evidence {evidence_id!r} source binding mismatch"
            )
        compact = {
            field: deepcopy(item[field])
            for field in _GENERATOR_EVIDENCE_FIELDS
            if field in item and item[field] not in (None, [], "")
        }
        compact["source_id"] = source_id
        projected_evidence.append(compact)
    if seen_ids != set(lineage_by_id):
        raise PreparationError("fact-card evidence/lineage ID sets differ")

    projected = {
        field: deepcopy(fact_card[field])
        for field in _GENERATOR_FACT_CARD_FIELDS
        if field != "evidence"
    }
    projected["evidence"] = projected_evidence
    if _contains_key(projected, "evidence_lineage"):
        raise PreparationError(
            "generator fact-card projection contains evidence_lineage"
        )
    return projected


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, Mapping):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _iter_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


def _validate_style_counts(
    value: object,
    *,
    allowed_reasons: frozenset[str],
    label: str,
) -> None:
    if not isinstance(value, Mapping) or not set(value).issubset(allowed_reasons):
        raise PreparationError(f"{label} contains an uncontrolled reason")
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in value.values()
    ):
        raise PreparationError(f"{label} counts must be non-negative integers")


def _validate_style_entry_safety(
    style_entry: Mapping[str, Any],
    *,
    raw_references: Sequence[str],
) -> None:
    """Prove that one prepared style entry is metadata/style only.

    Style extraction is already constrained by :mod:`style_guide`; this
    boundary check additionally freezes the exact safe projection and compares
    it with the complete Minutes corpus.  That prevents an otherwise valid
    prepared row from smuggling factual or source text through a style field.
    """

    observed_fields = set(style_entry)
    if observed_fields != _SAFE_STYLE_ENTRY_FIELDS:
        missing = sorted(_SAFE_STYLE_ENTRY_FIELDS - observed_fields)
        extra = sorted(observed_fields - _SAFE_STYLE_ENTRY_FIELDS)
        raise PreparationError(
            "section style entry has an unsafe schema; "
            f"missing={missing}, extra={extra}"
        )
    for field in _FORBIDDEN_STYLE_KEYS:
        if _contains_key(style_entry, field):
            raise PreparationError(
                f"section style entry contains prohibited field {field!r}"
            )
    style_id = str(style_entry.get("section_style_id") or "").strip()
    if not style_id:
        raise PreparationError("section style entry has an empty section_style_id")
    for field in ("section_key_sha256", "guide_text_sha256", "style_entry_sha256"):
        _require_sha256(style_entry.get(field), label=f"{style_id}.{field}")

    corpus_row_count = style_entry.get("corpus_row_count")
    usable_row_count = style_entry.get("usable_row_count")
    if (
        isinstance(corpus_row_count, bool)
        or not isinstance(corpus_row_count, int)
        or corpus_row_count <= 0
        or isinstance(usable_row_count, bool)
        or not isinstance(usable_row_count, int)
        or usable_row_count < 0
        or usable_row_count > corpus_row_count
    ):
        raise PreparationError(f"{style_id}: invalid style corpus counts")
    _validate_style_counts(
        style_entry.get("excluded_sentence_counts"),
        allowed_reasons=_SAFE_STYLE_EXCLUSION_REASONS,
        label=f"{style_id}.excluded_sentence_counts",
    )
    _validate_style_counts(
        style_entry.get("redacted_sentence_counts"),
        allowed_reasons=_SAFE_STYLE_REDACTION_REASONS,
        label=f"{style_id}.redacted_sentence_counts",
    )

    signature = style_entry.get("style_signature")
    if (
        not isinstance(signature, Mapping)
        or set(signature) != _SAFE_STYLE_SIGNATURE_FIELDS
    ):
        raise PreparationError(f"{style_id}: style signature has an unsafe schema")
    if signature.get("opening") not in _SAFE_STYLE_OPENINGS:
        raise PreparationError(f"{style_id}: uncontrolled style opening")
    if signature.get("sentence_shape") not in _SAFE_STYLE_SENTENCE_SHAPES:
        raise PreparationError(f"{style_id}: uncontrolled style sentence shape")
    for field, allowed_values in (
        ("moves", _SAFE_STYLE_MOVES),
        ("lexical_markers", _SAFE_STYLE_LEXICAL_MARKERS),
    ):
        values = signature.get(field)
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) for value in values)
            or len(values) != len(set(values))
            or not set(values).issubset(allowed_values)
        ):
            raise PreparationError(f"{style_id}: uncontrolled style {field}")

    guide_text = style_entry.get("guide_text")
    if not isinstance(guide_text, str) or not guide_text.strip():
        raise PreparationError(f"{style_id}: style guide text must be non-empty")
    if style_entry.get("guide_text_sha256") != sha256_text(guide_text):
        raise PreparationError(f"{style_id}: style guide text SHA-256 mismatch")
    entry_payload = {
        key: value for key, value in style_entry.items() if key != "style_entry_sha256"
    }
    if style_entry.get("style_entry_sha256") != sha256_text(
        canonical_json(entry_payload)
    ):
        raise PreparationError(f"{style_id}: style entry SHA-256 mismatch")

    style_strings = tuple(_iter_strings(style_entry))
    for reference in raw_references:
        if reference and any(reference in value for value in style_strings):
            raise PreparationError(
                f"{style_id}: section style entry contains raw Minutes text"
            )


def _validate_style_guide_source_free(
    style_guide: Mapping[str, Any],
    *,
    minutes_references: Mapping[str, tuple[str, str, str]],
) -> None:
    """Cross-check every style entry against every available Minutes excerpt."""

    raw_references = tuple(
        sorted({binding[2] for binding in minutes_references.values()}, key=sha256_text)
    )
    styles = style_guide.get("styles")
    if not isinstance(styles, list) or not styles:
        raise PreparationError("style guide must contain non-empty styles")
    for entry in styles:
        if not isinstance(entry, Mapping):
            raise PreparationError("style guide entry must be an object")
        _validate_style_entry_safety(entry, raw_references=raw_references)


def _validate_prepared_row_safety(
    row: Mapping[str, Any],
    *,
    raw_references: Sequence[str],
) -> None:
    for field in _FORBIDDEN_PREPARED_KEYS:
        if _contains_key(row, field):
            raise PreparationError(f"prepared row contains prohibited field {field!r}")
    for field in ("fact_card", "generator_prompt", "student_prompt", "provided_data"):
        value = row.get(field)
        if _contains_key(value, "evidence_lineage"):
            raise PreparationError(f"prepared {field} exposes evidence_lineage")
    lineage = row.get("evidence_lineage")
    if not isinstance(lineage, list) or not lineage:
        raise PreparationError("prepared row must retain non-empty top-level lineage")
    style_entry = row.get("section_style_guide")
    if not isinstance(style_entry, Mapping):
        raise PreparationError("prepared row lacks section_style_guide")
    _validate_style_entry_safety(style_entry, raw_references=raw_references)
    if style_entry.get("section_style_id") != row.get("section_style_id"):
        raise PreparationError("prepared row section-style identity mismatch")
    if row.get("input_truncated") is not False:
        raise PreparationError(
            "prepared row must explicitly record input_truncated=false"
        )

    selection = _require_mapping(
        row.get("fact_card_selection"), label="prepared fact_card_selection"
    )
    if (
        selection.get("schema_version") != FACT_CARD_SELECTION_SCHEMA_VERSION
        or selection.get("policy") != FACT_CARD_SELECTION_POLICY
    ):
        raise PreparationError("prepared fact-card selection policy mismatch")
    attempted_limits = _require_list(
        selection.get("attempted_limits"),
        label="prepared fact_card_selection.attempted_limits",
    )
    attempts = _require_list(
        selection.get("attempts"), label="prepared fact_card_selection.attempts"
    )
    if (
        not attempted_limits
        or attempted_limits != list(_FACT_CARD_RECENT_LIMITS[: len(attempted_limits)])
        or len(attempts) != len(attempted_limits)
        or selection.get("selected_max_recent_observations_per_series")
        != attempted_limits[-1]
    ):
        raise PreparationError("prepared fact-card selection attempts are invalid")
    for index, raw_attempt in enumerate(attempts):
        attempt = _require_mapping(
            raw_attempt, label=f"prepared fact_card_selection.attempts[{index}]"
        )
        if attempt.get("max_recent_observations_per_series") != attempted_limits[index]:
            raise PreparationError("prepared fact-card attempt limit mismatch")
        status = attempt.get("status")
        if status not in {"generator_overflow", "student_overflow", "selected"}:
            raise PreparationError("prepared fact-card attempt status is invalid")
        if index < len(attempts) - 1 and status == "selected":
            raise PreparationError("prepared fact-card selection continued after fit")
    selected_attempt = _require_mapping(
        attempts[-1], label="prepared fact_card_selection.selected_attempt"
    )
    if selected_attempt.get("status") != "selected":
        raise PreparationError("prepared fact-card selection has no selected attempt")

    fact_card = _require_mapping(row.get("fact_card"), label="prepared fact_card")
    atomic_topic = str(row.get("atomic_topic") or "")
    expected_generator_prompt = render_generator_payload(
        fact_card=fact_card,
        atomic_topic=atomic_topic,
        style_guide=style_entry,
    )
    expected_student_prompt = render_student_prompt(
        fact_card=fact_card,
        atomic_topic=atomic_topic,
        style_guide=style_entry,
    )
    if row.get("generator_prompt") != expected_generator_prompt:
        raise PreparationError("prepared generator prompt binding mismatch")
    if row.get("student_prompt") != expected_student_prompt:
        raise PreparationError("prepared student prompt binding mismatch")
    if row.get("provided_data") != canonical_json(dict(fact_card)):
        raise PreparationError("prepared provided_data does not bind fact_card")
    prompt_budget = _require_mapping(
        row.get("prompt_budget"), label="prepared prompt_budget"
    )
    generator_budget = _require_mapping(
        prompt_budget.get("generator"), label="prepared prompt_budget.generator"
    )
    student_budget = _require_mapping(
        prompt_budget.get("student"), label="prepared prompt_budget.student"
    )
    if (
        selected_attempt.get("generator_prompt_sha256")
        != generator_budget.get("prompt_sha256")
        or selected_attempt.get("generator_token_count")
        != generator_budget.get("token_count")
        or selected_attempt.get("student_prompt_sha256")
        != student_budget.get("prompt_sha256")
        or selected_attempt.get("student_token_count")
        != student_budget.get("token_count")
    ):
        raise PreparationError("prepared selected fact-card attempt is not prompt-bound")

    prepared_strings = tuple(_iter_strings(row))
    for reference in raw_references:
        if reference and any(reference in value for value in prepared_strings):
            raise PreparationError("prepared row contains same-sample Minutes text")


def _budget_or_exclusion(
    prompt: str,
    *,
    token_counter: Callable[[str], int],
    tokenizer_sha256: str,
    max_tokens: int,
) -> tuple[dict[str, Any] | None, int | None]:
    observed: dict[str, object] = {}

    def capture(text: str) -> int:
        count = token_counter(text)
        observed["count"] = count
        return count

    try:
        audit = validate_prompt_token_budget(
            prompt,
            token_counter=capture,
            max_tokens=max_tokens,
        )
    except PromptBudgetError as exc:
        count = observed.get("count")
        if (
            isinstance(count, int)
            and not isinstance(count, bool)
            and count > max_tokens
        ):
            return None, count
        raise PreparationError("Prompt token-budget validation failed") from exc
    return {**audit, "tokenizer_sha256": tokenizer_sha256}, None


def _decorate_evidence_exclusion(
    exclusion: Mapping[str, Any],
    *,
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": EVIDENCE_EXCLUSION_SCHEMA_VERSION,
        "sample_id": sample["sample_id"],
        "meeting_date": sample["meeting_date"],
        "atomic_topic": sample["atomic_topic"],
        "split": sample["split"],
        "evidence_id": str(exclusion.get("evidence_id") or ""),
        "source_id": str(exclusion.get("source_id") or ""),
        "reason_code": str(exclusion.get("reason") or "invalid_evidence"),
        "detail": str(exclusion.get("detail") or ""),
        "candidate_sha256": exclusion.get("candidate_sha256"),
    }
    return {**payload, "exclusion_sha256": sha256_text(canonical_json(payload))}


def _sample_exclusion(
    sample: Mapping[str, Any],
    *,
    reason_code: str,
    stage: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SAMPLE_EXCLUSION_SCHEMA_VERSION,
        "sample_id": str(sample.get("sample_id") or ""),
        "meeting_date": str(sample.get("meeting_date") or ""),
        "atomic_topic": str(sample.get("atomic_topic") or ""),
        "split": str(sample.get("split") or ""),
        "stage": stage,
        "reason_code": reason_code,
    }
    if details:
        payload["details"] = dict(details)
    return {**payload, "exclusion_sha256": sha256_text(canonical_json(payload))}


def _precanonical_exclusion(
    exclusion: Mapping[str, Any],
    *,
    source_split_by_sample_id: Mapping[str, str],
) -> dict[str, Any]:
    source_sample_id = str(exclusion.get("source_sample_id") or "").strip()
    if not source_sample_id:
        raise PreparationError("canonical exclusion lacks source_sample_id")
    split = source_split_by_sample_id.get(source_sample_id)
    if split not in _SPLITS:
        raise PreparationError(
            f"{source_sample_id}: canonical exclusion lacks a valid source split"
        )
    payload = {
        "schema_version": PRECANONICAL_EXCLUSION_SCHEMA_VERSION,
        "source_sample_id": source_sample_id,
        "meeting_date": _canonical_date(
            exclusion.get("meeting_date"),
            label=f"{source_sample_id}.precanonical.meeting_date",
        ),
        "atomic_topic": str(exclusion.get("atomic_topic") or "").strip(),
        "split": split,
        "stage": "canonical_population",
        "reason_code": str(exclusion.get("reason") or "canonical_exclusion"),
    }
    if not payload["atomic_topic"]:
        raise PreparationError(f"{source_sample_id}: canonical exclusion lacks topic")
    return {**payload, "exclusion_sha256": sha256_text(canonical_json(payload))}


def _prepare_rows(
    *,
    canonical_population: Mapping[str, Any],
    source_bundle: Mapping[str, Any],
    style_guide: Mapping[str, Any],
    minutes_references: Mapping[str, tuple[str, str, str]],
    source_split_by_sample_id: Mapping[str, str],
    generator_token_counter: Callable[[str], int],
    student_token_counter: Callable[[str], int],
    generator_tokenizer_sha256: str,
    student_tokenizer_sha256: str,
    max_tokens: int,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    styles = {
        str(entry["section_style_id"]): dict(entry)
        for entry in style_guide["styles"]
        if isinstance(entry, Mapping) and entry.get("section_style_id")
    }
    prepared: dict[str, list[dict[str, Any]]] = {split: [] for split in _SPLITS}
    evidence_exclusions: list[dict[str, Any]] = []
    sample_exclusions: list[dict[str, Any]] = []
    precanonical_exclusions = [
        _precanonical_exclusion(
            _require_mapping(exclusion, label="canonical population exclusion"),
            source_split_by_sample_id=source_split_by_sample_id,
        )
        for exclusion in canonical_population.get("exclusions", [])
    ]

    for sample in canonical_population["samples"]:
        split = str(sample["split"])
        source_key = (
            sample["meeting_date"],
            topic_key(sample["atomic_topic"]),
        )
        ledger_binding = source_bundle["lookup"].get(source_key)
        if ledger_binding is None:
            source_exclusion = source_bundle["declared_unavailable"].get(source_key)
            if source_exclusion is None:
                raise PreparationError(
                    f"{sample['sample_id']}: no bound source row or exclusion"
                )
            if source_exclusion["split"] != split:
                raise PreparationError(
                    f"{sample['sample_id']}: source exclusion split mismatch"
                )
            sample_exclusions.append(
                _sample_exclusion(
                    sample,
                    reason_code=str(source_exclusion["reason_code"]),
                    stage="source_evidence",
                    details={
                        "coverage_mode": source_exclusion["coverage_mode"],
                        "population_id": source_exclusion["population_id"],
                        "ledger_manifest_sha256": source_exclusion[
                            "manifest_sha256"
                        ],
                        "source_exclusion_sample_id": source_exclusion[
                            "source_exclusion_sample_id"
                        ],
                        "source_exclusion_sha256": source_exclusion[
                            "source_exclusion_sha256"
                        ],
                        "configured_source_count": source_exclusion[
                            "configured_source_count"
                        ],
                        "unusable_source_reason_codes": source_exclusion[
                            "unusable_source_reason_codes"
                        ],
                    },
                )
            )
            continue
        if ledger_binding["split"] != split:
            raise PreparationError(f"{sample['sample_id']}: ledger split mismatch")

        reference_binding = _minutes_reference_for_sample(sample, minutes_references)
        if reference_binding is None:
            sample_exclusions.append(
                _sample_exclusion(
                    sample,
                    reason_code="missing_minutes_reference",
                    stage="minutes_reference",
                )
            )
            continue
        minutes_reference_sha, raw_references = reference_binding

        style_id = str(sample["section_style_id"])
        style_entry = styles.get(style_id)
        if style_entry is None:
            raise PreparationError(
                f"{sample['sample_id']}: style guide lacks {style_id}"
            )

        selection_attempts: list[dict[str, Any]] = []
        selected_bundle: tuple[
            Mapping[str, Any],
            dict[str, Any],
            str,
            str,
            dict[str, Any],
            dict[str, Any],
            int,
        ] | None = None
        fact_card_result: Mapping[str, Any] | None = None
        last_overflow: dict[str, Any] | None = None
        for recent_limit in _FACT_CARD_RECENT_LIMITS:
            fact_card_result = build_fact_card(
                meeting_date=str(sample["meeting_date"]),
                atomic_topic=str(sample["atomic_topic"]),
                cutoff_ts=str(ledger_binding["ledger_row"]["meeting_timestamp"]),
                evidence_rows=ledger_binding["evidence_candidates"],
                max_recent_observations=recent_limit,
            )
            if (
                fact_card_result["status"] != "ready"
                or fact_card_result["fact_card"] is None
            ):
                break
            full_candidate = _require_mapping(
                fact_card_result["fact_card"], label="fact_card_result.fact_card"
            )
            generator_candidate = _project_generator_fact_card(full_candidate)
            generator_prompt_candidate = render_generator_payload(
                fact_card=generator_candidate,
                atomic_topic=str(sample["atomic_topic"]),
                style_guide=style_entry,
            )
            student_prompt_candidate = render_student_prompt(
                fact_card=generator_candidate,
                atomic_topic=str(sample["atomic_topic"]),
                style_guide=style_entry,
            )
            generator_budget_candidate, generator_overflow = _budget_or_exclusion(
                generator_prompt_candidate,
                token_counter=generator_token_counter,
                tokenizer_sha256=generator_tokenizer_sha256,
                max_tokens=max_tokens,
            )
            attempt: dict[str, Any] = {
                "max_recent_observations_per_series": recent_limit,
                "generator_prompt_sha256": sha256_text(generator_prompt_candidate),
                "generator_token_count": (
                    generator_overflow
                    if generator_budget_candidate is None
                    else generator_budget_candidate["token_count"]
                ),
            }
            if generator_budget_candidate is None:
                attempt["status"] = "generator_overflow"
                selection_attempts.append(attempt)
                last_overflow = {
                    "stage": "generator_prompt",
                    "token_count": generator_overflow,
                    "prompt_sha256": sha256_text(generator_prompt_candidate),
                    "tokenizer_sha256": generator_tokenizer_sha256,
                }
                continue

            student_budget_candidate, student_overflow = _budget_or_exclusion(
                student_prompt_candidate,
                token_counter=student_token_counter,
                tokenizer_sha256=student_tokenizer_sha256,
                max_tokens=max_tokens,
            )
            attempt.update(
                {
                    "student_prompt_sha256": sha256_text(student_prompt_candidate),
                    "student_token_count": (
                        student_overflow
                        if student_budget_candidate is None
                        else student_budget_candidate["token_count"]
                    ),
                }
            )
            if student_budget_candidate is None:
                attempt["status"] = "student_overflow"
                selection_attempts.append(attempt)
                last_overflow = {
                    "stage": "student_prompt",
                    "token_count": student_overflow,
                    "prompt_sha256": sha256_text(student_prompt_candidate),
                    "tokenizer_sha256": student_tokenizer_sha256,
                }
                continue

            attempt["status"] = "selected"
            selection_attempts.append(attempt)
            selected_bundle = (
                full_candidate,
                generator_candidate,
                generator_prompt_candidate,
                student_prompt_candidate,
                generator_budget_candidate,
                student_budget_candidate,
                recent_limit,
            )
            break

        if fact_card_result is None:
            raise AssertionError("Fact-card selection attempted no limits")
        evidence_exclusions.extend(
            _decorate_evidence_exclusion(item, sample=sample)
            for item in fact_card_result["exclusions"]
        )
        if (
            fact_card_result["status"] != "ready"
            or fact_card_result["fact_card"] is None
        ):
            reasons = sorted(
                {
                    str(item.get("reason") or "invalid_evidence")
                    for item in fact_card_result["exclusions"]
                }
            )
            sample_exclusions.append(
                _sample_exclusion(
                    sample,
                    reason_code="fact_card_excluded",
                    stage="fact_card",
                    details={"evidence_reason_codes": reasons},
                )
            )
            continue
        if selected_bundle is None:
            if last_overflow is None:
                raise AssertionError("Ready fact card has no selection outcome")
            sample_exclusions.append(
                _sample_exclusion(
                    sample,
                    reason_code="prompt_token_overflow",
                    stage=str(last_overflow["stage"]),
                    details={
                        "token_count": last_overflow["token_count"],
                        "max_tokens": max_tokens,
                        "truncated": False,
                        "prompt_sha256": last_overflow["prompt_sha256"],
                        "tokenizer_sha256": last_overflow["tokenizer_sha256"],
                        "fact_card_selection_policy": FACT_CARD_SELECTION_POLICY,
                        "selection_attempts": selection_attempts,
                    },
                )
            )
            continue

        (
            full_fact_card,
            generator_fact_card,
            generator_prompt,
            student_prompt,
            generator_budget,
            student_budget,
            selected_recent_limit,
        ) = selected_bundle

        provided_data = canonical_json(generator_fact_card)
        row_payload: dict[str, Any] = {
            "schema_version": PREPARED_SAMPLE_SCHEMA_VERSION,
            "sample_id": sample["sample_id"],
            "meeting_date": sample["meeting_date"],
            "atomic_topic": sample["atomic_topic"],
            "split": split,
            "section_style_id": style_id,
            "section_style_guide": deepcopy(style_entry),
            "cutoff_ts": generator_fact_card["cutoff_ts"],
            "fact_card": generator_fact_card,
            "evidence_lineage": deepcopy(full_fact_card["evidence_lineage"]),
            "generator_prompt": generator_prompt,
            "student_prompt": student_prompt,
            "provided_data": provided_data,
            "minutes_reference_sha256": minutes_reference_sha,
            "style_guide_sha256": style_guide["style_guide_sha256"],
            "source_binding": {
                "source_handoff_payload_sha256": source_bundle[
                    "handoff_payload_sha256"
                ],
                "population_id": ledger_binding["population_id"],
                "coverage_mode": ledger_binding["coverage_mode"],
                "ledger_manifest_sha256": ledger_binding["manifest_sha256"],
                "ledger_sample_id": ledger_binding["ledger_row"]["sample_id"],
                "source_sha256": ledger_binding["ledger_row"]["source_sha256"],
                "source_fact_card_sha256": full_fact_card["fact_card_sha256"],
            },
            "prompt_budget": {
                "generator": generator_budget,
                "student": student_budget,
            },
            "fact_card_selection": {
                "schema_version": FACT_CARD_SELECTION_SCHEMA_VERSION,
                "policy": FACT_CARD_SELECTION_POLICY,
                "attempted_limits": [
                    attempt["max_recent_observations_per_series"]
                    for attempt in selection_attempts
                ],
                "selected_max_recent_observations_per_series": (
                    selected_recent_limit
                ),
                "attempts": selection_attempts,
            },
            "input_truncated": False,
            "generator_fact_card_sha256": sha256_text(provided_data),
        }
        _validate_prepared_row_safety(row_payload, raw_references=raw_references)
        row = {
            **row_payload,
            "row_sha256": sha256_text(canonical_json(row_payload)),
        }
        prepared[split].append(row)

    for split in _SPLITS:
        prepared[split].sort(key=lambda item: item["sample_id"])
    evidence_exclusions.sort(
        key=lambda item: (
            item["split"],
            item["sample_id"],
            item["evidence_id"],
            item["reason_code"],
        )
    )
    sample_exclusions.sort(
        key=lambda item: (
            item["split"],
            item["sample_id"],
            item["stage"],
            item["reason_code"],
        )
    )
    precanonical_exclusions.sort(
        key=lambda item: (
            item["split"],
            item["meeting_date"],
            item["source_sample_id"],
            item["atomic_topic"],
            item["reason_code"],
        )
    )

    canonical_ids = {
        str(sample["sample_id"]) for sample in canonical_population["samples"]
    }
    terminal_ids = [
        str(row["sample_id"]) for split in _SPLITS for row in prepared[split]
    ] + [str(row["sample_id"]) for row in sample_exclusions]
    if (
        len(terminal_ids) != len(set(terminal_ids))
        or set(terminal_ids) != canonical_ids
    ):
        raise PreparationError(
            "prepared/sample-exclusion outputs do not close the canonical population"
        )
    return (
        prepared,
        evidence_exclusions,
        sample_exclusions,
        precanonical_exclusions,
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(canonical_json(dict(row)) + "\n" for row in materialized)
    path.write_text(text, encoding="utf-8", newline="\n")
    return len(materialized)


def _file_record(
    path: Path, *, root: Path, row_count: int | None = None
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if row_count is not None:
        record["row_count"] = row_count
    return record


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _finalize_directory(staging: Path, destination: Path) -> None:
    if destination.exists():
        if not destination.is_dir():
            raise PreparationError(
                f"Preparation destination is not a directory: {destination}"
            )
        if _tree_bytes(staging) != _tree_bytes(destination):
            raise PreparationError(
                "Preparation destination already exists with different immutable content"
            )
        shutil.rmtree(staging)
        return
    os.replace(staging, destination)


def prepare_chk1_data(
    *,
    student_prompt_dir: str | Path,
    minutes_reference_dir: str | Path,
    source_handoff_path: str | Path,
    output_dir: str | Path,
    repo_root: str | Path,
    generator_tokenizer_sha256: str,
    student_tokenizer_sha256: str,
    token_counter: Callable[[str], int] | None = None,
    generator_token_counter: Callable[[str], int] | None = None,
    student_token_counter: Callable[[str], int] | None = None,
    inventory_paths: Iterable[str | Path] = (),
    max_tokens: int = PROMPT_TOKEN_LIMIT,
) -> Path:
    """Build the immutable preparation bundle and return its handoff path.

    An existing byte-identical destination is an idempotent success.  An
    existing different destination is never overwritten.  Prompt overflow is
    an auditable sample exclusion; invalid source hashes, missing ledgers, and
    malformed token counters are hard failures.  ``token_counter`` remains a
    compatibility fallback, while model-specific counters take precedence.
    """

    repository = Path(repo_root).expanduser().resolve()
    if not repository.is_dir():
        raise PreparationError(f"repo_root is not a directory: {repository}")
    student_root = Path(student_prompt_dir).expanduser().resolve()
    minutes_root = Path(minutes_reference_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination == repository:
        raise PreparationError("output_dir must not be repo_root")
    generator_counter = (
        generator_token_counter
        if generator_token_counter is not None
        else token_counter
    )
    student_counter = (
        student_token_counter if student_token_counter is not None else token_counter
    )
    if not callable(generator_counter):
        raise PreparationError(
            "generator_token_counter or fallback token_counter must be callable"
        )
    if not callable(student_counter):
        raise PreparationError(
            "student_token_counter or fallback token_counter must be callable"
        )
    generator_tokenizer_digest = _require_sha256(
        generator_tokenizer_sha256,
        label="generator_tokenizer_sha256",
    )
    student_tokenizer_digest = _require_sha256(
        student_tokenizer_sha256,
        label="student_tokenizer_sha256",
    )
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
        or max_tokens > PROMPT_TOKEN_LIMIT
    ):
        raise PreparationError(
            f"max_tokens must be between 1 and the hard limit {PROMPT_TOKEN_LIMIT}"
        )

    try:
        source_bundle = _load_source_handoff(Path(source_handoff_path))
        student_rows, student_paths, student_meetings = _load_student_population(
            student_root
        )
        _validate_meeting_coverage(
            meetings_by_split=student_meetings,
            populations=source_bundle["populations"],
        )
        canonical_population = build_canonical_samples(
            student_rows,
            section_style_by_topic=topic_style_map(),
        )
        source_split_by_sample_id: dict[str, str] = {}
        for row in student_rows:
            source_sample_id = str(row["sample_id"])
            source_split = str(row["split"])
            prior_split = source_split_by_sample_id.setdefault(
                source_sample_id, source_split
            )
            if prior_split != source_split:
                raise PreparationError(
                    f"Student sample {source_sample_id!r} crosses split boundaries"
                )
        minutes_references, minutes_paths = _load_minutes_references(minutes_root)
        style_guide = build_style_guide_from_jsonl(
            minutes_root / "train.jsonl",
            source_id="chk1-train-minutes-reference",
        )
        verify_style_guide_artifact(style_guide)
        _validate_style_guide_source_free(
            style_guide,
            minutes_references=minutes_references,
        )
        (
            prepared,
            evidence_exclusions,
            sample_exclusions,
            precanonical_exclusions,
        ) = _prepare_rows(
            canonical_population=canonical_population,
            source_bundle=source_bundle,
            style_guide=style_guide,
            minutes_references=minutes_references,
            source_split_by_sample_id=source_split_by_sample_id,
            generator_token_counter=generator_counter,
            student_token_counter=student_counter,
            generator_tokenizer_sha256=generator_tokenizer_digest,
            student_tokenizer_sha256=student_tokenizer_digest,
            max_tokens=max_tokens,
        )
    except PreparationError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise PreparationError("chk1 preparation input validation failed") from exc

    all_inventory_paths = {
        Path(path).expanduser().resolve()
        for path in (
            *inventory_paths,
            *student_paths,
            *minutes_paths,
            *source_bundle["consumed_paths"],
        )
    }
    try:
        inventory = build_file_inventory(
            sorted(all_inventory_paths, key=lambda path: path.as_posix()),
            repo_root=repository,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise PreparationError("Unable to build immutable source inventory") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.prepare-", dir=destination.parent)
    )
    try:
        inventory_path = staging / "inventory.json"
        style_path = staging / "style" / "style_guide.json"
        canonical_path = staging / "canonical" / "canonical_population.json"
        _write_json(inventory_path, inventory)
        _write_json(style_path, style_guide)
        _write_json(canonical_path, canonical_population)

        prepared_records: dict[str, dict[str, Any]] = {}
        for split in _SPLITS:
            path = staging / "prepared" / f"{split}.jsonl"
            count = _write_jsonl(path, prepared[split])
            prepared_records[split] = _file_record(path, root=staging, row_count=count)
        evidence_path = staging / "audit" / "evidence_exclusions.jsonl"
        sample_path = staging / "audit" / "sample_exclusions.jsonl"
        precanonical_path = staging / "audit" / "precanonical_exclusions.jsonl"
        evidence_count = _write_jsonl(evidence_path, evidence_exclusions)
        sample_count = _write_jsonl(sample_path, sample_exclusions)
        precanonical_count = _write_jsonl(precanonical_path, precanonical_exclusions)

        artifacts = {
            "inventory": _file_record(inventory_path, root=staging),
            "style_guide": _file_record(style_path, root=staging),
            "canonical_population": _file_record(canonical_path, root=staging),
            "prepared": prepared_records,
            "audit": {
                "evidence_exclusions": _file_record(
                    evidence_path, root=staging, row_count=evidence_count
                ),
                "sample_exclusions": _file_record(
                    sample_path, root=staging, row_count=sample_count
                ),
                "precanonical_exclusions": _file_record(
                    precanonical_path,
                    root=staging,
                    row_count=precanonical_count,
                ),
            },
        }
        handoff_payload = {
            "schema_version": PREPARE_HANDOFF_SCHEMA_VERSION,
            "source_handoff": {
                "sha256": _sha256_file(source_bundle["handoff_path"]),
                "payload_sha256": source_bundle["handoff_payload_sha256"],
            },
            "inventory_payload_sha256": inventory["payload_sha256"],
            "canonical_population_payload_sha256": canonical_population[
                "payload_sha256"
            ],
            "style_guide_sha256": style_guide["style_guide_sha256"],
            "minutes_reference_hash_policy": MINUTES_REFERENCE_HASH_POLICY,
            "generator_tokenizer_sha256": generator_tokenizer_digest,
            "student_tokenizer_sha256": student_tokenizer_digest,
            "max_prompt_tokens": max_tokens,
            "counts": {
                "canonical_samples": len(canonical_population["samples"]),
                "prepared": {split: len(prepared[split]) for split in _SPLITS},
                "evidence_exclusions": evidence_count,
                "sample_exclusions": sample_count,
                "precanonical_exclusions": precanonical_count,
            },
            "artifacts": artifacts,
        }
        prepare_handoff = {
            **handoff_payload,
            "payload_sha256": sha256_text(canonical_json(handoff_payload)),
        }
        _write_json(staging / "prepare_handoff.json", prepare_handoff)
        _finalize_directory(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination / "prepare_handoff.json"


__all__ = [
    "EVIDENCE_EXCLUSION_SCHEMA_VERSION",
    "MINUTES_REFERENCE_HASH_POLICY",
    "LocalTokenCounterBundle",
    "MinutesResolver",
    "PRECANONICAL_EXCLUSION_SCHEMA_VERSION",
    "PREPARED_SAMPLE_SCHEMA_VERSION",
    "PREPARE_HANDOFF_SCHEMA_VERSION",
    "PreparationError",
    "SAMPLE_EXCLUSION_SCHEMA_VERSION",
    "build_minutes_resolver",
    "build_local_token_counters",
    "compute_minutes_reference_sha256",
    "normalize_minutes_reference_members",
    "prepare_chk1_data",
]
