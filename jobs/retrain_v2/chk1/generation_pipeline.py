"""Deterministic, resumable batch generation for prepared chk1 samples.

This layer owns sample selection and durable batch artifacts.  DeepSeek
Reasoner supplies the distilled analysis and answer; same-sample Minutes are
obtained only through a runtime resolver and are never serialized into an
artifact.  Qwen is not part of chk1 generation and remains reserved for chk2
GRPO reward scoring.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from jobs.main.checkpoint_provenance import fingerprint_tokenizer_payload

from .contracts import (
    GENERATOR_SYSTEM_PROMPT,
    MANIFEST_SCHEMA_VERSION,
    canonical_json,
    prompt_template_sha256,
    render_generator_payload,
    render_student_prompt,
    sha256_text,
)
from .deepseek_teacher import (
    OpenAIDeepSeekBackend,
    TeacherResponseRejectedError,
    build_teacher_config,
    candidate_to_sft_response,
    run_deepseek_chk1_generation,
    teacher_model_provenance,
)
from .local_models import (
    CHK0_MODEL_RELATIVE_PATH,
    ModelOutputContractError,
    resolve_local_model_path,
)
from .pipeline import (
    MINUTES_REFERENCE_HASH_POLICY,
    PREPARE_HANDOFF_SCHEMA_VERSION,
    PREPARED_SAMPLE_SCHEMA_VERSION,
    compute_minutes_reference_sha256,
    normalize_minutes_reference_members,
)
from .prompt_projection import (
    PreteacherProjection,
    SftTokenAuditor,
    build_chat_token_counter,
    build_sft_token_auditor,
    project_preteacher_inputs,
    projection_contract_sha256,
)
PILOT_SAMPLE_COUNT = 200
SMOKE_SAMPLE_COUNT = 20
SELECTION_SCHEMA_VERSION = "chk1-generation-selection-v2"
SAMPLE_CACHE_SCHEMA_VERSION = "chk1-generation-sample-cache-v5"
CACHE_MANIFEST_SCHEMA_VERSION = "chk1-generation-cache-manifest-v5"
EXCLUSION_SCHEMA_VERSION = "chk1-generation-exclusion-v1"
GENERATION_HANDOFF_SCHEMA_VERSION = "chk1-generation-handoff-v5"
GENERATION_PROVENANCE_SCHEMA_VERSION = "chk1-generation-provenance-v6"
GENERATION_CODE_SCHEMA_VERSION = "chk1-generation-code-bundle-v1"
SELECTION_ALGORITHM_VERSION = "chk1-topic-proportional-time-spread-v1"
SMOKE_ALGORITHM_VERSION = "chk1-fixed-sha256-priority-v1"
DEFAULT_DEEPSEEK_CONCURRENCY = 8
MAX_DEEPSEEK_CONCURRENCY = 32

_SPLITS = ("train", "eval", "test")
_SPLIT_ORDER = {split: index for index, split in enumerate(_SPLITS)}
_DIGEST_CHARS = frozenset("0123456789abcdef")
_FORBIDDEN_PREPARED_KEYS = {
    "analysis",
    "answer",
    "archived_response",
    "final_analysis",
    "reasoning",
    "reference_excerpt",
    "selected_candidate",
    "selected_response",
    "teacher_response",
    "teacher_model_sha256",
    "tokenizer_sha256",
    "generator_tokenizer_sha256",
    "student_tokenizer_sha256",
    "generation_provenance",
}
_GENERATION_CODE_FILES = (
    "jobs/main/checkpoint_provenance.py",
    "jobs/retrain_v2/chk1/contracts.py",
    "jobs/retrain_v2/chk1/deepseek_teacher.py",
    "jobs/retrain_v2/chk1/generation_pipeline.py",
    "jobs/retrain_v2/chk1/local_models.py",
    "jobs/retrain_v2/chk1/pipeline.py",
    "jobs/retrain_v2/chk1/prompt_projection.py",
)


class GenerationPipelineError(RuntimeError):
    """Base error for deterministic batch generation."""


class PreparedDataError(GenerationPipelineError):
    """Raised when prepared input cannot be identified safely."""


class SelectionError(GenerationPipelineError):
    """Raised when an exact deterministic sample cannot be selected."""


class GenerationResultError(GenerationPipelineError):
    """Raised when a generator result is not safely publishable."""


class ResumeIntegrityError(GenerationPipelineError):
    """Raised when an existing immutable artifact fails validation."""


class SampleGenerator(Protocol):
    def __call__(
        self,
        *,
        prompt: str,
        fact_card: Mapping[str, Any],
        same_sample_minutes: str,
        repo_root: str | Path,
        cache_dir: str | Path | None,
        environment: Mapping[str, str] | None,
    ) -> Mapping[str, Any]: ...


MinutesResolver = Callable[[Mapping[str, str]], str | Sequence[str]]
ProgressCallback = Callable[[Mapping[str, Any]], None]
ModelInputProjector = Callable[
    [Mapping[str, Any], Sequence[Mapping[str, Any]], str, Mapping[str, Any]],
    PreteacherProjection,
]
PreparedSource = str | Path | Sequence[Mapping[str, Any]] | Mapping[str, str | Path]


def _deepseek_concurrency(environment: Mapping[str, str] | None) -> int:
    env = os.environ if environment is None else environment
    raw = str(env.get("DEEPSEEK_CONCURRENCY") or DEFAULT_DEEPSEEK_CONCURRENCY)
    try:
        value = int(raw)
    except ValueError as exc:
        raise PreparedDataError("DEEPSEEK_CONCURRENCY must be an integer") from exc
    if value < 1 or value > MAX_DEEPSEEK_CONCURRENCY:
        raise PreparedDataError(
            f"DEEPSEEK_CONCURRENCY must be between 1 and {MAX_DEEPSEEK_CONCURRENCY}"
        )
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pretty_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(dict(row)) + "\n" for row in rows)


def _payload_with_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(payload)
    return {**copied, "payload_sha256": sha256_text(canonical_json(copied))}


def _validate_payload_hash(
    payload: Mapping[str, Any],
    *,
    schema_version: str,
    label: str,
) -> dict[str, Any]:
    if payload.get("schema_version") != schema_version:
        raise ResumeIntegrityError(f"{label} schema mismatch")
    expected = payload.get("payload_sha256")
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if expected != sha256_text(canonical_json(body)):
        raise ResumeIntegrityError(f"{label} payload hash mismatch")
    return dict(payload)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResumeIntegrityError(f"unable to read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ResumeIntegrityError(f"{label} is not one JSON object: {path}")
    return payload


def _write_immutable(path: Path, text: str, *, resume: bool) -> None:
    """Publish a complete file atomically, never replacing existing content."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not resume:
            raise FileExistsError(f"immutable generation artifact exists: {path}")
        try:
            existing = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ResumeIntegrityError(
                f"unable to validate existing artifact: {path}"
            ) from exc
        if existing != text:
            raise ResumeIntegrityError(f"existing artifact content mismatch: {path}")
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not resume:
                raise
            if path.read_text(encoding="utf-8") != text:
                raise ResumeIntegrityError(
                    f"concurrent immutable artifact mismatch: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _read_jsonl(
    path: Path, *, expected_split: str | None = None
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"prepared JSONL is missing: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PreparedDataError(
                        f"invalid prepared JSON at {path}:{line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise PreparedDataError(
                        f"prepared row is not an object at {path}:{line_number}"
                    )
                if expected_split is not None and row.get("split") != expected_split:
                    raise PreparedDataError(
                        f"prepared split mismatch at {path}:{line_number}"
                    )
                rows.append(row)
    except UnicodeDecodeError as exc:
        raise PreparedDataError(f"prepared JSONL is not valid UTF-8: {path}") from exc
    return rows


def load_prepared_rows(source: PreparedSource) -> list[dict[str, Any]]:
    """Load prepared rows from memory, one JSONL, or split JSONL files."""

    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve()
        if path.is_dir():
            split_paths = {split: path / f"{split}.jsonl" for split in _SPLITS}
            existing = {
                split: item for split, item in split_paths.items() if item.is_file()
            }
            if existing:
                if set(existing) != set(_SPLITS):
                    raise PreparedDataError(
                        "prepared directory must contain train/eval/test JSONL files"
                    )
                rows = []
                for split in _SPLITS:
                    rows.extend(_read_jsonl(existing[split], expected_split=split))
                return _snapshot_rows(rows)
            single = path / "prepared.jsonl"
            if single.is_file():
                return _snapshot_rows(_read_jsonl(single))
            raise FileNotFoundError(f"prepared directory has no JSONL inputs: {path}")
        return _snapshot_rows(_read_jsonl(path))

    if isinstance(source, Mapping):
        if set(source) != set(_SPLITS):
            raise PreparedDataError(
                "prepared path mapping must have exactly train/eval/test keys"
            )
        rows: list[dict[str, Any]] = []
        for split in _SPLITS:
            rows.extend(
                _read_jsonl(
                    Path(source[split]).expanduser().resolve(),
                    expected_split=split,
                )
            )
        return _snapshot_rows(rows)

    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        return _snapshot_rows(source)
    raise PreparedDataError(
        "prepared rows must be a sequence, JSONL path, or split mapping"
    )


def _prepare_artifact_path(
    root: Path,
    record: Any,
    *,
    label: str,
    require_row_count: bool = False,
) -> Path:
    if not isinstance(record, Mapping):
        raise PreparedDataError(f"{label} artifact record must be an object")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise PreparedDataError(f"{label} artifact path is missing")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise PreparedDataError(f"{label} artifact path is unsafe")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PreparedDataError(f"{label} artifact escapes preparation root") from exc
    if not path.is_file():
        raise FileNotFoundError(f"{label} artifact is missing: {path}")
    expected_bytes = record.get("bytes")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
        or path.stat().st_size != expected_bytes
    ):
        raise PreparedDataError(f"{label} artifact byte-count mismatch")
    expected_sha256 = _require_digest(record.get("sha256"), label=f"{label}.sha256")
    if _sha256_file(path) != expected_sha256:
        raise PreparedDataError(f"{label} artifact SHA-256 mismatch")
    if require_row_count:
        row_count = record.get("row_count")
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count < 0
        ):
            raise PreparedDataError(f"{label} row_count is invalid")
        if _count_jsonl_rows(path) != row_count:
            raise PreparedDataError(f"{label} artifact row-count mismatch")
    return path


def _load_prepare_bundle(
    handoff_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay a preparation handoff and bind every consumed byte."""

    path = Path(handoff_path).expanduser().resolve()
    handoff = _validate_payload_hash(
        _load_json(path, label="preparation handoff"),
        schema_version=PREPARE_HANDOFF_SCHEMA_VERSION,
        label="preparation handoff",
    )
    root = path.parent.resolve()
    artifacts = handoff.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise PreparedDataError("preparation handoff artifacts must be an object")
    prepared_records = artifacts.get("prepared")
    if not isinstance(prepared_records, Mapping) or set(prepared_records) != set(
        _SPLITS
    ):
        raise PreparedDataError(
            "preparation handoff must bind train/eval/test prepared files"
        )

    # Validate the complete preparation artifact roster, not only the rows used
    # by generation.  This keeps upstream exclusions and inventory immutable.
    for field in ("inventory", "style_guide", "canonical_population"):
        _prepare_artifact_path(root, artifacts.get(field), label=field)
    audit = artifacts.get("audit")
    expected_audit = {
        "evidence_exclusions",
        "sample_exclusions",
        "precanonical_exclusions",
    }
    if not isinstance(audit, Mapping) or set(audit) != expected_audit:
        raise PreparedDataError("preparation handoff audit records are incomplete")
    for field in sorted(expected_audit):
        _prepare_artifact_path(
            root,
            audit[field],
            label=field,
            require_row_count=True,
        )

    rows: list[dict[str, Any]] = []
    observed_counts: dict[str, int] = {}
    style_sha256 = _require_digest(
        handoff.get("style_guide_sha256"),
        label="preparation handoff style_guide_sha256",
    )
    generator_tokenizer_sha256 = _require_digest(
        handoff.get("generator_tokenizer_sha256"),
        label="preparation handoff generator_tokenizer_sha256",
    )
    student_tokenizer_sha256 = _require_digest(
        handoff.get("student_tokenizer_sha256"),
        label="preparation handoff student_tokenizer_sha256",
    )
    if handoff.get("minutes_reference_hash_policy") != MINUTES_REFERENCE_HASH_POLICY:
        raise PreparedDataError("preparation Minutes hash policy mismatch")
    for split in _SPLITS:
        prepared_path = _prepare_artifact_path(
            root,
            prepared_records[split],
            label=f"prepared.{split}",
            require_row_count=True,
        )
        split_rows = _read_jsonl(prepared_path, expected_split=split)
        for index, row in enumerate(split_rows):
            row_digest = _require_digest(
                row.get("row_sha256"),
                label=f"prepared.{split}[{index}].row_sha256",
            )
            body = dict(row)
            body.pop("row_sha256")
            if sha256_text(canonical_json(body)) != row_digest:
                raise PreparedDataError(
                    f"prepared.{split}[{index}] row_sha256 mismatch"
                )
            if row.get("style_guide_sha256") != style_sha256:
                raise PreparedDataError(
                    f"prepared.{split}[{index}] style-guide binding mismatch"
                )
        rows.extend(split_rows)
        observed_counts[split] = len(split_rows)

    counts = handoff.get("counts")
    if not isinstance(counts, Mapping) or counts.get("prepared") != observed_counts:
        raise PreparedDataError("preparation handoff prepared counts do not reconcile")
    for field in expected_audit:
        if counts.get(field) != audit[field].get("row_count"):
            raise PreparedDataError(
                f"preparation handoff {field} count does not reconcile"
            )
    source_handoff = handoff.get("source_handoff")
    if not isinstance(source_handoff, Mapping):
        raise PreparedDataError("preparation source_handoff binding is missing")
    source_sha256 = _require_digest(
        source_handoff.get("sha256"), label="source_handoff.sha256"
    )
    source_payload_sha256 = _require_digest(
        source_handoff.get("payload_sha256"),
        label="source_handoff.payload_sha256",
    )
    binding_body = {
        "handoff_sha256": _sha256_file(path),
        "handoff_payload_sha256": handoff["payload_sha256"],
        "source_handoff_sha256": source_sha256,
        "source_handoff_payload_sha256": source_payload_sha256,
        "style_guide_sha256": style_sha256,
        "generator_tokenizer_sha256": generator_tokenizer_sha256,
        "student_tokenizer_sha256": student_tokenizer_sha256,
        "minutes_reference_hash_policy": MINUTES_REFERENCE_HASH_POLICY,
        "prepared": json.loads(canonical_json(dict(prepared_records))),
        "sample_exclusions": json.loads(
            canonical_json(dict(audit["sample_exclusions"]))
        ),
        "precanonical_exclusions": json.loads(
            canonical_json(dict(audit["precanonical_exclusions"]))
        ),
    }
    binding = {
        **binding_body,
        "binding_sha256": sha256_text(canonical_json(binding_body)),
    }
    return _snapshot_rows(rows), binding


def _forbidden_keys(value: Any, *, path: str = "row") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            child_path = f"{path}.{raw_key}"
            is_top_level = path == "row" or (
                path.startswith("row[") and path.endswith("]") and "." not in path
            )
            is_allowed_minutes_digest = (
                is_top_level and key == "minutes_reference_sha256"
            )
            is_allowed_budget_tokenizer = key == "tokenizer_sha256" and (
                path.endswith(".prompt_budget.generator")
                or path.endswith(".prompt_budget.student")
            )
            if ("minutes" in key and not is_allowed_minutes_digest) or (
                key in _FORBIDDEN_PREPARED_KEYS and not is_allowed_budget_tokenizer
            ):
                found.append(child_path)
            found.extend(_forbidden_keys(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_keys(child, path=f"{path}[{index}]"))
    return found


def _required_text(row: Mapping[str, Any], field: str, *, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise PreparedDataError(f"{label} requires non-empty text {field!r}")
    cleaned = value.strip()
    if "\x00" in cleaned or "\ufffd" in cleaned:
        raise PreparedDataError(f"{label}.{field} contains invalid encoding")
    return cleaned


def _canonical_date(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise PreparedDataError(f"{label} must be a canonical YYYY-MM-DD date") from exc
    if parsed.isoformat() != text:
        raise PreparedDataError(f"{label} must be a canonical YYYY-MM-DD date")
    return text


def _snapshot_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_keys: set[tuple[str, str]] = set()
    topic_display: dict[str, str] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise PreparedDataError(f"prepared row {index} is not an object")
        forbidden = _forbidden_keys(raw, path=f"row[{index}]")
        if forbidden:
            raise PreparedDataError(
                "prepared rows may not contain Minutes or legacy/generated targets: "
                + ", ".join(forbidden[:5])
            )
        try:
            row = json.loads(canonical_json(dict(raw)))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PreparedDataError(f"prepared row {index} is not finite JSON") from exc
        sample_id = _required_text(row, "sample_id", label=f"row[{index}]")
        if sample_id in seen_ids:
            raise PreparedDataError(f"duplicate prepared sample_id: {sample_id}")
        seen_ids.add(sample_id)
        split = str(row.get("split") or "").strip()
        if split not in _SPLITS:
            raise PreparedDataError(f"{sample_id}: split must be train/eval/test")
        meeting = _canonical_date(
            row.get("meeting_date"), label=f"{sample_id}.meeting_date"
        )
        topic = _required_text(row, "atomic_topic", label=sample_id)
        topic_key = topic.casefold()
        prior_display = topic_display.setdefault(topic_key, topic)
        if prior_display != topic:
            raise PreparedDataError(
                f"inconsistent atomic_topic spelling: {prior_display!r} vs {topic!r}"
            )
        canonical_key = (meeting, topic_key)
        if canonical_key in seen_keys:
            raise PreparedDataError(
                f"duplicate prepared canonical key: {meeting} + {topic}"
            )
        seen_keys.add(canonical_key)
        snapshots.append(row)
    return sorted(snapshots, key=_row_order)


def _row_order(row: Mapping[str, Any]) -> tuple[int, str, str, str]:
    return (
        _SPLIT_ORDER[str(row["split"])],
        str(row["meeting_date"]),
        str(row["atomic_topic"]).casefold(),
        str(row["sample_id"]),
    )


def _pilot_topic_quotas(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    size: int,
) -> dict[str, int]:
    if len(groups) > size:
        raise SelectionError(
            f"pilot size {size} cannot cover all {len(groups)} atomic topics"
        )
    quotas = {topic: 1 for topic in groups}
    remaining = size - len(quotas)
    while remaining:
        available = [
            topic for topic, rows in groups.items() if quotas[topic] < len(rows)
        ]
        if not available:
            raise SelectionError("pilot population cannot satisfy the requested size")
        topic = min(
            available,
            key=lambda item: (
                Fraction(quotas[item], len(groups[item])),
                item,
            ),
        )
        quotas[topic] += 1
        remaining -= 1
    return quotas


def _time_spread(
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            str(row["meeting_date"]),
            str(row["sample_id"]),
        ),
    )
    if count == len(ordered):
        return ordered
    if count == 1:
        return [ordered[(len(ordered) - 1) // 2]]
    last = len(ordered) - 1
    denominator = count - 1
    indices = [
        (index * last + denominator // 2) // denominator for index in range(count)
    ]
    if len(indices) != len(set(indices)):
        raise AssertionError("time-spread selection produced duplicate indices")
    return [ordered[index] for index in indices]


def select_pilot_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    size: int = PILOT_SAMPLE_COUNT,
) -> list[dict[str, Any]]:
    """Select an exact topic-proportional sample spread across meeting time."""

    prepared = _snapshot_rows(rows)
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise SelectionError("pilot size must be a positive integer")
    if len(prepared) < size:
        raise SelectionError(
            f"pilot requires at least {size} prepared rows; found {len(prepared)}"
        )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prepared:
        groups[str(row["atomic_topic"]).casefold()].append(row)
    quotas = _pilot_topic_quotas(groups, size=size)
    selected = [
        row
        for topic in sorted(groups)
        for row in _time_spread(groups[topic], count=quotas[topic])
    ]
    if len(selected) != size or {
        str(row["atomic_topic"]).casefold() for row in selected
    } != set(groups):
        raise AssertionError(
            "pilot selection lost its size or topic coverage invariant"
        )
    return sorted(selected, key=_row_order)


def select_smoke_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select the fixed 20-row smoke cohort by stable SHA-256 priority."""

    prepared = _snapshot_rows(rows)
    if len(prepared) < SMOKE_SAMPLE_COUNT:
        raise SelectionError(
            f"smoke requires at least {SMOKE_SAMPLE_COUNT} prepared rows; "
            f"found {len(prepared)}"
        )
    ranked = sorted(
        prepared,
        key=lambda row: (
            sha256_text(f"{SMOKE_ALGORITHM_VERSION}\x00{row['sample_id']}"),
            str(row["sample_id"]),
        ),
    )
    return sorted(ranked[:SMOKE_SAMPLE_COUNT], key=_row_order)


def select_generation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    prepared = _snapshot_rows(rows)
    if mode == "full":
        return prepared
    if mode == "pilot":
        return select_pilot_rows(prepared)
    if mode == "smoke":
        return select_smoke_rows(prepared)
    raise SelectionError("generation mode must be full, pilot, or smoke")


def _fingerprint_summary(payload: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    digest = _require_digest(payload.get("sha256"), label=f"{label}.sha256")
    file_count = payload.get("file_count")
    total_bytes = payload.get("total_bytes")
    if (
        isinstance(file_count, bool)
        or not isinstance(file_count, int)
        or file_count <= 0
        or isinstance(total_bytes, bool)
        or not isinstance(total_bytes, int)
        or total_bytes <= 0
    ):
        raise PreparedDataError(f"{label} fingerprint inventory is invalid")
    return {
        "sha256": digest,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "kind": str(payload.get("kind") or ""),
        "algorithm": str(payload.get("algorithm") or ""),
    }


def _generation_code_fingerprint() -> dict[str, Any]:
    """Fingerprint the imported repository code that defines target semantics."""

    source_root = Path(__file__).resolve().parents[3]
    records: list[dict[str, Any]] = []
    for relative in _GENERATION_CODE_FILES:
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise PreparedDataError(
                f"generation source file is unavailable: {relative}"
            )
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    payload = {
        "schema_version": GENERATION_CODE_SCHEMA_VERSION,
        "files": records,
    }
    return {**payload, "payload_sha256": sha256_text(canonical_json(payload))}


def build_model_input_projector(
    *,
    repo_root: str | Path,
    tokenizer_loader: Callable[[Path], Any] | None = None,
) -> ModelInputProjector:
    """Load CPU tokenizers and return the mandatory pre-GPU input projector."""

    projector, _auditor = _build_model_input_contracts(
        repo_root=repo_root,
        tokenizer_loader=tokenizer_loader,
    )
    return projector


def _build_model_input_contracts(
    *,
    repo_root: str | Path,
    tokenizer_loader: Callable[[Path], Any] | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[ModelInputProjector, SftTokenAuditor]:
    """Load tokenizers once for both preteacher and accepted-target gates."""

    root = Path(repo_root).expanduser().resolve()
    student_model_path = resolve_local_model_path(
        repo_root=root,
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

    student_tokenizer = tokenizer_loader(student_model_path)
    # The hosted DeepSeek tokenizer is not a versioned local artifact.  Use
    # the immutable chk0 DeepSeek tokenizer as the teacher budget proxy and as
    # the exact SFT/student tokenizer.  Qwen belongs only to chk2 rewards.
    teacher_tokenizer = student_tokenizer
    student_chat_counter = build_chat_token_counter(student_tokenizer)

    def generator_prompt_counter(prompt: str) -> int:
        rendered = teacher_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str) or not rendered:
            raise PreparedDataError("DeepSeek chat template did not return text")
        token_ids = teacher_tokenizer.encode(rendered, add_special_tokens=True)
        if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence):
            raise PreparedDataError("DeepSeek tokenizer returned a non-sequence")
        return len(token_ids)

    def project(
        fact_card: Mapping[str, Any],
        evidence_lineage: Sequence[Mapping[str, Any]],
        atomic_topic: str,
        style_guide: Mapping[str, Any],
    ) -> PreteacherProjection:
        return project_preteacher_inputs(
            fact_card=fact_card,
            evidence_lineage=evidence_lineage,
            atomic_topic=atomic_topic,
            style_guide=style_guide,
            student_chat_token_counter=student_chat_counter,
            generator_prompt_token_counter=generator_prompt_counter,
        )

    return project, build_sft_token_auditor(student_tokenizer)


def _build_test_model_input_projector() -> ModelInputProjector:
    """Structural projection for explicit custom-generator unit tests only."""

    def project(
        fact_card: Mapping[str, Any],
        evidence_lineage: Sequence[Mapping[str, Any]],
        atomic_topic: str,
        style_guide: Mapping[str, Any],
    ) -> PreteacherProjection:
        return project_preteacher_inputs(
            fact_card=fact_card,
            evidence_lineage=evidence_lineage,
            atomic_topic=atomic_topic,
            style_guide=style_guide,
            student_chat_token_counter=lambda _system, _prompt: 0,
            generator_prompt_token_counter=lambda _prompt: 0,
        )

    return project


def _test_sft_token_auditor(_prompt: str, _response: str) -> Mapping[str, Any]:
    """Structural test-only budget result; production always tokenizes exactly."""

    return {
        "schema_version": "chk1-sft-token-budget-v1",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "max_prompt_tokens": 3072,
        "max_completion_tokens": 1024,
        "max_total_tokens": 4096,
        "overflow_policy": "error",
        "truncated": False,
        "passed": True,
    }


def build_generation_provenance(
    *,
    repo_root: str | Path,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Bind the DeepSeek request contract and local student tokenizer once."""

    root = Path(repo_root).expanduser().resolve()
    teacher = build_teacher_config(environment=environment)
    student_model_path = resolve_local_model_path(
        repo_root=root,
        model_path=CHK0_MODEL_RELATIVE_PATH,
        expected_relative_path=CHK0_MODEL_RELATIVE_PATH,
    )
    teacher_model = teacher_model_provenance(teacher)
    generator_tokenizer = _fingerprint_summary(
        fingerprint_tokenizer_payload(student_model_path),
        label="DeepSeek teacher budget tokenizer",
    )
    student_tokenizer = _fingerprint_summary(
        fingerprint_tokenizer_payload(student_model_path),
        label="student tokenizer",
    )
    payload = {
        "schema_version": GENERATION_PROVENANCE_SCHEMA_VERSION,
        "deepseek_concurrency": _deepseek_concurrency(environment),
        "prompt_template_sha256": prompt_template_sha256(),
        "teacher_model": teacher_model,
        "generation_code": _generation_code_fingerprint(),
        "teacher_provider": "deepseek",
        "candidate_processing": "provider_output_contract_only",
        "generator_tokenizer": generator_tokenizer,
        "student_tokenizer": student_tokenizer,
        "teacher_contract": teacher.contract(),
        "teacher_output_mapping": {
            "analysis": "message.reasoning_content",
            "answer": "message.content.answer",
            "sft_response": "analysis + '\\n</think>\\n' + answer",
        },
        "tokenizer_semantics": {
            "generator_tokenizer_sha256": generator_tokenizer["sha256"],
            "manifest_tokenizer_sha256": student_tokenizer["sha256"],
        },
        "model_input_projection_contract_sha256": projection_contract_sha256(),
    }
    return _payload_with_hash(payload)


def _selection_manifest(
    population: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    generation_provenance_sha256: str,
    preparation_binding_sha256: str | None,
) -> dict[str, Any]:
    topic_dates: dict[str, list[str]] = defaultdict(list)
    for row in selected:
        topic_dates[str(row["atomic_topic"])].append(str(row["meeting_date"]))
    payload = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "mode": mode,
        "algorithm": (
            SELECTION_ALGORITHM_VERSION
            if mode == "pilot"
            else SMOKE_ALGORITHM_VERSION
            if mode == "smoke"
            else "all-prepared-rows-v1"
        ),
        "population_count": len(population),
        "population_sha256": sha256_text(
            canonical_json([dict(row) for row in sorted(population, key=_row_order)])
        ),
        "generation_provenance_sha256": generation_provenance_sha256,
        "preparation_binding_sha256": preparation_binding_sha256,
        "selected_count": len(selected),
        "selected_sample_ids": [str(row["sample_id"]) for row in selected],
        "selected_sample_ids_sha256": sha256_text(
            canonical_json([str(row["sample_id"]) for row in selected])
        ),
        "selected_by_split": dict(
            sorted(Counter(str(row["split"]) for row in selected).items())
        ),
        "selected_by_topic": dict(
            sorted(Counter(str(row["atomic_topic"]) for row in selected).items())
        ),
        "selected_time_range_by_topic": {
            topic: {"earliest": min(dates), "latest": max(dates)}
            for topic, dates in sorted(topic_dates.items())
        },
    }
    return _payload_with_hash(payload)


def _require_digest(value: Any, *, label: str) -> str:
    text = str(value or "").strip().casefold()
    if len(text) != 64 or any(character not in _DIGEST_CHARS for character in text):
        raise PreparedDataError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _execution_inputs(
    row: Mapping[str, Any], *, model_input_projector: ModelInputProjector
) -> dict[str, Any]:
    sample_id = _required_text(row, "sample_id", label="prepared row")
    if row.get("schema_version") != PREPARED_SAMPLE_SCHEMA_VERSION:
        raise PreparedDataError(f"{sample_id}: prepared sample schema mismatch")
    fact_card_raw = row.get("fact_card")
    style_raw = row.get("section_style_guide")
    if not isinstance(fact_card_raw, Mapping):
        raise PreparedDataError(f"{sample_id}: fact_card must be an object")
    if not isinstance(style_raw, Mapping):
        raise PreparedDataError(f"{sample_id}: section_style_guide must be an object")
    fact_card = json.loads(canonical_json(dict(fact_card_raw)))
    style_guide = json.loads(canonical_json(dict(style_raw)))
    meeting = str(row["meeting_date"])
    topic = str(row["atomic_topic"])
    cutoff = _required_text(row, "cutoff_ts", label=sample_id)
    section_style_id = _required_text(row, "section_style_id", label=sample_id)
    if (
        fact_card.get("sample_id") != sample_id
        or fact_card.get("meeting_date") != meeting
        or fact_card.get("atomic_topic") != topic
        or fact_card.get("cutoff_ts") != cutoff
        or fact_card.get("canonical_key")
        != {"meeting_date": meeting, "atomic_topic": topic}
    ):
        raise PreparedDataError(f"{sample_id}: fact_card identity binding mismatch")
    if style_guide.get("section_style_id") != section_style_id:
        raise PreparedDataError(f"{sample_id}: section style identity binding mismatch")

    provided_data = canonical_json(fact_card)
    generator_fact_card_sha256 = _require_digest(
        row.get("generator_fact_card_sha256"),
        label=f"{sample_id}.generator_fact_card_sha256",
    )
    if generator_fact_card_sha256 != sha256_text(provided_data):
        raise PreparedDataError(f"{sample_id}: generator fact-card hash mismatch")
    if row.get("provided_data") != provided_data:
        raise PreparedDataError(f"{sample_id}: provided_data does not bind fact_card")

    expected_generator = render_generator_payload(
        fact_card=fact_card,
        atomic_topic=topic,
        style_guide=style_guide,
    )
    expected_student = render_student_prompt(
        fact_card=fact_card,
        atomic_topic=topic,
        style_guide=style_guide,
    )
    if row.get("generator_prompt") != expected_generator:
        raise PreparedDataError(f"{sample_id}: generator prompt binding mismatch")
    if row.get("student_prompt") != expected_student:
        raise PreparedDataError(f"{sample_id}: student prompt binding mismatch")
    if row.get("input_truncated") is not False:
        raise PreparedDataError(
            f"{sample_id}: input_truncated must be explicitly false"
        )
    prompt_budget = row.get("prompt_budget")
    if not isinstance(prompt_budget, Mapping) or set(prompt_budget) != {
        "generator",
        "student",
    }:
        raise PreparedDataError(f"{sample_id}: prompt_budget binding is invalid")
    budget_tokenizers: dict[str, str] = {}
    for budget_name, prompt in (
        ("generator", expected_generator),
        ("student", expected_student),
    ):
        audit = prompt_budget.get(budget_name)
        if not isinstance(audit, Mapping):
            raise PreparedDataError(
                f"{sample_id}: prompt_budget.{budget_name} must be an object"
            )
        if (
            audit.get("prompt_sha256") != sha256_text(prompt)
            or audit.get("truncated") is not False
            or audit.get("overflow_policy") != "error"
        ):
            raise PreparedDataError(
                f"{sample_id}: prompt_budget.{budget_name} content mismatch"
            )
        token_count = audit.get("token_count")
        max_tokens = audit.get("max_tokens")
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 0
            or isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
            or max_tokens > 4096
            or token_count > max_tokens
        ):
            raise PreparedDataError(
                f"{sample_id}: prompt_budget.{budget_name} exceeds its hard gate"
            )
        budget_tokenizers[budget_name] = _require_digest(
            audit.get("tokenizer_sha256"),
            label=f"{sample_id}.prompt_budget.{budget_name}.tokenizer_sha256",
        )

    lineage_raw = row.get("evidence_lineage")
    evidence_raw = fact_card.get("evidence")
    if not isinstance(lineage_raw, list) or not lineage_raw:
        raise PreparedDataError(f"{sample_id}: evidence lineage must be non-empty")
    if not isinstance(evidence_raw, list) or not evidence_raw:
        raise PreparedDataError(f"{sample_id}: fact-card evidence must be non-empty")
    evidence_by_id: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(evidence_raw):
        if not isinstance(item, Mapping):
            raise PreparedDataError(f"{sample_id}: invalid evidence row {index}")
        evidence_id = _required_text(
            item, "evidence_id", label=f"{sample_id}.evidence[{index}]"
        )
        if evidence_id in evidence_by_id:
            raise PreparedDataError(f"{sample_id}: duplicate evidence_id {evidence_id}")
        evidence_by_id[evidence_id] = item

    lineage: list[dict[str, Any]] = []
    lineage_ids: set[str] = set()
    for index, item in enumerate(lineage_raw):
        if not isinstance(item, Mapping):
            raise PreparedDataError(
                f"{sample_id}: invalid evidence lineage row {index}"
            )
        evidence_id = _required_text(
            item,
            "evidence_id",
            label=f"{sample_id}.evidence_lineage[{index}]",
        )
        if evidence_id in lineage_ids:
            raise PreparedDataError(
                f"{sample_id}: duplicate lineage evidence_id {evidence_id}"
            )
        lineage_ids.add(evidence_id)
        source_sha256 = _require_digest(
            item.get("source_sha256"),
            label=f"{sample_id}.evidence_lineage[{index}].source_sha256",
        )
        for digest_field in (
            "evidence_sha256",
            "raw_sha256",
            "request_id",
            "snapshot_manifest_payload_sha256",
            "registry_sha256",
        ):
            _require_digest(
                item.get(digest_field),
                label=f"{sample_id}.evidence_lineage[{index}].{digest_field}",
            )
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            raise PreparedDataError(
                f"{sample_id}: lineage references unknown evidence_id {evidence_id}"
            )
        if (
            evidence.get("source_sha256") != source_sha256
            or item.get("cutoff_ts") != cutoff
            or evidence.get("cutoff_ts") != cutoff
            or item.get("source_id") != evidence.get("source_id")
        ):
            raise PreparedDataError(
                f"{sample_id}: evidence/lineage binding mismatch for {evidence_id}"
            )
        lineage.append(json.loads(canonical_json(dict(item))))
    if lineage_ids != set(evidence_by_id):
        raise PreparedDataError(f"{sample_id}: evidence/lineage ID sets differ")

    try:
        projected = model_input_projector(fact_card, lineage, topic, style_guide)
    except RuntimeError:
        raise
    except (TypeError, ValueError) as exc:
        raise PreparedDataError(
            f"{sample_id}: safe model-input projection failed: {exc}"
        ) from exc
    projected_evidence = projected.fact_card.get("evidence")
    if not isinstance(projected_evidence, list) or not projected_evidence:
        raise PreparedDataError(f"{sample_id}: projected fact card has no evidence")
    projected_ids = {
        str(item.get("evidence_id") or "")
        for item in projected_evidence
        if isinstance(item, Mapping)
    }
    projected_lineage_ids = {
        str(item.get("evidence_id") or "") for item in projected.evidence_lineage
    }
    if (
        "" in projected_ids
        or projected_ids != projected_lineage_ids
        or not projected_ids <= lineage_ids
    ):
        raise PreparedDataError(
            f"{sample_id}: projected evidence/lineage binding mismatch"
        )

    return {
        "sample_id": sample_id,
        "split": str(row["split"]),
        "meeting_date": meeting,
        "atomic_topic": topic,
        "section_style_id": section_style_id,
        "cutoff_ts": cutoff,
        # This full-card subset is used only by deterministic validation/cache
        # logic.  Model messages use the separately rendered safe prompts.
        "fact_card": dict(projected.validation_fact_card),
        "model_fact_card": dict(projected.fact_card),
        "generator_prompt": projected.generator_prompt,
        "student_prompt": projected.student_prompt,
        "provided_data": projected.provided_data,
        "evidence_lineage": [dict(item) for item in projected.evidence_lineage],
        "model_input_projection": dict(projected.attestation),
        "style_guide_sha256": _require_digest(
            row.get("style_guide_sha256"), label=f"{sample_id}.style_guide_sha256"
        ),
        "minutes_reference_sha256": _require_digest(
            row.get("minutes_reference_sha256"),
            label=f"{sample_id}.minutes_reference_sha256",
        ),
        "generator_tokenizer_sha256": budget_tokenizers["generator"],
        "student_tokenizer_sha256": budget_tokenizers["student"],
        "prepared_row_sha256": sha256_text(canonical_json(dict(row))),
    }


def _sample_cache_path(
    output_dir: Path,
    sample_id: str,
    *,
    generation_provenance_sha256: str,
) -> Path:
    digest = sha256_text(sample_id)
    return (
        output_dir
        / "cache"
        / "samples"
        / generation_provenance_sha256
        / digest[:2]
        / f"{digest}.json"
    )


def _sample_exclusion(
    row: Mapping[str, Any],
    *,
    stage: str,
    reason_code: str,
    error_type: str,
    error_codes: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "schema_version": EXCLUSION_SCHEMA_VERSION,
        "sample_id": str(row["sample_id"]),
        "split": str(row["split"]),
        "meeting_date": str(row["meeting_date"]),
        "atomic_topic": str(row["atomic_topic"]),
        "stage": stage,
        "reason_code": reason_code,
        "error_type": error_type,
        "error_codes": sorted(set(str(code) for code in error_codes)),
        "prepared_row_sha256": sha256_text(canonical_json(dict(row))),
    }


def _rejection_codes(error: TeacherResponseRejectedError) -> list[str]:
    payload = error.payload
    if not isinstance(payload, Mapping):
        return []
    codes = payload.get("rejection_error_codes")
    if not isinstance(codes, list):
        return []
    return [str(code) for code in codes if isinstance(code, str) and code]


def _contains_runtime_reference(value: Any, members: Sequence[str]) -> bool:
    """Inspect in-memory values so JSON escaping cannot hide Minutes text."""

    if isinstance(value, str):
        return any(member and member in value for member in members)
    if isinstance(value, Mapping):
        return any(
            _contains_runtime_reference(item, members) for item in value.values()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_runtime_reference(item, members) for item in value)
    return False


def _accepted_artifact(
    row: Mapping[str, Any],
    inputs: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    leakage_members: Sequence[str],
    generation_provenance: Mapping[str, Any],
    preparation_binding_sha256: str | None,
    sft_token_auditor: SftTokenAuditor,
) -> dict[str, Any]:
    if result.get("status") != "accepted":
        raise GenerationResultError("generator did not return accepted status")
    candidate = result.get("selected_candidate")
    response = result.get("selected_response")
    if not isinstance(candidate, Mapping) or not isinstance(response, str):
        raise GenerationResultError("accepted result lacks candidate/response")
    try:
        expected_response = candidate_to_sft_response(candidate)
    except Exception as exc:
        raise GenerationResultError(
            "selected candidate violates the strict teacher contract"
        ) from exc
    if response != expected_response:
        raise GenerationResultError(
            "selected response does not bind selected candidate"
        )
    if result.get("prompt_sha256") != sha256_text(inputs["generator_prompt"]):
        raise GenerationResultError("generator result prompt hash mismatch")
    reasoning = str(candidate["reasoning"]).strip()
    final_analysis = str(candidate["final_analysis"]).strip()
    selected_evidence_ids = [str(item).strip() for item in candidate["evidence_ids"]]
    teacher_output = result.get("teacher_output")
    if (
        not isinstance(teacher_output, Mapping)
        or teacher_output.get("analysis") != reasoning
        or teacher_output.get("answer") != final_analysis
        or teacher_output.get("evidence_ids")
        != selected_evidence_ids
    ):
        raise GenerationResultError(
            "DeepSeek teacher output does not bind analysis/answer/evidence_ids"
        )
    sft_row = {
        "prompt": inputs["student_prompt"],
        "response": response,
        "provided_data": inputs["provided_data"],
    }
    try:
        sft_token_budget = dict(
            sft_token_auditor(sft_row["prompt"], sft_row["response"])
        )
    except Exception as exc:
        sft_token_budget = {
            "schema_version": "chk1-sft-token-budget-observation-v1",
            "status": "unavailable",
            "error_type": type(exc).__name__,
        }
    # Acquisition is deliberately lossless: token counts are retained as
    # informational metadata, but an over-budget DeepSeek response is still
    # written in full.  Any later SFT truncation/packing policy belongs to the
    # training stage, not teacher-data collection.
    try:
        cache_key = _require_digest(
            result.get("cache_key"),
            label=f"{inputs['sample_id']}.generation.cache_key",
        )
    except PreparedDataError as exc:
        raise GenerationResultError("generator cache key is invalid") from exc
    generation = {
        "cache_key": cache_key,
        "selected_from": result.get("selected_from"),
        "generation_provenance_sha256": generation_provenance["payload_sha256"],
        "prompt_template_sha256": generation_provenance["prompt_template_sha256"],
        "generator_tokenizer_sha256": generation_provenance["generator_tokenizer"][
            "sha256"
        ],
        "student_tokenizer_sha256": generation_provenance["student_tokenizer"][
            "sha256"
        ],
        "contracts": {
            "deepseek_teacher": generation_provenance["teacher_contract"],
        },
        "teacher_output_mapping": generation_provenance[
            "teacher_output_mapping"
        ],
        "teacher_response_provenance": result.get("teacher_provenance"),
        "model_input_projection": inputs["model_input_projection"],
        "prepared_row_sha256": inputs["prepared_row_sha256"],
        "preparation_binding_sha256": preparation_binding_sha256,
        "sft_token_budget": sft_token_budget,
    }
    manifest_row = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "sample_id": inputs["sample_id"],
        "meeting_date": inputs["meeting_date"],
        "atomic_topic": inputs["atomic_topic"],
        "section_style_id": inputs["section_style_id"],
        "split": inputs["split"],
        "cutoff_ts": inputs["cutoff_ts"],
        "evidence_lineage": inputs["evidence_lineage"],
        "style_guide_sha256": inputs["style_guide_sha256"],
        "teacher_model_sha256": generation_provenance["teacher_model"]["sha256"],
        # The release tokenizer is the final SFT/student tokenizer.  The same
        # immutable chk0 tokenizer is the hosted-teacher budget proxy.
        "tokenizer_sha256": generation_provenance["student_tokenizer"]["sha256"],
        "generation": generation,
        "prompt_sha256": sha256_text(sft_row["prompt"]),
        "reasoning_sha256": sha256_text(reasoning),
        "final_analysis_sha256": sha256_text(final_analysis),
        "response_sha256": sha256_text(response),
        "provided_data_sha256": sha256_text(sft_row["provided_data"]),
        "selected_evidence_ids": selected_evidence_ids,
        "selected_evidence_ids_sha256": sha256_text(
            canonical_json(selected_evidence_ids)
        ),
        "input_truncated": False,
    }
    artifact = {
        "sft_row": sft_row,
        "manifest_row": manifest_row,
        "exclusion": None,
    }
    if _contains_runtime_reference(artifact, leakage_members):
        raise GenerationResultError("runtime leakage reference escaped into artifacts")
    return artifact


def _sample_cache_payload(
    row: Mapping[str, Any],
    *,
    mode: str,
    minutes_reference_sha256: str | None,
    generation_provenance_sha256: str,
    preparation_binding_sha256: str | None,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    return _payload_with_hash(
        {
            "schema_version": SAMPLE_CACHE_SCHEMA_VERSION,
            "sample_id": str(row["sample_id"]),
            "mode": mode,
            "prepared_row_sha256": sha256_text(canonical_json(dict(row))),
            "minutes_reference_sha256": minutes_reference_sha256,
            "generation_provenance_sha256": generation_provenance_sha256,
            "preparation_binding_sha256": preparation_binding_sha256,
            "artifact": dict(artifact),
        }
    )


def _load_sample_cache(
    path: Path,
    row: Mapping[str, Any],
    *,
    mode: str,
    minutes_reference_sha256: str | None,
    generation_provenance_sha256: str,
    preparation_binding_sha256: str | None,
) -> dict[str, Any]:
    payload = _validate_payload_hash(
        _load_json(path, label="sample cache"),
        schema_version=SAMPLE_CACHE_SCHEMA_VERSION,
        label="sample cache",
    )
    expected = {
        "sample_id": str(row["sample_id"]),
        "mode": mode,
        "prepared_row_sha256": sha256_text(canonical_json(dict(row))),
        "minutes_reference_sha256": minutes_reference_sha256,
        "generation_provenance_sha256": generation_provenance_sha256,
        "preparation_binding_sha256": preparation_binding_sha256,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ResumeIntegrityError(
                f"sample cache {field} mismatch for {row['sample_id']}"
            )
    artifact = payload.get("artifact")
    if not isinstance(artifact, dict):
        raise ResumeIntegrityError(f"sample cache artifact is invalid: {path}")
    return artifact


def _process_sample(
    row: Mapping[str, Any],
    *,
    mode: str,
    output_dir: Path,
    repo_root: Path,
    model_cache_dir: Path | None,
    generator: SampleGenerator,
    minutes_resolver: MinutesResolver,
    generation_provenance: Mapping[str, Any],
    model_input_projector: ModelInputProjector,
    sft_token_auditor: SftTokenAuditor,
    preparation_binding_sha256: str | None,
    environment: Mapping[str, str] | None,
    resume: bool,
) -> dict[str, Any]:
    provenance_sha256 = str(generation_provenance["payload_sha256"])
    cache_path = _sample_cache_path(
        output_dir,
        str(row["sample_id"]),
        generation_provenance_sha256=provenance_sha256,
    )
    try:
        inputs = _execution_inputs(row, model_input_projector=model_input_projector)
    except RuntimeError:
        raise
    except (PreparedDataError, TypeError, ValueError) as exc:
        artifact = {
            "sft_row": None,
            "manifest_row": None,
            "exclusion": _sample_exclusion(
                row,
                stage="prepared_validation",
                reason_code="invalid_prepared_sample",
                error_type=type(exc).__name__,
            ),
        }
        payload = _sample_cache_payload(
            row,
            mode=mode,
            minutes_reference_sha256=None,
            generation_provenance_sha256=provenance_sha256,
            preparation_binding_sha256=preparation_binding_sha256,
            artifact=artifact,
        )
        _write_immutable(cache_path, _pretty_json(payload), resume=resume)
        return artifact

    expected_generator_tokenizer = generation_provenance["generator_tokenizer"][
        "sha256"
    ]
    expected_student_tokenizer = generation_provenance["student_tokenizer"]["sha256"]
    if (
        inputs["generator_tokenizer_sha256"] != expected_generator_tokenizer
        or inputs["student_tokenizer_sha256"] != expected_student_tokenizer
    ):
        raise PreparedDataError(
            f"{inputs['sample_id']}: prompt-budget tokenizer provenance mismatch"
        )

    lookup = MappingProxyType(
        {
            "sample_id": inputs["sample_id"],
            "split": inputs["split"],
            "meeting_date": inputs["meeting_date"],
            "atomic_topic": inputs["atomic_topic"],
        }
    )
    try:
        resolved_minutes = minutes_resolver(lookup)
        members = normalize_minutes_reference_members(resolved_minutes)
        observed_reference_sha256 = compute_minutes_reference_sha256(members)
    except RuntimeError:
        raise
    except (TypeError, ValueError) as exc:
        artifact = {
            "sft_row": None,
            "manifest_row": None,
            "exclusion": _sample_exclusion(
                row,
                stage="runtime_resolver",
                reason_code="split_leakage",
                error_type=type(exc).__name__,
                error_codes=("minutes_reference_invalid",),
            ),
        }
        payload = _sample_cache_payload(
            row,
            mode=mode,
            minutes_reference_sha256=None,
            generation_provenance_sha256=provenance_sha256,
            preparation_binding_sha256=preparation_binding_sha256,
            artifact=artifact,
        )
        _write_immutable(cache_path, _pretty_json(payload), resume=resume)
        return artifact

    leakage_reference = "\n\n".join(members)
    if observed_reference_sha256 != inputs["minutes_reference_sha256"]:
        artifact = {
            "sft_row": None,
            "manifest_row": None,
            "exclusion": _sample_exclusion(
                row,
                stage="runtime_resolver",
                reason_code="split_leakage",
                error_type="MinutesReferenceMismatch",
                error_codes=("minutes_reference_sha256_mismatch",),
            ),
        }
        payload = _sample_cache_payload(
            row,
            mode=mode,
            minutes_reference_sha256=observed_reference_sha256,
            generation_provenance_sha256=provenance_sha256,
            preparation_binding_sha256=preparation_binding_sha256,
            artifact=artifact,
        )
        serialized = _pretty_json(payload)
        if any(member in serialized for member in members):
            raise GenerationPipelineError(
                "runtime leakage reference escaped into sample cache"
            )
        _write_immutable(cache_path, serialized, resume=resume)
        return artifact

    if cache_path.exists():
        if not resume:
            raise FileExistsError(f"sample cache already exists: {cache_path}")
        return _load_sample_cache(
            cache_path,
            row,
            mode=mode,
            minutes_reference_sha256=observed_reference_sha256,
            generation_provenance_sha256=provenance_sha256,
            preparation_binding_sha256=preparation_binding_sha256,
        )

    try:
        result = generator(
            prompt=inputs["generator_prompt"],
            fact_card=inputs["fact_card"],
            same_sample_minutes=leakage_reference,
            repo_root=repo_root,
            cache_dir=model_cache_dir,
            environment=environment,
        )
        if not isinstance(result, Mapping):
            raise GenerationResultError("generator result must be an object")
        if result.get("status") == "rejected":
            raw_codes = result.get("rejection_error_codes", [])
            codes = (
                [str(code) for code in raw_codes if isinstance(code, str) and code]
                if isinstance(raw_codes, list)
                else []
            )
            artifact = {
                "sft_row": None,
                "manifest_row": None,
                "exclusion": _sample_exclusion(
                    row,
                    stage="generation",
                    reason_code="candidate_rejected",
                    error_type="GeneratorRejectedResult",
                    error_codes=codes,
                ),
            }
        else:
            artifact = _accepted_artifact(
                row,
                inputs,
                result,
                leakage_members=members,
                generation_provenance=generation_provenance,
                preparation_binding_sha256=preparation_binding_sha256,
                sft_token_auditor=sft_token_auditor,
            )
    except TeacherResponseRejectedError as exc:
        artifact = {
            "sft_row": None,
            "manifest_row": None,
            "exclusion": _sample_exclusion(
                row,
                stage="generation",
                reason_code="candidate_rejected",
                error_type=type(exc).__name__,
                error_codes=_rejection_codes(exc),
            ),
        }
    except GenerationResultError as exc:
        artifact = {
            "sft_row": None,
            "manifest_row": None,
            "exclusion": _sample_exclusion(
                row,
                stage="result_validation",
                reason_code="invalid_generator_result",
                error_type=type(exc).__name__,
            ),
        }
    except ModelOutputContractError as exc:
        artifact = {
            "sft_row": None,
            "manifest_row": None,
            "exclusion": _sample_exclusion(
                row,
                stage="result_validation",
                reason_code="invalid_generator_result",
                error_type=type(exc).__name__,
            ),
        }
    payload = _sample_cache_payload(
        row,
        mode=mode,
        minutes_reference_sha256=observed_reference_sha256,
        generation_provenance_sha256=provenance_sha256,
        preparation_binding_sha256=preparation_binding_sha256,
        artifact=artifact,
    )
    if _contains_runtime_reference(payload, members):
        raise GenerationPipelineError(
            "runtime leakage reference escaped into sample cache"
        )
    serialized = _pretty_json(payload)
    _write_immutable(cache_path, serialized, resume=resume)
    return artifact


def _file_record(path: Path, *, output_dir: Path, rows: int) -> dict[str, Any]:
    return {
        "path": path.relative_to(output_dir).as_posix(),
        "sha256": _sha256_file(path),
        "rows": rows,
    }


def _count_jsonl_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except (OSError, UnicodeDecodeError) as exc:
        raise ResumeIntegrityError(f"unable to count artifact rows: {path}") from exc


def _load_jsonl_objects(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ResumeIntegrityError(
                        f"{label} contains a blank row at {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ResumeIntegrityError(
                        f"{label} row {line_number} is not an object"
                    )
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResumeIntegrityError(f"unable to parse {label}: {path}") from exc
    return rows


def _validate_complete_handoff(
    handoff_path: Path,
    *,
    output_dir: Path,
    expected_selection: Mapping[str, Any],
    expected_generation_provenance: Mapping[str, Any],
    expected_preparation_binding: Mapping[str, Any] | None,
    expected_sft_token_auditor: SftTokenAuditor,
) -> Path:
    del expected_sft_token_auditor  # Token observations never gate acquisition.
    handoff = _validate_payload_hash(
        _load_json(handoff_path, label="generation handoff"),
        schema_version=GENERATION_HANDOFF_SCHEMA_VERSION,
        label="generation handoff",
    )
    if handoff.get("status") != "complete":
        raise ResumeIntegrityError("generation handoff is not complete")
    if handoff.get("generation_provenance") != expected_generation_provenance:
        raise ResumeIntegrityError("generation handoff provenance mismatch")
    if handoff.get("preparation_binding") != expected_preparation_binding:
        raise ResumeIntegrityError("generation handoff preparation binding mismatch")
    if (
        handoff.get("mode") != expected_selection.get("mode")
        or handoff.get("selected_count") != expected_selection.get("selected_count")
        or handoff.get("selection_payload_sha256")
        != expected_selection.get("payload_sha256")
    ):
        raise ResumeIntegrityError("generation handoff selection binding mismatch")
    selection_path = output_dir / "selection_manifest.json"
    try:
        observed_selection = selection_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ResumeIntegrityError("selection manifest is unavailable") from exc
    if observed_selection != _pretty_json(expected_selection):
        raise ResumeIntegrityError("selection manifest does not match prepared inputs")

    files = handoff.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        "sft",
        "manifests",
        "exclusions",
        "selection",
        "cache_manifest",
    }:
        raise ResumeIntegrityError("generation handoff files record is invalid")

    sft_records = files["sft"]
    manifest_records = files["manifests"]
    if (
        not isinstance(sft_records, Mapping)
        or set(sft_records) != set(_SPLITS)
        or not isinstance(manifest_records, Mapping)
        or set(manifest_records) != set(_SPLITS)
    ):
        raise ResumeIntegrityError("generation handoff split file records are invalid")
    records = [
        *(sft_records[split] for split in _SPLITS),
        *(manifest_records[split] for split in _SPLITS),
        files["exclusions"],
        files["selection"],
        files["cache_manifest"],
    ]
    expected_paths = {
        *(f"sft/{split}.jsonl" for split in _SPLITS),
        *(f"manifests/{split}.jsonl" for split in _SPLITS),
        "audit/exclusions.jsonl",
        "selection_manifest.json",
        "cache/cache_manifest.json",
    }
    observed_paths: set[str] = set()

    def validate_record(record: Any, *, expected_jsonl: bool | None = None) -> Path:
        if not isinstance(record, Mapping):
            raise ResumeIntegrityError("generation handoff file record is invalid")
        relative = Path(str(record.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ResumeIntegrityError(
                "generation handoff path is not relative and safe"
            )
        path = (output_dir / relative).resolve()
        try:
            path.relative_to(output_dir)
        except ValueError as exc:
            raise ResumeIntegrityError(
                "generation handoff path escapes output_dir"
            ) from exc
        if not path.is_file() or _sha256_file(path) != record.get("sha256"):
            raise ResumeIntegrityError(f"generation artifact hash mismatch: {relative}")
        expected_rows = record.get("rows")
        if isinstance(expected_rows, bool) or not isinstance(expected_rows, int):
            raise ResumeIntegrityError(
                f"generation artifact row count is invalid: {relative}"
            )
        is_jsonl = path.suffix == ".jsonl"
        if expected_jsonl is not None and is_jsonl is not expected_jsonl:
            raise ResumeIntegrityError(f"generation artifact type mismatch: {relative}")
        if is_jsonl and _count_jsonl_rows(path) != expected_rows:
            raise ResumeIntegrityError(f"generation artifact row mismatch: {relative}")
        observed_paths.add(relative.as_posix())
        return path

    for record in records:
        validate_record(record)
    if observed_paths != expected_paths:
        raise ResumeIntegrityError("generation handoff file paths are incomplete")

    accepted_counts = handoff.get("accepted_counts")
    if not isinstance(accepted_counts, Mapping) or set(accepted_counts) != set(_SPLITS):
        raise ResumeIntegrityError("generation handoff accepted counts are invalid")
    for split in _SPLITS:
        expected_count = sft_records[split].get("rows")
        if (
            accepted_counts.get(split) != expected_count
            or manifest_records[split].get("rows") != expected_count
        ):
            raise ResumeIntegrityError(
                "generation handoff split counts do not reconcile"
            )
    projection_attestations: list[str] = []
    expected_preparation_sha = (
        expected_preparation_binding.get("binding_sha256")
        if expected_preparation_binding is not None
        else None
    )
    for split in _SPLITS:
        manifest_path = output_dir / str(manifest_records[split]["path"])
        sft_path = output_dir / str(sft_records[split]["path"])
        split_manifests = _load_jsonl_objects(
            manifest_path, label=f"generation manifests/{split}"
        )
        split_sft = _load_jsonl_objects(sft_path, label=f"generation sft/{split}")
        if len(split_manifests) != len(split_sft):
            raise ResumeIntegrityError("manifest/SFT split cardinality mismatch")
        for row, sft_row in zip(split_manifests, split_sft, strict=True):
            generation = row.get("generation")
            if not isinstance(generation, Mapping):
                raise ResumeIntegrityError("manifest generation binding is invalid")
            if generation.get("preparation_binding_sha256") != expected_preparation_sha:
                raise ResumeIntegrityError(
                    "manifest preparation binding does not match generation handoff"
                )
            projection = generation.get("model_input_projection")
            if not isinstance(projection, Mapping):
                raise ResumeIntegrityError("manifest model-input projection is invalid")
            attestation_sha = str(projection.get("attestation_sha256") or "")
            projection_payload = {
                key: value
                for key, value in projection.items()
                if key != "attestation_sha256"
            }
            if projection.get(
                "projection_contract_sha256"
            ) != projection_contract_sha256() or attestation_sha != sha256_text(
                canonical_json(projection_payload)
            ):
                raise ResumeIntegrityError(
                    "manifest model-input projection attestation mismatch"
                )
            prompt = sft_row.get("prompt")
            response = sft_row.get("response")
            if not isinstance(prompt, str) or not isinstance(response, str):
                raise ResumeIntegrityError("SFT prompt/response binding is invalid")
            projection_attestations.append(attestation_sha)
    expected_projection_binding = {
        "projection_contract_sha256": projection_contract_sha256(),
        "accepted_row_count": len(projection_attestations),
        "row_attestations_sha256": sha256_text(canonical_json(projection_attestations)),
    }
    if handoff.get("model_input_projection") != expected_projection_binding:
        raise ResumeIntegrityError("generation handoff projection binding mismatch")
    accepted_count = sum(int(accepted_counts[split]) for split in _SPLITS)
    excluded_count = files["exclusions"].get("rows")
    if (
        handoff.get("accepted_count") != accepted_count
        or handoff.get("excluded_count") != excluded_count
        or accepted_count + int(excluded_count) != handoff.get("selected_count")
    ):
        raise ResumeIntegrityError("generation handoff counts do not reconcile")

    cache_manifest_path = output_dir / str(files["cache_manifest"]["path"])
    cache_manifest = _validate_payload_hash(
        _load_json(cache_manifest_path, label="sample cache manifest"),
        schema_version=CACHE_MANIFEST_SCHEMA_VERSION,
        label="sample cache manifest",
    )
    if cache_manifest.get(
        "generation_provenance_sha256"
    ) != expected_generation_provenance.get("payload_sha256") or cache_manifest.get(
        "preparation_binding_sha256"
    ) != (
        expected_preparation_binding.get("binding_sha256")
        if expected_preparation_binding is not None
        else None
    ):
        raise ResumeIntegrityError("sample cache manifest provenance mismatch")
    entries = cache_manifest.get("entries")
    if not isinstance(entries, list):
        raise ResumeIntegrityError("sample cache manifest entries are invalid")
    expected_ids = set(expected_selection["selected_sample_ids"])
    observed_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ResumeIntegrityError("sample cache manifest entry is invalid")
        sample_id = str(entry.get("sample_id") or "")
        if not sample_id or sample_id in observed_ids:
            raise ResumeIntegrityError("sample cache manifest has duplicate/empty IDs")
        observed_ids.add(sample_id)
        expected_path = _sample_cache_path(
            output_dir,
            sample_id,
            generation_provenance_sha256=str(
                expected_generation_provenance["payload_sha256"]
            ),
        )
        path = validate_record(entry, expected_jsonl=False)
        if path != expected_path.resolve() or entry.get("rows") != 1:
            raise ResumeIntegrityError("sample cache manifest path/count mismatch")
    if observed_ids != expected_ids:
        raise ResumeIntegrityError("sample cache manifest selection mismatch")
    return handoff_path


def run_generation_pipeline(
    *,
    prepared_rows: PreparedSource | None = None,
    prepared_path: str | Path | None = None,
    prepare_handoff_path: str | Path | None = None,
    output_dir: str | Path,
    repo_root: str | Path,
    mode: str = "full",
    generator: SampleGenerator | None = None,
    minutes_resolver: MinutesResolver | None = None,
    allow_test_generator: bool = False,
    model_cache_dir: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    resume: bool = False,
    dry_run: bool = False,
    progress_callback: ProgressCallback | None = None,
    model_input_projector: ModelInputProjector | None = None,
    sft_token_auditor: SftTokenAuditor | None = None,
) -> dict[str, Any] | Path:
    """Generate selected prepared rows and publish a content-bound handoff."""

    if generator is not None and not allow_test_generator:
        raise PreparedDataError(
            "custom generators are test-only; set allow_test_generator=True explicitly"
        )
    if not dry_run and minutes_resolver is None:
        raise PreparedDataError(
            "minutes_resolver is required for every non-dry-run generation"
        )
    provided_sources = sum(
        value is not None
        for value in (prepared_rows, prepared_path, prepare_handoff_path)
    )
    if provided_sources != 1:
        raise PreparedDataError(
            "provide exactly one of prepared_rows, prepared_path, or "
            "prepare_handoff_path"
        )
    preparation_binding: dict[str, Any] | None = None
    if prepare_handoff_path is not None:
        population, preparation_binding = _load_prepare_bundle(prepare_handoff_path)
    elif prepared_path is not None:
        source: PreparedSource = prepared_path
        population = load_prepared_rows(source)
    else:
        assert prepared_rows is not None
        population = load_prepared_rows(prepared_rows)

    root = Path(repo_root).expanduser().resolve()
    generation_provenance = build_generation_provenance(
        repo_root=root, environment=environment
    )
    provenance_sha256 = str(generation_provenance["payload_sha256"])
    preparation_binding_sha256 = (
        str(preparation_binding["binding_sha256"])
        if preparation_binding is not None
        else None
    )
    if preparation_binding is not None and (
        preparation_binding["generator_tokenizer_sha256"]
        != generation_provenance["generator_tokenizer"]["sha256"]
        or preparation_binding["student_tokenizer_sha256"]
        != generation_provenance["student_tokenizer"]["sha256"]
    ):
        raise PreparedDataError(
            "preparation handoff tokenizer provenance does not match local models"
        )
    if model_input_projector is None:
        if allow_test_generator:
            model_input_projector = _build_test_model_input_projector()
            sft_token_auditor = sft_token_auditor or _test_sft_token_auditor
        else:
            model_input_projector, production_auditor = _build_model_input_contracts(
                repo_root=root, environment=environment
            )
            sft_token_auditor = sft_token_auditor or production_auditor
    elif sft_token_auditor is None:
        if allow_test_generator:
            sft_token_auditor = _test_sft_token_auditor
        elif not dry_run:
            raise PreparedDataError(
                "custom production model_input_projector requires sft_token_auditor"
            )
    if sft_token_auditor is None:
        # Dry runs never accept a target, but keep the call graph total.
        sft_token_auditor = _test_sft_token_auditor

    selected = select_generation_rows(population, mode=mode)
    selection = _selection_manifest(
        population,
        selected,
        mode=mode,
        generation_provenance_sha256=provenance_sha256,
        preparation_binding_sha256=preparation_binding_sha256,
    )
    destination = Path(output_dir).expanduser().resolve()

    if dry_run:
        invalid = 0
        for row in selected:
            try:
                inputs = _execution_inputs(
                    row, model_input_projector=model_input_projector
                )
                if (
                    inputs["generator_tokenizer_sha256"]
                    != generation_provenance["generator_tokenizer"]["sha256"]
                    or inputs["student_tokenizer_sha256"]
                    != generation_provenance["student_tokenizer"]["sha256"]
                ):
                    raise PreparedDataError("prompt-budget tokenizer mismatch")
            except RuntimeError:
                raise
            except (PreparedDataError, TypeError, ValueError):
                invalid += 1
        return {
            "status": "dry_run",
            "mode": mode,
            "selection": selection,
            "generation_provenance": generation_provenance,
            "preparation_binding": preparation_binding,
            "planned_model_calls": len(selected) - invalid,
            "planned_prepared_exclusions": invalid,
            "would_write": [
                "sft/{train,eval,test}.jsonl",
                "manifests/{train,eval,test}.jsonl",
                "audit/exclusions.jsonl",
                "selection_manifest.json",
                "generation_handoff.json",
            ],
        }

    handoff_path = destination / "generation_handoff.json"
    if handoff_path.exists():
        if not resume:
            raise FileExistsError(f"generation handoff already exists: {handoff_path}")
        return _validate_complete_handoff(
            handoff_path,
            output_dir=destination,
            expected_selection=selection,
            expected_generation_provenance=generation_provenance,
            expected_preparation_binding=preparation_binding,
            expected_sft_token_auditor=sft_token_auditor,
        )
    if destination.exists() and any(destination.iterdir()) and not resume:
        raise FileExistsError(
            f"generation output directory is not empty: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    if generator is None:
        persistent_teacher_backend = OpenAIDeepSeekBackend()

        def generator_fn(
            *,
            prompt: str,
            fact_card: Mapping[str, Any],
            same_sample_minutes: str,
            repo_root: str | Path,
            cache_dir: str | Path | None,
            environment: Mapping[str, str] | None,
        ) -> Mapping[str, Any]:
            return run_deepseek_chk1_generation(
                prompt=prompt,
                fact_card=fact_card,
                same_sample_minutes=same_sample_minutes,
                repo_root=repo_root,
                cache_dir=cache_dir,
                environment=environment,
                teacher_backend=persistent_teacher_backend,
                generation_provenance_sha256=provenance_sha256,
            )

    else:
        generator_fn = generator
    assert minutes_resolver is not None
    resolver_fn = minutes_resolver
    local_cache_root = (
        Path(model_cache_dir).expanduser().resolve()
        if model_cache_dir is not None
        else destination / "cache" / "local_models"
    )
    # Keep one shared teacher-cache root.  The DeepSeek cache key still binds
    # current provenance; the shared root additionally allows lossless reuse
    # of already-fetched responses after acquisition-policy-only revisions.
    local_cache = local_cache_root

    artifacts_by_index: list[dict[str, Any] | None] = [None] * len(selected)

    def process(index: int, row: Mapping[str, Any]) -> dict[str, Any]:
        return _process_sample(
            row,
            mode=mode,
            output_dir=destination,
            repo_root=root,
            model_cache_dir=local_cache,
            generator=generator_fn,
            minutes_resolver=resolver_fn,
            generation_provenance=generation_provenance,
            model_input_projector=model_input_projector,
            sft_token_auditor=sft_token_auditor,
            preparation_binding_sha256=preparation_binding_sha256,
            environment=environment,
            resume=resume,
        )

    def report_progress(completed: int, row: Mapping[str, Any], artifact: Mapping[str, Any]) -> None:
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "chk1_generation_progress",
                    "mode": mode,
                    "completed": completed,
                    "total": len(selected),
                    "sample_id_sha256": sha256_text(str(row["sample_id"])),
                    "result": (
                        "excluded"
                        if artifact.get("exclusion") is not None
                        else "accepted"
                    ),
                }
            )

    concurrency = (
        int(generation_provenance["deepseek_concurrency"])
        if generator is None
        else 1
    )
    if concurrency == 1:
        for index, row in enumerate(selected):
            artifact = process(index, row)
            artifacts_by_index[index] = artifact
            report_progress(index + 1, row, artifact)
    else:
        completed = 0
        with ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix="chk1-deepseek",
        ) as executor:
            futures = {
                executor.submit(process, index, row): (index, row)
                for index, row in enumerate(selected)
            }
            for future in as_completed(futures):
                index, row = futures[future]
                artifact = future.result()
                artifacts_by_index[index] = artifact
                completed += 1
                report_progress(completed, row, artifact)

    if any(artifact is None for artifact in artifacts_by_index):
        raise GenerationPipelineError("concurrent generation lost a selected sample")
    artifacts = [
        artifact for artifact in artifacts_by_index if artifact is not None
    ]
    sft_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in _SPLITS}
    manifest_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in _SPLITS}
    exclusions: list[dict[str, Any]] = []
    for artifact in artifacts:
        if artifact.get("exclusion") is not None:
            exclusions.append(dict(artifact["exclusion"]))
            continue
        manifest = dict(artifact["manifest_row"])
        split = str(manifest["split"])
        sft_rows[split].append(dict(artifact["sft_row"]))
        manifest_rows[split].append(manifest)
    for split in _SPLITS:
        paired = sorted(
            zip(manifest_rows[split], sft_rows[split], strict=True),
            key=lambda item: str(item[0]["sample_id"]),
        )
        manifest_rows[split] = [item[0] for item in paired]
        sft_rows[split] = [item[1] for item in paired]
    exclusions.sort(key=lambda item: str(item["sample_id"]))

    file_records: dict[str, Any] = {"sft": {}, "manifests": {}}
    for split in _SPLITS:
        sft_path = destination / "sft" / f"{split}.jsonl"
        manifest_path = destination / "manifests" / f"{split}.jsonl"
        _write_immutable(sft_path, _jsonl_text(sft_rows[split]), resume=resume)
        _write_immutable(
            manifest_path,
            _jsonl_text(manifest_rows[split]),
            resume=resume,
        )
        file_records["sft"][split] = _file_record(
            sft_path, output_dir=destination, rows=len(sft_rows[split])
        )
        file_records["manifests"][split] = _file_record(
            manifest_path,
            output_dir=destination,
            rows=len(manifest_rows[split]),
        )

    exclusions_path = destination / "audit" / "exclusions.jsonl"
    _write_immutable(exclusions_path, _jsonl_text(exclusions), resume=resume)
    file_records["exclusions"] = _file_record(
        exclusions_path,
        output_dir=destination,
        rows=len(exclusions),
    )
    selection_path = destination / "selection_manifest.json"
    _write_immutable(selection_path, _pretty_json(selection), resume=resume)
    file_records["selection"] = _file_record(
        selection_path,
        output_dir=destination,
        rows=1,
    )

    cache_entries = [
        {
            "sample_id": str(row["sample_id"]),
            **_file_record(
                _sample_cache_path(
                    destination,
                    str(row["sample_id"]),
                    generation_provenance_sha256=provenance_sha256,
                ),
                output_dir=destination,
                rows=1,
            ),
        }
        for row in selected
    ]
    cache_manifest = _payload_with_hash(
        {
            "schema_version": CACHE_MANIFEST_SCHEMA_VERSION,
            "generation_provenance_sha256": provenance_sha256,
            "preparation_binding_sha256": preparation_binding_sha256,
            "entries": cache_entries,
        }
    )
    cache_manifest_path = destination / "cache" / "cache_manifest.json"
    _write_immutable(
        cache_manifest_path,
        _pretty_json(cache_manifest),
        resume=resume,
    )
    file_records["cache_manifest"] = _file_record(
        cache_manifest_path,
        output_dir=destination,
        rows=1,
    )

    accepted_counts = {split: len(sft_rows[split]) for split in _SPLITS}
    projection_attestations = [
        str(row["generation"]["model_input_projection"]["attestation_sha256"])
        for split in _SPLITS
        for row in manifest_rows[split]
    ]
    projection_binding = {
        "projection_contract_sha256": projection_contract_sha256(),
        "accepted_row_count": len(projection_attestations),
        "row_attestations_sha256": sha256_text(canonical_json(projection_attestations)),
    }
    final_generation_provenance = build_generation_provenance(
        repo_root=root, environment=environment
    )
    if final_generation_provenance != generation_provenance:
        raise GenerationPipelineError(
            "generation code/model/tokenizer provenance changed during execution"
        )
    handoff = _payload_with_hash(
        {
            "schema_version": GENERATION_HANDOFF_SCHEMA_VERSION,
            "status": "complete",
            "mode": mode,
            "selected_count": len(selected),
            "accepted_counts": accepted_counts,
            "accepted_count": sum(accepted_counts.values()),
            "excluded_count": len(exclusions),
            "selection_payload_sha256": selection["payload_sha256"],
            "generation_provenance": generation_provenance,
            "preparation_binding": preparation_binding,
            "model_input_projection": projection_binding,
            "files": file_records,
        }
    )
    _write_immutable(handoff_path, _pretty_json(handoff), resume=resume)
    return handoff_path


read_prepared_rows = load_prepared_rows
select_pilot_samples = select_pilot_rows
select_smoke_samples = select_smoke_rows
run_chk1_generation_pipeline = run_generation_pipeline


__all__ = [
    "CACHE_MANIFEST_SCHEMA_VERSION",
    "EXCLUSION_SCHEMA_VERSION",
    "GENERATION_HANDOFF_SCHEMA_VERSION",
    "GenerationPipelineError",
    "GenerationResultError",
    "PILOT_SAMPLE_COUNT",
    "PreparedDataError",
    "ResumeIntegrityError",
    "SAMPLE_CACHE_SCHEMA_VERSION",
    "SELECTION_SCHEMA_VERSION",
    "SMOKE_SAMPLE_COUNT",
    "SelectionError",
    "load_prepared_rows",
    "read_prepared_rows",
    "run_chk1_generation_pipeline",
    "run_generation_pipeline",
    "select_generation_rows",
    "select_pilot_rows",
    "select_pilot_samples",
    "select_smoke_rows",
    "select_smoke_samples",
]
