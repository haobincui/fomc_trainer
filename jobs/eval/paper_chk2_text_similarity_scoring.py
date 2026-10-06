"""Score and bootstrap the paper-CHK2 in-sample K10 similarity experiment.

The module is deliberately generation-free.  It consumes a completed, sealed
three-model generation matrix and computes answer-only MPNet cosine and
BERTScore-F1 against the synthetic Minutes reference.  Two prespecified views
are retained:

``raw_best_effort``
    Score the recovered final text.  An empty recovered text receives zero.

``delivery_penalized``
    Retain the raw score only when the generation runner attested
    ``delivery_valid=true``; otherwise assign zero.

The primary uncertainty calculation is a split-stratified, meeting-cluster,
shared-index paired percentile bootstrap.  Pooled K10 intervals add a second,
shared paired resample over stochastic replicates.  A split-stratified row
bootstrap is published only as a dependence-ignoring sensitivity analysis. The
data are a training-only release, so every result is explicitly descriptive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Keep the CLI runnable both as ``python -m jobs.eval...`` and by absolute path.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs.eval import eval_checkpoint_generation as semantic_base  # noqa: E402
from jobs.eval import (  # noqa: E402
    score_chk3_native_checkpoint_sweep_semantic as semantic_shared,
)
from open_r1.validator.loo_generation_spec import (  # noqa: E402
    seal_manifest,
    validate_manifest_integrity,
)

ROW_SCHEMA_VERSION = "paper-chk2-text-similarity-score-row-v1"
RESULT_SCHEMA_VERSION = "paper-chk2-text-similarity-bootstrap-results-v1"
MANIFEST_SCHEMA_VERSION = "paper-chk2-text-similarity-score-manifest-v1"
DRAW_SCHEMA_VERSION = "paper-chk2-text-similarity-bootstrap-draw-v1"

MODEL_ORDER = ("chk0", "chk1", "chk2")
SPLIT_ORDER = ("train", "validation", "test")
POLICY_ORDER = ("raw_best_effort", "delivery_penalized")
PRIMARY_POLICY = "raw_best_effort"
METRIC_ORDER = ("mpnet_cosine", "bertscore_f1")
REPLICATE_IDS = tuple(range(10))

EXPECTED_SPLIT_ROWS = {"train": 305, "validation": 42, "test": 44}
EXPECTED_SPLIT_MEETINGS = {"train": 98, "validation": 13, "test": 13}
EXPECTED_SAMPLES = sum(EXPECTED_SPLIT_ROWS.values())
EXPECTED_MEETINGS = sum(EXPECTED_SPLIT_MEETINGS.values())
EXPECTED_ROWS_PER_MODEL = EXPECTED_SAMPLES * len(REPLICATE_IDS)
EXPECTED_TOTAL_GENERATIONS = EXPECTED_ROWS_PER_MODEL * len(MODEL_ORDER)

TEMPERATURE = 0.6
TOP_P = 0.9
BOOTSTRAP_DRAWS = 1_000
BOOTSTRAP_SEED = 20_260_901
SIGN_FLIP_DRAWS = 100_000
CONFIDENCE = 0.95

CONTRASTS = (
    ("chk1_minus_chk0", "chk0", "chk1"),
    ("chk2_minus_chk1", "chk1", "chk2"),
    ("chk2_minus_chk0", "chk0", "chk2"),
)

DEFAULT_SEMANTIC_MANIFEST = ROOT / "configs/main/checkpoint_eval_semantic_models.json"
EXPECTED_SEMANTIC_MANIFEST_SHA256 = (
    "639ca25cf4f5e695b0c6cfeeb47aba75d3e6dfeda1ad87510c23bb4f4a378bf7"
)
EXPECTED_SEMANTIC_MODELS = {
    "bertscore": {
        "repo_id": "FacebookAI/roberta-large",
        "resolved_revision": "722cf37b1afa9454edce342e7895e588b6ff1d59",
        "directory_sha256": "b970a47c99ab5d994a7bcd689ad86b48fb709076c5d309f6a39d5763478eab88",
        "num_layers": 17,
    },
    "embedding_cosine": {
        "repo_id": "sentence-transformers/all-mpnet-base-v2",
        "resolved_revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
        "directory_sha256": "1c8bfc2c3cb29e484b3ac3585c5166de44389c32cbfdc33d1f2b51634e37403f",
    },
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PaperChk2SimilarityError(RuntimeError):
    """A scoring, provenance, or statistical invariant failed closed."""


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
        raise PaperChk2SimilarityError(f"value is not canonical JSON: {exc}") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_regular_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if path.is_symlink() or resolved.is_symlink() or not resolved.is_file():
        raise PaperChk2SimilarityError(f"{label} must be a regular non-symlink file: {path}")
    return resolved


def _read_json(path: Path, *, label: str, sealed: bool = False) -> dict[str, Any]:
    resolved = _require_regular_file(path, label=label)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperChk2SimilarityError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise PaperChk2SimilarityError(f"{label} must be a JSON object")
    if sealed or "integrity" in value:
        try:
            validate_manifest_integrity(value)
        except Exception as exc:
            raise PaperChk2SimilarityError(f"{label} integrity failed: {exc}") from exc
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    resolved = _require_regular_file(path, label=label)
    rows: list[dict[str, Any]] = []
    try:
        for line_number, raw in enumerate(
            resolved.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw.strip():
                raise PaperChk2SimilarityError(
                    f"{label} contains a blank line at {line_number}"
                )
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise PaperChk2SimilarityError(
                    f"{label} line {line_number} is not an object"
                )
            rows.append(value)
    except json.JSONDecodeError as exc:
        raise PaperChk2SimilarityError(f"cannot parse {label}: {exc}") from exc
    return rows


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(_canonical_json(dict(row)) + "\n" for row in rows)


def _csv_text(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field)) for field in fields})
    return stream.getvalue()


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PaperChk2SimilarityError("CSV contains a non-finite float")
        return format(value, ".17g")
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_or_verify_text(path: Path, text: str, *, resume: bool) -> None:
    payload = text.encode("utf-8")
    if path.exists() or path.is_symlink():
        if not resume or path.is_symlink() or not path.is_file():
            raise PaperChk2SimilarityError(f"refusing to overwrite artifact: {path}")
        if path.read_bytes() != payload:
            raise PaperChk2SimilarityError(
                f"resume artifact differs from deterministic recomputation: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise PaperChk2SimilarityError(
                f"artifact appeared during create-only commit: {path}"
            ) from exc
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _file_binding(
    path: Path, *, rows: int | None = None, sealed: bool = False
) -> dict[str, Any]:
    resolved = _require_regular_file(path, label="artifact")
    result: dict[str, Any] = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }
    if rows is not None:
        result["rows"] = rows
    if sealed:
        value = _read_json(resolved, label="sealed artifact", sealed=True)
        result["payload_sha256"] = value["integrity"]["payload_sha256"]
    return result


def _walk_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mappings(child)


def _resolve_declared_path(raw: str, manifest_path: Path) -> list[Path]:
    value = Path(raw).expanduser()
    candidates = [value] if value.is_absolute() else [manifest_path.parent / value, ROOT / value]
    return [candidate.resolve() for candidate in candidates]


def _validate_manifest_file_binding(
    manifest: Mapping[str, Any], manifest_path: Path, actual_path: Path, *, rows: int
) -> None:
    actual = _file_binding(actual_path, rows=rows)
    matches: list[Mapping[str, Any]] = []
    for record in _walk_mappings(manifest):
        raw = record.get("path")
        if not isinstance(raw, str):
            continue
        if actual_path.resolve() in _resolve_declared_path(raw, manifest_path):
            matches.append(record)
    if not matches:
        raise PaperChk2SimilarityError(
            f"evaluation manifest does not bind input file: {actual_path}"
        )
    for record in matches:
        if record.get("sha256") != actual["sha256"]:
            raise PaperChk2SimilarityError(
                f"evaluation-manifest SHA drift for {actual_path}"
            )
        declared_rows = record.get("rows", record.get("row_count"))
        if declared_rows is not None and declared_rows != rows:
            raise PaperChk2SimilarityError(
                f"evaluation-manifest row drift for {actual_path}"
            )


def _numeric_values_for_keys(value: Any, keys: set[str]) -> list[float]:
    result: list[float] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in keys and isinstance(child, (int, float)) and not isinstance(child, bool):
                result.append(float(child))
            result.extend(_numeric_values_for_keys(child, keys))
    elif isinstance(value, list):
        for child in value:
            result.extend(_numeric_values_for_keys(child, keys))
    return result


def _model_orders(value: Any) -> list[tuple[str, ...]]:
    result: list[tuple[str, ...]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"model_order", "models"} and isinstance(child, list) and all(
                isinstance(item, str) for item in child
            ):
                result.append(tuple(child))
            result.extend(_model_orders(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_model_orders(child))
    return result


def _require_numeric_contract(
    manifest: Mapping[str, Any], keys: set[str], expected: float, *, label: str
) -> None:
    values = _numeric_values_for_keys(manifest, keys)
    if not values or any(not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12) for value in values):
        raise PaperChk2SimilarityError(
            f"evaluation manifest {label} must be consistently {expected}; found {values}"
        )


def _validate_evaluation_manifest(
    path: Path,
    *,
    samples_path: Path,
    generation_path: Path,
    sample_rows: int,
    generation_rows: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = _require_regular_file(path, label="evaluation manifest")
    manifest = _read_json(resolved, label="evaluation manifest")
    if not isinstance(manifest.get("schema_version"), str) or not manifest.get(
        "schema_version"
    ):
        raise PaperChk2SimilarityError("evaluation manifest has no schema_version")
    if manifest.get("status") != "complete":
        raise PaperChk2SimilarityError("evaluation manifest status must be complete")
    orders = _model_orders(manifest)
    if not orders or any(order != MODEL_ORDER for order in orders):
        raise PaperChk2SimilarityError(
            f"evaluation model order must be exactly {list(MODEL_ORDER)}; found {orders}"
        )
    _require_numeric_contract(
        manifest, {"replicates", "replicate_count", "k"}, 10, label="K"
    )
    _require_numeric_contract(
        manifest, {"temperature"}, TEMPERATURE, label="temperature"
    )
    _require_numeric_contract(manifest, {"top_p"}, TOP_P, label="top_p")
    _require_numeric_contract(
        manifest,
        {"total_rows", "total_generation_rows", "total_generations"},
        EXPECTED_TOTAL_GENERATIONS,
        label="total generation rows",
    )
    _validate_manifest_file_binding(
        manifest, resolved, samples_path, rows=sample_rows
    )
    _validate_manifest_file_binding(
        manifest, resolved, generation_path, rows=generation_rows
    )
    return manifest, _file_binding(resolved, sealed="integrity" in manifest)


def _first(row: Mapping[str, Any], names: Sequence[str], *, label: str) -> Any:
    found = [(name, row[name]) for name in names if name in row]
    if not found:
        raise PaperChk2SimilarityError(f"missing {label}; accepted fields={list(names)}")
    reference = found[0][1]
    if any(value != reference for _, value in found[1:]):
        raise PaperChk2SimilarityError(f"conflicting aliases for {label}")
    return reference


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PaperChk2SimilarityError(f"{label} must be a non-empty string")
    return value


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise PaperChk2SimilarityError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PaperChk2SimilarityError(f"{label} must be an integer >= {minimum}")
    return value


def _token_ids(value: Any, *, label: str, allow_empty: bool) -> list[int]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise PaperChk2SimilarityError(f"{label} must be a token-ID list")
    result = [_integer(item, label=f"{label} item") for item in value]
    return result


def load_samples(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved = _require_regular_file(path, label="samples")
    raw_rows = _read_jsonl(resolved, label="samples")
    if len(raw_rows) != EXPECTED_SAMPLES:
        raise PaperChk2SimilarityError(
            f"sample count must be {EXPECTED_SAMPLES}, found {len(raw_rows)}"
        )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    split_counts: Counter[str] = Counter()
    meetings_by_split: dict[str, set[str]] = defaultdict(set)
    for line_number, raw in enumerate(raw_rows, start=1):
        sample_id = _nonempty_string(
            _first(raw, ("sample_id",), label="sample_id"),
            label=f"samples[{line_number}].sample_id",
        )
        if sample_id in seen:
            raise PaperChk2SimilarityError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        split = _nonempty_string(
            _first(raw, ("split",), label="split"), label=f"{sample_id}.split"
        )
        if split == "eval":
            split = "validation"
        if split not in SPLIT_ORDER:
            raise PaperChk2SimilarityError(f"{sample_id}: invalid split {split!r}")
        meeting_id = _nonempty_string(
            _first(raw, ("meeting_id", "meeting_date"), label="meeting_id"),
            label=f"{sample_id}.meeting_id",
        )
        prompt = _nonempty_string(
            _first(raw, ("prompt", "source_prompt"), label="prompt"),
            label=f"{sample_id}.prompt",
        )
        reference = _nonempty_string(
            _first(
                raw,
                ("reference", "reference_minutes", "target_answer"),
                label="reference",
            ),
            label=f"{sample_id}.reference",
        )
        prompt_sha = _sha(
            _first(raw, ("prompt_sha256", "source_prompt_sha256"), label="prompt SHA"),
            label=f"{sample_id}.prompt_sha256",
        )
        reference_sha = _sha(
            _first(
                raw,
                ("reference_sha256", "reference_minutes_sha256"),
                label="reference SHA",
            ),
            label=f"{sample_id}.reference_sha256",
        )
        if prompt_sha != _sha256_text(prompt) or reference_sha != _sha256_text(reference):
            raise PaperChk2SimilarityError(f"{sample_id}: prompt/reference hash drift")
        prompt_token_ids = _token_ids(
            _first(raw, ("prompt_token_ids",), label="prompt_token_ids"),
            label=f"{sample_id}.prompt_token_ids",
            allow_empty=False,
        )
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "meeting_id": meeting_id,
                "prompt": prompt,
                "reference": reference,
                "prompt_sha256": prompt_sha,
                "reference_sha256": reference_sha,
                "prompt_token_ids": prompt_token_ids,
                "source_line_number": line_number,
            }
        )
        split_counts[split] += 1
        meetings_by_split[split].add(meeting_id)
    if dict(split_counts) != EXPECTED_SPLIT_ROWS:
        raise PaperChk2SimilarityError(
            f"sample split counts drift: {dict(split_counts)}"
        )
    meeting_counts = {split: len(meetings_by_split[split]) for split in SPLIT_ORDER}
    if meeting_counts != EXPECTED_SPLIT_MEETINGS:
        raise PaperChk2SimilarityError(f"sample meeting counts drift: {meeting_counts}")
    for left_index, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[left_index + 1 :]:
            overlap = meetings_by_split[left] & meetings_by_split[right]
            if overlap:
                raise PaperChk2SimilarityError(
                    f"meeting split overlap between {left}/{right}: {sorted(overlap)[:5]}"
                )
    return rows, _file_binding(resolved, rows=len(rows))


def load_generations(
    path: Path, *, samples: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved = _require_regular_file(path, label="generation rows")
    raw_rows = _read_jsonl(resolved, label="generation rows")
    if len(raw_rows) != EXPECTED_TOTAL_GENERATIONS:
        raise PaperChk2SimilarityError(
            f"generation count must be {EXPECTED_TOTAL_GENERATIONS}, found {len(raw_rows)}"
        )
    sample_by_id = {str(row["sample_id"]): row for row in samples}
    normalized: dict[tuple[str, str, int], dict[str, Any]] = {}
    row_seed_by_pair: dict[tuple[str, int], int] = {}
    replicate_seed_by_id: dict[int, int] = {}
    counts: Counter[str] = Counter()
    for line_number, raw in enumerate(raw_rows, start=1):
        model_id = _nonempty_string(
            _first(raw, ("model_id", "stage_id"), label="model_id"),
            label=f"generation[{line_number}].model_id",
        )
        if model_id not in MODEL_ORDER:
            raise PaperChk2SimilarityError(f"invalid model_id: {model_id}")
        sample_id = _nonempty_string(
            _first(raw, ("sample_id",), label="sample_id"),
            label=f"generation[{line_number}].sample_id",
        )
        if sample_id not in sample_by_id:
            raise PaperChk2SimilarityError(f"unknown generation sample_id: {sample_id}")
        replicate_id = _integer(
            _first(
                raw,
                ("replicate_id", "replicate_index", "k"),
                label="replicate_id",
            ),
            label=f"{model_id}/{sample_id}.replicate_id",
        )
        if replicate_id not in REPLICATE_IDS:
            raise PaperChk2SimilarityError(
                f"{model_id}/{sample_id}: replicate_id outside K10"
            )
        replicate_seed = _integer(
            _first(raw, ("replicate_seed",), label="replicate_seed"),
            label=f"{model_id}/{sample_id}/{replicate_id}.replicate_seed",
        )
        prior_replicate = replicate_seed_by_id.setdefault(replicate_id, replicate_seed)
        if prior_replicate != replicate_seed:
            raise PaperChk2SimilarityError(
                f"replicate_seed drift for replicate {replicate_id}"
            )
        row_seed = _integer(
            _first(raw, ("row_seed", "seed"), label="row_seed"),
            label=f"{model_id}/{sample_id}/{replicate_id}.row_seed",
        )
        prior_row_seed = row_seed_by_pair.setdefault((sample_id, replicate_id), row_seed)
        if prior_row_seed != row_seed:
            raise PaperChk2SimilarityError(
                f"cross-model row_seed drift for {sample_id}/{replicate_id}"
            )
        completion = _first(
            raw, ("completion", "generated_text"), label="completion"
        )
        if not isinstance(completion, str):
            raise PaperChk2SimilarityError("completion must be a string")
        completion_ids = _token_ids(
            _first(
                raw,
                ("completion_token_ids", "output_token_ids", "generated_token_ids"),
                label="completion_token_ids",
            ),
            label=f"{model_id}/{sample_id}/{replicate_id}.completion_token_ids",
            allow_empty=True,
        )
        recovered = _first(
            raw, ("recovered_text", "answer", "final_answer"), label="recovered_text"
        )
        if not isinstance(recovered, str):
            raise PaperChk2SimilarityError("recovered_text must be a string")
        finish_reason = _nonempty_string(
            _first(raw, ("finish_reason",), label="finish_reason"),
            label=f"{model_id}/{sample_id}/{replicate_id}.finish_reason",
        )
        delivery_valid = _first(raw, ("delivery_valid",), label="delivery_valid")
        if not isinstance(delivery_valid, bool):
            raise PaperChk2SimilarityError("delivery_valid must be boolean")
        delivery_failures = _first(
            raw, ("delivery_failures",), label="delivery_failures"
        )
        if not isinstance(delivery_failures, list) or not all(
            isinstance(item, str) and item for item in delivery_failures
        ):
            raise PaperChk2SimilarityError("delivery_failures must be a string list")
        if delivery_valid is (len(delivery_failures) > 0):
            raise PaperChk2SimilarityError(
                f"delivery verdict/failures disagree for {model_id}/{sample_id}/{replicate_id}"
            )
        if raw.get("input_truncated") not in (None, False):
            raise PaperChk2SimilarityError("input truncation is forbidden")
        key = (model_id, sample_id, replicate_id)
        if key in normalized:
            raise PaperChk2SimilarityError(f"duplicate generation tuple: {key}")
        sample = sample_by_id[sample_id]
        normalized[key] = {
            "model_id": model_id,
            "sample_id": sample_id,
            "split": sample["split"],
            "meeting_id": sample["meeting_id"],
            "replicate_id": replicate_id,
            "replicate_seed": replicate_seed,
            "row_seed": row_seed,
            "completion": completion,
            "completion_token_ids": completion_ids,
            "finish_reason": finish_reason,
            # Preserve the post-processor's recovered text byte-for-byte.  Only
            # the empty-text predicate is whitespace-normalized downstream.
            "recovered_text": recovered,
            "delivery_valid": delivery_valid,
            "delivery_failures": list(delivery_failures),
            "source_line_number": line_number,
        }
        counts[model_id] += 1
    expected_keys = {
        (model_id, str(sample["sample_id"]), replicate_id)
        for model_id in MODEL_ORDER
        for sample in samples
        for replicate_id in REPLICATE_IDS
    }
    if set(normalized) != expected_keys:
        missing = sorted(expected_keys - set(normalized))[:5]
        extra = sorted(set(normalized) - expected_keys)[:5]
        raise PaperChk2SimilarityError(
            f"generation tuple coverage drift; missing={missing}, extra={extra}"
        )
    if dict(counts) != {model_id: EXPECTED_ROWS_PER_MODEL for model_id in MODEL_ORDER}:
        raise PaperChk2SimilarityError(f"per-model generation counts drift: {counts}")
    ordered = [
        normalized[(model_id, str(sample["sample_id"]), replicate_id)]
        for model_id in MODEL_ORDER
        for sample in samples
        for replicate_id in REPLICATE_IDS
    ]
    return ordered, _file_binding(resolved, rows=len(ordered))


def _validate_semantic_manifest_offline(path: Path) -> dict[str, Any]:
    resolved = _require_regular_file(path, label="semantic manifest")
    if _sha256_file(resolved) != EXPECTED_SEMANTIC_MANIFEST_SHA256:
        raise PaperChk2SimilarityError("semantic manifest external SHA drift")
    manifest = _read_json(resolved, label="semantic manifest", sealed=True)
    if manifest.get("schema_version") != semantic_base.SEMANTIC_MANIFEST_SCHEMA_VERSION:
        raise PaperChk2SimilarityError("unsupported semantic manifest schema")
    if manifest.get("network_at_scoring_time") is not False:
        raise PaperChk2SimilarityError("semantic manifest must require offline scoring")
    models = manifest.get("models")
    if not isinstance(models, Mapping) or set(models) != {
        "bertscore",
        "embedding_cosine",
    }:
        raise PaperChk2SimilarityError("semantic model inventory drift")
    checked: dict[str, Any] = {}
    for key in ("bertscore", "embedding_cosine"):
        record = models[key]
        if not isinstance(record, Mapping):
            raise PaperChk2SimilarityError(f"semantic model {key} is malformed")
        expected_record = EXPECTED_SEMANTIC_MODELS[key]
        for field, expected_value in expected_record.items():
            if record.get(field) != expected_value:
                raise PaperChk2SimilarityError(
                    f"semantic model {key} binding drift: {field}"
                )
        try:
            model_path = semantic_base._resolve_semantic_model_path(
                record.get("local_path"),
                manifest_path=resolved,
                label=f"semantic model {key}",
            )
            expected = _sha(record.get("directory_sha256"), label=f"{key} directory hash")
            observed = semantic_base._sha256_path(model_path)
        except Exception as exc:
            raise PaperChk2SimilarityError(
                f"cannot validate semantic model {key}: {exc}"
            ) from exc
        if observed != expected:
            raise PaperChk2SimilarityError(f"semantic model directory drift: {key}")
        checked[key] = {
            "repo_id": record.get("repo_id"),
            "resolved_revision": record.get("resolved_revision"),
            "local_path": str(model_path),
            "directory_sha256": observed,
            "num_layers": record.get("num_layers") if key == "bertscore" else None,
        }
    return {
        "manifest": _file_binding(resolved, sealed=True),
        "network_at_scoring_time": False,
        "models": checked,
    }


def _score_contract(
    *,
    sample_binding: Mapping[str, Any],
    generation_binding: Mapping[str, Any],
    evaluation_binding: Mapping[str, Any],
    semantic_binding: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": "paper-chk2-text-similarity-score-contract-v1",
        "models": list(MODEL_ORDER),
        "splits": list(SPLIT_ORDER),
        "policies": list(POLICY_ORDER),
        "metrics": list(METRIC_ORDER),
        "replicate_ids": list(REPLICATE_IDS),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "candidate_text": "recovered_text",
        "reference_text": "reference",
        "empty_raw_policy": "both_metrics_zero",
        "delivery_penalty": "both_metrics_zero_when_delivery_valid_false",
        "samples": dict(sample_binding),
        "generations": dict(generation_binding),
        "evaluation_manifest": dict(evaluation_binding),
        "semantic_manifest": dict(semantic_binding),
    }
    return {**payload, "sha256": _sha256_text(_canonical_json(payload))}


def _metric_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise PaperChk2SimilarityError(f"{label} is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PaperChk2SimilarityError(f"{label} is not numeric") from exc
    if not math.isfinite(result) or result < -1.000001 or result > 1.000001:
        raise PaperChk2SimilarityError(f"{label} is outside [-1,1]")
    return result


def build_row_scores(
    *,
    samples: Sequence[Mapping[str, Any]],
    generations: Sequence[Mapping[str, Any]],
    bert: Any,
    mpnet: Any,
    score_contract_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample_by_id = {str(row["sample_id"]): row for row in samples}
    nonempty = [row for row in generations if str(row["recovered_text"]).strip()]
    candidates = [str(row["recovered_text"]) for row in nonempty]
    references = [str(sample_by_id[str(row["sample_id"])]["reference"]) for row in nonempty]
    semantic_by_key: dict[tuple[str, str, int], dict[str, float]] = {}
    if candidates:
        try:
            bert_values = bert.score(candidates, references)
            mpnet_values = mpnet.score(candidates, references)
        except Exception as exc:
            raise PaperChk2SimilarityError(f"semantic scoring failed: {exc}") from exc
        if not isinstance(bert_values, Mapping) or set(bert_values) != {
            "bertscore_precision",
            "bertscore_recall",
            "bertscore_f1",
        }:
            raise PaperChk2SimilarityError("BERTScore metric inventory drift")
        bert_f1 = list(bert_values["bertscore_f1"])
        mpnet_scores = list(mpnet_values)
        if len(bert_f1) != len(nonempty) or len(mpnet_scores) != len(nonempty):
            raise PaperChk2SimilarityError("semantic vector length drift")
        for row, cosine, f1 in zip(nonempty, mpnet_scores, bert_f1, strict=True):
            key = (str(row["model_id"]), str(row["sample_id"]), int(row["replicate_id"]))
            semantic_by_key[key] = {
                "mpnet_cosine": _metric_number(cosine, label=f"{key} MPNet"),
                "bertscore_f1": _metric_number(f1, label=f"{key} BERTScore-F1"),
            }
    backend_audit = {
        "scored_nonempty_pairs": len(nonempty),
        "empty_pairs_fixed_to_zero": len(generations) - len(nonempty),
        "bertscore": bert.semantic_metadata(),
        "mpnet": mpnet.semantic_metadata(),
    }
    for backend in ("bertscore", "mpnet"):
        metadata = backend_audit[backend]
        if not isinstance(metadata, Mapping):
            raise PaperChk2SimilarityError(f"{backend} metadata missing")
        chunk = metadata.get("chunk_audit")
        if not isinstance(chunk, Mapping) or chunk.get("silent_truncation") is not False:
            raise PaperChk2SimilarityError(
                f"{backend} did not prove zero silent truncation"
            )
    output: list[dict[str, Any]] = []
    for row_number, generation in enumerate(generations, start=1):
        key = (
            str(generation["model_id"]),
            str(generation["sample_id"]),
            int(generation["replicate_id"]),
        )
        sample = sample_by_id[key[1]]
        empty = not bool(str(generation["recovered_text"]).strip())
        raw = semantic_by_key.get(key, {metric: 0.0 for metric in METRIC_ORDER})
        if empty is (key in semantic_by_key):
            raise PaperChk2SimilarityError(f"semantic empty/nonempty mapping drift: {key}")
        penalized = (
            dict(raw)
            if bool(generation["delivery_valid"])
            else {metric: 0.0 for metric in METRIC_ORDER}
        )
        output.append(
            {
                "schema_version": ROW_SCHEMA_VERSION,
                "score_contract_sha256": score_contract_sha256,
                "score_row_number": row_number,
                "model_id": key[0],
                "sample_id": key[1],
                "split": sample["split"],
                "meeting_id": sample["meeting_id"],
                "replicate_id": key[2],
                "replicate_seed": generation["replicate_seed"],
                "row_seed": generation["row_seed"],
                "prompt_sha256": sample["prompt_sha256"],
                "reference_sha256": sample["reference_sha256"],
                "completion_sha256": _sha256_text(str(generation["completion"])),
                "completion_token_ids_sha256": _sha256_text(
                    _canonical_json(generation["completion_token_ids"])
                ),
                "recovered_text_sha256": _sha256_text(str(generation["recovered_text"])),
                "finish_reason": generation["finish_reason"],
                "recovered_text_empty": empty,
                "delivery_valid": generation["delivery_valid"],
                "delivery_failures": list(generation["delivery_failures"]),
                "raw_best_effort": {metric: float(raw[metric]) for metric in METRIC_ORDER},
                "delivery_penalized": {
                    metric: float(penalized[metric]) for metric in METRIC_ORDER
                },
            }
        )
    return output, backend_audit


def validate_row_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    samples: Sequence[Mapping[str, Any]],
    generations: Sequence[Mapping[str, Any]],
    score_contract_sha256: str,
) -> None:
    if len(rows) != EXPECTED_TOTAL_GENERATIONS:
        raise PaperChk2SimilarityError("row-score count drift")
    sample_by_id = {str(row["sample_id"]): row for row in samples}
    generation_by_key = {
        (str(row["model_id"]), str(row["sample_id"]), int(row["replicate_id"])): row
        for row in generations
    }
    seen: set[tuple[str, str, int]] = set()
    for row_number, row in enumerate(rows, start=1):
        if row.get("schema_version") != ROW_SCHEMA_VERSION:
            raise PaperChk2SimilarityError("row-score schema drift")
        if row.get("score_contract_sha256") != score_contract_sha256:
            raise PaperChk2SimilarityError("row-score contract drift")
        key = (str(row.get("model_id")), str(row.get("sample_id")), row.get("replicate_id"))
        if (
            key[0] not in MODEL_ORDER
            or key[1] not in sample_by_id
            or isinstance(key[2], bool)
            or key[2] not in REPLICATE_IDS
            or key in seen
        ):
            raise PaperChk2SimilarityError(f"invalid/duplicate row-score key: {key}")
        seen.add(key)
        generation = generation_by_key.get(key)
        sample = sample_by_id[key[1]]
        if generation is None:
            raise PaperChk2SimilarityError(f"row score has no generation: {key}")
        expected_surface = {
            "score_row_number": row_number,
            "split": sample["split"],
            "meeting_id": sample["meeting_id"],
            "replicate_seed": generation["replicate_seed"],
            "row_seed": generation["row_seed"],
            "prompt_sha256": sample["prompt_sha256"],
            "reference_sha256": sample["reference_sha256"],
            "completion_sha256": _sha256_text(str(generation["completion"])),
            "completion_token_ids_sha256": _sha256_text(
                _canonical_json(generation["completion_token_ids"])
            ),
            "recovered_text_sha256": _sha256_text(str(generation["recovered_text"])),
            "finish_reason": generation["finish_reason"],
            "recovered_text_empty": not bool(str(generation["recovered_text"]).strip()),
            "delivery_valid": generation["delivery_valid"],
            "delivery_failures": generation["delivery_failures"],
        }
        for field, expected in expected_surface.items():
            if row.get(field) != expected:
                raise PaperChk2SimilarityError(f"row-score field drift {field}: {key}")
        for policy in POLICY_ORDER:
            metrics = row.get(policy)
            if not isinstance(metrics, Mapping) or set(metrics) != set(METRIC_ORDER):
                raise PaperChk2SimilarityError(f"row-score metric schema drift: {key}")
            for metric in METRIC_ORDER:
                _metric_number(metrics[metric], label=f"{key}/{policy}/{metric}")
        raw = row["raw_best_effort"]
        penalized = row["delivery_penalized"]
        if row["recovered_text_empty"] and any(float(raw[m]) != 0.0 for m in METRIC_ORDER):
            raise PaperChk2SimilarityError(f"empty raw score is not zero: {key}")
        expected_penalized = raw if row["delivery_valid"] else {m: 0.0 for m in METRIC_ORDER}
        if dict(penalized) != dict(expected_penalized):
            raise PaperChk2SimilarityError(f"delivery penalty drift: {key}")
    if seen != set(generation_by_key):
        raise PaperChk2SimilarityError("row-score tuple coverage is incomplete")


def _percentile(values: np.ndarray) -> tuple[float, float]:
    if values.ndim != 1 or len(values) != BOOTSTRAP_DRAWS or not np.all(
        np.isfinite(values)
    ):
        raise PaperChk2SimilarityError("bootstrap vector is incomplete")
    alpha = 1.0 - CONFIDENCE
    return (
        float(np.quantile(values, alpha / 2.0, method="linear")),
        float(np.quantile(values, 1.0 - alpha / 2.0, method="linear")),
    )


def _plan_digest(metadata: Mapping[str, Any], arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256(_canonical_json(dict(metadata)).encode("utf-8"))
    for array in arrays:
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _bootstrap_plans(
    samples: Sequence[Mapping[str, Any]], *, seed: int = BOOTSTRAP_SEED
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    meetings = {
        split: sorted(
            {str(sample["meeting_id"]) for sample in samples if sample["split"] == split}
        )
        for split in SPLIT_ORDER
    }
    rows = {
        split: [index for index, sample in enumerate(samples) if sample["split"] == split]
        for split in SPLIT_ORDER
    }
    rng_meeting = np.random.default_rng(seed)
    meeting_plan = {
        split: rng_meeting.integers(
            0,
            len(meetings[split]),
            size=(BOOTSTRAP_DRAWS, len(meetings[split])),
            dtype=np.uint16,
        )
        for split in SPLIT_ORDER
    }
    rng_row = np.random.default_rng(seed + 1)
    row_plan = {
        split: rng_row.integers(
            0,
            len(rows[split]),
            size=(BOOTSTRAP_DRAWS, len(rows[split])),
            dtype=np.uint16,
        )
        for split in SPLIT_ORDER
    }
    rng_replicate = np.random.default_rng(seed + 2)
    replicate_plan = rng_replicate.integers(
        0,
        len(REPLICATE_IDS),
        size=(BOOTSTRAP_DRAWS, len(REPLICATE_IDS)),
        dtype=np.uint8,
    )
    meeting_metadata = {
        "schema_version": "paper-chk2-split-stratified-meeting-bootstrap-plan-v1",
        "seed": seed,
        "draws": BOOTSTRAP_DRAWS,
        "split_order": list(SPLIT_ORDER),
        "meeting_ids": meetings,
        "shared_across_models_policies_metrics_and_k": True,
        "numpy_version": np.__version__,
        "bit_generator": type(rng_meeting.bit_generator).__name__,
    }
    row_metadata = {
        "schema_version": "paper-chk2-split-stratified-row-bootstrap-plan-v1",
        "seed": seed + 1,
        "draws": BOOTSTRAP_DRAWS,
        "split_order": list(SPLIT_ORDER),
        "sample_ids": {
            split: [str(samples[index]["sample_id"]) for index in rows[split]]
            for split in SPLIT_ORDER
        },
        "shared_across_models_policies_metrics_and_k": True,
        "dependence_limitation": "ignores_within_meeting_row_dependence",
        "numpy_version": np.__version__,
        "bit_generator": type(rng_row.bit_generator).__name__,
    }
    replicate_metadata = {
        "schema_version": "paper-chk2-paired-replicate-bootstrap-plan-v1",
        "seed": seed + 2,
        "draws": BOOTSTRAP_DRAWS,
        "replicate_ids": list(REPLICATE_IDS),
        "replicates_resampled_per_draw": len(REPLICATE_IDS),
        "shared_across_models_policies_metrics_splits_and_bootstrap_schemes": True,
        "used_for_pooled_k10_only": True,
        "numpy_version": np.__version__,
        "bit_generator": type(rng_replicate.bit_generator).__name__,
    }
    meeting_metadata["sha256"] = _plan_digest(
        meeting_metadata, [meeting_plan[split] for split in SPLIT_ORDER]
    )
    row_metadata["sha256"] = _plan_digest(
        row_metadata, [row_plan[split] for split in SPLIT_ORDER]
    )
    replicate_metadata["sha256"] = _plan_digest(
        replicate_metadata, [replicate_plan]
    )
    return (
        meeting_plan,
        row_plan,
        replicate_plan,
        meeting_metadata,
        row_metadata,
        replicate_metadata,
    )


def _score_arrays(
    row_scores: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    sample_index = {str(row["sample_id"]): index for index, row in enumerate(samples)}
    arrays = {
        policy: {
            model_id: {
                metric: np.full(
                    (len(samples), len(REPLICATE_IDS)), np.nan, dtype=np.float64
                )
                for metric in METRIC_ORDER
            }
            for model_id in MODEL_ORDER
        }
        for policy in POLICY_ORDER
    }
    for row in row_scores:
        i = sample_index[str(row["sample_id"])]
        k = int(row["replicate_id"])
        for policy in POLICY_ORDER:
            for metric in METRIC_ORDER:
                arrays[policy][str(row["model_id"])][metric][i, k] = float(
                    row[policy][metric]
                )
    for policy in POLICY_ORDER:
        for model_id in MODEL_ORDER:
            for metric in METRIC_ORDER:
                if not np.all(np.isfinite(arrays[policy][model_id][metric])):
                    raise PaperChk2SimilarityError("score matrix is incomplete")
    return arrays


def _meeting_layout(
    samples: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, list[int]], dict[str, list[int]]]:
    by_meeting: dict[str, list[int]] = defaultdict(list)
    split_by_meeting: dict[str, str] = {}
    for index, sample in enumerate(samples):
        meeting = str(sample["meeting_id"])
        split = str(sample["split"])
        prior = split_by_meeting.setdefault(meeting, split)
        if prior != split:
            raise PaperChk2SimilarityError("meeting crosses splits")
        by_meeting[meeting].append(index)
    meeting_ids = sorted(by_meeting)
    positions_by_split = {
        split: [index for index, meeting in enumerate(meeting_ids) if split_by_meeting[meeting] == split]
        for split in SPLIT_ORDER
    }
    return meeting_ids, dict(by_meeting), positions_by_split


def _meeting_matrices(
    arrays: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    *,
    meeting_ids: Sequence[str],
    by_meeting: Mapping[str, Sequence[int]],
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    result = {
        policy: {
            model_id: {
                metric: np.empty(
                    (len(meeting_ids), len(REPLICATE_IDS)), dtype=np.float64
                )
                for metric in METRIC_ORDER
            }
            for model_id in MODEL_ORDER
        }
        for policy in POLICY_ORDER
    }
    for policy in POLICY_ORDER:
        for model_id in MODEL_ORDER:
            for metric in METRIC_ORDER:
                source = arrays[policy][model_id][metric]
                for position, meeting in enumerate(meeting_ids):
                    result[policy][model_id][metric][position] = source[
                        list(by_meeting[meeting])
                    ].mean(axis=0)
    return result


def _draw_matrix(
    matrix: np.ndarray,
    *,
    plan: Mapping[str, np.ndarray],
    positions_by_split: Mapping[str, Sequence[int]],
) -> np.ndarray:
    output = np.zeros(
        (BOOTSTRAP_DRAWS, len(REPLICATE_IDS)), dtype=np.float64
    )
    denominator = sum(len(positions_by_split[split]) for split in SPLIT_ORDER)
    for split in SPLIT_ORDER:
        positions = np.asarray(positions_by_split[split], dtype=np.int64)
        selected = positions[plan[split].astype(np.int64)]
        output += matrix[selected].mean(axis=1) * (len(positions) / denominator)
    return output


def _draw_rows(
    matrix: np.ndarray,
    *,
    plan: Mapping[str, np.ndarray],
    sample_positions_by_split: Mapping[str, Sequence[int]],
) -> np.ndarray:
    output = np.zeros(
        (BOOTSTRAP_DRAWS, len(REPLICATE_IDS)), dtype=np.float64
    )
    denominator = sum(len(sample_positions_by_split[split]) for split in SPLIT_ORDER)
    for split in SPLIT_ORDER:
        positions = np.asarray(sample_positions_by_split[split], dtype=np.int64)
        selected = positions[plan[split].astype(np.int64)]
        output += matrix[selected].mean(axis=1) * (len(positions) / denominator)
    return output


def _pooled_replicate_draws(
    draws_by_replicate: np.ndarray, *, replicate_plan: np.ndarray
) -> np.ndarray:
    expected_shape = (BOOTSTRAP_DRAWS, len(REPLICATE_IDS))
    if (
        draws_by_replicate.shape != expected_shape
        or replicate_plan.shape != expected_shape
        or not np.all(np.isfinite(draws_by_replicate))
        or np.any(replicate_plan < 0)
        or np.any(replicate_plan >= len(REPLICATE_IDS))
    ):
        raise PaperChk2SimilarityError("hierarchical replicate bootstrap shape drift")
    selected = np.take_along_axis(
        draws_by_replicate, replicate_plan.astype(np.int64), axis=1
    )
    return selected.mean(axis=1)


def _sign_flip_p_value(
    differences: Sequence[float], *, contrast_id: str, metric: str
) -> tuple[float, int, int]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) != EXPECTED_MEETINGS or not np.all(
        np.isfinite(values)
    ):
        raise PaperChk2SimilarityError("sign-flip vector is invalid")
    seed_payload = f"{BOOTSTRAP_SEED + 3}:{contrast_id}:{metric}"
    seed = int.from_bytes(hashlib.sha256(seed_payload.encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    observed = abs(float(values.mean()))
    extreme = 0
    remaining = SIGN_FLIP_DRAWS
    while remaining:
        count = min(10_000, remaining)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(values)))
        null = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(null >= observed - 1e-15))
        remaining -= count
    return (extreme + 1.0) / (SIGN_FLIP_DRAWS + 1.0), SIGN_FLIP_DRAWS, seed


def _holm(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(float(value) for value in values), key=lambda item: item[1])
    output = [0.0] * len(values)
    running = 0.0
    total = len(values)
    for rank, (original, value) in enumerate(indexed):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise PaperChk2SimilarityError("invalid p-value for Holm correction")
        running = max(running, min(1.0, (total - rank) * value))
        output[original] = running
    return output


def build_statistics(
    *,
    row_scores: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    arrays = _score_arrays(row_scores, samples)
    meeting_ids, by_meeting, meeting_positions_by_split = _meeting_layout(samples)
    if len(meeting_ids) != EXPECTED_MEETINGS:
        raise PaperChk2SimilarityError("meeting denominator drift")
    sample_positions_by_split = {
        split: [index for index, sample in enumerate(samples) if sample["split"] == split]
        for split in SPLIT_ORDER
    }
    meeting_values = _meeting_matrices(
        arrays, meeting_ids=meeting_ids, by_meeting=by_meeting
    )
    (
        meeting_plan,
        row_plan,
        replicate_plan,
        meeting_plan_meta,
        row_plan_meta,
        replicate_plan_meta,
    ) = _bootstrap_plans(samples)

    schemes: dict[
        str, dict[str, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]]
    ] = {
        "meeting_cluster_primary": {
            policy: {model: {} for model in MODEL_ORDER} for policy in POLICY_ORDER
        },
        "row_stratified_sensitivity": {
            policy: {model: {} for model in MODEL_ORDER} for policy in POLICY_ORDER
        },
    }
    for policy in POLICY_ORDER:
        for model in MODEL_ORDER:
            for metric in METRIC_ORDER:
                meeting_matrix = meeting_values[policy][model][metric]
                meeting_draws = _draw_matrix(
                    meeting_matrix,
                    plan=meeting_plan,
                    positions_by_split=meeting_positions_by_split,
                )
                row_matrix = arrays[policy][model][metric]
                row_draws = _draw_rows(
                    row_matrix,
                    plan=row_plan,
                    sample_positions_by_split=sample_positions_by_split,
                )
                schemes["meeting_cluster_primary"][policy][model][metric] = (
                    meeting_matrix.mean(axis=0),
                    meeting_draws,
                )
                schemes["row_stratified_sensitivity"][policy][model][metric] = (
                    row_matrix.mean(axis=0),
                    row_draws,
                )

    model_summary: list[dict[str, Any]] = []
    per_k_results: list[dict[str, Any]] = []
    for scheme_id in ("meeting_cluster_primary", "row_stratified_sensitivity"):
        for policy in POLICY_ORDER:
            for model in MODEL_ORDER:
                for metric in METRIC_ORDER:
                    point_k, draws_k = schemes[scheme_id][policy][model][metric]
                    pooled_draws = _pooled_replicate_draws(
                        draws_k, replicate_plan=replicate_plan
                    )
                    lower, upper = _percentile(pooled_draws)
                    model_summary.append(
                        {
                            "bootstrap_scheme": scheme_id,
                            "scoring_policy": policy,
                            "model_id": model,
                            "metric": metric,
                            "estimate": float(point_k.mean()),
                            "ci_lower": lower,
                            "ci_upper": upper,
                            "confidence": CONFIDENCE,
                            "bootstrap_draws": BOOTSTRAP_DRAWS,
                            "samples": EXPECTED_SAMPLES,
                            "meetings": EXPECTED_MEETINGS,
                            "replicates": len(REPLICATE_IDS),
                        }
                    )
                    for replicate_id in REPLICATE_IDS:
                        k_lower, k_upper = _percentile(draws_k[:, replicate_id])
                        per_k_results.append(
                            {
                                "bootstrap_scheme": scheme_id,
                                "scoring_policy": policy,
                                "model_id": model,
                                "metric": metric,
                                "replicate_id": replicate_id,
                                "estimate": float(point_k[replicate_id]),
                                "ci_lower": k_lower,
                                "ci_upper": k_upper,
                                "confidence": CONFIDENCE,
                                "bootstrap_draws": BOOTSTRAP_DRAWS,
                                "samples": EXPECTED_SAMPLES,
                                "meetings": EXPECTED_MEETINGS,
                            }
                        )

    split_results: list[dict[str, Any]] = []
    for split in SPLIT_ORDER:
        positions = np.asarray(meeting_positions_by_split[split], dtype=np.int64)
        selected = positions[meeting_plan[split].astype(np.int64)]
        for policy in POLICY_ORDER:
            for model in MODEL_ORDER:
                for metric in METRIC_ORDER:
                    matrix = meeting_values[policy][model][metric]
                    split_draws_k = matrix[selected].mean(axis=1)
                    draws = _pooled_replicate_draws(
                        split_draws_k, replicate_plan=replicate_plan
                    )
                    lower, upper = _percentile(draws)
                    split_results.append(
                        {
                            "split": split,
                            "bootstrap_scheme": "meeting_cluster_primary",
                            "scoring_policy": policy,
                            "model_id": model,
                            "metric": metric,
                            "estimate": float(matrix[positions].mean()),
                            "ci_lower": lower,
                            "ci_upper": upper,
                            "confidence": CONFIDENCE,
                            "bootstrap_draws": BOOTSTRAP_DRAWS,
                            "samples": EXPECTED_SPLIT_ROWS[split],
                            "meetings": EXPECTED_SPLIT_MEETINGS[split],
                            "replicates": len(REPLICATE_IDS),
                        }
                    )

    pairwise: list[dict[str, Any]] = []
    inferential_rows: list[dict[str, Any]] = []
    for scheme_id in ("meeting_cluster_primary", "row_stratified_sensitivity"):
        for policy in POLICY_ORDER:
            for contrast_id, before, after in CONTRASTS:
                for metric in METRIC_ORDER:
                    before_point, before_draws = schemes[scheme_id][policy][before][metric]
                    after_point, after_draws = schemes[scheme_id][policy][after][metric]
                    delta_points = after_point - before_point
                    delta_draws = after_draws - before_draws
                    for replicate_id in REPLICATE_IDS:
                        lower, upper = _percentile(delta_draws[:, replicate_id])
                        pairwise.append(
                            {
                                "bootstrap_scheme": scheme_id,
                                "scoring_policy": policy,
                                "aggregate_scope": "replicate_k",
                                "replicate_id": replicate_id,
                                "contrast_id": contrast_id,
                                "before_model": before,
                                "after_model": after,
                                "metric": metric,
                                "estimate": float(delta_points[replicate_id]),
                                "ci_lower": lower,
                                "ci_upper": upper,
                                "confidence": CONFIDENCE,
                                "bootstrap_draws": BOOTSTRAP_DRAWS,
                                "p_value_method": None,
                                "p_value": None,
                                "holm_adjusted_p": None,
                                "holm_family_size": None,
                            }
                        )
                    pooled_draws = _pooled_replicate_draws(
                        delta_draws, replicate_plan=replicate_plan
                    )
                    lower, upper = _percentile(pooled_draws)
                    record = {
                        "bootstrap_scheme": scheme_id,
                        "scoring_policy": policy,
                        "aggregate_scope": "pooled_k10",
                        "replicate_id": None,
                        "contrast_id": contrast_id,
                        "before_model": before,
                        "after_model": after,
                        "metric": metric,
                        "estimate": float(delta_points.mean()),
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "confidence": CONFIDENCE,
                        "bootstrap_draws": BOOTSTRAP_DRAWS,
                        "p_value_method": None,
                        "p_value": None,
                        "holm_adjusted_p": None,
                        "holm_family_size": None,
                    }
                    if scheme_id == "meeting_cluster_primary" and policy == PRIMARY_POLICY:
                        meeting_delta = (
                            meeting_values[policy][after][metric].mean(axis=1)
                            - meeting_values[policy][before][metric].mean(axis=1)
                        )
                        p_value, assignments, p_seed = _sign_flip_p_value(
                            meeting_delta, contrast_id=contrast_id, metric=metric
                        )
                        record.update(
                            {
                                "p_value_method": (
                                    "two_sided_monte_carlo_meeting_sign_flip_plus_one"
                                ),
                                "p_value": p_value,
                                "sign_flip_assignments": assignments,
                                "sign_flip_seed": p_seed,
                                "holm_family_size": 6,
                            }
                        )
                        inferential_rows.append(record)
                    pairwise.append(record)
    if len(inferential_rows) != 6:
        raise PaperChk2SimilarityError("primary Holm family is not exactly six tests")
    adjusted = _holm([float(row["p_value"]) for row in inferential_rows])
    for row, value in zip(inferential_rows, adjusted, strict=True):
        row["holm_adjusted_p"] = value

    draw_rows: list[dict[str, Any]] = []
    for draw_id in range(BOOTSTRAP_DRAWS):
        views: dict[str, Any] = {}
        for scheme_id in ("meeting_cluster_primary", "row_stratified_sensitivity"):
            views[scheme_id] = {
                policy: {
                    model: {
                        metric: {
                            "per_k": [
                                float(value)
                                for value in schemes[scheme_id][policy][model][metric][1][
                                    draw_id
                                ]
                            ],
                            "pooled_k10": float(
                                schemes[scheme_id][policy][model][metric][1][draw_id][
                                    replicate_plan[draw_id].astype(np.int64)
                                ].mean()
                            ),
                        }
                        for metric in METRIC_ORDER
                    }
                    for model in MODEL_ORDER
                }
                for policy in POLICY_ORDER
            }
        draw_rows.append(
            {
                "schema_version": DRAW_SCHEMA_VERSION,
                "draw_id": draw_id,
                "meeting_plan_sha256": meeting_plan_meta["sha256"],
                "row_plan_sha256": row_plan_meta["sha256"],
                "replicate_plan_sha256": replicate_plan_meta["sha256"],
                "replicate_resample_indices": [
                    int(value) for value in replicate_plan[draw_id]
                ],
                "views": views,
            }
        )

    diagnostics = {
        model: {
            "generations": sum(row["model_id"] == model for row in row_scores),
            "recovered_text_empty": sum(
                row["model_id"] == model and row["recovered_text_empty"]
                for row in row_scores
            ),
            "delivery_invalid": sum(
                row["model_id"] == model and not row["delivery_valid"]
                for row in row_scores
            ),
            "delivery_failure_counts": dict(
                sorted(
                    Counter(
                        failure
                        for row in row_scores
                        if row["model_id"] == model
                        for failure in row["delivery_failures"]
                    ).items()
                )
            ),
        }
        for model in MODEL_ORDER
    }
    results = seal_manifest(
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "complete",
            "evaluation_role": "in_sample_training_release_reconstruction_diagnostic",
            "models": list(MODEL_ORDER),
            "splits": list(SPLIT_ORDER),
            "policies": list(POLICY_ORDER),
            "primary_policy": PRIMARY_POLICY,
            "metrics": list(METRIC_ORDER),
            "coverage": {
                "samples": EXPECTED_SAMPLES,
                "meetings": EXPECTED_MEETINGS,
                "replicates": len(REPLICATE_IDS),
                "generation_rows": EXPECTED_TOTAL_GENERATIONS,
                "split_rows": EXPECTED_SPLIT_ROWS,
                "split_meetings": EXPECTED_SPLIT_MEETINGS,
            },
            "bootstrap_contract": {
                "draws": BOOTSTRAP_DRAWS,
                "confidence": CONFIDENCE,
                "interval": "two_sided_percentile_numpy_linear",
                "primary": meeting_plan_meta,
                "sensitivity": row_plan_meta,
                "replicate_resampling": replicate_plan_meta,
                "point_aggregation": (
                    "within_meeting_rows_mean_then_k_mean_then_meeting_equal_mean"
                ),
                "pooled_interval_aggregation": (
                    "shared_paired_meeting_or_row_resample_then_shared_paired_"
                    "replicate_resample"
                ),
                "paired_shared_indices": True,
                "split_stratified": True,
            },
            "multiple_testing": {
                "primary_family": (
                    "raw_best_effort pooled K10; three model contrasts x two metrics"
                ),
                "tests": 6,
                "method": "Holm",
                "delivery_penalized_significance_authorized": False,
            },
            "model_summary": model_summary,
            "per_k_results": per_k_results,
            "split_results": split_results,
            "pairwise_contrasts": pairwise,
            "delivery_diagnostics": diagnostics,
            "limitations": [
                "All 391 samples come from the training-only paper-CHK2 release.",
                "The merged train/validation/test result is not a held-out generalization estimate.",
                "The reference is the synthetic teacher rewrite after </think>, not official Minutes.",
                "Cosine similarity and BERTScore-F1 measure reference resemblance, not factual correctness.",
                "The row bootstrap ignores within-meeting dependence and is sensitivity-only.",
                "The ten stochastic replicates are repeated generations, not ten independent datasets.",
            ],
        }
    )
    return (
        results,
        draw_rows,
        model_summary,
        per_k_results,
        split_results,
        pairwise,
    )


MODEL_SUMMARY_FIELDS = (
    "bootstrap_scheme",
    "scoring_policy",
    "model_id",
    "metric",
    "estimate",
    "ci_lower",
    "ci_upper",
    "confidence",
    "bootstrap_draws",
    "samples",
    "meetings",
    "replicates",
)
PER_K_FIELDS = (
    "bootstrap_scheme",
    "scoring_policy",
    "model_id",
    "metric",
    "replicate_id",
    "estimate",
    "ci_lower",
    "ci_upper",
    "confidence",
    "bootstrap_draws",
    "samples",
    "meetings",
)
SPLIT_FIELDS = (
    "split",
    "bootstrap_scheme",
    "scoring_policy",
    "model_id",
    "metric",
    "estimate",
    "ci_lower",
    "ci_upper",
    "confidence",
    "bootstrap_draws",
    "samples",
    "meetings",
    "replicates",
)
CONTRAST_FIELDS = (
    "bootstrap_scheme",
    "scoring_policy",
    "aggregate_scope",
    "replicate_id",
    "contrast_id",
    "before_model",
    "after_model",
    "metric",
    "estimate",
    "ci_lower",
    "ci_upper",
    "confidence",
    "bootstrap_draws",
    "p_value_method",
    "p_value",
    "holm_adjusted_p",
    "holm_family_size",
    "sign_flip_assignments",
    "sign_flip_seed",
)


def _load_inputs(
    *,
    generation_rows: Path,
    samples: Path,
    evaluation_manifest: Path,
    semantic_manifest: Path,
) -> dict[str, Any]:
    sample_rows, sample_binding = load_samples(samples)
    generation_values, generation_binding = load_generations(
        generation_rows, samples=sample_rows
    )
    _manifest, evaluation_binding = _validate_evaluation_manifest(
        evaluation_manifest,
        samples_path=samples.expanduser().resolve(),
        generation_path=generation_rows.expanduser().resolve(),
        sample_rows=len(sample_rows),
        generation_rows=len(generation_values),
    )
    semantic_provenance = _validate_semantic_manifest_offline(semantic_manifest)
    contract = _score_contract(
        sample_binding=sample_binding,
        generation_binding=generation_binding,
        evaluation_binding=evaluation_binding,
        semantic_binding=semantic_provenance["manifest"],
    )
    return {
        "samples": sample_rows,
        "generations": generation_values,
        "sample_binding": sample_binding,
        "generation_binding": generation_binding,
        "evaluation_binding": evaluation_binding,
        "semantic_provenance": semantic_provenance,
        "score_contract": contract,
    }


def score_command(
    *,
    generation_rows: Path,
    samples: Path,
    evaluation_manifest: Path,
    semantic_manifest: Path,
    output_dir: Path,
    semantic_device: str,
    semantic_batch_size: int,
    resume: bool,
) -> dict[str, Any]:
    if semantic_batch_size < 1:
        raise PaperChk2SimilarityError("semantic batch size must be positive")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    inputs = _load_inputs(
        generation_rows=generation_rows,
        samples=samples,
        evaluation_manifest=evaluation_manifest,
        semantic_manifest=semantic_manifest,
    )
    output = output_dir.expanduser().resolve()
    if output.is_symlink():
        raise PaperChk2SimilarityError("output directory must not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    row_path = output / "row_scores.jsonl"
    if row_path.exists() or row_path.is_symlink():
        if not resume:
            raise PaperChk2SimilarityError(
                "row_scores already exists; pass --resume for validate-only reuse"
            )
        rows = _read_jsonl(row_path, label="row scores")
        validate_row_scores(
            rows,
            samples=inputs["samples"],
            generations=inputs["generations"],
            score_contract_sha256=inputs["score_contract"]["sha256"],
        )
        return {
            "status": "already_scored",
            "row_scores": _file_binding(row_path, rows=len(rows)),
            "score_contract_sha256": inputs["score_contract"]["sha256"],
        }
    try:
        bert, mpnet, loaded_provenance = semantic_shared.load_formal_semantic_backends(
            semantic_manifest_path=semantic_manifest,
            batch_size=semantic_batch_size,
            device=semantic_device,
        )
    except Exception as exc:
        raise PaperChk2SimilarityError(
            f"cannot load pinned semantic backends: {exc}"
        ) from exc
    rows, audit = build_row_scores(
        samples=inputs["samples"],
        generations=inputs["generations"],
        bert=bert,
        mpnet=mpnet,
        score_contract_sha256=inputs["score_contract"]["sha256"],
    )
    validate_row_scores(
        rows,
        samples=inputs["samples"],
        generations=inputs["generations"],
        score_contract_sha256=inputs["score_contract"]["sha256"],
    )
    _write_or_verify_text(row_path, _jsonl_text(rows), resume=False)
    return {
        "status": "scored",
        "row_scores": _file_binding(row_path, rows=len(rows)),
        "score_contract_sha256": inputs["score_contract"]["sha256"],
        "semantic_backend_audit": audit,
        "semantic_provenance": loaded_provenance,
    }


def _artifact_texts(
    statistics: tuple[
        dict[str, Any],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]
) -> dict[str, tuple[str, int | None, bool]]:
    results, draws, model_summary, per_k, splits, contrasts = statistics
    return {
        "bootstrap_draws.jsonl": (_jsonl_text(draws), len(draws), False),
        "model_summary.csv": (_csv_text(model_summary, MODEL_SUMMARY_FIELDS), len(model_summary), False),
        "per_k_results.csv": (_csv_text(per_k, PER_K_FIELDS), len(per_k), False),
        "split_results.csv": (_csv_text(splits, SPLIT_FIELDS), len(splits), False),
        "pairwise_contrasts.csv": (
            _csv_text(contrasts, CONTRAST_FIELDS),
            len(contrasts),
            False,
        ),
        "results.json": (json.dumps(results, indent=2, sort_keys=True) + "\n", None, True),
    }


def _manifest_payload(
    *,
    output_dir: Path,
    inputs: Mapping[str, Any],
    row_scores: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    artifacts = {
        "row_scores": _file_binding(
            output_dir / "row_scores.jsonl", rows=len(row_scores)
        )
    }
    for name in (
        "bootstrap_draws.jsonl",
        "model_summary.csv",
        "per_k_results.csv",
        "split_results.csv",
        "pairwise_contrasts.csv",
        "results.json",
    ):
        path = output_dir / name
        if name.endswith(".jsonl"):
            rows = BOOTSTRAP_DRAWS
        elif name == "model_summary.csv":
            rows = 24
        elif name == "per_k_results.csv":
            rows = 240
        elif name == "split_results.csv":
            rows = 36
        elif name == "pairwise_contrasts.csv":
            rows = 264
        else:
            rows = None
        artifacts[name] = _file_binding(
            path, rows=rows, sealed=name == "results.json"
        )
    return seal_manifest(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": "complete",
            "operation": (
                "offline_pinned_semantic_scoring_and_split_stratified_paired_bootstrap"
            ),
            "evaluation_role": "in_sample_training_release_reconstruction_diagnostic",
            "training_release_evaluation_eligible": False,
            "network_at_scoring_time": False,
            "model_order": list(MODEL_ORDER),
            "split_order": list(SPLIT_ORDER),
            "policy_order": list(POLICY_ORDER),
            "metric_order": list(METRIC_ORDER),
            "score_contract": dict(inputs["score_contract"]),
            "inputs": {
                "samples": dict(inputs["sample_binding"]),
                "generations": dict(inputs["generation_binding"]),
                "evaluation_manifest": dict(inputs["evaluation_binding"]),
                "semantic_manifest": dict(inputs["semantic_provenance"]["manifest"]),
            },
            "semantic_models": dict(inputs["semantic_provenance"]),
            "execution": {
                "bootstrap_device": "cpu",
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "sign_flip_draws": SIGN_FLIP_DRAWS,
                "python_executable": sys.executable,
                "numpy_version": np.__version__,
            },
            "coverage": {
                "samples": EXPECTED_SAMPLES,
                "meetings": EXPECTED_MEETINGS,
                "replicates": len(REPLICATE_IDS),
                "rows_per_model": EXPECTED_ROWS_PER_MODEL,
                "row_scores": len(row_scores),
                "split_rows": EXPECTED_SPLIT_ROWS,
                "split_meetings": EXPECTED_SPLIT_MEETINGS,
            },
            "artifacts": artifacts,
            "implementation": _file_binding(Path(__file__).resolve()),
        }
    )


def bootstrap_command(
    *,
    generation_rows: Path,
    samples: Path,
    evaluation_manifest: Path,
    semantic_manifest: Path,
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    inputs = _load_inputs(
        generation_rows=generation_rows,
        samples=samples,
        evaluation_manifest=evaluation_manifest,
        semantic_manifest=semantic_manifest,
    )
    output = output_dir.expanduser().resolve()
    if output.is_symlink() or not output.is_dir():
        raise PaperChk2SimilarityError("score output directory does not exist")
    row_path = output / "row_scores.jsonl"
    row_scores = _read_jsonl(row_path, label="row scores")
    validate_row_scores(
        row_scores,
        samples=inputs["samples"],
        generations=inputs["generations"],
        score_contract_sha256=inputs["score_contract"]["sha256"],
    )
    manifest_path = output / "score_manifest.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        if not resume:
            raise PaperChk2SimilarityError(
                "score manifest already exists; pass --resume to validate it"
            )
        return validate_command(output_dir=output)
    statistics = build_statistics(row_scores=row_scores, samples=inputs["samples"])
    for name, (text, _rows, _sealed) in _artifact_texts(statistics).items():
        _write_or_verify_text(output / name, text, resume=resume)
    manifest = _manifest_payload(output_dir=output, inputs=inputs, row_scores=row_scores)
    _write_or_verify_text(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n", resume=False
    )
    return manifest


def _binding_path(binding: Mapping[str, Any], *, label: str) -> Path:
    raw = binding.get("path")
    if not isinstance(raw, str):
        raise PaperChk2SimilarityError(f"{label} has no path")
    path = _require_regular_file(Path(raw), label=label)
    observed = _file_binding(path, sealed="payload_sha256" in binding)
    for field in ("sha256", "bytes", "payload_sha256"):
        if field in binding and binding.get(field) != observed.get(field):
            raise PaperChk2SimilarityError(f"{label} binding drift: {field}")
    return path


def validate_command(*, output_dir: Path) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    if output.is_symlink() or not output.is_dir():
        raise PaperChk2SimilarityError("score output directory is missing")
    manifest_path = output / "score_manifest.json"
    manifest = _read_json(manifest_path, label="score manifest", sealed=True)
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("model_order") != list(MODEL_ORDER)
        or manifest.get("split_order") != list(SPLIT_ORDER)
        or manifest.get("policy_order") != list(POLICY_ORDER)
        or manifest.get("metric_order") != list(METRIC_ORDER)
        or manifest.get("training_release_evaluation_eligible") is not False
        or manifest.get("network_at_scoring_time") is not False
    ):
        raise PaperChk2SimilarityError("score manifest contract drift")
    inputs_record = manifest.get("inputs")
    artifacts = manifest.get("artifacts")
    if not isinstance(inputs_record, Mapping) or not isinstance(artifacts, Mapping):
        raise PaperChk2SimilarityError("score manifest inputs/artifacts missing")
    samples_path = _binding_path(inputs_record["samples"], label="samples")
    generations_path = _binding_path(inputs_record["generations"], label="generations")
    evaluation_path = _binding_path(
        inputs_record["evaluation_manifest"], label="evaluation manifest"
    )
    semantic_path = _binding_path(
        inputs_record["semantic_manifest"], label="semantic manifest"
    )
    inputs = _load_inputs(
        generation_rows=generations_path,
        samples=samples_path,
        evaluation_manifest=evaluation_path,
        semantic_manifest=semantic_path,
    )
    if manifest.get("score_contract") != inputs["score_contract"]:
        raise PaperChk2SimilarityError("score manifest input contract drift")
    for label, binding in artifacts.items():
        if not isinstance(binding, Mapping):
            raise PaperChk2SimilarityError(f"artifact binding malformed: {label}")
        _binding_path(binding, label=f"artifact {label}")
    row_scores = _read_jsonl(output / "row_scores.jsonl", label="row scores")
    validate_row_scores(
        row_scores,
        samples=inputs["samples"],
        generations=inputs["generations"],
        score_contract_sha256=inputs["score_contract"]["sha256"],
    )
    statistics = build_statistics(row_scores=row_scores, samples=inputs["samples"])
    expected_texts = _artifact_texts(statistics)
    for name, (text, _rows, _sealed) in expected_texts.items():
        path = output / name
        if path.read_text(encoding="utf-8") != text:
            raise PaperChk2SimilarityError(
                f"artifact differs from deterministic recomputation: {name}"
            )
    expected_manifest = _manifest_payload(
        output_dir=output, inputs=inputs, row_scores=row_scores
    )
    if manifest != expected_manifest:
        raise PaperChk2SimilarityError("score manifest differs from reconstruction")
    return {
        "status": "complete_validated",
        "score_manifest": _file_binding(manifest_path, sealed=True),
        "row_scores": len(row_scores),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--generation-rows", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--evaluation-manifest", required=True, type=Path)
    parser.add_argument(
        "--semantic-manifest", type=Path, default=DEFAULT_SEMANTIC_MANIFEST
    )
    parser.add_argument("--output-dir", required=True, type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score", help="Compute the two semantic metrics")
    _common_arguments(score)
    score.add_argument("--semantic-device", default="cuda:0")
    score.add_argument("--semantic-batch-size", type=int, default=8)
    score.add_argument("--resume", action="store_true")
    bootstrap = subparsers.add_parser(
        "bootstrap", help="Compute fixed B=1000 statistics and seal the bundle"
    )
    _common_arguments(bootstrap)
    bootstrap.add_argument("--resume", action="store_true")
    validate = subparsers.add_parser(
        "validate", help="Deep-replay a completed score bundle"
    )
    validate.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "score":
            result = score_command(
                generation_rows=args.generation_rows,
                samples=args.samples,
                evaluation_manifest=args.evaluation_manifest,
                semantic_manifest=args.semantic_manifest,
                output_dir=args.output_dir,
                semantic_device=args.semantic_device,
                semantic_batch_size=args.semantic_batch_size,
                resume=args.resume,
            )
        elif args.command == "bootstrap":
            result = bootstrap_command(
                generation_rows=args.generation_rows,
                samples=args.samples,
                evaluation_manifest=args.evaluation_manifest,
                semantic_manifest=args.semantic_manifest,
                output_dir=args.output_dir,
                resume=args.resume,
            )
        else:
            result = validate_command(output_dir=args.output_dir)
    except PaperChk2SimilarityError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(_canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
