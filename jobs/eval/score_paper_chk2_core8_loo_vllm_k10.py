"""Score and analyse the paper-chk2 Core8 leave-one-out K=10 panel.

The module is deliberately scoped to the fresh paper-chk2/cp50 run.  It never
reads the historical chk3 semantic rows.  All 21,760 generated rows are scored
against the frozen synthetic Minutes reference; generation gates are retained
as diagnostics and never filter or zero-penalise the primary estimand (except
that a truly empty answer has a deterministic semantic score of zero).

The primary estimand is ``score(Full) - score(intervention)``.  Ten paired
decoding replicates are averaged within each meeting and the 128 meetings are
then equally weighted.  Uncertainty uses one shared hierarchical bootstrap
index plan across all 8 x 2 x 2 topic/arm/metric cells.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / (
    "output/evaluation/retrain_v2/"
    "paper_chk2_cp50_core8_loo_vllm_k10_n128_1993_2008_"
    "t06_p095_b10000_v1_20260901"
)
DEFAULT_SEMANTIC_MANIFEST = ROOT / "configs/main/checkpoint_eval_semantic_models.json"
DEFAULT_SEMANTIC_MANIFEST_SHA256 = (
    "639ca25cf4f5e695b0c6cfeeb47aba75d3e6dfeda1ad87510c23bb4f4a378bf7"
)

EXPECTED_MEETINGS = 128
EXPECTED_VARIANTS = 17
EXPECTED_REPLICATES = 10
EXPECTED_PROMPTS = EXPECTED_MEETINGS * EXPECTED_VARIANTS
EXPECTED_ROWS = EXPECTED_PROMPTS * EXPECTED_REPLICATES
EXPECTED_MEETING_CELLS = EXPECTED_MEETINGS * 8 * 2
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_260_824
CONFIDENCE = 0.95
DEFAULT_WORKERS = 6

TOPICS = (
    "Consumer-Price-Index-(CPI)",
    "GDP-Growth",
    "Government-Purchases",
    "Housing-Starts",
    "Industrial-Production",
    "Labour-Market",
    "Money-Supply",
    "Unemployment-Rate",
)
ARMS = ("exact_deletion", "token_matched_neutral")
METRICS = ("mpnet_cosine", "bertscore_f1")

K10_K9_POINT_DRIFT_THRESHOLD = 0.005
K10_K9_CI_ENDPOINT_DRIFT_THRESHOLD = 0.010
LORO_MAX_DRIFT_THRESHOLD = 0.010
DECODING_VARIANCE_SHARE_THRESHOLD = 0.20

GENERATION_ROW_SCHEMA = "paper-chk2-core8-loo-vllm-k10-generation-row-v1"
CONSOLIDATED_SCHEMA = "paper-chk2-core8-loo-consolidated-generation-row-v1"
METRIC_ROW_SCHEMA = "paper-chk2-core8-loo-semantic-metric-row-v1"
SCORE_ROW_SCHEMA = "paper-chk2-core8-loo-semantic-score-row-v1"
MEETING_CELL_SCHEMA = "paper-chk2-core8-loo-meeting-cell-delta-v1"
FAILURE_ROW_SCHEMA = "paper-chk2-core8-loo-generation-diagnostic-failure-v1"
BOOTSTRAP_DRAW_SCHEMA = "paper-chk2-core8-loo-bootstrap-draw-v1"
STATISTICS_SCHEMA = "paper-chk2-core8-loo-statistics-v1"
SCORE_MANIFEST_SCHEMA = "paper-chk2-core8-loo-score-manifest-v1"


class PaperChk2Core8ScoreError(RuntimeError):
    """A frozen input, score, matrix, or statistical invariant failed closed."""


@dataclass(frozen=True)
class BootstrapPlan:
    meeting_indices: np.ndarray
    prefix_replicate_indices: dict[int, np.ndarray]
    sha256: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CellComputation:
    key: tuple[str, str, str]
    row: dict[str, Any]
    nested_draws: np.ndarray


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise PaperChk2Core8ScoreError(f"required regular file is missing: {resolved}")
    result: dict[str, Any] = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha_file(resolved),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _read_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise PaperChk2Core8ScoreError(f"missing JSON file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperChk2Core8ScoreError(f"invalid JSON file {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise PaperChk2Core8ScoreError(f"JSON root must be an object: {resolved}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise PaperChk2Core8ScoreError(f"missing JSONL file: {resolved}")
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PaperChk2Core8ScoreError(
                    f"invalid JSONL at {resolved}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise PaperChk2Core8ScoreError(
                    f"JSONL row must be an object at {resolved}:{line_number}"
                )
            rows.append(value)
    return rows


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(_canonical(dict(row)) + "\n" for row in rows)


def _write_or_verify_text(path: Path, text: str, *, resume: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        if not resume or not path.is_file() or path.is_symlink():
            raise PaperChk2Core8ScoreError(f"refusing to overwrite existing output: {path}")
        if path.read_text(encoding="utf-8") != text:
            raise PaperChk2Core8ScoreError(f"existing output differs from recomputation: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Mapping[str, Any], *, resume: bool) -> None:
    _write_or_verify_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", resume=resume)


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.pop("manifest_sha256", None)
    return {**body, "manifest_sha256": _sha_text(_canonical(body))}


def _stable_created_at(path: Path, *, resume: bool) -> str:
    if resume and path.is_file() and not path.is_symlink():
        value = _read_json(path).get("created_at_utc")
        if isinstance(value, str) and value:
            return value
        raise PaperChk2Core8ScoreError(f"existing artifact lacks stable timestamp: {path}")
    return _utc_now()


def _validate_seal(value: Mapping[str, Any], *, label: str) -> None:
    expected = value.get("manifest_sha256")
    body = dict(value)
    body.pop("manifest_sha256", None)
    if not isinstance(expected, str) or expected != _sha_text(_canonical(body)):
        raise PaperChk2Core8ScoreError(f"{label} manifest seal mismatch")


def _normalise_topic(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in {"", "none", "null", "full"} else text


def _expected_variant(variant_rank: int) -> tuple[str, str | None]:
    if variant_rank == 0:
        return "full", None
    if not 1 <= variant_rank < EXPECTED_VARIANTS:
        raise PaperChk2Core8ScoreError(f"variant rank outside 0..16: {variant_rank}")
    topic = TOPICS[(variant_rank - 1) // 2]
    arm = ARMS[(variant_rank - 1) % 2]
    return arm, topic


def _resolve_binding_path(value: Any, *, base: Path, label: str) -> Path:
    if isinstance(value, Mapping):
        raw = value.get("path")
    else:
        raw = value
    if not isinstance(raw, str) or not raw:
        raise PaperChk2Core8ScoreError(f"{label} path binding is absent")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    else:
        path = path.resolve()
    if isinstance(value, Mapping):
        if value.get("sha256") is not None and value["sha256"] != _sha_file(path):
            raise PaperChk2Core8ScoreError(f"{label} SHA binding mismatch")
        if value.get("bytes") is not None and int(value["bytes"]) != path.stat().st_size:
            raise PaperChk2Core8ScoreError(f"{label} byte binding mismatch")
    return path


def _load_preparation(run_root: Path) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    try:
        from jobs.eval import paper_chk2_core8_loo_common as common

        loaded = common.load_prepared_ledger(output_root=run_root)
    except (ImportError, AttributeError) as exc:  # pragma: no cover - staged integration only
        raise PaperChk2Core8ScoreError(f"paper-chk2 preparation loader unavailable: {exc}") from exc
    if (
        not isinstance(loaded, tuple)
        or len(loaded) != 2
        or not isinstance(loaded[0], Mapping)
        or not isinstance(loaded[1], (list, tuple))
    ):
        raise PaperChk2Core8ScoreError("prepared ledger loader returned an invalid contract")
    manifest = dict(loaded[0])
    ledger = tuple(dict(row) for row in loaded[1])
    if len(ledger) != EXPECTED_PROMPTS:
        raise PaperChk2Core8ScoreError("prepared ledger is not the exact 128x17 prompt matrix")
    return manifest, ledger


def _generation_paths(run_root: Path) -> tuple[dict[str, Any], list[Path]]:
    validation_path = run_root / "generation/validation.json"
    validation = _read_json(validation_path)
    combined = validation.get("combined_generation_rows")
    if combined is None:
        artifacts = validation.get("artifacts")
        if isinstance(artifacts, Mapping):
            combined = artifacts.get("combined_generation_rows") or artifacts.get("generation_rows")
    if combined is not None:
        return validation, [
            _resolve_binding_path(combined, base=validation_path.parent, label="combined generation rows")
        ]
    paths: list[Path] = []
    for shard in range(2):
        manifest_path = run_root / f"generation/shard-{shard}/manifest.json"
        manifest = _read_json(manifest_path)
        candidate: Any = manifest.get("canonical_generations")
        if candidate is None and isinstance(manifest.get("artifacts"), Mapping):
            candidate = manifest["artifacts"].get("canonical_generations")
        if candidate is None:
            candidate = run_root / f"generation/shard-{shard}/generations.canonical.jsonl"
        paths.append(
            _resolve_binding_path(candidate, base=manifest_path.parent, label=f"shard-{shard} canonical rows")
        )
    return validation, paths


def _gate_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = row.get("generation_metrics")
    if not isinstance(metrics, Mapping):
        metrics = row.get("gate") if isinstance(row.get("gate"), Mapping) else {}
    aliases = {
        "structure_delivery": ("structure_delivery", "delivery", "delivery_valid"),
        "numeric_fidelity": ("numeric_fidelity", "numeric_valid"),
        "date_fidelity": ("date_fidelity", "date_valid"),
        "degeneration_free": ("degeneration_free", "nondegenerate"),
    }
    projected: dict[str, bool] = {}
    for target, names in aliases.items():
        observed: Any = None
        for name in names:
            if name in metrics:
                observed = metrics[name]
                break
            if name in row:
                observed = row[name]
                break
        if isinstance(observed, Mapping):
            observed = observed.get("pass", observed.get("valid"))
        projected[target] = bool(observed) if isinstance(observed, bool) else False
    explicit = row.get("preregistered_core_valid")
    if not isinstance(explicit, bool) and isinstance(row.get("gate"), Mapping):
        explicit = row["gate"].get("preregistered_core_valid")
    computed = all(projected.values())
    if isinstance(explicit, bool) and explicit != computed:
        raise PaperChk2Core8ScoreError("generation core-valid flag conflicts with component gates")
    raw_failures = row.get("preregistered_core_failures")
    if raw_failures is None:
        raw_failures = row.get("failure_codes")
    if raw_failures is None and isinstance(row.get("gate"), Mapping):
        raw_failures = row["gate"].get("failure_codes")
    if not isinstance(raw_failures, list):
        raw_failures = []
    failures = sorted({str(value) for value in raw_failures if str(value)})
    if not computed and not failures:
        failures = [f"{name}_failed" for name, passed in projected.items() if not passed]
    return {
        **projected,
        "preregistered_core_valid": computed,
        "failure_codes": failures,
        "used_to_filter_primary": False,
        "used_to_zero_penalise_primary": False,
    }


def _validate_generation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    ledger: Sequence[Mapping[str, Any]],
    evaluation_manifest_sha256: str,
    replicate_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    if len(rows) != EXPECTED_ROWS:
        raise PaperChk2Core8ScoreError(f"generation coverage must be {EXPECTED_ROWS}, observed {len(rows)}")
    if (
        len(replicate_seeds) != EXPECTED_REPLICATES
        or len(set(replicate_seeds)) != EXPECTED_REPLICATES
        or any(isinstance(value, bool) or not isinstance(value, int) for value in replicate_seeds)
    ):
        raise PaperChk2Core8ScoreError("prepared replicate-seed contract drift")
    ledger_by_id = {str(row["sample_id"]): row for row in ledger}
    if len(ledger_by_id) != EXPECTED_PROMPTS:
        raise PaperChk2Core8ScoreError("prepared sample IDs are not unique")
    output: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    seen_keys: set[str] = set()
    paired_seeds: dict[tuple[int, int], tuple[int, int]] = {}
    for source in rows:
        row = dict(source)
        if row.get("schema_version") != GENERATION_ROW_SCHEMA:
            raise PaperChk2Core8ScoreError("generation row schema drift")
        index = int(row.get("absolute_case_index", -1))
        meeting_rank = int(row.get("meeting_rank", -1))
        variant_rank = int(row.get("variant_rank", -1))
        replicate_id = int(row.get("replicate_id", -1))
        expected_index = (meeting_rank * EXPECTED_VARIANTS + variant_rank) * EXPECTED_REPLICATES + replicate_id
        if index != expected_index or index in seen_indices:
            raise PaperChk2Core8ScoreError(f"generation absolute index drift/duplicate: {index}")
        seen_indices.add(index)
        generation_key = _canonical(row.get("generation_key"))
        if generation_key in seen_keys:
            raise PaperChk2Core8ScoreError("duplicate generation key")
        seen_keys.add(generation_key)
        if not (0 <= meeting_rank < EXPECTED_MEETINGS and 0 <= replicate_id < EXPECTED_REPLICATES):
            raise PaperChk2Core8ScoreError(f"generation coordinate outside frozen matrix: {index}")
        expected_arm, expected_topic = _expected_variant(variant_rank)
        observed_arm = str(row.get("arm", ""))
        observed_topic = _normalise_topic(row.get("intervention_topic"))
        if observed_arm != expected_arm or observed_topic != expected_topic:
            raise PaperChk2Core8ScoreError(f"variant mapping drift at generation index {index}")
        sample_id = str(row.get("sample_id", ""))
        prepared = ledger_by_id.get(sample_id)
        if prepared is None:
            raise PaperChk2Core8ScoreError(f"generation sample not in frozen ledger: {sample_id}")
        for field in ("meeting_id", "meeting_rank", "variant_rank", "arm"):
            if str(row.get(field)) != str(prepared.get(field)):
                raise PaperChk2Core8ScoreError(f"generation/preparation {field} drift: {sample_id}")
        if _normalise_topic(prepared.get("intervention_topic")) != expected_topic:
            raise PaperChk2Core8ScoreError(f"prepared topic drift: {sample_id}")
        hash_pairs = (
            ("prompt_sha256", "prompt_sha256"),
            ("prompt_token_ids_sha256", "prompt_token_ids_sha256"),
            ("reference_minutes_sha256", "reference_minutes_sha256"),
            ("source_analysis_sha256", "source_analysis_sha256"),
        )
        for generated_field, prepared_field in hash_pairs:
            expected_hash = prepared.get(prepared_field)
            if expected_hash is not None and row.get(generated_field) != expected_hash:
                raise PaperChk2Core8ScoreError(f"{generated_field} drift: {sample_id}")
        if int(row.get("prompt_token_count", -1)) != int(prepared.get("prompt_token_count", -2)):
            raise PaperChk2Core8ScoreError(f"prompt token-count drift: {sample_id}")
        reference = str(row.get("reference_minutes", ""))
        answer = str(row.get("answer", ""))
        completion = str(row.get("completion", ""))
        source_analysis = str(row.get("source_analysis", ""))
        generated_token_ids = row.get("generated_token_ids")
        if not isinstance(generated_token_ids, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in generated_token_ids
        ):
            raise PaperChk2Core8ScoreError(f"generated token inventory drift: {index}")
        if row.get("reference_minutes_sha256") != _sha_text(reference):
            raise PaperChk2Core8ScoreError(f"reference text hash drift: {sample_id}")
        if row.get("source_analysis_sha256") != _sha_text(source_analysis):
            raise PaperChk2Core8ScoreError(f"source-analysis text hash drift: {sample_id}")
        if row.get("answer_sha256") != _sha_text(answer):
            raise PaperChk2Core8ScoreError(f"answer text hash drift: {index}")
        if row.get("completion_sha256") != _sha_text(completion):
            raise PaperChk2Core8ScoreError(f"completion text hash drift: {index}")
        if row.get("generated_text") != completion:
            raise PaperChk2Core8ScoreError(f"generated-text/completion drift: {index}")
        if row.get("generated_token_ids_sha256") != _sha_text(_canonical(generated_token_ids)):
            raise PaperChk2Core8ScoreError(f"generated-token hash drift: {index}")
        if row.get("evaluation_manifest_sha256") != evaluation_manifest_sha256:
            raise PaperChk2Core8ScoreError(f"evaluation-manifest binding drift: {index}")
        if row.get("model_id") != "paper_chk2" or row.get("model_label") != "paper-chk2-cp50-lora-over-chk1-cp200":
            raise PaperChk2Core8ScoreError(f"paper-chk2 model identity drift: {index}")
        row_seed = int(row.get("row_seed", -1))
        replicate_seed = int(row.get("replicate_seed", -1))
        if replicate_seed != int(replicate_seeds[replicate_id]):
            raise PaperChk2Core8ScoreError(f"replicate-seed contract drift: {index}")
        if int(row.get("shard_id", -1)) != (meeting_rank * EXPECTED_REPLICATES + replicate_id) % 2:
            raise PaperChk2Core8ScoreError(f"paired-block shard assignment drift: {index}")
        pair = (row_seed, replicate_seed)
        prior = paired_seeds.setdefault((meeting_rank, replicate_id), pair)
        if prior != pair:
            raise PaperChk2Core8ScoreError(f"paired CRN seed drift: {meeting_rank}/{replicate_id}")
        projected = {
            "schema_version": CONSOLIDATED_SCHEMA,
            "absolute_case_index": index,
            "generation_key": row.get("generation_key"),
            "sample_id": sample_id,
            "meeting_id": str(row["meeting_id"]),
            "meeting_rank": meeting_rank,
            "variant_rank": variant_rank,
            "arm": observed_arm,
            "intervention_topic": expected_topic,
            "replicate_id": replicate_id,
            "replicate_seed": replicate_seed,
            "row_seed": row_seed,
            "shard_id": int(row.get("shard_id", -1)),
            "prompt_sha256": row["prompt_sha256"],
            "prompt_token_ids_sha256": row["prompt_token_ids_sha256"],
            "reference_minutes": reference,
            "reference_minutes_sha256": row["reference_minutes_sha256"],
            "source_analysis_sha256": row["source_analysis_sha256"],
            "completion_sha256": row["completion_sha256"],
            "answer": answer,
            "answer_sha256": row["answer_sha256"],
            "finish_reason": row.get("finish_reason"),
            "generation_diagnostics": _gate_projection(row),
            "evaluation_manifest_sha256": evaluation_manifest_sha256,
            "model_id": "paper_chk2",
            "model_label": "paper-chk2-cp50-lora-over-chk1-cp200",
        }
        projected["row_sha256"] = _sha_text(_canonical(projected))
        output.append(projected)
    if seen_indices != set(range(EXPECTED_ROWS)) or len(paired_seeds) != EXPECTED_MEETINGS * EXPECTED_REPLICATES:
        raise PaperChk2Core8ScoreError("generation 128x17x10 coordinate closure failed")
    output.sort(key=lambda value: int(value["absolute_case_index"]))
    return output


def consolidate_generation(run_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    preparation_manifest, ledger = _load_preparation(run_root)
    generation_design = preparation_manifest.get("generation_design")
    replicate_seeds = generation_design.get("replicate_seeds") if isinstance(generation_design, Mapping) else None
    if not isinstance(replicate_seeds, list):
        raise PaperChk2Core8ScoreError("preparation manifest lacks the replicate-seed contract")
    evaluation_path = run_root / "preparation/evaluation_manifest.json"
    evaluation_sha = _sha_file(evaluation_path)
    validation, paths = _generation_paths(run_root)
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_jsonl(path))
    consolidated = _validate_generation_rows(
        rows,
        ledger=ledger,
        evaluation_manifest_sha256=evaluation_sha,
        replicate_seeds=replicate_seeds,
    )
    output_path = run_root / "score/generation_rows.jsonl"
    _write_or_verify_text(output_path, _jsonl_text(consolidated), resume=resume)
    receipt_path = run_root / "score/consolidation_receipt.json"
    receipt = _seal(
        {
            "schema_version": "paper-chk2-core8-loo-consolidation-receipt-v1",
            "status": "complete",
            "created_at_utc": _stable_created_at(receipt_path, resume=resume),
            "inputs": {
                "preparation_manifest": _binding(evaluation_path),
                "generation_validation": _binding(run_root / "generation/validation.json"),
                "generation_sources": [_binding(path) for path in paths],
            },
            "coverage": {
                "rows": EXPECTED_ROWS,
                "prompts": EXPECTED_PROMPTS,
                "meetings": EXPECTED_MEETINGS,
                "replicates": EXPECTED_REPLICATES,
                "validation_status": validation.get("status"),
            },
            "artifact": _binding(output_path, rows=EXPECTED_ROWS),
        }
    )
    _write_json(receipt_path, receipt, resume=resume)
    return receipt


def _metric_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise PaperChk2Core8ScoreError(f"{label} is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PaperChk2Core8ScoreError(f"{label} is not numeric") from exc
    if not math.isfinite(result) or not -1.000001 <= result <= 1.000001:
        raise PaperChk2Core8ScoreError(f"{label} lies outside [-1,1]")
    return result


def _validate_semantic_manifest(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if _sha_file(resolved) != DEFAULT_SEMANTIC_MANIFEST_SHA256:
        raise PaperChk2Core8ScoreError("pinned semantic manifest SHA drift")
    value = _read_json(resolved)
    return {"binding": _binding(resolved), "value": value}


def _backend_audit(backend: Any, *, label: str, scored_rows: int) -> dict[str, Any]:
    metadata = backend.semantic_metadata() if hasattr(backend, "semantic_metadata") else None
    chunk = metadata.get("chunk_audit") if isinstance(metadata, Mapping) else None
    if not isinstance(chunk, Mapping) or chunk.get("silent_truncation") is not False:
        raise PaperChk2Core8ScoreError(f"{label} did not prove zero silent truncation")
    observed = chunk.get("document_count")
    if observed is not None and int(observed) != scored_rows:
        raise PaperChk2Core8ScoreError(f"{label} semantic document-count drift")
    return dict(metadata)


def _score_metric_rows(
    rows: Sequence[Mapping[str, Any]], *, metric: str, backend: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if metric not in METRICS:
        raise PaperChk2Core8ScoreError(f"unknown semantic metric: {metric}")
    nonempty = [row for row in rows if str(row["answer"]).strip()]
    candidates = [str(row["answer"]) for row in nonempty]
    references = [str(row["reference_minutes"]) for row in nonempty]
    values_by_index: dict[int, float] = {}
    if candidates:
        try:
            raw = backend.score(candidates, references)
        except Exception as exc:
            raise PaperChk2Core8ScoreError(f"{metric} inference failed: {exc}") from exc
        vector = raw.get("bertscore_f1") if metric == "bertscore_f1" and isinstance(raw, Mapping) else raw
        if not isinstance(vector, (list, tuple, np.ndarray)) or len(vector) != len(nonempty):
            raise PaperChk2Core8ScoreError(f"{metric} vector length/inventory drift")
        for source, value in zip(nonempty, vector, strict=True):
            values_by_index[int(source["absolute_case_index"])] = _metric_number(
                value, label=f"{metric}/{source['absolute_case_index']}"
            )
    output = []
    for source in rows:
        index = int(source["absolute_case_index"])
        value = values_by_index.get(index, 0.0)
        if bool(str(source["answer"]).strip()) != (index in values_by_index):
            raise PaperChk2Core8ScoreError(f"empty/nonempty semantic mapping drift: {index}")
        projected = {
            "schema_version": METRIC_ROW_SCHEMA,
            "metric": metric,
            "absolute_case_index": index,
            "generation_key": source["generation_key"],
            "sample_id": source["sample_id"],
            "answer_sha256": source["answer_sha256"],
            "reference_minutes_sha256": source["reference_minutes_sha256"],
            "value": value,
            "empty_answer_fixed_to_zero": not bool(str(source["answer"]).strip()),
        }
        projected["row_sha256"] = _sha_text(_canonical(projected))
        output.append(projected)
    audit = _backend_audit(backend, label=metric, scored_rows=len(nonempty))
    return output, audit


def score_metric(
    run_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    metric: str,
    semantic_manifest: Path = DEFAULT_SEMANTIC_MANIFEST,
    semantic_device: str = "cuda:0",
    semantic_batch_size: int = 16,
    resume: bool = False,
) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    if metric not in METRICS:
        raise PaperChk2Core8ScoreError(f"metric must be one of {METRICS}")
    metric_dir = run_root / f"score/metrics/{metric}"
    existing_manifest = metric_dir / "manifest.json"
    existing_rows = metric_dir / "rows.jsonl"
    if existing_manifest.exists() or existing_rows.exists():
        if not resume or not (existing_manifest.is_file() and existing_rows.is_file()):
            raise PaperChk2Core8ScoreError(f"partial/existing {metric} output requires --resume and a complete bundle")
        _load_metric_rows(run_root, metric)
        return _read_json(existing_manifest)
    _validate_semantic_manifest(semantic_manifest)
    generation_path = run_root / "score/generation_rows.jsonl"
    rows = _read_jsonl(generation_path)
    if len(rows) != EXPECTED_ROWS:
        raise PaperChk2Core8ScoreError("consolidated generation coverage drift")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from jobs.eval import score_chk3_native_checkpoint_sweep_semantic as semantic_shared

        bert, mpnet, provenance = semantic_shared.load_formal_semantic_backends(
            semantic_manifest_path=semantic_manifest,
            batch_size=semantic_batch_size,
            device=semantic_device,
        )
    except Exception as exc:
        raise PaperChk2Core8ScoreError(f"cannot load pinned semantic backends: {exc}") from exc
    backend = mpnet if metric == "mpnet_cosine" else bert
    metric_rows, audit = _score_metric_rows(rows, metric=metric, backend=backend)
    row_path = metric_dir / "rows.jsonl"
    _write_or_verify_text(row_path, _jsonl_text(metric_rows), resume=resume)
    manifest = _seal(
        {
            "schema_version": "paper-chk2-core8-loo-semantic-metric-manifest-v1",
            "status": "complete",
            "created_at_utc": _utc_now(),
            "metric": metric,
            "inputs": {
                "generation_rows": _binding(generation_path, rows=EXPECTED_ROWS),
                "semantic_manifest": _binding(semantic_manifest),
            },
            "coverage": {
                "rows": EXPECTED_ROWS,
                "nonempty_scored": sum(bool(str(row["answer"]).strip()) for row in rows),
                "empty_fixed_to_zero": sum(not bool(str(row["answer"]).strip()) for row in rows),
                "generation_gate_filtered": 0,
            },
            "semantic_provenance": provenance,
            "backend_audit": audit,
            "artifact": _binding(row_path, rows=EXPECTED_ROWS),
        }
    )
    _write_json(metric_dir / "manifest.json", manifest, resume=resume)
    return manifest


def _load_metric_rows(run_root: Path, metric: str) -> list[dict[str, Any]]:
    metric_dir = run_root / f"score/metrics/{metric}"
    manifest = _read_json(metric_dir / "manifest.json")
    _validate_seal(manifest, label=f"{metric} semantic")
    if manifest.get("status") != "complete" or manifest.get("metric") != metric:
        raise PaperChk2Core8ScoreError(f"{metric} metric manifest contract drift")
    rows = _read_jsonl(metric_dir / "rows.jsonl")
    if len(rows) != EXPECTED_ROWS:
        raise PaperChk2Core8ScoreError(f"{metric} semantic coverage drift")
    binding = manifest.get("artifact")
    if not isinstance(binding, Mapping):
        raise PaperChk2Core8ScoreError(f"{metric} metric artifact binding missing")
    if binding.get("sha256") != _sha_file(metric_dir / "rows.jsonl") or int(binding.get("rows", -1)) != EXPECTED_ROWS:
        raise PaperChk2Core8ScoreError(f"{metric} metric artifact binding drift")
    seen: set[int] = set()
    for row in rows:
        index = int(row.get("absolute_case_index", -1))
        if (
            row.get("schema_version") != METRIC_ROW_SCHEMA
            or row.get("metric") != metric
            or index in seen
            or not 0 <= index < EXPECTED_ROWS
        ):
            raise PaperChk2Core8ScoreError(f"{metric} semantic row identity drift")
        seal = row.get("row_sha256")
        body = dict(row)
        body.pop("row_sha256", None)
        if seal != _sha_text(_canonical(body)):
            raise PaperChk2Core8ScoreError(f"{metric} semantic row seal drift: {index}")
        _metric_number(row.get("value"), label=f"{metric}/{index}")
        seen.add(index)
    if seen != set(range(EXPECTED_ROWS)):
        raise PaperChk2Core8ScoreError(f"{metric} semantic tuple closure failed")
    return sorted(rows, key=lambda value: int(value["absolute_case_index"]))


def combine_semantic_scores(
    run_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False
) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    generations = _read_jsonl(run_root / "score/generation_rows.jsonl")
    if len(generations) != EXPECTED_ROWS:
        raise PaperChk2Core8ScoreError("consolidated generation coverage drift")
    metric_rows = {metric: _load_metric_rows(run_root, metric) for metric in METRICS}
    output: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for position, generation in enumerate(generations):
        if int(generation.get("absolute_case_index", -1)) != position:
            raise PaperChk2Core8ScoreError("consolidated generation order drift")
        values: dict[str, float] = {}
        for metric in METRICS:
            semantic = metric_rows[metric][position]
            for field in ("generation_key", "sample_id", "answer_sha256", "reference_minutes_sha256"):
                if semantic.get(field) != generation.get(field):
                    raise PaperChk2Core8ScoreError(f"{metric} semantic/generation {field} drift at {position}")
            values[metric] = float(semantic["value"])
        projected = {
            "schema_version": SCORE_ROW_SCHEMA,
            "score_row_number": position + 1,
            "absolute_case_index": position,
            "generation_key": generation["generation_key"],
            "sample_id": generation["sample_id"],
            "meeting_id": generation["meeting_id"],
            "meeting_rank": generation["meeting_rank"],
            "variant_rank": generation["variant_rank"],
            "arm": generation["arm"],
            "intervention_topic": generation["intervention_topic"],
            "replicate_id": generation["replicate_id"],
            "replicate_seed": generation["replicate_seed"],
            "row_seed": generation["row_seed"],
            "answer_sha256": generation["answer_sha256"],
            "reference_minutes_sha256": generation["reference_minutes_sha256"],
            "finish_reason": generation["finish_reason"],
            "answer_empty": not bool(str(generation["answer"]).strip()),
            "raw_semantic_scores": values,
            "generation_diagnostics": generation["generation_diagnostics"],
            "primary_inclusion": "included_regardless_of_generation_diagnostics",
        }
        projected["row_sha256"] = _sha_text(_canonical(projected))
        output.append(projected)
        diagnostics = generation["generation_diagnostics"]
        if not diagnostics["preregistered_core_valid"]:
            failure = {
                "schema_version": FAILURE_ROW_SCHEMA,
                "absolute_case_index": position,
                "sample_id": generation["sample_id"],
                "meeting_id": generation["meeting_id"],
                "variant_rank": generation["variant_rank"],
                "replicate_id": generation["replicate_id"],
                "failure_codes": diagnostics["failure_codes"],
                "component_gates": {
                    key: diagnostics[key]
                    for key in (
                        "structure_delivery",
                        "numeric_fidelity",
                        "date_fidelity",
                        "degeneration_free",
                    )
                },
                "raw_semantic_scores": values,
                "included_in_primary": True,
            }
            failure["row_sha256"] = _sha_text(_canonical(failure))
            failures.append(failure)
    score_path = run_root / "score/row_scores.jsonl"
    failure_path = run_root / "score/generation_diagnostic_failures.jsonl"
    _write_or_verify_text(score_path, _jsonl_text(output), resume=resume)
    _write_or_verify_text(failure_path, _jsonl_text(failures), resume=resume)
    receipt_path = run_root / "score/score_combination_receipt.json"
    receipt = _seal(
        {
            "schema_version": "paper-chk2-core8-loo-score-combination-receipt-v1",
            "status": "complete",
            "created_at_utc": _stable_created_at(receipt_path, resume=resume),
            "coverage": {
                "rows": EXPECTED_ROWS,
                "diagnostic_failures": len(failures),
                "diagnostic_failures_excluded": 0,
                "empty_answers_fixed_to_zero": sum(row["answer_empty"] for row in output),
            },
            "inputs": {
                "generation_rows": _binding(run_root / "score/generation_rows.jsonl", rows=EXPECTED_ROWS),
                "metric_manifests": {
                    metric: _binding(run_root / f"score/metrics/{metric}/manifest.json")
                    for metric in METRICS
                },
            },
            "artifacts": {
                "row_scores": _binding(score_path, rows=EXPECTED_ROWS),
                "diagnostic_failures": _binding(failure_path, rows=len(failures)),
            },
        }
    )
    _write_json(receipt_path, receipt, resume=resume)
    return receipt


def build_meeting_cells(
    score_rows: Sequence[Mapping[str, Any]],
    *,
    expected_meetings: int = EXPECTED_MEETINGS,
    expected_replicates: int = EXPECTED_REPLICATES,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], np.ndarray], list[str]]:
    expected_rows = expected_meetings * EXPECTED_VARIANTS * expected_replicates
    if len(score_rows) != expected_rows:
        raise PaperChk2Core8ScoreError(f"score matrix must contain {expected_rows} rows")
    coordinates: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    meeting_ids: dict[int, str] = {}
    for row in score_rows:
        coordinate = (
            int(row["meeting_rank"]),
            int(row["variant_rank"]),
            int(row["replicate_id"]),
        )
        if coordinate in coordinates:
            raise PaperChk2Core8ScoreError(f"duplicate score coordinate: {coordinate}")
        coordinates[coordinate] = row
        meeting = coordinate[0]
        prior = meeting_ids.setdefault(meeting, str(row["meeting_id"]))
        if prior != str(row["meeting_id"]):
            raise PaperChk2Core8ScoreError(f"meeting identity drift at rank {meeting}")
    expected = {
        (meeting, variant, replicate)
        for meeting in range(expected_meetings)
        for variant in range(EXPECTED_VARIANTS)
        for replicate in range(expected_replicates)
    }
    if set(coordinates) != expected or set(meeting_ids) != set(range(expected_meetings)):
        raise PaperChk2Core8ScoreError("score coordinate closure failed")
    matrices = {
        (topic, arm, metric): np.empty((expected_meetings, expected_replicates), dtype=np.float64)
        for topic in TOPICS
        for arm in ARMS
        for metric in METRICS
    }
    cells: list[dict[str, Any]] = []
    for meeting in range(expected_meetings):
        for topic_index, topic in enumerate(TOPICS):
            for arm_index, arm in enumerate(ARMS):
                variant = 1 + topic_index * 2 + arm_index
                full_scores = {metric: [] for metric in METRICS}
                intervention_scores = {metric: [] for metric in METRICS}
                deltas = {metric: [] for metric in METRICS}
                full_valid = 0
                intervention_valid = 0
                for replicate in range(expected_replicates):
                    full = coordinates[(meeting, 0, replicate)]
                    intervention = coordinates[(meeting, variant, replicate)]
                    if (
                        full["replicate_seed"] != intervention["replicate_seed"]
                        or full["row_seed"] != intervention["row_seed"]
                        or full["reference_minutes_sha256"] != intervention["reference_minutes_sha256"]
                    ):
                        raise PaperChk2Core8ScoreError(
                            f"Full/intervention paired binding drift at {meeting}/{variant}/{replicate}"
                        )
                    full_valid += int(full["generation_diagnostics"]["preregistered_core_valid"])
                    intervention_valid += int(intervention["generation_diagnostics"]["preregistered_core_valid"])
                    for metric in METRICS:
                        full_value = _metric_number(full["raw_semantic_scores"][metric], label="Full score")
                        intervention_value = _metric_number(
                            intervention["raw_semantic_scores"][metric], label="intervention score"
                        )
                        delta = full_value - intervention_value
                        full_scores[metric].append(full_value)
                        intervention_scores[metric].append(intervention_value)
                        deltas[metric].append(delta)
                        matrices[(topic, arm, metric)][meeting, replicate] = delta
                cell = {
                    "schema_version": MEETING_CELL_SCHEMA,
                    "meeting_id": meeting_ids[meeting],
                    "meeting_rank": meeting,
                    "topic": topic,
                    "arm": arm,
                    "replicate_ids": list(range(expected_replicates)),
                    "full_scores": full_scores,
                    "intervention_scores": intervention_scores,
                    "paired_deltas": deltas,
                    "meeting_mean_delta": {
                        metric: statistics.fmean(deltas[metric]) for metric in METRICS
                    },
                    "meeting_replicate_sd": {
                        metric: (
                            statistics.stdev(deltas[metric]) if expected_replicates > 1 else 0.0
                        )
                        for metric in METRICS
                    },
                    "generation_diagnostics": {
                        "full_core_valid_replicates": full_valid,
                        "intervention_core_valid_replicates": intervention_valid,
                        "used_to_filter_primary": False,
                    },
                }
                cell["row_sha256"] = _sha_text(_canonical(cell))
                cells.append(cell)
    if len(cells) != expected_meetings * len(TOPICS) * len(ARMS):
        raise PaperChk2Core8ScoreError("meeting-cell denominator drift")
    return cells, matrices, [meeting_ids[index] for index in range(expected_meetings)]


def make_bootstrap_plan(
    *,
    meeting_ids: Sequence[str],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = EXPECTED_REPLICATES,
) -> BootstrapPlan:
    meetings_count = len(meeting_ids)
    if draws < 2 or meetings_count < 2 or len(set(meeting_ids)) != meetings_count or replicates < 2:
        raise PaperChk2Core8ScoreError("invalid hierarchical bootstrap dimensions")
    rng = np.random.default_rng(seed)
    meeting_dtype = np.uint8 if meetings_count <= 255 else np.uint16
    meeting_indices = rng.integers(
        0, meetings_count, size=(draws, meetings_count), dtype=meeting_dtype
    )
    uniforms = rng.random(size=(draws, meetings_count, replicates))
    prefixes = {
        prefix: np.floor(uniforms[:, :, :prefix] * prefix).astype(np.uint8)
        for prefix in range(1, replicates + 1)
    }
    metadata = {
        "schema_version": "paper-chk2-core8-loo-shared-hierarchical-index-plan-v1",
        "draws": draws,
        "seed": seed,
        "meeting_ids": list(meeting_ids),
        "replicates": replicates,
        "meeting_index_shape": list(meeting_indices.shape),
        "prefix_coupling": "one_shared_uniform_array_with_floor_U_times_prefix_K",
        "shared_across_all_topic_arm_metric_cells": True,
    }
    digest = hashlib.sha256(_canonical(metadata).encode("utf-8"))
    digest.update(meeting_indices.tobytes(order="C"))
    for prefix in range(1, replicates + 1):
        digest.update(bytes([prefix]))
        digest.update(prefixes[prefix].tobytes(order="C"))
    return BootstrapPlan(meeting_indices, prefixes, digest.hexdigest(), metadata)


def _ci(values: np.ndarray) -> tuple[float, float]:
    alpha = (1.0 - CONFIDENCE) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha], method="linear")
    return float(low), float(high)


def _interval_classification(low: float, high: float) -> str:
    if low > 0:
        return "positive_excludes_zero"
    if high < 0:
        return "negative_excludes_zero"
    return "crosses_zero"


def _same_nonzero_sign(left: float, right: float) -> bool:
    return left != 0.0 and right != 0.0 and math.copysign(1.0, left) == math.copysign(1.0, right)


def _one_sample_t(values: np.ndarray) -> tuple[float | None, float]:
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise PaperChk2Core8ScoreError("invalid meeting vector for one-sample t test")
    if float(np.ptp(values)) == 0.0:
        return (0.0, 1.0) if float(values[0]) == 0.0 else (None, 0.0)
    result = scipy_stats.ttest_1samp(values, popmean=0.0, alternative="two-sided")
    statistic, p_value = float(result.statistic), float(result.pvalue)
    if not math.isfinite(statistic) or not math.isfinite(p_value):
        raise PaperChk2Core8ScoreError("non-finite meeting-level t statistic")
    return statistic, p_value


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    if not p_values or any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in p_values):
        raise PaperChk2Core8ScoreError("Holm family contains invalid p-values")
    count = len(p_values)
    order = sorted(range(count), key=lambda index: (float(p_values[index]), index))
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _compute_cell(
    item: tuple[tuple[str, str, str], np.ndarray],
    *,
    plan: BootstrapPlan,
) -> CellComputation:
    key, matrix = item
    topic, arm, metric = key
    meetings, replicates = matrix.shape
    if meetings != len(plan.metadata["meeting_ids"]) or replicates != int(plan.metadata["replicates"]):
        raise PaperChk2Core8ScoreError(f"bootstrap matrix dimension drift for {key}")
    if not np.all(np.isfinite(matrix)):
        raise PaperChk2Core8ScoreError(f"non-finite bootstrap matrix for {key}")
    meeting_means = matrix.mean(axis=1)
    full_indices = plan.prefix_replicate_indices[replicates]
    nested = matrix[plan.meeting_indices[:, :, None], full_indices].mean(axis=(1, 2))
    meeting_only = meeting_means[plan.meeting_indices].mean(axis=1)
    point = float(matrix.mean())
    nested_low, nested_high = _ci(nested)
    meeting_low, meeting_high = _ci(meeting_only)
    prefix_results: dict[str, Any] = {}
    for prefix in range(1, replicates + 1):
        prefix_matrix = matrix[:, :prefix]
        prefix_draw = prefix_matrix[
            plan.meeting_indices[:, :, None], plan.prefix_replicate_indices[prefix]
        ].mean(axis=(1, 2))
        low, high = _ci(prefix_draw)
        prefix_results[f"k{prefix}"] = {
            "replicates": prefix,
            "point_estimate": float(prefix_matrix.mean()),
            "ci_lower": low,
            "ci_upper": high,
            "ci_width": high - low,
            "interval_classification": _interval_classification(low, high),
        }
    k9 = prefix_results[f"k{replicates - 1}"]
    k10 = prefix_results[f"k{replicates}"]
    point_drift = abs(float(k10["point_estimate"]) - float(k9["point_estimate"]))
    lower_drift = abs(float(k10["ci_lower"]) - float(k9["ci_lower"]))
    upper_drift = abs(float(k10["ci_upper"]) - float(k9["ci_upper"]))
    endpoint_drift = max(lower_drift, upper_drift)
    leave_one = [float(np.delete(matrix, replicate, axis=1).mean()) for replicate in range(replicates)]
    loro_max = max(abs(value - point) for value in leave_one)
    nested_variance = float(np.var(nested, ddof=1))
    decoding_variance = float(np.var(matrix, axis=1, ddof=1).sum() / (meetings**2 * replicates))
    if float(np.ptp(matrix)) == 0.0:
        decoding_variance = 0.0
    variance_share = decoding_variance / nested_variance if nested_variance > 0 else (0.0 if decoding_variance == 0 else None)
    decisive = nested_low > 0 or nested_high < 0
    classification = _interval_classification(nested_low, nested_high)
    critical_prefixes = range(max(1, replicates - 2), replicates + 1)
    prefix_sign_stable = all(
        _same_nonzero_sign(point, float(prefix_results[f"k{prefix}"]["point_estimate"]))
        for prefix in critical_prefixes
    )
    prefix_ci_stable = not decisive or all(
        prefix_results[f"k{prefix}"]["interval_classification"] == classification
        for prefix in critical_prefixes
    )
    loro_sign_stable = all(_same_nonzero_sign(point, value) for value in leave_one)
    reasons: list[str] = []
    if point_drift > K10_K9_POINT_DRIFT_THRESHOLD:
        reasons.append("k10_minus_k9_point_drift_above_threshold")
    if endpoint_drift > K10_K9_CI_ENDPOINT_DRIFT_THRESHOLD:
        reasons.append("k10_minus_k9_ci_endpoint_drift_above_threshold")
    if loro_max > LORO_MAX_DRIFT_THRESHOLD:
        reasons.append("leave_one_replicate_max_drift_above_threshold")
    if variance_share is None or variance_share > DECODING_VARIANCE_SHARE_THRESHOLD:
        reasons.append("decoding_variance_share_above_threshold")
    if decisive and not prefix_ci_stable:
        reasons.append("critical_cell_k8_k9_k10_ci_classification_unstable")
    if decisive and (not prefix_sign_stable or not loro_sign_stable):
        reasons.append("critical_cell_prefix_or_loro_point_sign_unstable")
    t_stat, raw_p = _one_sample_t(meeting_means)
    row = {
        "schema_version": "paper-chk2-core8-loo-cell-statistic-v1",
        "topic": topic,
        "arm": arm,
        "metric": metric,
        "estimand": "raw_score_full_minus_raw_score_intervention",
        "point_estimate": point,
        "meeting_mean_sd": float(np.std(meeting_means, ddof=1)),
        "positive_meetings": int(np.count_nonzero(meeting_means > 0)),
        "negative_meetings": int(np.count_nonzero(meeting_means < 0)),
        "zero_meetings": int(np.count_nonzero(meeting_means == 0)),
        "hierarchical_bootstrap": {
            "draws": int(nested.size),
            "mean": float(nested.mean()),
            "standard_error": float(np.std(nested, ddof=1)),
            "ci_lower": nested_low,
            "ci_upper": nested_high,
            "draw_share_above_zero": float(np.mean(nested > 0)),
        },
        "meeting_only_bootstrap_sensitivity": {
            "draws": int(meeting_only.size),
            "mean": float(meeting_only.mean()),
            "standard_error": float(np.std(meeting_only, ddof=1)),
            "ci_lower": meeting_low,
            "ci_upper": meeting_high,
            "draw_share_above_zero": float(np.mean(meeting_only > 0)),
        },
        "meeting_level_t_test": {
            "t_statistic": t_stat,
            "degrees_of_freedom": meetings - 1,
            "raw_two_sided_p": raw_p,
        },
        "k_adequacy": {
            "prefixes": prefix_results,
            "k10_minus_k9_point_drift": point_drift,
            "k10_minus_k9_ci_lower_drift": lower_drift,
            "k10_minus_k9_ci_upper_drift": upper_drift,
            "k10_minus_k9_ci_endpoint_max_drift": endpoint_drift,
            "replicate_mean_deltas": [float(value) for value in matrix.mean(axis=0)],
            "leave_one_replicate_estimates": leave_one,
            "leave_one_replicate_max_abs_shift": loro_max,
            "conditional_decoding_mcse": math.sqrt(decoding_variance),
            "nested_bootstrap_variance": nested_variance,
            "estimated_decoding_variance_share": variance_share,
            "decisive_nested_ci": decisive,
            "prefix_point_sign_stable": prefix_sign_stable,
            "prefix_ci_classification_stable": prefix_ci_stable,
            "leave_one_replicate_sign_stable": loro_sign_stable,
            "increase_k_reasons": reasons,
        },
    }
    return CellComputation(key=key, row=row, nested_draws=nested)


def _validate_score_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != EXPECTED_ROWS:
        raise PaperChk2Core8ScoreError("row-score coverage drift")
    for position, row in enumerate(rows):
        if (
            row.get("schema_version") != SCORE_ROW_SCHEMA
            or int(row.get("absolute_case_index", -1)) != position
            or int(row.get("score_row_number", -1)) != position + 1
        ):
            raise PaperChk2Core8ScoreError(f"row-score identity/order drift at {position}")
        body = dict(row)
        observed = body.pop("row_sha256", None)
        if observed != _sha_text(_canonical(body)):
            raise PaperChk2Core8ScoreError(f"row-score seal drift at {position}")
        scores = row.get("raw_semantic_scores")
        if not isinstance(scores, Mapping) or set(scores) != set(METRICS):
            raise PaperChk2Core8ScoreError(f"row-score metric inventory drift at {position}")
        for metric in METRICS:
            _metric_number(scores[metric], label=f"{metric}/{position}")
        diagnostics = row.get("generation_diagnostics")
        if not isinstance(diagnostics, Mapping) or diagnostics.get("used_to_filter_primary") is not False:
            raise PaperChk2Core8ScoreError(f"row-score diagnostic policy drift at {position}")


def diagnostic_funnel(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _validate_score_rows(rows)
    component_names = (
        "structure_delivery",
        "numeric_fidelity",
        "date_fidelity",
        "degeneration_free",
    )
    component_pass = Counter({name: 0 for name in component_names})
    failure_codes: Counter[str] = Counter()
    core_valid = 0
    empty = 0
    normal_finish = 0
    for row in rows:
        diagnostics = row["generation_diagnostics"]
        for name in component_names:
            component_pass[name] += int(bool(diagnostics[name]))
        core_valid += int(bool(diagnostics["preregistered_core_valid"]))
        empty += int(bool(row["answer_empty"]))
        finish = str(row.get("finish_reason", "")).lower()
        normal_finish += int(finish in {"stop", "eos", "length"})
        failure_codes.update(str(value) for value in diagnostics["failure_codes"])
    return {
        "requested_rows": len(rows),
        "raw_semantic_scored_rows": len(rows),
        "normal_finish_rows": normal_finish,
        "nonempty_answer_rows": len(rows) - empty,
        "empty_answer_rows": empty,
        "preregistered_core_valid_rows": core_valid,
        "preregistered_core_invalid_rows": len(rows) - core_valid,
        "component_gate_pass": dict(component_pass),
        "component_gate_fail": {name: len(rows) - component_pass[name] for name in component_names},
        "failure_code_counts": dict(sorted(failure_codes.items())),
        "rows_removed_from_primary": 0,
        "diagnostic_only": True,
    }


def _stat_csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "topic",
        "arm",
        "metric",
        "point_estimate",
        "hierarchical_ci_lower",
        "hierarchical_ci_upper",
        "hierarchical_standard_error",
        "meeting_only_ci_lower",
        "meeting_only_ci_upper",
        "meeting_mean_sd",
        "positive_meetings",
        "negative_meetings",
        "zero_meetings",
        "t_statistic",
        "degrees_of_freedom",
        "raw_two_sided_p",
        "holm_adjusted_p",
        "holm_reject_005",
        "significance_marker",
        "estimated_decoding_variance_share",
        "increase_k_recommended",
        "increase_k_reasons",
    )
    output = []
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        nested = row["hierarchical_bootstrap"]
        meeting = row["meeting_only_bootstrap_sensitivity"]
        t_test = row["meeting_level_t_test"]
        adequacy = row["k_adequacy"]
        output_row = {
            "topic": row["topic"],
            "arm": row["arm"],
            "metric": row["metric"],
            "point_estimate": format(float(row["point_estimate"]), ".17g"),
            "hierarchical_ci_lower": format(float(nested["ci_lower"]), ".17g"),
            "hierarchical_ci_upper": format(float(nested["ci_upper"]), ".17g"),
            "hierarchical_standard_error": format(float(nested["standard_error"]), ".17g"),
            "meeting_only_ci_lower": format(float(meeting["ci_lower"]), ".17g"),
            "meeting_only_ci_upper": format(float(meeting["ci_upper"]), ".17g"),
            "meeting_mean_sd": format(float(row["meeting_mean_sd"]), ".17g"),
            "positive_meetings": row["positive_meetings"],
            "negative_meetings": row["negative_meetings"],
            "zero_meetings": row["zero_meetings"],
            "t_statistic": "" if t_test["t_statistic"] is None else format(float(t_test["t_statistic"]), ".17g"),
            "degrees_of_freedom": t_test["degrees_of_freedom"],
            "raw_two_sided_p": format(float(t_test["raw_two_sided_p"]), ".17g"),
            "holm_adjusted_p": format(float(t_test["holm_adjusted_p"]), ".17g"),
            "holm_reject_005": str(bool(t_test["holm_reject_005"])).lower(),
            "significance_marker": t_test["significance_marker"],
            "estimated_decoding_variance_share": "" if adequacy["estimated_decoding_variance_share"] is None else format(float(adequacy["estimated_decoding_variance_share"]), ".17g"),
            "increase_k_recommended": str(bool(adequacy["increase_k_reasons"])).lower(),
            "increase_k_reasons": "|".join(adequacy["increase_k_reasons"]),
        }
        writer.writerow(output_row)
    output.append(buffer.getvalue())
    return "".join(output)


def compute_statistics(
    run_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    workers: int = DEFAULT_WORKERS,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    resume: bool = False,
) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    if workers < 1 or workers > 32 or draws < 2:
        raise PaperChk2Core8ScoreError("invalid statistics worker/draw configuration")
    rows = _read_jsonl(run_root / "score/row_scores.jsonl")
    _validate_score_rows(rows)
    meeting_cells, matrices, meeting_ids = build_meeting_cells(rows)
    plan = make_bootstrap_plan(
        meeting_ids=meeting_ids, draws=draws, seed=seed, replicates=EXPECTED_REPLICATES
    )
    ordered_items = [
        ((topic, arm, metric), matrices[(topic, arm, metric)])
        for topic in TOPICS
        for arm in ARMS
        for metric in METRICS
    ]
    with ThreadPoolExecutor(max_workers=min(workers, len(ordered_items))) as executor:
        computations = list(executor.map(lambda item: _compute_cell(item, plan=plan), ordered_items))
    raw_p = [float(value.row["meeting_level_t_test"]["raw_two_sided_p"]) for value in computations]
    adjusted = holm_adjust(raw_p)
    statistical_rows: list[dict[str, Any]] = []
    increase_cells: list[dict[str, Any]] = []
    for computation, adjusted_p in zip(computations, adjusted, strict=True):
        row = computation.row
        t_test = row["meeting_level_t_test"]
        t_test["holm_family"] = "8_topics_x_2_arms_x_2_metrics"
        t_test["holm_family_size"] = len(computations)
        t_test["holm_adjusted_p"] = adjusted_p
        t_test["holm_reject_005"] = adjusted_p < 0.05
        t_test["significance_marker"] = (
            "***" if adjusted_p < 0.001 else "**" if adjusted_p < 0.01 else "*" if adjusted_p < 0.05 else ""
        )
        reasons = row["k_adequacy"]["increase_k_reasons"]
        if reasons:
            increase_cells.append(
                {
                    "topic": row["topic"],
                    "arm": row["arm"],
                    "metric": row["metric"],
                    "reasons": list(reasons),
                }
            )
        row["row_sha256"] = _sha_text(_canonical(row))
        statistical_rows.append(row)
    draws_rows: list[dict[str, Any]] = []
    for draw_id in range(draws):
        estimates = {
            "|".join(computation.key): float(computation.nested_draws[draw_id])
            for computation in computations
        }
        draw_row = {
            "schema_version": BOOTSTRAP_DRAW_SCHEMA,
            "draw_id": draw_id,
            "shared_index_plan_sha256": plan.sha256,
            "cell_estimates": estimates,
        }
        draw_row["row_sha256"] = _sha_text(_canonical(draw_row))
        draws_rows.append(draw_row)
    funnel = diagnostic_funnel(rows)
    score_dir = run_root / "score"
    results_path = score_dir / "bootstrap_results.json"
    created_at = _stable_created_at(results_path, resume=resume)
    results = _seal(
        {
            "schema_version": STATISTICS_SCHEMA,
            "status": "complete",
            "created_at_utc": created_at,
            "model_id": "paper_chk2",
            "model_label": "paper-chk2-cp50-lora-over-chk1-cp200",
            "estimand": {
                "replicate_delta": "raw_score_full_minus_raw_score_intervention",
                "positive_delta": "intervention_reduced_target_relative_similarity",
                "meeting_aggregation": "arithmetic_mean_of_10_paired_replicates",
                "panel_aggregation": "equal_weight_arithmetic_mean_of_128_meetings",
                "reference": "frozen_source_grounded_synthetic_minutes_reference",
                "generation_gates": "diagnostic_only_no_filter_no_zero_penalty",
            },
            "coverage": {
                "generation_rows": EXPECTED_ROWS,
                "meetings": EXPECTED_MEETINGS,
                "variants_per_meeting": EXPECTED_VARIANTS,
                "replicates": EXPECTED_REPLICATES,
                "meeting_cells": EXPECTED_MEETING_CELLS,
                "primary_topic_arm_metric_cells": len(computations),
            },
            "bootstrap_contract": {
                **plan.metadata,
                "index_plan_sha256": plan.sha256,
                "confidence": CONFIDENCE,
                "primary_interval": "shared_hierarchical_paired_percentile",
                "meeting_only_sensitivity": True,
                "thread_workers": workers,
                "worker_result_order": "frozen_topic_arm_metric_order",
            },
            "multiplicity": {
                "test": "two_sided_one_sample_t_on_128_meeting_means",
                "degrees_of_freedom": EXPECTED_MEETINGS - 1,
                "joint_family": "8_topics_x_2_arms_x_2_metrics",
                "family_size": len(computations),
                "adjustment": "Holm",
                "bootstrap_confidence_intervals_adjusted": False,
            },
            "k_adequacy": {
                "thresholds": {
                    "k10_minus_k9_point_drift": K10_K9_POINT_DRIFT_THRESHOLD,
                    "k10_minus_k9_ci_endpoint_drift": K10_K9_CI_ENDPOINT_DRIFT_THRESHOLD,
                    "leave_one_replicate_max_drift": LORO_MAX_DRIFT_THRESHOLD,
                    "decoding_variance_share": DECODING_VARIANCE_SHARE_THRESHOLD,
                },
                "increase_k_recommended": bool(increase_cells),
                "aggregate_status": "increase_k_recommended" if increase_cells else "k10_adequate_for_aggregate_estimand",
                "increase_k_reason_cells": increase_cells,
                "per_meeting_inference_authorized": False,
            },
            "diagnostic_funnel": funnel,
            "cells": statistical_rows,
            "limitations": [
                "This is target-relative post-selection sensitivity, not causal topic attribution.",
                "The reference is a deterministic source-grounded synthetic Minutes target, not official Minutes.",
                "Generation hard gates are diagnostics and do not filter the semantic estimand.",
                "Core8 removes or neutralises one block inside a multi-block out-of-distribution stress prompt; it does not identify information sufficiency.",
                "The 32 percentile intervals are unadjusted; only the 32 meeting-level t-test p-values receive joint Holm adjustment.",
                "K adequacy concerns aggregate topic-arm-metric estimands, not per-meeting inference.",
            ],
        }
    )
    _write_or_verify_text(score_dir / "meeting_cell_deltas.jsonl", _jsonl_text(meeting_cells), resume=resume)
    _write_or_verify_text(score_dir / "bootstrap_draws.jsonl", _jsonl_text(draws_rows), resume=resume)
    _write_json(results_path, results, resume=resume)
    _write_json(
        score_dir / "cell_statistics.json",
        {
            "schema_version": "paper-chk2-core8-loo-cell-statistics-table-v1",
            "rows": statistical_rows,
        },
        resume=resume,
    )
    _write_or_verify_text(score_dir / "cell_statistics.csv", _stat_csv_text(statistical_rows), resume=resume)
    manifest = _seal(
        {
            "schema_version": SCORE_MANIFEST_SCHEMA,
            "status": "complete",
            "created_at_utc": created_at,
            "immutable": True,
            "paper_chk2_scope": True,
            "historical_generation_or_score_rows_reused": False,
            "inputs": {
                "preparation_manifest": _binding(run_root / "preparation/evaluation_manifest.json"),
                "generation_validation": _binding(run_root / "generation/validation.json"),
                "consolidation_receipt": _binding(score_dir / "consolidation_receipt.json"),
                "score_combination_receipt": _binding(score_dir / "score_combination_receipt.json"),
                "semantic_manifest": _binding(DEFAULT_SEMANTIC_MANIFEST),
            },
            "artifacts": {
                "generation_rows": _binding(score_dir / "generation_rows.jsonl", rows=EXPECTED_ROWS),
                "row_scores": _binding(score_dir / "row_scores.jsonl", rows=EXPECTED_ROWS),
                "generation_diagnostic_failures": _binding(
                    score_dir / "generation_diagnostic_failures.jsonl",
                    rows=int(funnel["preregistered_core_invalid_rows"]),
                ),
                "meeting_cell_deltas": _binding(score_dir / "meeting_cell_deltas.jsonl", rows=EXPECTED_MEETING_CELLS),
                "bootstrap_draws": _binding(score_dir / "bootstrap_draws.jsonl", rows=draws),
                "bootstrap_results": _binding(score_dir / "bootstrap_results.json"),
                "cell_statistics_json": _binding(score_dir / "cell_statistics.json"),
                "cell_statistics_csv": _binding(score_dir / "cell_statistics.csv"),
            },
            "implementation": _binding(Path(__file__)),
        }
    )
    _write_json(score_dir / "score_manifest.json", manifest, resume=resume)
    return results


def score_semantics(
    run_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    metric: str = "all",
    semantic_manifest: Path = DEFAULT_SEMANTIC_MANIFEST,
    semantic_device: str = "cuda:0",
    semantic_batch_size: int = 16,
    resume: bool = False,
) -> dict[str, Any]:
    selected = METRICS if metric == "all" else (metric,)
    if any(value not in METRICS for value in selected):
        raise PaperChk2Core8ScoreError(f"metric must be all or one of {METRICS}")
    manifests = {
        value: score_metric(
            run_root,
            metric=value,
            semantic_manifest=semantic_manifest,
            semantic_device=semantic_device,
            semantic_batch_size=semantic_batch_size,
            resume=resume,
        )
        for value in selected
    }
    if set(selected) == set(METRICS):
        combine_semantic_scores(run_root, resume=resume)
    return {"status": "complete", "metrics": manifests}


def _assert_artifact_binding(binding: Mapping[str, Any], *, expected_path: Path, expected_rows: int | None = None) -> None:
    path = Path(str(binding.get("path", ""))).expanduser().resolve()
    if path != expected_path.expanduser().resolve():
        raise PaperChk2Core8ScoreError(f"artifact path drift: {expected_path}")
    observed = _binding(path, rows=expected_rows) if expected_rows is not None else _binding(path)
    for field in ("path", "bytes", "sha256"):
        if binding.get(field) != observed[field]:
            raise PaperChk2Core8ScoreError(f"artifact {field} drift: {expected_path}")
    if expected_rows is not None and int(binding.get("rows", -1)) != expected_rows:
        raise PaperChk2Core8ScoreError(f"artifact row-count drift: {expected_path}")


def validate_score_bundle(run_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    score_dir = run_root / "score"
    manifest = _read_json(score_dir / "score_manifest.json")
    _validate_seal(manifest, label="paper-chk2 Core8 score")
    if (
        manifest.get("schema_version") != SCORE_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("immutable") is not True
        or manifest.get("paper_chk2_scope") is not True
        or manifest.get("historical_generation_or_score_rows_reused") is not False
    ):
        raise PaperChk2Core8ScoreError("score manifest role/status drift")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "generation_rows",
        "row_scores",
        "generation_diagnostic_failures",
        "meeting_cell_deltas",
        "bootstrap_draws",
        "bootstrap_results",
        "cell_statistics_json",
        "cell_statistics_csv",
    }:
        raise PaperChk2Core8ScoreError("score artifact inventory drift")
    results = _read_json(score_dir / "bootstrap_results.json")
    _validate_seal(results, label="paper-chk2 Core8 statistics")
    if results.get("schema_version") != STATISTICS_SCHEMA or results.get("status") != "complete":
        raise PaperChk2Core8ScoreError("bootstrap result status/schema drift")
    draws = int(results.get("bootstrap_contract", {}).get("draws", -1))
    failures = int(results.get("diagnostic_funnel", {}).get("preregistered_core_invalid_rows", -1))
    expected_paths = {
        "generation_rows": (score_dir / "generation_rows.jsonl", EXPECTED_ROWS),
        "row_scores": (score_dir / "row_scores.jsonl", EXPECTED_ROWS),
        "generation_diagnostic_failures": (score_dir / "generation_diagnostic_failures.jsonl", failures),
        "meeting_cell_deltas": (score_dir / "meeting_cell_deltas.jsonl", EXPECTED_MEETING_CELLS),
        "bootstrap_draws": (score_dir / "bootstrap_draws.jsonl", draws),
        "bootstrap_results": (score_dir / "bootstrap_results.json", None),
        "cell_statistics_json": (score_dir / "cell_statistics.json", None),
        "cell_statistics_csv": (score_dir / "cell_statistics.csv", None),
    }
    for role, (path, row_count) in expected_paths.items():
        binding = artifacts[role]
        if not isinstance(binding, Mapping):
            raise PaperChk2Core8ScoreError(f"invalid artifact binding: {role}")
        _assert_artifact_binding(binding, expected_path=path, expected_rows=row_count)
    implementation = manifest.get("implementation")
    if not isinstance(implementation, Mapping):
        raise PaperChk2Core8ScoreError("implementation source binding missing")
    _assert_artifact_binding(implementation, expected_path=Path(__file__))
    if _sha_file(DEFAULT_SEMANTIC_MANIFEST) != DEFAULT_SEMANTIC_MANIFEST_SHA256:
        raise PaperChk2Core8ScoreError("semantic manifest changed after scoring")
    rows = _read_jsonl(score_dir / "row_scores.jsonl")
    _validate_score_rows(rows)
    semantic_rows = {metric: _load_metric_rows(run_root, metric) for metric in METRICS}
    for position, row in enumerate(rows):
        for metric in METRICS:
            source = semantic_rows[metric][position]
            if (
                source["generation_key"] != row["generation_key"]
                or source["sample_id"] != row["sample_id"]
                or float(source["value"]) != float(row["raw_semantic_scores"][metric])
            ):
                raise PaperChk2Core8ScoreError(
                    f"semantic worker/final score projection drift at {metric}/{position}"
                )
    meeting_cells = _read_jsonl(score_dir / "meeting_cell_deltas.jsonl")
    rebuilt_cells, _matrices, _meeting_ids = build_meeting_cells(rows)
    if meeting_cells != rebuilt_cells:
        raise PaperChk2Core8ScoreError("meeting-cell artifact differs from deterministic replay")
    failure_rows = _read_jsonl(score_dir / "generation_diagnostic_failures.jsonl")
    if len(failure_rows) != failures:
        raise PaperChk2Core8ScoreError("diagnostic-failure row count drift")
    for row in failure_rows:
        body = dict(row)
        observed = body.pop("row_sha256", None)
        if observed != _sha_text(_canonical(body)) or row.get("included_in_primary") is not True:
            raise PaperChk2Core8ScoreError("diagnostic-failure row seal/policy drift")
    draw_rows = _read_jsonl(score_dir / "bootstrap_draws.jsonl")
    if len(draw_rows) != draws:
        raise PaperChk2Core8ScoreError("bootstrap draw coverage drift")
    expected_cell_keys = {
        "|".join((topic, arm, metric))
        for topic in TOPICS for arm in ARMS for metric in METRICS
    }
    index_plan_sha = results["bootstrap_contract"]["index_plan_sha256"]
    for position, row in enumerate(draw_rows):
        body = dict(row)
        observed = body.pop("row_sha256", None)
        if (
            observed != _sha_text(_canonical(body))
            or int(row.get("draw_id", -1)) != position
            or row.get("shared_index_plan_sha256") != index_plan_sha
            or set(row.get("cell_estimates", {})) != expected_cell_keys
        ):
            raise PaperChk2Core8ScoreError(f"bootstrap draw row drift at {position}")
    table = _read_json(score_dir / "cell_statistics.json")
    statistics_rows = table.get("rows")
    if table.get("schema_version") != "paper-chk2-core8-loo-cell-statistics-table-v1" or not isinstance(statistics_rows, list) or len(statistics_rows) != 32:
        raise PaperChk2Core8ScoreError("cell-statistics JSON inventory drift")
    if statistics_rows != results.get("cells"):
        raise PaperChk2Core8ScoreError("cell-statistics JSON/result projection drift")
    for row in statistics_rows:
        body = dict(row)
        observed = body.pop("row_sha256", None)
        if observed != _sha_text(_canonical(body)):
            raise PaperChk2Core8ScoreError("cell-statistic row seal drift")
    with (score_dir / "cell_statistics.csv").open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != 32:
        raise PaperChk2Core8ScoreError("cell-statistics CSV row coverage drift")
    if [
        (row["topic"], row["arm"], row["metric"]) for row in csv_rows
    ] != [
        (row["topic"], row["arm"], row["metric"]) for row in statistics_rows
    ]:
        raise PaperChk2Core8ScoreError("cell-statistics CSV ordering drift")
    adjusted = [float(row["meeting_level_t_test"]["holm_adjusted_p"]) for row in statistics_rows]
    raw = [float(row["meeting_level_t_test"]["raw_two_sided_p"]) for row in statistics_rows]
    if any(abs(left - right) > 1e-15 for left, right in zip(adjusted, holm_adjust(raw), strict=True)):
        raise PaperChk2Core8ScoreError("joint Holm adjustment replay drift")
    return {
        "status": "verified",
        "score_manifest": _binding(score_dir / "score_manifest.json"),
        "rows": EXPECTED_ROWS,
        "meeting_cells": EXPECTED_MEETING_CELLS,
        "statistical_cells": 32,
        "bootstrap_draws": draws,
        "diagnostic_failures": failures,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(command: str) -> argparse.ArgumentParser:
        child = subparsers.add_parser(command)
        child.add_argument("--run-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        child.add_argument("--resume", action="store_true")
        return child

    common("consolidate")
    metric_parser = common("score-metric")
    metric_parser.add_argument("--metric", choices=METRICS, required=True)
    metric_parser.add_argument("--semantic-manifest", type=Path, default=DEFAULT_SEMANTIC_MANIFEST)
    metric_parser.add_argument("--semantic-device", default="cuda:0")
    metric_parser.add_argument("--semantic-batch-size", type=int, default=16)
    score_parser = common("score")
    score_parser.add_argument("--metric", choices=("all", *METRICS), default="all")
    score_parser.add_argument("--semantic-manifest", type=Path, default=DEFAULT_SEMANTIC_MANIFEST)
    score_parser.add_argument("--semantic-device", default="cuda:0")
    score_parser.add_argument("--semantic-batch-size", type=int, default=16)
    stats_parser = common("statistics")
    stats_parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    stats_parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    stats_parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    common("validate")
    all_parser = common("all")
    all_parser.add_argument("--semantic-manifest", type=Path, default=DEFAULT_SEMANTIC_MANIFEST)
    all_parser.add_argument("--semantic-device", default="cuda:0")
    all_parser.add_argument("--semantic-batch-size", type=int, default=16)
    all_parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    all_parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    all_parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "consolidate":
        result = consolidate_generation(args.run_root, resume=args.resume)
    elif args.command == "score-metric":
        result = score_metric(
            args.run_root,
            metric=args.metric,
            semantic_manifest=args.semantic_manifest,
            semantic_device=args.semantic_device,
            semantic_batch_size=args.semantic_batch_size,
            resume=args.resume,
        )
    elif args.command == "score":
        result = score_semantics(
            args.run_root,
            metric=args.metric,
            semantic_manifest=args.semantic_manifest,
            semantic_device=args.semantic_device,
            semantic_batch_size=args.semantic_batch_size,
            resume=args.resume,
        )
    elif args.command == "statistics":
        result = compute_statistics(
            args.run_root,
            workers=args.workers,
            draws=args.bootstrap_draws,
            seed=args.bootstrap_seed,
            resume=args.resume,
        )
    elif args.command == "validate":
        result = validate_score_bundle(args.run_root)
    else:
        consolidate_generation(args.run_root, resume=args.resume)
        score_semantics(
            args.run_root,
            metric="all",
            semantic_manifest=args.semantic_manifest,
            semantic_device=args.semantic_device,
            semantic_batch_size=args.semantic_batch_size,
            resume=args.resume,
        )
        compute_statistics(
            args.run_root,
            workers=args.workers,
            draws=args.bootstrap_draws,
            seed=args.bootstrap_seed,
            resume=args.resume,
        )
        result = validate_score_bundle(args.run_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
