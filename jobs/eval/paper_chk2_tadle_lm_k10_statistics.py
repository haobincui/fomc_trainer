#!/usr/bin/env python3
"""Statistical stages for the paper-chk2 K=10 Tadle-form diagnostic.

This module deliberately does not generate model completions.  It consumes a
closed 2,520-row meeting-document ledger produced by the companion generator,
scores the documents with the frozen WGAN Loughran--McDonald contract, and
replays the historical Federal Funds futures interaction design.  All model
comparisons are paired, and all bootstrap p-values use the prespecified
null-centred bootstrap distribution.

The output is a historical document-source association diagnostic.  Generated
documents were never observed by market participants, so none of the routines
below support a causal market-effect interpretation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import multiprocessing
import os
import platform
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from jobs.eval import chk3_beta_statistics as beta_statistics
from jobs.eval import run_tadle_form_ff_futures_v1 as tadle_base
from open_r1.validator.loo_generation_spec import validate_manifest_integrity


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION = Path(__file__).resolve()
LEGACY_SOURCE_RELEASE = (
    ROOT
    / "output/evaluation/main/tadle_form_official_generated_ff_futures_2004_2015_v1"
)
LEGACY_DOCUMENT_RELEASE = (
    ROOT
    / "output/evaluation/main/official_minutes_full_document_treasury_beta_1993_2025_v1"
)
DEFAULT_EVALUATION_ROOT = (
    ROOT / "output/evaluation/retrain_v2/"
    "paper_chk2_cp50_tadle_form_lm_ff_futures_k10_v1_20260902"
)
DEFAULT_SOURCE_MANIFEST = LEGACY_SOURCE_RELEASE / "manifest.json"
DEFAULT_SOURCE_PANEL = LEGACY_SOURCE_RELEASE / "estimation/analysis_panel.jsonl"
DEFAULT_STATEMENTS = LEGACY_SOURCE_RELEASE / "statements/documents.jsonl"
DEFAULT_OFFICIAL_MINUTES = (
    LEGACY_DOCUMENT_RELEASE / "documents/official_documents.jsonl"
)
DEFAULT_DICTIONARY = (
    ROOT.parent / "wgan_option-rq123-london-timezone/data/reference/"
    "Loughran-McDonald_MasterDictionary_1993-2025.csv"
)

FROZEN_SHA256 = {
    "source_manifest": "eeb29722e95903dadedfa9e496b6a8e017ef0b731d8e5a67befc6ffd198de44f",
    "source_panel": "72fdcd8aac1063b8d0024ae1f8b7151a064bce7e5b1a45c58e183282a2e25c6b",
    "statements": "972300ce03ba08740ef292e231a78bac20004ed2ae175a5db27bd8af76daec71",
    "official_minutes": "5432c4b04f61da51aae82eeb2e4a7b192564a60c4b6cd2aa39c2a7eb9f4bbb40",
    "dictionary": "e2d1328682bab7d2187684fb9f5420bb730401c9eefc00daf835edd203f4859d",
}

HORIZONS = (1, 3, 6, 12)
EXPECTED_NOBS = {1: 2_613, 3: 2_612, 6: 2_612, 12: 2_338}
EXPECTED_ESTIMABLE_EVENTS_BY_HORIZON = {1: 82, 3: 82, 6: 82, 12: 72}
ARMS = ("official", "chk0", "chk1", "paper_chk2_cp50")
GENERATED_ARMS = ("chk0", "chk1", "paper_chk2_cp50")
LABELS = {
    "official": "Official FOMC Minutes",
    "chk0": "Model chk-0",
    "chk1": "Model chk-1 cp200",
    "paper_chk2_cp50": "Model chk-2 cp50",
}
TOPIC_ORDER = (
    "Consumer-Price-Index-(CPI)",
    "GDP-Growth",
    "Government-Purchases",
    "Housing-Starts",
    "Industrial-Production",
    "Labour-Market",
    "Money-Supply",
    "Unemployment-Rate",
)
REPLICATE_SEEDS = (
    20_260_811,
    21_260_811,
    22_260_811,
    23_260_811,
    24_260_811,
    25_260_811,
    26_260_811,
    27_260_811,
    28_260_811,
    29_260_811,
)
REPLICATE_COUNT = len(REPLICATE_SEEDS)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_260_902
TADLE_GAMMA = 0.368
POST_CUTOFF = "2011-08-08"
BACKEND_ID = "wgan_historical_lm_polarity_v1"
CONVENTION = "calendar_month_offset"
SPECIFICATION = "tadle_post2011_interaction"
SCHEMA = "paper-chk2-tadle-lm-k10-statistics-v1"
EVALUATION_ID = "paper-chk2-cp50-tadle-form-lm-ff-futures-k10-v1"
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z'-]*")
WGAN_SCORER_COMMIT = "2619e5e"
PHASE_ARTIFACTS = {
    "sentiment": {
        "document_scores",
        "statement_removal_ledger",
        "dictionary_manifest",
        "k_stability",
    },
    "estimate": {
        "standardization_scales",
        "analysis_panel",
        "coefficient_estimates",
        "leave_one_event_out",
        "leave_one_year_out",
        "point_diagnostics",
    },
    "statistics": {
        "bootstrap_plan",
        "bootstrap_plan_arrays",
        "bootstrap_draws",
        "model_minus_official_contrasts",
        "absolute_distances",
        "distance_gains",
        "validation_report",
    },
}


class StatisticsError(RuntimeError):
    """A frozen data, scoring, or statistical contract failed closed."""


_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
_CPU_THREAD_LIMITER: Any | None = None


def _absolute_without_resolving(path: Path) -> Path:
    """Return an absolute path while retaining a final-component symlink."""

    return Path(os.path.abspath(path.expanduser()))


def _require_regular_file(path: Path) -> None:
    if path.is_symlink():
        raise StatisticsError(f"symlink inputs/artifacts are forbidden: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)


def _worker_count(workers: int, tasks: int, *, label: str) -> int:
    if isinstance(workers, bool) or int(workers) != workers or workers < 1:
        raise StatisticsError(f"{label} workers must be a positive integer")
    available = os.cpu_count() or 1
    if workers > available:
        raise StatisticsError(
            f"{label} workers={workers} exceeds available logical CPUs={available}"
        )
    if tasks < 1:
        raise StatisticsError(f"{label} task inventory is empty")
    return min(int(workers), int(tasks))


def _configure_cpu_worker() -> None:
    """Prevent nested native thread pools inside each process worker."""

    global _CPU_THREAD_LIMITER
    for name in _THREAD_ENVIRONMENT:
        os.environ[name] = "1"
    _CPU_THREAD_LIMITER = threadpool_limits(limits=1)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    _require_regular_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    _require_regular_file(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise StatisticsError(f"unterminated JSONL row at {path}:{line_number}")
            if not line.strip():
                raise StatisticsError(f"blank JSONL row at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StatisticsError(
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise StatisticsError(f"non-object JSONL row at {path}:{line_number}")
            yield row


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def read_json(path: Path) -> dict[str, Any]:
    _require_regular_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StatisticsError(f"non-object JSON: {path}")
    return value


def file_binding(
    path: Path,
    *,
    relative_to: Path | None = None,
    rows: int | None = None,
) -> dict[str, Any]:
    _require_regular_file(path)
    resolved = path.resolve()
    shown: Path | str = resolved
    if relative_to is not None:
        shown = resolved.relative_to(relative_to.resolve())
    result: dict[str, Any] = {
        "path": str(shown),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if rows is not None:
        observed_rows = 0
        with resolved.open("rb") as handle:
            for line in handle:
                if line.strip():
                    observed_rows += 1
        if observed_rows != int(rows):
            raise StatisticsError(
                f"JSONL row-count drift for {resolved}: {observed_rows} != {rows}"
            )
        result["rows"] = observed_rows
    return result


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(canonical(dict(row)) + "\n" for row in rows).encode("utf-8")


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    handle = io.BytesIO()
    np.savez_compressed(handle, **arrays)
    return handle.getvalue()


def _write_create_only(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}"
    if temporary.exists():
        raise StatisticsError(f"temporary output already exists: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"create-only output exists: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _verify_binding(path: Path, expected: Mapping[str, Any]) -> None:
    observed = file_binding(path, rows=expected.get("rows"))
    for key in ("bytes", "sha256", "rows"):
        if key in expected and observed.get(key) != expected.get(key):
            raise StatisticsError(
                f"artifact binding drift for {path}: {key}={observed.get(key)} "
                f"expected={expected.get(key)}"
            )


def verify_phase_manifest(
    evaluation_root: Path,
    manifest_path: Path,
    *,
    phase: str,
    inputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if manifest_path.is_symlink():
        raise StatisticsError(f"symlink phase manifest is forbidden: {manifest_path}")
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != f"{SCHEMA}:{phase}-manifest"
        or manifest.get("status") != "complete"
    ):
        raise StatisticsError(f"invalid {phase} manifest: {manifest_path}")
    recorded_inputs = manifest.get("inputs")
    if not isinstance(recorded_inputs, dict) or set(recorded_inputs) != set(inputs):
        raise StatisticsError(f"{phase} input inventory drift")
    for name, expected in inputs.items():
        recorded = recorded_inputs[name]
        if any(recorded.get(key) != expected.get(key) for key in ("bytes", "sha256")):
            raise StatisticsError(f"{phase} input binding drift: {name}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise StatisticsError(f"{phase} artifact inventory is empty")
    expected_artifacts = PHASE_ARTIFACTS.get(phase)
    if expected_artifacts is not None and set(artifacts) != expected_artifacts:
        raise StatisticsError(f"{phase} artifact inventory drift")
    for artifact in artifacts.values():
        if not isinstance(artifact, dict):
            raise StatisticsError(f"invalid {phase} artifact binding")
        _verify_binding(evaluation_root / str(artifact["path"]), artifact)
    return manifest


def verify_committed_phase(
    evaluation_root: Path, manifest_path: Path, *, phase: str
) -> dict[str, Any]:
    """Verify a committed upstream phase, including all recorded inputs."""

    manifest = read_json(manifest_path)
    recorded_inputs = manifest.get("inputs")
    if not isinstance(recorded_inputs, dict) or not recorded_inputs:
        raise StatisticsError(f"{phase} recorded input inventory is empty")
    current_inputs: dict[str, dict[str, Any]] = {}
    for name, value in recorded_inputs.items():
        if not isinstance(value, dict) or not value.get("path"):
            raise StatisticsError(f"invalid {phase} input binding: {name}")
        path = Path(str(value["path"]))
        if not path.is_absolute():
            raise StatisticsError(f"{phase} input path is not absolute: {name}")
        current_inputs[name] = file_binding(path, rows=value.get("rows"))
    return verify_phase_manifest(
        evaluation_root,
        manifest_path,
        phase=phase,
        inputs=current_inputs,
    )


def _write_phase(
    evaluation_root: Path,
    *,
    phase: str,
    manifest_path: Path,
    inputs: Mapping[str, Mapping[str, Any]],
    artifacts: Mapping[str, tuple[Path, bytes, int | None]],
    metadata: Mapping[str, Any],
    resume: bool,
) -> dict[str, Any]:
    written: dict[str, dict[str, Any]] = {}
    for name, (path, content, rows) in artifacts.items():
        if path.is_symlink():
            raise StatisticsError(f"symlink phase artifact is forbidden: {path}")
        if path.exists():
            if not resume:
                raise FileExistsError(
                    f"create-only output exists without terminal manifest: {path}"
                )
            if path.stat().st_size != len(content) or sha256_file(path) != sha256_bytes(
                content
            ):
                raise StatisticsError(
                    f"partial-phase artifact differs on resume: {path}"
                )
        else:
            _write_create_only(path, content)
        written[name] = file_binding(path, relative_to=evaluation_root, rows=rows)
    manifest = {
        "schema_version": f"{SCHEMA}:{phase}-manifest",
        "created_at_utc": utc_now(),
        "status": "complete",
        "inputs": dict(inputs),
        "artifacts": written,
        **dict(metadata),
    }
    _write_create_only(manifest_path, _json_bytes(manifest))
    return manifest


def _assert_frozen(path: Path, name: str, *, enforce: bool) -> None:
    _require_regular_file(path)
    if enforce and sha256_file(path) != FROZEN_SHA256[name]:
        raise StatisticsError(f"frozen {name} SHA-256 drift: {path}")


def _active(value: str | None) -> bool:
    text = "" if value is None else str(value).strip()
    if not text:
        return False
    try:
        return float(text) > 0.0
    except ValueError:
        return text.lower() not in {"false", "no", "none", "nan", "0"}


def load_dictionary(
    path: Path = DEFAULT_DICTIONARY,
) -> tuple[set[str], set[str], dict[str, Any]]:
    _require_regular_file(path)
    positive: set[str] = set()
    negative: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"Word", "Positive", "Negative"}.issubset(reader.fieldnames or []):
            raise StatisticsError(f"dictionary schema drift: {reader.fieldnames}")
        count = 0
        for row in reader:
            count += 1
            word = str(row["Word"]).strip().lower()
            if word and _active(row["Positive"]):
                positive.add(word)
            if word and _active(row["Negative"]):
                negative.add(word)
    if count != 86_553 or len(positive) != 347 or len(negative) != 2_345:
        raise StatisticsError(
            f"dictionary inventory drift: rows={count}, positive={len(positive)}, "
            f"negative={len(negative)}"
        )
    if positive & negative:
        raise StatisticsError("positive and negative dictionary lexemes overlap")
    return (
        positive,
        negative,
        {
            "dictionary_rows": count,
            "positive_unique_lexemes": len(positive),
            "negative_unique_lexemes": len(negative),
            "sha256": sha256_file(path),
        },
    )


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(str(text))]


def score_text(text: str, positive: set[str], negative: set[str]) -> dict[str, Any]:
    tokens = tokenize(text)
    positive_count = sum(token in positive for token in tokens)
    negative_count = sum(token in negative for token in tokens)
    dictionary_count = positive_count + negative_count
    return {
        "token_count": len(tokens),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "dictionary_count": dictionary_count,
        "polarity": float((positive_count - negative_count) / max(1, dictionary_count)),
    }


def _candidate_statement_spans(minutes: str) -> list[tuple[int, int]]:
    lowered = minutes.lower()
    marker = "the vote encompassed approval"
    starts: list[int] = []
    cursor = 0
    while True:
        found = lowered.find(marker, cursor)
        if found < 0:
            break
        starts.append(found)
        cursor = found + len(marker)
    spans: list[tuple[int, int]] = []
    for start in starts:
        ends = [
            lowered.find(marker_text, start)
            for marker_text in (
                "votes for this action:",
                "voting for this action:",
                "votes for this action.",
                "voting for this action.",
            )
        ]
        finite = [value for value in ends if value >= 0]
        if finite:
            spans.append((start, min(finite)))
    return spans


def remove_repeated_statement(
    minutes: str, statement: str
) -> tuple[str, dict[str, Any]]:
    spans = _candidate_statement_spans(minutes)
    if not spans:
        raise StatisticsError("no repeated-Statement candidate span")
    statement_tokens = tokenize(statement)
    candidates: list[tuple[float, int, int, int]] = []
    for index, (start, end) in enumerate(spans):
        candidate_tokens = tokenize(minutes[start:end])
        similarity = SequenceMatcher(
            None, statement_tokens, candidate_tokens, autojunk=False
        ).ratio()
        candidates.append((float(similarity), index, start, end))
    candidates.sort(reverse=True)
    similarity, selected_index, start, end = candidates[0]
    if similarity <= 0.10:
        raise StatisticsError(f"repeated-Statement match too weak: {similarity}")
    if len(candidates) > 1 and math.isclose(
        similarity, candidates[1][0], rel_tol=0.0, abs_tol=1e-12
    ):
        raise StatisticsError("ambiguous repeated-Statement match")
    cleaned = re.sub(r"\s+", " ", minutes[:start] + " " + minutes[end:]).strip()
    return cleaned, {
        "candidate_count": len(spans),
        "selected_candidate_index": selected_index,
        "similarity_ratio": similarity,
        "runner_up_similarity_ratio": None
        if len(candidates) == 1
        else candidates[1][0],
        "removed_start": start,
        "removed_end": end,
        "removed_characters": end - start,
        "original_text_sha256": sha256_bytes(minutes.encode("utf-8")),
        "cleaned_text_sha256": sha256_bytes(cleaned.encode("utf-8")),
    }


def load_source_skeleton(source_panel: Path = DEFAULT_SOURCE_PANEL) -> dict[str, Any]:
    source_backend = tadle_base.official_runner.DISTIL_ID
    by_horizon: dict[int, list[dict[str, Any]]] = {horizon: [] for horizon in HORIZONS}
    for row in iter_jsonl(source_panel):
        horizon = int(row.get("horizon_months", -1))
        if (
            horizon in by_horizon
            and row.get("backend_id") == source_backend
            and row.get("convention") == CONVENTION
            and row.get("arm") == "official"
        ):
            by_horizon[horizon].append(row)
    for horizon, rows in by_horizon.items():
        rows.sort(key=lambda row: str(row["trading_date"]))
        if len(rows) != EXPECTED_NOBS[horizon]:
            raise StatisticsError(
                f"source panel FF{horizon} rows={len(rows)}, expected={EXPECTED_NOBS[horizon]}"
            )
        if len({str(row["trading_date"]) for row in rows}) != len(rows):
            raise StatisticsError(f"duplicate FF{horizon} trading dates")
    events = [row for row in by_horizon[1] if row["event_meeting_id"] is not None]
    current_ids = tuple(str(row["event_meeting_id"]) for row in events)
    lag_ids = tuple(str(row["lag_meeting_id"]) for row in events)
    if len(current_ids) != 82 or len(set(current_ids)) != 82:
        raise StatisticsError("estimable event inventory drift")
    if len(lag_ids) != 82 or any(value in {"", "None"} for value in lag_ids):
        raise StatisticsError("lag meeting inventory drift")
    needed_ids = tuple(sorted(set(current_ids) | set(lag_ids)))
    if len(needed_ids) != 84:
        raise StatisticsError("current-plus-lag meeting inventory drift")
    scale_ids = needed_ids[1:]
    if len(scale_ids) != 83 or not set(current_ids).issubset(scale_ids):
        raise StatisticsError("83-meeting standardization population drift")
    lag_only_ids = tuple(sorted(set(needed_ids) - set(current_ids)))
    if lag_only_ids != ("2004-09-21", "2009-12-16"):
        raise StatisticsError(f"lag-only meeting inventory drift: {lag_only_ids}")
    return {
        "by_horizon": by_horizon,
        "current_ids": current_ids,
        "lag_ids": lag_ids,
        "needed_ids": needed_ids,
        "scale_ids": scale_ids,
        "lag_only_ids": lag_only_ids,
        "lag_map": dict(zip(current_ids, lag_ids, strict=True)),
    }


def validate_generated_documents(
    path: Path,
    source: Mapping[str, Any],
    *,
    replicate_count: int = REPLICATE_COUNT,
    replicate_seeds: Sequence[int] = REPLICATE_SEEDS,
) -> list[dict[str, Any]]:
    if replicate_count != len(replicate_seeds):
        raise StatisticsError("replicate count and seed inventory differ")
    needed = set(source["needed_ids"])
    rows = read_jsonl(path)
    expected_rows = len(needed) * len(GENERATED_ARMS) * replicate_count
    if len(rows) != expected_rows:
        raise StatisticsError(
            f"generated document rows={len(rows)}, expected={expected_rows}"
        )
    observed: set[tuple[str, str, int]] = set()
    document_ids: set[str] = set()
    for row in rows:
        arm = str(row.get("arm"))
        meeting = str(row.get("meeting_end_date"))
        replicate = int(row.get("replicate_id", -1))
        key = (arm, meeting, replicate)
        if arm not in GENERATED_ARMS or meeting not in needed:
            raise StatisticsError(f"unexpected generated document key: {key}")
        if key in observed:
            raise StatisticsError(f"duplicate generated document key: {key}")
        observed.add(key)
        document_id = str(row.get("document_id", ""))
        if not document_id or document_id in document_ids:
            raise StatisticsError(f"missing/duplicate document_id: {document_id}")
        document_ids.add(document_id)
        if str(row.get("meeting_id")) != meeting:
            raise StatisticsError(f"meeting identity drift: {key}")
        if not 0 <= replicate < replicate_count:
            raise StatisticsError(f"replicate id outside K={replicate_count}: {key}")
        if int(row.get("replicate_seed", -1)) != int(replicate_seeds[replicate]):
            raise StatisticsError(f"replicate seed pairing drift: {key}")
        if int(row.get("section_count", -1)) != len(TOPIC_ORDER):
            raise StatisticsError(f"section-count drift: {key}")
        if tuple(row.get("topic_order", ())) != TOPIC_ORDER:
            raise StatisticsError(f"topic-order drift: {key}")
        if row.get("assembly_separator") != "two_newlines_exact_section_answers":
            raise StatisticsError(f"assembly-separator drift: {key}")
        document_text = row.get("document_text")
        if not isinstance(document_text, str):
            raise StatisticsError(f"non-string generated document: {key}")
        if row.get("document_text_sha256") != sha256_bytes(
            document_text.encode("utf-8")
        ):
            raise StatisticsError(f"generated document hash drift: {key}")
    expected = {
        (arm, meeting, replicate)
        for arm in GENERATED_ARMS
        for meeting in needed
        for replicate in range(replicate_count)
    }
    if observed != expected:
        raise StatisticsError("generated document Cartesian product is incomplete")
    return rows


def validate_documents_manifest(
    evaluation_root: Path, generated_documents: Path
) -> dict[str, Any]:
    """Validate the committed generation-to-document lineage for a formal run."""

    manifest_path = evaluation_root / "documents/manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise StatisticsError(f"missing/unsafe documents manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    try:
        validate_manifest_integrity(manifest)
    except Exception as exc:
        raise StatisticsError(f"documents manifest integrity failed: {exc}") from exc
    if (
        manifest.get("schema_version") != "paper-chk2-tadle-lm-k10-document-manifest-v1"
        or manifest.get("status") != "complete"
        or manifest.get("evaluation_id") != EVALUATION_ID
    ):
        raise StatisticsError("documents manifest identity/status drift")
    coverage = manifest.get("coverage")
    expected_coverage = {
        "models": list(GENERATED_ARMS),
        "meetings": 84,
        "replicates": 10,
        "topics_per_document": 8,
        "documents_per_model": 840,
        "documents_total": 2_520,
    }
    if coverage != expected_coverage:
        raise StatisticsError(f"documents manifest coverage drift: {coverage}")
    record = manifest.get("generated_documents")
    if not isinstance(record, Mapping):
        raise StatisticsError("documents manifest lacks generated-document binding")
    recorded_path = Path(str(record.get("path", "")))
    if recorded_path.resolve() != generated_documents.resolve():
        raise StatisticsError(
            "documents manifest points to a different generated ledger"
        )
    observed = file_binding(generated_documents, rows=2_520)
    for key in ("bytes", "sha256", "rows"):
        if record.get(key) != observed.get(key):
            raise StatisticsError(f"generated-document manifest binding drift: {key}")
    for name, expected_rows in (
        ("source_generation_rows", 20_160),
        ("source_generation_validation", None),
    ):
        upstream = manifest.get(name)
        if not isinstance(upstream, Mapping) or not upstream.get("path"):
            raise StatisticsError(f"documents manifest lacks upstream binding: {name}")
        upstream_path = Path(str(upstream["path"]))
        current = file_binding(upstream_path, rows=expected_rows)
        for key in ("bytes", "sha256", "rows"):
            if key in upstream and upstream.get(key) != current.get(key):
                raise StatisticsError(f"documents upstream binding drift: {name}/{key}")
    return manifest


def _unique_documents(
    rows: Iterable[Mapping[str, Any]], needed: set[str], *, label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        meeting = str(row.get("meeting_end_date"))
        if meeting not in needed:
            continue
        if meeting in result:
            raise StatisticsError(f"duplicate {label} meeting: {meeting}")
        text = row.get("document_text")
        if not isinstance(text, str) or not text.strip():
            raise StatisticsError(f"empty {label} document: {meeting}")
        expected_sha = row.get("document_text_sha256")
        if expected_sha is not None and expected_sha != sha256_bytes(
            text.encode("utf-8")
        ):
            raise StatisticsError(f"{label} document hash drift: {meeting}")
        result[meeting] = row
    if set(result) != needed:
        raise StatisticsError(f"{label} meeting join is incomplete")
    return result


_SCORE_POSITIVE: frozenset[str] | None = None
_SCORE_NEGATIVE: frozenset[str] | None = None


def _init_score_worker(positive: set[str], negative: set[str]) -> None:
    global _SCORE_POSITIVE, _SCORE_NEGATIVE
    _configure_cpu_worker()
    _SCORE_POSITIVE = frozenset(positive)
    _SCORE_NEGATIVE = frozenset(negative)


def _score_task_with_lexicon(
    task: tuple[Any, ...],
    positive: set[str] | frozenset[str],
    negative: set[str] | frozenset[str],
) -> dict[str, Any]:
    kind = str(task[0])
    if kind == "official_pair":
        _, meeting, statement_text, official_text = task
        statement_score = score_text(str(statement_text), positive, negative)
        cleaned, removal = remove_repeated_statement(
            str(official_text), str(statement_text)
        )
        official_score = score_text(cleaned, positive, negative)
        return {
            "kind": kind,
            "meeting_id": str(meeting),
            "statement_score": statement_score,
            "official_score": official_score,
            "removal": removal,
        }
    if kind == "generated":
        _, arm, meeting, replicate, replicate_seed, document_text = task
        return {
            "kind": kind,
            "arm": str(arm),
            "meeting_id": str(meeting),
            "replicate_id": int(replicate),
            "replicate_seed": int(replicate_seed),
            "score": score_text(str(document_text), positive, negative),
        }
    raise StatisticsError(f"unknown sentiment-scoring task kind: {kind}")


def _score_task(task: tuple[Any, ...]) -> dict[str, Any]:
    if _SCORE_POSITIVE is None or _SCORE_NEGATIVE is None:
        raise StatisticsError("sentiment worker lexicon is missing")
    return _score_task_with_lexicon(task, _SCORE_POSITIVE, _SCORE_NEGATIVE)


def _run_score_tasks(
    tasks: Sequence[tuple[Any, ...]],
    positive: set[str],
    negative: set[str],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    active_workers = _worker_count(workers, len(tasks), label="sentiment")
    if active_workers == 1:
        output = [_score_task_with_lexicon(task, positive, negative) for task in tasks]
    else:
        chunksize = max(1, len(tasks) // (active_workers * 8))
        with ProcessPoolExecutor(
            max_workers=active_workers,
            mp_context=multiprocessing.get_context("fork"),
            initializer=_init_score_worker,
            initargs=(positive, negative),
        ) as executor:
            output = list(executor.map(_score_task, tasks, chunksize=chunksize))
    if len(output) != len(tasks):
        raise StatisticsError("sentiment worker result-count drift")
    for task, result in zip(tasks, output, strict=True):
        if str(task[0]) != str(result.get("kind")):
            raise StatisticsError("sentiment worker result-order drift")
        if str(task[0]) == "official_pair":
            if str(task[1]) != str(result.get("meeting_id")):
                raise StatisticsError("official sentiment worker result-order drift")
        elif (
            str(task[1]),
            str(task[2]),
            int(task[3]),
        ) != (
            str(result.get("arm")),
            str(result.get("meeting_id")),
            int(result.get("replicate_id", -1)),
        ):
            raise StatisticsError("generated sentiment worker result-order drift")
    return output


def build_score_bundle(
    *,
    source: Mapping[str, Any],
    generated_documents: Path,
    statement_documents: Path = DEFAULT_STATEMENTS,
    official_documents: Path = DEFAULT_OFFICIAL_MINUTES,
    dictionary: Path = DEFAULT_DICTIONARY,
    replicate_count: int = REPLICATE_COUNT,
    replicate_seeds: Sequence[int] = REPLICATE_SEEDS,
    workers: int = 1,
) -> dict[str, Any]:
    positive, negative, dictionary_manifest = load_dictionary(dictionary)
    generated_rows = validate_generated_documents(
        generated_documents,
        source,
        replicate_count=replicate_count,
        replicate_seeds=replicate_seeds,
    )
    needed = set(source["needed_ids"])
    statements = _unique_documents(
        iter_jsonl(statement_documents), needed, label="Statement"
    )
    official = _unique_documents(
        iter_jsonl(official_documents), needed, label="official Minutes"
    )

    generated_rows.sort(
        key=lambda row: (
            GENERATED_ARMS.index(str(row["arm"])),
            str(row["meeting_end_date"]),
            int(row["replicate_id"]),
        )
    )
    official_tasks = [
        (
            "official_pair",
            meeting,
            str(statements[meeting]["document_text"]),
            str(official[meeting]["document_text"]),
        )
        for meeting in source["needed_ids"]
    ]
    generated_tasks = [
        (
            "generated",
            str(row["arm"]),
            str(row["meeting_end_date"]),
            int(row["replicate_id"]),
            int(row["replicate_seed"]),
            str(row["document_text"]),
        )
        for row in generated_rows
    ]
    results = _run_score_tasks(
        [*official_tasks, *generated_tasks],
        positive,
        negative,
        workers=workers,
    )
    official_results = results[: len(official_tasks)]
    generated_results = results[len(official_tasks) :]

    raw: dict[str, Any] = {
        "statement": {},
        "official": {},
        "generated": {arm: {} for arm in GENERATED_ARMS},
    }
    score_rows: list[dict[str, Any]] = []
    removal_rows: list[dict[str, Any]] = []
    for meeting, result in zip(source["needed_ids"], official_results, strict=True):
        statement_score = result["statement_score"]
        raw["statement"][meeting] = statement_score["polarity"]
        score_rows.append(
            {
                "schema_version": f"{SCHEMA}:document-score-row",
                "document_source": "official_statement",
                "arm": "statement",
                "meeting_id": meeting,
                "replicate_id": None,
                **statement_score,
            }
        )
        official_score = result["official_score"]
        raw["official"][meeting] = official_score["polarity"]
        score_rows.append(
            {
                "schema_version": f"{SCHEMA}:document-score-row",
                "document_source": "official_minutes_statement_removed",
                "arm": "official",
                "meeting_id": meeting,
                "replicate_id": None,
                **official_score,
            }
        )
        removal_rows.append(
            {
                "schema_version": f"{SCHEMA}:statement-removal-row",
                "meeting_id": meeting,
                **result["removal"],
                "cleaned_token_count": official_score["token_count"],
            }
        )

    for result in generated_results:
        arm = str(result["arm"])
        meeting = str(result["meeting_id"])
        replicate = int(result["replicate_id"])
        values = result["score"]
        raw["generated"][arm].setdefault(meeting, {})[replicate] = values["polarity"]
        score_rows.append(
            {
                "schema_version": f"{SCHEMA}:document-score-row",
                "document_source": "generated_core8_document",
                "arm": arm,
                "paper_label": LABELS[arm],
                "meeting_id": meeting,
                "replicate_id": replicate,
                "replicate_seed": int(result["replicate_seed"]),
                **values,
            }
        )
    for arm in GENERATED_ARMS:
        for meeting in source["needed_ids"]:
            if set(raw["generated"][arm].get(meeting, {})) != set(
                range(replicate_count)
            ):
                raise StatisticsError(f"incomplete score inventory: {arm}/{meeting}")
    expected_scores = (
        2 * len(needed) + len(needed) * len(GENERATED_ARMS) * replicate_count
    )
    if len(score_rows) != expected_scores or len(removal_rows) != len(needed):
        raise StatisticsError("score/removal ledger row-count drift")
    return {
        "raw": raw,
        "score_rows": score_rows,
        "removal_rows": removal_rows,
        "dictionary": dictionary_manifest,
        "replicate_count": replicate_count,
    }


def _score_bundle_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "statement": {},
        "official": {},
        "generated": {arm: {} for arm in GENERATED_ARMS},
    }
    for row in rows:
        arm = str(row["arm"])
        meeting = str(row["meeting_id"])
        polarity = float(row["polarity"])
        if arm == "statement":
            raw["statement"][meeting] = polarity
        elif arm == "official":
            raw["official"][meeting] = polarity
        elif arm in GENERATED_ARMS:
            raw["generated"][arm].setdefault(meeting, {})[int(row["replicate_id"])] = (
                polarity
            )
        else:
            raise StatisticsError(f"unknown score arm: {arm}")
    return {"raw": raw, "score_rows": list(rows), "replicate_count": REPLICATE_COUNT}


def k_stability_rows(
    source: Mapping[str, Any],
    scores: Mapping[str, Any],
    *,
    replicate_count: int = REPLICATE_COUNT,
) -> list[dict[str, Any]]:
    """Summarize cumulative K sentiment stability relative to the K-max mean."""

    output: list[dict[str, Any]] = []
    meetings = tuple(source["needed_ids"])
    for arm in GENERATED_ARMS:
        matrix = np.asarray(
            [
                [
                    scores["raw"]["generated"][arm][meeting][replicate]
                    for replicate in range(replicate_count)
                ]
                for meeting in meetings
            ],
            dtype=np.float64,
        )
        target = matrix.mean(axis=1)
        for k in range(1, replicate_count + 1):
            cumulative = matrix[:, :k].mean(axis=1)
            drift = cumulative - target
            correlation = (
                1.0
                if k == replicate_count
                else float(np.corrcoef(cumulative, target)[0, 1])
            )
            if not math.isfinite(correlation):
                raise StatisticsError(
                    f"non-finite K-stability correlation: {arm}/K={k}"
                )
            output.append(
                {
                    "schema_version": f"{SCHEMA}:k-stability-row",
                    "arm": arm,
                    "paper_label": LABELS[arm],
                    "k": k,
                    "meeting_count": len(meetings),
                    "reference_k": replicate_count,
                    "mean_absolute_drift_from_kmax": float(np.mean(np.abs(drift))),
                    "root_mean_squared_drift_from_kmax": float(
                        np.sqrt(np.mean(np.square(drift)))
                    ),
                    "max_absolute_drift_from_kmax": float(np.max(np.abs(drift))),
                    "pearson_correlation_with_kmax": correlation,
                }
            )
    if len(output) != len(GENERATED_ARMS) * replicate_count:
        raise StatisticsError("K-stability row-count drift")
    return output


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.all(np.isfinite(array)):
        raise StatisticsError("invalid standardization vector")
    sd = float(array.std(ddof=1))
    if sd <= 0.0:
        raise StatisticsError("zero standardization scale")
    return float(array.mean()), sd


def construct_semantics(
    source: Mapping[str, Any],
    scores: Mapping[str, Any],
    *,
    replicate_count: int = REPLICATE_COUNT,
) -> dict[str, Any]:
    raw = scores["raw"]
    needed_ids = tuple(source["needed_ids"])
    scale_ids = tuple(source["scale_ids"])
    if set(raw["statement"]) != set(needed_ids) or set(raw["official"]) != set(
        needed_ids
    ):
        raise StatisticsError("official score inventory drift")
    point_minutes: dict[str, dict[str, float]] = {"official": dict(raw["official"])}
    for arm in GENERATED_ARMS:
        point_minutes[arm] = {}
        for meeting in needed_ids:
            inventory = raw["generated"][arm].get(meeting, {})
            if set(inventory) != set(range(replicate_count)):
                raise StatisticsError(
                    f"K={replicate_count} score inventory drift: {arm}/{meeting}"
                )
            point_minutes[arm][meeting] = float(
                np.mean([inventory[replicate] for replicate in range(replicate_count)])
            )
    statement_scale = mean_sd([raw["statement"][meeting] for meeting in scale_ids])
    minute_scales = {
        arm: mean_sd([point_minutes[arm][meeting] for meeting in scale_ids])
        for arm in ARMS
    }
    statement_z = {
        meeting: (raw["statement"][meeting] - statement_scale[0]) / statement_scale[1]
        for meeting in needed_ids
    }
    rz: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    ns: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        mean, sd = minute_scales[arm]
        for meeting in needed_ids:
            rz[arm][meeting] = (point_minutes[arm][meeting] - mean) / sd - statement_z[
                meeting
            ]
        for meeting in source["current_ids"]:
            ns[arm][meeting] = (
                rz[arm][meeting] - TADLE_GAMMA * rz[arm][source["lag_map"][meeting]]
            )
    return {
        "point_minutes": point_minutes,
        "statement_scale": statement_scale,
        "minute_scales": minute_scales,
        "statement_z": statement_z,
        "rz": rz,
        "ns": ns,
    }


def build_panel(
    source: Mapping[str, Any], semantics: Mapping[str, Any]
) -> list[dict[str, Any]]:
    panel: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for source_row in source["by_horizon"][horizon]:
            meeting = source_row["event_meeting_id"]
            for arm in ARMS:
                panel.append(
                    {
                        "schema_version": f"{SCHEMA}:analysis-panel-row",
                        "backend_id": BACKEND_ID,
                        "convention": CONVENTION,
                        "horizon_months": horizon,
                        "horizon_label": f"FF{horizon}",
                        "arm": arm,
                        "paper_label": LABELS[arm],
                        "trading_date": str(source_row["trading_date"]),
                        "release_year": str(source_row["release_year"]),
                        "post_2011": int(str(source_row["trading_date"]) > POST_CUTOFF),
                        "is_minutes_release_event": meeting is not None,
                        "event_meeting_id": meeting,
                        "lag_meeting_id": source_row["lag_meeting_id"],
                        "news_shock": (
                            0.0
                            if meeting is None
                            else semantics["ns"][arm][str(meeting)]
                        ),
                        "vix_log_percent_change": float(
                            source_row["vix_log_percent_change"]
                        ),
                        "futures_return_log_percent": float(
                            source_row["futures_return_log_percent"]
                        ),
                    }
                )
    if len(panel) != len(ARMS) * sum(EXPECTED_NOBS.values()):
        raise StatisticsError("analysis panel row-count drift")
    return panel


def fit_interaction(
    y: Sequence[float] | np.ndarray,
    news: Sequence[float] | np.ndarray,
    vix: Sequence[float] | np.ndarray,
    years: Sequence[str],
    post: Sequence[float] | np.ndarray,
    *,
    hc1: bool,
) -> dict[str, float]:
    y_array = np.asarray(y, dtype=np.float64)
    news_array = np.asarray(news, dtype=np.float64)
    vix_array = np.asarray(vix, dtype=np.float64)
    post_array = np.asarray(post, dtype=np.float64)
    if not (
        len(y_array)
        == len(news_array)
        == len(vix_array)
        == len(post_array)
        == len(years)
    ):
        raise StatisticsError("interaction vectors have different lengths")
    matrix, names = tadle_base.design_matrix(
        news_array, vix_array, years, post_array, interaction=True
    )
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, y_array, rcond=None)
    if int(rank) != matrix.shape[1] or not np.all(np.isfinite(coefficients)):
        raise StatisticsError(
            f"rank-deficient/nonfinite interaction design: rank={rank}, k={matrix.shape[1]}"
        )
    pre_index = names.index("news_shock")
    interaction_index = names.index("post2011_x_news_shock")
    result = {
        "beta_pre": float(coefficients[pre_index]),
        "beta_interaction": float(coefficients[interaction_index]),
        "beta_post": float(coefficients[pre_index] + coefficients[interaction_index]),
    }
    if hc1:
        fitted = tadle_base.fit_ols_hc1(y_array, matrix, names)
        covariance = fitted["covariance"]
        post_variance = float(
            covariance[pre_index, pre_index]
            + covariance[interaction_index, interaction_index]
            + 2.0 * covariance[pre_index, interaction_index]
        )
        result.update(
            {
                "se_pre": float(fitted["standard_errors"][pre_index]),
                "se_interaction": float(fitted["standard_errors"][interaction_index]),
                "se_post": math.sqrt(max(post_variance, 0.0)),
                "nobs": int(fitted["nobs"]),
                "r_squared": float(fitted["r_squared"]),
                "condition_number": float(np.linalg.cond(matrix)),
            }
        )
    return result


def _fit_panel_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    ordered = sorted(rows, key=lambda row: str(row["trading_date"]))
    return fit_interaction(
        [float(row["futures_return_log_percent"]) for row in ordered],
        [float(row["news_shock"]) for row in ordered],
        [float(row["vix_log_percent_change"]) for row in ordered],
        [str(row["release_year"]) for row in ordered],
        [float(row["post_2011"]) for row in ordered],
        hc1=True,
    )


def point_estimates(panel: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for arm in ARMS:
            rows = [
                row
                for row in panel
                if int(row["horizon_months"]) == horizon and row["arm"] == arm
            ]
            fitted = _fit_panel_rows(rows)
            result.append(
                {
                    "schema_version": f"{SCHEMA}:coefficient-row",
                    "backend_id": BACKEND_ID,
                    "convention": CONVENTION,
                    "specification": SPECIFICATION,
                    "horizon_months": horizon,
                    "horizon_label": f"FF{horizon}",
                    "arm": arm,
                    "paper_label": LABELS[arm],
                    "nobs": int(fitted["nobs"]),
                    "beta_pre_basis_points": 100.0 * fitted["beta_pre"],
                    "hc1_se_pre_basis_points": 100.0 * fitted["se_pre"],
                    "beta_interaction_basis_points": 100.0 * fitted["beta_interaction"],
                    "hc1_se_interaction_basis_points": 100.0 * fitted["se_interaction"],
                    "beta_post_basis_points": 100.0 * fitted["beta_post"],
                    "hc1_se_post_basis_points": 100.0 * fitted["se_post"],
                    "r_squared": fitted["r_squared"],
                    "condition_number": fitted["condition_number"],
                    "covariance": "HC1 heteroskedasticity-robust",
                }
            )
    if len(result) != 16:
        raise StatisticsError("coefficient inventory drift")
    return result


def _point_lookup(
    points: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], Mapping[str, Any]]:
    lookup = {(int(row["horizon_months"]), str(row["arm"])): row for row in points}
    if set(lookup) != {(horizon, arm) for horizon in HORIZONS for arm in ARMS}:
        raise StatisticsError("point-estimate key inventory drift")
    return lookup


_DIAGNOSTIC_ROWS: dict[tuple[int, str], tuple[Mapping[str, Any], ...]] | None = None
_DIAGNOSTIC_BASELINES: dict[tuple[int, str], Mapping[str, Any]] | None = None


def _diagnostic_jobs(
    panel: Sequence[Mapping[str, Any]], *, include_event: bool, include_year: bool
) -> tuple[
    dict[tuple[int, str], tuple[Mapping[str, Any], ...]],
    list[tuple[str, int, str, str]],
]:
    grouped: dict[tuple[int, str], tuple[Mapping[str, Any], ...]] = {}
    jobs: list[tuple[str, int, str, str]] = []
    for horizon in HORIZONS:
        horizon_rows = [row for row in panel if int(row["horizon_months"]) == horizon]
        event_ids = sorted(
            {
                str(row["event_meeting_id"])
                for row in horizon_rows
                if row["event_meeting_id"] is not None
            }
        )
        if len(event_ids) != EXPECTED_ESTIMABLE_EVENTS_BY_HORIZON[horizon]:
            raise StatisticsError(f"FF{horizon} leave-one-event inventory drift")
        years = sorted({str(row["release_year"]) for row in horizon_rows} - {"2011"})
        expected_years = 10 if horizon == 12 else 11
        if len(years) != expected_years:
            raise StatisticsError(
                f"FF{horizon} leave-one-year inventory drift: {years}"
            )
        for arm in ARMS:
            key = (horizon, arm)
            grouped[key] = tuple(row for row in horizon_rows if row["arm"] == arm)
            if len(grouped[key]) != EXPECTED_NOBS[horizon]:
                raise StatisticsError(
                    f"diagnostic panel inventory drift: FF{horizon}/{arm}"
                )
            if include_event:
                jobs.extend(("event", horizon, arm, meeting) for meeting in event_ids)
            if include_year:
                jobs.extend(("year", horizon, arm, year) for year in years)
    return grouped, jobs


def _init_diagnostic_worker(
    grouped: dict[tuple[int, str], tuple[Mapping[str, Any], ...]],
    baselines: dict[tuple[int, str], Mapping[str, Any]],
) -> None:
    global _DIAGNOSTIC_ROWS, _DIAGNOSTIC_BASELINES
    _configure_cpu_worker()
    _DIAGNOSTIC_ROWS = grouped
    _DIAGNOSTIC_BASELINES = baselines


def _diagnostic_task_with_context(
    task: tuple[str, int, str, str],
    grouped: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]],
    baselines: Mapping[tuple[int, str], Mapping[str, Any]],
) -> tuple[tuple[str, int, str, str], dict[str, Any]]:
    kind, horizon, arm, omitted = task
    arm_rows = grouped[(horizon, arm)]
    baseline = baselines[(horizon, arm)]
    if kind == "event":
        selected = [row for row in arm_rows if str(row["event_meeting_id"]) != omitted]
        identity_field = "omitted_meeting_id"
        schema_suffix = "leave-one-event-out-row"
    elif kind == "year":
        selected = [row for row in arm_rows if str(row["release_year"]) != omitted]
        identity_field = "omitted_release_year"
        schema_suffix = "leave-one-year-out-row"
    else:  # pragma: no cover - defensive worker guard
        raise StatisticsError(f"unknown estimate diagnostic task: {kind}")
    fitted = _fit_panel_rows(selected)
    row = {
        "schema_version": f"{SCHEMA}:{schema_suffix}",
        "horizon_months": horizon,
        "horizon_label": f"FF{horizon}",
        "arm": arm,
        identity_field: omitted,
        "nobs": int(fitted["nobs"]),
        "beta_pre_basis_points": 100.0 * fitted["beta_pre"],
        "beta_interaction_basis_points": 100.0 * fitted["beta_interaction"],
        "beta_post_basis_points": 100.0 * fitted["beta_post"],
        "delta_pre_basis_points": 100.0 * fitted["beta_pre"]
        - float(baseline["beta_pre_basis_points"]),
        "delta_interaction_basis_points": 100.0 * fitted["beta_interaction"]
        - float(baseline["beta_interaction_basis_points"]),
        "delta_post_basis_points": 100.0 * fitted["beta_post"]
        - float(baseline["beta_post_basis_points"]),
        "condition_number": fitted["condition_number"],
    }
    return task, row


def _diagnostic_task(
    task: tuple[str, int, str, str],
) -> tuple[tuple[str, int, str, str], dict[str, Any]]:
    if _DIAGNOSTIC_ROWS is None or _DIAGNOSTIC_BASELINES is None:
        raise StatisticsError("estimate diagnostic worker context is missing")
    return _diagnostic_task_with_context(task, _DIAGNOSTIC_ROWS, _DIAGNOSTIC_BASELINES)


def sensitivity_diagnostics(
    panel: Sequence[Mapping[str, Any]],
    points: Sequence[Mapping[str, Any]],
    *,
    workers: int,
    include_event: bool = True,
    include_year: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not include_event and not include_year:
        raise StatisticsError("no estimate diagnostic family requested")
    baselines = _point_lookup(points)
    grouped, jobs = _diagnostic_jobs(
        panel, include_event=include_event, include_year=include_year
    )
    active_workers = _worker_count(workers, len(jobs), label="estimate diagnostic")
    if active_workers == 1:
        with threadpool_limits(limits=1):
            results = [
                _diagnostic_task_with_context(task, grouped, baselines) for task in jobs
            ]
    else:
        chunksize = max(1, len(jobs) // (active_workers * 8))
        with ProcessPoolExecutor(
            max_workers=active_workers,
            mp_context=multiprocessing.get_context("fork"),
            initializer=_init_diagnostic_worker,
            initargs=(grouped, baselines),
        ) as executor:
            results = list(executor.map(_diagnostic_task, jobs, chunksize=chunksize))
    if len(results) != len(jobs):
        raise StatisticsError("estimate diagnostic worker result-count drift")
    if [task for task, _ in results] != jobs:
        raise StatisticsError("estimate diagnostic worker result-order drift")
    event_rows = [row for task, row in results if task[0] == "event"]
    year_rows = [row for task, row in results if task[0] == "year"]
    if include_event:
        expected = len(ARMS) * sum(EXPECTED_ESTIMABLE_EVENTS_BY_HORIZON.values())
        if len(event_rows) != expected:
            raise StatisticsError("leave-one-event-out row-count drift")
    if include_year:
        expected = len(ARMS) * sum(10 if horizon == 12 else 11 for horizon in HORIZONS)
        if len(year_rows) != expected:
            raise StatisticsError("leave-one-year-out row-count drift")
    return event_rows, year_rows


def leave_one_event_out(
    panel: Sequence[Mapping[str, Any]],
    points: Sequence[Mapping[str, Any]],
    *,
    workers: int = 1,
) -> list[dict[str, Any]]:
    output, _ = sensitivity_diagnostics(
        panel,
        points,
        workers=workers,
        include_event=True,
        include_year=False,
    )
    return output


def leave_one_year_out(
    panel: Sequence[Mapping[str, Any]],
    points: Sequence[Mapping[str, Any]],
    *,
    workers: int = 1,
) -> list[dict[str, Any]]:
    _, output = sensitivity_diagnostics(
        panel,
        points,
        workers=workers,
        include_event=False,
        include_year=True,
    )
    return output


def diagnostics_summary(
    points: Sequence[Mapping[str, Any]],
    loeo: Sequence[Mapping[str, Any]],
    loyo: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA}:point-diagnostics",
        "point_max_condition_number": max(
            float(row["condition_number"]) for row in points
        ),
        "leave_one_event_out_rows": len(loeo),
        "leave_one_year_out_rows": len(loyo),
        "max_abs_leave_one_event_delta_basis_points": {
            estimand: max(
                abs(float(row[f"delta_{estimand}_basis_points"])) for row in loeo
            )
            for estimand in ("pre", "interaction", "post")
        },
        "max_abs_leave_one_year_delta_basis_points": {
            estimand: max(
                abs(float(row[f"delta_{estimand}_basis_points"])) for row in loyo
            )
            for estimand in ("pre", "interaction", "post")
        },
        "interpretation": (
            "Sensitivity diagnostics omit one estimable release event or one calendar year "
            "other than the anchored 2011 bridge year; they are not additional hypothesis tests."
        ),
    }


def make_bootstrap_plan(
    source: Mapping[str, Any],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    replicate_count: int = REPLICATE_COUNT,
) -> dict[str, Any]:
    if draws < 1 or replicate_count < 1 or replicate_count > 127:
        raise StatisticsError("invalid bootstrap dimensions")
    rng = np.random.default_rng(seed)
    pre_uniform = rng.random((draws, 7), dtype=np.float64)
    post_uniform = rng.random((draws, 4), dtype=np.float64)
    replicate_choices = rng.integers(
        0,
        replicate_count,
        size=(draws, len(source["needed_ids"]), replicate_count),
        dtype=np.int8,
    )
    digest = hashlib.sha256()
    digest.update(
        canonical(
            {
                "draws": draws,
                "seed": seed,
                "needed_ids": list(source["needed_ids"]),
                "replicate_count": replicate_count,
            }
        ).encode("utf-8")
    )
    digest.update(pre_uniform.tobytes(order="C"))
    digest.update(post_uniform.tobytes(order="C"))
    digest.update(replicate_choices.tobytes(order="C"))
    return {
        "draws": draws,
        "seed": seed,
        "replicate_count": replicate_count,
        "pre_uniform": pre_uniform,
        "post_uniform": post_uniform,
        "replicate_choices": replicate_choices,
        "plan_sha256": digest.hexdigest(),
    }


def sampled_years(plan: Mapping[str, Any], draw: int, horizon: int) -> tuple[str, ...]:
    pre = tuple(str(year) for year in range(2005 if horizon == 12 else 2004, 2011))
    post = tuple(str(year) for year in range(2012, 2016))
    selected_pre = tuple(
        pre[min(int(value * len(pre)), len(pre) - 1)]
        for value in plan["pre_uniform"][draw, : len(pre)]
    )
    selected_post = tuple(
        post[min(int(value * len(post)), len(post) - 1)]
        for value in plan["post_uniform"][draw]
    )
    return selected_pre + ("2011",) + selected_post


def bootstrap_news(
    source: Mapping[str, Any],
    semantics: Mapping[str, Any],
    scores: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    needed_ids = tuple(source["needed_ids"])
    needed_index = {meeting: index for index, meeting in enumerate(needed_ids)}
    current_index = np.asarray(
        [needed_index[meeting] for meeting in source["current_ids"]], dtype=np.int64
    )
    lag_index = np.asarray(
        [needed_index[meeting] for meeting in source["lag_ids"]], dtype=np.int64
    )
    draws = int(plan["draws"])
    result: dict[str, np.ndarray] = {}
    official_rz = np.asarray(
        [semantics["rz"]["official"][meeting] for meeting in needed_ids],
        dtype=np.float64,
    )
    official_ns = official_rz[current_index] - TADLE_GAMMA * official_rz[lag_index]
    result["official"] = np.broadcast_to(official_ns, (draws, len(current_index)))
    statement_z = np.asarray(
        [semantics["statement_z"][meeting] for meeting in needed_ids],
        dtype=np.float64,
    )
    choices = np.asarray(plan["replicate_choices"], dtype=np.int64)
    replicate_count = int(plan["replicate_count"])
    for arm in GENERATED_ARMS:
        values = np.asarray(
            [
                [
                    scores["raw"]["generated"][arm][meeting][rep]
                    for rep in range(replicate_count)
                ]
                for meeting in needed_ids
            ],
            dtype=np.float64,
        )
        expanded = np.broadcast_to(values, (draws, *values.shape))
        resampled_mean = np.take_along_axis(expanded, choices, axis=2).mean(axis=2)
        mean, sd = semantics["minute_scales"][arm]
        rz = (resampled_mean - mean) / sd - statement_z[None, :]
        result[arm] = rz[:, current_index] - TADLE_GAMMA * rz[:, lag_index]
    return result


def _demean_by_year(values: np.ndarray, years: Sequence[str]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    for year in sorted(set(years)):
        mask = np.asarray([value == year for value in years], dtype=bool)
        result[mask] -= result[mask].mean(axis=0)
    return result


def _bootstrap_context(
    source: Mapping[str, Any],
    news_draws: Mapping[str, np.ndarray],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    current_index = {
        meeting: index for index, meeting in enumerate(source["current_ids"])
    }
    skeleton: dict[int, dict[str, Any]] = {}
    for horizon in HORIZONS:
        rows = source["by_horizon"][horizon]
        event_indexes = np.asarray(
            [
                -1
                if row["event_meeting_id"] is None
                else current_index[str(row["event_meeting_id"])]
                for row in rows
            ],
            dtype=np.int64,
        )
        years = tuple(str(row["release_year"]) for row in rows)
        y = np.asarray(
            [row["futures_return_log_percent"] for row in rows], dtype=np.float64
        )
        vix = np.asarray(
            [row["vix_log_percent_change"] for row in rows], dtype=np.float64
        )
        post = np.asarray([row["post_2011"] for row in rows], dtype=np.float64)
        skeleton[horizon] = {
            "event_indexes": event_indexes,
            "y": y,
            "vix": vix,
            "post": post,
            "years": years,
            "y_demeaned": _demean_by_year(y, years),
            "vix_demeaned": _demean_by_year(vix, years),
            "post_demeaned": _demean_by_year(post, years),
        }
    return {"skeleton": skeleton, "news_draws": dict(news_draws), "plan": plan}


_WORKER_CONTEXT: dict[str, Any] | None = None


def _init_bootstrap_worker(context: dict[str, Any]) -> None:
    global _WORKER_CONTEXT
    _configure_cpu_worker()
    _WORKER_CONTEXT = context


def _bootstrap_one(draw: int, context: Mapping[str, Any]) -> dict[str, Any]:
    plan = context["plan"]
    news_draws = context["news_draws"]
    skeleton = context["skeleton"]
    draw_betas: dict[str, Any] = {}
    draw_years: dict[str, list[str]] = {}
    for horizon in HORIZONS:
        data = skeleton[horizon]
        years = sampled_years(plan, draw, horizon)
        draw_years[f"FF{horizon}"] = list(years)
        multiplicity = {year: years.count(year) for year in set(years)}
        weights = np.sqrt(
            np.asarray(
                [multiplicity.get(year, 0) for year in data["years"]],
                dtype=np.float64,
            )
        )
        news_matrix = np.zeros(
            (len(data["event_indexes"]), len(ARMS)), dtype=np.float64
        )
        mask = data["event_indexes"] >= 0
        for arm_index, arm in enumerate(ARMS):
            news_matrix[mask, arm_index] = news_draws[arm][
                draw, data["event_indexes"][mask]
            ]
        interaction_matrix = data["post"][:, None] * news_matrix
        paired_columns = np.empty((len(news_matrix), 2 * len(ARMS)), dtype=np.float64)
        for arm_index in range(len(ARMS)):
            paired_columns[:, 2 * arm_index] = news_matrix[:, arm_index]
            paired_columns[:, 2 * arm_index + 1] = interaction_matrix[:, arm_index]
        paired_demeaned = _demean_by_year(paired_columns, data["years"])
        controls = (
            np.column_stack((data["vix_demeaned"], data["post_demeaned"]))
            * weights[:, None]
        )
        outcomes = (
            np.column_stack((data["y_demeaned"], paired_demeaned)) * weights[:, None]
        )
        control_coefficients, _, control_rank, _ = np.linalg.lstsq(
            controls, outcomes, rcond=None
        )
        if int(control_rank) != controls.shape[1]:
            raise StatisticsError("rank-deficient within-year bootstrap controls")
        residualized = outcomes - controls @ control_coefficients
        y_residual = residualized[:, 0]
        draw_betas[f"FF{horizon}"] = {}
        for arm_index, arm in enumerate(ARMS):
            columns = residualized[:, [1 + 2 * arm_index, 2 + 2 * arm_index]]
            gram = columns.T @ columns
            if np.linalg.matrix_rank(gram) != 2:
                raise StatisticsError(
                    f"rank-deficient bootstrap news terms: FF{horizon}/{arm}/{draw}"
                )
            coefficients = np.linalg.solve(gram, columns.T @ y_residual)
            draw_betas[f"FF{horizon}"][arm] = {
                "pre_bp": 100.0 * float(coefficients[0]),
                "interaction_bp": 100.0 * float(coefficients[1]),
                "post_bp": 100.0 * float(coefficients.sum()),
            }
    return {
        "schema_version": f"{SCHEMA}:bootstrap-row",
        "draw_index": draw,
        "plan_sha256": plan["plan_sha256"],
        "sampled_years": draw_years,
        "betas": draw_betas,
    }


def _bootstrap_chunk(indexes: Sequence[int]) -> list[dict[str, Any]]:
    if _WORKER_CONTEXT is None:  # pragma: no cover - defensive worker guard
        raise StatisticsError("bootstrap worker context is missing")
    return [_bootstrap_one(index, _WORKER_CONTEXT) for index in indexes]


def run_bootstrap_draws(
    source: Mapping[str, Any],
    semantics: Mapping[str, Any],
    scores: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    draws = int(plan["draws"])
    active_workers = _worker_count(workers, draws, label="bootstrap")
    news_draws = bootstrap_news(source, semantics, scores, plan)
    context = _bootstrap_context(source, news_draws, plan)
    if active_workers == 1:
        with threadpool_limits(limits=1):
            output = [_bootstrap_one(draw, context) for draw in range(draws)]
    else:
        chunks = [
            array.tolist()
            for array in np.array_split(
                np.arange(draws, dtype=np.int64), active_workers
            )
            if len(array)
        ]
        mp_context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=active_workers,
            mp_context=mp_context,
            initializer=_init_bootstrap_worker,
            initargs=(context,),
        ) as executor:
            output = [
                row for chunk in executor.map(_bootstrap_chunk, chunks) for row in chunk
            ]
        output.sort(key=lambda row: int(row["draw_index"]))
    if [int(row["draw_index"]) for row in output] != list(range(draws)):
        raise StatisticsError("bootstrap draw closure/order drift")
    return output


def null_centered_bootstrap_p_value(
    values: Sequence[float] | np.ndarray, estimate: float
) -> float:
    array = np.asarray(values, dtype=np.float64)
    point = float(estimate)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise StatisticsError("invalid bootstrap contrast vector")
    if not math.isfinite(point):
        raise StatisticsError("non-finite point contrast")
    extreme = int(np.count_nonzero(np.abs(array - point) >= abs(point)))
    return float((1 + extreme) / (len(array) + 1))


def summarize_bootstrap(
    points: Sequence[Mapping[str, Any]],
    draws: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    point = _point_lookup(points)
    contrasts: list[dict[str, Any]] = []
    estimands = {
        "pre": "beta_pre_basis_points",
        "interaction": "beta_interaction_basis_points",
        "post": "beta_post_basis_points",
    }
    for estimand, field in estimands.items():
        family = (
            "primitive_pre_and_interaction_24"
            if estimand != "post"
            else "derived_post_marginal_12"
        )
        for horizon in HORIZONS:
            official = float(point[(horizon, "official")][field])
            for arm in GENERATED_ARMS:
                estimate = float(point[(horizon, arm)][field]) - official
                values = np.asarray(
                    [
                        row["betas"][f"FF{horizon}"][arm][f"{estimand}_bp"]
                        - row["betas"][f"FF{horizon}"]["official"][f"{estimand}_bp"]
                        for row in draws
                    ],
                    dtype=np.float64,
                )
                low, high = beta_statistics.percentile_interval(values)
                contrasts.append(
                    {
                        "schema_version": f"{SCHEMA}:model-minus-official-row",
                        "backend_id": BACKEND_ID,
                        "convention": CONVENTION,
                        "specification": SPECIFICATION,
                        "family": family,
                        "estimand": estimand,
                        "horizon_months": horizon,
                        "horizon_label": f"FF{horizon}",
                        "arm": arm,
                        "paper_label": LABELS[arm],
                        "estimate_basis_points": estimate,
                        "ci_95_low_basis_points": low,
                        "ci_95_high_basis_points": high,
                        "bootstrap_p_raw": null_centered_bootstrap_p_value(
                            values, estimate
                        ),
                        "bootstrap_draws": len(draws),
                    }
                )
    for family in ("primitive_pre_and_interaction_24", "derived_post_marginal_12"):
        members = [row for row in contrasts if row["family"] == family]
        adjusted = beta_statistics.holm_adjust(
            [float(row["bootstrap_p_raw"]) for row in members]
        )
        for row, value in zip(members, adjusted, strict=True):
            row["holm_p"] = float(value)
    if len(contrasts) != 36:
        raise StatisticsError("model-minus-official contrast count drift")

    distances: list[dict[str, Any]] = []
    gains: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        official_point = float(point[(horizon, "official")]["beta_post_basis_points"])
        point_distances = {
            arm: abs(
                float(point[(horizon, arm)]["beta_post_basis_points"]) - official_point
            )
            for arm in GENERATED_ARMS
        }
        draw_distances = {
            arm: np.asarray(
                [
                    abs(
                        row["betas"][f"FF{horizon}"][arm]["post_bp"]
                        - row["betas"][f"FF{horizon}"]["official"]["post_bp"]
                    )
                    for row in draws
                ],
                dtype=np.float64,
            )
            for arm in GENERATED_ARMS
        }
        for arm in GENERATED_ARMS:
            low, high = beta_statistics.percentile_interval(draw_distances[arm])
            distances.append(
                {
                    "schema_version": f"{SCHEMA}:absolute-distance-row",
                    "backend_id": BACKEND_ID,
                    "convention": CONVENTION,
                    "specification": SPECIFICATION,
                    "estimand": "post",
                    "horizon_months": horizon,
                    "horizon_label": f"FF{horizon}",
                    "arm": arm,
                    "paper_label": LABELS[arm],
                    "distance_definition": "abs(beta_post_model-beta_post_official)",
                    "estimate_basis_points": point_distances[arm],
                    "ci_95_low_basis_points": low,
                    "ci_95_high_basis_points": high,
                    "bootstrap_draws": len(draws),
                }
            )
        for comparator in ("chk0", "chk1"):
            estimate = point_distances[comparator] - point_distances["paper_chk2_cp50"]
            values = draw_distances[comparator] - draw_distances["paper_chk2_cp50"]
            low, high = beta_statistics.percentile_interval(values)
            gains.append(
                {
                    "schema_version": f"{SCHEMA}:distance-gain-row",
                    "backend_id": BACKEND_ID,
                    "convention": CONVENTION,
                    "specification": SPECIFICATION,
                    "family": "chk2_distance_gains_8",
                    "estimand": "post",
                    "horizon_months": horizon,
                    "horizon_label": f"FF{horizon}",
                    "comparator_arm": comparator,
                    "comparator_label": LABELS[comparator],
                    "target_arm": "paper_chk2_cp50",
                    "target_label": LABELS["paper_chk2_cp50"],
                    "gain_definition": (
                        f"abs(beta_post_{comparator}-beta_post_official)-"
                        "abs(beta_post_paper_chk2_cp50-beta_post_official)"
                    ),
                    "positive_means_chk2_closer": True,
                    "estimate_basis_points": estimate,
                    "ci_95_low_basis_points": low,
                    "ci_95_high_basis_points": high,
                    "bootstrap_p_raw": null_centered_bootstrap_p_value(
                        values, estimate
                    ),
                    "bootstrap_draws": len(draws),
                }
            )
    adjusted_gains = beta_statistics.holm_adjust(
        [float(row["bootstrap_p_raw"]) for row in gains]
    )
    for row, value in zip(gains, adjusted_gains, strict=True):
        row["holm_p"] = float(value)
    if len(distances) != 12 or len(gains) != 8:
        raise StatisticsError("distance/gain row-count drift")
    return contrasts, distances, gains


def _standardization_manifest(semantics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA}:standardization-scales",
        "sample_n": 83,
        "ddof": 1,
        "scales_fixed_during_bootstrap": True,
        "statement": {
            "mean": semantics["statement_scale"][0],
            "sd": semantics["statement_scale"][1],
        },
        "minutes": {
            arm: {"mean": values[0], "sd": values[1]}
            for arm, values in semantics["minute_scales"].items()
        },
    }


def run_sentiment(
    evaluation_root: Path,
    generated_documents: Path | None = None,
    resume: bool = False,
    *,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    source_panel: Path = DEFAULT_SOURCE_PANEL,
    statement_documents: Path = DEFAULT_STATEMENTS,
    official_documents: Path = DEFAULT_OFFICIAL_MINUTES,
    dictionary: Path = DEFAULT_DICTIONARY,
    enforce_frozen_bindings: bool = True,
    workers: int = 6,
) -> dict[str, Any]:
    evaluation_root = evaluation_root.resolve()
    default_generated_documents = (
        evaluation_root / "documents/generated_documents.jsonl"
    )
    generated_documents = (
        default_generated_documents
        if generated_documents is None
        else _absolute_without_resolving(generated_documents)
    )
    formal_documents_lineage = generated_documents == default_generated_documents
    documents_manifest_path = evaluation_root / "documents/manifest.json"
    if formal_documents_lineage:
        validate_documents_manifest(evaluation_root, generated_documents)
    frozen = {
        "source_manifest": source_manifest,
        "source_panel": source_panel,
        "statements": statement_documents,
        "official_minutes": official_documents,
        "dictionary": dictionary,
    }
    for name, path in frozen.items():
        _assert_frozen(path, name, enforce=enforce_frozen_bindings)
    inputs = {
        "implementation": file_binding(IMPLEMENTATION),
        "generated_documents": file_binding(generated_documents, rows=2_520),
        **{name: file_binding(path) for name, path in frozen.items()},
    }
    if formal_documents_lineage:
        inputs["documents_manifest"] = file_binding(documents_manifest_path)
    manifest_path = evaluation_root / "sentiment/manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"create-only output exists: {manifest_path}")
        return verify_phase_manifest(
            evaluation_root, manifest_path, phase="sentiment", inputs=inputs
        )
    source = load_source_skeleton(source_panel)
    bundle = build_score_bundle(
        source=source,
        generated_documents=generated_documents,
        statement_documents=statement_documents,
        official_documents=official_documents,
        dictionary=dictionary,
        workers=workers,
    )
    dictionary_payload = {
        "schema_version": f"{SCHEMA}:dictionary-manifest",
        "backend_id": BACKEND_ID,
        "dictionary": bundle["dictionary"],
        "token_regex": TOKEN_PATTERN.pattern,
        "case": "lowercase exact lexical match",
        "score": "(positive_count-negative_count)/max(1,positive_count+negative_count)",
        "fallback_allowed": False,
        "historical_wgan_scorer_commit": WGAN_SCORER_COMMIT,
        "not_tadle_custom_dictionary": True,
    }
    stability = k_stability_rows(source, bundle)
    artifacts = {
        "document_scores": (
            evaluation_root / "sentiment/document_scores.jsonl",
            _jsonl_bytes(bundle["score_rows"]),
            len(bundle["score_rows"]),
        ),
        "statement_removal_ledger": (
            evaluation_root / "sentiment/official_statement_removal_ledger.jsonl",
            _jsonl_bytes(bundle["removal_rows"]),
            len(bundle["removal_rows"]),
        ),
        "dictionary_manifest": (
            evaluation_root / "sentiment/dictionary_manifest.json",
            _json_bytes(dictionary_payload),
            None,
        ),
        "k_stability": (
            evaluation_root / "sentiment/k_stability.jsonl",
            _jsonl_bytes(stability),
            len(stability),
        ),
    }
    return _write_phase(
        evaluation_root,
        phase="sentiment",
        manifest_path=manifest_path,
        inputs=inputs,
        artifacts=artifacts,
        metadata={
            "evaluation_id": EVALUATION_ID,
            "backend_id": BACKEND_ID,
            "replicates": REPLICATE_COUNT,
            "atomic_generation_rows": 20_160,
            "generation_meetings": 84,
            "estimable_current_meeting_ids": 82,
            "lag_only_meeting_ids": list(source["lag_only_ids"]),
            "broader_standardization_roster": 83,
            "generated_document_rows": 2_520,
            "document_score_rows": len(bundle["score_rows"]),
            "statement_removal_rows": len(bundle["removal_rows"]),
            "replicate_count": REPLICATE_COUNT,
            "k_stability_rows": len(stability),
            "formal_documents_lineage_verified": formal_documents_lineage,
            "unsealed_custom_generated_documents": not formal_documents_lineage,
        },
        resume=resume,
    )


def run_estimate(
    evaluation_root: Path,
    resume: bool = False,
    *,
    source_panel: Path = DEFAULT_SOURCE_PANEL,
    enforce_frozen_bindings: bool = True,
    workers: int = 6,
) -> dict[str, Any]:
    evaluation_root = evaluation_root.resolve()
    _assert_frozen(source_panel, "source_panel", enforce=enforce_frozen_bindings)
    sentiment_manifest = evaluation_root / "sentiment/manifest.json"
    if not sentiment_manifest.is_file():
        raise StatisticsError("sentiment phase is incomplete")
    sentiment_receipt = verify_committed_phase(
        evaluation_root, sentiment_manifest, phase="sentiment"
    )
    if sentiment_receipt["inputs"]["source_panel"]["sha256"] != sha256_file(
        source_panel
    ):
        raise StatisticsError("sentiment/estimate source panel binding mismatch")
    inputs = {
        "implementation": file_binding(IMPLEMENTATION),
        "sentiment_manifest": file_binding(sentiment_manifest),
        "document_scores": file_binding(
            evaluation_root / "sentiment/document_scores.jsonl", rows=2_688
        ),
        "source_panel": file_binding(source_panel),
    }
    receipt_path = evaluation_root / "estimation/estimate_receipt.json"
    if receipt_path.exists():
        if not resume:
            raise FileExistsError(f"create-only output exists: {receipt_path}")
        return verify_phase_manifest(
            evaluation_root, receipt_path, phase="estimate", inputs=inputs
        )
    source = load_source_skeleton(source_panel)
    score_rows = read_jsonl(evaluation_root / "sentiment/document_scores.jsonl")
    scores = _score_bundle_from_rows(score_rows)
    semantics = construct_semantics(source, scores)
    panel = build_panel(source, semantics)
    points = point_estimates(panel)
    loeo, loyo = sensitivity_diagnostics(panel, points, workers=workers)
    summary = diagnostics_summary(points, loeo, loyo)
    artifacts = {
        "standardization_scales": (
            evaluation_root / "estimation/standardization_scales.json",
            _json_bytes(_standardization_manifest(semantics)),
            None,
        ),
        "analysis_panel": (
            evaluation_root / "estimation/analysis_panel.jsonl",
            _jsonl_bytes(panel),
            len(panel),
        ),
        "coefficient_estimates": (
            evaluation_root / "estimation/coefficient_estimates.jsonl",
            _jsonl_bytes(points),
            len(points),
        ),
        "leave_one_event_out": (
            evaluation_root / "estimation/leave_one_event_out.jsonl",
            _jsonl_bytes(loeo),
            len(loeo),
        ),
        "leave_one_year_out": (
            evaluation_root / "estimation/leave_one_year_out.jsonl",
            _jsonl_bytes(loyo),
            len(loyo),
        ),
        "point_diagnostics": (
            evaluation_root / "estimation/point_diagnostics.json",
            _json_bytes(summary),
            None,
        ),
    }
    return _write_phase(
        evaluation_root,
        phase="estimate",
        manifest_path=receipt_path,
        inputs=inputs,
        artifacts=artifacts,
        metadata={
            "evaluation_id": EVALUATION_ID,
            "backend_id": BACKEND_ID,
            "convention": CONVENTION,
            "specification": SPECIFICATION,
            "replicates": REPLICATE_COUNT,
            "atomic_generation_rows": 20_160,
            "generated_documents": 2_520,
            "generation_meetings": 84,
            "estimable_current_meeting_ids": 82,
            "lag_only_meeting_ids": list(source["lag_only_ids"]),
            "broader_standardization_roster": 83,
            "analysis_panel_rows": len(panel),
            "coefficient_rows": len(points),
            "leave_one_event_out_rows": len(loeo),
            "leave_one_year_out_rows": len(loyo),
        },
        resume=resume,
    )


def _load_plan(npz_path: Path, plan_manifest: Mapping[str, Any]) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as archive:
        plan = {
            "draws": int(plan_manifest["draws"]),
            "seed": int(plan_manifest["seed"]),
            "replicate_count": int(plan_manifest["replicate_count"]),
            "plan_sha256": str(plan_manifest["plan_sha256"]),
            "pre_uniform": archive["pre_uniform"].copy(),
            "post_uniform": archive["post_uniform"].copy(),
            "replicate_choices": archive["replicate_choices"].copy(),
        }
    return plan


def run_bootstrap(
    evaluation_root: Path,
    workers: int = 6,
    resume: bool = False,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    source_panel: Path = DEFAULT_SOURCE_PANEL,
    enforce_frozen_bindings: bool = True,
) -> dict[str, Any]:
    evaluation_root = evaluation_root.resolve()
    _assert_frozen(source_panel, "source_panel", enforce=enforce_frozen_bindings)
    estimate_receipt = evaluation_root / "estimation/estimate_receipt.json"
    if not estimate_receipt.is_file():
        raise StatisticsError("estimate phase is incomplete")
    estimate_manifest = verify_committed_phase(
        evaluation_root, estimate_receipt, phase="estimate"
    )
    if estimate_manifest["inputs"]["source_panel"]["sha256"] != sha256_file(
        source_panel
    ):
        raise StatisticsError("estimate/bootstrap source panel binding mismatch")
    inputs = {
        "implementation": file_binding(IMPLEMENTATION),
        "estimate_receipt": file_binding(estimate_receipt),
        "analysis_panel": file_binding(
            evaluation_root / "estimation/analysis_panel.jsonl", rows=40_700
        ),
        "coefficient_estimates": file_binding(
            evaluation_root / "estimation/coefficient_estimates.jsonl", rows=16
        ),
        "document_scores": file_binding(
            evaluation_root / "sentiment/document_scores.jsonl", rows=2_688
        ),
        "source_panel": file_binding(source_panel),
    }
    manifest_path = evaluation_root / "estimation/statistics_manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"create-only output exists: {manifest_path}")
        manifest = verify_phase_manifest(
            evaluation_root, manifest_path, phase="statistics", inputs=inputs
        )
        if (
            int(manifest.get("bootstrap_draws", -1)) != draws
            or int(manifest.get("bootstrap_seed", -1)) != seed
        ):
            raise StatisticsError("resume bootstrap parameters differ from sealed run")
        return manifest

    source = load_source_skeleton(source_panel)
    score_rows = read_jsonl(evaluation_root / "sentiment/document_scores.jsonl")
    scores = _score_bundle_from_rows(score_rows)
    semantics = construct_semantics(source, scores)
    points = read_jsonl(evaluation_root / "estimation/coefficient_estimates.jsonl")
    plan = make_bootstrap_plan(source, draws=draws, seed=seed)
    draw_rows = run_bootstrap_draws(source, semantics, scores, plan, workers=workers)
    contrasts, distances, gains = summarize_bootstrap(points, draw_rows)

    plan_payload = {
        "schema_version": f"{SCHEMA}:bootstrap-plan",
        "draws": draws,
        "seed": seed,
        "replicate_count": REPLICATE_COUNT,
        "plan_sha256": plan["plan_sha256"],
        "method": "paired regime-stratified 2011-anchored calendar-year block bootstrap",
        "year_blocks": {
            "FF1_FF3_FF6_pre": list(range(2004, 2011)),
            "FF12_pre": list(range(2005, 2011)),
            "anchor": [2011],
            "post": list(range(2012, 2016)),
        },
        "generated_document_resampling": (
            "10 draws with replacement from K=10 per meeting; indices shared across models"
        ),
        "official_and_statement_resampling": False,
        "standardization_scales_fixed": True,
        "worker_count_is_not_part_of_random_plan": True,
        "p_value": "(1+count(abs(c_b-c_hat)>=abs(c_hat)))/(B+1)",
    }
    validation = validate_statistics(
        source=source,
        points=points,
        draws=draw_rows,
        contrasts=contrasts,
        distances=distances,
        gains=gains,
        plan=plan,
    )
    artifacts = {
        "bootstrap_plan": (
            evaluation_root / "estimation/bootstrap_plan.json",
            _json_bytes(plan_payload),
            None,
        ),
        "bootstrap_plan_arrays": (
            evaluation_root / "estimation/bootstrap_plan_arrays.npz",
            _npz_bytes(
                pre_uniform=plan["pre_uniform"],
                post_uniform=plan["post_uniform"],
                replicate_choices=plan["replicate_choices"],
            ),
            None,
        ),
        "bootstrap_draws": (
            evaluation_root / "estimation/bootstrap_draws.jsonl",
            _jsonl_bytes(draw_rows),
            len(draw_rows),
        ),
        "model_minus_official_contrasts": (
            evaluation_root / "estimation/model_minus_official_contrasts.jsonl",
            _jsonl_bytes(contrasts),
            len(contrasts),
        ),
        "absolute_distances": (
            evaluation_root / "estimation/absolute_distances.jsonl",
            _jsonl_bytes(distances),
            len(distances),
        ),
        "distance_gains": (
            evaluation_root / "estimation/distance_gains.jsonl",
            _jsonl_bytes(gains),
            len(gains),
        ),
        "validation_report": (
            evaluation_root / "estimation/validation_report.json",
            _json_bytes(validation),
            None,
        ),
    }
    return _write_phase(
        evaluation_root,
        phase="statistics",
        manifest_path=manifest_path,
        inputs=inputs,
        artifacts=artifacts,
        metadata={
            "evaluation_id": EVALUATION_ID,
            "backend_id": BACKEND_ID,
            "convention": CONVENTION,
            "specification": SPECIFICATION,
            "replicates": REPLICATE_COUNT,
            "atomic_generation_rows": 20_160,
            "generated_documents": 2_520,
            "bootstrap_draws": draws,
            "bootstrap_seed": seed,
            "bootstrap_workers_recorded_not_estimand_defining": workers,
            "plan_sha256": plan["plan_sha256"],
            "coefficient_rows": 16,
            "contrast_rows": 36,
            "absolute_distance_rows": 12,
            "distance_gain_rows": 8,
            "generation_meetings": 84,
            "estimable_current_meeting_ids": 82,
            "lag_only_meeting_ids": list(source["lag_only_ids"]),
            "broader_standardization_roster": 83,
            "exact_tadle_regression_form": True,
            "exact_tadle_2022_replication": False,
            "interpretation": "mixed-scope historical document-source association diagnostic",
            "limitations": validation["limitations"],
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
        },
        resume=resume,
    )


def validate_statistics(
    *,
    source: Mapping[str, Any],
    points: Sequence[Mapping[str, Any]],
    draws: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    distances: Sequence[Mapping[str, Any]],
    gains: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    def all_finite_numeric(value: Any) -> bool:
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return True
        if isinstance(value, (int, float, np.integer, np.floating)):
            return math.isfinite(float(value))
        if isinstance(value, Mapping):
            return all(all_finite_numeric(item) for item in value.values())
        if isinstance(value, Sequence):
            return all(all_finite_numeric(item) for item in value)
        return True

    numeric_point_fields = (
        "beta_pre_basis_points",
        "hc1_se_pre_basis_points",
        "beta_interaction_basis_points",
        "hc1_se_interaction_basis_points",
        "beta_post_basis_points",
        "hc1_se_post_basis_points",
        "r_squared",
        "condition_number",
    )
    numeric_interval_fields = (
        "estimate_basis_points",
        "ci_95_low_basis_points",
        "ci_95_high_basis_points",
    )
    checks = {
        "needed_meetings": len(source["needed_ids"]) == 84,
        "standardization_meetings": len(source["scale_ids"]) == 83,
        "estimable_events": len(source["current_ids"]) == 82,
        "lag_only_meetings": len(source["lag_only_ids"]) == 2,
        "coefficient_rows": len(points) == 16,
        "bootstrap_rows": len(draws) == int(plan["draws"]),
        "contrast_rows": len(contrasts) == 36,
        "absolute_distance_rows": len(distances) == 12,
        "distance_gain_rows": len(gains) == 8,
        "point_cartesian_product": {
            (int(row["horizon_months"]), str(row["arm"])) for row in points
        }
        == {(horizon, arm) for horizon in HORIZONS for arm in ARMS},
        "nobs_consistent_within_horizon": all(
            {
                int(row["nobs"])
                for row in points
                if int(row["horizon_months"]) == horizon
            }
            == {EXPECTED_NOBS[horizon]}
            for horizon in HORIZONS
        ),
        "finite_point_values": all(
            math.isfinite(float(row[field]))
            for row in points
            for field in numeric_point_fields
        ),
        "standard_errors_nonnegative": all(
            float(row[field]) >= 0.0
            for row in points
            for field in (
                "hc1_se_pre_basis_points",
                "hc1_se_interaction_basis_points",
                "hc1_se_post_basis_points",
            )
        ),
        "finite_draw_values": all(
            math.isfinite(float(value[field]))
            for row in draws
            for horizon in row["betas"].values()
            for value in horizon.values()
            for field in ("pre_bp", "interaction_bp", "post_bp")
        ),
        "finite_contrast_distance_gain_values": all(
            math.isfinite(float(row[field]))
            for row in [*contrasts, *distances, *gains]
            for field in numeric_interval_fields
        ),
        "all_result_numerics_finite": all_finite_numeric(
            [points, draws, contrasts, distances, gains]
        ),
        "ordered_confidence_intervals": all(
            float(row["ci_95_low_basis_points"])
            <= float(row["ci_95_high_basis_points"])
            for row in [*contrasts, *distances, *gains]
        ),
        "nonnegative_distance_intervals": all(
            float(row["ci_95_low_basis_points"]) >= 0.0
            and float(row["ci_95_high_basis_points"]) >= 0.0
            for row in distances
        ),
        "post_identity_points": all(
            math.isclose(
                float(row["beta_post_basis_points"]),
                float(row["beta_pre_basis_points"])
                + float(row["beta_interaction_basis_points"]),
                rel_tol=0.0,
                abs_tol=1e-10,
            )
            for row in points
        ),
        "post_identity_draws": all(
            math.isclose(
                float(value["post_bp"]),
                float(value["pre_bp"]) + float(value["interaction_bp"]),
                rel_tol=0.0,
                abs_tol=1e-10,
            )
            for row in draws
            for horizon in row["betas"].values()
            for value in horizon.values()
        ),
        "bootstrap_plan_closed": [int(row["draw_index"]) for row in draws]
        == list(range(int(plan["draws"]))),
        "bootstrap_coordinate_closure": all(
            set(row["betas"]) == {f"FF{horizon}" for horizon in HORIZONS}
            and all(set(values) == set(ARMS) for values in row["betas"].values())
            and row.get("plan_sha256") == plan.get("plan_sha256")
            for row in draws
        )
        and isinstance(plan.get("plan_sha256"), str)
        and bool(plan.get("plan_sha256")),
        "contrast_coordinate_closure": {
            (str(row["estimand"]), int(row["horizon_months"]), str(row["arm"]))
            for row in contrasts
        }
        == {
            (estimand, horizon, arm)
            for estimand in ("pre", "interaction", "post")
            for horizon in HORIZONS
            for arm in GENERATED_ARMS
        },
        "distance_coordinate_closure": {
            (int(row["horizon_months"]), str(row["arm"])) for row in distances
        }
        == {(horizon, arm) for horizon in HORIZONS for arm in GENERATED_ARMS},
        "gain_coordinate_closure": {
            (int(row["horizon_months"]), str(row["comparator_arm"])) for row in gains
        }
        == {
            (horizon, comparator)
            for horizon in HORIZONS
            for comparator in ("chk0", "chk1")
        },
        "holm_probabilities_valid": all(
            0.0 <= float(row["holm_p"]) <= 1.0 for row in [*contrasts, *gains]
        ),
        "distance_nonnegative": all(
            float(row["estimate_basis_points"]) >= 0.0 for row in distances
        ),
        "gain_sign_convention": all(
            row.get("positive_means_chk2_closer") is True for row in gains
        ),
    }
    if not all(checks.values()):
        raise StatisticsError(f"statistics validation failed: {checks}")
    return {
        "schema_version": f"{SCHEMA}:validation-report",
        "status": "passed",
        "checks": checks,
        "limitations": [
            "The Loughran--McDonald dictionary is not Tadle's custom monetary-policy dictionary.",
            "Generated Core8 documents and official full Minutes differ in information scope.",
            "The bootstrap conditions on fixed point-sample standardization scales and the observed 2011 bridge year.",
            "Generated texts were not observed by market participants; coefficients are not causal market effects.",
            "Training-meeting overlap makes the diagnostic mixed-scope rather than leakage-safe external evaluation.",
            "Failure to reject a model-minus-official difference is not evidence of equivalence.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("sentiment", "estimate", "bootstrap", "all"))
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument("--generated-documents", type=Path, default=None)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--source-panel", type=Path, default=DEFAULT_SOURCE_PANEL)
    parser.add_argument("--statement-documents", type=Path, default=DEFAULT_STATEMENTS)
    parser.add_argument(
        "--official-documents", type=Path, default=DEFAULT_OFFICIAL_MINUTES
    )
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--bootstrap-workers",
        type=int,
        default=6,
        help=(
            "shared CPU process budget for sentiment scoring, estimate diagnostics, "
            "and bootstrap draws"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.evaluation_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    _worker_count(args.bootstrap_workers, 1, label="CPU")
    source_manifest = _absolute_without_resolving(args.source_manifest)
    source_panel = _absolute_without_resolving(args.source_panel)
    statement_documents = _absolute_without_resolving(args.statement_documents)
    official_documents = _absolute_without_resolving(args.official_documents)
    dictionary = _absolute_without_resolving(args.dictionary)
    if args.phase in {"sentiment", "all"}:
        run_sentiment(
            root,
            args.generated_documents,
            args.resume,
            source_manifest=source_manifest,
            source_panel=source_panel,
            statement_documents=statement_documents,
            official_documents=official_documents,
            dictionary=dictionary,
            workers=args.bootstrap_workers,
        )
    if args.phase in {"estimate", "all"}:
        run_estimate(
            root,
            args.resume,
            source_panel=source_panel,
            workers=args.bootstrap_workers,
        )
    if args.phase in {"bootstrap", "all"}:
        run_bootstrap(
            root,
            args.bootstrap_workers,
            args.resume,
            draws=args.bootstrap_draws,
            seed=args.bootstrap_seed,
            source_panel=source_panel,
        )
    print(
        canonical(
            {
                "status": "complete",
                "phase": args.phase,
                "evaluation_root": str(root),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
