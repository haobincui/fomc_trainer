#!/usr/bin/env python3
"""Build and score leakage-controlled FOMC decision comparison baselines.

This job is intentionally independent of the SHA-bound chk4 evaluators and of
the legacy ``jobs/main/eval_decision_baselines.py`` implementation.  It has two
explicit fit contracts:

* ``matched_roster`` fits learned baselines on the 211 unique task-training
  meetings, including no validation, test, or external-panel labels.
* ``expanding_window`` refits on all policy actions strictly before each
  evaluated meeting and records unavailable early windows instead of inventing
  pre-1993 training data.

The historical N19 and post-cutoff N12 panels are always scored separately.
There is deliberately no pooled N31 estimand.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import statistics
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from open_r1.utils import decision_comparison_baselines as model_utils


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION = Path(__file__).resolve()
DEFAULT_CONFIG = ROOT / "configs/main/decision_comparison_baselines_v3.json"
SCHEMA = "decision-comparison-baselines-v1"
PREDICTION_SCHEMA = "decision-comparison-prediction-v1"
DIRECTIONS = ("cut", "hold", "hike")
PROBABILITY_LABELS = (*DIRECTIONS, "invalid")
FIT_CONTRACTS = ("matched_roster", "expanding_window")
BASELINE_MODELS = (
    "always_hold",
    "lag1",
    "hamilton_jorda_ach_op",
    "kauppi_mnl",
    "ordered_probit",
    "ordinal_rf",
    "futures",
)
PAPER_MODEL_LABELS = (
    "model_chk0",
    "model_chk1_cp200",
    "model_chk3_sft_cp38",
    "model_chk3_grpo_cp450",
)
MODEL_DISPLAYS = {
    "always_hold": "Always hold",
    "lag1": "Lag-1 action",
    "hamilton_jorda_ach_op": "Hamilton\u2013Jord\u00e0 ACH\u2013OP",
    "kauppi_mnl": "Kauppi-style multinomial logit",
    "ordered_probit": "Ordered probit",
    "ordinal_rf": "Python cumulative ordinal RF approximation",
    "futures": "Fed Funds futures",
    "model_chk0": "Model chk-0",
    "model_chk1_cp200": "Model chk-1 cp200",
    "model_chk3_sft_cp38": "Model chk-3 SFT cp38",
    "model_chk3_grpo_cp450": "Model chk-3 GRPO cp450",
}
SIGNED_MOVE_GRID = tuple(range(-100, 101, 25))
LOG_LOSS_EPSILON = 1e-15
HAMILTON_JORDA_FIT_CONTRACT = (
    "published_hj_ach_1989_2001_op_code_n102_current_vintage_inputs"
)
EXPECTED_UNIQUE_COUNTS = {"train": 211, "validation": 13, "test": 13}
EXPECTED_TRAIN_CLASSES = {"cut": 23, "hold": 156, "hike": 32}
EXPECTED_PANEL_COUNTS = {"historical_n19": 19, "postcutoff_n12": 12}
EXPECTED_PANEL_CLASSES = {
    "historical_n19": {"cut": 4, "hold": 10, "hike": 5},
    "postcutoff_n12": {"cut": 3, "hold": 9, "hike": 0},
}


class DecisionComparisonError(RuntimeError):
    """A frozen input, leakage, model, or publication contract failed."""


@dataclass(frozen=True)
class RuntimePaths:
    output_root: Path
    macro_root: Path
    action_roster: Path
    features: Path
    predictions: Path
    summary: Path
    comparison_csv: Path
    manifest: Path
    status: Path


def utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def canonical_json(value: Any) -> str:
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DecisionComparisonError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DecisionComparisonError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DecisionComparisonError(f"cannot read JSONL {path}: {exc}") from exc
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DecisionComparisonError(
                f"malformed JSONL {path}:{number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise DecisionComparisonError(f"non-object JSONL row {path}:{number}")
        rows.append(row)
    return rows


def _atomic_write(path: Path, content: bytes, *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(f"create-only output already exists: {path}")
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            try:
                os.link(temporary_path, path)
            except FileExistsError:
                raise FileExistsError(f"create-only output already exists: {path}")
        else:
            os.replace(temporary_path, path)
        if exclusive:
            temporary_path.unlink()
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_json(path: Path, value: Mapping[str, Any], *, resume: bool = False) -> None:
    content = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode()
    if path.exists() and resume:
        if path.is_symlink() or not path.is_file():
            raise DecisionComparisonError(f"unsafe resume output: {path}")
        if path.read_bytes() != content:
            raise DecisionComparisonError(f"resume output drift: {path}")
        return
    _atomic_write(path, content, exclusive=True)


def write_jsonl(
    path: Path, rows: Iterable[Mapping[str, Any]], *, resume: bool = False
) -> int:
    values = [dict(row) for row in rows]
    content = "".join(canonical_json(row) + "\n" for row in values).encode("utf-8")
    if path.exists() and resume:
        if path.is_symlink() or not path.is_file():
            raise DecisionComparisonError(f"unsafe resume output: {path}")
        if path.read_bytes() != content:
            raise DecisionComparisonError(f"resume output drift: {path}")
        return len(values)
    _atomic_write(path, content, exclusive=True)
    return len(values)


def write_status(
    path: Path, *, phase: str, state: str, detail: Mapping[str, Any] | None = None
) -> None:
    payload = {
        "schema_version": "decision-comparison-status-v1",
        "updated_at_utc": utc_now(),
        "phase": phase,
        "state": state,
        "detail": dict(detail or {}),
    }
    _atomic_write(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        exclusive=False,
    )


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") not in {
        "decision-comparison-baselines-config-v1",
        "decision-comparison-baselines-config-v2",
        "decision-comparison-baselines-config-v3",
    }:
        raise DecisionComparisonError("unsupported decision comparison config schema")
    if config.get("scoring") != {
        "bootstrap_draws": 10000,
        "bootstrap_seed": 20260827,
        "log_loss_epsilon": LOG_LOSS_EPSILON,
        "pooled_panels_allowed": False,
    }:
        raise DecisionComparisonError("frozen scoring configuration drift")
    return config


def runtime_paths(config: Mapping[str, Any], output_root: Path | None) -> RuntimePaths:
    root = resolve_path(output_root or str(config["default_output_root"]))
    return RuntimePaths(
        output_root=root,
        macro_root=root / "sources/macro",
        action_roster=root / "action_roster.jsonl",
        features=root / "features.jsonl",
        predictions=root / "predictions.jsonl",
        summary=root / "summary.json",
        comparison_csv=root / "comparison.csv",
        manifest=root / "manifest.json",
        status=root / "status.json",
    )


def _canonical_date(value: Any, *, label: str) -> str:
    text = str(value or "").strip()[:10]
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise DecisionComparisonError(f"invalid {label}: {value!r}") from exc
    if parsed.isoformat() != text:
        raise DecisionComparisonError(f"non-canonical {label}: {value!r}")
    return text


def _validate_action(
    direction: Any, magnitude: Any, *, context: str
) -> tuple[str, int]:
    direction_text = str(direction)
    if direction_text not in DIRECTIONS:
        raise DecisionComparisonError(f"invalid direction in {context}: {direction!r}")
    if isinstance(magnitude, bool):
        raise DecisionComparisonError(f"boolean magnitude in {context}")
    try:
        magnitude_int = int(magnitude)
    except (TypeError, ValueError) as exc:
        raise DecisionComparisonError(
            f"invalid magnitude in {context}: {magnitude!r}"
        ) from exc
    if magnitude_int not in {0, 25, 50, 75, 100}:
        raise DecisionComparisonError(
            f"illegal magnitude in {context}: {magnitude_int}"
        )
    if (direction_text == "hold") != (magnitude_int == 0):
        raise DecisionComparisonError(f"direction/magnitude mismatch in {context}")
    return direction_text, magnitude_int


def _official_roster_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path in config["official_rosters"]:
        path = resolve_path(raw_path)
        for raw in read_jsonl(path):
            meeting_id = str(raw.get("meeting_id") or "")
            start = _canonical_date(
                raw.get("meeting_start_date"), label="meeting_start_date"
            )
            end = _canonical_date(raw.get("meeting_end_date"), label="meeting_end_date")
            cutoff = _canonical_date(
                raw.get("evidence_cutoff"), label="evidence_cutoff"
            )
            if (
                not meeting_id
                or meeting_id in seen
                or not cutoff < start <= end
                or date.fromisoformat(cutoff)
                != date.fromisoformat(start) - timedelta(days=1)
            ):
                raise DecisionComparisonError(
                    f"invalid or duplicate official roster row: {meeting_id}"
                )
            rows.append(
                {
                    "meeting_id": meeting_id,
                    "meeting_start_date": start,
                    "meeting_end_date": end,
                    "evidence_cutoff": cutoff,
                    "meeting_type": str(raw.get("meeting_type") or "regular"),
                    "scheduled": bool(raw.get("scheduled", True)),
                    "official_minutes_url": raw.get("official_minutes_url"),
                    "roster_source_path": str(path.relative_to(ROOT)),
                    "roster_source_sha256": sha256_file(path),
                }
            )
            seen.add(meeting_id)
    rows.sort(key=lambda row: (row["meeting_start_date"], row["meeting_id"]))
    if len(rows) != 256:
        raise DecisionComparisonError(
            f"official 1993-2025 roster drift: {len(rows)} != 256"
        )
    return rows


def _external_panel_rows(config: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for panel, raw_path in config["external_panels"].items():
        path = resolve_path(raw_path)
        payload = read_json(path)
        meetings = payload.get("meetings")
        if not isinstance(meetings, list):
            raise DecisionComparisonError(f"panel meetings missing: {path}")
        rows: list[dict[str, Any]] = []
        for index, raw_value in enumerate(meetings):
            if not isinstance(raw_value, dict):
                raise DecisionComparisonError(f"non-object panel row {path}:{index}")
            raw = dict(raw_value)
            start = _canonical_date(
                raw.get("meeting_start_date"), label=f"{panel} start"
            )
            end = _canonical_date(
                raw.get("meeting_end_date", raw.get("decision_date")),
                label=f"{panel} end",
            )
            cutoff = _canonical_date(
                raw.get("evidence_cutoff"), label=f"{panel} cutoff"
            )
            if date.fromisoformat(cutoff) != date.fromisoformat(start) - timedelta(
                days=1
            ):
                raise DecisionComparisonError(
                    f"{panel} evidence cutoff is not start-date D-1: {start}"
                )
            direction, magnitude = _validate_action(
                raw.get("direction"),
                raw.get("magnitude_bp"),
                context=f"{panel}:{start}",
            )
            meeting_id = str(
                raw.get("meeting_id")
                or f"fomc-{start.replace('-', '')}-{end.replace('-', '')}"
            )
            rows.append(
                {
                    **raw,
                    "meeting_id": meeting_id,
                    "meeting_start_date": start,
                    "meeting_end_date": end,
                    "evidence_cutoff": cutoff,
                    "direction": direction,
                    "magnitude_bp": magnitude,
                    "panel": panel,
                    "panel_source_path": str(path.relative_to(ROOT)),
                    "panel_source_sha256": sha256_file(path),
                }
            )
        rows.sort(key=lambda row: row["meeting_start_date"])
        expected = EXPECTED_PANEL_COUNTS.get(panel)
        if expected is None or len(rows) != expected:
            raise DecisionComparisonError(
                f"panel count drift for {panel}: {len(rows)} != {expected}"
            )
        result[panel] = rows
    if set(result) != set(EXPECTED_PANEL_COUNTS):
        raise DecisionComparisonError(f"external panel set drift: {sorted(result)}")
    return result


def build_action_roster(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the canonical 268-meeting, label-complete policy chronology."""

    official = _official_roster_rows(config)
    panels = _external_panel_rows(config)
    by_id = {row["meeting_id"]: dict(row) for row in official}
    date_index: dict[str, set[str]] = defaultdict(set)
    for row in official:
        date_index[row["meeting_start_date"]].add(row["meeting_id"])
        date_index[row["meeting_end_date"]].add(row["meeting_id"])

    label_by_id: dict[str, dict[str, Any]] = {}
    release = resolve_path(config["decision_release"])
    unique_root = release / "manifests/unique"
    for split, expected in EXPECTED_UNIQUE_COUNTS.items():
        path = unique_root / f"{split}.jsonl"
        rows = read_jsonl(path)
        if len(rows) != expected:
            raise DecisionComparisonError(
                f"unique {split} count drift: {len(rows)} != {expected}"
            )
        if split == "train" and Counter(
            str(row.get("direction")) for row in rows
        ) != Counter(EXPECTED_TRAIN_CLASSES):
            raise DecisionComparisonError("unique train direction counts drift")
        for raw in rows:
            stored_date = _canonical_date(
                raw.get("meeting_date"), label=f"{split} meeting_date"
            )
            candidates = date_index.get(stored_date, set())
            if len(candidates) != 1:
                raise DecisionComparisonError(
                    f"unique {split} date maps to {len(candidates)} official meetings: {stored_date}"
                )
            meeting_id = next(iter(candidates))
            if meeting_id in label_by_id:
                raise DecisionComparisonError(
                    f"duplicate internal label meeting: {meeting_id}"
                )
            direction, magnitude = _validate_action(
                raw.get("direction"),
                raw.get("magnitude_bp"),
                context=f"{split}:{meeting_id}",
            )
            label_by_id[meeting_id] = {
                "direction": direction,
                "magnitude_bp": magnitude,
                "role": split,
                "sample_id": raw.get("sample_id"),
                "stored_meeting_date": stored_date,
                "label_source_path": str(path.relative_to(ROOT)),
                "label_source_sha256": sha256_file(path),
            }

    if len(label_by_id) != 237:
        raise DecisionComparisonError(
            f"internal unique label closure drift: {len(label_by_id)}"
        )

    for raw in panels["historical_n19"]:
        meeting_id = raw["meeting_id"]
        if meeting_id not in by_id or meeting_id in label_by_id:
            raise DecisionComparisonError(
                f"historical external identity overlap/drift: {meeting_id}"
            )
        label_by_id[meeting_id] = {
            "direction": raw["direction"],
            "magnitude_bp": raw["magnitude_bp"],
            "role": "historical_n19",
            "sample_id": f"decision-comparison::{meeting_id}",
            "stored_meeting_date": raw["meeting_start_date"],
            "label_source_path": raw["panel_source_path"],
            "label_source_sha256": raw["panel_source_sha256"],
        }

    if set(label_by_id) != set(by_id):
        missing = sorted(set(by_id) - set(label_by_id))
        extra = sorted(set(label_by_id) - set(by_id))
        raise DecisionComparisonError(
            f"1993-2025 label closure failed; missing={missing}, extra={extra}"
        )

    rows: list[dict[str, Any]] = []
    for meeting_id, base in by_id.items():
        label = label_by_id[meeting_id]
        role = str(label["role"])
        rows.append(
            {
                **base,
                **label,
                "panel": role if role in EXPECTED_PANEL_COUNTS else None,
            }
        )

    for raw in panels["postcutoff_n12"]:
        if raw["meeting_id"] in by_id:
            raise DecisionComparisonError(
                f"postcutoff meeting overlaps official 1993-2025 roster: {raw['meeting_id']}"
            )
        rows.append(
            {
                "meeting_id": raw["meeting_id"],
                "meeting_start_date": raw["meeting_start_date"],
                "meeting_end_date": raw["meeting_end_date"],
                "evidence_cutoff": raw["evidence_cutoff"],
                "meeting_type": "regular",
                "scheduled": True,
                "official_minutes_url": raw.get("official_statement_url"),
                "roster_source_path": raw["panel_source_path"],
                "roster_source_sha256": raw["panel_source_sha256"],
                "direction": raw["direction"],
                "magnitude_bp": raw["magnitude_bp"],
                "role": "postcutoff_n12",
                "sample_id": f"decision-comparison::{raw['meeting_id']}",
                "stored_meeting_date": raw["meeting_start_date"],
                "label_source_path": raw["panel_source_path"],
                "label_source_sha256": raw["panel_source_sha256"],
                "panel": "postcutoff_n12",
            }
        )

    rows.sort(key=lambda row: (row["meeting_start_date"], row["meeting_id"]))
    if len(rows) != 268 or len({row["meeting_id"] for row in rows}) != 268:
        raise DecisionComparisonError(
            "canonical action roster must contain 268 unique meetings"
        )
    for index, row in enumerate(rows):
        previous = rows[index - 1] if index else None
        row["lag1_meeting_id"] = previous["meeting_id"] if previous else None
        row["lag1_direction"] = previous["direction"] if previous else None
        row["lag1_magnitude_bp"] = previous["magnitude_bp"] if previous else None
        row["row_sha256"] = sha256_bytes(
            canonical_json({k: v for k, v in row.items() if k != "row_sha256"}).encode()
        )

    counts = Counter(row["role"] for row in rows)
    expected_roles = {
        "train": 211,
        "validation": 13,
        "test": 13,
        "historical_n19": 19,
        "postcutoff_n12": 12,
    }
    if counts != Counter(expected_roles):
        raise DecisionComparisonError(f"canonical role counts drift: {dict(counts)}")
    for panel, expected_classes in EXPECTED_PANEL_CLASSES.items():
        observed_classes = Counter(
            str(row["direction"]) for row in rows if row.get("panel") == panel
        )
        if observed_classes != Counter(expected_classes):
            raise DecisionComparisonError(
                f"canonical {panel} direction counts drift: {dict(observed_classes)}"
            )
    return rows


def _subtract_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def _http_get(
    url: str,
    *,
    timeout_seconds: float,
    retries: int,
    backoff_initial_seconds: float,
    backoff_max_seconds: float,
) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(
        url,
        headers={
            # ALFRED's graph endpoint can hang or terminate its HTTP/2 stream
            # when application/zip is advertised for a CSV URL.  Request the
            # only representation this acquisition contract accepts.
            "Accept": "text/csv",
        },
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                status = int(getattr(response, "status", 200))
                if status != 200:
                    raise DecisionComparisonError(f"HTTP {status} for macro source")
                body = response.read()
                if not body:
                    raise DecisionComparisonError("empty macro-source HTTP response")
                return body, {
                    str(key).lower(): str(value)
                    for key, value in response.headers.items()
                }
        except (urllib.error.URLError, TimeoutError, DecisionComparisonError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                delay = min(backoff_initial_seconds * (2**attempt), backoff_max_seconds)
                print(
                    canonical_json(
                        {
                            "event": "macro_http_retry",
                            "attempt": attempt + 1,
                            "attempts_allowed": retries,
                            "error_type": type(exc).__name__,
                            "next_delay_seconds": delay,
                        }
                    ),
                    flush=True,
                )
                time.sleep(delay)
    raise DecisionComparisonError(
        f"macro-source request failed after {retries} attempts: {last_error}"
    )


def _parse_csv_bytes(body: bytes, *, context: str) -> tuple[list[str], list[list[str]]]:
    try:
        rows = list(csv.reader(io.StringIO(body.decode("utf-8-sig"), newline="")))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise DecisionComparisonError(
            f"invalid CSV response for {context}: {exc}"
        ) from exc
    if not rows or not rows[0] or rows[0][0] != "observation_date":
        raise DecisionComparisonError(f"missing observation_date header for {context}")
    width = len(rows[0])
    data = [row for row in rows[1:] if row and any(cell.strip() for cell in row)]
    if any(len(row) != width for row in data):
        raise DecisionComparisonError(f"ragged CSV response for {context}")
    return rows[0], data


def _normalise_alfred_response(
    body: bytes, *, expected_columns: Sequence[str]
) -> bytes:
    expected = set(expected_columns)
    members: list[bytes]
    if body[:2] == b"PK":
        try:
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                infos = archive.infolist()
                if not 1 <= len(infos) <= 32:
                    raise DecisionComparisonError("unsafe ALFRED ZIP member count")
                if any(
                    info.is_dir()
                    or "/" in info.filename
                    or "\\" in info.filename
                    or info.file_size > 64 * 1024 * 1024
                    for info in infos
                ):
                    raise DecisionComparisonError("unsafe ALFRED ZIP member")
                members = [
                    archive.read(info)
                    for info in infos
                    if info.filename.lower().endswith(".csv")
                ]
        except (OSError, zipfile.BadZipFile) as exc:
            raise DecisionComparisonError(f"malformed ALFRED ZIP: {exc}") from exc
        if not members:
            raise DecisionComparisonError("ALFRED ZIP contains no CSV")
    else:
        members = [body]

    by_column: dict[str, dict[str, str]] = {}
    all_dates: set[str] = set()
    for index, member in enumerate(members):
        header, rows = _parse_csv_bytes(member, context=f"ALFRED member {index}")
        columns = header[1:]
        if (
            not columns
            or not set(columns).issubset(expected)
            or any(column in by_column for column in columns)
        ):
            raise DecisionComparisonError(
                f"ALFRED vintage-column partition drift: {columns}"
            )
        for column in columns:
            by_column[column] = {}
        for row in rows:
            observation = _canonical_date(row[0], label="ALFRED observation_date")
            all_dates.add(observation)
            for position, column in enumerate(columns, 1):
                value = row[position].strip()
                if value:
                    try:
                        numeric = float(value)
                    except ValueError as exc:
                        raise DecisionComparisonError(
                            f"non-numeric ALFRED value {value!r}"
                        ) from exc
                    if not math.isfinite(numeric):
                        raise DecisionComparisonError("non-finite ALFRED value")
                    by_column[column][observation] = value
    if set(by_column) != expected:
        raise DecisionComparisonError(
            f"ALFRED response columns do not close; missing={sorted(expected - set(by_column))}"
        )
    if any(not by_column[column] for column in expected_columns):
        raise DecisionComparisonError("ALFRED returned an empty requested vintage")

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["observation_date", *expected_columns])
    for observation in sorted(all_dates):
        writer.writerow(
            [
                observation,
                *(
                    by_column[column].get(observation, "")
                    for column in expected_columns
                ),
            ]
        )
    return output.getvalue().encode("utf-8")


def _validate_fred_csv(body: bytes, *, series_id: str) -> bytes:
    header, rows = _parse_csv_bytes(body, context=f"FRED {series_id}")
    if header != ["observation_date", series_id] or not rows:
        raise DecisionComparisonError(f"FRED {series_id} header/row contract drift")
    previous: str | None = None
    numeric_count = 0
    for row in rows:
        observation = _canonical_date(
            row[0], label=f"FRED {series_id} observation_date"
        )
        if previous is not None and observation <= previous:
            raise DecisionComparisonError(
                f"FRED {series_id} dates are not strictly increasing"
            )
        previous = observation
        value = row[1].strip()
        if value not in {"", "."}:
            try:
                numeric = float(value)
            except ValueError as exc:
                raise DecisionComparisonError(
                    f"non-numeric FRED {series_id} value"
                ) from exc
            if not math.isfinite(numeric):
                raise DecisionComparisonError(f"non-finite FRED {series_id} value")
            numeric_count += 1
    if numeric_count == 0:
        raise DecisionComparisonError(f"FRED {series_id} contains no numeric values")
    return body


def _macro_acquisition_contract(
    config: Mapping[str, Any], roster: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    cutoffs = sorted({str(row["evidence_cutoff"]) for row in roster})
    return {
        "source_config_sha256": sha256_bytes(
            canonical_json(config["macro_sources"]).encode()
        ),
        "roster_sha256": sha256_bytes(canonical_json(list(roster)).encode()),
        "meeting_count": len(roster),
        "evidence_cutoffs": cutoffs,
        "evidence_cutoffs_sha256": sha256_bytes(canonical_json(cutoffs).encode()),
    }


def validate_macro_manifest(
    macro_root: Path,
    manifest: Mapping[str, Any],
    *,
    expected_contract: Mapping[str, Any] | None = None,
) -> None:
    if (
        manifest.get("schema_version") != "decision-comparison-macro-sources-v1"
        or manifest.get("status") != "complete"
        or manifest.get("credentials_used") is not False
    ):
        raise DecisionComparisonError("macro source manifest schema drift")
    if (
        expected_contract is not None
        and manifest.get("acquisition_contract") != expected_contract
    ):
        raise DecisionComparisonError(
            "macro acquisition contract differs from current config/roster"
        )
    if int(manifest.get("meeting_count", -1)) <= 0:
        raise DecisionComparisonError("macro source manifest meeting count is invalid")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise DecisionComparisonError("macro source manifest contains no files")
    seen_paths: set[str] = set()
    alfred_bindings: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    alfred_vintages: defaultdict[str, list[str]] = defaultdict(list)
    fred_series: Counter[str] = Counter()
    fred_target_series: Counter[str] = Counter()
    for record in records:
        if not isinstance(record, dict):
            raise DecisionComparisonError("invalid macro file record")
        relative_path = str(record.get("path") or "")
        if not relative_path or relative_path in seen_paths:
            raise DecisionComparisonError("duplicate/empty macro file path binding")
        seen_paths.add(relative_path)
        bound_path = macro_root / relative_path
        path = bound_path.resolve()
        if (
            bound_path.is_symlink()
            or macro_root.resolve() not in path.parents
            or not path.is_file()
        ):
            raise DecisionComparisonError(f"bound macro file missing or unsafe: {path}")
        if path.stat().st_size != int(record.get("bytes", -1)) or sha256_file(
            path
        ) != record.get("sha256"):
            raise DecisionComparisonError(f"bound macro file hash drift: {path}")
        provider = record.get("provider")
        series_id = str(record.get("series_id") or "")
        if provider == "ALFRED":
            request_id = str(record.get("request_id") or "")
            kind = str(record.get("kind") or "")
            if (
                series_id not in {"UNRATE", "GDPC1"}
                or not request_id
                or kind
                not in {
                    "raw_response",
                    "normalized_csv",
                }
            ):
                raise DecisionComparisonError("invalid ALFRED file binding")
            alfred_bindings[(series_id, request_id)].add(kind)
            if kind == "normalized_csv":
                vintages = record.get("vintage_dates")
                if not isinstance(vintages, list) or not vintages:
                    raise DecisionComparisonError(
                        "ALFRED normalized file lacks vintages"
                    )
                alfred_vintages[series_id].extend(str(value) for value in vintages)
        elif provider == "FRED/H15":
            request_id = str(record.get("request_id") or "")
            if (
                record.get("kind") != "current_vintage_csv"
                or series_id not in {"DTB6", "DFF"}
                or not request_id
                or request_id not in relative_path.split("/")
            ):
                raise DecisionComparisonError("invalid FRED/H15 file binding")
            fred_series[series_id] += 1
        elif provider == "FRED":
            request_id = str(record.get("request_id") or "")
            if (
                record.get("kind") != "current_vintage_csv"
                or series_id != "DFEDTAR"
                or not request_id
                or request_id not in relative_path.split("/")
            ):
                raise DecisionComparisonError("invalid FRED target file binding")
            fred_target_series[series_id] += 1
        else:
            raise DecisionComparisonError(f"unexpected macro provider: {provider}")
    if any(
        kinds != {"raw_response", "normalized_csv"}
        for kinds in alfred_bindings.values()
    ):
        raise DecisionComparisonError(
            "ALFRED raw/normalized request pair closure failed"
        )
    if set(alfred_vintages) != {"UNRATE", "GDPC1"} or fred_series != Counter(
        {"DTB6": 1, "DFF": 1}
    ):
        raise DecisionComparisonError("macro source series closure failed")
    if fred_target_series not in (Counter(), Counter({"DFEDTAR": 1})):
        raise DecisionComparisonError("FRED target source series closure failed")
    if expected_contract is not None:
        expected_cutoffs = list(expected_contract["evidence_cutoffs"])
        for series_id in ("UNRATE", "GDPC1"):
            if sorted(alfred_vintages[series_id]) != expected_cutoffs:
                raise DecisionComparisonError(
                    f"ALFRED {series_id} vintage population drift"
                )


def acquire_macro_sources(
    config: Mapping[str, Any],
    roster: Sequence[Mapping[str, Any]],
    macro_root: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    """Acquire and seal the four sources required by the Kauppi-style feature set."""

    manifest_path = macro_root / "manifest.json"
    acquisition_contract = _macro_acquisition_contract(config, roster)
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"macro source root is already sealed: {macro_root}")
        manifest = read_json(manifest_path)
        validate_macro_manifest(
            macro_root, manifest, expected_contract=acquisition_contract
        )
        return manifest

    macro_root.mkdir(parents=True, exist_ok=True)
    evidence_cutoffs = sorted({str(row["evidence_cutoff"]) for row in roster})
    source_config = config["macro_sources"]
    try:
        http_options = {
            "timeout_seconds": float(source_config["http"]["timeout_seconds"]),
            "retries": int(source_config["http"]["retries"]),
            "backoff_initial_seconds": float(
                source_config["http"]["backoff_initial_seconds"]
            ),
            "backoff_max_seconds": float(source_config["http"]["backoff_max_seconds"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise DecisionComparisonError(
            "invalid frozen macro HTTP configuration"
        ) from exc
    if (
        http_options["timeout_seconds"] <= 0
        or http_options["retries"] < 1
        or http_options["backoff_initial_seconds"] < 0
        or http_options["backoff_max_seconds"] < http_options["backoff_initial_seconds"]
    ):
        raise DecisionComparisonError("invalid frozen macro HTTP configuration")
    file_records: list[dict[str, Any]] = []
    request_records: list[dict[str, Any]] = []

    alfred = source_config["alfred"]
    chunk_size = int(alfred["max_vintages_per_request"])
    if chunk_size < 1:
        raise DecisionComparisonError("ALFRED vintage request size must be positive")
    lookback_years = int(alfred["lookback_years"])
    target_source = source_config.get("fred_target_current_vintage")
    total_requests = len(alfred["series"]) * math.ceil(
        len(evidence_cutoffs) / chunk_size
    ) + len(source_config["fred_h15_current_vintage"]["series"]) + int(
        target_source is not None
    )
    completed_requests = 0
    for series_id in alfred["series"]:
        for chunk_index in range(0, len(evidence_cutoffs), chunk_size):
            vintages = evidence_cutoffs[chunk_index : chunk_index + chunk_size]
            starts = [
                _subtract_years(date.fromisoformat(vintage), lookback_years).isoformat()
                for vintage in vintages
            ]
            params = {
                "id": ",".join([str(series_id)] * len(vintages)),
                "cosd": ",".join(starts),
                "coed": ",".join(vintages),
                "vintage_date": ",".join(vintages),
            }
            url = f"{alfred['base_url']}?{urllib.parse.urlencode(params)}"
            request_id = sha256_bytes(url.encode())
            relative = Path("alfred") / str(series_id) / request_id
            raw_path = macro_root / relative / "response.bin"
            normalized_path = macro_root / relative / "normalized.csv"
            expected_columns = [
                f"{series_id}_{vintage.replace('-', '')}" for vintage in vintages
            ]
            response_content_type: str | None = None
            if raw_path.exists() or normalized_path.exists():
                if (
                    not resume
                    or not raw_path.is_file()
                    or raw_path.is_symlink()
                    or normalized_path.is_symlink()
                ):
                    raise DecisionComparisonError(
                        f"partial/existing macro request without valid resume: {relative}"
                    )
                raw_body = raw_path.read_bytes()
                normalized = _normalise_alfred_response(
                    raw_body, expected_columns=expected_columns
                )
                if not normalized_path.exists():
                    _atomic_write(normalized_path, normalized, exclusive=True)
                elif (
                    not normalized_path.is_file()
                    or normalized_path.read_bytes() != normalized
                ):
                    raise DecisionComparisonError(
                        f"cached ALFRED normalization drift: {relative}"
                    )
            else:
                raw_body, headers = _http_get(url, **http_options)
                normalized = _normalise_alfred_response(
                    raw_body, expected_columns=expected_columns
                )
                _atomic_write(raw_path, raw_body, exclusive=True)
                _atomic_write(normalized_path, normalized, exclusive=True)
                response_content_type = headers.get("content-type")
            for path, kind in (
                (raw_path, "raw_response"),
                (normalized_path, "normalized_csv"),
            ):
                file_records.append(
                    {
                        "path": str(path.relative_to(macro_root)),
                        "kind": kind,
                        "provider": "ALFRED",
                        "series_id": str(series_id),
                        "request_id": request_id,
                        "vintage_dates": vintages,
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
            request_records.append(
                {
                    "request_id": request_id,
                    "provider": "ALFRED",
                    "series_id": str(series_id),
                    "vintage_dates": vintages,
                    "canonical_url": url,
                    "response_content_type": response_content_type,
                }
            )
            completed_requests += 1
            print(
                canonical_json(
                    {
                        "event": "macro_request_complete",
                        "provider": "ALFRED",
                        "series_id": str(series_id),
                        "completed_requests": completed_requests,
                        "total_requests": total_requests,
                    }
                ),
                flush=True,
            )

    fred = source_config["fred_h15_current_vintage"]
    start = min("1988-01-01", evidence_cutoffs[0])
    end = evidence_cutoffs[-1]
    for series_id in fred["series"]:
        params = {"id": str(series_id), "cosd": start, "coed": end}
        url = f"{fred['base_url']}?{urllib.parse.urlencode(params)}"
        request_id = sha256_bytes(url.encode())
        path = (
            macro_root
            / "fred_h15_current_vintage"
            / str(series_id)
            / request_id
            / "response.csv"
        )
        if path.exists():
            if not resume:
                raise DecisionComparisonError(
                    f"existing FRED source without resume: {path}"
                )
            body = _validate_fred_csv(path.read_bytes(), series_id=str(series_id))
        else:
            body, _headers = _http_get(url, **http_options)
            body = _validate_fred_csv(body, series_id=str(series_id))
            _atomic_write(path, body, exclusive=True)
        file_records.append(
            {
                "path": str(path.relative_to(macro_root)),
                "kind": "current_vintage_csv",
                "provider": "FRED/H15",
                "series_id": str(series_id),
                "request_id": request_id,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        request_records.append(
            {
                "provider": "FRED/H15",
                "series_id": str(series_id),
                "canonical_url": url,
                "revision_caveat": fred["revision_caveat"],
            }
        )
        completed_requests += 1
        print(
            canonical_json(
                {
                    "event": "macro_request_complete",
                    "provider": "FRED/H15",
                    "series_id": str(series_id),
                    "completed_requests": completed_requests,
                    "total_requests": total_requests,
                }
            ),
            flush=True,
        )

    if target_source is not None:
        series_id = str(target_source["series"])
        if series_id != "DFEDTAR":
            raise DecisionComparisonError("frozen FRED target series drift")
        target_start = _canonical_date(
            target_source["observation_start_date"],
            label="FRED target observation_start_date",
        )
        target_end = _canonical_date(
            target_source["observation_end_date"],
            label="FRED target observation_end_date",
        )
        if target_start != "1982-09-27" or target_end != "2008-12-15":
            raise DecisionComparisonError("frozen FRED target date range drift")
        params = {"id": series_id, "cosd": target_start, "coed": target_end}
        url = f"{target_source['base_url']}?{urllib.parse.urlencode(params)}"
        request_id = sha256_bytes(url.encode())
        path = (
            macro_root
            / "fred_target_current_vintage"
            / series_id
            / request_id
            / "response.csv"
        )
        if path.exists():
            if not resume:
                raise DecisionComparisonError(
                    f"existing FRED target source without resume: {path}"
                )
            body = _validate_fred_csv(path.read_bytes(), series_id=series_id)
        else:
            body, _headers = _http_get(url, **http_options)
            body = _validate_fred_csv(body, series_id=series_id)
            _atomic_write(path, body, exclusive=True)
        file_records.append(
            {
                "path": str(path.relative_to(macro_root)),
                "kind": "current_vintage_csv",
                "provider": "FRED",
                "series_id": series_id,
                "request_id": request_id,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        request_records.append(
            {
                "provider": "FRED",
                "series_id": series_id,
                "canonical_url": url,
                "revision_caveat": target_source["revision_caveat"],
            }
        )
        completed_requests += 1
        print(
            canonical_json(
                {
                    "event": "macro_request_complete",
                    "provider": "FRED",
                    "series_id": series_id,
                    "completed_requests": completed_requests,
                    "total_requests": total_requests,
                }
            ),
            flush=True,
        )

    manifest = {
        "schema_version": "decision-comparison-macro-sources-v1",
        "status": "complete",
        "created_at_utc": utc_now(),
        "meeting_count": len(roster),
        "acquisition_contract": acquisition_contract,
        "evidence_cutoff_range": [evidence_cutoffs[0], evidence_cutoffs[-1]],
        "information_policy": {
            "alfred": "requested vintage equals official meeting-start D-1 evidence cutoff",
            "fred_h15_current_vintage": "latest observation no later than evidence cutoff minus one calendar day",
            "fred_target_current_vintage": (
                "current-vintage historical target levels used only through the prior completed Hamilton-Jorda week"
                if target_source is not None
                else None
            ),
            "same_day_data": "excluded",
            "runtime_fallback": "forbidden",
        },
        "http_transport": source_config["http"],
        "requests": request_records,
        "files": file_records,
        "credentials_used": False,
    }
    write_json(manifest_path, manifest)
    validate_macro_manifest(
        macro_root, manifest, expected_contract=acquisition_contract
    )
    return manifest


def _read_numeric_csv(path: Path, value_column: str) -> list[tuple[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["observation_date", value_column]:
            raise DecisionComparisonError(f"unexpected source CSV columns in {path}")
        rows: list[tuple[str, float]] = []
        for raw in reader:
            observation = _canonical_date(
                raw["observation_date"], label=f"{path} observation"
            )
            value = str(raw.get(value_column) or "").strip()
            if value in {"", "."}:
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                raise DecisionComparisonError(f"non-finite source value in {path}")
            rows.append((observation, numeric))
    if not rows:
        raise DecisionComparisonError(f"no numeric source observations in {path}")
    return rows


def load_macro_source_values(
    macro_root: Path, manifest: Mapping[str, Any]
) -> tuple[
    dict[tuple[str, str], list[tuple[str, float, str]]],
    dict[str, list[tuple[str, float, str]]],
]:
    validate_macro_manifest(macro_root, manifest)
    alfred_values: dict[tuple[str, str], list[tuple[str, float, str]]] = {}
    fred_values: dict[str, list[tuple[str, float, str]]] = {}
    normalized_records = [
        record
        for record in manifest["files"]
        if record.get("provider") == "ALFRED" and record.get("kind") == "normalized_csv"
    ]
    for record in normalized_records:
        path = macro_root / str(record["path"])
        series_id = str(record["series_id"])
        header, rows = _parse_csv_bytes(path.read_bytes(), context=str(path))
        for position, column in enumerate(header[1:], 1):
            prefix = f"{series_id}_"
            if not column.startswith(prefix) or len(column) != len(prefix) + 8:
                raise DecisionComparisonError(
                    f"unexpected ALFRED normalized column: {column}"
                )
            compact = column[len(prefix) :]
            vintage = f"{compact[:4]}-{compact[4:6]}-{compact[6:]}"
            key = (series_id, vintage)
            if key in alfred_values:
                raise DecisionComparisonError(f"duplicate ALFRED series/vintage: {key}")
            values: list[tuple[str, float, str]] = []
            for row in rows:
                raw_value = row[position].strip()
                if not raw_value:
                    continue
                observation = _canonical_date(row[0], label="ALFRED observation")
                if observation > vintage:
                    raise DecisionComparisonError(
                        f"future observation in ALFRED vintage {key}: {observation}"
                    )
                values.append((observation, float(raw_value), str(record["sha256"])))
            if not values:
                raise DecisionComparisonError(f"empty ALFRED series/vintage: {key}")
            alfred_values[key] = values

    for record in manifest["files"]:
        if record.get("provider") not in {"FRED/H15", "FRED"} or record.get(
            "kind"
        ) != "current_vintage_csv":
            continue
        series_id = str(record["series_id"])
        path = macro_root / str(record["path"])
        fred_values[series_id] = [
            (observation, value, str(record["sha256"]))
            for observation, value in _read_numeric_csv(path, series_id)
        ]
    if not {"DTB6", "DFF"}.issubset(fred_values) or set(fred_values) not in (
        {"DTB6", "DFF"},
        {"DTB6", "DFF", "DFEDTAR"},
    ):
        raise DecisionComparisonError(
            f"FRED source closure drift: {sorted(fred_values)}"
        )
    return alfred_values, fred_values


def _latest(
    values: Sequence[tuple[str, float, str]], cutoff: str, *, count: int = 1
) -> list[tuple[str, float, str]]:
    eligible = [row for row in values if row[0] <= cutoff]
    if len(eligible) < count:
        raise DecisionComparisonError(
            f"only {len(eligible)} observations available by {cutoff}; need {count}"
        )
    return eligible[-count:]


def _hamilton_jorda_week_end(value: str | date) -> date:
    """Return the Wednesday ending the Hamilton--Jorda Thursday--Wednesday week."""

    current = date.fromisoformat(value) if isinstance(value, str) else value
    return current + timedelta(days=(2 - current.weekday()) % 7)


def _weekly_mean(
    values: Sequence[tuple[str, float, str]],
    *,
    start: date,
    end: date,
    availability_cutoff: date,
    series_id: str,
) -> tuple[float, str, str, str]:
    """Average a series over its own cutoff-available weekday observations.

    FRED's daily DFF history repeats values on weekends whereas DTB6 is a
    business-day series. Hamilton and Jorda's revised weekly spread averages
    the two legs separately, so calendar-weekend DFF repeats must not receive
    additional weight and the legs must not be inner-joined by date.
    """
    eligible = [
        row
        for row in values
        if start <= date.fromisoformat(row[0]) <= end
        and date.fromisoformat(row[0]) <= availability_cutoff
        and date.fromisoformat(row[0]).weekday() < 5
    ]
    if not eligible:
        raise DecisionComparisonError(
            f"no {series_id} observations in Hamilton-Jorda prior-week window "
            f"{start.isoformat()}..{end.isoformat()} available by "
            f"{availability_cutoff.isoformat()}"
        )
    hashes = {row[2] for row in eligible}
    if len(hashes) != 1:
        raise DecisionComparisonError(
            f"multiple {series_id} hashes in Hamilton-Jorda weekly mean"
        )
    return (
        statistics.fmean(row[1] for row in eligible),
        eligible[0][0],
        eligible[-1][0],
        next(iter(hashes)),
    )


def build_hamilton_jorda_state(
    meeting: Mapping[str, Any],
    fred: Mapping[str, Sequence[tuple[str, float, str]]],
    *,
    market_lag_days: int,
) -> dict[str, Any]:
    """Construct a cutoff-safe weekly target-clock state for fixed HJ coefficients."""

    target_values = fred.get("DFEDTAR")
    if not target_values:
        return {
            "coverage_status": "unavailable",
            "unavailable_reason": "DFEDTAR_target_history_not_acquired",
        }
    meeting_start = date.fromisoformat(str(meeting["meeting_start_date"]))
    evidence_cutoff = date.fromisoformat(str(meeting["evidence_cutoff"]))
    last_target_observation = date.fromisoformat(target_values[-1][0])
    if evidence_cutoff > last_target_observation:
        return {
            "coverage_status": "unavailable",
            "unavailable_reason": (
                "DFEDTAR_single_target_history_ends_before_evidence_cutoff:"
                f"{last_target_observation.isoformat()}"
            ),
            "target_history_last_observation_date": last_target_observation.isoformat(),
            "source_hashes": {"DFEDTAR": target_values[-1][2]},
        }

    current_week_end = _hamilton_jorda_week_end(meeting_start)
    prior_week_end = current_week_end - timedelta(days=7)
    prior_week_start = prior_week_end - timedelta(days=6)
    eligible_targets = [
        row for row in target_values if date.fromisoformat(row[0]) <= evidence_cutoff
    ]
    if not eligible_targets:
        raise DecisionComparisonError("no DFEDTAR history available by evidence cutoff")
    first_target_day = date.fromisoformat(eligible_targets[0][0])
    first_week_end = _hamilton_jorda_week_end(first_target_day)
    weekly_targets: list[tuple[date, float]] = []
    week_end = first_week_end
    position = 0
    latest: tuple[str, float, str] | None = None
    while week_end <= prior_week_end:
        while position < len(eligible_targets) and date.fromisoformat(
            eligible_targets[position][0]
        ) <= week_end:
            latest = eligible_targets[position]
            position += 1
        if latest is not None:
            weekly_targets.append((week_end, float(latest[1])))
        week_end += timedelta(days=7)
    events: list[dict[str, Any]] = []
    for previous, current in zip(weekly_targets, weekly_targets[1:], strict=False):
        change = current[1] - previous[1]
        if not math.isclose(change, 0.0, abs_tol=1e-10):
            events.append(
                {
                    "week_end": current[0],
                    "change_pp": change,
                    "target_pp": current[1],
                }
            )
    if len(events) < 2:
        return {
            "coverage_status": "unavailable",
            "unavailable_reason": "fewer_than_two_prior_DFEDTAR_change_weeks",
            "source_hashes": {"DFEDTAR": eligible_targets[-1][2]},
        }
    previous_event, latest_event = events[-2], events[-1]
    duration_weeks = (latest_event["week_end"] - previous_event["week_end"]).days / 7
    if duration_weeks <= 0 or not float(duration_weeks).is_integer():
        raise DecisionComparisonError("invalid Hamilton-Jorda target-change duration")

    market_cutoff = evidence_cutoff - timedelta(days=market_lag_days)
    dff = _weekly_mean(
        fred["DFF"],
        start=prior_week_start,
        end=prior_week_end,
        availability_cutoff=market_cutoff,
        series_id="DFF",
    )
    dtb6 = _weekly_mean(
        fred["DTB6"],
        start=prior_week_start,
        end=prior_week_end,
        availability_cutoff=market_cutoff,
        series_id="DTB6",
    )
    spread = dtb6[0] - dff[0]
    state = {
        "coverage_status": "available",
        "specification": "authors-revised-2001-weekly-ach01-op-published-coefficients",
        "current_week_end": current_week_end.isoformat(),
        "prior_week_start": prior_week_start.isoformat(),
        "prior_week_end": prior_week_end.isoformat(),
        "market_availability_cutoff": market_cutoff.isoformat(),
        "previous_change_week_end": latest_event["week_end"].isoformat(),
        "penultimate_change_week_end": previous_event["week_end"].isoformat(),
        "previous_change_pp": latest_event["change_pp"],
        "previous_duration_weeks": int(duration_weeks),
        "spread_pp": spread,
        "spread_definition": (
            "separate mean(DTB6)-mean(DFF) over each series' own weekday "
            "numeric observations in the prior Thu-Wed week using only "
            "cutoff-available observations"
        ),
        "spread_observation_dates": {
            "DTB6": [dtb6[1], dtb6[2]],
            "DFF": [dff[1], dff[2]],
        },
        "target_history_latest_used_date": min(
            prior_week_end, evidence_cutoff
        ).isoformat(),
        "source_hashes": {
            "DFEDTAR": eligible_targets[-1][2],
            "DTB6": dtb6[3],
            "DFF": dff[3],
        },
        "exact_literature_replication": False,
    }
    if (
        not latest_event["week_end"] <= prior_week_end
        or not previous_event["week_end"] < latest_event["week_end"]
        or not date.fromisoformat(dff[2]) <= market_cutoff
        or not date.fromisoformat(dtb6[2]) <= market_cutoff
        or not math.isfinite(spread)
    ):
        raise DecisionComparisonError("Hamilton-Jorda state information-set drift")
    return state


def build_feature_rows(
    config: Mapping[str, Any],
    roster: Sequence[Mapping[str, Any]],
    macro_root: Path,
) -> list[dict[str, Any]]:
    manifest = read_json(macro_root / "manifest.json")
    validate_macro_manifest(
        macro_root,
        manifest,
        expected_contract=_macro_acquisition_contract(config, roster),
    )
    alfred, fred = load_macro_source_values(macro_root, manifest)
    retrieval_timestamp = str(manifest.get("created_at_utc") or "")
    if not retrieval_timestamp:
        raise DecisionComparisonError("macro manifest has no retrieval timestamp")
    market_lag = int(
        config["macro_sources"]["fred_h15_current_vintage"][
            "observation_cutoff_calendar_days_before_evidence_cutoff"
        ]
    )
    rows: list[dict[str, Any]] = []
    for meeting in roster:
        cutoff = str(meeting["evidence_cutoff"])
        market_cutoff = (
            date.fromisoformat(cutoff) - timedelta(days=market_lag)
        ).isoformat()
        unrate = _latest(alfred[("UNRATE", cutoff)], cutoff)[0]
        gdp = _latest(alfred[("GDPC1", cutoff)], cutoff, count=2)
        if not gdp[0][0] < gdp[1][0] or gdp[0][1] <= 0 or gdp[1][1] <= 0:
            raise DecisionComparisonError(
                f"invalid GDPC1 pair for {meeting['meeting_id']}"
            )
        dtb6 = _latest(fred["DTB6"], market_cutoff)[0]
        dff = _latest(fred["DFF"], market_cutoff)[0]
        real_gdp_growth = ((gdp[1][1] / gdp[0][1]) ** 4 - 1.0) * 100.0
        features = {
            "tbill6m_minus_effr_pp": dtb6[1] - dff[1],
            "unemployment_rate_pct": unrate[1],
            "real_gdp_growth_annualized_pct": real_gdp_growth,
        }
        if any(not math.isfinite(float(value)) for value in features.values()):
            raise DecisionComparisonError(
                f"non-finite feature for {meeting['meeting_id']}"
            )
        observations = {
            "UNRATE": {
                "observation_date": unrate[0],
                "availability_as_of_date": cutoff,
                "value": unrate[1],
                "source_sha256": unrate[2],
                "source_interface": "alfred-graph-csv-vintage",
                "retrieved_at_utc": retrieval_timestamp,
            },
            "GDPC1": {
                "observation_dates": [gdp[0][0], gdp[1][0]],
                "availability_as_of_date": cutoff,
                "values": [gdp[0][1], gdp[1][1]],
                "source_sha256": gdp[1][2],
                "source_interface": "alfred-graph-csv-vintage",
                "retrieved_at_utc": retrieval_timestamp,
            },
            "DTB6": {
                "observation_date": dtb6[0],
                "availability_as_of_date": market_cutoff,
                "value": dtb6[1],
                "source_sha256": dtb6[2],
                "source_interface": "fred-graph-csv-current-vintage",
                "retrieved_at_utc": retrieval_timestamp,
            },
            "DFF": {
                "observation_date": dff[0],
                "availability_as_of_date": market_cutoff,
                "value": dff[1],
                "source_sha256": dff[2],
                "source_interface": "fred-graph-csv-current-vintage",
                "retrieved_at_utc": retrieval_timestamp,
            },
        }
        hamilton_jorda_state = build_hamilton_jorda_state(
            meeting,
            fred,
            market_lag_days=market_lag,
        )
        payload = {
            "schema_version": "decision-comparison-feature-row-v1",
            "meeting_id": meeting["meeting_id"],
            "meeting_start_date": meeting["meeting_start_date"],
            "meeting_end_date": meeting["meeting_end_date"],
            "evidence_cutoff": cutoff,
            "market_observation_cutoff": market_cutoff,
            "role": meeting["role"],
            "panel": meeting["panel"],
            "target_direction": meeting["direction"],
            "target_magnitude_bp": meeting["magnitude_bp"],
            "features": features,
            "observations": observations,
            "hamilton_jorda_state": hamilton_jorda_state,
            "source_manifest_sha256": sha256_file(macro_root / "manifest.json"),
        }
        payload["feature_sha256"] = sha256_bytes(canonical_json(payload).encode())
        rows.append(payload)
    if len(rows) != 268:
        raise DecisionComparisonError("feature row closure failed")
    return rows


def validate_feature_information_set(rows: Sequence[Mapping[str, Any]]) -> None:
    """Fail closed when a feature row contains same-meeting or future evidence."""

    seen: set[str] = set()
    for row in rows:
        meeting_id = str(row.get("meeting_id") or "")
        if not meeting_id or meeting_id in seen:
            raise DecisionComparisonError(
                f"duplicate/empty feature meeting_id: {meeting_id!r}"
            )
        seen.add(meeting_id)
        meeting_start = _canonical_date(
            row.get("meeting_start_date"), label="feature meeting_start_date"
        )
        cutoff = _canonical_date(
            row.get("evidence_cutoff"), label="feature evidence_cutoff"
        )
        market_cutoff = _canonical_date(
            row.get("market_observation_cutoff"),
            label="feature market_observation_cutoff",
        )
        if not market_cutoff < cutoff < meeting_start:
            raise DecisionComparisonError(f"feature cutoff leakage for {meeting_id}")
        observations = row.get("observations")
        if not isinstance(observations, Mapping) or set(observations) != {
            "UNRATE",
            "GDPC1",
            "DTB6",
            "DFF",
        }:
            raise DecisionComparisonError(
                f"feature observation closure drift for {meeting_id}"
            )
        for series_id, raw_record in observations.items():
            if not isinstance(raw_record, Mapping):
                raise DecisionComparisonError(
                    f"invalid {series_id} evidence record for {meeting_id}"
                )
            availability = _canonical_date(
                raw_record.get("availability_as_of_date"),
                label=f"{series_id} availability_as_of_date",
            )
            allowed = market_cutoff if series_id in {"DTB6", "DFF"} else cutoff
            if availability > allowed or availability >= meeting_start:
                raise DecisionComparisonError(
                    f"future {series_id} availability for {meeting_id}"
                )
            raw_dates = raw_record.get("observation_dates")
            if raw_dates is None:
                raw_dates = [raw_record.get("observation_date")]
            if not isinstance(raw_dates, list) or not raw_dates:
                raise DecisionComparisonError(
                    f"missing {series_id} observation date for {meeting_id}"
                )
            observation_dates = [
                _canonical_date(value, label=f"{series_id} observation_date")
                for value in raw_dates
            ]
            if any(
                observation > availability or observation >= meeting_start
                for observation in observation_dates
            ):
                raise DecisionComparisonError(
                    f"same-day/future {series_id} observation for {meeting_id}"
                )
            if not str(raw_record.get("source_sha256") or ""):
                raise DecisionComparisonError(
                    f"missing {series_id} source hash for {meeting_id}"
                )
            retrieval = str(raw_record.get("retrieved_at_utc") or "")
            try:
                datetime.fromisoformat(retrieval.replace("Z", "+00:00"))
            except ValueError as exc:
                raise DecisionComparisonError(
                    f"missing/invalid {series_id} retrieval timestamp for {meeting_id}"
                ) from exc
        features = row.get("features")
        if not isinstance(features, Mapping) or set(features) != {
            "tbill6m_minus_effr_pp",
            "unemployment_rate_pct",
            "real_gdp_growth_annualized_pct",
        }:
            raise DecisionComparisonError(
                f"feature vector closure drift for {meeting_id}"
            )
        if any(not math.isfinite(float(value)) for value in features.values()):
            raise DecisionComparisonError(f"non-finite feature vector for {meeting_id}")
        hj_state = row.get("hamilton_jorda_state")
        if not isinstance(hj_state, Mapping) or hj_state.get(
            "coverage_status"
        ) not in {"available", "unavailable"}:
            raise DecisionComparisonError(
                f"Hamilton-Jorda feature state missing for {meeting_id}"
            )
        if hj_state["coverage_status"] == "unavailable":
            if not str(hj_state.get("unavailable_reason") or ""):
                raise DecisionComparisonError(
                    f"Hamilton-Jorda unavailable reason missing for {meeting_id}"
                )
        else:
            prior_week_end = _canonical_date(
                hj_state.get("prior_week_end"), label="HJ prior_week_end"
            )
            previous_change = _canonical_date(
                hj_state.get("previous_change_week_end"),
                label="HJ previous_change_week_end",
            )
            penultimate_change = _canonical_date(
                hj_state.get("penultimate_change_week_end"),
                label="HJ penultimate_change_week_end",
            )
            hj_market_cutoff = _canonical_date(
                hj_state.get("market_availability_cutoff"),
                label="HJ market_availability_cutoff",
            )
            spread_dates = hj_state.get("spread_observation_dates")
            source_hashes = hj_state.get("source_hashes")
            if (
                not penultimate_change < previous_change <= prior_week_end
                or not prior_week_end < meeting_start
                or hj_market_cutoff != market_cutoff
                or not isinstance(spread_dates, Mapping)
                or set(spread_dates) != {"DTB6", "DFF"}
                or any(
                    not isinstance(values, list)
                    or len(values) != 2
                    or _canonical_date(values[-1], label="HJ spread observation")
                    > market_cutoff
                    for values in spread_dates.values()
                )
                or not isinstance(source_hashes, Mapping)
                or set(source_hashes) != {"DFEDTAR", "DTB6", "DFF"}
                or any(not str(value) for value in source_hashes.values())
                or not math.isfinite(float(hj_state.get("previous_change_pp")))
                or float(hj_state.get("previous_duration_weeks", 0)) <= 0
                or not math.isfinite(float(hj_state.get("spread_pp")))
            ):
                raise DecisionComparisonError(
                    f"Hamilton-Jorda information-set drift for {meeting_id}"
                )
        if row.get("feature_sha256"):
            expected_hash = sha256_bytes(
                canonical_json(
                    {
                        key: value
                        for key, value in row.items()
                        if key != "feature_sha256"
                    }
                ).encode()
            )
            if row["feature_sha256"] != expected_hash:
                raise DecisionComparisonError(
                    f"feature row hash drift for {meeting_id}"
                )


def _signed_change(direction: str, magnitude_bp: int) -> int:
    if direction == "cut":
        return -magnitude_bp
    if direction == "hike":
        return magnitude_bp
    return 0


def _direction_probabilities_from_actions(
    action_probabilities: Mapping[int, float], *, invalid_probability: float = 0.0
) -> dict[str, float]:
    probabilities = {
        "cut": sum(
            float(value) for move, value in action_probabilities.items() if move < 0
        ),
        "hold": float(action_probabilities.get(0, 0.0)),
        "hike": sum(
            float(value) for move, value in action_probabilities.items() if move > 0
        ),
        "invalid": float(invalid_probability),
    }
    if any(
        not math.isfinite(value) or value < -1e-12 for value in probabilities.values()
    ):
        raise DecisionComparisonError("invalid action-derived probabilities")
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-8):
        raise DecisionComparisonError(
            f"four-label probability sum drift: {probabilities}"
        )
    return {key: max(0.0, value) for key, value in probabilities.items()}


def _magnitude_probability_payload(
    action_probabilities: Mapping[int, float], *, invalid_probability: float = 0.0
) -> dict[str, float]:
    payload = {
        str(move): float(action_probabilities.get(move, 0.0))
        for move in SIGNED_MOVE_GRID
    }
    payload["invalid"] = float(invalid_probability)
    if not math.isclose(sum(payload.values()), 1.0, abs_tol=1e-8):
        raise DecisionComparisonError("magnitude probability sum drift")
    return payload


def _unique_mode(
    values: Mapping[Any, float], *, tolerance: float = 1e-12
) -> Any | None:
    if not values:
        return None
    largest = max(float(value) for value in values.values())
    winners = [
        key
        for key, value in values.items()
        if math.isclose(float(value), largest, abs_tol=tolerance)
    ]
    return winners[0] if len(winners) == 1 else None


def _prediction_base(
    meeting: Mapping[str, Any],
    *,
    model: str,
    comparison_contract: str,
    fit_contract: str,
) -> dict[str, Any]:
    return {
        "schema_version": PREDICTION_SCHEMA,
        "model": model,
        "model_display_name": MODEL_DISPLAYS[model],
        "panel": meeting["panel"],
        "comparison_contract": comparison_contract,
        "fit_contract": fit_contract,
        "sample_id": meeting.get("sample_id"),
        "meeting_id": meeting["meeting_id"],
        "meeting_start_date": meeting["meeting_start_date"],
        "meeting_end_date": meeting["meeting_end_date"],
        "evidence_cutoff": meeting["evidence_cutoff"],
        "target_direction": meeting["direction"],
        "target_magnitude_bp": int(meeting["magnitude_bp"]),
    }


def _available_prediction(
    meeting: Mapping[str, Any],
    *,
    model: str,
    comparison_contract: str,
    fit_contract: str,
    action_probabilities: Mapping[int, float],
    invalid_probability: float = 0.0,
    training_rows: int | None = None,
    training_class_counts: Mapping[str, int] | None = None,
    feature_sha256: str | None = None,
    source_hashes: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    actions = {
        int(move): float(probability)
        for move, probability in action_probabilities.items()
    }
    probabilities = _direction_probabilities_from_actions(
        actions, invalid_probability=invalid_probability
    )
    direction_winner = _unique_mode(probabilities)
    predicted_direction = direction_winner if direction_winner in DIRECTIONS else None
    direction_actions = {
        move: probability
        for move, probability in actions.items()
        if predicted_direction is not None
        and (
            (move < 0 and predicted_direction == "cut")
            or (move == 0 and predicted_direction == "hold")
            or (move > 0 and predicted_direction == "hike")
        )
    }
    signed_winner = _unique_mode(direction_actions)
    predicted_signed_bp = (
        int(signed_winner) if isinstance(signed_winner, (int, np.integer)) else None
    )
    valid_mass = sum(actions.values())
    expected_change = (
        sum(move * probability for move, probability in actions.items()) / valid_mass
        if valid_mass > 0
        else None
    )
    row = {
        **_prediction_base(
            meeting,
            model=model,
            comparison_contract=comparison_contract,
            fit_contract=fit_contract,
        ),
        "coverage_status": "available",
        "unavailable_reason": None,
        "direction_probabilities": probabilities,
        "magnitude_probabilities": _magnitude_probability_payload(
            actions, invalid_probability=invalid_probability
        ),
        "predicted_direction": predicted_direction,
        "predicted_magnitude_bp": abs(predicted_signed_bp)
        if predicted_signed_bp is not None
        else None,
        "predicted_signed_bp": predicted_signed_bp,
        "expected_change_bp": expected_change,
        "training_rows": training_rows,
        "training_class_counts": dict(training_class_counts)
        if training_class_counts is not None
        else None,
        "feature_sha256": feature_sha256,
        "source_hashes": dict(source_hashes or {}),
    }
    row.update(dict(extra or {}))
    return row


def _unavailable_prediction(
    meeting: Mapping[str, Any],
    *,
    model: str,
    comparison_contract: str,
    fit_contract: str,
    reason: str,
    training_rows: int | None = None,
    training_class_counts: Mapping[str, int] | None = None,
    feature_sha256: str | None = None,
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return {
        **_prediction_base(
            meeting,
            model=model,
            comparison_contract=comparison_contract,
            fit_contract=fit_contract,
        ),
        "coverage_status": "unavailable",
        "unavailable_reason": reason,
        "direction_probabilities": None,
        "magnitude_probabilities": None,
        "predicted_direction": None,
        "predicted_magnitude_bp": None,
        "predicted_signed_bp": None,
        "expected_change_bp": None,
        "training_rows": training_rows,
        "training_class_counts": dict(training_class_counts)
        if training_class_counts is not None
        else None,
        "feature_sha256": feature_sha256,
        "source_hashes": dict(source_hashes or {}),
        "generation_count": None,
        "generation_invalid_count": None,
    }


def _paper_sample_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        row.get(field)
        for field in (
            "sample_id",
            "panel",
            "meeting_id",
            "meeting_start_date",
            "meeting_end_date",
            "evidence_cutoff",
            "direction",
            "magnitude_bp",
            "prompt_sha256",
            "source_analysis_sha256",
        )
    )


def _validate_integrity_envelope(payload: Mapping[str, Any], *, label: str) -> None:
    integrity = payload.get("integrity")
    if not isinstance(integrity, Mapping):
        raise DecisionComparisonError(f"{label} integrity envelope missing")
    unsigned = dict(payload)
    unsigned.pop("integrity", None)
    if integrity.get("payload_sha256") != sha256_bytes(
        canonical_json(unsigned).encode()
    ):
        raise DecisionComparisonError(f"{label} integrity payload drift")


def _validate_paper_run_seal(root: Path, *, expected_rows: int) -> dict[str, str]:
    manifest_path = root / "evaluation_manifest.json"
    authorization_path = root / "formal_generation_authorization.json"
    receipt_path = root / "run/run_receipt.json"
    samples_path = root / "inputs/panel_samples.jsonl"
    results_path = root / "run/results.jsonl"
    for path in (
        manifest_path,
        authorization_path,
        receipt_path,
        samples_path,
        results_path,
    ):
        if not path.is_file() or path.is_symlink():
            raise DecisionComparisonError(
                f"paper run seal file missing or unsafe: {path}"
            )
    manifest = read_json(manifest_path)
    authorization = read_json(authorization_path)
    receipt = read_json(receipt_path)
    if not str(manifest.get("status") or "").startswith("prepared_pending_"):
        raise DecisionComparisonError("paper evaluation manifest status drift")
    if not str(authorization.get("status") or "").startswith("authorized_"):
        raise DecisionComparisonError("paper generation authorization status drift")
    if receipt.get("status") != "complete":
        raise DecisionComparisonError("paper run receipt is not complete")
    for label, payload in (
        ("paper evaluation manifest", manifest),
        ("paper generation authorization", authorization),
        ("paper run receipt", receipt),
    ):
        _validate_integrity_envelope(payload, label=label)
    bound_manifest = receipt.get("manifest")
    bound_authorization = receipt.get("authorization")
    bound_results = receipt.get("results")
    manifest_samples = manifest.get("inputs", {}).get("samples")
    bindings = (
        (bound_manifest, manifest_path, None, "receipt manifest"),
        (bound_authorization, authorization_path, None, "receipt authorization"),
        (bound_results, results_path, expected_rows, "receipt results"),
        (manifest_samples, samples_path, 31, "manifest samples"),
    )
    for record, path, rows, label in bindings:
        if (
            not isinstance(record, Mapping)
            or int(record.get("bytes", -1)) != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
            or (rows is not None and int(record.get("rows", -1)) != rows)
        ):
            raise DecisionComparisonError(f"paper {label} file binding drift")
    if (
        receipt.get("status") != "complete"
        or int(receipt.get("rows", -1)) != expected_rows
    ):
        raise DecisionComparisonError("paper run receipt population drift")
    return {
        "evaluation_manifest_sha256": sha256_file(manifest_path),
        "authorization_sha256": sha256_file(authorization_path),
        "run_receipt_sha256": sha256_file(receipt_path),
        "panel_samples_sha256": sha256_file(samples_path),
        "results_sha256": sha256_file(results_path),
    }


def aggregate_paper_predictions(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the sealed N31 identity then aggregate K=10 at meeting level."""

    roots = {name: resolve_path(path) for name, path in config["paper_runs"].items()}
    sample_paths = {
        name: root / "inputs/panel_samples.jsonl" for name, root in roots.items()
    }
    result_paths = {name: root / "run/results.jsonl" for name, root in roots.items()}
    seal_records = {
        "chk0": _validate_paper_run_seal(roots["chk0"], expected_rows=310),
        "chk4": _validate_paper_run_seal(roots["chk4"], expected_rows=930),
    }
    samples_by_run = {name: read_jsonl(path) for name, path in sample_paths.items()}
    if set(samples_by_run) != {"chk0", "chk4"}:
        raise DecisionComparisonError(
            "paper run configuration must contain chk0 and chk4"
        )
    for name, rows in samples_by_run.items():
        if len(rows) != 31 or Counter(str(row.get("panel")) for row in rows) != Counter(
            {"historical_n19": 19, "postcutoff_n12": 12}
        ):
            raise DecisionComparisonError(f"paper {name} sample closure drift")
        for panel, expected_classes in EXPECTED_PANEL_CLASSES.items():
            observed_classes = Counter(
                str(row.get("direction")) for row in rows if row.get("panel") == panel
            )
            if observed_classes != Counter(expected_classes):
                raise DecisionComparisonError(
                    f"paper {name}/{panel} target-class closure drift"
                )
        identities = [_paper_sample_identity(row) for row in rows]
        if len(set(identities)) != 31:
            raise DecisionComparisonError(f"paper {name} sample identity duplication")
    if sorted(_paper_sample_identity(row) for row in samples_by_run["chk0"]) != sorted(
        _paper_sample_identity(row) for row in samples_by_run["chk4"]
    ):
        raise DecisionComparisonError(
            "chk0/chk4 N31 sample, meeting, target, or prompt hash mismatch; merge forbidden"
        )

    samples = samples_by_run["chk4"]
    sample_by_id = {str(row["sample_id"]): row for row in samples}
    result_rows = read_jsonl(result_paths["chk0"]) + read_jsonl(result_paths["chk4"])
    if len(result_rows) != 1240:
        raise DecisionComparisonError(
            f"paper K=10 result closure drift: {len(result_rows)} != 1240"
        )
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        model = str(row.get("model_label") or "")
        sample_id = str(row.get("sample_id") or "")
        if model not in PAPER_MODEL_LABELS or sample_id not in sample_by_id:
            raise DecisionComparisonError(
                f"unexpected paper result identity: {(model, sample_id)}"
            )
        sample = sample_by_id[sample_id]
        expected_values = {
            "meeting_id": sample["meeting_id"],
            "meeting_start_date": sample["meeting_start_date"],
            "panel": sample["panel"],
            "target_direction": sample["direction"],
            "target_magnitude_bp": sample["magnitude_bp"],
            "prompt_sha256": sample["prompt_sha256"],
        }
        if any(row.get(field) != value for field, value in expected_values.items()):
            raise DecisionComparisonError(
                f"paper result/sample hash closure failed: {model}/{sample_id}"
            )
        groups[(model, sample_id)].append(row)
    expected_groups = {
        (model, str(sample["sample_id"]))
        for model in PAPER_MODEL_LABELS
        for sample in samples
    }
    if set(groups) != expected_groups:
        raise DecisionComparisonError("paper model/sample population closure drift")

    predictions: list[dict[str, Any]] = []
    source_hashes = {
        "panel_samples_sha256": sha256_file(sample_paths["chk4"]),
        "chk0_results_sha256": sha256_file(result_paths["chk0"]),
        "chk4_results_sha256": sha256_file(result_paths["chk4"]),
        "chk0_run_receipt_sha256": seal_records["chk0"]["run_receipt_sha256"],
        "chk4_run_receipt_sha256": seal_records["chk4"]["run_receipt_sha256"],
        "chk0_evaluation_manifest_sha256": seal_records["chk0"][
            "evaluation_manifest_sha256"
        ],
        "chk4_evaluation_manifest_sha256": seal_records["chk4"][
            "evaluation_manifest_sha256"
        ],
    }
    for model, sample_id in sorted(groups):
        values = sorted(
            groups[(model, sample_id)], key=lambda row: int(row.get("replicate_id", -1))
        )
        if [int(row.get("replicate_id", -1)) for row in values] != list(range(10)):
            raise DecisionComparisonError(
                f"paper K=10 replicate closure failed: {model}/{sample_id}"
            )
        action_counts: Counter[int] = Counter()
        invalid_count = 0
        strict_contract_invalid_count = 0
        for value in values:
            strict_valid = (
                value.get("status") == "valid"
                and value.get("contract_valid") is True
                and value.get("decision_domain_valid") is True
                and value.get("delivery_valid") is True
            )
            if not strict_valid:
                strict_contract_invalid_count += 1
            prediction = value.get("decision_prediction")
            if not isinstance(prediction, Mapping):
                invalid_count += 1
                continue
            try:
                direction, magnitude = _validate_action(
                    prediction.get("direction"),
                    prediction.get("magnitude_bp"),
                    context=f"paper generation {model}/{sample_id}",
                )
            except DecisionComparisonError:
                invalid_count += 1
                continue
            action_counts[_signed_change(direction, magnitude)] += 1
        action_probabilities = {
            move: count / 10.0 for move, count in action_counts.items()
        }
        sample = sample_by_id[sample_id]
        meeting = {
            "sample_id": sample_id,
            "panel": sample["panel"],
            "meeting_id": sample["meeting_id"],
            "meeting_start_date": sample["meeting_start_date"],
            "meeting_end_date": sample["meeting_end_date"],
            "evidence_cutoff": sample["evidence_cutoff"],
            "direction": sample["direction"],
            "magnitude_bp": sample["magnitude_bp"],
        }
        predictions.append(
            _available_prediction(
                meeting,
                model=model,
                comparison_contract="matched_roster",
                fit_contract="frozen_paper_checkpoint_k10",
                action_probabilities=action_probabilities,
                invalid_probability=invalid_count / 10.0,
                source_hashes=source_hashes,
                extra={
                    "prompt_sha256": sample["prompt_sha256"],
                    "source_analysis_sha256": sample["source_analysis_sha256"],
                    "generation_count": 10,
                    "generation_invalid_count": invalid_count,
                    "generation_invalid_rate": invalid_count / 10.0,
                    "generation_invalid_definition": (
                        "no legal extractable decision_prediction; mirrors the sealed stochastic "
                        "evaluator's meeting-level directional label"
                    ),
                    "strict_contract_invalid_count": strict_contract_invalid_count,
                    "strict_contract_invalid_rate": strict_contract_invalid_count
                    / 10.0,
                    "strict_contract_invalid_definition": (
                        "source status/contract/domain/delivery validity did not all pass"
                    ),
                    "invalid_is_coverage_failure": False,
                    "prediction_seed": None,
                    "prediction_seed_source": "frozen_paper_result_rows",
                },
            )
        )
    if len(predictions) != 124:
        raise DecisionComparisonError("paper meeting-level aggregation closure failed")
    return predictions


def _validate_model_configuration(config: Mapping[str, Any]) -> None:
    models = config["models"]
    expected_features = [
        "tbill6m_minus_effr_pp",
        "unemployment_rate_pct",
        "real_gdp_growth_annualized_pct",
    ]
    if (
        models.get("directions") != list(DIRECTIONS)
        or models.get("features") != expected_features
    ):
        raise DecisionComparisonError(
            "frozen model direction/feature configuration drift"
        )
    mnl = models.get("mnl", {})
    if mnl != {"C": 1.0, "max_iter": 5000, "solver": "lbfgs"}:
        raise DecisionComparisonError("frozen Kauppi MNL configuration drift")
    ordered = models.get("ordered_probit", {})
    if ordered != {"distribution": "probit", "max_iter": 5000, "method": "bfgs"}:
        raise DecisionComparisonError("frozen ordered-probit configuration drift")
    forest = models.get("ordinal_rf", {})
    if forest != {
        "n_estimators": 500,
        "min_samples_leaf": 5,
        "class_weight": "balanced_subsample",
        "max_features": "sqrt",
    }:
        raise DecisionComparisonError("frozen ordinal-RF configuration drift")
    hj = models.get("hamilton_jorda_ach_op", {})
    expected_hj = {
        "specification": "authors-revised-2001-weekly-ach01-ordered-probit-published-coefficients",
        "coefficient_source": "https://econweb.ucsd.edu/~jhamilto/jordec01.pdf",
        "replication_archive": "https://econweb.ucsd.edu/~jhamilto/jorda.zip",
        "replication_archive_sha256": "50e450c88d67c76b88c8170a8f31df8db81f43183b129de9b94c034fca4df108",
        "replication_archive_bytes": 3726720,
        "prior_week_spread_contract": (
            "separate weekday numeric means by series; DTB6 minus DFF; "
            "cutoff-available observations only"
        ),
        "published_parameter_estimation_windows": {
            "ach": "1989-11-30_to_2001-04-26",
            "ordered_probit_replication_code": (
                "event_observations_1_to_102_ending_1997-03-26"
            ),
            "ordered_probit_paper_table_label": "1984_to_2001",
        },
        "ach": {
            "lag_duration": model_utils.HJ_ACH_LAG_DURATION,
            "constant": model_utils.HJ_ACH_CONSTANT,
            "fomc_week": model_utils.HJ_ACH_FOMC_WEEK,
            "absolute_spread": model_utils.HJ_ACH_ABSOLUTE_SPREAD,
            "pasting_delta": model_utils.HJ_ACH_PASTING_DELTA,
            "pasting_epsilon": model_utils.HJ_ACH_PASTING_EPSILON,
        },
        "ordered_probit": {
            "previous_change": model_utils.HJ_OP_PREVIOUS_CHANGE,
            "spread": model_utils.HJ_OP_SPREAD,
            "thresholds": list(model_utils.HJ_OP_THRESHOLDS),
            "marks_bp": list(model_utils.HJ_OP_MARKS_BP),
        },
        "target_history_series": "DFEDTAR",
        "supported_through": "2008-12-15",
        "exact_literature_replication": False,
    }
    if hj != expected_hj:
        raise DecisionComparisonError(
            "frozen Hamilton-Jorda ACH-OP configuration drift"
        )
    window = models.get("expanding_window", {})
    if window != {"minimum_training_rows": 30, "minimum_rows_per_class": 3}:
        raise DecisionComparisonError("frozen expanding-window gate drift")
    futures = config.get("futures", {})
    if (
        futures.get("maximum_settlement_gap_calendar_days") != 4
        or futures.get("magnitude_grid_bp") != list(SIGNED_MOVE_GRID)
        or futures.get("risk_premium_adjustment") is not None
        or futures.get("probability_identification")
        != "linear interpolation between adjacent magnitude-grid points"
    ):
        raise DecisionComparisonError("frozen Fed Funds futures configuration drift")


def _feature_matrix(
    meetings: Sequence[Mapping[str, Any]],
    feature_by_meeting: Mapping[str, Mapping[str, Any]],
    feature_names: Sequence[str],
) -> np.ndarray:
    matrix: list[list[float]] = []
    for meeting in meetings:
        meeting_id = str(meeting["meeting_id"])
        feature_row = feature_by_meeting.get(meeting_id)
        if feature_row is None:
            raise DecisionComparisonError(f"feature missing for {meeting_id}")
        values = feature_row.get("features")
        if not isinstance(values, Mapping):
            raise DecisionComparisonError(f"feature mapping missing for {meeting_id}")
        matrix.append([float(values[name]) for name in feature_names])
    result = np.asarray(matrix, dtype=float)
    if (
        result.ndim != 2
        or result.shape != (len(meetings), len(feature_names))
        or not np.isfinite(result).all()
    ):
        raise DecisionComparisonError("feature matrix closure failed")
    return result


def _class_counts(meetings: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["direction"]) for row in meetings)
    return {direction: counts[direction] for direction in DIRECTIONS}


def expanding_training_rows(
    roster: Sequence[Mapping[str, Any]], target: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    """Return only actions whose decision ended before the target meeting began."""

    target_start = _canonical_date(
        target.get("meeting_start_date"), label="expanding target meeting_start_date"
    )
    training = [
        row
        for row in roster
        if _canonical_date(
            row.get("meeting_end_date"), label="training meeting_end_date"
        )
        < target_start
    ]
    training.sort(
        key=lambda row: (str(row["meeting_start_date"]), str(row["meeting_id"]))
    )
    if any(
        str(row["meeting_id"]) == str(target["meeting_id"])
        or str(row["meeting_end_date"]) >= target_start
        for row in training
    ):
        raise DecisionComparisonError(
            f"expanding-window leakage for {target['meeting_id']}"
        )
    return training


def _actions_from_direction_probabilities(
    probabilities: Sequence[float], training_meetings: Sequence[Mapping[str, Any]]
) -> dict[int, float]:
    matrix = model_utils.validate_probability_matrix(probabilities)
    if matrix.ndim != 1:
        raise DecisionComparisonError("expected one model probability row")
    conditional: dict[str, Counter[int]] = {
        direction: Counter() for direction in DIRECTIONS
    }
    for meeting in training_meetings:
        direction, magnitude = _validate_action(
            meeting["direction"],
            meeting["magnitude_bp"],
            context="training action distribution",
        )
        conditional[direction][_signed_change(direction, magnitude)] += 1
    actions: defaultdict[int, float] = defaultdict(float)
    for index, direction in enumerate(DIRECTIONS):
        total = sum(conditional[direction].values())
        if total == 0:
            raise DecisionComparisonError(
                f"training window has no {direction} magnitude distribution"
            )
        for move, count in conditional[direction].items():
            actions[move] += float(matrix[index]) * count / total
    if not math.isclose(sum(actions.values()), 1.0, abs_tol=1e-8):
        raise DecisionComparisonError("learned action probability allocation drift")
    return dict(actions)


def _fit_direction_model(
    model: str,
    x_train: np.ndarray,
    y_train: Sequence[str],
    x_test: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    if model == "kauppi_mnl":
        return model_utils.kauppi_mnl_predict_proba(x_train, y_train, x_test)
    if model == "ordered_probit":
        return model_utils.ordered_probit_predict_proba(x_train, y_train, x_test)
    if model == "ordinal_rf":
        return model_utils.cumulative_ordinal_rf_predict_proba(
            x_train, y_train, x_test, random_state=seed
        )
    raise DecisionComparisonError(f"not a learned baseline model: {model}")


def _read_gzip_csv(path: Path) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise DecisionComparisonError(f"missing gzip CSV header: {path}")
            return [dict(row) for row in reader]
    except (OSError, csv.Error) as exc:
        raise DecisionComparisonError(f"cannot read gzip CSV {path}: {exc}") from exc


def load_futures_source(wrds_root: Path) -> dict[str, Any]:
    """Load only the sealed, restricted root-level futures panel."""

    manifest_path = wrds_root / "manifest.json"
    info_path = wrds_root / "contract_info.csv.gz"
    prices_path = wrds_root / "contract_prices_daily.csv.gz"
    if not wrds_root.is_dir():
        return {
            "available": False,
            "reason": "wrds_root_missing",
            "root": str(wrds_root),
        }
    if any(
        path.is_symlink() or not path.is_file()
        for path in (manifest_path, info_path, prices_path)
    ):
        raise DecisionComparisonError(
            f"incomplete or unsafe sealed WRDS root: {wrds_root}"
        )
    try:
        from jobs.eval import download_wrds_fed_funds_futures as wrds_downloader

        wrds_spec = wrds_downloader.load_run_spec(wrds_root)
        manifest = wrds_downloader.validate_sealed_artifact(wrds_root, wrds_spec)
    except (OSError, RuntimeError, ValueError) as exc:
        raise DecisionComparisonError(
            f"WRDS v2 seal validation failed: {type(exc).__name__}: {exc}"
        ) from exc
    if (
        manifest.get("schema_version") != "wrds-ff-futures-contract-panel-v2"
        or manifest.get("status") != "complete"
    ):
        raise DecisionComparisonError("WRDS final manifest is not a complete v2 seal")
    redistribution = manifest.get("redistribution")
    if (
        not isinstance(redistribution, Mapping)
        or redistribution.get("classification") != "restricted_licensed_source_data"
    ):
        raise DecisionComparisonError("WRDS data is not marked restricted")
    if redistribution.get("raw_data_redistribution_permitted") is not False:
        raise DecisionComparisonError("WRDS redistribution prohibition is missing")
    credentials = manifest.get("credentials")
    if not isinstance(credentials, Mapping) or any(
        credentials.get(field) is not False
        for field in (
            "embedded",
            "pgpass_path_recorded",
            "username_recorded",
            "password_recorded",
        )
    ):
        raise DecisionComparisonError("WRDS credential redaction contract drift")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise DecisionComparisonError("WRDS file manifest missing")
    for path in (info_path, prices_path):
        record = files.get(path.name)
        if not isinstance(record, Mapping):
            raise DecisionComparisonError(
                f"WRDS file not bound by manifest: {path.name}"
            )
        if path.stat().st_size != int(record.get("bytes", -1)) or sha256_file(
            path
        ) != record.get("sha256"):
            raise DecisionComparisonError(f"WRDS file hash drift: {path.name}")

    metadata = _read_gzip_csv(info_path)
    prices = _read_gzip_csv(prices_path)
    contract_by_month: dict[str, str] = {}
    for row in metadata:
        try:
            product_matches = (
                int(str(row.get("contrcode") or "")) == 331
                and int(str(row.get("clscode") or "")) == 245
                and str(row.get("contrname") or "") == "30 DAY US FEDERAL FUNDS"
                and str(row.get("exchtickersymb") or "").upper() == "FF"
            )
        except ValueError as exc:
            raise DecisionComparisonError(
                "malformed WRDS FF contract product fields"
            ) from exc
        if not product_matches:
            raise DecisionComparisonError(
                "non-FF contract escaped the sealed product filter"
            )
        last_trade_date = str(row.get("lasttrddate") or "")[:10]
        try:
            last_trade_day = date.fromisoformat(last_trade_date)
            contract_month = last_trade_day.strftime("%Y-%m")
        except ValueError as exc:
            raise DecisionComparisonError(
                "FF contract has no canonical last-trade date"
            ) from exc
        futcode = str(row.get("futcode") or "")
        if not futcode:
            raise DecisionComparisonError("FF contract has an empty futcode")
        mnemonic = str(row.get("dsmnem") or "")
        compact = mnemonic[-4:]
        if (
            len(compact) != 4
            or not compact.isdigit()
            or int(compact[:2]) != last_trade_day.month
            or int(compact[2:]) != last_trade_day.year % 100
        ):
            raise DecisionComparisonError(
                f"FF contract mnemonic/month mismatch for futcode {futcode}"
            )
        previous = contract_by_month.setdefault(contract_month, futcode)
        if previous != futcode:
            raise DecisionComparisonError(
                f"multiple FF contracts for month {contract_month}"
            )
    settlements: defaultdict[str, list[tuple[str, float]]] = defaultdict(list)
    seen_keys: set[tuple[str, str]] = set()
    for row in prices:
        futcode = str(row.get("futcode") or "")
        observation = str(row.get("date_") or "")[:10]
        raw_settlement = str(row.get("settlement") or "").strip()
        if not futcode or not raw_settlement:
            continue
        observation = _canonical_date(observation, label="WRDS futures settlement date")
        key = (futcode, observation)
        if key in seen_keys:
            raise DecisionComparisonError(
                f"duplicate WRDS futures settlement key: {key}"
            )
        seen_keys.add(key)
        numeric = float(raw_settlement)
        if not math.isfinite(numeric):
            raise DecisionComparisonError(f"non-finite WRDS settlement: {key}")
        settlements[futcode].append((observation, numeric))
    unknown_price_contracts = set(settlements) - set(contract_by_month.values())
    if unknown_price_contracts:
        raise DecisionComparisonError(
            f"WRDS prices contain contracts absent from metadata: {sorted(unknown_price_contracts)}"
        )
    for values in settlements.values():
        values.sort()
    return {
        "available": True,
        "root": str(wrds_root),
        "contract_by_month": contract_by_month,
        "settlements": dict(settlements),
        "source_hashes": {
            "wrds_manifest_sha256": sha256_file(manifest_path),
            "contract_info_sha256": sha256_file(info_path),
            "contract_prices_daily_sha256": sha256_file(prices_path),
        },
        "restriction": "restricted_licensed_source_data",
    }


def _next_month(value: date) -> str:
    year = value.year + (1 if value.month == 12 else 0)
    month = 1 if value.month == 12 else value.month + 1
    return f"{year:04d}-{month:02d}"


def _futures_prediction(
    meeting: Mapping[str, Any],
    feature: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    comparison_contract: str,
    maximum_gap_days: int,
) -> dict[str, Any]:
    fit_contract = "frozen_wrds_futures_grid_v1"
    if not source.get("available"):
        return _unavailable_prediction(
            meeting,
            model="futures",
            comparison_contract=comparison_contract,
            fit_contract=fit_contract,
            reason=str(source.get("reason") or "wrds_source_unavailable"),
            feature_sha256=str(feature.get("feature_sha256") or ""),
        )
    decision_date = date.fromisoformat(str(meeting["meeting_end_date"]))
    days_in_month = calendar.monthrange(decision_date.year, decision_date.month)[1]
    month_end_case = decision_date.day == days_in_month
    contract_month = (
        _next_month(decision_date)
        if month_end_case
        else decision_date.strftime("%Y-%m")
    )
    futcode = source["contract_by_month"].get(contract_month)
    if futcode is None:
        return _unavailable_prediction(
            meeting,
            model="futures",
            comparison_contract=comparison_contract,
            fit_contract=fit_contract,
            reason=f"meeting_month_contract_missing:{contract_month}",
            feature_sha256=str(feature.get("feature_sha256") or ""),
            source_hashes=source.get("source_hashes"),
        )
    cutoff = str(meeting["evidence_cutoff"])
    eligible = [
        row for row in source["settlements"].get(futcode, []) if row[0] <= cutoff
    ]
    if not eligible:
        return _unavailable_prediction(
            meeting,
            model="futures",
            comparison_contract=comparison_contract,
            fit_contract=fit_contract,
            reason=f"settlement_missing_by_cutoff:{cutoff}",
            feature_sha256=str(feature.get("feature_sha256") or ""),
            source_hashes=source.get("source_hashes"),
        )
    settlement_date, settlement = eligible[-1]
    gap = (date.fromisoformat(cutoff) - date.fromisoformat(settlement_date)).days
    if gap > maximum_gap_days:
        return _unavailable_prediction(
            meeting,
            model="futures",
            comparison_contract=comparison_contract,
            fit_contract=fit_contract,
            reason=f"settlement_gap_exceeds_{maximum_gap_days}_days:{gap}",
            feature_sha256=str(feature.get("feature_sha256") or ""),
            source_hashes=source.get("source_hashes"),
        )
    dff = feature.get("observations", {}).get("DFF", {}).get("value")
    try:
        current_rate = float(dff)
    except (TypeError, ValueError) as exc:
        raise DecisionComparisonError(
            f"DFF feature unavailable for {meeting['meeting_id']}"
        ) from exc
    post_rate = (
        model_utils.fed_funds_futures_implied_rate(settlement)
        if month_end_case
        else model_utils.monthly_implied_post_rate(
            settlement, current_rate, decision_date.day, days_in_month
        )
    )
    expected_move = (post_rate - current_rate) * 100.0
    grid_probabilities = model_utils.interpolate_futures_move_probabilities(
        expected_move
    )
    actions = {
        int(move): float(probability)
        for move, probability in zip(
            model_utils.FUTURES_MOVE_GRID_BP, grid_probabilities, strict=True
        )
        if probability > 0
    }
    return _available_prediction(
        meeting,
        model="futures",
        comparison_contract=comparison_contract,
        fit_contract=fit_contract,
        action_probabilities=actions,
        feature_sha256=str(feature.get("feature_sha256") or ""),
        source_hashes=source.get("source_hashes"),
        extra={
            "contract_month": contract_month,
            "settlement_gap_calendar_days": gap,
            "month_end_next_contract": month_end_case,
            "risk_premium_adjustment": None,
            "probability_identification": "adjacent-node interpolation on fixed 25bp grid",
            "restricted_raw_quote_persisted": False,
        },
    )


def _hamilton_jorda_source_hashes(
    feature: Mapping[str, Any], state: Mapping[str, Any]
) -> dict[str, str]:
    """Collect the hashes binding the HJ state and its market spread inputs."""

    hashes: dict[str, str] = {}
    raw_hashes = state.get("source_hashes")
    if isinstance(raw_hashes, Mapping):
        hashes.update(
            {
                str(key): str(value)
                for key, value in raw_hashes.items()
                if str(key) and str(value or "")
            }
        )
    for key, value in state.items():
        if str(key).endswith("sha256") and str(value or ""):
            hashes[str(key)] = str(value)
    return hashes


def _hamilton_jorda_prediction(
    meeting: Mapping[str, Any],
    feature: Mapping[str, Any],
    *,
    comparison_contract: str,
) -> dict[str, Any]:
    """Apply the published final HJ ACH--OP parameters to a prepared state."""

    feature_sha256 = str(feature.get("feature_sha256") or "")
    state = feature.get("hamilton_jorda_state")
    state_mapping = state if isinstance(state, Mapping) else {}
    source_hashes = _hamilton_jorda_source_hashes(feature, state_mapping)
    if meeting.get("panel") == "postcutoff_n12":
        return _unavailable_prediction(
            meeting,
            model="hamilton_jorda_ach_op",
            comparison_contract=comparison_contract,
            fit_contract=HAMILTON_JORDA_FIT_CONTRACT,
            reason="published_hj_target_level_regime_not_applicable_post_2008",
            feature_sha256=feature_sha256,
            source_hashes=source_hashes,
        )
    if not isinstance(state, Mapping):
        return _unavailable_prediction(
            meeting,
            model="hamilton_jorda_ach_op",
            comparison_contract=comparison_contract,
            fit_contract=HAMILTON_JORDA_FIT_CONTRACT,
            reason="hamilton_jorda_state_missing",
            feature_sha256=feature_sha256,
            source_hashes=source_hashes,
        )
    state_status = str(
        state.get("coverage_status", state.get("status", "available"))
    )
    if state_status != "available":
        return _unavailable_prediction(
            meeting,
            model="hamilton_jorda_ach_op",
            comparison_contract=comparison_contract,
            fit_contract=HAMILTON_JORDA_FIT_CONTRACT,
            reason=str(
                state.get("unavailable_reason")
                or state.get("reason")
                or f"hamilton_jorda_state_{state_status}"
            ),
            feature_sha256=feature_sha256,
            source_hashes=source_hashes,
        )
    try:
        previous_change_pp = float(state["previous_change_pp"])
        previous_duration_weeks = float(state["previous_duration_weeks"])
        spread_pp = float(state["spread_pp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DecisionComparisonError(
            f"invalid Hamilton--Jord\u00e0 state for {meeting['meeting_id']}"
        ) from exc
    actions = model_utils.hamilton_jorda_ach_op_action_probabilities(
        previous_change_pp=previous_change_pp,
        previous_duration_weeks=previous_duration_weeks,
        spread_pp=spread_pp,
        fomc_meeting=True,
    )
    return _available_prediction(
        meeting,
        model="hamilton_jorda_ach_op",
        comparison_contract=comparison_contract,
        fit_contract=HAMILTON_JORDA_FIT_CONTRACT,
        action_probabilities=actions,
        feature_sha256=feature_sha256,
        source_hashes=source_hashes,
        extra={
            "published_parameter_application": True,
            "published_parameter_samples": {
                "ach": "1989-11-30_to_2001-04-26",
                "ordered_probit_replication_code": (
                    "event_observations_1_to_102_ending_1997-03-26"
                ),
                "ordered_probit_paper_table_label": "1984_to_2001",
            },
            "exact_literature_replication": False,
            "fomc_meeting_indicator": 1,
            "previous_change_pp": previous_change_pp,
            "previous_duration_weeks": previous_duration_weeks,
            "spread_pp": spread_pp,
            "previous_change_week_end": state.get("previous_change_week_end"),
            "penultimate_change_week_end": state.get(
                "penultimate_change_week_end"
            ),
            "prior_week_start": state.get("prior_week_start"),
            "prior_week_end": state.get("prior_week_end"),
        },
    )


def predict_baselines(
    config: Mapping[str, Any],
    roster: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    training_contracts: Sequence[str],
    models: Sequence[str],
    wrds_root: Path,
    seed: int,
) -> list[dict[str, Any]]:
    _validate_model_configuration(config)
    validate_feature_information_set(feature_rows)
    if any(contract not in FIT_CONTRACTS for contract in training_contracts):
        raise DecisionComparisonError(
            f"invalid training contract set: {training_contracts}"
        )
    if any(model not in BASELINE_MODELS for model in models):
        raise DecisionComparisonError(f"invalid baseline model set: {models}")
    if isinstance(seed, bool) or not 0 <= int(seed) < 2**32 - 1:
        raise DecisionComparisonError("seed must satisfy 0 <= seed < 2**32 - 1")
    feature_by_meeting = {str(row["meeting_id"]): row for row in feature_rows}
    if len(feature_by_meeting) != len(roster) or set(feature_by_meeting) != {
        str(row["meeting_id"]) for row in roster
    }:
        raise DecisionComparisonError("roster/feature meeting closure failed")
    targets = [row for row in roster if row.get("panel") in EXPECTED_PANEL_COUNTS]
    targets.sort(key=lambda row: (row["meeting_start_date"], row["meeting_id"]))
    if Counter(str(row["panel"]) for row in targets) != Counter(EXPECTED_PANEL_COUNTS):
        raise DecisionComparisonError("prediction target panel closure failed")
    feature_names = list(config["models"]["features"])
    predictions: list[dict[str, Any]] = []
    futures_source = (
        load_futures_source(wrds_root) if "futures" in models else {"available": False}
    )

    for contract in training_contracts:
        for meeting in targets:
            feature = feature_by_meeting[str(meeting["meeting_id"])]
            if "always_hold" in models:
                predictions.append(
                    _available_prediction(
                        meeting,
                        model="always_hold",
                        comparison_contract=contract,
                        fit_contract="no_fit_constant",
                        action_probabilities={0: 1.0},
                    )
                )
            if "lag1" in models:
                lag_direction = meeting.get("lag1_direction")
                lag_magnitude = meeting.get("lag1_magnitude_bp")
                if lag_direction is None or lag_magnitude is None:
                    predictions.append(
                        _unavailable_prediction(
                            meeting,
                            model="lag1",
                            comparison_contract=contract,
                            fit_contract="full_action_chronology_lag1",
                            reason="no_prior_policy_action",
                        )
                    )
                else:
                    direction, magnitude = _validate_action(
                        lag_direction,
                        lag_magnitude,
                        context=f"lag1 {meeting['meeting_id']}",
                    )
                    predictions.append(
                        _available_prediction(
                            meeting,
                            model="lag1",
                            comparison_contract=contract,
                            fit_contract="full_action_chronology_lag1",
                            action_probabilities={
                                _signed_change(direction, magnitude): 1.0
                            },
                            extra={"lag1_meeting_id": meeting.get("lag1_meeting_id")},
                        )
                    )
            if "hamilton_jorda_ach_op" in models:
                predictions.append(
                    _hamilton_jorda_prediction(
                        meeting,
                        feature,
                        comparison_contract=contract,
                    )
                )
            if "futures" in models:
                predictions.append(
                    _futures_prediction(
                        meeting,
                        feature,
                        futures_source,
                        comparison_contract=contract,
                        maximum_gap_days=int(
                            config["futures"]["maximum_settlement_gap_calendar_days"]
                        ),
                    )
                )

        learned = [
            model
            for model in models
            if model in {"kauppi_mnl", "ordered_probit", "ordinal_rf"}
        ]
        if not learned:
            continue
        if contract == "matched_roster":
            training = [row for row in roster if row["role"] == "train"]
            if (
                len(training) != 211
                or _class_counts(training) != EXPECTED_TRAIN_CLASSES
            ):
                raise DecisionComparisonError(
                    "matched-roster N211 training contract drift"
                )
            training.sort(
                key=lambda row: (row["meeting_start_date"], row["meeting_id"])
            )
            x_train = _feature_matrix(training, feature_by_meeting, feature_names)
            y_train = [str(row["direction"]) for row in training]
            x_test = _feature_matrix(targets, feature_by_meeting, feature_names)
            counts = _class_counts(training)
            for model in learned:
                try:
                    probabilities = _fit_direction_model(
                        model, x_train, y_train, x_test, seed=int(seed)
                    )
                except (
                    ImportError,
                    ValueError,
                    np.linalg.LinAlgError,
                    model_utils.ModelConvergenceError,
                ) as exc:
                    for meeting in targets:
                        predictions.append(
                            _unavailable_prediction(
                                meeting,
                                model=model,
                                comparison_contract=contract,
                                fit_contract="matched_unique_train_n211",
                                reason=f"model_fit_failed:{type(exc).__name__}:{exc}",
                                training_rows=len(training),
                                training_class_counts=counts,
                                feature_sha256=str(
                                    feature_by_meeting[str(meeting["meeting_id"])][
                                        "feature_sha256"
                                    ]
                                ),
                            )
                        )
                    continue
                for meeting, probability in zip(targets, probabilities, strict=True):
                    predictions.append(
                        _available_prediction(
                            meeting,
                            model=model,
                            comparison_contract=contract,
                            fit_contract="matched_unique_train_n211",
                            action_probabilities=_actions_from_direction_probabilities(
                                probability, training
                            ),
                            training_rows=len(training),
                            training_class_counts=counts,
                            feature_sha256=str(
                                feature_by_meeting[str(meeting["meeting_id"])][
                                    "feature_sha256"
                                ]
                            ),
                            extra={
                                "ordinal_approximation": model == "ordinal_rf",
                                "exact_literature_replication": False,
                            },
                        )
                    )
        else:
            for meeting in targets:
                training = expanding_training_rows(roster, meeting)
                labels = [str(row["direction"]) for row in training]
                eligibility = model_utils.expanding_window_eligibility(labels)
                counts = dict(eligibility.class_counts)
                for model in learned:
                    if not eligibility.eligible:
                        predictions.append(
                            _unavailable_prediction(
                                meeting,
                                model=model,
                                comparison_contract=contract,
                                fit_contract="strictly_prior_actions_expanding_window",
                                reason="; ".join(eligibility.reasons),
                                training_rows=len(training),
                                training_class_counts=counts,
                                feature_sha256=str(
                                    feature_by_meeting[str(meeting["meeting_id"])][
                                        "feature_sha256"
                                    ]
                                ),
                            )
                        )
                        continue
                    x_train = _feature_matrix(
                        training, feature_by_meeting, feature_names
                    )
                    x_test = _feature_matrix(
                        [meeting], feature_by_meeting, feature_names
                    )
                    try:
                        probability = _fit_direction_model(
                            model, x_train, labels, x_test, seed=int(seed)
                        )[0]
                    except (
                        ImportError,
                        ValueError,
                        np.linalg.LinAlgError,
                        model_utils.ModelConvergenceError,
                    ) as exc:
                        predictions.append(
                            _unavailable_prediction(
                                meeting,
                                model=model,
                                comparison_contract=contract,
                                fit_contract="strictly_prior_actions_expanding_window",
                                reason=f"model_fit_failed:{type(exc).__name__}:{exc}",
                                training_rows=len(training),
                                training_class_counts=counts,
                                feature_sha256=str(
                                    feature_by_meeting[str(meeting["meeting_id"])][
                                        "feature_sha256"
                                    ]
                                ),
                            )
                        )
                        continue
                    predictions.append(
                        _available_prediction(
                            meeting,
                            model=model,
                            comparison_contract=contract,
                            fit_contract="strictly_prior_actions_expanding_window",
                            action_probabilities=_actions_from_direction_probabilities(
                                probability, training
                            ),
                            training_rows=len(training),
                            training_class_counts=counts,
                            feature_sha256=str(
                                feature_by_meeting[str(meeting["meeting_id"])][
                                    "feature_sha256"
                                ]
                            ),
                            extra={
                                "ordinal_approximation": model == "ordinal_rf",
                                "exact_literature_replication": False,
                                "training_latest_meeting_end_date": training[-1][
                                    "meeting_end_date"
                                ],
                            },
                        )
                    )
    expected_rows = len(targets) * len(models) * len(training_contracts)
    if len(predictions) != expected_rows:
        raise DecisionComparisonError(
            f"baseline prediction row closure drift: {len(predictions)} != {expected_rows}"
        )
    keys = [
        (row["comparison_contract"], row["panel"], row["model"], row["meeting_id"])
        for row in predictions
    ]
    if len(set(keys)) != len(keys):
        raise DecisionComparisonError("duplicate baseline prediction key")
    for row in predictions:
        row["prediction_seed"] = int(seed)
        row["prediction_seed_source"] = "decision_comparison_cli"
    return predictions


def validate_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (
            str(row.get("comparison_contract")),
            str(row.get("panel")),
            str(row.get("model")),
            str(row.get("meeting_id")),
        )
        if key in seen:
            raise DecisionComparisonError(f"duplicate prediction key: {key}")
        seen.add(key)
        contract, panel, model, meeting_id = key
        if contract not in FIT_CONTRACTS or panel not in EXPECTED_PANEL_COUNTS:
            raise DecisionComparisonError(f"invalid prediction contract/panel: {key}")
        if model not in {*BASELINE_MODELS, *PAPER_MODEL_LABELS} or not meeting_id:
            raise DecisionComparisonError(f"invalid prediction model/meeting: {key}")
        start = _canonical_date(
            row.get("meeting_start_date"), label="prediction meeting_start_date"
        )
        end = _canonical_date(
            row.get("meeting_end_date"), label="prediction meeting_end_date"
        )
        cutoff = _canonical_date(
            row.get("evidence_cutoff"), label="prediction evidence_cutoff"
        )
        expected_fit_contract = {
            "always_hold": "no_fit_constant",
            "lag1": "full_action_chronology_lag1",
            "hamilton_jorda_ach_op": HAMILTON_JORDA_FIT_CONTRACT,
            "futures": "frozen_wrds_futures_grid_v1",
            "kauppi_mnl": "matched_unique_train_n211"
            if contract == "matched_roster"
            else "strictly_prior_actions_expanding_window",
            "ordered_probit": "matched_unique_train_n211"
            if contract == "matched_roster"
            else "strictly_prior_actions_expanding_window",
            "ordinal_rf": "matched_unique_train_n211"
            if contract == "matched_roster"
            else "strictly_prior_actions_expanding_window",
            **{
                paper_model: "frozen_paper_checkpoint_k10"
                for paper_model in PAPER_MODEL_LABELS
            },
        }[model]
        if (
            row.get("schema_version") != PREDICTION_SCHEMA
            or row.get("model_display_name") != MODEL_DISPLAYS[model]
            or row.get("fit_contract") != expected_fit_contract
            or not str(row.get("sample_id") or "")
            or not cutoff < start <= end
            or date.fromisoformat(cutoff)
            != date.fromisoformat(start) - timedelta(days=1)
            or not isinstance(row.get("source_hashes"), Mapping)
        ):
            raise DecisionComparisonError(
                f"prediction identity/provenance metadata drift: {key}"
            )
        _validate_action(
            row.get("target_direction"),
            row.get("target_magnitude_bp"),
            context=f"target {key}",
        )
        if model not in PAPER_MODEL_LABELS and any(
            row.get(field) not in {None, 0, 0.0}
            for field in (
                "generation_count",
                "generation_invalid_count",
                "generation_invalid_rate",
                "strict_contract_invalid_count",
                "strict_contract_invalid_rate",
            )
        ):
            raise DecisionComparisonError(
                f"non-LLM baseline carries generation accounting: {key}"
            )
        status = row.get("coverage_status")
        if model in PAPER_MODEL_LABELS and status != "available":
            raise DecisionComparisonError(
                f"paper generation aggregate cannot be unavailable: {key}"
            )
        if status == "unavailable":
            if not row.get("unavailable_reason"):
                raise DecisionComparisonError(
                    f"unavailable prediction lacks reason: {key}"
                )
            if any(
                row.get(field) is not None
                for field in (
                    "direction_probabilities",
                    "magnitude_probabilities",
                    "predicted_direction",
                    "predicted_magnitude_bp",
                    "predicted_signed_bp",
                    "expected_change_bp",
                )
            ):
                raise DecisionComparisonError(
                    f"unavailable prediction carries a prediction: {key}"
                )
            continue
        if status != "available" or row.get("unavailable_reason") is not None:
            raise DecisionComparisonError(f"invalid coverage state: {key}")
        probabilities = row.get("direction_probabilities")
        if not isinstance(probabilities, Mapping) or set(probabilities) != set(
            PROBABILITY_LABELS
        ):
            raise DecisionComparisonError(
                f"four-label probability closure drift: {key}"
            )
        values = [float(probabilities[label]) for label in PROBABILITY_LABELS]
        if any(
            not math.isfinite(value) or value < -1e-12 or value > 1 + 1e-12
            for value in values
        ):
            raise DecisionComparisonError(f"invalid four-label probability: {key}")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-8):
            raise DecisionComparisonError(f"four-label probability sum drift: {key}")
        magnitudes = row.get("magnitude_probabilities")
        expected_magnitude_keys = {str(move) for move in SIGNED_MOVE_GRID} | {"invalid"}
        if (
            not isinstance(magnitudes, Mapping)
            or set(magnitudes) != expected_magnitude_keys
        ):
            raise DecisionComparisonError(f"magnitude probability closure drift: {key}")
        magnitude_values = [
            float(magnitudes[label]) for label in expected_magnitude_keys
        ]
        if any(
            not math.isfinite(value) or value < -1e-12 or value > 1 + 1e-12
            for value in magnitude_values
        ) or not math.isclose(sum(magnitude_values), 1.0, abs_tol=1e-8):
            raise DecisionComparisonError(f"invalid magnitude probabilities: {key}")
        if not math.isclose(
            float(magnitudes["invalid"]), float(probabilities["invalid"]), abs_tol=1e-8
        ):
            raise DecisionComparisonError(
                f"direction/magnitude invalid mass drift: {key}"
            )
        derived_direction = {
            "cut": sum(
                float(magnitudes[str(move)]) for move in SIGNED_MOVE_GRID if move < 0
            ),
            "hold": float(magnitudes["0"]),
            "hike": sum(
                float(magnitudes[str(move)]) for move in SIGNED_MOVE_GRID if move > 0
            ),
            "invalid": float(magnitudes["invalid"]),
        }
        if any(
            not math.isclose(
                float(probabilities[label]), derived_direction[label], abs_tol=1e-8
            )
            for label in PROBABILITY_LABELS
        ):
            raise DecisionComparisonError(
                f"direction/magnitude probability mismatch: {key}"
            )
        expected_direction_winner = _unique_mode(
            {label: float(probabilities[label]) for label in PROBABILITY_LABELS}
        )
        expected_direction = (
            expected_direction_winner
            if expected_direction_winner in DIRECTIONS
            else None
        )
        predicted = row.get("predicted_direction")
        if predicted != expected_direction:
            raise DecisionComparisonError(
                f"point direction is not the unique four-label mode: {key}"
            )
        expected_signed: int | None = None
        if predicted is not None:
            eligible_moves = {
                move: float(magnitudes[str(move)])
                for move in SIGNED_MOVE_GRID
                if (move < 0 and predicted == "cut")
                or (move == 0 and predicted == "hold")
                or (move > 0 and predicted == "hike")
            }
            winner = _unique_mode(eligible_moves)
            expected_signed = (
                int(winner) if isinstance(winner, (int, np.integer)) else None
            )
        if row.get("predicted_signed_bp") != expected_signed or row.get(
            "predicted_magnitude_bp"
        ) != (abs(expected_signed) if expected_signed is not None else None):
            raise DecisionComparisonError(f"point direction/magnitude mismatch: {key}")
        valid_mass = 1.0 - float(probabilities["invalid"])
        expected_change = (
            sum(move * float(magnitudes[str(move)]) for move in SIGNED_MOVE_GRID)
            / valid_mass
            if valid_mass > 1e-12
            else None
        )
        observed_change = row.get("expected_change_bp")
        if (expected_change is None) != (observed_change is None) or (
            expected_change is not None
            and not math.isclose(float(observed_change), expected_change, abs_tol=1e-8)
        ):
            raise DecisionComparisonError(
                f"expected signed-bp probability mismatch: {key}"
            )
        if model in PAPER_MODEL_LABELS:
            if contract != "matched_roster" or row.get("generation_count") != 10:
                raise DecisionComparisonError(f"paper prediction contract drift: {key}")
            invalid_count = row.get("generation_invalid_count")
            strict_invalid_count = row.get("strict_contract_invalid_count")
            if (
                isinstance(invalid_count, bool)
                or not isinstance(invalid_count, int)
                or not 0 <= invalid_count <= 10
                or not math.isclose(
                    float(probabilities["invalid"]), invalid_count / 10.0, abs_tol=1e-8
                )
                or not math.isclose(
                    float(row.get("generation_invalid_rate", -1)),
                    invalid_count / 10.0,
                    abs_tol=1e-8,
                )
                or row.get("invalid_is_coverage_failure") is not False
                or not row.get("generation_invalid_definition")
                or isinstance(strict_invalid_count, bool)
                or not isinstance(strict_invalid_count, int)
                or not 0 <= strict_invalid_count <= 10
                or strict_invalid_count < invalid_count
                or not math.isclose(
                    float(row.get("strict_contract_invalid_rate", -1)),
                    strict_invalid_count / 10.0,
                    abs_tol=1e-8,
                )
                or not row.get("strict_contract_invalid_definition")
                or row.get("prediction_seed") is not None
                or row.get("prediction_seed_source") != "frozen_paper_result_rows"
                or not row.get("source_hashes")
            ):
                raise DecisionComparisonError(
                    f"paper invalid-generation accounting drift: {key}"
                )
        elif float(probabilities["invalid"]) != 0.0:
            raise DecisionComparisonError(
                f"non-LLM baseline has invalid-generation mass: {key}"
            )


def validate_selected_prediction_population(
    rows: Sequence[Mapping[str, Any]],
    *,
    training_contracts: Sequence[str],
    baseline_models: Sequence[str],
    prediction_seed: int | None = None,
) -> None:
    expected: Counter[tuple[str, str, str]] = Counter()
    for contract in training_contracts:
        for panel, meeting_count in EXPECTED_PANEL_COUNTS.items():
            for model in baseline_models:
                expected[(contract, panel, model)] = meeting_count
            if contract == "matched_roster":
                for model in PAPER_MODEL_LABELS:
                    expected[(contract, panel, model)] = meeting_count
    observed = Counter(
        (str(row["comparison_contract"]), str(row["panel"]), str(row["model"]))
        for row in rows
    )
    if observed != expected:
        missing = expected - observed
        extra = observed - expected
        raise DecisionComparisonError(
            "prediction population differs from requested CLI contract; "
            f"missing={dict(missing)}, extra={dict(extra)}"
        )
    if prediction_seed is not None:
        for row in rows:
            expected_seed = (
                None if row["model"] in PAPER_MODEL_LABELS else int(prediction_seed)
            )
            if row.get("prediction_seed") != expected_seed:
                raise DecisionComparisonError(
                    f"prediction seed drift for {row['model']}/{row['meeting_id']}"
                )


def prediction_run_provenance(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    paths: RuntimePaths,
    training_contracts: Sequence[str],
    baseline_models: Sequence[str],
    wrds_root: Path,
    seed: int,
) -> dict[str, Any]:
    wrds_manifest = wrds_root / "manifest.json"
    paper_sources = (
        {
            name: _validate_paper_run_seal(
                resolve_path(root), expected_rows=310 if name == "chk0" else 930
            )
            for name, root in config["paper_runs"].items()
        }
        if "matched_roster" in training_contracts
        else None
    )
    return {
        "schema_version": "decision-comparison-prediction-run-v1",
        "config_sha256": sha256_file(config_path),
        "implementation_sha256": sha256_file(IMPLEMENTATION),
        "model_utils_sha256": sha256_file(Path(model_utils.__file__).resolve()),
        "requirements_sha256": sha256_file(ROOT / "requirements.txt"),
        "action_roster_sha256": sha256_file(paths.action_roster),
        "features_sha256": sha256_file(paths.features),
        "macro_manifest_sha256": sha256_file(paths.macro_root / "manifest.json"),
        "training_contracts": list(training_contracts),
        "baseline_models": list(baseline_models),
        "paper_models": list(PAPER_MODEL_LABELS)
        if "matched_roster" in training_contracts
        else [],
        "paper_sources": paper_sources,
        "wrds_root": str(wrds_root),
        "wrds_manifest_sha256": sha256_file(wrds_manifest)
        if wrds_manifest.is_file()
        else None,
        "seed": int(seed),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": package_version("scikit-learn"),
            "statsmodels": package_version("statsmodels"),
        },
    }


def validate_prediction_run_provenance(
    rows: Sequence[Mapping[str, Any]], expected: Mapping[str, Any]
) -> None:
    observed_values = {
        canonical_json(row.get("prediction_run_provenance")) for row in rows
    }
    if observed_values != {canonical_json(expected)}:
        raise DecisionComparisonError(
            "prediction run provenance differs from current config, implementation, inputs, seed, or WRDS root"
        )


def validate_predictions_against_prepared_inputs(
    rows: Sequence[Mapping[str, Any]],
    roster: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
) -> None:
    targets = {
        str(row["meeting_id"]): row
        for row in roster
        if row.get("panel") in EXPECTED_PANEL_COUNTS
    }
    features = {str(row["meeting_id"]): row for row in feature_rows}
    if len(targets) != 31:
        raise DecisionComparisonError("prepared N31 target closure drift")
    learned = {"kauppi_mnl", "ordered_probit", "ordinal_rf"}
    for row in rows:
        meeting_id = str(row["meeting_id"])
        target = targets.get(meeting_id)
        if target is None:
            raise DecisionComparisonError(
                f"prediction meeting absent from prepared N31: {meeting_id}"
            )
        expected_fields = {
            "panel": target["panel"],
            "meeting_start_date": target["meeting_start_date"],
            "meeting_end_date": target["meeting_end_date"],
            "evidence_cutoff": target["evidence_cutoff"],
            "target_direction": target["direction"],
            "target_magnitude_bp": target["magnitude_bp"],
        }
        if any(row.get(field) != value for field, value in expected_fields.items()):
            raise DecisionComparisonError(
                f"prediction/prepared target drift: {meeting_id}"
            )
        model = str(row["model"])
        contract = str(row["comparison_contract"])
        if (
            row.get("schema_version") != PREDICTION_SCHEMA
            or row.get("model_display_name") != MODEL_DISPLAYS[model]
            or not str(row.get("fit_contract") or "")
        ):
            raise DecisionComparisonError(
                f"prediction schema/model metadata drift: {model}/{meeting_id}"
            )
        expected_fit = {
            "always_hold": "no_fit_constant",
            "lag1": "full_action_chronology_lag1",
            "hamilton_jorda_ach_op": HAMILTON_JORDA_FIT_CONTRACT,
            "futures": "frozen_wrds_futures_grid_v1",
            **{
                learned_model: (
                    "matched_unique_train_n211"
                    if contract == "matched_roster"
                    else "strictly_prior_actions_expanding_window"
                )
                for learned_model in learned
            },
            **{
                paper_model: "frozen_paper_checkpoint_k10"
                for paper_model in PAPER_MODEL_LABELS
            },
        }[model]
        if row["fit_contract"] != expected_fit:
            raise DecisionComparisonError(
                f"prediction fit contract drift: {model}/{meeting_id}"
            )
        if model not in PAPER_MODEL_LABELS and row.get("sample_id") != target.get(
            "sample_id"
        ):
            raise DecisionComparisonError(
                f"baseline sample identity drift: {model}/{meeting_id}"
            )
        expected_feature_hash = str(features[meeting_id]["feature_sha256"])
        if model in learned | {"futures", "hamilton_jorda_ach_op"}:
            if row.get("feature_sha256") != expected_feature_hash:
                raise DecisionComparisonError(
                    f"prediction feature hash drift: {model}/{meeting_id}"
                )
        elif row.get("feature_sha256") not in {None, ""}:
            raise DecisionComparisonError(
                f"unexpected prediction feature hash: {model}/{meeting_id}"
            )
        if model in learned:
            training = (
                [candidate for candidate in roster if candidate["role"] == "train"]
                if contract == "matched_roster"
                else expanding_training_rows(roster, target)
            )
            if row.get("training_rows") != len(training) or row.get(
                "training_class_counts"
            ) != _class_counts(training):
                raise DecisionComparisonError(
                    f"prediction training-window provenance drift: {model}/{meeting_id}"
                )
            if (
                contract == "expanding_window"
                and row.get("coverage_status") == "available"
                and row.get("training_latest_meeting_end_date")
                != training[-1]["meeting_end_date"]
            ):
                raise DecisionComparisonError(
                    f"expanding latest-label boundary drift: {model}/{meeting_id}"
                )
        elif (
            row.get("training_rows") is not None
            or row.get("training_class_counts") is not None
        ):
            raise DecisionComparisonError(
                f"non-learning baseline carries training counts: {model}/{meeting_id}"
            )
        source_hashes = row.get("source_hashes")
        if not isinstance(source_hashes, Mapping):
            raise DecisionComparisonError(
                f"prediction source hashes missing: {model}/{meeting_id}"
            )
        if model in PAPER_MODEL_LABELS and not source_hashes:
            raise DecisionComparisonError(
                f"paper source hashes missing: {model}/{meeting_id}"
            )
        if (
            model == "hamilton_jorda_ach_op"
            and row.get("panel") == "historical_n19"
            and row.get("coverage_status") == "available"
            and not source_hashes
        ):
            raise DecisionComparisonError(
                f"Hamilton--Jord\u00e0 source hashes missing: {meeting_id}"
            )
        if model == "hamilton_jorda_ach_op" and row.get(
            "panel"
        ) == "postcutoff_n12" and (
            row.get("coverage_status") != "unavailable"
            or row.get("unavailable_reason")
            != "published_hj_target_level_regime_not_applicable_post_2008"
        ):
            raise DecisionComparisonError(
                f"Hamilton--Jord\u00e0 post-2008 coverage contract drift: {meeting_id}"
            )


def validate_paper_prediction_replay(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> None:
    observed = [dict(row) for row in rows if row.get("model") in PAPER_MODEL_LABELS]
    if not observed:
        return
    for row in observed:
        row.pop("prediction_run_provenance", None)
    expected = aggregate_paper_predictions(config)

    def ordering(row: Mapping[str, Any]) -> tuple[str, str]:
        return str(row["model"]), str(row["meeting_id"])

    if sorted(observed, key=ordering) != sorted(expected, key=ordering):
        raise DecisionComparisonError(
            "paper meeting-level predictions do not replay from sealed K=10 rows"
        )


def validate_baseline_prediction_replay(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    roster: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    training_contracts: Sequence[str],
    baseline_models: Sequence[str],
    wrds_root: Path,
    seed: int,
) -> None:
    observed = [dict(row) for row in rows if row.get("model") in BASELINE_MODELS]
    for row in observed:
        row.pop("prediction_run_provenance", None)
    expected = predict_baselines(
        config,
        roster,
        feature_rows,
        training_contracts=training_contracts,
        models=baseline_models,
        wrds_root=wrds_root,
        seed=seed,
    )

    def ordering(row: Mapping[str, Any]) -> tuple[str, str, str]:
        return (
            str(row["comparison_contract"]),
            str(row["model"]),
            str(row["meeting_id"]),
        )

    if sorted(observed, key=ordering) != sorted(expected, key=ordering):
        raise DecisionComparisonError(
            "baseline predictions do not replay from frozen inputs, contracts, and seed"
        )


def _metric_block(rows: Sequence[Mapping[str, Any]], *, panel: str) -> dict[str, Any]:
    if panel not in EXPECTED_PANEL_COUNTS:
        raise DecisionComparisonError(f"cannot score unsupported panel: {panel}")
    total = len(rows)
    available = [row for row in rows if row.get("coverage_status") == "available"]
    unavailable = [row for row in rows if row.get("coverage_status") == "unavailable"]
    target_class_totals = Counter(str(row["target_direction"]) for row in rows)
    available_target_class_counts = Counter(
        str(row["target_direction"]) for row in available
    )
    supported = (
        ("cut", "hold", "hike") if panel == "historical_n19" else ("cut", "hold")
    )
    coverage = {
        "total_meetings": total,
        "available_meetings": len(available),
        "unavailable_meetings": len(unavailable),
        "coverage_rate": len(available) / total if total else None,
        "unavailable_reasons": dict(
            Counter(str(row["unavailable_reason"]) for row in unavailable)
        ),
        "invalid_generation_count": sum(
            int(row.get("generation_invalid_count") or 0) for row in available
        ),
        "strict_contract_invalid_generation_count": sum(
            int(row.get("strict_contract_invalid_count") or 0) for row in available
        ),
        "generation_count": sum(
            int(row.get("generation_count") or 0) for row in available
        ),
        "target_class_coverage": {
            direction: {
                "total_meetings": target_class_totals[direction],
                "available_meetings": available_target_class_counts[direction],
                "unavailable_meetings": (
                    target_class_totals[direction]
                    - available_target_class_counts[direction]
                ),
                "coverage_rate": (
                    available_target_class_counts[direction]
                    / target_class_totals[direction]
                    if target_class_totals[direction]
                    else None
                ),
            }
            for direction in DIRECTIONS
        },
    }
    coverage["invalid_generation_rate"] = (
        coverage["invalid_generation_count"] / coverage["generation_count"]
        if coverage["generation_count"]
        else None
    )
    coverage["strict_contract_invalid_generation_rate"] = (
        coverage["strict_contract_invalid_generation_count"]
        / coverage["generation_count"]
        if coverage["generation_count"]
        else None
    )
    if not available:
        return {
            "coverage": coverage,
            "available_target_class_counts": {direction: 0 for direction in DIRECTIONS},
            "probability_metrics": None,
            "point_metrics": None,
            "magnitude_metrics": None,
            "bootstrap_intervals": None,
        }
    class_counts = Counter(str(row["target_direction"]) for row in available)
    target_probabilities = [
        float(row["direction_probabilities"][str(row["target_direction"])])
        for row in available
    ]
    per_class_expected: dict[str, float | None] = {}
    per_class_recall: dict[str, float | None] = {}
    confusion = {
        target: {prediction: 0 for prediction in (*DIRECTIONS, "invalid_or_tie")}
        for target in DIRECTIONS
    }
    for row in available:
        target = str(row["target_direction"])
        prediction = (
            str(row["predicted_direction"])
            if row.get("predicted_direction") in DIRECTIONS
            else "invalid_or_tie"
        )
        confusion[target][prediction] += 1
    for direction in DIRECTIONS:
        values = [row for row in available if row["target_direction"] == direction]
        per_class_expected[direction] = (
            statistics.fmean(
                float(row["direction_probabilities"][direction]) for row in values
            )
            if values
            else None
        )
        per_class_recall[direction] = (
            statistics.fmean(
                row.get("predicted_direction") == direction for row in values
            )
            if values
            else None
        )
    per_target_direction_generation_accounting: dict[str, dict[str, int | None]] = {}
    for direction in DIRECTIONS:
        direction_rows = [
            row for row in available if row["target_direction"] == direction
        ]
        generation_rows = [
            row for row in direction_rows if row.get("generation_count") is not None
        ]
        if not generation_rows:
            per_target_direction_generation_accounting[direction] = {
                "generation_count": None,
                "correct_direction_generation_count": None,
                "invalid_generation_count": None,
                "strict_contract_invalid_generation_count": None,
            }
            continue
        generation_count = sum(int(row["generation_count"]) for row in generation_rows)
        correct_direction_float = sum(
            float(row["direction_probabilities"][direction])
            * int(row["generation_count"])
            for row in generation_rows
        )
        correct_direction_count = round(correct_direction_float)
        if not math.isclose(
            correct_direction_float,
            correct_direction_count,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise DecisionComparisonError(
                "paper direction probability does not close to an integer "
                f"generation count: {(direction, correct_direction_float)}"
            )
        per_target_direction_generation_accounting[direction] = {
            "generation_count": generation_count,
            "correct_direction_generation_count": correct_direction_count,
            "invalid_generation_count": sum(
                int(row.get("generation_invalid_count") or 0)
                for row in generation_rows
            ),
            "strict_contract_invalid_generation_count": sum(
                int(row.get("strict_contract_invalid_count") or 0)
                for row in generation_rows
            ),
        }
    supported_present = [
        direction for direction in supported if class_counts[direction] > 0
    ]
    supported_complete = len(supported_present) == len(supported)
    expected_balanced = (
        statistics.fmean(
            float(per_class_expected[direction]) for direction in supported_present
        )
        if supported_complete
        else None
    )
    point_balanced = (
        statistics.fmean(
            float(per_class_recall[direction]) for direction in supported_present
        )
        if supported_complete
        else None
    )
    f1_by_class: dict[str, float | None] = {}
    for direction in DIRECTIONS:
        support = class_counts[direction]
        if support == 0:
            f1_by_class[direction] = None
            continue
        true_positive = confusion[direction][direction]
        false_positive = sum(
            confusion[target][direction] for target in DIRECTIONS if target != direction
        )
        false_negative = support - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1_by_class[direction] = 2 * true_positive / denominator if denominator else 0.0
    target_signed = [
        _signed_change(str(row["target_direction"]), int(row["target_magnitude_bp"]))
        for row in available
    ]
    true_action_probabilities = [
        float(row["magnitude_probabilities"][str(target)])
        for row, target in zip(available, target_signed, strict=True)
    ]
    per_target_direction_true_action_probability: dict[str, float | None] = {}
    for direction in DIRECTIONS:
        direction_rows = [
            row for row in available if row["target_direction"] == direction
        ]
        per_target_direction_true_action_probability[direction] = (
            statistics.fmean(
                float(
                    row["magnitude_probabilities"][
                        str(
                            _signed_change(
                                str(row["target_direction"]),
                                int(row["target_magnitude_bp"]),
                            )
                        )
                    ]
                )
                for row in direction_rows
            )
            if direction_rows
            else None
        )
    expected_signed_pairs = [
        (target, float(row["expected_change_bp"]))
        for row, target in zip(available, target_signed, strict=True)
        if row.get("expected_change_bp") is not None
    ]
    epsilon = LOG_LOSS_EPSILON
    brier = statistics.fmean(
        sum(
            (
                float(row["direction_probabilities"][label])
                - (1.0 if label == row["target_direction"] else 0.0)
            )
            ** 2
            for label in PROBABILITY_LABELS
        )
        for row in available
    )
    fixed_three_expected = (
        statistics.fmean(
            float(per_class_expected[direction]) for direction in DIRECTIONS
        )
        if panel == "historical_n19"
        and all(class_counts[direction] > 0 for direction in DIRECTIONS)
        else None
    )
    fixed_three_point = (
        statistics.fmean(float(per_class_recall[direction]) for direction in DIRECTIONS)
        if panel == "historical_n19"
        and all(class_counts[direction] > 0 for direction in DIRECTIONS)
        else None
    )
    return {
        "coverage": coverage,
        "available_target_class_counts": {
            direction: class_counts[direction] for direction in DIRECTIONS
        },
        "probability_metrics": {
            "mean_true_class_probability": statistics.fmean(target_probabilities),
            "supported_class_expected_balanced_accuracy": expected_balanced,
            "supported_classes": list(supported),
            "supported_classes_present": supported_present,
            "supported_class_coverage_complete": supported_complete,
            "per_class_mean_true_probability": per_class_expected,
            "per_target_direction_mean_true_direction_probability": (
                per_class_expected
            ),
            "per_target_direction_generation_accounting": (
                per_target_direction_generation_accounting
            ),
            "four_class_brier_score": brier,
            "clipped_log_loss": statistics.fmean(
                -math.log(min(1.0, max(epsilon, probability)))
                for probability in target_probabilities
            ),
            "fixed_three_class_expected_balanced_accuracy_secondary": fixed_three_expected,
            "direction_only_interpretation": (
                "per-target-direction values average the probability assigned to "
                "the realized cut, hold, or hike direction and ignore magnitude; "
                "for K=10 paper models they equal correct-direction generations "
                "divided by all K generations, with non-extractable generations "
                "receiving zero"
            ),
        },
        "point_metrics": {
            "argmax_modal_accuracy": statistics.fmean(
                row.get("predicted_direction") == row["target_direction"]
                for row in available
            ),
            "supported_class_balanced_accuracy": point_balanced,
            "fixed_three_class_balanced_accuracy_secondary": fixed_three_point,
            "macro_f1_supported_classes": statistics.fmean(
                float(f1_by_class[direction]) for direction in supported_present
            )
            if supported_complete
            else None,
            "per_class_recall": per_class_recall,
            "per_class_f1": f1_by_class,
            "confusion_matrix": confusion,
            "invalid_or_tied_point_predictions": sum(
                row.get("predicted_direction") not in DIRECTIONS for row in available
            ),
        },
        "magnitude_metrics": {
            "mean_true_action_probability": statistics.fmean(
                true_action_probabilities
            ),
            "per_target_direction_mean_true_action_probability": (
                per_target_direction_true_action_probability
            ),
            "exact_action_accuracy": statistics.fmean(
                row.get("predicted_direction") == row["target_direction"]
                and row.get("predicted_magnitude_bp") == row["target_magnitude_bp"]
                for row in available
            ),
            "signed_bp_mae": statistics.fmean(
                abs(target - predicted) for target, predicted in expected_signed_pairs
            )
            if expected_signed_pairs
            else None,
            "signed_bp_mae_meetings": len(expected_signed_pairs),
            "interpretation": (
                "mean_true_action_probability is the non-modal exact-action score; "
                "for K=10 paper models it equals the empirical fraction of "
                "all K generations whose extractable action matches both direction "
                "and magnitude, with non-extractable generations receiving zero; "
                "per-target-direction values condition on the realized direction"
            ),
        },
        "bootstrap_intervals": None,
    }


def _scalar_metrics(
    rows: Sequence[Mapping[str, Any]], *, panel: str
) -> dict[str, float | None]:
    metrics = _metric_block(rows, panel=panel)
    probability = metrics["probability_metrics"]
    point = metrics["point_metrics"]
    magnitude = metrics["magnitude_metrics"]
    if probability is None or point is None or magnitude is None:
        return {
            "mean_true_class_probability": None,
            "supported_class_expected_balanced_accuracy": None,
            "four_class_brier_score": None,
            "clipped_log_loss": None,
            "argmax_modal_accuracy": None,
            "supported_class_balanced_accuracy": None,
            "macro_f1_supported_classes": None,
            "mean_true_action_probability": None,
            "cut_mean_true_direction_probability": None,
            "hold_mean_true_direction_probability": None,
            "hike_mean_true_direction_probability": None,
            "cut_mean_true_action_probability": None,
            "hold_mean_true_action_probability": None,
            "hike_mean_true_action_probability": None,
            "exact_action_accuracy": None,
            "signed_bp_mae": None,
        }
    return {
        "mean_true_class_probability": probability["mean_true_class_probability"],
        "supported_class_expected_balanced_accuracy": probability[
            "supported_class_expected_balanced_accuracy"
        ],
        "four_class_brier_score": probability["four_class_brier_score"],
        "clipped_log_loss": probability["clipped_log_loss"],
        "argmax_modal_accuracy": point["argmax_modal_accuracy"],
        "supported_class_balanced_accuracy": point["supported_class_balanced_accuracy"],
        "macro_f1_supported_classes": point["macro_f1_supported_classes"],
        "mean_true_action_probability": magnitude[
            "mean_true_action_probability"
        ],
        **{
            f"{direction}_mean_true_direction_probability": probability[
                "per_target_direction_mean_true_direction_probability"
            ][direction]
            for direction in DIRECTIONS
        },
        **{
            f"{direction}_mean_true_action_probability": magnitude[
                "per_target_direction_mean_true_action_probability"
            ][direction]
            for direction in DIRECTIONS
        },
        "exact_action_accuracy": magnitude["exact_action_accuracy"],
        "signed_bp_mae": magnitude["signed_bp_mae"],
    }


def _percentile_interval(values: Sequence[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "lower_2_5pct": float(np.quantile(array, 0.025)),
        "upper_97_5pct": float(np.quantile(array, 0.975)),
        "valid_draws": len(values),
    }


def _shared_stratified_draws(
    reference_rows: Sequence[Mapping[str, Any]], *, draws: int, seed: int
) -> list[list[str]]:
    by_class: dict[str, list[str]] = defaultdict(list)
    for row in reference_rows:
        by_class[str(row["target_direction"])].append(str(row["meeting_id"]))
    if any(not identifiers for identifiers in by_class.values()) or not by_class:
        raise DecisionComparisonError("cannot bootstrap an empty target stratum")
    rng = np.random.default_rng(seed)
    output: list[list[str]] = []
    for _ in range(draws):
        identifiers: list[str] = []
        for direction in DIRECTIONS:
            population = by_class.get(direction, [])
            if not population:
                continue
            selected = rng.integers(0, len(population), size=len(population))
            identifiers.extend(population[int(index)] for index in selected)
        output.append(identifiers)
    return output


def _bootstrap_seed(seed: int, contract: str, panel: str) -> int:
    material = f"{seed}:{contract}:{panel}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _bootstrap_panel(
    model_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    panel: str,
    contract: str,
    draws: int,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    reference_by_id: dict[str, Mapping[str, Any]] = {}
    for rows in model_rows.values():
        for row in rows:
            meeting_id = str(row["meeting_id"])
            previous = reference_by_id.setdefault(meeting_id, row)
            if (
                previous["target_direction"] != row["target_direction"]
                or previous["target_magnitude_bp"] != row["target_magnitude_bp"]
            ):
                raise DecisionComparisonError(
                    f"cross-model target drift for {meeting_id}"
                )
    expected = EXPECTED_PANEL_COUNTS[panel]
    if len(reference_by_id) != expected:
        raise DecisionComparisonError(f"bootstrap panel target closure drift: {panel}")
    shared_draws = _shared_stratified_draws(
        list(reference_by_id.values()),
        draws=draws,
        seed=_bootstrap_seed(seed, contract, panel),
    )
    intervals: dict[str, dict[str, Any]] = {}
    lookups: dict[str, dict[str, Mapping[str, Any]]] = {}
    for model, rows in model_rows.items():
        lookup = {
            str(row["meeting_id"]): row
            for row in rows
            if row.get("coverage_status") == "available"
        }
        lookups[model] = lookup
        samples: defaultdict[str, list[float]] = defaultdict(list)
        model_draws = (
            shared_draws
            if set(lookup) == set(reference_by_id)
            else _shared_stratified_draws(
                list(lookup.values()),
                draws=draws,
                seed=_bootstrap_seed(seed, contract, panel),
            )
            if lookup
            else []
        )
        for draw_ids in model_draws:
            sampled = [lookup[meeting_id] for meeting_id in draw_ids]
            scalars = _scalar_metrics(sampled, panel=panel)
            for metric, value in scalars.items():
                if value is not None and math.isfinite(float(value)):
                    samples[metric].append(float(value))
        intervals[model] = {
            metric: _percentile_interval(values) for metric, values in samples.items()
        }

    paired_metrics = (
        "supported_class_expected_balanced_accuracy",
        "mean_true_action_probability",
        "cut_mean_true_direction_probability",
        "hold_mean_true_direction_probability",
        "hike_mean_true_direction_probability",
        "cut_mean_true_action_probability",
        "hold_mean_true_action_probability",
        "hike_mean_true_action_probability",
        "argmax_modal_accuracy",
        "supported_class_balanced_accuracy",
    )
    paired: list[dict[str, Any]] = []
    for left, right in combinations(sorted(model_rows), 2):
        common = set(lookups[left]) & set(lookups[right])
        differences: defaultdict[str, list[float]] = defaultdict(list)
        pair_draws: list[list[str]] = []
        if common:
            pair_reference = [
                reference_by_id[meeting_id] for meeting_id in sorted(common)
            ]
            pair_material = f"{seed}:{contract}:{panel}:{left}:{right}".encode()
            pair_seed = int.from_bytes(
                hashlib.sha256(pair_material).digest()[:4], "big"
            )
            pair_draws = _shared_stratified_draws(
                pair_reference, draws=draws, seed=pair_seed
            )
        for common_draw in pair_draws:
            left_metrics = _scalar_metrics(
                [lookups[left][meeting_id] for meeting_id in common_draw], panel=panel
            )
            right_metrics = _scalar_metrics(
                [lookups[right][meeting_id] for meeting_id in common_draw], panel=panel
            )
            for metric in paired_metrics:
                left_value = left_metrics[metric]
                right_value = right_metrics[metric]
                if left_value is not None and right_value is not None:
                    differences[metric].append(float(left_value) - float(right_value))
        paired.append(
            {
                "left_model": left,
                "right_model": right,
                "common_available_meetings": len(common),
                "common_available_target_class_counts": dict(
                    Counter(
                        str(reference_by_id[meeting_id]["target_direction"])
                        for meeting_id in common
                    )
                ),
                "left_minus_right_intervals": {
                    metric: _percentile_interval(differences[metric])
                    for metric in paired_metrics
                },
            }
        )
    return intervals, paired


def score_predictions(
    predictions: Sequence[Mapping[str, Any]],
    *,
    bootstrap_draws: int = 10_000,
    seed: int = 20260827,
) -> dict[str, Any]:
    """Score meeting-level rows without ever constructing an N31 pool."""

    validate_prediction_rows(predictions)
    if isinstance(bootstrap_draws, bool) or int(bootstrap_draws) < 0:
        raise DecisionComparisonError("bootstrap_draws must be a non-negative integer")
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    targets_by_panel: dict[str, dict[str, tuple[str, int]]] = {
        panel: {} for panel in EXPECTED_PANEL_COUNTS
    }
    for row in predictions:
        grouped[
            (str(row["comparison_contract"]), str(row["panel"]), str(row["model"]))
        ].append(row)
        panel = str(row["panel"])
        meeting_id = str(row["meeting_id"])
        target = (str(row["target_direction"]), int(row["target_magnitude_bp"]))
        previous = targets_by_panel[panel].setdefault(meeting_id, target)
        if previous != target:
            raise DecisionComparisonError(f"cross-model target drift for {meeting_id}")
    for panel, expected_count in EXPECTED_PANEL_COUNTS.items():
        panel_targets = targets_by_panel[panel]
        if panel_targets and (
            len(panel_targets) != expected_count
            or Counter(direction for direction, _magnitude in panel_targets.values())
            != Counter(EXPECTED_PANEL_CLASSES[panel])
        ):
            raise DecisionComparisonError(
                f"{panel} target population/class contract drift"
            )
    summary: dict[str, Any] = {
        "schema_version": "decision-comparison-summary-v4",
        "created_at_utc": utc_now(),
        "bootstrap": {
            "draws": int(bootstrap_draws),
            "seed": int(seed),
            "unit": "meeting",
            "stratification": "target_direction_within_panel",
            "draws_shared_across_models": True,
            "single_model_missingness_policy": (
                "fixed available-meeting cohort; identical cohorts reuse identical class-stratified draws"
            ),
            "paired_missingness_policy": "common_available_meetings_only",
            "log_loss_epsilon": LOG_LOSS_EPSILON,
        },
        "matched_roster": {},
        "expanding_window": {},
        "pooled_result": None,
        "pooling_prohibited": True,
    }
    for contract in FIT_CONTRACTS:
        for panel in EXPECTED_PANEL_COUNTS:
            model_names = sorted(
                model
                for candidate_contract, candidate_panel, model in grouped
                if candidate_contract == contract and candidate_panel == panel
            )
            block_models: dict[str, Any] = {}
            model_rows = {
                model: grouped[(contract, panel, model)] for model in model_names
            }
            for model, rows in model_rows.items():
                if len(rows) != EXPECTED_PANEL_COUNTS[panel]:
                    raise DecisionComparisonError(
                        f"model panel row closure drift: {(contract, panel, model, len(rows))}"
                    )
                metrics = _metric_block(rows, panel=panel)
                metrics["model_display_name"] = MODEL_DISPLAYS[model]
                block_models[model] = metrics
            paired: list[dict[str, Any]] = []
            if model_rows and bootstrap_draws:
                intervals, paired = _bootstrap_panel(
                    model_rows,
                    panel=panel,
                    contract=contract,
                    draws=int(bootstrap_draws),
                    seed=int(seed),
                )
                for model, model_intervals in intervals.items():
                    block_models[model]["bootstrap_intervals"] = model_intervals
            summary[contract][panel] = {
                "panel": panel,
                "expected_meetings": EXPECTED_PANEL_COUNTS[panel],
                "status": "scored" if model_rows else "not_requested",
                "supported_target_classes": ["cut", "hold", "hike"]
                if panel == "historical_n19"
                else ["cut", "hold"],
                "models": block_models,
                "paired_differences": paired,
                "pooled_with_other_panel": False,
            }
    return summary


def comparison_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for contract in FIT_CONTRACTS:
        for panel in EXPECTED_PANEL_COUNTS:
            block = summary[contract][panel]
            for model, metrics in sorted(block["models"].items()):
                probability = metrics.get("probability_metrics") or {}
                point = metrics.get("point_metrics") or {}
                magnitude = metrics.get("magnitude_metrics") or {}
                coverage = metrics["coverage"]
                bootstrap = metrics.get("bootstrap_intervals") or {}
                exact_action_interval = bootstrap.get(
                    "mean_true_action_probability"
                ) or {}
                comparison = {
                        "comparison_contract": contract,
                        "panel": panel,
                        "model": model,
                        "model_display_name": metrics["model_display_name"],
                        "total_meetings": coverage["total_meetings"],
                        "available_meetings": coverage["available_meetings"],
                        "unavailable_meetings": coverage["unavailable_meetings"],
                        "coverage_rate": coverage["coverage_rate"],
                        "invalid_generation_count": coverage[
                            "invalid_generation_count"
                        ],
                        "generation_count": coverage["generation_count"],
                        "invalid_generation_rate": coverage[
                            "invalid_generation_rate"
                        ],
                        "strict_contract_invalid_generation_count": coverage[
                            "strict_contract_invalid_generation_count"
                        ],
                        "strict_contract_invalid_generation_rate": coverage[
                            "strict_contract_invalid_generation_rate"
                        ],
                        "mean_true_class_probability": probability.get(
                            "mean_true_class_probability"
                        ),
                        "supported_class_expected_balanced_accuracy": probability.get(
                            "supported_class_expected_balanced_accuracy"
                        ),
                        "four_class_brier_score": probability.get(
                            "four_class_brier_score"
                        ),
                        "clipped_log_loss": probability.get("clipped_log_loss"),
                        "argmax_modal_accuracy": point.get("argmax_modal_accuracy"),
                        "supported_class_balanced_accuracy": point.get(
                            "supported_class_balanced_accuracy"
                        ),
                        "fixed_three_class_balanced_accuracy_secondary": point.get(
                            "fixed_three_class_balanced_accuracy_secondary"
                        ),
                        "macro_f1_supported_classes": point.get(
                            "macro_f1_supported_classes"
                        ),
                        "mean_true_action_probability": magnitude.get(
                            "mean_true_action_probability"
                        ),
                        "cut_mean_true_direction_probability": (
                            probability.get(
                                "per_target_direction_mean_true_direction_probability",
                                {},
                            ).get("cut")
                        ),
                        "hold_mean_true_direction_probability": (
                            probability.get(
                                "per_target_direction_mean_true_direction_probability",
                                {},
                            ).get("hold")
                        ),
                        "hike_mean_true_direction_probability": (
                            probability.get(
                                "per_target_direction_mean_true_direction_probability",
                                {},
                            ).get("hike")
                        ),
                        "mean_true_action_probability_ci_lower_2_5pct": (
                            exact_action_interval.get("lower_2_5pct")
                        ),
                        "mean_true_action_probability_ci_upper_97_5pct": (
                            exact_action_interval.get("upper_97_5pct")
                        ),
                        "mean_true_action_probability_ci_valid_draws": (
                            exact_action_interval.get("valid_draws")
                        ),
                        "cut_mean_true_action_probability": (
                            magnitude.get(
                                "per_target_direction_mean_true_action_probability",
                                {},
                            ).get("cut")
                        ),
                        "hold_mean_true_action_probability": (
                            magnitude.get(
                                "per_target_direction_mean_true_action_probability",
                                {},
                            ).get("hold")
                        ),
                        "hike_mean_true_action_probability": (
                            magnitude.get(
                                "per_target_direction_mean_true_action_probability",
                                {},
                            ).get("hike")
                        ),
                        "exact_action_accuracy": magnitude.get("exact_action_accuracy"),
                        "signed_bp_mae": magnitude.get("signed_bp_mae"),
                        "target_class_coverage_json": canonical_json(
                            coverage.get("target_class_coverage")
                        ),
                        "target_class_generation_accounting_json": canonical_json(
                            probability.get(
                                "per_target_direction_generation_accounting"
                            )
                        ),
                        "per_class_recall_json": canonical_json(
                            point.get("per_class_recall")
                        ),
                        "confusion_matrix_json": canonical_json(
                            point.get("confusion_matrix")
                        ),
                    }
                target_class_coverage = coverage.get("target_class_coverage") or {}
                for direction in DIRECTIONS:
                    direction_coverage = target_class_coverage.get(direction) or {}
                    interval = bootstrap.get(
                        f"{direction}_mean_true_action_probability"
                    ) or {}
                    direction_interval = bootstrap.get(
                        f"{direction}_mean_true_direction_probability"
                    ) or {}
                    comparison.update(
                        {
                            f"{direction}_available_meetings": direction_coverage.get(
                                "available_meetings"
                            ),
                            f"{direction}_total_meetings": direction_coverage.get(
                                "total_meetings"
                            ),
                            f"{direction}_score_ci_lower_2_5pct": interval.get(
                                "lower_2_5pct"
                            ),
                            f"{direction}_score_ci_upper_97_5pct": interval.get(
                                "upper_97_5pct"
                            ),
                            f"{direction}_score_ci_valid_draws": interval.get(
                                "valid_draws"
                            ),
                            f"{direction}_mean_true_direction_probability_ci_lower_2_5pct": (
                                direction_interval.get("lower_2_5pct")
                            ),
                            f"{direction}_mean_true_direction_probability_ci_upper_97_5pct": (
                                direction_interval.get("upper_97_5pct")
                            ),
                            f"{direction}_mean_true_direction_probability_ci_valid_draws": (
                                direction_interval.get("valid_draws")
                            ),
                        }
                    )
                rows.append(comparison)
    return rows


def write_comparison_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], *, resume: bool
) -> None:
    fieldnames = [
        "comparison_contract",
        "panel",
        "model",
        "model_display_name",
        "total_meetings",
        "available_meetings",
        "unavailable_meetings",
        "coverage_rate",
        "invalid_generation_count",
        "generation_count",
        "invalid_generation_rate",
        "strict_contract_invalid_generation_count",
        "strict_contract_invalid_generation_rate",
        "mean_true_class_probability",
        "supported_class_expected_balanced_accuracy",
        "four_class_brier_score",
        "clipped_log_loss",
        "argmax_modal_accuracy",
        "supported_class_balanced_accuracy",
        "fixed_three_class_balanced_accuracy_secondary",
        "macro_f1_supported_classes",
        "mean_true_action_probability",
        "mean_true_action_probability_ci_lower_2_5pct",
        "mean_true_action_probability_ci_upper_97_5pct",
        "mean_true_action_probability_ci_valid_draws",
        "cut_mean_true_direction_probability",
        "hold_mean_true_direction_probability",
        "hike_mean_true_direction_probability",
        "cut_mean_true_action_probability",
        "hold_mean_true_action_probability",
        "hike_mean_true_action_probability",
        "cut_available_meetings",
        "cut_total_meetings",
        "cut_score_ci_lower_2_5pct",
        "cut_score_ci_upper_97_5pct",
        "cut_score_ci_valid_draws",
        "cut_mean_true_direction_probability_ci_lower_2_5pct",
        "cut_mean_true_direction_probability_ci_upper_97_5pct",
        "cut_mean_true_direction_probability_ci_valid_draws",
        "hold_available_meetings",
        "hold_total_meetings",
        "hold_score_ci_lower_2_5pct",
        "hold_score_ci_upper_97_5pct",
        "hold_score_ci_valid_draws",
        "hold_mean_true_direction_probability_ci_lower_2_5pct",
        "hold_mean_true_direction_probability_ci_upper_97_5pct",
        "hold_mean_true_direction_probability_ci_valid_draws",
        "hike_available_meetings",
        "hike_total_meetings",
        "hike_score_ci_lower_2_5pct",
        "hike_score_ci_upper_97_5pct",
        "hike_score_ci_valid_draws",
        "hike_mean_true_direction_probability_ci_lower_2_5pct",
        "hike_mean_true_direction_probability_ci_upper_97_5pct",
        "hike_mean_true_direction_probability_ci_valid_draws",
        "exact_action_accuracy",
        "signed_bp_mae",
        "target_class_coverage_json",
        "target_class_generation_accounting_json",
        "per_class_recall_json",
        "confusion_matrix_json",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue().encode("utf-8")
    if path.exists() and resume:
        if path.is_symlink() or not path.is_file():
            raise DecisionComparisonError(f"unsafe resume output: {path}")
        if path.read_bytes() != content:
            raise DecisionComparisonError(f"resume output drift: {path}")
        return
    _atomic_write(path, content, exclusive=True)


def _artifact_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise DecisionComparisonError(f"artifact missing or unsafe: {path}")
    return {
        "path": str(path.relative_to(relative_to))
        if relative_to is not None
        else str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _prepared_inputs(
    config: Mapping[str, Any], paths: RuntimePaths
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not paths.action_roster.is_file() or not paths.features.is_file():
        raise DecisionComparisonError(
            "prepare phase has not sealed action_roster.jsonl and features.jsonl"
        )
    if paths.action_roster.is_symlink() or paths.features.is_symlink():
        raise DecisionComparisonError(
            "prepared inputs must be regular non-symlink files"
        )
    roster = read_jsonl(paths.action_roster)
    expected_roster = build_action_roster(config)
    if roster != expected_roster:
        raise DecisionComparisonError(
            "sealed action roster differs from current frozen inputs"
        )
    features = read_jsonl(paths.features)
    macro_manifest_path = paths.macro_root / "manifest.json"
    if not macro_manifest_path.is_file() or macro_manifest_path.is_symlink():
        raise DecisionComparisonError("prepared macro manifest is missing or unsafe")
    macro_manifest = read_json(macro_manifest_path)
    validate_macro_manifest(
        paths.macro_root,
        macro_manifest,
        expected_contract=_macro_acquisition_contract(config, roster),
    )
    macro_manifest_sha256 = sha256_file(macro_manifest_path)
    validate_feature_information_set(features)
    expected_features = build_feature_rows(config, roster, paths.macro_root)
    if features != expected_features:
        raise DecisionComparisonError(
            "sealed features do not replay from bound macro sources"
        )
    roster_by_id = {str(row["meeting_id"]): row for row in roster}
    if len(features) != len(roster_by_id):
        raise DecisionComparisonError("prepared feature population drift")
    for feature in features:
        meeting = roster_by_id.get(str(feature["meeting_id"]))
        if meeting is None or any(
            feature.get(feature_field) != meeting.get(roster_field)
            for feature_field, roster_field in (
                ("meeting_start_date", "meeting_start_date"),
                ("meeting_end_date", "meeting_end_date"),
                ("evidence_cutoff", "evidence_cutoff"),
                ("target_direction", "direction"),
                ("target_magnitude_bp", "magnitude_bp"),
            )
        ):
            raise DecisionComparisonError(
                f"prepared feature/roster drift: {feature['meeting_id']}"
            )
        if feature.get("source_manifest_sha256") != macro_manifest_sha256:
            raise DecisionComparisonError(
                f"prepared feature/macro manifest drift: {feature['meeting_id']}"
            )
    return roster, features


def _parse_models(value: str) -> list[str]:
    models = [part.strip() for part in value.split(",") if part.strip()]
    if not models:
        raise argparse.ArgumentTypeError("--models must select at least one baseline")
    duplicates = [model for model, count in Counter(models).items() if count > 1]
    unknown = [model for model in models if model not in BASELINE_MODELS]
    if duplicates or unknown:
        raise argparse.ArgumentTypeError(
            f"invalid --models; duplicates={duplicates}, unknown={unknown}, allowed={list(BASELINE_MODELS)}"
        )
    return models


def _selected_contracts(value: str) -> list[str]:
    return list(FIT_CONTRACTS) if value == "both" else [value]


def _validate_manifest_input_bindings(value: Any, *, label: str = "inputs") -> int:
    """Re-hash every local artifact record recursively bound by a manifest."""

    if isinstance(value, Mapping):
        artifact_fields = {"path", "bytes", "sha256"}
        present = artifact_fields.intersection(value)
        if present:
            if not artifact_fields.issubset(value):
                raise DecisionComparisonError(
                    f"release input binding fields missing: {label}"
                )
            path = resolve_path(str(value["path"]))
            try:
                expected_bytes = int(value["bytes"])
            except (TypeError, ValueError) as exc:
                raise DecisionComparisonError(
                    f"release input byte binding invalid: {label}"
                ) from exc
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != expected_bytes
                or sha256_file(path) != value["sha256"]
            ):
                raise DecisionComparisonError(
                    f"release input binding drift: {label}"
                )
            return 1
        return sum(
            _validate_manifest_input_bindings(child, label=f"{label}.{key}")
            for key, child in value.items()
        )
    if isinstance(value, list):
        return sum(
            _validate_manifest_input_bindings(child, label=f"{label}[{index}]")
            for index, child in enumerate(value)
        )
    return 0


def _status_snapshot(paths: RuntimePaths) -> dict[str, Any]:
    artifacts = {}
    for name, path in (
        ("manifest.json", paths.manifest),
        ("action_roster.jsonl", paths.action_roster),
        ("features.jsonl", paths.features),
        ("predictions.jsonl", paths.predictions),
        ("summary.json", paths.summary),
        ("comparison.csv", paths.comparison_csv),
        ("status.json", paths.status),
        ("sources/macro/manifest.json", paths.macro_root / "manifest.json"),
    ):
        artifacts[name] = (
            {
                "exists": True,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            if path.is_file() and not path.is_symlink()
            else {"exists": False}
        )
    attempt_status: dict[str, Any] | None = None
    if paths.status.is_file() and not paths.status.is_symlink():
        try:
            attempt_status = read_json(paths.status)
        except DecisionComparisonError as exc:
            attempt_status = {"state": "invalid", "error": str(exc)}
    release: dict[str, Any] = {"state": "unsealed"}
    if paths.manifest.is_file() and not paths.manifest.is_symlink():
        try:
            manifest = read_json(paths.manifest)
            if (
                manifest.get("schema_version")
                != "decision-comparison-baselines-manifest-v1"
                or manifest.get("status") != "complete"
                or manifest.get("panel_contract")
                != {
                    "historical_n19": 19,
                    "postcutoff_n12": 12,
                    "pooled_result": None,
                    "pooling_prohibited": True,
                }
                or not set(manifest.get("training_contracts", [])).issubset(
                    set(FIT_CONTRACTS)
                )
                or not set(manifest.get("baseline_models", [])).issubset(
                    set(BASELINE_MODELS)
                )
            ):
                raise DecisionComparisonError("release manifest contract/status drift")
            records = manifest.get("outputs")
            expected_output_paths = {
                "action_roster.jsonl",
                "features.jsonl",
                "predictions.jsonl",
                "summary.json",
                "comparison.csv",
            }
            if (
                not isinstance(records, Mapping)
                or set(records) != expected_output_paths
            ):
                raise DecisionComparisonError("release output bindings missing")
            inputs = manifest.get("inputs")
            if not isinstance(inputs, Mapping):
                raise DecisionComparisonError("release input bindings missing")
            inputs_verified = _validate_manifest_input_bindings(inputs)
            if inputs_verified == 0:
                raise DecisionComparisonError("release input bindings are empty")
            for name, record in records.items():
                if not isinstance(record, Mapping) or record.get("path") != name:
                    raise DecisionComparisonError(f"release output path drift: {name}")
                path = paths.output_root / name
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or path.stat().st_size != int(record.get("bytes", -1))
                    or sha256_file(path) != record.get("sha256")
                ):
                    raise DecisionComparisonError(
                        f"release output binding drift: {name}"
                    )
            release = {
                "state": "complete",
                "manifest_sha256": sha256_file(paths.manifest),
                "inputs_verified": inputs_verified,
                "outputs_verified": len(records),
            }
        except (DecisionComparisonError, OSError, TypeError, ValueError) as exc:
            release = {"state": "invalid", "error": str(exc)}
    return {
        "schema_version": "decision-comparison-status-snapshot-v1",
        "output_root": str(paths.output_root),
        "artifacts": artifacts,
        "attempt_status": attempt_status,
        "release": release,
        "effective_state": "release_complete"
        if release["state"] == "complete"
        else (
            "release_invalid"
            if release["state"] == "invalid"
            else (
                f"phase_{attempt_status.get('state')}_unsealed"
                if isinstance(attempt_status, Mapping)
                else "not_started_unsealed"
            )
        ),
    }


def _build_manifest(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    paths: RuntimePaths,
    contracts: Sequence[str],
    models: Sequence[str],
    seed: int,
    bootstrap_draws: int,
    wrds_root: Path,
    created_at_utc: str,
) -> dict[str, Any]:
    input_records: dict[str, Any] = {
        "config": _artifact_record(config_path),
        "implementation": _artifact_record(IMPLEMENTATION),
        "model_utils": _artifact_record(Path(model_utils.__file__).resolve()),
        "requirements": _artifact_record(ROOT / "requirements.txt"),
        "macro_manifest": _artifact_record(paths.macro_root / "manifest.json"),
        "decision_release": {
            split: _artifact_record(
                resolve_path(config["decision_release"])
                / f"manifests/unique/{split}.jsonl"
            )
            for split in EXPECTED_UNIQUE_COUNTS
        },
        "official_rosters": [
            _artifact_record(resolve_path(path)) for path in config["official_rosters"]
        ],
        "external_panels": {
            panel: _artifact_record(resolve_path(path))
            for panel, path in config["external_panels"].items()
        },
        "paper_runs": {
            name: {
                "samples": _artifact_record(
                    resolve_path(root) / "inputs/panel_samples.jsonl"
                ),
                "results": _artifact_record(resolve_path(root) / "run/results.jsonl"),
                "evaluation_manifest": _artifact_record(
                    resolve_path(root) / "evaluation_manifest.json"
                ),
                "authorization": _artifact_record(
                    resolve_path(root) / "formal_generation_authorization.json"
                ),
                "run_receipt": _artifact_record(
                    resolve_path(root) / "run/run_receipt.json"
                ),
            }
            for name, root in config["paper_runs"].items()
        }
        if "matched_roster" in contracts
        else None,
    }
    if wrds_root.is_dir() and "futures" in models:
        input_records["wrds"] = {
            "restriction": "restricted_licensed_source_data",
            "manifest": _artifact_record(wrds_root / "manifest.json"),
            "raw_files_not_copied": True,
        }
    else:
        input_records["wrds"] = {
            "available": False,
            "requested_root": str(wrds_root),
            "raw_files_not_copied": True,
        }
    outputs = {
        path.name: _artifact_record(path, relative_to=paths.output_root)
        for path in (
            paths.action_roster,
            paths.features,
            paths.predictions,
            paths.summary,
            paths.comparison_csv,
        )
    }
    return {
        "schema_version": "decision-comparison-baselines-manifest-v1",
        "status": "complete",
        "created_at_utc": created_at_utc,
        "training_contracts": list(contracts),
        "baseline_models": list(models),
        "paper_models": list(PAPER_MODEL_LABELS)
        if "matched_roster" in contracts
        else [],
        "seed": int(seed),
        "seed_scope": "statistical model fitting and shared meeting bootstrap",
        "bootstrap_draws": int(bootstrap_draws),
        "log_loss_epsilon": LOG_LOSS_EPSILON,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": package_version("scikit-learn"),
            "statsmodels": package_version("statsmodels"),
            "wrds": package_version("wrds"),
        },
        "panel_contract": {
            "historical_n19": 19,
            "postcutoff_n12": 12,
            "pooled_result": None,
            "pooling_prohibited": True,
        },
        "method_labels": {
            "all_statistical_models": "literature-inspired operational baselines",
            "hamilton_jorda_ach_op": (
                "published ACH(0,1)-ordered-probit parameter application; "
                "replication code estimates ACH through 2001 and OP on 102 "
                "events through 1997; not a re-estimation or exact replication"
            ),
            "ordinal_rf": "Python cumulative ordinal approximation; not a Yoon-Fan replication",
            "futures": "fixed-grid interpolation; not an exact CME FedWatch replication",
        },
        "limitations": [
            "matched_roster N19 is a task-held-out retrospective comparison, not a chronological forecast",
            "expanding_window uses no synthetic 1989-1992 bootstrap observations",
            "DTB6 and DFF are conservatively lagged current-vintage histories and can retain later source corrections",
            (
                "Hamilton-Jorda ACH-OP uses final published parameters with current-vintage "
                "inputs, is reported only for historical_n19, and is unavailable for the "
                "post-2008 target-range regime"
            ),
            (
                "the Hamilton-Jorda paper's ordered-probit table label and public "
                "replication-code estimation endpoint differ; both are disclosed"
            ),
            "futures probabilities require the disclosed discrete grid and contain no risk-premium adjustment",
            "unavailable baseline coverage and invalid LLM generations are distinct states",
            (
                "LLM direction-probability invalid means no legal extractable decision, matching the "
                "sealed stochastic evaluator; strict contract/delivery invalidity is reported separately"
            ),
        ],
        "inputs": input_records,
        "outputs": outputs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--phase",
        choices=("acquire-macro", "prepare", "predict", "score", "all", "status"),
        default="status",
    )
    parser.add_argument(
        "--training-contract",
        choices=(*FIT_CONTRACTS, "both"),
        default="both",
    )
    parser.add_argument(
        "--models",
        type=_parse_models,
        default=list(BASELINE_MODELS),
        help=(
            "comma-separated subset of always_hold,lag1,hamilton_jorda_ach_op,"
            "kauppi_mnl,ordered_probit,ordinal_rf,futures"
        ),
    )
    parser.add_argument("--wrds-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        help="test/debug override; defaults to the frozen 10,000 draws",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    paths = runtime_paths(config, args.output_root)
    contracts = _selected_contracts(args.training_contract)
    models = list(args.models)
    wrds_root = resolve_path(
        args.wrds_root or str(config["futures"]["default_wrds_root"])
    )
    bootstrap_draws = (
        int(args.bootstrap_draws)
        if args.bootstrap_draws is not None
        else int(config["scoring"]["bootstrap_draws"])
    )
    if args.seed < 0 or args.seed >= 2**32 - 1:
        parser.error("--seed must satisfy 0 <= seed < 2**32 - 1")
    if bootstrap_draws < 0:
        parser.error("--bootstrap-draws must be non-negative")
    if args.phase == "status":
        print(json.dumps(_status_snapshot(paths), indent=2, sort_keys=True))
        return 0

    paths.output_root.mkdir(parents=True, exist_ok=True)
    if paths.manifest.exists() and not args.resume:
        raise FileExistsError(
            f"sealed decision-comparison release already exists; use --resume to verify it: {paths.manifest}"
        )
    write_status(
        paths.status,
        phase=args.phase,
        state="running",
        detail={
            "training_contracts": contracts,
            "models": models,
            "external_acquisition_started": args.phase in {"acquire-macro", "all"},
            "wrds_acquisition_started": False,
        },
    )
    try:
        if args.phase in {"acquire-macro", "all"}:
            roster = build_action_roster(config)
            acquire_macro_sources(config, roster, paths.macro_root, resume=args.resume)
        if args.phase in {"prepare", "all"}:
            roster = build_action_roster(config)
            if not (paths.macro_root / "manifest.json").is_file():
                raise DecisionComparisonError(
                    "macro sources are absent; run --phase acquire-macro explicitly before prepare"
                )
            features = build_feature_rows(config, roster, paths.macro_root)
            validate_feature_information_set(features)
            write_jsonl(paths.action_roster, roster, resume=args.resume)
            write_jsonl(paths.features, features, resume=args.resume)
        if args.phase in {"predict", "all"}:
            roster, features = _prepared_inputs(config, paths)
            predictions = predict_baselines(
                config,
                roster,
                features,
                training_contracts=contracts,
                models=models,
                wrds_root=wrds_root,
                seed=args.seed,
            )
            if "matched_roster" in contracts:
                predictions.extend(aggregate_paper_predictions(config))
            provenance = prediction_run_provenance(
                config_path=config_path,
                config=config,
                paths=paths,
                training_contracts=contracts,
                baseline_models=models,
                wrds_root=wrds_root,
                seed=args.seed,
            )
            for row in predictions:
                row["prediction_run_provenance"] = provenance
            predictions.sort(
                key=lambda row: (
                    FIT_CONTRACTS.index(str(row["comparison_contract"])),
                    list(EXPECTED_PANEL_COUNTS).index(str(row["panel"])),
                    str(row["model"]),
                    str(row["meeting_start_date"]),
                    str(row["meeting_id"]),
                )
            )
            validate_prediction_rows(predictions)
            validate_selected_prediction_population(
                predictions,
                training_contracts=contracts,
                baseline_models=models,
                prediction_seed=args.seed,
            )
            validate_prediction_run_provenance(predictions, provenance)
            validate_predictions_against_prepared_inputs(predictions, roster, features)
            validate_paper_prediction_replay(predictions, config)
            write_jsonl(paths.predictions, predictions, resume=args.resume)
        if args.phase in {"score", "all"}:
            if not paths.predictions.is_file() or paths.predictions.is_symlink():
                raise DecisionComparisonError(
                    "predict phase has not sealed predictions.jsonl"
                )
            roster, features = _prepared_inputs(config, paths)
            predictions = read_jsonl(paths.predictions)
            validate_prediction_rows(predictions)
            validate_selected_prediction_population(
                predictions,
                training_contracts=contracts,
                baseline_models=models,
                prediction_seed=args.seed,
            )
            validate_prediction_run_provenance(
                predictions,
                prediction_run_provenance(
                    config_path=config_path,
                    config=config,
                    paths=paths,
                    training_contracts=contracts,
                    baseline_models=models,
                    wrds_root=wrds_root,
                    seed=args.seed,
                ),
            )
            validate_predictions_against_prepared_inputs(predictions, roster, features)
            validate_paper_prediction_replay(predictions, config)
            validate_baseline_prediction_replay(
                predictions,
                config=config,
                roster=roster,
                feature_rows=features,
                training_contracts=contracts,
                baseline_models=models,
                wrds_root=wrds_root,
                seed=args.seed,
            )
            summary = score_predictions(
                predictions, bootstrap_draws=bootstrap_draws, seed=args.seed
            )
            if paths.summary.exists() and args.resume:
                summary["created_at_utc"] = read_json(paths.summary).get(
                    "created_at_utc"
                )
            write_json(paths.summary, summary, resume=args.resume)
            write_comparison_csv(
                paths.comparison_csv, comparison_rows(summary), resume=args.resume
            )
            created_at = utc_now()
            if paths.manifest.exists() and args.resume:
                created_at = str(
                    read_json(paths.manifest).get("created_at_utc") or created_at
                )
            manifest = _build_manifest(
                config_path=config_path,
                config=config,
                paths=paths,
                contracts=contracts,
                models=models,
                seed=args.seed,
                bootstrap_draws=bootstrap_draws,
                wrds_root=wrds_root,
                created_at_utc=created_at,
            )
            write_json(paths.manifest, manifest, resume=args.resume)
        write_status(
            paths.status,
            phase=args.phase,
            state="complete",
            detail={
                "release_artifacts": {
                    name: record
                    for name, record in _status_snapshot(paths)["artifacts"].items()
                    if name != "status.json"
                }
            },
        )
        return 0
    except Exception as exc:
        write_status(
            paths.status,
            phase=args.phase,
            state="failed",
            detail={"error_type": type(exc).__name__, "error": str(exc)},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
