"""Plan and materialize sealed ALFRED D-1 ledgers for canonical chk1.

The existing snapshot and ledger validators intentionally accept only 11- or
13-meeting populations (snapshot fetching may pair two 13-meeting populations
into one 26-vintage request set).  The chk1 training split has 102 meetings, so
this module partitions it deterministically into seven 13-meeting populations
and one 11-meeting population, without changing the shared LOO implementation.

Dense acquisition remains the default.  The explicit sparse-train mode routes
only train populations through the chk1-local sparse acquisition and ledger
implementation, while retaining dense or reused eval/test coverage.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.main.fetch_loo_source_snapshots import (
    SNAPSHOT_SCHEMA_VERSION as DENSE_SNAPSHOT_SCHEMA_VERSION,
    fetch_source_snapshots,
)
from open_r1.provenance import sha256_file, sha256_text
from open_r1.validator.loo_ledger import (
    EXPECTED_INDICATOR_COUNT,
    LEDGER_MANIFEST_SCHEMA_VERSION as DENSE_LEDGER_MANIFEST_SCHEMA_VERSION,
    build_loo_indicator_ledger,
    validate_loo_indicator_ledger,
)

from .sparse_source import (
    LEDGER_MANIFEST_SCHEMA_VERSION as SPARSE_LEDGER_MANIFEST_SCHEMA_VERSION,
    SNAPSHOT_MANIFEST_SCHEMA_VERSION as SPARSE_SNAPSHOT_MANIFEST_SCHEMA_VERSION,
    acquire_sparse_source_snapshots,
    build_sparse_loo_ledger,
    validate_sparse_loo_ledger,
    validate_sparse_snapshot_manifest,
)


SOURCE_PLAN_SCHEMA_VERSION = "chk1-source-plan-v1"
SOURCE_HANDOFF_SCHEMA_VERSION = "chk1-source-handoff-v1"
POPULATION_SCHEMA_VERSION = "loo-population-v1"
SUPPORTED_POPULATION_SIZES = (13, 11)
SUPPORTED_SNAPSHOT_VINTAGE_COUNTS = (26, 13, 11)
_SPLITS = ("train", "eval", "test")
_COVERAGE_MODES = frozenset({"dense", "sparse"})
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class SourcePlanError(ValueError):
    """Raised when the meeting roster cannot be bound without ambiguity."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourcePlanError(f"Unable to read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise SourcePlanError(f"{label} must contain one JSON object: {path}")
    return value


def _coverage_mode_for_population(
    population: Mapping[str, Any],
    *,
    sparse_train: bool,
) -> str:
    return "sparse" if sparse_train and population.get("split") == "train" else "dense"


def _require_manifest_schema(
    path: Path,
    *,
    expected_schema: str,
    label: str,
) -> dict[str, Any]:
    manifest = _load_json(path, label=label)
    observed = manifest.get("schema_version")
    if observed != expected_schema:
        raise SourcePlanError(
            f"{label} schema mismatch: expected {expected_schema!r}, "
            f"observed {observed!r}"
        )
    return manifest


def _snapshot_schema_for_mode(coverage_mode: str) -> str:
    return (
        SPARSE_SNAPSHOT_MANIFEST_SCHEMA_VERSION
        if coverage_mode == "sparse"
        else DENSE_SNAPSHOT_SCHEMA_VERSION
    )


def _ledger_schema_for_mode(coverage_mode: str) -> str:
    return (
        SPARSE_LEDGER_MANIFEST_SCHEMA_VERSION
        if coverage_mode == "sparse"
        else DENSE_LEDGER_MANIFEST_SCHEMA_VERSION
    )


def _sample_exclusions_binding(
    ledger_dir: Path,
    *,
    row_count: int,
) -> dict[str, Any]:
    path = (ledger_dir / "sample_exclusions.jsonl").resolve()
    if not path.is_file():
        raise SourcePlanError(f"Sparse sample exclusions are missing: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "row_count": row_count,
    }


def _validate_sample_exclusions_binding(
    record: Mapping[str, Any],
    *,
    ledger_dir: Path,
    expected_row_count: int,
    population_id: str,
) -> None:
    binding = record.get("sample_exclusions")
    if not isinstance(binding, Mapping):
        raise SourcePlanError(
            f"Sparse source handoff lacks sample exclusions: {population_id}"
        )
    expected = _sample_exclusions_binding(
        ledger_dir,
        row_count=expected_row_count,
    )
    if dict(binding) != expected:
        raise SourcePlanError(
            f"Sparse sample-exclusion binding changed: {population_id}"
        )


def _validate_meeting_date(value: object) -> str:
    text = str(value or "").strip()
    try:
        from datetime import date

        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise SourcePlanError(f"Invalid meeting date: {value!r}") from exc
    if parsed.isoformat() != text:
        raise SourcePlanError(f"Meeting date is not canonical: {value!r}")
    return text


def _partition_sizes(count: int) -> list[int]:
    """Return the fewest supported chunks, preferring 13-meeting chunks."""

    choices: list[list[int]] = []
    for elevens in range(count // 11 + 1):
        remainder = count - elevens * 11
        if remainder >= 0 and remainder % 13 == 0:
            choices.append([13] * (remainder // 13) + [11] * elevens)
    if not choices:
        raise SourcePlanError(
            f"Meeting count {count} cannot be partitioned into 11/13-meeting "
            "sealed ledger populations"
        )
    return min(choices, key=lambda item: (len(item), item.count(11)))


def load_meetings_by_split(student_prompt_dir: str | Path) -> dict[str, list[str]]:
    """Read only the frozen meeting/split membership from student prompts."""

    root = Path(student_prompt_dir).expanduser().resolve()
    result: dict[str, list[str]] = {}
    owner: dict[str, str] = {}
    for split in _SPLITS:
        path = root / f"{split}.jsonl"
        if not path.is_file():
            raise SourcePlanError(f"Missing student prompt split: {path}")
        meetings: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SourcePlanError(
                        f"Invalid JSON in {path}:{row_number}"
                    ) from exc
                if not isinstance(row, dict) or row.get("split") != split:
                    raise SourcePlanError(f"Split mismatch in {path}:{row_number}")
                meeting = _validate_meeting_date(row.get("meeting_date"))
                prior = owner.setdefault(meeting, split)
                if prior != split:
                    raise SourcePlanError(
                        f"Meeting {meeting} occurs in both {prior} and {split}"
                    )
                meetings.add(meeting)
        result[split] = sorted(meetings)
    return result


def _existing_population(path: Path, *, split: str) -> dict[str, Any]:
    payload = _load_json(path, label=f"{split} population")
    population_id = str(payload.get("population_id") or "").strip()
    raw_dates = payload.get("meeting_dates")
    if not _IDENTIFIER_RE.fullmatch(population_id) or not isinstance(raw_dates, list):
        raise SourcePlanError(f"Malformed existing population: {path}")
    meeting_dates = [_validate_meeting_date(item) for item in raw_dates]
    if len(meeting_dates) not in SUPPORTED_POPULATION_SIZES:
        raise SourcePlanError(f"Unsupported population size in {path}")
    if meeting_dates != sorted(set(meeting_dates)):
        raise SourcePlanError(f"Population dates must be sorted and unique: {path}")
    return {
        "population_id": population_id,
        "split": split,
        "meeting_dates": meeting_dates,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "existing": True,
    }


def create_source_plan(
    *,
    meetings_by_split: Mapping[str, Sequence[str]],
    output_dir: str | Path,
    existing_populations: Mapping[str, str | Path] | None = None,
) -> Path:
    """Write an immutable population/source plan for all three splits."""

    destination = Path(output_dir).expanduser().resolve()
    plan_path = destination / "source_plan.json"
    if plan_path.exists():
        raise FileExistsError(f"Source plan already exists: {plan_path}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Source plan directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    supplied = dict(existing_populations or {})
    populations: list[dict[str, Any]] = []
    seen_meetings: set[str] = set()

    for split in _SPLITS:
        dates = [
            _validate_meeting_date(item) for item in meetings_by_split.get(split, ())
        ]
        if dates != sorted(set(dates)):
            raise SourcePlanError(f"{split} meetings must be sorted and unique")
        overlap = sorted(seen_meetings & set(dates))
        if overlap:
            raise SourcePlanError(f"Meeting split overlap at {overlap[:5]}")
        seen_meetings.update(dates)

        existing_path = supplied.get(split)
        if existing_path is not None:
            record = _existing_population(Path(existing_path).resolve(), split=split)
            if record["meeting_dates"] != dates:
                raise SourcePlanError(
                    f"Existing {split} population does not match frozen split membership"
                )
            populations.append(record)
            continue

        cursor = 0
        for index, size in enumerate(_partition_sizes(len(dates)), 1):
            chunk = dates[cursor : cursor + size]
            cursor += size
            population_id = f"chk1_{split}_{size}_{index:02d}"
            population_payload = {
                "schema_version": POPULATION_SCHEMA_VERSION,
                "population_id": population_id,
                "phase": "chk1_local_data",
                "split_label": split,
                "meeting_dates": chunk,
            }
            population_path = destination / "populations" / f"{population_id}.json"
            _atomic_write(population_path, _pretty_json(population_payload))
            populations.append(
                {
                    "population_id": population_id,
                    "split": split,
                    "meeting_dates": chunk,
                    "path": str(population_path),
                    "sha256": sha256_file(population_path),
                    "existing": False,
                }
            )
        if cursor != len(dates):
            raise AssertionError(f"Internal population partition failure for {split}")

    generated = [item for item in populations if not item["existing"]]
    thirteen = [item for item in generated if len(item["meeting_dates"]) == 13]
    eleven = [item for item in generated if len(item["meeting_dates"]) == 11]
    snapshot_batches: list[dict[str, Any]] = []
    for index in range(0, len(thirteen), 2):
        group = thirteen[index : index + 2]
        count = sum(len(item["meeting_dates"]) for item in group)
        snapshot_batches.append(
            {
                "batch_id": "snapshot_"
                + "__".join(item["population_id"] for item in group),
                "population_ids": [item["population_id"] for item in group],
                "expected_vintage_count": count,
            }
        )
    for item in eleven:
        snapshot_batches.append(
            {
                "batch_id": f"snapshot_{item['population_id']}",
                "population_ids": [item["population_id"]],
                "expected_vintage_count": 11,
            }
        )
    if any(
        item["expected_vintage_count"] not in SUPPORTED_SNAPSHOT_VINTAGE_COUNTS
        for item in snapshot_batches
    ):
        raise AssertionError("Internal snapshot batching failure")

    payload = {
        "schema_version": SOURCE_PLAN_SCHEMA_VERSION,
        "meeting_counts": {
            split: len(meetings_by_split.get(split, ())) for split in _SPLITS
        },
        "populations": populations,
        "snapshot_batches": snapshot_batches,
    }
    plan = {**payload, "payload_sha256": sha256_text(_canonical_json(payload))}
    _atomic_write(plan_path, _pretty_json(plan))
    return plan_path


def _parse_reuse(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        population_id, separator, raw_path = value.partition("=")
        if not separator or not _IDENTIFIER_RE.fullmatch(population_id):
            raise SourcePlanError(
                "--reuse-snapshot must be POPULATION_ID=/path/to/snapshot_manifest.json"
            )
        path = Path(raw_path).expanduser().resolve()
        if population_id in result and result[population_id] != path:
            raise SourcePlanError(f"Conflicting reuse snapshot for {population_id}")
        result[population_id] = path
    return result


def _validated_source_plan(path: Path) -> dict[str, Any]:
    plan = _load_json(path, label="source plan")
    if plan.get("schema_version") != SOURCE_PLAN_SCHEMA_VERSION:
        raise SourcePlanError("Unsupported source plan schema")
    payload = {key: value for key, value in plan.items() if key != "payload_sha256"}
    if sha256_text(_canonical_json(payload)) != plan.get("payload_sha256"):
        raise SourcePlanError("Source plan payload hash mismatch")
    populations = plan.get("populations")
    if not isinstance(populations, list) or not populations:
        raise SourcePlanError("Source plan contains no populations")
    seen: set[str] = set()
    for index, population in enumerate(populations):
        if not isinstance(population, Mapping):
            raise SourcePlanError(f"Source plan population {index} is not an object")
        population_id = str(population.get("population_id") or "").strip()
        if not _IDENTIFIER_RE.fullmatch(population_id) or population_id in seen:
            raise SourcePlanError(
                f"Source plan population {index} has an invalid/duplicate ID"
            )
        seen.add(population_id)
        population_path = Path(str(population.get("path") or "")).resolve()
        expected_sha = str(population.get("sha256") or "")
        if (
            not population_path.is_file()
            or sha256_file(population_path) != expected_sha
        ):
            raise SourcePlanError(f"Source population changed: {population_id}")
    return plan


def validate_source_handoff(
    *,
    handoff_path: str | Path,
    plan_path: str | Path,
    registry_path: str | Path,
    roster_path: str | Path,
) -> dict[str, Any]:
    """Replay every ledger bound by a completed chk1 source handoff."""

    handoff_file = Path(handoff_path).expanduser().resolve()
    plan_file = Path(plan_path).expanduser().resolve()
    registry = Path(registry_path).expanduser().resolve()
    roster = Path(roster_path).expanduser().resolve()
    plan = _validated_source_plan(plan_file)
    handoff = _load_json(handoff_file, label="source handoff")
    if handoff.get("schema_version") != SOURCE_HANDOFF_SCHEMA_VERSION:
        raise SourcePlanError("Unsupported source handoff schema")
    payload = {key: value for key, value in handoff.items() if key != "payload_sha256"}
    if sha256_text(_canonical_json(payload)) != handoff.get("payload_sha256"):
        raise SourcePlanError("Source handoff payload hash mismatch")
    source_plan = handoff.get("source_plan")
    if not isinstance(source_plan, Mapping) or source_plan != {
        "path": str(plan_file),
        "sha256": sha256_file(plan_file),
        "payload_sha256": plan["payload_sha256"],
    }:
        raise SourcePlanError("Source handoff plan binding mismatch")
    if handoff.get("registry_sha256") != sha256_file(registry):
        raise SourcePlanError("Source handoff registry binding mismatch")
    if handoff.get("roster_sha256") != sha256_file(roster):
        raise SourcePlanError("Source handoff roster binding mismatch")

    populations = {item["population_id"]: item for item in plan.get("populations", [])}
    raw_ledgers = handoff.get("ledgers")
    if not isinstance(raw_ledgers, list):
        raise SourcePlanError("Source handoff ledgers must be a list")
    observed: set[str] = set()
    for index, record in enumerate(raw_ledgers):
        if not isinstance(record, Mapping):
            raise SourcePlanError(f"Source handoff ledger {index} is not an object")
        population_id = str(record.get("population_id") or "")
        population = populations.get(population_id)
        if population is None or population_id in observed:
            raise SourcePlanError(
                f"Source handoff ledger has unknown/duplicate population {population_id!r}"
            )
        observed.add(population_id)
        coverage_mode = record.get("coverage_mode", "dense")
        if not isinstance(coverage_mode, str) or coverage_mode not in _COVERAGE_MODES:
            raise SourcePlanError(
                f"Source handoff has invalid coverage_mode for {population_id}: "
                f"{coverage_mode!r}"
            )
        if coverage_mode == "sparse" and population.get("split") != "train":
            raise SourcePlanError(
                f"Sparse source coverage is restricted to train: {population_id}"
            )
        if record.get("split") != population["split"] or record.get(
            "meeting_count"
        ) != len(population["meeting_dates"]):
            raise SourcePlanError(
                f"Source handoff population metadata mismatch: {population_id}"
            )
        ledger_dir = Path(str(record.get("ledger_dir") or "")).resolve()
        expected_ledger_dir = handoff_file.parent / "ledgers" / population_id
        if ledger_dir != expected_ledger_dir.resolve():
            raise SourcePlanError(
                f"Source handoff ledger path mismatch: {population_id}"
            )
        ledger_manifest = ledger_dir / "ledger_manifest.json"
        if not ledger_manifest.is_file() or sha256_file(ledger_manifest) != record.get(
            "ledger_manifest_sha256"
        ):
            raise SourcePlanError(
                f"Source handoff ledger manifest changed: {population_id}"
            )
        snapshot_manifest = Path(str(record.get("snapshot_manifest") or "")).resolve()
        if not snapshot_manifest.is_file() or sha256_file(
            snapshot_manifest
        ) != record.get("snapshot_manifest_sha256"):
            raise SourcePlanError(
                f"Source handoff snapshot manifest changed: {population_id}"
            )
        _require_manifest_schema(
            snapshot_manifest,
            expected_schema=_snapshot_schema_for_mode(str(coverage_mode)),
            label=f"{coverage_mode} snapshot manifest for {population_id}",
        )
        _require_manifest_schema(
            ledger_manifest,
            expected_schema=_ledger_schema_for_mode(str(coverage_mode)),
            label=f"{coverage_mode} ledger manifest for {population_id}",
        )
        if coverage_mode == "dense":
            if "sample_exclusions" in record:
                raise SourcePlanError(
                    f"Dense source handoff carries sparse exclusions: {population_id}"
                )
            validation = validate_loo_indicator_ledger(
                ledger_manifest_file=ledger_manifest,
                registry_file=registry,
                snapshot_manifest_file=snapshot_manifest,
                population_file=population["path"],
                roster_file=roster,
                expected_manifest_payload_sha256=record.get("ledger_payload_sha256"),
            )
            if validation.get("row_count") != record.get("row_count"):
                raise SourcePlanError(
                    f"Source handoff ledger row count changed: {population_id}"
                )
            continue

        snapshot_validation = validate_sparse_snapshot_manifest(
            snapshot_manifest,
            registry_file=registry,
            roster_file=roster,
            expected_meeting_dates=population["meeting_dates"],
            expected_population_id=population_id,
        )
        if (
            snapshot_validation.get("status") != "valid"
            or snapshot_validation.get("population_id") != population_id
            or snapshot_validation.get("manifest_payload_sha256")
            != record.get("snapshot_payload_sha256")
        ):
            raise SourcePlanError(
                f"Sparse snapshot payload binding changed: {population_id}"
            )
        sparse_validation = validate_sparse_loo_ledger(
            ledger_manifest,
            snapshot_manifest_file=snapshot_manifest,
            registry_file=registry,
            roster_file=roster,
        )
        expected_sample_count = sparse_validation.get("expected_sample_count")
        ready_sample_count = sparse_validation.get("ready_sample_count")
        excluded_sample_count = sparse_validation.get("excluded_sample_count")
        if (
            sparse_validation.get("status") != "valid"
            or sparse_validation.get("population_id") != population_id
            or sparse_validation.get("manifest_payload_sha256")
            != record.get("ledger_payload_sha256")
            or ready_sample_count != record.get("row_count")
            or excluded_sample_count != record.get("excluded_sample_count")
            or expected_sample_count != record.get("expected_sample_count")
            or not isinstance(ready_sample_count, int)
            or isinstance(ready_sample_count, bool)
            or not isinstance(excluded_sample_count, int)
            or isinstance(excluded_sample_count, bool)
            or not isinstance(expected_sample_count, int)
            or isinstance(expected_sample_count, bool)
            or expected_sample_count
            != len(population["meeting_dates"]) * EXPECTED_INDICATOR_COUNT
            or ready_sample_count + excluded_sample_count != expected_sample_count
        ):
            raise SourcePlanError(
                f"Sparse source handoff replay differs: {population_id}"
            )
        _validate_sample_exclusions_binding(
            record,
            ledger_dir=ledger_dir,
            expected_row_count=excluded_sample_count,
            population_id=population_id,
        )
    if observed != set(populations):
        raise SourcePlanError(
            "Source handoff population coverage mismatch: "
            f"missing={sorted(set(populations) - observed)}"
        )
    return handoff


def materialize_source_plan(
    *,
    plan_path: str | Path,
    registry_path: str | Path,
    roster_path: str | Path,
    output_dir: str | Path,
    reused_snapshots: Mapping[str, str | Path] | None = None,
    sparse_train: bool = False,
    resume: bool = False,
    max_workers: int = 2,
    requests_per_second: float = 2.0,
    sparse_http_get: Callable[[str, float], Any] | None = None,
) -> Path:
    """Fetch missing official snapshots and build validated ledgers.

    Dense coverage remains the default.  With ``sparse_train=True``, only train
    populations use the chk1 sparse snapshot/ledger policy; eval and test stay
    on the canonical dense path.
    """

    if not isinstance(sparse_train, bool):
        raise SourcePlanError("sparse_train must be a boolean")

    plan_file = Path(plan_path).expanduser().resolve()
    plan = _validated_source_plan(plan_file)
    registry = Path(registry_path).expanduser().resolve()
    roster = Path(roster_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    handoff_path = destination / "source_handoff.json"
    if handoff_path.exists():
        if not resume:
            raise FileExistsError(f"Source handoff already exists: {handoff_path}")
        validated_handoff = validate_source_handoff(
            handoff_path=handoff_path,
            plan_path=plan_file,
            registry_path=registry,
            roster_path=roster,
        )
        populations_by_id = {
            item["population_id"]: item for item in plan.get("populations", [])
        }
        for record in validated_handoff["ledgers"]:
            population = populations_by_id[str(record["population_id"])]
            expected_mode = _coverage_mode_for_population(
                population,
                sparse_train=sparse_train,
            )
            if record.get("coverage_mode", "dense") != expected_mode:
                raise SourcePlanError(
                    "Existing source handoff coverage mode does not match the "
                    f"requested run: {record['population_id']}"
                )
        return handoff_path

    populations = {item["population_id"]: item for item in plan.get("populations", [])}
    reuse = {
        key: Path(value).expanduser().resolve()
        for key, value in dict(reused_snapshots or {}).items()
    }
    snapshot_for_population: dict[str, Path] = {}
    for population_id, manifest in reuse.items():
        if population_id not in populations:
            raise SourcePlanError(f"Unknown reused population: {population_id}")
        if not manifest.is_file():
            raise FileNotFoundError(f"Reused snapshot manifest is missing: {manifest}")
        coverage_mode = _coverage_mode_for_population(
            populations[population_id],
            sparse_train=sparse_train,
        )
        _require_manifest_schema(
            manifest,
            expected_schema=_snapshot_schema_for_mode(coverage_mode),
            label=f"reused {coverage_mode} snapshot for {population_id}",
        )
        if coverage_mode == "sparse":
            validate_sparse_snapshot_manifest(
                manifest,
                registry_file=registry,
                roster_file=roster,
                expected_meeting_dates=populations[population_id]["meeting_dates"],
                expected_population_id=population_id,
            )
        snapshot_for_population[population_id] = manifest

    for population_id in sorted(populations):
        population = populations[population_id]
        if (
            _coverage_mode_for_population(
                population,
                sparse_train=sparse_train,
            )
            != "sparse"
            or population_id in snapshot_for_population
        ):
            continue
        snapshot_dir = destination / "snapshots" / f"sparse_{population_id}"
        sparse_transport: dict[str, Any] = {}
        if sparse_http_get is not None:
            sparse_transport["http_get"] = sparse_http_get
        snapshot_for_population[population_id] = acquire_sparse_source_snapshots(
            registry_file=registry,
            roster_file=roster,
            meeting_dates=population["meeting_dates"],
            output_dir=snapshot_dir,
            population_id=population_id,
            split="train",
            resume=resume,
            requests_per_second=requests_per_second,
            **sparse_transport,
        )

    for batch in plan.get("snapshot_batches", []):
        population_ids = [
            population_id
            for population_id in batch["population_ids"]
            if _coverage_mode_for_population(
                populations[population_id],
                sparse_train=sparse_train,
            )
            == "dense"
        ]
        if not population_ids:
            continue
        if all(
            population_id in snapshot_for_population for population_id in population_ids
        ):
            continue
        if any(
            population_id in snapshot_for_population for population_id in population_ids
        ):
            raise SourcePlanError(
                f"Snapshot batch {batch['batch_id']} is only partially reused"
            )
        population_paths = [Path(populations[item]["path"]) for item in population_ids]
        snapshot_dir = destination / "snapshots" / str(batch["batch_id"])
        manifest = fetch_source_snapshots(
            registry_path=registry,
            population_paths=population_paths,
            output_dir=snapshot_dir,
            resume=resume,
            max_workers=max_workers,
            requests_per_second=requests_per_second,
            expected_vintage_count=sum(
                len(populations[population_id]["meeting_dates"])
                for population_id in population_ids
            ),
        )
        if manifest.get("status") != "complete":
            raise SourcePlanError(
                f"Snapshot batch did not complete: {batch['batch_id']}"
            )
        manifest_path = snapshot_dir / "snapshot_manifest.json"
        for population_id in population_ids:
            snapshot_for_population[population_id] = manifest_path

    ledger_records: list[dict[str, Any]] = []
    for population_id in sorted(populations):
        population = populations[population_id]
        coverage_mode = _coverage_mode_for_population(
            population,
            sparse_train=sparse_train,
        )
        snapshot_manifest = snapshot_for_population.get(population_id)
        if snapshot_manifest is None:
            raise SourcePlanError(f"No snapshot manifest for {population_id}")
        _require_manifest_schema(
            snapshot_manifest,
            expected_schema=_snapshot_schema_for_mode(coverage_mode),
            label=f"{coverage_mode} snapshot manifest for {population_id}",
        )
        ledger_dir = destination / "ledgers" / population_id
        ledger_manifest_path = ledger_dir / "ledger_manifest.json"
        if coverage_mode == "sparse":
            snapshot_validation = validate_sparse_snapshot_manifest(
                snapshot_manifest,
                registry_file=registry,
                roster_file=roster,
                expected_meeting_dates=population["meeting_dates"],
                expected_population_id=population_id,
            )
            if (
                snapshot_validation.get("status") != "valid"
                or snapshot_validation.get("population_id") != population_id
            ):
                raise SourcePlanError(
                    f"Sparse snapshot replay differs: {population_id}"
                )
            if ledger_manifest_path.exists():
                if not resume:
                    raise FileExistsError(
                        f"Ledger already exists: {ledger_manifest_path}"
                    )
            else:
                build_sparse_loo_ledger(
                    snapshot_manifest_file=snapshot_manifest,
                    registry_file=registry,
                    roster_file=roster,
                    output_dir=ledger_dir,
                )
            _require_manifest_schema(
                ledger_manifest_path,
                expected_schema=SPARSE_LEDGER_MANIFEST_SCHEMA_VERSION,
                label=f"sparse ledger manifest for {population_id}",
            )
            validation = validate_sparse_loo_ledger(
                ledger_manifest_path,
                snapshot_manifest_file=snapshot_manifest,
                registry_file=registry,
                roster_file=roster,
            )
            ready_count = validation.get("ready_sample_count")
            excluded_count = validation.get("excluded_sample_count")
            expected_count = validation.get("expected_sample_count")
            if (
                validation.get("status") != "valid"
                or validation.get("population_id") != population_id
                or not isinstance(ready_count, int)
                or isinstance(ready_count, bool)
                or not isinstance(excluded_count, int)
                or isinstance(excluded_count, bool)
                or not isinstance(expected_count, int)
                or isinstance(expected_count, bool)
                or expected_count
                != len(population["meeting_dates"]) * EXPECTED_INDICATOR_COUNT
                or ready_count + excluded_count != expected_count
            ):
                raise SourcePlanError(
                    f"Sparse ledger sample closure failed: {population_id}"
                )
            ledger_records.append(
                {
                    "population_id": population_id,
                    "split": population["split"],
                    "coverage_mode": coverage_mode,
                    "meeting_count": len(population["meeting_dates"]),
                    "ledger_dir": str(ledger_dir),
                    "ledger_manifest_sha256": sha256_file(ledger_manifest_path),
                    "ledger_payload_sha256": validation["manifest_payload_sha256"],
                    "row_count": ready_count,
                    "excluded_sample_count": excluded_count,
                    "expected_sample_count": expected_count,
                    "sample_exclusions": _sample_exclusions_binding(
                        ledger_dir,
                        row_count=excluded_count,
                    ),
                    "snapshot_manifest": str(snapshot_manifest),
                    "snapshot_manifest_sha256": sha256_file(snapshot_manifest),
                    "snapshot_payload_sha256": snapshot_validation[
                        "manifest_payload_sha256"
                    ],
                }
            )
            continue

        if ledger_manifest_path.exists():
            if not resume:
                raise FileExistsError(f"Ledger already exists: {ledger_manifest_path}")
        else:
            build_loo_indicator_ledger(
                registry_file=registry,
                snapshot_manifest_file=snapshot_manifest,
                population_file=population["path"],
                roster_file=roster,
                output_dir=ledger_dir,
            )
        _require_manifest_schema(
            ledger_manifest_path,
            expected_schema=DENSE_LEDGER_MANIFEST_SCHEMA_VERSION,
            label=f"dense ledger manifest for {population_id}",
        )
        validation = validate_loo_indicator_ledger(
            ledger_manifest_file=ledger_manifest_path,
            registry_file=registry,
            snapshot_manifest_file=snapshot_manifest,
            population_file=population["path"],
            roster_file=roster,
        )
        ledger_records.append(
            {
                "population_id": population_id,
                "split": population["split"],
                "coverage_mode": coverage_mode,
                "meeting_count": len(population["meeting_dates"]),
                "ledger_dir": str(ledger_dir),
                "ledger_manifest_sha256": sha256_file(ledger_manifest_path),
                "ledger_payload_sha256": validation["manifest_payload_sha256"],
                "row_count": validation["row_count"],
                "snapshot_manifest": str(snapshot_manifest),
                "snapshot_manifest_sha256": sha256_file(snapshot_manifest),
            }
        )

    handoff_payload = {
        "schema_version": SOURCE_HANDOFF_SCHEMA_VERSION,
        "source_plan": {
            "path": str(plan_file),
            "sha256": sha256_file(plan_file),
            "payload_sha256": plan["payload_sha256"],
        },
        "registry_sha256": sha256_file(registry),
        "roster_sha256": sha256_file(roster),
        "ledgers": ledger_records,
    }
    handoff = {
        **handoff_payload,
        "payload_sha256": sha256_text(_canonical_json(handoff_payload)),
    }
    _atomic_write(handoff_path, _pretty_json(handoff))
    return handoff_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Freeze 11/13-meeting populations")
    plan.add_argument("--student-prompt-dir", required=True, type=Path)
    plan.add_argument("--output-dir", required=True, type=Path)
    plan.add_argument("--eval-population", type=Path)
    plan.add_argument("--test-population", type=Path)

    materialize = subparsers.add_parser(
        "materialize", help="Fetch ALFRED snapshots and build sealed ledgers"
    )
    materialize.add_argument("--plan", required=True, type=Path)
    materialize.add_argument("--registry", required=True, type=Path)
    materialize.add_argument("--roster", required=True, type=Path)
    materialize.add_argument("--output-dir", required=True, type=Path)
    materialize.add_argument("--reuse-snapshot", action="append", default=[])
    materialize.add_argument(
        "--sparse-train",
        action="store_true",
        help=(
            "Use chk1 sparse ALFRED acquisition/ledgers for train populations; "
            "eval and test remain dense"
        ),
    )
    materialize.add_argument("--resume", action="store_true")
    materialize.add_argument("--max-workers", type=int, default=2)
    materialize.add_argument("--requests-per-second", type=float, default=2.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        existing = {
            split: value
            for split, value in (
                ("eval", args.eval_population),
                ("test", args.test_population),
            )
            if value is not None
        }
        path = create_source_plan(
            meetings_by_split=load_meetings_by_split(args.student_prompt_dir),
            output_dir=args.output_dir,
            existing_populations=existing,
        )
        print(f"source_plan={path}")
        return 0
    handoff = materialize_source_plan(
        plan_path=args.plan,
        registry_path=args.registry,
        roster_path=args.roster,
        output_dir=args.output_dir,
        reused_snapshots=_parse_reuse(args.reuse_snapshot),
        sparse_train=args.sparse_train,
        resume=args.resume,
        max_workers=args.max_workers,
        requests_per_second=args.requests_per_second,
    )
    print(f"source_handoff={handoff}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SOURCE_HANDOFF_SCHEMA_VERSION",
    "SOURCE_PLAN_SCHEMA_VERSION",
    "SourcePlanError",
    "create_source_plan",
    "load_meetings_by_split",
    "materialize_source_plan",
    "validate_source_handoff",
]
