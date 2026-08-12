"""Create and verify the immutable chk4 hierarchical-balanced SFT release.

This is a narrowly scoped child of the sealed chk4 Decision v3 release.  It
does not invent prompts, responses, labels, or evaluation rows.  The only
training transformation is a manifest-bound, deterministic 192-row schedule
over the 102 unique parent train samples:

* every eight-row optimizer window contains 4 hold, 2 hike, and 2 cut rows;
* every unique parent train sample is used at least once;
* validation and test are inherited byte-for-byte from the parent; and
* a fixed sampler must consume ``schedule_index`` 0 through 191 in order.

Publication is create-only and uses ``renameat2(RENAME_NOREPLACE)`` after a
deep validation of the staging tree.  The parent release is never modified.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jobs.retrain_v2 import publish_chk4_training_data as parent_publisher
from open_r1.trainer.sft_prompt_renderer import (
    render_sft_prompt,
    tokenize_sft_text,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ID = "chk4_decision_sft_hier_balanced_v1_20260811"
DEFAULT_PARENT = (
    REPO_ROOT / "dataset/processed/retrain_v2/"
    "chk4_decision_warmstart_grpo_core_v3_20260810"
)
DEFAULT_TOKENIZER = parent_publisher.DEFAULT_TOKENIZER
DEFAULT_OUTPUT = REPO_ROOT / "dataset/processed/retrain_v2" / RELEASE_ID

PARENT_RELEASE_ID = "chk4_decision_warmstart_grpo_core_v3_20260810"
PARENT_MANIFEST_SHA256 = (
    "8b05dce09bcb0a0b27ee240da1e5d730be93921ce19e650a3b3e8b2689adb893"
)
PARENT_UNIQUE_TRAIN_SHA256 = (
    "1e7ed2c451a29b3807a780c715d46c274c9382eeaf98b1e63da65e121048b0b7"
)
PARENT_SFT_SPLIT_SHA256 = {
    "train": "0b7ffb7a3c1f54f8c2dc65f2badbf213ea9fdd0d74e462a7a060ffa5693d56b6",
    "validation": "8913a2c31087ddbf1f678e7dbed975c2ff466254fcf13dd214ed322b5323045e",
    "test": "f89475752af03d58f183ee5182159431cd8da8f66d4efb909c346e41c4103141",
}
PARENT_UNIQUE_ROWS = 102
PARENT_SPLIT_ROWS = {"train": 141, "validation": 13, "test": 13}

RELEASE_SCHEMA = "chk4-decision-sft-hier-balanced-release-v1"
HANDOFF_SCHEMA = "chk4-decision-sft-hier-balanced-handoff-v1"
SCHEDULE_SCHEMA = "chk4-decision-sft-fixed-schedule-row-v1"
AUDIT_SCHEMA = "chk4-decision-sft-hier-balanced-audit-v1"
DATASET_ROLE = "decision_sft_hier_balanced"
SAMPLER_TYPE = "manifest_fixed_schedule_v1"
SAMPLER_SEED = 20260811

SPLITS = ("train", "validation", "test")
TRAIN_ROWS = 192
OPTIMIZER_STEPS = 24
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
EFFECTIVE_BATCH_SIZE = 8
WORLD_SIZE = 1
DIRECTION_COUNTS = {"hold": 96, "hike": 48, "cut": 48}
PER_WINDOW_COUNTS = {"hold": 4, "hike": 2, "cut": 2}
WINDOW_PATTERN = ("hold", "hike", "hold", "cut", "hold", "hike", "hold", "cut")
EXPECTED_UNIQUE_DIRECTIONS = {"hold": 89, "hike": 9, "cut": 4}
EXPECTED_REPEAT_HISTOGRAMS = {
    "hold": {1: 82, 2: 7},
    "hike": {5: 6, 6: 3},
    "cut": {12: 4},
}

BOUNDARY = "\n</think>\n"
MAX_PROMPT_TOKENS = 2560
MAX_COMPLETION_TOKENS = 512
MAX_SFT_TOKENS = 3072
MIN_DIRECTION_TOKEN_SHARE = 0.20
MAX_DIRECTION_TOKEN_SHARE = 0.55

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAMPLE_ID_RE = re.compile(r"dec-[0-9a-f]{24}")
_TRAINING_ROW_ID_RE = re.compile(r"row-[0-9a-f]{24}")
_ALLOWED_MAGNITUDES = {
    "hold": {0},
    "hike": {25, 50, 75, 100},
    "cut": {25, 50, 75, 100},
}


class HierBalancedReleaseError(RuntimeError):
    """The parent, derived data, or immutable release contract is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HierBalancedReleaseError(message)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    _require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HierBalancedReleaseError(f"invalid {label}: {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise HierBalancedReleaseError(f"cannot read {label}: {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        _require(line != "", f"{label}:{line_number}: blank line")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HierBalancedReleaseError(
                f"{label}:{line_number}: invalid JSON: {exc}"
            ) from exc
        _require(isinstance(row, dict), f"{label}:{line_number}: row must be object")
        rows.append(row)
    return rows


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _write_bytes(path, payload)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(
        (_canonical_json(dict(row)) + "\n").encode("utf-8") for row in rows
    )
    _write_bytes(path, payload)


def _file_record(path: Path, *, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    record: dict[str, Any] = {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if path.suffix == ".jsonl":
        record["rows"] = len(path.read_text(encoding="utf-8").splitlines())
    return record


def _file_records(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        _require(not path.is_symlink(), f"staging contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in {"release_manifest.json", "handoff.json"}:
                records[relative] = _file_record(path, root=root)
    return records


def _tokenizer_files(tokenizer_root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name in (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "config.json",
    ):
        path = tokenizer_root / name
        if path.is_file() and not path.is_symlink():
            records[name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    _require("tokenizer_config.json" in records, "tokenizer_config.json missing")
    _require(
        "tokenizer.json" in records or "tokenizer.model" in records,
        "tokenizer vocabulary missing",
    )
    return records


def _percentiles(values: Sequence[int]) -> dict[str, int | float]:
    _require(bool(values), "cannot summarize empty token lengths")
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    def clean(value: float) -> int | float:
        rounded = round(value, 3)
        return int(rounded) if rounded.is_integer() else rounded

    return {
        "min": ordered[0],
        "p50": clean(percentile(0.50)),
        "p95": clean(percentile(0.95)),
        "max": ordered[-1],
    }


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise HierBalancedReleaseError(
            f"source path must remain within repository root: {resolved}"
        ) from exc


def _parse_decision(response: str, *, label: str) -> tuple[str, int]:
    _require(response.count(BOUNDARY) == 1, f"{label}: reasoning boundary drift")
    reasoning, answer = response.split(BOUNDARY, 1)
    _require(bool(reasoning.strip()), f"{label}: empty reasoning")
    _require(answer == answer.strip(), f"{label}: answer has outer whitespace")
    _require(
        answer.startswith("{") and answer.endswith("}") and "```" not in answer,
        f"{label}: answer is not bare JSON",
    )
    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError as exc:
        raise HierBalancedReleaseError(f"{label}: invalid answer JSON: {exc}") from exc
    _require(isinstance(parsed, dict), f"{label}: answer must be an object")
    _require(
        set(parsed) == {"direction", "magnitude_bp"},
        f"{label}: answer key drift",
    )
    direction = parsed["direction"]
    magnitude = parsed["magnitude_bp"]
    _require(type(direction) is str, f"{label}: direction type drift")
    _require(type(magnitude) is int, f"{label}: magnitude type drift")
    _require(
        direction in _ALLOWED_MAGNITUDES
        and magnitude in _ALLOWED_MAGNITUDES[direction],
        f"{label}: decision domain drift",
    )
    return direction, magnitude


def _gold_text(direction: str, magnitude_bp: int) -> str:
    return _canonical_json({"direction": direction, "magnitude_bp": magnitude_bp})


def _verify_parent(parent: Path) -> dict[str, Any]:
    parent = parent.resolve()
    _require(parent.name == PARENT_RELEASE_ID, "unexpected parent release identity")
    manifest_path = parent / "release_manifest.json"
    _require(
        _sha256_file(manifest_path) == PARENT_MANIFEST_SHA256,
        "parent release manifest SHA drift",
    )
    try:
        parent_manifest = parent_publisher.verify_release(
            parent,
            expected_manifest_sha256=PARENT_MANIFEST_SHA256,
        )
    except (parent_publisher.Chk4ReleaseError, OSError, ValueError) as exc:
        raise HierBalancedReleaseError(
            f"parent release verification failed: {exc}"
        ) from exc
    _require(
        parent_manifest.get("release_id") == PARENT_RELEASE_ID,
        "parent manifest release_id drift",
    )

    unique_path = parent / "manifests/unique/train.jsonl"
    _require(
        _sha256_file(unique_path) == PARENT_UNIQUE_TRAIN_SHA256,
        "parent unique train SHA drift",
    )
    unique_rows = _read_jsonl(unique_path, label="parent unique train")
    _require(len(unique_rows) == PARENT_UNIQUE_ROWS, "parent unique train row drift")
    unique_by_id: dict[str, dict[str, Any]] = {}
    direction_counts: Counter[str] = Counter()
    for index, unique in enumerate(unique_rows):
        sample_id = unique.get("sample_id")
        _require(
            isinstance(sample_id, str)
            and _SAMPLE_ID_RE.fullmatch(sample_id) is not None,
            f"parent unique train:{index}: sample_id drift",
        )
        _require(sample_id not in unique_by_id, "parent unique sample_id collision")
        _require(unique.get("split") == "train", f"{sample_id}: parent split drift")
        direction = unique.get("direction")
        magnitude = unique.get("magnitude_bp")
        _require(type(direction) is str, f"{sample_id}: parent direction type")
        _require(type(magnitude) is int, f"{sample_id}: parent magnitude type")
        _require(
            direction in _ALLOWED_MAGNITUDES
            and magnitude in _ALLOWED_MAGNITUDES[direction],
            f"{sample_id}: parent action domain drift",
        )
        for field in ("prompt_sha256", "response_sha256", "gold_sha256"):
            value = unique.get(field)
            _require(
                isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
                f"{sample_id}: invalid {field}",
            )
        direction_counts[direction] += 1
        unique_by_id[sample_id] = unique
    _require(
        dict(direction_counts) == EXPECTED_UNIQUE_DIRECTIONS,
        "parent unique direction distribution drift",
    )

    parent_train = _read_jsonl(
        parent / "decision_sft/train.jsonl", label="parent physical SFT train"
    )
    _require(
        len(parent_train) == PARENT_SPLIT_ROWS["train"], "parent train count drift"
    )
    sample_payloads: dict[str, dict[str, str]] = {}
    offset = 0
    for unique in unique_rows:
        sample_id = str(unique["sample_id"])
        repeat_factor = unique.get("repeat_factor")
        _require(
            type(repeat_factor) is int and repeat_factor > 0,
            f"{sample_id}: parent repeat factor drift",
        )
        for repeat_index in range(repeat_factor):
            _require(offset < len(parent_train), "parent physical train underflow")
            row = parent_train[offset]
            _require(
                set(row) == {"prompt", "response"},
                f"parent train:{offset}: schema drift",
            )
            prompt = row.get("prompt")
            response = row.get("response")
            _require(
                isinstance(prompt, str) and isinstance(response, str),
                f"parent train:{offset}: text type drift",
            )
            _require(
                _sha256_text(prompt) == unique["prompt_sha256"],
                f"{sample_id}: parent prompt hash drift",
            )
            _require(
                _sha256_text(response) == unique["response_sha256"],
                f"{sample_id}: parent response hash drift",
            )
            direction, magnitude = _parse_decision(
                response, label=f"parent train:{offset}"
            )
            _require(
                (direction, magnitude) == (unique["direction"], unique["magnitude_bp"]),
                f"{sample_id}: parent label/response mismatch",
            )
            _require(
                _sha256_text(_gold_text(direction, magnitude)) == unique["gold_sha256"],
                f"{sample_id}: parent gold hash drift",
            )
            if repeat_index == 0:
                sample_payloads[sample_id] = {
                    "prompt": prompt,
                    "response": response,
                }
            else:
                _require(
                    sample_payloads[sample_id]
                    == {"prompt": prompt, "response": response},
                    f"{sample_id}: parent physical repeat bytes drift",
                )
            offset += 1
    _require(offset == len(parent_train), "parent physical train overflow")

    for split in SPLITS:
        path = parent / "decision_sft" / f"{split}.jsonl"
        _require(
            _sha256_file(path) == PARENT_SFT_SPLIT_SHA256[split],
            f"parent {split} SFT SHA drift",
        )
        rows = _read_jsonl(path, label=f"parent {split} SFT")
        _require(len(rows) == PARENT_SPLIT_ROWS[split], f"parent {split} count drift")
        for index, row in enumerate(rows):
            _require(
                set(row) == {"prompt", "response"},
                f"parent {split}:{index}: SFT schema drift",
            )
            _parse_decision(str(row["response"]), label=f"parent {split}:{index}")

    contract_path = parent / "contracts/decision_input_contract.json"
    contract = _read_json(contract_path, label="parent decision input contract")
    system_prompt = contract.get("student_system_prompt")
    _require(
        isinstance(system_prompt, str) and system_prompt,
        "parent student system prompt missing",
    )
    _require(
        _sha256_text(system_prompt) == contract.get("student_system_prompt_sha256"),
        "parent student system prompt SHA drift",
    )
    return {
        "root": parent,
        "manifest": parent_manifest,
        "unique_rows": unique_rows,
        "unique_by_id": unique_by_id,
        "sample_payloads": sample_payloads,
        "system_prompt": system_prompt,
        "input_contract_path": contract_path,
    }


def _cycle_order(
    rows: Sequence[Mapping[str, Any]], *, direction: str, cycle: int
) -> list[str]:
    def score(row: Mapping[str, Any]) -> str:
        sample_id = str(row["sample_id"])
        return _sha256_text(
            f"{SCHEDULE_SCHEMA}\0{SAMPLER_SEED}\0{direction}\0{cycle}\0{sample_id}"
        )

    return [str(row["sample_id"]) for row in sorted(rows, key=score)]


def _balanced_class_sequence(
    rows: Sequence[Mapping[str, Any]], *, direction: str, count: int, per_window: int
) -> list[str]:
    _require(rows and count > 0, f"{direction}: empty schedule pool")
    sequence: list[str] = []
    cycle = 0
    while len(sequence) < count:
        sequence.extend(_cycle_order(rows, direction=direction, cycle=cycle))
        cycle += 1
    del sequence[count:]

    # A cycle boundary can place the same source twice in one optimizer
    # window.  Deterministically swap with the earliest safe future source;
    # this preserves the exact per-source repeat multiset.
    for start in range(0, count, per_window):
        end = start + per_window
        for position in range(start, end):
            used = set(sequence[start:position])
            if sequence[position] not in used:
                continue
            replacement = next(
                (
                    index
                    for index in range(end, count)
                    if sequence[index] not in used
                    and sequence[index] not in sequence[position + 1 : end]
                ),
                None,
            )
            _require(
                replacement is not None,
                f"{direction}: cannot de-duplicate optimizer window",
            )
            sequence[position], sequence[replacement] = (
                sequence[replacement],
                sequence[position],
            )
        _require(
            len(set(sequence[start:end])) == per_window,
            f"{direction}: duplicate source within optimizer window",
        )
    return sequence


def _expected_training_row_id(
    *, source_sample_id: str, source_repeat_index: int, schedule_index: int
) -> str:
    payload = (
        f"{RELEASE_ID}\0{SCHEDULE_SCHEMA}\0{source_sample_id}\0"
        f"{source_repeat_index}\0{schedule_index}"
    )
    return "row-" + _sha256_text(payload)[:24]


def build_schedule(
    unique_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return the deterministic fixed schedule and its ordered source IDs."""

    pools = {
        direction: [row for row in unique_rows if row.get("direction") == direction]
        for direction in DIRECTION_COUNTS
    }
    _require(
        {direction: len(rows) for direction, rows in pools.items()}
        == EXPECTED_UNIQUE_DIRECTIONS,
        "unique direction pool drift",
    )
    sequences = {
        direction: _balanced_class_sequence(
            pools[direction],
            direction=direction,
            count=DIRECTION_COUNTS[direction],
            per_window=PER_WINDOW_COUNTS[direction],
        )
        for direction in DIRECTION_COUNTS
    }
    cursors = Counter()
    ordered_ids: list[str] = []
    for _step in range(OPTIMIZER_STEPS):
        for direction in WINDOW_PATTERN:
            ordered_ids.append(sequences[direction][cursors[direction]])
            cursors[direction] += 1
    _require(len(ordered_ids) == TRAIN_ROWS, "schedule row count drift")

    unique_by_id = {str(row["sample_id"]): row for row in unique_rows}
    totals = Counter(ordered_ids)
    seen = Counter()
    schedule: list[dict[str, Any]] = []
    for schedule_index, sample_id in enumerate(ordered_ids):
        source = unique_by_id[sample_id]
        repeat_index = seen[sample_id]
        training_row_id = _expected_training_row_id(
            source_sample_id=sample_id,
            source_repeat_index=repeat_index,
            schedule_index=schedule_index,
        )
        schedule.append(
            {
                "schema_version": SCHEDULE_SCHEMA,
                "schedule_index": schedule_index,
                "optimizer_step": schedule_index // EFFECTIVE_BATCH_SIZE,
                "microbatch_slot": schedule_index % EFFECTIVE_BATCH_SIZE,
                "training_row_id": training_row_id,
                "source_sample_id": sample_id,
                "source_repeat_index": repeat_index,
                "source_repeat_total": totals[sample_id],
                "direction": source["direction"],
                "magnitude_bp": source["magnitude_bp"],
                "prompt_sha256": source["prompt_sha256"],
                "response_sha256": source["response_sha256"],
                "gold_sha256": source["gold_sha256"],
            }
        )
        seen[sample_id] += 1

    _validate_schedule(schedule, unique_rows=unique_rows)
    return schedule, ordered_ids


def _validate_schedule(
    schedule: Sequence[Mapping[str, Any]],
    *,
    unique_rows: Sequence[Mapping[str, Any]],
) -> None:
    _require(len(schedule) == TRAIN_ROWS, "fixed schedule must contain 192 rows")
    unique_by_id = {str(row["sample_id"]): row for row in unique_rows}
    counts: Counter[str] = Counter()
    repeat_counts: Counter[str] = Counter()
    training_ids: set[str] = set()
    for index, row in enumerate(schedule):
        expected_keys = {
            "schema_version",
            "schedule_index",
            "optimizer_step",
            "microbatch_slot",
            "training_row_id",
            "source_sample_id",
            "source_repeat_index",
            "source_repeat_total",
            "direction",
            "magnitude_bp",
            "prompt_sha256",
            "response_sha256",
            "gold_sha256",
        }
        _require(set(row) == expected_keys, f"schedule:{index}: schema drift")
        _require(row.get("schema_version") == SCHEDULE_SCHEMA, "schedule schema drift")
        _require(row.get("schedule_index") == index, f"schedule:{index}: index drift")
        _require(
            row.get("optimizer_step") == index // EFFECTIVE_BATCH_SIZE,
            f"schedule:{index}: optimizer step drift",
        )
        _require(
            row.get("microbatch_slot") == index % EFFECTIVE_BATCH_SIZE,
            f"schedule:{index}: microbatch slot drift",
        )
        sample_id = row.get("source_sample_id")
        _require(
            isinstance(sample_id, str) and sample_id in unique_by_id,
            f"schedule:{index}: unknown source sample",
        )
        source = unique_by_id[sample_id]
        _require(
            row.get("source_repeat_index") == repeat_counts[sample_id],
            f"schedule:{index}: repeat index drift",
        )
        expected_id = _expected_training_row_id(
            source_sample_id=sample_id,
            source_repeat_index=repeat_counts[sample_id],
            schedule_index=index,
        )
        training_id = row.get("training_row_id")
        _require(
            training_id == expected_id
            and isinstance(training_id, str)
            and _TRAINING_ROW_ID_RE.fullmatch(training_id) is not None,
            f"schedule:{index}: training row ID drift",
        )
        _require(training_id not in training_ids, "schedule training row ID collision")
        training_ids.add(training_id)
        for field in (
            "direction",
            "magnitude_bp",
            "prompt_sha256",
            "response_sha256",
            "gold_sha256",
        ):
            _require(
                row.get(field) == source.get(field),
                f"schedule:{index}: parent {field} lineage drift",
            )
        counts[str(row["direction"])] += 1
        repeat_counts[sample_id] += 1

    _require(dict(counts) == DIRECTION_COUNTS, "schedule direction count drift")
    _require(set(repeat_counts) == set(unique_by_id), "schedule unique coverage drift")
    for row in schedule:
        _require(
            row["source_repeat_total"] == repeat_counts[row["source_sample_id"]],
            "schedule repeat total drift",
        )
    histograms: dict[str, Counter[int]] = {
        direction: Counter() for direction in DIRECTION_COUNTS
    }
    for sample_id, count in repeat_counts.items():
        direction = str(unique_by_id[sample_id]["direction"])
        histograms[direction][count] += 1
    _require(
        {direction: dict(hist) for direction, hist in histograms.items()}
        == EXPECTED_REPEAT_HISTOGRAMS,
        "schedule repeat histogram drift",
    )
    for step in range(OPTIMIZER_STEPS):
        window = schedule[
            step * EFFECTIVE_BATCH_SIZE : (step + 1) * EFFECTIVE_BATCH_SIZE
        ]
        _require(
            Counter(str(row["direction"]) for row in window) == PER_WINDOW_COUNTS,
            f"optimizer window {step}: direction composition drift",
        )
        _require(
            len({str(row["source_sample_id"]) for row in window})
            == EFFECTIVE_BATCH_SIZE,
            f"optimizer window {step}: duplicate source sample",
        )


def _materialized_train_rows(
    schedule: Sequence[Mapping[str, Any]],
    *,
    sample_payloads: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for schedule_row in schedule:
        sample_id = str(schedule_row["source_sample_id"])
        payload = sample_payloads.get(sample_id)
        _require(payload is not None, f"schedule source payload missing: {sample_id}")
        rows.append(
            {
                "prompt": payload["prompt"],
                "response": payload["response"],
                "training_row_id": schedule_row["training_row_id"],
                "source_sample_id": sample_id,
                "schedule_index": schedule_row["schedule_index"],
            }
        )
    return rows


def _validate_train_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    schedule: Sequence[Mapping[str, Any]],
    sample_payloads: Mapping[str, Mapping[str, str]],
) -> None:
    _require(len(rows) == len(schedule) == TRAIN_ROWS, "train/schedule row count drift")
    expected_keys = {
        "prompt",
        "response",
        "training_row_id",
        "source_sample_id",
        "schedule_index",
    }
    for index, (row, schedule_row) in enumerate(zip(rows, schedule, strict=True)):
        _require(set(row) == expected_keys, f"derived train:{index}: schema drift")
        _require(
            row.get("schedule_index") == index, f"derived train:{index}: index drift"
        )
        for field in ("training_row_id", "source_sample_id", "schedule_index"):
            _require(
                row.get(field) == schedule_row.get(field),
                f"derived train:{index}: schedule {field} drift",
            )
        payload = sample_payloads[str(row["source_sample_id"])]
        _require(
            row.get("prompt") == payload["prompt"]
            and row.get("response") == payload["response"],
            f"derived train:{index}: parent payload drift",
        )
        _require(
            _sha256_text(str(row["prompt"])) == schedule_row["prompt_sha256"]
            and _sha256_text(str(row["response"])) == schedule_row["response_sha256"],
            f"derived train:{index}: parent hash lineage drift",
        )
        decision = _parse_decision(str(row["response"]), label=f"derived train:{index}")
        _require(
            decision == (schedule_row["direction"], schedule_row["magnitude_bp"]),
            f"derived train:{index}: response label drift",
        )


def _audit_tokens(
    root: Path,
    *,
    tokenizer_root: Path,
    system_prompt: str,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer_root = tokenizer_root.resolve()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
    _require(tokenizer.eos_token is not None, "tokenizer EOS token missing")
    _require(tokenizer.eos_token_id is not None, "tokenizer EOS ID missing")
    split_stats: dict[str, Any] = {}
    train_completion_by_direction: Counter[str] = Counter()
    train_tail_by_direction: Counter[str] = Counter()

    for split in SPLITS:
        rows = _read_jsonl(
            root / "decision_sft" / f"{split}.jsonl",
            label=f"derived {split} SFT token audit",
        )
        prompts: list[int] = []
        completions: list[int] = []
        totals: list[int] = []
        for index, row in enumerate(rows):
            prompt = row.get("prompt")
            response = row.get("response")
            _require(
                isinstance(prompt, str) and isinstance(response, str),
                f"derived {split}:{index}: text type drift",
            )
            direction, _magnitude = _parse_decision(
                response, label=f"derived {split}:{index}"
            )
            rendered = render_sft_prompt(
                tokenizer,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            prompt_ids = tokenize_sft_text(tokenizer, rendered)
            full_ids = tokenize_sft_text(
                tokenizer, rendered + response + tokenizer.eos_token
            )
            _require(
                full_ids[: len(prompt_ids)] == prompt_ids,
                f"derived {split}:{index}: prompt/completion token prefix drift",
            )
            completion_tokens = len(full_ids) - len(prompt_ids)
            _require(
                completion_tokens > 0, f"derived {split}:{index}: empty completion"
            )
            _require(
                len(prompt_ids) <= MAX_PROMPT_TOKENS,
                f"derived {split}:{index}: prompt token overflow",
            )
            _require(
                completion_tokens <= MAX_COMPLETION_TOKENS,
                f"derived {split}:{index}: completion token overflow",
            )
            _require(
                len(full_ids) <= MAX_SFT_TOKENS,
                f"derived {split}:{index}: SFT token overflow",
            )
            _require(
                full_ids[-1] == tokenizer.eos_token_id,
                f"derived {split}:{index}: final EOS drift",
            )
            tail = BOUNDARY.strip("\n") + BOUNDARY[-1] + response.split(BOUNDARY, 1)[1]
            tail_ids = tokenizer(
                tail + tokenizer.eos_token,
                add_special_tokens=False,
            )["input_ids"]
            _require(bool(tail_ids), f"derived {split}:{index}: empty decision tail")
            prompts.append(len(prompt_ids))
            completions.append(completion_tokens)
            totals.append(len(full_ids))
            if split == "train":
                train_completion_by_direction[direction] += completion_tokens
                train_tail_by_direction[direction] += len(tail_ids)
        split_stats[split] = {
            "physical_rows": len(rows),
            "prompt_tokens": _percentiles(prompts),
            "completion_tokens": _percentiles(completions),
            "total_tokens": _percentiles(totals),
            "prompt_overflow_rows": 0,
            "completion_overflow_rows": 0,
            "sft_overflow_rows": 0,
        }

    def shares(counts: Counter[str]) -> dict[str, float]:
        total = sum(counts.values())
        _require(total > 0, "empty direction token exposure")
        output = {
            direction: round(counts[direction] / total, 8)
            for direction in DIRECTION_COUNTS
        }
        _require(
            all(
                MIN_DIRECTION_TOKEN_SHARE <= value <= MAX_DIRECTION_TOKEN_SHARE
                for value in output.values()
            ),
            f"direction token exposure outside gate: {output}",
        )
        return output

    tokenizer_files = _tokenizer_files(tokenizer_root)
    return {
        "split_token_stats": split_stats,
        "train_supervised_completion_tokens_by_direction": dict(
            train_completion_by_direction
        ),
        "train_supervised_completion_token_share": shares(
            train_completion_by_direction
        ),
        "train_decision_tail_tokens_by_direction": dict(train_tail_by_direction),
        "train_decision_tail_token_share": shares(train_tail_by_direction),
        "token_contract": {
            "tokenizer_path": str(tokenizer_root),
            "tokenizer_files": tokenizer_files,
            "tokenizer_bundle_sha256": _sha256_text(_canonical_json(tokenizer_files)),
            "tokenizer_class": (
                f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}"
            ),
            "loader_parameters": {
                "local_files_only": True,
                "fix_mistral_regex": "omitted_to_match_parent_training_runtime",
            },
            "runtime_versions": {
                "python": platform.python_version(),
                "transformers": importlib.metadata.version("transformers"),
                "tokenizers": importlib.metadata.version("tokenizers"),
                "trl": importlib.metadata.version("trl"),
            },
            "single_bos": True,
            "final_eos": True,
            "completion_mask_covers_reasoning_boundary_answer_eos": True,
            "max_prompt_length": MAX_PROMPT_TOKENS,
            "max_completion_length": MAX_COMPLETION_TOKENS,
            "sft_max_length": MAX_SFT_TOKENS,
            "truncation": False,
        },
    }


def _build_audit(
    root: Path,
    *,
    parent_info: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    tokenizer_root: Path,
) -> dict[str, Any]:
    token_audit = _audit_tokens(
        root,
        tokenizer_root=tokenizer_root,
        system_prompt=str(parent_info["system_prompt"]),
    )
    return {
        "schema_version": AUDIT_SCHEMA,
        "status": "passed",
        "created_at_utc": _utc_now(),
        "parent_release_manifest_sha256": PARENT_MANIFEST_SHA256,
        "parent_unique_train_sha256": PARENT_UNIQUE_TRAIN_SHA256,
        "lineage_rows_checked": len(schedule),
        "new_prompt_rows": 0,
        "new_response_rows": 0,
        "label_mismatches": 0,
        "invalid_reasoning_boundaries": 0,
        "unique_train_rows_covered": PARENT_UNIQUE_ROWS,
        "train_physical_direction_counts": DIRECTION_COUNTS,
        "per_optimizer_window_direction_counts": PER_WINDOW_COUNTS,
        "optimizer_windows": OPTIMIZER_STEPS,
        "optimizer_windows_with_duplicate_source": 0,
        "repeat_histograms": {
            direction: {str(key): value for key, value in histogram.items()}
            for direction, histogram in EXPECTED_REPEAT_HISTOGRAMS.items()
        },
        "validation_byte_inherited": True,
        "test_byte_inherited": True,
        "test_is_sealed_evaluation_only": True,
        "known_coverage_limits": {
            "train_unique_cut_rows": 4,
            "train_unique_hike_rows": 9,
            "train_actions": ["cut:25", "cut:100", "hike:25", "hold:0"],
            "evaluation_actions_absent_from_train": [
                "cut:50",
                "hike:50",
                "hike:75",
            ],
            "sampler_changes_exposure_not_unique_information": True,
        },
        **token_audit,
    }


def _seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _unseal_tree(root: Path) -> None:
    if not root.exists():
        return
    root.chmod(0o700)
    for path in sorted(root.rglob("*")):
        path.chmod(0o700 if path.is_dir() else 0o600)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    _require(renameat2 is not None, "renameat2(RENAME_NOREPLACE) unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise HierBalancedReleaseError(
            f"immutable destination already exists: {destination}"
        )
    raise HierBalancedReleaseError(f"atomic publish failed: {os.strerror(error)}")


def _assert_sealed_tree(root: Path) -> None:
    _require((root.stat().st_mode & 0o777) == 0o555, "release root is mutable")
    for path in root.rglob("*"):
        _require(not path.is_symlink(), f"release contains a symlink: {path}")
        expected = 0o555 if path.is_dir() else 0o444
        _require(
            (path.stat().st_mode & 0o777) == expected,
            f"release member has mutable mode: {path}",
        )


def _resolve_repo_source(value: Any, *, label: str) -> Path:
    _require(isinstance(value, str) and value, f"{label} path missing")
    raw = Path(value)
    _require(not raw.is_absolute(), f"{label} must be repository-relative")
    resolved = (REPO_ROOT / raw).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise HierBalancedReleaseError(f"{label} escapes repository") from exc
    return resolved


def verify_release(
    root: Path,
    *,
    expected_manifest_sha256: str | None = None,
    tokenizer: Path | None = None,
) -> dict[str, Any]:
    """Deeply verify a published or sealed staging release."""

    root = root.resolve()
    _require(root.is_dir() and not root.is_symlink(), f"release root missing: {root}")
    _assert_sealed_tree(root)
    manifest_path = root / "release_manifest.json"
    manifest_sha = _sha256_file(manifest_path)
    if expected_manifest_sha256 is not None:
        _require(
            _SHA256_RE.fullmatch(expected_manifest_sha256) is not None,
            "expected manifest SHA is invalid",
        )
        _require(
            manifest_sha == expected_manifest_sha256,
            "release manifest does not match externally pinned SHA",
        )
    manifest = _read_json(manifest_path, label="derived release manifest")
    _require(manifest.get("schema_version") == RELEASE_SCHEMA, "release schema drift")
    is_staging = root.name.startswith(f".{RELEASE_ID}.staging.")
    _require(
        manifest.get("release_id") == RELEASE_ID
        and (root.name == RELEASE_ID or is_staging),
        "release_id drift",
    )
    _require(manifest.get("dataset_role") == DATASET_ROLE, "dataset role drift")
    _require(
        manifest.get("quality_status") == "passed"
        and manifest.get("immutable") is True
        and manifest.get("training_ready") is True,
        "release must be immutable, passed, and training-ready",
    )
    _require(
        manifest.get("test_is_sealed_evaluation_only") is True,
        "test split seal drift",
    )

    files = manifest.get("files")
    _require(isinstance(files, dict) and files, "release file table missing")
    actual_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    _require(
        actual_files == set(files) | {"release_manifest.json", "handoff.json"},
        "release contains missing or unlisted files",
    )
    for relative, descriptor in files.items():
        _require(
            isinstance(relative, str) and isinstance(descriptor, dict),
            "invalid release file record",
        )
        path = root / relative
        _require(path.is_file() and not path.is_symlink(), f"missing file: {relative}")
        expected_keys = {"path", "bytes", "sha256"}
        if relative.endswith(".jsonl"):
            expected_keys.add("rows")
        _require(
            set(descriptor) == expected_keys and descriptor.get("path") == relative,
            f"file descriptor schema drift: {relative}",
        )
        _require(path.stat().st_size == descriptor["bytes"], f"size drift: {relative}")
        _require(_sha256_file(path) == descriptor["sha256"], f"hash drift: {relative}")
        if relative.endswith(".jsonl"):
            _require(
                len(path.read_text(encoding="utf-8").splitlines())
                == descriptor["rows"],
                f"row count drift: {relative}",
            )

    parent_record = manifest.get("parent_release")
    _require(isinstance(parent_record, dict), "parent release binding missing")
    expected_parent_record = {
        "release_id": PARENT_RELEASE_ID,
        "manifest_path": _repo_relative(DEFAULT_PARENT / "release_manifest.json"),
        "manifest_sha256": PARENT_MANIFEST_SHA256,
        "unique_train_path": _repo_relative(
            DEFAULT_PARENT / "manifests/unique/train.jsonl"
        ),
        "unique_train_sha256": PARENT_UNIQUE_TRAIN_SHA256,
        "unique_train_rows": PARENT_UNIQUE_ROWS,
        "decision_sft_split_sha256": PARENT_SFT_SPLIT_SHA256,
        "decision_sft_split_rows": PARENT_SPLIT_ROWS,
    }
    _require(parent_record == expected_parent_record, "parent release binding drift")
    parent_manifest = _resolve_repo_source(
        parent_record["manifest_path"], label="parent manifest"
    )
    parent_info = _verify_parent(parent_manifest.parent)

    copied_unique = root / "manifests/parent_unique_train.jsonl"
    _require(
        copied_unique.read_bytes()
        == (DEFAULT_PARENT / "manifests/unique/train.jsonl").read_bytes(),
        "copied parent unique manifest byte drift",
    )
    for split in ("validation", "test"):
        _require(
            (root / "decision_sft" / f"{split}.jsonl").read_bytes()
            == (DEFAULT_PARENT / "decision_sft" / f"{split}.jsonl").read_bytes(),
            f"{split} is not byte-inherited from parent",
        )
    _require(
        (root / "contracts/decision_input_contract.json").read_bytes()
        == (DEFAULT_PARENT / "contracts/decision_input_contract.json").read_bytes(),
        "decision input contract is not byte-inherited",
    )

    schedule = _read_jsonl(
        root / "manifests/sampler_schedule.jsonl", label="fixed sampler schedule"
    )
    expected_schedule, _ordered_ids = build_schedule(parent_info["unique_rows"])
    _require(schedule == expected_schedule, "fixed sampler schedule replay drift")
    train_rows = _read_jsonl(
        root / "decision_sft/train.jsonl", label="derived SFT train"
    )
    expected_train = _materialized_train_rows(
        expected_schedule,
        sample_payloads=parent_info["sample_payloads"],
    )
    _require(train_rows == expected_train, "derived train replay drift")
    _validate_train_rows(
        train_rows,
        schedule=schedule,
        sample_payloads=parent_info["sample_payloads"],
    )

    sampler = manifest.get("sampler_contract")
    _require(isinstance(sampler, dict), "sampler contract missing")
    train_record = files["decision_sft/train.jsonl"]
    schedule_record = files["manifests/sampler_schedule.jsonl"]
    expected_sampler = {
        "type": SAMPLER_TYPE,
        "seed": SAMPLER_SEED,
        "schedule_path": "manifests/sampler_schedule.jsonl",
        "schedule_sha256": schedule_record["sha256"],
        "schedule_rows": TRAIN_ROWS,
        "train_path": "decision_sft/train.jsonl",
        "train_sha256": train_record["sha256"],
        "train_rows": TRAIN_ROWS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "world_size": WORLD_SIZE,
        "optimizer_steps": OPTIMIZER_STEPS,
        "shuffle_dataset": False,
        "direction_counts": DIRECTION_COUNTS,
        "per_optimizer_window": PER_WINDOW_COUNTS,
        "order_is_authoritative": True,
        "required_sampler": "fixed_sequential_schedule_index_v1",
        "secondary_shuffle_forbidden": True,
    }
    _require(sampler == expected_sampler, "sampler contract drift")

    split_files = manifest.get("split_files")
    _require(isinstance(split_files, dict), "split file binding missing")
    _require(
        split_files
        == {split: files[f"decision_sft/{split}.jsonl"] for split in SPLITS},
        "split file binding drift",
    )
    role = manifest.get("training_roles")
    _require(
        role
        == {
            DATASET_ROLE: {
                "dataset_path": "decision_sft",
                "parent_role": "selected_chk1_merged",
                "max_length": MAX_SFT_TOKENS,
                "completion_only_loss": True,
                "sampler_contract": "sampler_contract",
            }
        },
        "training role contract drift",
    )

    tokenizer_contract = manifest.get("token_contract")
    _require(isinstance(tokenizer_contract, dict), "token contract missing")
    tokenizer_root = (
        tokenizer.resolve()
        if tokenizer is not None
        else Path(str(tokenizer_contract.get("tokenizer_path"))).resolve()
    )
    tokenizer_files = _tokenizer_files(tokenizer_root)
    _require(
        tokenizer_files == tokenizer_contract.get("tokenizer_files"),
        "runtime tokenizer file binding drift",
    )
    _require(
        _sha256_text(_canonical_json(tokenizer_files))
        == tokenizer_contract.get("tokenizer_bundle_sha256"),
        "runtime tokenizer bundle SHA drift",
    )
    replay_audit = _build_audit(
        root,
        parent_info=parent_info,
        schedule=schedule,
        tokenizer_root=tokenizer_root,
    )
    persisted_audit = _read_json(root / "audits/data_quality.json", label="data audit")
    replay_created = replay_audit.pop("created_at_utc")
    persisted_created = persisted_audit.pop("created_at_utc", None)
    _require(
        isinstance(replay_created, str)
        and isinstance(persisted_created, str)
        and replay_audit == persisted_audit,
        "data-quality/token audit replay drift",
    )
    _require(
        manifest.get("token_contract") == persisted_audit["token_contract"],
        "manifest/audit token contract drift",
    )

    handoff = _read_json(root / "handoff.json", label="derived handoff")
    handoff_record = manifest.get("handoff")
    _require(isinstance(handoff_record, dict), "handoff binding missing")
    unsigned_handoff = dict(handoff)
    unsigned_handoff.pop("release_manifest_sha256", None)
    _require(
        handoff_record
        == {
            "path": "handoff.json",
            "schema_version": HANDOFF_SCHEMA,
            "unsigned_payload_sha256": _sha256_text(_canonical_json(unsigned_handoff)),
        },
        "handoff unsigned payload binding drift",
    )
    _require(
        handoff.get("schema_version") == HANDOFF_SCHEMA
        and handoff.get("release_id") == RELEASE_ID
        and handoff.get("quality_status") == "passed"
        and handoff.get("immutable") is True
        and handoff.get("training_ready") is True
        and handoff.get("dataset_role") == DATASET_ROLE
        and handoff.get("release_manifest") == "release_manifest.json"
        and handoff.get("release_manifest_sha256") == manifest_sha
        and handoff.get("dataset_path") == "decision_sft"
        and handoff.get("sampler_schedule") == "manifests/sampler_schedule.jsonl"
        and handoff.get("test_is_sealed_evaluation_only") is True,
        "handoff contract drift",
    )

    implementation = manifest.get("implementation")
    _require(isinstance(implementation, dict), "implementation binding missing")
    _require(
        _sha256_file(root / "provenance/materializer_snapshot.py")
        == implementation.get("materializer_snapshot_sha256"),
        "materializer snapshot drift",
    )
    _require(
        _sha256_file(root / "provenance/parent_publisher_snapshot.py")
        == implementation.get("parent_publisher_snapshot_sha256"),
        "parent publisher snapshot drift",
    )

    return {
        **manifest,
        "verified_manifest_sha256": manifest_sha,
        "verified_split_files": {
            "train": root / "decision_sft/train.jsonl",
            "validation": root / "decision_sft/validation.jsonl",
        },
        "verified_schedule_path": root / "manifests/sampler_schedule.jsonl",
    }


def publish(
    *,
    parent: Path = DEFAULT_PARENT,
    tokenizer: Path = DEFAULT_TOKENIZER,
    destination: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Materialize, validate, seal, and atomically publish the child release."""

    raw_destination = destination.absolute()
    _require(
        not os.path.lexists(raw_destination),
        f"immutable destination already exists: {raw_destination}",
    )
    destination = raw_destination.parent.resolve() / raw_destination.name
    _require(
        destination.name == RELEASE_ID, "destination must use the versioned release_id"
    )
    _require(
        not os.path.lexists(destination),
        f"immutable destination already exists: {destination}",
    )
    parent = parent.resolve()
    tokenizer = tokenizer.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent_info = _verify_parent(parent)

    materializer_bytes = Path(__file__).resolve().read_bytes()
    parent_publisher_path = Path(parent_publisher.__file__).resolve()
    parent_publisher_bytes = parent_publisher_path.read_bytes()
    renderer_path = REPO_ROOT / "src/open_r1/trainer/sft_prompt_renderer.py"
    renderer_bytes = renderer_path.read_bytes()
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=destination.parent)
    )
    renamed = False
    try:
        schedule, _ordered_ids = build_schedule(parent_info["unique_rows"])
        train_rows = _materialized_train_rows(
            schedule,
            sample_payloads=parent_info["sample_payloads"],
        )
        _validate_train_rows(
            train_rows,
            schedule=schedule,
            sample_payloads=parent_info["sample_payloads"],
        )
        _write_jsonl(staging / "decision_sft/train.jsonl", train_rows)
        for split in ("validation", "test"):
            _write_bytes(
                staging / "decision_sft" / f"{split}.jsonl",
                (parent / "decision_sft" / f"{split}.jsonl").read_bytes(),
            )
        _write_jsonl(staging / "manifests/sampler_schedule.jsonl", schedule)
        _write_bytes(
            staging / "manifests/parent_unique_train.jsonl",
            (parent / "manifests/unique/train.jsonl").read_bytes(),
        )
        _write_bytes(
            staging / "contracts/decision_input_contract.json",
            Path(parent_info["input_contract_path"]).read_bytes(),
        )
        _write_bytes(
            staging / "provenance/materializer_snapshot.py", materializer_bytes
        )
        _write_bytes(
            staging / "provenance/parent_publisher_snapshot.py",
            parent_publisher_bytes,
        )
        _write_bytes(
            staging / "provenance/sft_prompt_renderer_snapshot.py",
            renderer_bytes,
        )

        audit = _build_audit(
            staging,
            parent_info=parent_info,
            schedule=schedule,
            tokenizer_root=tokenizer,
        )
        _write_json(staging / "audits/data_quality.json", audit)
        files = _file_records(staging)
        parent_record = {
            "release_id": PARENT_RELEASE_ID,
            "manifest_path": _repo_relative(parent / "release_manifest.json"),
            "manifest_sha256": PARENT_MANIFEST_SHA256,
            "unique_train_path": _repo_relative(
                parent / "manifests/unique/train.jsonl"
            ),
            "unique_train_sha256": PARENT_UNIQUE_TRAIN_SHA256,
            "unique_train_rows": PARENT_UNIQUE_ROWS,
            "decision_sft_split_sha256": PARENT_SFT_SPLIT_SHA256,
            "decision_sft_split_rows": PARENT_SPLIT_ROWS,
        }
        sampler_contract = {
            "type": SAMPLER_TYPE,
            "seed": SAMPLER_SEED,
            "schedule_path": "manifests/sampler_schedule.jsonl",
            "schedule_sha256": files["manifests/sampler_schedule.jsonl"]["sha256"],
            "schedule_rows": TRAIN_ROWS,
            "train_path": "decision_sft/train.jsonl",
            "train_sha256": files["decision_sft/train.jsonl"]["sha256"],
            "train_rows": TRAIN_ROWS,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "world_size": WORLD_SIZE,
            "optimizer_steps": OPTIMIZER_STEPS,
            "shuffle_dataset": False,
            "direction_counts": DIRECTION_COUNTS,
            "per_optimizer_window": PER_WINDOW_COUNTS,
            "order_is_authoritative": True,
            "required_sampler": "fixed_sequential_schedule_index_v1",
            "secondary_shuffle_forbidden": True,
        }
        created_at = _utc_now()
        handoff_unsigned = {
            "schema_version": HANDOFF_SCHEMA,
            "release_id": RELEASE_ID,
            "created_at_utc": created_at,
            "quality_status": "passed",
            "immutable": True,
            "training_ready": True,
            "dataset_role": DATASET_ROLE,
            "release_manifest": "release_manifest.json",
            "dataset_path": "decision_sft",
            "sampler_schedule": "manifests/sampler_schedule.jsonl",
            "parent_release_manifest_sha256": PARENT_MANIFEST_SHA256,
            "physical_split_counts": {
                "train": TRAIN_ROWS,
                "validation": PARENT_SPLIT_ROWS["validation"],
                "test": PARENT_SPLIT_ROWS["test"],
            },
            "test_is_sealed_evaluation_only": True,
            "known_limitations": [
                "cut has four and hike has nine unique train meetings",
                "cut:50, hike:50, and hike:75 evaluation actions are absent from train",
                "the sampler changes exposure but does not create independent information",
            ],
        }
        manifest = {
            "schema_version": RELEASE_SCHEMA,
            "release_id": RELEASE_ID,
            "created_at_utc": created_at,
            "quality_status": "passed",
            "immutable": True,
            "training_ready": True,
            "canonical_dag_bindable": False,
            "dataset_role": DATASET_ROLE,
            "grain": (
                "one parent unique target-decision-blind meeting brief; "
                "physical train exposure is fixed-schedule manifest-bound"
            ),
            "parent_release": parent_record,
            "unique_split_counts": {
                "train": PARENT_UNIQUE_ROWS,
                "validation": PARENT_SPLIT_ROWS["validation"],
                "test": PARENT_SPLIT_ROWS["test"],
            },
            "physical_split_counts": {
                "train": TRAIN_ROWS,
                "validation": PARENT_SPLIT_ROWS["validation"],
                "test": PARENT_SPLIT_ROWS["test"],
            },
            "training_roles": {
                DATASET_ROLE: {
                    "dataset_path": "decision_sft",
                    "parent_role": "selected_chk1_merged",
                    "max_length": MAX_SFT_TOKENS,
                    "completion_only_loss": True,
                    "sampler_contract": "sampler_contract",
                }
            },
            "sampler_contract": sampler_contract,
            "split_files": {
                split: files[f"decision_sft/{split}.jsonl"] for split in SPLITS
            },
            "token_contract": audit["token_contract"],
            "input_contract": {
                "path": "contracts/decision_input_contract.json",
                "sha256": files["contracts/decision_input_contract.json"]["sha256"],
            },
            "implementation": {
                "materializer_snapshot": "provenance/materializer_snapshot.py",
                "materializer_snapshot_sha256": _sha256_bytes(materializer_bytes),
                "parent_publisher_snapshot": (
                    "provenance/parent_publisher_snapshot.py"
                ),
                "parent_publisher_snapshot_sha256": _sha256_bytes(
                    parent_publisher_bytes
                ),
                "sft_prompt_renderer_snapshot": (
                    "provenance/sft_prompt_renderer_snapshot.py"
                ),
                "sft_prompt_renderer_snapshot_sha256": _sha256_bytes(renderer_bytes),
            },
            "known_coverage_limits": audit["known_coverage_limits"],
            "handoff": {
                "path": "handoff.json",
                "schema_version": HANDOFF_SCHEMA,
                "unsigned_payload_sha256": _sha256_text(
                    _canonical_json(handoff_unsigned)
                ),
            },
            "test_is_sealed_evaluation_only": True,
            "files": files,
        }
        _write_json(staging / "release_manifest.json", manifest)
        manifest_sha = _sha256_file(staging / "release_manifest.json")
        _write_json(
            staging / "handoff.json",
            {**handoff_unsigned, "release_manifest_sha256": manifest_sha},
        )
        _seal_tree(staging)
        verify_release(
            staging,
            expected_manifest_sha256=manifest_sha,
            tokenizer=tokenizer,
        )
        _rename_noreplace(staging, destination)
        renamed = True
        verified = verify_release(
            destination,
            expected_manifest_sha256=manifest_sha,
            tokenizer=tokenizer,
        )
        return {
            "status": "published",
            "release_id": verified["release_id"],
            "destination": str(destination),
            "release_manifest_sha256": manifest_sha,
            "schedule_sha256": verified["sampler_contract"]["schedule_sha256"],
            "train_sha256": verified["sampler_contract"]["train_sha256"],
        }
    except Exception:
        if not renamed and staging.exists():
            _unseal_tree(staging)
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    publish_parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    publish_parser.add_argument("--destination", type=Path, default=DEFAULT_OUTPUT)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--release", type=Path, default=DEFAULT_OUTPUT)
    verify_parser.add_argument("--expected-manifest-sha256")
    verify_parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "publish":
            result = publish(
                parent=args.parent,
                tokenizer=args.tokenizer,
                destination=args.destination,
            )
        else:
            verified = verify_release(
                args.release,
                expected_manifest_sha256=args.expected_manifest_sha256,
                tokenizer=args.tokenizer,
            )
            result = {
                "status": "verified",
                "release_id": verified["release_id"],
                "release_manifest_sha256": verified["verified_manifest_sha256"],
                "schedule_sha256": verified["sampler_contract"]["schedule_sha256"],
                "train_sha256": verified["sampler_contract"]["train_sha256"],
            }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (
        HierBalancedReleaseError,
        parent_publisher.Chk4ReleaseError,
        OSError,
        ValueError,
    ) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
