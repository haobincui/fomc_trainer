"""Independent sequential sampler for chk4 pre-2009 correction-v2.

This module is additive.  It deliberately does not alter the v1/v2/v3 fixed
sampler modules whose byte hashes are retained by earlier training receipts.
The v2 correction schedule contains 48 physical rows backed by 33 unique
sources and six authoritative eight-row optimizer windows.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sized
from pathlib import Path
from typing import Any

from torch.utils.data import SequentialSampler


SAMPLER_TYPE = "manifest_fixed_schedule_correction_v2"
EXPECTED_ROWS = 48
EXPECTED_UNIQUE_SOURCES = 33
EXPECTED_STEPS = 6
EXPECTED_PER_DEVICE_BATCH = 1
EXPECTED_GRADIENT_ACCUMULATION = 8
EXPECTED_EFFECTIVE_BATCH = 8
EXPECTED_WORLD_SIZE = 1
EXPECTED_DIRECTIONS = {"hold": 12, "hike": 24, "cut": 12}
EXPECTED_WINDOW_DIRECTIONS = {"hold": 2, "hike": 4, "cut": 2}
EXPECTED_POPULATIONS = {"core": 24, "supplement": 24}
EXPECTED_WINDOW_POPULATIONS = {"core": 4, "supplement": 4}
EXPECTED_WINDOW_CELLS = {
    "core:hold": 1,
    "supplement:hold": 1,
    "core:hike": 2,
    "supplement:hike": 2,
    "core:cut": 1,
    "supplement:cut": 1,
}
EXPECTED_UNIQUE_CELLS = {
    "core:hold": 6,
    "supplement:hold": 4,
    "core:hike": 8,
    "supplement:hike": 8,
    "core:cut": 3,
    "supplement:cut": 4,
}
EXPECTED_REPEATED_SOURCE_CELLS = {
    "core:hold": 0,
    "supplement:hold": 2,
    "core:hike": 4,
    "supplement:hike": 4,
    "core:cut": 3,
    "supplement:cut": 2,
}
EXPECTED_REPEAT_HISTOGRAM = {"1": 18, "2": 15}
EXPECTED_SOURCE_POPULATION = "pre2009_correction_v2_burned_plus_fresh"


class CorrectionV2FixedScheduleError(RuntimeError):
    """The correction-v2 fixed schedule failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorrectionV2FixedScheduleError(message)


def _integer(value: object, *, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be an integer",
    )
    return int(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sampler_source_binding() -> dict[str, Any]:
    source = Path(__file__).resolve()
    return {
        "path": str(source),
        "sha256": _sha256_file(source),
        "class": "ManifestFixedScheduleCorrectionV2SFTTrainer",
        "sampler": "torch.utils.data.SequentialSampler",
        "historical_sampler_modules_modified": False,
    }


def validate_dataset_order(dataset: Sized) -> None:
    """Require exact physical schedule order and declared repeat metadata."""

    _require(len(dataset) == EXPECTED_ROWS, "correction-v2 dataset must have 48 rows")
    columns = set(getattr(dataset, "column_names", ()))
    required = {
        "schedule_index",
        "optimizer_step",
        "microbatch_slot",
        "source_sample_id",
        "source_repeat_index",
        "source_repeat_total",
        "population_role",
        "direction",
    }
    _require(
        required <= columns,
        f"correction-v2 dataset lost columns: {sorted(required - columns)}",
    )
    schedule_indices = [
        _integer(value, label="schedule_index") for value in dataset["schedule_index"]
    ]
    optimizer_steps = [
        _integer(value, label="optimizer_step") for value in dataset["optimizer_step"]
    ]
    microbatch_slots = [
        _integer(value, label="microbatch_slot") for value in dataset["microbatch_slot"]
    ]
    _require(
        schedule_indices == list(range(EXPECTED_ROWS)),
        "correction-v2 rows do not preserve schedule_index order",
    )
    source_occurrences: dict[str, list[tuple[int, int]]] = defaultdict(list)
    source_cells: dict[str, set[str]] = defaultdict(set)
    global_directions: Counter[str] = Counter()
    global_populations: Counter[str] = Counter()
    for step in range(EXPECTED_STEPS):
        start = step * EXPECTED_EFFECTIVE_BATCH
        stop = start + EXPECTED_EFFECTIVE_BATCH
        _require(
            optimizer_steps[start:stop] == [step] * 8,
            f"optimizer window {step} metadata drift",
        )
        _require(
            microbatch_slots[start:stop] == list(range(8)),
            f"optimizer window {step} slot drift",
        )
        source_ids = list(dataset["source_sample_id"])[start:stop]
        _require(
            len(source_ids) == len(set(source_ids)),
            f"optimizer window {step} repeats one source",
        )
        directions = [str(value) for value in dataset["direction"][start:stop]]
        populations = [str(value) for value in dataset["population_role"][start:stop]]
        _require(
            Counter(directions) == EXPECTED_WINDOW_DIRECTIONS,
            f"optimizer window {step} direction composition drift",
        )
        _require(
            Counter(populations) == EXPECTED_WINDOW_POPULATIONS,
            f"optimizer window {step} population composition drift",
        )
        _require(
            Counter(
                f"{population}:{direction}"
                for population, direction in zip(populations, directions, strict=True)
            )
            == EXPECTED_WINDOW_CELLS,
            f"optimizer window {step} population/direction cells drift",
        )
        global_directions.update(directions)
        global_populations.update(populations)
        for slot, source_id in enumerate(source_ids):
            _require(
                isinstance(source_id, str) and bool(source_id),
                f"optimizer window {step} has invalid source_sample_id",
            )
            source_occurrences[str(source_id)].append((step, slot))
            source_cells[str(source_id)].add(f"{populations[slot]}:{directions[slot]}")

    _require(
        global_directions == EXPECTED_DIRECTIONS,
        "correction-v2 global direction counts drift",
    )
    _require(
        global_populations == EXPECTED_POPULATIONS,
        "correction-v2 global population counts drift",
    )
    _require(
        len(source_occurrences) == EXPECTED_UNIQUE_SOURCES,
        "correction-v2 unique source count drift",
    )
    repeat_histogram = Counter(len(value) for value in source_occurrences.values())
    _require(
        {str(key): value for key, value in sorted(repeat_histogram.items())}
        == EXPECTED_REPEAT_HISTOGRAM,
        "correction-v2 source repeat histogram drift",
    )
    _require(
        all(len(cells) == 1 for cells in source_cells.values()),
        "correction-v2 source changes population/direction across repeats",
    )
    unique_cells = Counter(next(iter(cells)) for cells in source_cells.values())
    repeated_cells = Counter(
        next(iter(source_cells[source_id]))
        for source_id, occurrences in source_occurrences.items()
        if len(occurrences) == 2
    )
    _require(
        {key: unique_cells.get(key, 0) for key in EXPECTED_UNIQUE_CELLS}
        == EXPECTED_UNIQUE_CELLS,
        "correction-v2 unique population/direction cells drift",
    )
    _require(
        {key: repeated_cells.get(key, 0) for key in EXPECTED_REPEATED_SOURCE_CELLS}
        == EXPECTED_REPEATED_SOURCE_CELLS,
        "correction-v2 repeated-source cells drift",
    )
    repeat_indices = list(dataset["source_repeat_index"])
    repeat_totals = list(dataset["source_repeat_total"])
    for source_id, occurrences in source_occurrences.items():
        declared = [
            (
                _integer(
                    repeat_indices[step * EXPECTED_EFFECTIVE_BATCH + slot],
                    label="source_repeat_index",
                ),
                _integer(
                    repeat_totals[step * EXPECTED_EFFECTIVE_BATCH + slot],
                    label="source_repeat_total",
                ),
            )
            for step, slot in occurrences
        ]
        expected_total = len(occurrences)
        _require(
            [index for index, _ in declared] == list(range(expected_total))
            and {total for _, total in declared} == {expected_total},
            f"correction-v2 repeat metadata drift for {source_id}",
        )


class ManifestFixedScheduleCorrectionV2SamplerMixin:
    """Install an exact sequential sampler for the sealed 48-row schedule."""

    def _get_train_sampler(self, train_dataset: Sized | None = None):
        original = getattr(self, "train_dataset", None)
        if original is None:
            raise CorrectionV2FixedScheduleError("correction-v2 train dataset missing")
        validate_dataset_order(original)
        selected = original if train_dataset is None else train_dataset
        _require(
            len(selected) == EXPECTED_ROWS,
            "correction-v2 sampler view must contain 48 rows",
        )
        return SequentialSampler(selected)


def validate_runtime_contract(
    *, training_args: Any, release_binding: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the full release/runtime boundary before trainer construction."""

    _require(
        getattr(training_args, "train_sampler", None) == SAMPLER_TYPE,
        f"correction-v2 requires train_sampler={SAMPLER_TYPE}",
    )
    contract = release_binding.get("sampler_contract")
    _require(isinstance(contract, Mapping), "correction-v2 sampler contract missing")
    exact = {
        "type": SAMPLER_TYPE,
        "schedule_rows": EXPECTED_ROWS,
        "train_rows": EXPECTED_ROWS,
        "unique_source_rows": EXPECTED_UNIQUE_SOURCES,
        "optimizer_steps": EXPECTED_STEPS,
        "effective_batch_size": EXPECTED_EFFECTIVE_BATCH,
        "per_device_train_batch_size": EXPECTED_PER_DEVICE_BATCH,
        "gradient_accumulation_steps": EXPECTED_GRADIENT_ACCUMULATION,
        "world_size": EXPECTED_WORLD_SIZE,
        "shuffle_dataset": False,
        "direction_counts": EXPECTED_DIRECTIONS,
        "per_optimizer_window": EXPECTED_WINDOW_DIRECTIONS,
        "population_counts": EXPECTED_POPULATIONS,
        "per_optimizer_window_population": EXPECTED_WINDOW_POPULATIONS,
        "per_optimizer_window_cells": EXPECTED_WINDOW_CELLS,
        "unique_source_cells": EXPECTED_UNIQUE_CELLS,
        "repeated_source_cells": EXPECTED_REPEATED_SOURCE_CELLS,
        "order_is_authoritative": True,
        "optimizer_windows_with_duplicate_source": 0,
        "source_population": EXPECTED_SOURCE_POPULATION,
        "max_source_repeat": 2,
        "repeat_histogram": EXPECTED_REPEAT_HISTOGRAM,
        "required_sampler": "fixed_sequential_schedule_index_correction_v2",
        "secondary_shuffle_forbidden": True,
    }
    for key, expected in exact.items():
        _require(
            contract.get(key) == expected,
            f"correction-v2 sampler {key} drift: expected={expected!r}, observed={contract.get(key)!r}",
        )
    for key in ("schedule_path", "schedule_sha256", "train_path", "train_sha256"):
        _require(
            isinstance(contract.get(key), str) and bool(contract[key]),
            f"correction-v2 sampler {key} missing",
        )

    runtime = {
        "world_size": _integer(
            getattr(training_args, "world_size", None), label="world_size"
        ),
        "visible_gpus": _integer(getattr(training_args, "n_gpu", None), label="n_gpu"),
        "per_device_train_batch_size": _integer(
            getattr(training_args, "per_device_train_batch_size", None),
            label="per_device_train_batch_size",
        ),
        "gradient_accumulation_steps": _integer(
            getattr(training_args, "gradient_accumulation_steps", None),
            label="gradient_accumulation_steps",
        ),
        "max_steps": _integer(
            getattr(training_args, "max_steps", None), label="max_steps"
        ),
        "shuffle_dataset": bool(getattr(training_args, "shuffle_dataset", True)),
        "group_by_length": bool(getattr(training_args, "group_by_length", False)),
        "dataloader_drop_last": bool(
            getattr(training_args, "dataloader_drop_last", False)
        ),
        "use_liger_kernel": bool(getattr(training_args, "use_liger_kernel", False)),
    }
    expected_runtime = {
        "world_size": 1,
        "visible_gpus": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_steps": 6,
        "shuffle_dataset": False,
        "group_by_length": False,
        "dataloader_drop_last": False,
        "use_liger_kernel": False,
    }
    _require(runtime == expected_runtime, f"correction-v2 runtime drift: {runtime!r}")
    return {
        "schema_version": "manifest-fixed-schedule-correction-v2-runtime-v1",
        "type": SAMPLER_TYPE,
        "release_manifest": {
            "path": str(release_binding["release_manifest_path"]),
            "sha256": str(release_binding["release_manifest_sha256"]),
            "release_id": str(release_binding["release_id"]),
        },
        "schedule": {
            "path": str(contract["schedule_path"]),
            "sha256": str(contract["schedule_sha256"]),
            "rows": EXPECTED_ROWS,
            "unique_sources": EXPECTED_UNIQUE_SOURCES,
            "order_is_authoritative": True,
        },
        "train_file": {
            "path": str(contract["train_path"]),
            "sha256": str(contract["train_sha256"]),
            "rows": EXPECTED_ROWS,
        },
        "repeat_histogram": EXPECTED_REPEAT_HISTOGRAM,
        "sampler_source": sampler_source_binding(),
        "runtime": runtime,
    }
