"""Fail-closed sequential sampler for sealed optimizer-step schedules.

This module is deliberately small and versioned by its own source hash.  A
release publisher decides the row order; the trainer consumes indices
``0..N-1`` exactly once.  No seed or epoch callback is allowed to reorder it.

Version 1 remains the fixed 192-row/24-step chk4 schedule.  Version 2 admits a
manifest-bound dynamic ``N`` and optimizer-step count while retaining the same
single-GPU, batch-1, GA-8 and sequential-order safety boundary.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sized
from pathlib import Path
from typing import Any

from torch.utils.data import SequentialSampler


SAMPLER_TYPE_V1 = "manifest_fixed_schedule_v1"
SAMPLER_TYPE_V2 = "manifest_fixed_schedule_v2"
# Backward-compatible public name used by the existing v1 profile and tests.
SAMPLER_TYPE = SAMPLER_TYPE_V1
SUPPORTED_SAMPLER_TYPES = frozenset({SAMPLER_TYPE_V1, SAMPLER_TYPE_V2})
EXPECTED_ROWS = 192
EXPECTED_WORLD_SIZE = 1
EXPECTED_PER_DEVICE_BATCH = 1
EXPECTED_GRADIENT_ACCUMULATION = 8
EXPECTED_EFFECTIVE_BATCH = 8
EXPECTED_OPTIMIZER_STEPS = 24


class FixedScheduleSamplerError(RuntimeError):
    """Raised when a sealed schedule cannot be consumed without reordering."""


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
        "class": "ManifestFixedScheduleSFTTrainer",
        "sampler": "torch.utils.data.SequentialSampler",
    }


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FixedScheduleSamplerError(f"{label} must be an integer")
    return value


def validate_runtime_contract(
    *, training_args: Any, release_binding: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and return an immutable-ready fixed-schedule runtime receipt."""

    sampler_type = getattr(training_args, "train_sampler", None)
    if sampler_type not in SUPPORTED_SAMPLER_TYPES:
        raise FixedScheduleSamplerError(
            "sealed schedule requires train_sampler="
            f"{SAMPLER_TYPE_V1} or {SAMPLER_TYPE_V2}"
        )
    contract = release_binding.get("sampler_contract")
    if not isinstance(contract, Mapping):
        raise FixedScheduleSamplerError("release has no verified sampler contract")
    expected = {
        "type": sampler_type,
        "effective_batch_size": EXPECTED_EFFECTIVE_BATCH,
        "per_device_train_batch_size": EXPECTED_PER_DEVICE_BATCH,
        "gradient_accumulation_steps": EXPECTED_GRADIENT_ACCUMULATION,
        "world_size": EXPECTED_WORLD_SIZE,
        "shuffle_dataset": False,
        "order_is_authoritative": True,
    }
    if sampler_type == SAMPLER_TYPE_V1:
        expected.update(
            {
                "schedule_rows": EXPECTED_ROWS,
                "train_rows": EXPECTED_ROWS,
                "optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
            }
        )
    for key, expected_value in expected.items():
        if contract.get(key) != expected_value:
            raise FixedScheduleSamplerError(
                f"release sampler contract {key} drift: "
                f"expected {expected_value!r}, observed {contract.get(key)!r}"
            )
    if sampler_type == SAMPLER_TYPE_V1:
        expected_rows = EXPECTED_ROWS
        optimizer_steps = EXPECTED_OPTIMIZER_STEPS
    else:
        expected_rows = _integer(
            contract.get("schedule_rows"), label="sampler_contract.schedule_rows"
        )
        train_rows = _integer(
            contract.get("train_rows"), label="sampler_contract.train_rows"
        )
        optimizer_steps = _integer(
            contract.get("optimizer_steps"),
            label="sampler_contract.optimizer_steps",
        )
        if expected_rows <= 0 or optimizer_steps <= 0:
            raise FixedScheduleSamplerError(
                "v2 schedule rows and optimizer steps must be positive"
            )
        if (
            train_rows != expected_rows
            or expected_rows != optimizer_steps * EXPECTED_EFFECTIVE_BATCH
        ):
            raise FixedScheduleSamplerError(
                "v2 dynamic schedule requires train_rows=schedule_rows="
                "optimizer_steps*effective_batch_size"
            )
        expected_window = {"hold": 4, "hike": 2, "cut": 2}
        direction_counts = contract.get("direction_counts")
        expected_direction_counts = {
            direction: optimizer_steps * count
            for direction, count in expected_window.items()
        }
        if (
            contract.get("per_optimizer_window") != expected_window
            or direction_counts != expected_direction_counts
            or sum(expected_direction_counts.values()) != expected_rows
            or contract.get("all_unique_sources_covered") is not True
            or contract.get("optimizer_windows_with_duplicate_source") != 0
            or contract.get("source_population") != "combined_unique_train_only"
            or contract.get("oversampling_layers") != 1
            or contract.get("required_sampler") != "fixed_sequential_schedule_index_v2"
            or contract.get("secondary_shuffle_forbidden") is not True
        ):
            raise FixedScheduleSamplerError("v2 dynamic schedule safety contract drift")
        max_source_repeat = _integer(
            contract.get("max_source_repeat"),
            label="sampler_contract.max_source_repeat",
        )
        if not 1 <= max_source_repeat <= 4:
            raise FixedScheduleSamplerError("v2 source repeat cap drift")

    world_size = _integer(
        getattr(training_args, "world_size", None), label="world_size"
    )
    visible_gpus = _integer(getattr(training_args, "n_gpu", None), label="n_gpu")
    per_device = _integer(
        getattr(training_args, "per_device_train_batch_size", None),
        label="per_device_train_batch_size",
    )
    accumulation = _integer(
        getattr(training_args, "gradient_accumulation_steps", None),
        label="gradient_accumulation_steps",
    )
    max_steps = _integer(getattr(training_args, "max_steps", None), label="max_steps")
    if (
        world_size != EXPECTED_WORLD_SIZE
        or visible_gpus != 1
        or per_device != EXPECTED_PER_DEVICE_BATCH
        or accumulation != EXPECTED_GRADIENT_ACCUMULATION
        or max_steps != optimizer_steps
    ):
        suffix = "24" if sampler_type == SAMPLER_TYPE_V1 else str(optimizer_steps)
        raise FixedScheduleSamplerError(
            "fixed schedule requires world1, one visible GPU, per-device batch1, "
            f"GA8, max_steps{suffix}"
        )
    if bool(getattr(training_args, "group_by_length", False)):
        raise FixedScheduleSamplerError("fixed schedule forbids group_by_length")
    if bool(getattr(training_args, "dataloader_drop_last", False)):
        raise FixedScheduleSamplerError("fixed schedule forbids dataloader_drop_last")
    if bool(getattr(training_args, "shuffle_dataset", True)):
        raise FixedScheduleSamplerError("fixed schedule forbids TRL shuffle_dataset")
    if bool(getattr(training_args, "use_liger_kernel", False)):
        raise FixedScheduleSamplerError(
            "fixed schedule forbids Liger preprocessing that drops schedule_index"
        )
    effective_batch = world_size * per_device * accumulation
    if effective_batch != EXPECTED_EFFECTIVE_BATCH:
        raise FixedScheduleSamplerError("fixed schedule effective batch drift")
    required_strings = (
        "schedule_path",
        "schedule_sha256",
        "train_path",
        "train_sha256",
    )
    if any(
        not isinstance(contract.get(key), str) or not contract[key]
        for key in required_strings
    ):
        raise FixedScheduleSamplerError(
            "fixed schedule path/hash binding is incomplete"
        )
    return {
        "schema_version": (
            "manifest-fixed-schedule-runtime-v1"
            if sampler_type == SAMPLER_TYPE_V1
            else "manifest-fixed-schedule-runtime-v2"
        ),
        "type": sampler_type,
        "release_manifest": {
            "path": str(release_binding["release_manifest_path"]),
            "sha256": str(release_binding["release_manifest_sha256"]),
            "release_id": str(release_binding["release_id"]),
        },
        "schedule": {
            "path": str(contract["schedule_path"]),
            "sha256": str(contract["schedule_sha256"]),
            "rows": expected_rows,
            "order_is_authoritative": True,
        },
        "train_file": {
            "path": str(contract["train_path"]),
            "sha256": str(contract["train_sha256"]),
            "rows": expected_rows,
        },
        "sampler_source": sampler_source_binding(),
        "runtime": {
            "world_size": world_size,
            "visible_gpus": visible_gpus,
            "per_device_train_batch_size": per_device,
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": effective_batch,
            "max_steps": max_steps,
            "shuffle_dataset": False,
            "trl_shuffle_dataset": False,
            "secondary_shuffle": False,
            "use_liger_kernel": False,
        },
    }


def validate_dataset_order(
    dataset: Sized, *, expected_rows: int = EXPECTED_ROWS
) -> None:
    """Require the loaded HF train split to preserve schedule_index ``0..N-1``."""

    expected = _integer(expected_rows, label="expected_rows")
    if expected <= 0:
        raise FixedScheduleSamplerError("expected_rows must be positive")
    if len(dataset) != expected:
        raise FixedScheduleSamplerError(
            f"fixed schedule dataset must contain {expected} rows"
        )
    column_names = getattr(dataset, "column_names", ())
    if "schedule_index" not in column_names:
        raise FixedScheduleSamplerError("fixed schedule dataset lost schedule_index")
    indexes = list(dataset["schedule_index"])
    if indexes != list(range(expected)):
        raise FixedScheduleSamplerError(
            "loaded train rows do not preserve sealed schedule_index order"
        )


class ManifestFixedScheduleSamplerMixin:
    """Trainer mixin whose only train sampler is deterministic sequential order."""

    def _get_train_sampler(self, train_dataset: Sized | None = None):
        # Transformers 4.57 validates/removes columns before invoking the
        # sampler callback.  Verify the authoritative schedule on the original
        # processed dataset, then accept only a same-length column-stripped
        # view.  Column removal is row-preserving, while passing schedule_index
        # into the model would be incorrect.
        original = getattr(self, "train_dataset", None)
        if original is None:
            raise FixedScheduleSamplerError("fixed schedule train dataset is missing")
        expected_rows = getattr(
            self, "_manifest_fixed_schedule_expected_rows", EXPECTED_ROWS
        )
        validate_dataset_order(original, expected_rows=expected_rows)
        selected = self.train_dataset if train_dataset is None else train_dataset
        if len(selected) != expected_rows:
            raise FixedScheduleSamplerError(
                f"fixed schedule sampler view must contain {expected_rows} rows"
            )
        return SequentialSampler(selected)


__all__ = [
    "EXPECTED_EFFECTIVE_BATCH",
    "EXPECTED_GRADIENT_ACCUMULATION",
    "EXPECTED_OPTIMIZER_STEPS",
    "EXPECTED_PER_DEVICE_BATCH",
    "EXPECTED_ROWS",
    "EXPECTED_WORLD_SIZE",
    "FixedScheduleSamplerError",
    "ManifestFixedScheduleSamplerMixin",
    "SAMPLER_TYPE",
    "SAMPLER_TYPE_V1",
    "SAMPLER_TYPE_V2",
    "SUPPORTED_SAMPLER_TYPES",
    "sampler_source_binding",
    "validate_dataset_order",
    "validate_runtime_contract",
]
