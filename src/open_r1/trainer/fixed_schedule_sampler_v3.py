"""Correction-only fixed sequential sampler contract.

This module is additive: the historical v1/v2 sampler source remains byte-for-
byte unchanged so its sealed runtime receipts stay verifiable.  Version 3 is
restricted to the 48-row, six-step chk4 pre-2009 correction release with
3 hold / 3 hike / 2 cut and core=3 / supplement=5 in every optimizer window.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sized
from pathlib import Path
from typing import Any

from torch.utils.data import SequentialSampler

from open_r1.trainer.fixed_schedule_sampler import (
    ManifestFixedScheduleSamplerMixin as LegacyManifestFixedScheduleSamplerMixin,
)


SAMPLER_TYPE_V3 = "manifest_fixed_schedule_v3"
# Additive compatibility names used by the shared SFT/trainer dispatch.  The
# correction workflow's original ``*_V3`` API remains unchanged.
SAMPLER_TYPE = SAMPLER_TYPE_V3
SUPPORTED_SAMPLER_TYPES = frozenset({SAMPLER_TYPE_V3})
EXPECTED_ROWS = 48
EXPECTED_STEPS = 6
EXPECTED_OPTIMIZER_STEPS = EXPECTED_STEPS
EXPECTED_BATCH = 1
EXPECTED_PER_DEVICE_BATCH = EXPECTED_BATCH
EXPECTED_ACCUMULATION = 8
EXPECTED_GRADIENT_ACCUMULATION = EXPECTED_ACCUMULATION
EXPECTED_EFFECTIVE_BATCH = 8
EXPECTED_WORLD_SIZE = 1
EXPECTED_DIRECTIONS = {"hold": 18, "hike": 18, "cut": 12}
EXPECTED_WINDOW_DIRECTIONS = {"hold": 3, "hike": 3, "cut": 2}
EXPECTED_WINDOW = EXPECTED_WINDOW_DIRECTIONS
EXPECTED_POPULATIONS = {"core": 18, "supplement": 30}
EXPECTED_WINDOW_POPULATIONS = {"core": 3, "supplement": 5}
EXPECTED_SOURCE_POPULATION = "pre2009_correction_unique_train_only"

# Preserve the workflow-specific public alias without using it for the shared
# v3 trainer: the legacy mixin defaults to 192 rows until a caller mutates an
# instance attribute, which is unsafe for this fixed 48-row release.
ManifestFixedScheduleSamplerMixin = LegacyManifestFixedScheduleSamplerMixin


class CorrectionFixedScheduleError(RuntimeError):
    """The correction schedule cannot be consumed without reordering."""


FixedScheduleV3SamplerError = CorrectionFixedScheduleError


def validate_dataset_order(dataset: Sized) -> None:
    """Require the correction train split to preserve indices ``0..47``."""

    if len(dataset) != EXPECTED_ROWS:
        raise CorrectionFixedScheduleError(
            f"correction dataset must contain {EXPECTED_ROWS} rows"
        )
    column_names = getattr(dataset, "column_names", ())
    if "schedule_index" not in column_names:
        raise CorrectionFixedScheduleError("correction dataset lost schedule_index")
    if list(dataset["schedule_index"]) != list(range(EXPECTED_ROWS)):
        raise CorrectionFixedScheduleError(
            "correction rows do not preserve sealed schedule_index order"
        )


class ManifestFixedScheduleV3SamplerMixin:
    """Correction-only sequential sampler that cannot inherit v1 row defaults."""

    def _get_train_sampler(self, train_dataset: Sized | None = None):
        original = getattr(self, "train_dataset", None)
        if original is None:
            raise CorrectionFixedScheduleError(
                "correction fixed-schedule train dataset is missing"
            )
        validate_dataset_order(original)
        selected = self.train_dataset if train_dataset is None else train_dataset
        if len(selected) != EXPECTED_ROWS:
            raise CorrectionFixedScheduleError(
                f"correction sampler view must contain {EXPECTED_ROWS} rows"
            )
        return SequentialSampler(selected)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sampler_source_binding() -> dict[str, Any]:
    source = Path(__file__).resolve()
    legacy_source = source.with_name("fixed_schedule_sampler.py")
    return {
        "path": str(source),
        "sha256": _sha256_file(source),
        "class": "CorrectionManifestFixedScheduleSFTTrainer",
        "sampler": "torch.utils.data.SequentialSampler",
        "legacy_mixin_source": {
            "path": str(legacy_source),
            "sha256": _sha256_file(legacy_source),
            "class": "ManifestFixedScheduleSamplerMixin",
        },
    }


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CorrectionFixedScheduleError(f"{label} must be an integer")
    return value


def validate_runtime_contract(
    *, training_args: Any, release_binding: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the exact correction release/runtime sampler boundary."""

    if getattr(training_args, "train_sampler", None) != SAMPLER_TYPE_V3:
        raise CorrectionFixedScheduleError(
            f"correction release requires train_sampler={SAMPLER_TYPE_V3}"
        )
    contract = release_binding.get("sampler_contract")
    if not isinstance(contract, Mapping):
        raise CorrectionFixedScheduleError("correction release has no sampler contract")
    exact = {
        "type": SAMPLER_TYPE_V3,
        "schedule_rows": EXPECTED_ROWS,
        "train_rows": EXPECTED_ROWS,
        "optimizer_steps": EXPECTED_STEPS,
        "effective_batch_size": EXPECTED_EFFECTIVE_BATCH,
        "per_device_train_batch_size": EXPECTED_BATCH,
        "gradient_accumulation_steps": EXPECTED_ACCUMULATION,
        "world_size": EXPECTED_WORLD_SIZE,
        "shuffle_dataset": False,
        "direction_counts": EXPECTED_DIRECTIONS,
        "per_optimizer_window": EXPECTED_WINDOW_DIRECTIONS,
        "population_counts": EXPECTED_POPULATIONS,
        "per_optimizer_window_population": EXPECTED_WINDOW_POPULATIONS,
        "order_is_authoritative": True,
        "all_unique_sources_covered": True,
        "optimizer_windows_with_duplicate_source": 0,
        "source_population": EXPECTED_SOURCE_POPULATION,
        "oversampling_layers": 0,
        "max_source_repeat": 1,
        "repeat_histogram": {"1": EXPECTED_ROWS},
        "required_sampler": "fixed_sequential_schedule_index_v3",
        "secondary_shuffle_forbidden": True,
    }
    for key, expected in exact.items():
        if contract.get(key) != expected:
            raise CorrectionFixedScheduleError(
                f"correction sampler {key} drift: expected {expected!r}, "
                f"observed {contract.get(key)!r}"
            )
    for key in ("schedule_path", "schedule_sha256", "train_path", "train_sha256"):
        if not isinstance(contract.get(key), str) or not contract[key]:
            raise CorrectionFixedScheduleError(f"correction sampler {key} missing")

    runtime = {
        "world_size": _integer(getattr(training_args, "world_size", None), label="world_size"),
        "visible_gpus": _integer(getattr(training_args, "n_gpu", None), label="n_gpu"),
        "per_device_train_batch_size": _integer(
            getattr(training_args, "per_device_train_batch_size", None),
            label="per_device_train_batch_size",
        ),
        "gradient_accumulation_steps": _integer(
            getattr(training_args, "gradient_accumulation_steps", None),
            label="gradient_accumulation_steps",
        ),
        "max_steps": _integer(getattr(training_args, "max_steps", None), label="max_steps"),
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
    if runtime != expected_runtime:
        raise CorrectionFixedScheduleError(
            f"correction sampler runtime drift: {runtime!r}"
        )
    return {
        "schema_version": "manifest-fixed-schedule-runtime-v3",
        "type": SAMPLER_TYPE_V3,
        "release_manifest": {
            "path": str(release_binding["release_manifest_path"]),
            "sha256": str(release_binding["release_manifest_sha256"]),
            "release_id": str(release_binding["release_id"]),
        },
        "schedule": {
            "path": str(contract["schedule_path"]),
            "sha256": str(contract["schedule_sha256"]),
            "rows": EXPECTED_ROWS,
            "order_is_authoritative": True,
        },
        "train_file": {
            "path": str(contract["train_path"]),
            "sha256": str(contract["train_sha256"]),
            "rows": EXPECTED_ROWS,
        },
        "sampler_source": sampler_source_binding(),
        "runtime": {
            "world_size": 1,
            "visible_gpus": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "effective_batch_size": 8,
            "max_steps": 6,
            "shuffle_dataset": False,
            "trl_shuffle_dataset": False,
            "secondary_shuffle": False,
            "use_liger_kernel": False,
        },
    }


__all__ = [
    "CorrectionFixedScheduleError",
    "EXPECTED_ACCUMULATION",
    "EXPECTED_BATCH",
    "EXPECTED_DIRECTIONS",
    "EXPECTED_EFFECTIVE_BATCH",
    "EXPECTED_GRADIENT_ACCUMULATION",
    "EXPECTED_OPTIMIZER_STEPS",
    "EXPECTED_PER_DEVICE_BATCH",
    "EXPECTED_POPULATIONS",
    "EXPECTED_ROWS",
    "EXPECTED_SOURCE_POPULATION",
    "EXPECTED_STEPS",
    "EXPECTED_WINDOW",
    "EXPECTED_WINDOW_DIRECTIONS",
    "EXPECTED_WINDOW_POPULATIONS",
    "EXPECTED_WORLD_SIZE",
    "FixedScheduleV3SamplerError",
    "ManifestFixedScheduleSamplerMixin",
    "ManifestFixedScheduleV3SamplerMixin",
    "SAMPLER_TYPE",
    "SAMPLER_TYPE_V3",
    "SUPPORTED_SAMPLER_TYPES",
    "sampler_source_binding",
    "validate_dataset_order",
    "validate_runtime_contract",
]
