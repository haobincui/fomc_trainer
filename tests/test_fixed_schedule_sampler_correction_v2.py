from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from transformers import TrainingArguments

from jobs.retrain_v2.train_chk4_pre2009_correction_v2 import (
    ManifestFixedScheduleCorrectionV2SFTTrainer,
)
from open_r1.trainer import fixed_schedule_sampler_correction_v2 as sampler


class FakeDataset:
    column_names = [
        "schedule_index",
        "optimizer_step",
        "microbatch_slot",
        "source_sample_id",
        "source_repeat_index",
        "source_repeat_total",
        "population_role",
        "direction",
    ]

    def __init__(self) -> None:
        # Fifteen sources repeat in a later optimizer window while eighteen
        # appear once. No source repeats inside one eight-row window, and each
        # repeat keeps its population/direction cell.
        cells = {
            "core:hold": [f"ch-{index}" for index in range(6)],
            "supplement:hold": ["sh-0", "sh-1", "sh-2", "sh-3", "sh-0", "sh-1"],
            "core:hike": [
                "chi-0",
                "chi-1",
                "chi-2",
                "chi-3",
                "chi-4",
                "chi-5",
                "chi-6",
                "chi-7",
                "chi-0",
                "chi-1",
                "chi-2",
                "chi-3",
            ],
            "supplement:hike": [
                "shi-0",
                "shi-1",
                "shi-2",
                "shi-3",
                "shi-4",
                "shi-5",
                "shi-6",
                "shi-7",
                "shi-0",
                "shi-1",
                "shi-2",
                "shi-3",
            ],
            "core:cut": ["cc-0", "cc-1", "cc-2", "cc-0", "cc-1", "cc-2"],
            "supplement:cut": ["sc-0", "sc-1", "sc-2", "sc-3", "sc-0", "sc-1"],
        }
        scheduled_ids = []
        for step in range(6):
            scheduled_ids.extend(
                [
                    cells["core:hold"][step],
                    cells["supplement:hold"][step],
                    cells["core:hike"][2 * step],
                    cells["core:hike"][2 * step + 1],
                    cells["supplement:hike"][2 * step],
                    cells["supplement:hike"][2 * step + 1],
                    cells["core:cut"][step],
                    cells["supplement:cut"][step],
                ]
            )
        source_ids = sorted(set(scheduled_ids))
        occurrences: dict[str, int] = {}
        totals = {source_id: scheduled_ids.count(source_id) for source_id in source_ids}
        repeat_indices = []
        repeat_totals = []
        for source_id in scheduled_ids:
            repeat_indices.append(occurrences.get(source_id, 0))
            repeat_totals.append(totals[source_id])
            occurrences[source_id] = occurrences.get(source_id, 0) + 1
        self.values = {
            "schedule_index": list(range(48)),
            "optimizer_step": [step for step in range(6) for _ in range(8)],
            "microbatch_slot": list(range(8)) * 6,
            "source_sample_id": scheduled_ids,
            "source_repeat_index": repeat_indices,
            "source_repeat_total": repeat_totals,
            "population_role": [
                "core",
                "supplement",
                "core",
                "core",
                "supplement",
                "supplement",
                "core",
                "supplement",
            ]
            * 6,
            "direction": ["hold", "hold", "hike", "hike", "hike", "hike", "cut", "cut"]
            * 6,
        }

    def __len__(self) -> int:
        return 48

    def __getitem__(self, key: str):
        return self.values[key]


def _binding() -> dict[str, object]:
    return {
        "release_manifest_path": "/tmp/release.json",
        "release_manifest_sha256": "a" * 64,
        "release_id": "correction-v2",
        "sampler_contract": {
            "type": sampler.SAMPLER_TYPE,
            "schedule_rows": 48,
            "train_rows": 48,
            "unique_source_rows": 33,
            "optimizer_steps": 6,
            "effective_batch_size": 8,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "world_size": 1,
            "shuffle_dataset": False,
            "direction_counts": {"hold": 12, "hike": 24, "cut": 12},
            "per_optimizer_window": {"hold": 2, "hike": 4, "cut": 2},
            "population_counts": {"core": 24, "supplement": 24},
            "per_optimizer_window_population": {"core": 4, "supplement": 4},
            "per_optimizer_window_cells": {
                "core:hold": 1,
                "supplement:hold": 1,
                "core:hike": 2,
                "supplement:hike": 2,
                "core:cut": 1,
                "supplement:cut": 1,
            },
            "unique_source_cells": {
                "core:hold": 6,
                "supplement:hold": 4,
                "core:hike": 8,
                "supplement:hike": 8,
                "core:cut": 3,
                "supplement:cut": 4,
            },
            "repeated_source_cells": {
                "core:hold": 0,
                "supplement:hold": 2,
                "core:hike": 4,
                "supplement:hike": 4,
                "core:cut": 3,
                "supplement:cut": 2,
            },
            "order_is_authoritative": True,
            "optimizer_windows_with_duplicate_source": 0,
            "source_population": "pre2009_correction_v2_burned_plus_fresh",
            "max_source_repeat": 2,
            "repeat_histogram": {"1": 18, "2": 15},
            "required_sampler": "fixed_sequential_schedule_index_correction_v2",
            "secondary_shuffle_forbidden": True,
            "schedule_path": "/tmp/schedule.jsonl",
            "schedule_sha256": "b" * 64,
            "train_path": "/tmp/train.jsonl",
            "train_sha256": "c" * 64,
        },
    }


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        train_sampler=sampler.SAMPLER_TYPE,
        world_size=1,
        n_gpu=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=6,
        shuffle_dataset=False,
        group_by_length=False,
        dataloader_drop_last=False,
        use_liger_kernel=False,
    )


def test_dataset_order_accepts_six_complete_unique_windows() -> None:
    sampler.validate_dataset_order(FakeDataset())


def test_dataset_order_rejects_duplicate_source_within_window() -> None:
    dataset = FakeDataset()
    dataset.values["source_sample_id"][1] = dataset.values["source_sample_id"][0]
    with pytest.raises(sampler.CorrectionV2FixedScheduleError):
        sampler.validate_dataset_order(dataset)


def test_dataset_order_rejects_repeat_metadata_drift() -> None:
    dataset = FakeDataset()
    dataset.values["source_repeat_total"][0] = 2
    with pytest.raises(sampler.CorrectionV2FixedScheduleError):
        sampler.validate_dataset_order(dataset)


def test_dataset_order_rejects_window_cell_drift() -> None:
    dataset = FakeDataset()
    dataset.values["population_role"][2] = "supplement"
    with pytest.raises(sampler.CorrectionV2FixedScheduleError):
        sampler.validate_dataset_order(dataset)


def test_runtime_contract_accepts_frozen_v2_shape() -> None:
    result = sampler.validate_runtime_contract(
        training_args=_args(), release_binding=_binding()
    )
    assert result["schedule"]["rows"] == 48
    assert result["schedule"]["unique_sources"] == 33
    assert result["repeat_histogram"] == {"1": 18, "2": 15}


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("unique_source_rows", 34),
        ("max_source_repeat", 1),
        ("repeat_histogram", {"1": 48}),
        ("per_optimizer_window_population", {"core": 3, "supplement": 5}),
        ("unique_source_cells", {"core:hold": 5}),
        ("repeated_source_cells", {"core:hike": 5}),
    ],
)
def test_runtime_contract_fails_closed_on_release_drift(
    key: str, value: object
) -> None:
    binding = _binding()
    binding["sampler_contract"][key] = value  # type: ignore[index]
    with pytest.raises(sampler.CorrectionV2FixedScheduleError):
        sampler.validate_runtime_contract(
            training_args=_args(), release_binding=binding
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("world_size", 2),
        ("n_gpu", 0),
        ("per_device_train_batch_size", 2),
        ("gradient_accumulation_steps", 4),
        ("max_steps", 7),
        ("shuffle_dataset", True),
        ("group_by_length", True),
        ("dataloader_drop_last", True),
        ("use_liger_kernel", True),
    ],
)
def test_runtime_contract_fails_closed_on_runtime_drift(
    key: str, value: object
) -> None:
    args = _args()
    setattr(args, key, value)
    with pytest.raises(sampler.CorrectionV2FixedScheduleError, match="runtime drift"):
        sampler.validate_runtime_contract(
            training_args=args, release_binding=_binding()
        )


def test_real_trainer_column_stripping_keeps_sequential_sampler_validation(
    tmp_path,
) -> None:
    source = FakeDataset()
    full = Dataset.from_dict(
        source.values
        | {
            "input_ids": [[index] for index in range(48)],
            "labels": [[index] for index in range(48)],
        }
    )

    class TinyModel(torch.nn.Module):
        def forward(self, input_ids=None, labels=None):
            return {"loss": torch.tensor(0.0), "logits": input_ids, "labels": labels}

    trainer = object.__new__(ManifestFixedScheduleCorrectionV2SFTTrainer)
    trainer.train_dataset = full
    trainer.model = TinyModel()
    trainer.args = TrainingArguments(output_dir=str(tmp_path), report_to=[])
    trainer.label_names = ["labels"]
    trainer._signature_columns = None
    trainer._is_vision_dataset = False

    stripped = trainer._remove_unused_columns(full, description="training")
    assert "schedule_index" not in stripped.column_names
    selected = trainer._get_train_sampler(stripped)
    assert list(selected) == list(range(48))
