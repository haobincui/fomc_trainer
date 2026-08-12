from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments, default_data_collator

from open_r1.trainer.fixed_schedule_sampler import (
    FixedScheduleSamplerError,
    ManifestFixedScheduleSamplerMixin,
    SAMPLER_TYPE,
    SAMPLER_TYPE_V2,
    sampler_source_binding,
    validate_dataset_order,
    validate_runtime_contract,
)


class _Dataset:
    column_names = ["schedule_index", "prompt", "response"]

    def __init__(self, indexes=None):
        self.indexes = list(range(192)) if indexes is None else indexes

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, key):
        if key == "schedule_index":
            return self.indexes
        return {"schedule_index": self.indexes[key]}


class _Trainer(ManifestFixedScheduleSamplerMixin):
    def __init__(self, dataset):
        self.train_dataset = dataset


class _RealFixedTrainer(ManifestFixedScheduleSamplerMixin, Trainer):
    pass


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids=None, labels=None):
        values = input_ids.float() * self.weight
        return {"loss": values.mean() * 0.0, "logits": values.unsqueeze(-1)}


def _args(**overrides):
    values = {
        "train_sampler": SAMPLER_TYPE,
        "world_size": 1,
        "n_gpu": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_steps": 24,
        "group_by_length": False,
        "dataloader_drop_last": False,
        "shuffle_dataset": False,
        "use_liger_kernel": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _release():
    return {
        "release_id": "hier-balanced-v1",
        "release_manifest_path": "/sealed/release_manifest.json",
        "release_manifest_sha256": "a" * 64,
        "sampler_contract": {
            "type": SAMPLER_TYPE,
            "schedule_path": "/sealed/manifests/sampler_schedule.jsonl",
            "schedule_sha256": "b" * 64,
            "schedule_rows": 192,
            "train_path": "/sealed/decision_sft/train.jsonl",
            "train_sha256": "c" * 64,
            "train_rows": 192,
            "effective_batch_size": 8,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "world_size": 1,
            "optimizer_steps": 24,
            "shuffle_dataset": False,
            "order_is_authoritative": True,
        },
    }


def _release_v2(*, rows: int = 312, steps: int = 39):
    return {
        "release_id": "pre2009-balanced-v1",
        "release_manifest_path": "/sealed/release_manifest.json",
        "release_manifest_sha256": "d" * 64,
        "sampler_contract": {
            "type": SAMPLER_TYPE_V2,
            "schedule_path": "/sealed/manifests/sampler_schedule.jsonl",
            "schedule_sha256": "e" * 64,
            "schedule_rows": rows,
            "train_path": "/sealed/decision_sft/train.jsonl",
            "train_sha256": "f" * 64,
            "train_rows": rows,
            "effective_batch_size": 8,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "world_size": 1,
            "optimizer_steps": steps,
            "shuffle_dataset": False,
            "direction_counts": {
                "hold": steps * 4,
                "hike": steps * 2,
                "cut": steps * 2,
            },
            "per_optimizer_window": {"hold": 4, "hike": 2, "cut": 2},
            "max_source_repeat": 4,
            "all_unique_sources_covered": True,
            "optimizer_windows_with_duplicate_source": 0,
            "source_population": "combined_unique_train_only",
            "oversampling_layers": 1,
            "order_is_authoritative": True,
            "required_sampler": "fixed_sequential_schedule_index_v2",
            "secondary_shuffle_forbidden": True,
        },
    }


def test_fixed_sampler_yields_sealed_order_without_shuffle() -> None:
    dataset = _Dataset()
    sampler = _Trainer(dataset)._get_train_sampler()
    assert list(sampler) == list(range(192))
    validate_dataset_order(dataset)


def test_real_trainer_column_stripping_keeps_fixed_row_order(tmp_path) -> None:
    dataset = Dataset.from_dict(
        {
            "input_ids": [[index] for index in range(192)],
            "labels": [[index] for index in range(192)],
            "schedule_index": list(range(192)),
            "training_row_id": [f"row-{index}" for index in range(192)],
        }
    )
    trainer = _RealFixedTrainer(
        model=_TinyModel(),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=1,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            remove_unused_columns=True,
            report_to=[],
        ),
        train_dataset=dataset,
        data_collator=default_data_collator,
    )

    observed: list[int] = []
    for batch in trainer.get_train_dataloader():
        assert "schedule_index" not in batch
        assert "training_row_id" not in batch
        observed.extend(int(value) for value in batch["input_ids"].flatten().tolist())
    assert observed == list(range(192))


def test_runtime_receipt_binds_schedule_source_world_and_effective_batch() -> None:
    receipt = validate_runtime_contract(
        training_args=_args(), release_binding=_release()
    )
    assert receipt["type"] == SAMPLER_TYPE
    assert receipt["schedule"]["sha256"] == "b" * 64
    assert receipt["runtime"] == {
        "world_size": 1,
        "visible_gpus": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "effective_batch_size": 8,
        "max_steps": 24,
        "shuffle_dataset": False,
        "trl_shuffle_dataset": False,
        "secondary_shuffle": False,
        "use_liger_kernel": False,
    }
    source = sampler_source_binding()
    assert receipt["sampler_source"] == source
    assert len(source["sha256"]) == 64


def test_v2_runtime_uses_manifest_bound_dynamic_rows_and_steps() -> None:
    args = _args(train_sampler=SAMPLER_TYPE_V2, max_steps=39)
    receipt = validate_runtime_contract(
        training_args=args,
        release_binding=_release_v2(),
    )
    assert receipt["schema_version"] == "manifest-fixed-schedule-runtime-v2"
    assert receipt["type"] == SAMPLER_TYPE_V2
    assert receipt["schedule"]["rows"] == 312
    assert receipt["runtime"]["max_steps"] == 39

    dataset = _Dataset(list(range(312)))
    trainer = _Trainer(dataset)
    trainer._manifest_fixed_schedule_expected_rows = 312
    assert list(trainer._get_train_sampler()) == list(range(312))
    validate_dataset_order(dataset, expected_rows=312)


def test_v2_runtime_rejects_non_integral_window_or_config_step_drift() -> None:
    with pytest.raises(FixedScheduleSamplerError, match="optimizer_steps"):
        validate_runtime_contract(
            training_args=_args(train_sampler=SAMPLER_TYPE_V2, max_steps=39),
            release_binding=_release_v2(rows=311),
        )
    with pytest.raises(FixedScheduleSamplerError, match="max_steps39"):
        validate_runtime_contract(
            training_args=_args(train_sampler=SAMPLER_TYPE_V2, max_steps=38),
            release_binding=_release_v2(),
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"world_size": 2}, "world1"),
        ({"n_gpu": 2}, "one visible GPU"),
        ({"per_device_train_batch_size": 2}, "batch1"),
        ({"gradient_accumulation_steps": 1}, "GA8"),
        ({"max_steps": 23}, "max_steps24"),
        ({"group_by_length": True}, "group_by_length"),
        ({"dataloader_drop_last": True}, "dataloader_drop_last"),
        ({"shuffle_dataset": True}, "TRL shuffle_dataset"),
        ({"use_liger_kernel": True}, "Liger preprocessing"),
    ],
)
def test_runtime_contract_rejects_reordering_or_batch_drift(override, message) -> None:
    with pytest.raises(FixedScheduleSamplerError, match=message):
        validate_runtime_contract(
            training_args=_args(**override), release_binding=_release()
        )


def test_loaded_dataset_order_drift_fails_closed() -> None:
    indexes = list(range(192))
    indexes[0], indexes[1] = indexes[1], indexes[0]
    with pytest.raises(FixedScheduleSamplerError, match="schedule_index order"):
        validate_dataset_order(_Dataset(indexes))


def test_release_cannot_claim_shuffle_or_wrong_schedule_size() -> None:
    release = _release()
    release["sampler_contract"]["shuffle_dataset"] = True
    with pytest.raises(FixedScheduleSamplerError, match="shuffle_dataset"):
        validate_runtime_contract(training_args=_args(), release_binding=release)
    release = _release()
    release["sampler_contract"]["schedule_rows"] = 191
    with pytest.raises(FixedScheduleSamplerError, match="schedule_rows"):
        validate_runtime_contract(training_args=_args(), release_binding=release)
