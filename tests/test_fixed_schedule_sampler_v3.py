from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, default_data_collator
from trl import SFTConfig

from open_r1.trainer import fixed_schedule_sampler_v3 as sampler
from open_r1.trainer.sft_trainer import ManifestFixedScheduleV3SFTTrainer


class FakeDataset:
    column_names = ["prompt", "completion", "schedule_index"]

    def __init__(self, indexes=range(48)):
        self.indexes = list(indexes)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, key):
        if key == "schedule_index":
            return self.indexes
        return self.indexes[key]


def test_shared_sft_dispatch_imports_additive_v3_compatibility_api():
    assert sampler.SAMPLER_TYPE == sampler.SAMPLER_TYPE_V3
    assert sampler.FixedScheduleV3SamplerError is sampler.CorrectionFixedScheduleError
    assert ManifestFixedScheduleV3SFTTrainer.__mro__[1] is (
        sampler.ManifestFixedScheduleV3SamplerMixin
    )


def _args(**overrides):
    values = {
        "train_sampler": sampler.SAMPLER_TYPE,
        "world_size": 1,
        "n_gpu": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_steps": 6,
        "group_by_length": False,
        "dataloader_drop_last": False,
        "shuffle_dataset": False,
        "use_liger_kernel": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _binding():
    return {
        "release_id": "correction",
        "release_manifest_path": "/release/release_manifest.json",
        "release_manifest_sha256": "a" * 64,
        "sampler_contract": {
            "type": sampler.SAMPLER_TYPE,
            "schedule_rows": 48,
            "train_rows": 48,
            "optimizer_steps": 6,
            "effective_batch_size": 8,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "world_size": 1,
            "shuffle_dataset": False,
            "direction_counts": {"hold": 18, "hike": 18, "cut": 12},
            "population_counts": {"core": 18, "supplement": 30},
            "per_optimizer_window": {"hold": 3, "hike": 3, "cut": 2},
            "per_optimizer_window_population": {"core": 3, "supplement": 5},
            "order_is_authoritative": True,
            "all_unique_sources_covered": True,
            "optimizer_windows_with_duplicate_source": 0,
            "source_population": "pre2009_correction_unique_train_only",
            "oversampling_layers": 0,
            "max_source_repeat": 1,
            "repeat_histogram": {"1": 48},
            "required_sampler": "fixed_sequential_schedule_index_v3",
            "secondary_shuffle_forbidden": True,
            "schedule_path": "/release/manifests/sampler_schedule.jsonl",
            "schedule_sha256": "b" * 64,
            "train_path": "/release/decision_sft/train.jsonl",
            "train_sha256": "c" * 64,
        },
    }


def test_v3_runtime_contract_and_sequential_sampler():
    receipt = sampler.validate_runtime_contract(
        training_args=_args(), release_binding=_binding()
    )
    assert receipt["schema_version"] == "manifest-fixed-schedule-runtime-v3"
    assert receipt["schedule"]["rows"] == 48

    class Trainer(sampler.ManifestFixedScheduleV3SamplerMixin):
        train_dataset = FakeDataset()

    assert list(Trainer()._get_train_sampler()) == list(range(48))


class _TinySFTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(_attn_implementation="eager")

    def add_model_tags(self, *_tags):
        return None

    def forward(self, input_ids=None, labels=None, **_kwargs):
        values = input_ids.float() * self.weight
        return {"loss": values.mean() * 0.0, "logits": values.unsqueeze(-1)}


def test_real_sft_trainer_column_view_keeps_original_v3_schedule(tmp_path):
    """TRL may strip schedule metadata, but its sampler must stay sequential."""

    original = Dataset.from_dict(
        {
            "input_ids": [[index] for index in range(48)],
            "labels": [[index] for index in range(48)],
            "schedule_index": list(range(48)),
            "training_row_id": [f"row-{index}" for index in range(48)],
        }
    )
    backend = Tokenizer(WordLevel({"<unk>": 0, "<pad>": 1, "<eos>": 2}, unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
    )
    trainer = ManifestFixedScheduleV3SFTTrainer(
        model=_TinySFTModel(),
        args=SFTConfig(
            output_dir=str(tmp_path),
            per_device_train_batch_size=1,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            remove_unused_columns=True,
            report_to=[],
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=None,
            gradient_checkpointing=False,
        ),
        train_dataset=original,
        data_collator=default_data_collator,
        processing_class=tokenizer,
    )

    observed = []
    for batch in trainer.get_train_dataloader():
        assert "schedule_index" not in batch
        assert "training_row_id" not in batch
        observed.extend(int(value) for value in batch["input_ids"].flatten().tolist())
    assert observed == list(range(48))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("direction_counts", {"hold": 24, "hike": 12, "cut": 12}),
        ("population_counts", {"core": 19, "supplement": 29}),
        ("per_optimizer_window", {"hold": 4, "hike": 2, "cut": 2}),
        ("per_optimizer_window_population", {"core": 4, "supplement": 4}),
        ("oversampling_layers", 1),
        ("max_source_repeat", 2),
        ("repeat_histogram", {"1": 47, "2": 1}),
        ("source_population", "combined_unique_train_only"),
    ],
)
def test_v3_rejects_contract_drift(key, value):
    binding = deepcopy(_binding())
    binding["sampler_contract"][key] = value
    with pytest.raises(sampler.FixedScheduleV3SamplerError, match=key):
        sampler.validate_runtime_contract(
            training_args=_args(), release_binding=binding
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("max_steps", 7),
        ("gradient_accumulation_steps", 4),
        ("n_gpu", 2),
        ("shuffle_dataset", True),
        ("group_by_length", True),
        ("dataloader_drop_last", True),
        ("use_liger_kernel", True),
    ],
)
def test_v3_rejects_runtime_drift(key, value):
    with pytest.raises(sampler.FixedScheduleV3SamplerError):
        sampler.validate_runtime_contract(
            training_args=_args(**{key: value}), release_binding=_binding()
        )


def test_v3_rejects_missing_or_reordered_schedule_index():
    sampler.validate_dataset_order(FakeDataset())
    with pytest.raises(sampler.FixedScheduleV3SamplerError, match="preserve"):
        sampler.validate_dataset_order(FakeDataset(reversed(range(48))))

    missing = FakeDataset()
    missing.column_names = ["prompt", "completion"]
    with pytest.raises(sampler.FixedScheduleV3SamplerError, match="lost"):
        sampler.validate_dataset_order(missing)
