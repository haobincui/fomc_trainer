from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import open_r1.data_loader as data_loader
import open_r1.trainer.trainer as trainer_module
from open_r1.trainer.trainer import Trainer


class _DatasetTrainer(Trainer):
    def load_trainer(self):
        raise NotImplementedError

    def plot_customized_curve(self):
        raise NotImplementedError


class _Logger:
    def info(self, *_args, **_kwargs) -> None:
        pass


class _Split:
    column_names: list[str] = []


class _Dataset(dict):
    def __init__(self) -> None:
        super().__init__({"train": _Split(), "validation": _Split()})
        self.map_calls = 0

    def map(self, _function):
        self.map_calls += 1
        return self


def _write_split(path: Path) -> None:
    path.write_text('{"prompt":"p","response":"r","provided_data":"d"}\n')


def test_legacy_glob_discovery_rejects_multiple_candidates(tmp_path: Path) -> None:
    _write_split(tmp_path / "train.jsonl")
    _write_split(tmp_path / "shadow_train.jsonl")

    with pytest.raises(ValueError, match="Multiple files match split 'train'"):
        data_loader.load_train_eval_datasets(tmp_path)


def test_explicit_manifest_bindings_bypass_decoy_glob_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    _write_split(train)
    _write_split(evaluation)
    _write_split(tmp_path / "shadow_train.jsonl")
    _write_split(tmp_path / "shadow_eval.jsonl")
    captured: dict[str, object] = {}

    def fake_load_dataset(kind: str, *, data_files):
        captured.update({"kind": kind, "data_files": data_files})
        return {"train": [1], "validation": [1]}

    monkeypatch.setattr(data_loader, "load_dataset", fake_load_dataset)
    result = data_loader.load_train_eval_datasets(
        tmp_path,
        split_files={"train": train, "validation": evaluation},
    )

    assert captured == {
        "kind": "json",
        "data_files": {
            "train": str(train.resolve()),
            "validation": str(evaluation.resolve()),
        },
    }
    assert result == {"train": [1], "validation": [1]}


def test_explicit_manifest_binding_rejects_path_outside_dataset(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    train = dataset / "train.jsonl"
    evaluation = dataset / "eval.jsonl"
    outside = tmp_path / "outside.jsonl"
    for path in (train, evaluation, outside):
        _write_split(path)

    with pytest.raises(ValueError, match="escapes dataset directory"):
        data_loader.load_train_eval_datasets(
            dataset,
            split_files={"train": outside, "validation": evaluation},
        )


def test_trainer_loads_the_exact_verified_manifest_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_root = tmp_path / "release"
    dataset = release_root / "analysis_sft"
    dataset.mkdir(parents=True)
    train = dataset / "train.jsonl"
    evaluation = dataset / "eval.jsonl"
    for path in (train, evaluation):
        _write_split(path)
    manifest = release_root / "release_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    loaded = _Dataset()
    captured: dict[str, object] = {}

    def fake_verify_clean_sft_release(**kwargs):
        captured["verify"] = kwargs
        return {
            "release_id": "clean-test",
            "split_files": {
                "train": {"path": "analysis_sft/train.jsonl"},
                "eval": {"path": "analysis_sft/eval.jsonl"},
                "test": {"path": "analysis_sft/test.jsonl"},
            },
        }

    def fake_load_train_eval_datasets(data_path, *, split_files):
        captured["loader"] = {
            "data_path": data_path,
            "split_files": split_files,
        }
        return loaded

    monkeypatch.setattr(
        trainer_module, "verify_clean_sft_release", fake_verify_clean_sft_release
    )
    monkeypatch.setattr(
        trainer_module, "load_train_eval_datasets", fake_load_train_eval_datasets
    )
    script_args = SimpleNamespace(
        dataset_name=str(dataset),
        dataset_release_manifest=str(manifest),
        dataset_release_manifest_sha256="a" * 64,
        dataset_prompt_column="prompt",
        user_prompt_suffix=None,
    )
    trainer = _DatasetTrainer(
        script_args,
        SimpleNamespace(system_prompt=None),
        SimpleNamespace(),
        SimpleNamespace(),
    )
    trainer._logger = _Logger()

    result = trainer.load_dataset()

    assert result is loaded
    assert loaded.map_calls == 1
    assert captured["verify"] == {
        "dataset_dir": str(dataset),
        "manifest_path": str(manifest),
        "expected_manifest_sha256": "a" * 64,
    }
    assert captured["loader"] == {
        "data_path": str(dataset),
        "split_files": {
            "train": train.resolve(),
            "validation": evaluation.resolve(),
        },
    }
