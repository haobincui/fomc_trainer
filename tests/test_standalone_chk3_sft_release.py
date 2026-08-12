from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_r1.trainer.dataset_release import (
    STANDALONE_CHK3_BINDING_SCHEMA,
    STANDALONE_CHK3_DIRECT_SCOPE,
    STANDALONE_CHK3_RELEASE_SCHEMA,
    DatasetReleaseValidationError,
    sha256_file,
    verify_standalone_chk3_direct_sft_release,
)
from open_r1.trainer.trainer import Trainer


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_release(root: Path) -> tuple[Path, Path]:
    dataset = root / "minutes_alignment"
    manifests = dataset / "manifests"
    manifests.mkdir(parents=True)
    split_counts = {"train": 1, "validation": 1, "test": 1}
    source_names = {"train": "train", "validation": "eval", "test": "test"}
    response = "Preserve the supplied facts.\n</think>\nParticipants noted the supplied developments."
    reasoning, minutes = response.split("\n</think>\n", 1)
    for split in split_counts:
        prompt = f"Rewrite the {split} analysis."
        row = {"prompt": prompt, "response": response}
        (dataset / f"{split}.jsonl").write_text(
            json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        manifest_row = {
            "schema_version": STANDALONE_CHK3_RELEASE_SCHEMA,
            "sample_id": f"sample-{split}",
            "split": split,
            "source_split": source_names[split],
            "source_index": 0,
            "prompt_sha256": _sha(prompt),
            "response_sha256": _sha(response),
            "reasoning_sha256": _sha(reasoning),
            "minutes_sha256": _sha(minutes),
            "analysis_sha256": _sha(f"analysis-{split}"),
            "source_analysis_sha256": _sha(f"source-analysis-{split}"),
            "source_response_sha256": _sha(f"source-response-{split}"),
            "prompt_tokens": 8,
            "reasoning_tokens": 5,
            "completion_tokens": 14,
            "total_tokens": 22,
        }
        (manifests / f"{split}.jsonl").write_text(
            json.dumps(manifest_row, sort_keys=True) + "\n", encoding="utf-8"
        )

    audit = {
        "schema_version": STANDALONE_CHK3_RELEASE_SCHEMA,
        "status": "passed",
        "intended_use": "chk3 SFT: analysis to formal FOMC Minutes paragraph",
        "split_counts": split_counts,
        "total_rows": 3,
        "unique_sample_ids": 3,
        "duplicate_sample_ids": 0,
        "missing_required_fields": 0,
        "invalid_response_boundaries": 0,
        "analysis_evidence_citations": 0,
        "reasoning_meta_contamination": 0,
        "transport_wrapped_analyses": 0,
        "checks": {"schema": "passed", "token_budget": "passed"},
        "token_contract": {
            "total_max": 4096,
            "truncation": False,
            "overflow_policy": "error",
        },
        "token_stats": {"total": {"max": 22}},
    }
    _write_json(root / "audits/data_quality.json", audit)
    (root / "chk3_minutes_sft.template.yaml").write_text(
        "max_length: 4096\n", encoding="utf-8"
    )

    file_names = {
        "audits/data_quality.json",
        "chk3_minutes_sft.template.yaml",
        *(f"minutes_alignment/{split}.jsonl" for split in split_counts),
        *(f"minutes_alignment/manifests/{split}.jsonl" for split in split_counts),
    }
    files = {}
    for name in sorted(file_names):
        path = root / name
        descriptor = {
            "path": name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if name.endswith(".jsonl"):
            descriptor["rows"] = 1
        files[name] = descriptor

    manifest = root / "release_manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": STANDALONE_CHK3_RELEASE_SCHEMA,
            "release_id": root.name,
            "quality_status": "passed",
            "immutable": True,
            "dag_bindable": False,
            "dag_binding_blocker": "canonical chk2 parent is unavailable",
            "dataset_role": "standalone_chk3_minutes_alignment",
            "training_mapping": (
                "chk1 final analysis -> reasoning -> formal Minutes paragraph"
            ),
            "split_counts": split_counts,
            "total_rows": 3,
            "files": files,
            "source": {
                "chk1_handoff_sha256": _sha("chk1-handoff"),
                "chk3_prompt_contract_sha256": _sha("prompt-contract"),
                "chk3_summary_sha256": _sha("summary"),
                "prior_chk3_release": {
                    "handoff_sha256": _sha("prior-handoff"),
                    "release_manifest_sha256": _sha("prior-manifest"),
                },
            },
        },
    )
    return dataset, manifest


def _verify(dataset: Path, manifest: Path) -> dict[str, object]:
    return verify_standalone_chk3_direct_sft_release(
        dataset_dir=dataset.resolve(),
        manifest_path=manifest.resolve(),
        expected_manifest_sha256=sha256_file(manifest),
        training_scope=STANDALONE_CHK3_DIRECT_SCOPE,
    )


def test_standalone_chk3_release_binds_bytes_and_nonpromotable_scope(
    tmp_path: Path,
) -> None:
    dataset, manifest = _write_release(tmp_path / "standalone_chk3")

    result = _verify(dataset, manifest)

    assert result["split_files"]["validation"]["path"] == (
        "minutes_alignment/validation.jsonl"
    )
    assert result["verified_scope"] == {
        "schema_version": STANDALONE_CHK3_BINDING_SCHEMA,
        "scope_id": STANDALONE_CHK3_DIRECT_SCOPE,
        "training_stage": "chk3",
        "operation": "direct_sft_training",
        "parent_stage": "chk1",
        "canonical_dag_bindable": False,
        "promotable_as_canonical_chk3": False,
        "downstream_stages_allowed": [],
    }


def test_standalone_chk3_release_rejects_canonical_or_wrong_scope(
    tmp_path: Path,
) -> None:
    dataset, manifest = _write_release(tmp_path / "standalone_chk3")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["dag_bindable"] = True
    _write_json(manifest, payload)
    with pytest.raises(DatasetReleaseValidationError, match="non-DAG-bindable"):
        _verify(dataset, manifest)

    payload["dag_bindable"] = False
    _write_json(manifest, payload)
    with pytest.raises(DatasetReleaseValidationError, match="non-promotable"):
        verify_standalone_chk3_direct_sft_release(
            dataset_dir=dataset.resolve(),
            manifest_path=manifest.resolve(),
            expected_manifest_sha256=sha256_file(manifest),
            training_scope="canonical_chk3",
        )


def test_standalone_chk3_release_rejects_split_or_row_manifest_drift(
    tmp_path: Path,
) -> None:
    dataset, manifest = _write_release(tmp_path / "standalone_chk3")
    (dataset / "train.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(DatasetReleaseValidationError, match="byte-count mismatch"):
        _verify(dataset, manifest)

    dataset, manifest = _write_release(tmp_path / "other_standalone_chk3")
    row_path = dataset / "manifests/train.jsonl"
    row = json.loads(row_path.read_text(encoding="utf-8"))
    row["prompt_sha256"] = _sha("wrong")
    row_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    release = json.loads(manifest.read_text(encoding="utf-8"))
    descriptor = release["files"]["minutes_alignment/manifests/train.jsonl"]
    descriptor["bytes"] = row_path.stat().st_size
    descriptor["sha256"] = sha256_file(row_path)
    _write_json(manifest, release)
    with pytest.raises(DatasetReleaseValidationError, match="prompt_sha256 mismatch"):
        _verify(dataset, manifest)


class _BindingTrainer(Trainer):
    def load_trainer(self):
        raise NotImplementedError

    def plot_customized_curve(self):
        raise NotImplementedError


def test_runtime_receipt_names_direct_standalone_scope() -> None:
    script = SimpleNamespace(
        dataset_name="/release/minutes_alignment",
        dataset_prompt_column="prompt",
        dataset_train_split="train",
        dataset_test_split="validation",
        dataset_release_manifest=None,
        dataset_release_manifest_sha256=None,
        dataset_standalone_chk3_scope=STANDALONE_CHK3_DIRECT_SCOPE,
        dataset_standalone_chk3_release_manifest="/release/release_manifest.json",
        dataset_standalone_chk3_release_manifest_sha256="a" * 64,
        user_prompt_suffix=None,
        reward_funcs=[],
    )
    training = SimpleNamespace(
        output_dir="/output",
        report_to=[],
    )
    model = SimpleNamespace(
        model_name_or_path="/models/chk1",
        load_in_4bit=False,
        load_in_8bit=False,
    )
    trainer = _BindingTrainer(script, training, model, SimpleNamespace())

    runtime = trainer._build_runtime_config()

    assert runtime["dataset"]["binding_mode"] == (
        "standalone_chk3_direct_non_promotable"
    )
    receipt = runtime["dataset"]["standalone_chk3_direct"]
    assert receipt["scope"]["canonical_dag_bindable"] is False
    assert receipt["scope"]["promotable_as_canonical_chk3"] is False
    assert receipt["scope"]["downstream_stages_allowed"] == []


def test_standalone_chk3_fields_are_all_or_none_and_exclusive() -> None:
    partial = SimpleNamespace(
        dataset_release_manifest=None,
        dataset_release_manifest_sha256=None,
        dataset_standalone_chk3_scope=STANDALONE_CHK3_DIRECT_SCOPE,
    )
    trainer = _BindingTrainer(
        partial, SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
    )
    with pytest.raises(ValueError, match="fields must all be configured"):
        trainer._dataset_binding()

    conflicting = SimpleNamespace(
        dataset_release_manifest="/clean/release_manifest.json",
        dataset_release_manifest_sha256="b" * 64,
        dataset_standalone_chk3_scope=STANDALONE_CHK3_DIRECT_SCOPE,
        dataset_standalone_chk3_release_manifest="/standalone/release_manifest.json",
        dataset_standalone_chk3_release_manifest_sha256="a" * 64,
    )
    trainer = _BindingTrainer(
        conflicting, SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        trainer._dataset_binding()
