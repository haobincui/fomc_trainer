from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import open_r1.trainer.dataset_release as release_module
import open_r1.trainer.trainer as trainer_module
from open_r1.trainer.dataset_release import (
    CHK1_OVERRIDE_AUTHORIZATION_SCHEMA,
    CHK1_OVERRIDE_CANDIDATE_SCHEMA,
    CLEAN_SFT_AUDIT_SCHEMA,
    CLEAN_SFT_VALIDATION_SCHEMA,
    DatasetReleaseValidationError,
    sha256_file,
    verify_chk1_semantic_override,
)
from open_r1.trainer.trainer import Trainer


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _write_authorization(
    path: Path,
    *,
    dataset: Path,
    candidate_id: str,
    candidate_sha: str,
    validation_sha: str,
    audit_sha: str,
    split_counts: dict[str, int],
    stage: str = "chk1",
) -> None:
    payload = {
        "schema_version": CHK1_OVERRIDE_AUTHORIZATION_SCHEMA,
        "status": "authorized",
        "authorization_basis": "explicit_user_instruction",
        "scope": {
            "stage": stage,
            "operation": "sft_training",
            "downstream_stages_allowed": [],
        },
        "bindings": {
            "candidate_id": candidate_id,
            "dataset_dir": str(dataset.resolve()),
            "candidate_manifest_sha256": candidate_sha,
            "deterministic_validation_receipt_sha256": validation_sha,
            "semantic_audit_summary_sha256": audit_sha,
            "split_counts": split_counts,
        },
        "risk_acknowledgements": [
            "semantic_audit_failed",
            "semantic_audit_contains_blocking_violations",
            "semantic_audit_contains_judge_errors",
            "not_authorized_for_downstream_training",
        ],
    }
    payload["authorization_sha256"] = _sha(_canonical(payload))
    _write_json(path, payload)


def _write_override_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, str]:
    split_counts = {"train": 2, "eval": 1, "test": 1}
    monkeypatch.setattr(release_module, "_CHK1_OVERRIDE_SPLIT_COUNTS", split_counts)
    monkeypatch.setattr(release_module, "_CHK1_OVERRIDE_CHANGED_ROWS", 2)

    root = (tmp_path / "candidate").resolve()
    dataset = root / "analysis_sft"
    dataset.mkdir(parents=True)
    split_files: dict[str, object] = {}
    for split, count in split_counts.items():
        split_path = dataset / f"{split}.jsonl"
        _write_jsonl(
            split_path,
            [{"prompt": f"{split}-{index}", "response": "r", "provided_data": "d"}
             for index in range(count)],
        )
        split_files[split] = {
            "path": f"analysis_sft/{split}.jsonl",
            "rows": count,
            "sha256": sha256_file(split_path),
        }

    coordinates = [("train", 1), ("train", 2), ("eval", 1), ("test", 1)]
    repair_rows = []
    for index, (split, line_number) in enumerate(coordinates):
        new_response_sha = _sha(f"new-response-{index}")
        repair_rows.append(
            {
                "sample_id": f"sample-{index}",
                "split": split,
                "source_line_number": line_number,
                "prompt_sha256": _sha(f"prompt-{index}"),
                "provided_data_sha256": _sha(f"data-{index}"),
                "old_response_sha256": (
                    _sha(f"old-response-{index}") if index < 2 else new_response_sha
                ),
                "new_response_sha256": new_response_sha,
            }
        )
    repair_path = root / "audits" / "repair_manifest.jsonl"
    semantic_input_path = root / "audits" / "semantic_audit_input.jsonl"
    _write_jsonl(repair_path, repair_rows)
    _write_jsonl(
        semantic_input_path,
        [{"sample_id": "sample-0"}, {"sample_id": "sample-1"}],
    )
    manifest_payload = {
        "schema_version": CHK1_OVERRIDE_CANDIDATE_SCHEMA,
        "candidate_id": root.name,
        "quality_status": "pending_semantic_audit",
        "immutable_candidate": True,
        "split_counts": split_counts,
        "split_files": split_files,
        "changed_rows": 2,
        "changed_sample_ids_sha256": _sha(_canonical(["sample-0", "sample-1"])),
        "repair_manifest": {
            "path": "audits/repair_manifest.jsonl",
            "rows": 4,
            "sha256": sha256_file(repair_path),
        },
        "semantic_audit_input": {
            "path": "audits/semantic_audit_input.jsonl",
            "rows": 2,
            "sha256": sha256_file(semantic_input_path),
        },
    }
    manifest_path = root / "candidate_manifest.json"
    _write_json(manifest_path, manifest_payload)
    candidate_sha = sha256_file(manifest_path)

    content_binding = [
        {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "line_number": row["source_line_number"],
            "prompt_sha256": row["prompt_sha256"],
            "provided_data_sha256": row["provided_data_sha256"],
            "response_sha256": row["new_response_sha256"],
        }
        for row in repair_rows
    ]
    tokenizer_files = [
        {"name": "tokenizer.json", "bytes": 3, "sha256": _sha("{}\n")}
    ]
    validation = {
        "schema_version": CLEAN_SFT_VALIDATION_SCHEMA,
        "status": "passed",
        "contracts": {
            "data": "chk1-clean-sft-data-contract-v2",
            "tokenization": "chk1-sft-single-bos-completion-mask-v1",
        },
        "configuration": {
            "expected_split_counts": split_counts,
            "max_length": 4608,
            "reasoning_token_range": [512, 2400],
            "answer_token_range": [16, 512],
        },
        "inputs": {
            "clean_release": {
                "release_id": root.name,
                "manifests": [
                    {"name": "candidate_manifest.json", "sha256": candidate_sha}
                ],
                "splits": {
                    split: {"sha256": split_files[split]["sha256"]}
                    for split in split_counts
                },
            },
            "repair_manifest": {"rows": 4, "sha256": sha256_file(repair_path)},
            "tokenizer": {
                "files": tokenizer_files,
                "tokenizer_bundle_sha256": _sha(_canonical(tokenizer_files)),
            },
        },
        "counts": {
            "expected_rows": 4,
            "observed_rows": 4,
            "valid_rows": 4,
            "prompt_hashes_unchanged": 4,
            "provided_data_hashes_unchanged": 4,
            "order_unchanged": 4,
            "truncated_rows": 0,
            "issue_rows_or_groups": 0,
            "split_counts": split_counts,
        },
        "content_binding_sha256": _sha(_canonical(content_binding)),
        "quality_gates": {"data": True, "tokenization": True},
        "issues": [],
        "token_statistics": {"max_total_tokens": 4019},
    }
    validation["validation_sha256"] = _sha(_canonical(validation))
    validation_path = tmp_path / "validation.json"
    _write_json(validation_path, validation)
    validation_sha = sha256_file(validation_path)

    audit_rows_path = tmp_path / "audit" / "row_audits.jsonl"
    _write_jsonl(
        audit_rows_path,
        [
            {
                "schema_version": CLEAN_SFT_AUDIT_SCHEMA,
                "sample_id": "sample-0",
                "split": "train",
                "candidate_sha256": repair_rows[0]["new_response_sha256"],
                "evidence_sha256": repair_rows[0]["provided_data_sha256"],
                "status": "failed",
                "validated_violations": [{"kind": "factual"}],
                "blocking_violations": [{"kind": "factual"}],
            }
        ],
    )
    audit = {
        "schema_version": CLEAN_SFT_AUDIT_SCHEMA,
        "status": "failed",
        "release_dir": str(root),
        "repair_manifest_sha256": sha256_file(repair_path),
        "counts": {
            "expected": 2,
            "completed": 1,
            "passed": 0,
            "failed": 1,
            "judge_errors": 1,
            "blocking_violations": 1,
        },
        "errors": [
            {
                "sample_id": "sample-1",
                "split": "train",
                "error_type": "JudgeError",
            }
        ],
        "row_audit": {
            "path": "row_audits.jsonl",
            "rows": 1,
            "sha256": sha256_file(audit_rows_path),
        },
        "judge": {
            "model": "Qwen3.5-9B",
            "health": {
                "model": "Qwen3.5-9B",
                "loaded_model_root": "/models/Qwen3.5-9B",
                "status": "ready",
                "tokenizer_parity": True,
                "weight_attested": True,
            },
        },
    }
    audit_path = tmp_path / "audit" / "summary.json"
    _write_json(audit_path, audit)
    audit_sha = sha256_file(audit_path)

    authorization_path = tmp_path / "authorization.json"
    _write_authorization(
        authorization_path,
        dataset=dataset,
        candidate_id=root.name,
        candidate_sha=candidate_sha,
        validation_sha=validation_sha,
        audit_sha=audit_sha,
        split_counts=split_counts,
    )
    return {
        "dataset_dir": str(dataset),
        "candidate_manifest_path": str(manifest_path),
        "expected_candidate_manifest_sha256": candidate_sha,
        "deterministic_validation_path": str(validation_path.resolve()),
        "expected_deterministic_validation_sha256": validation_sha,
        "semantic_audit_summary_path": str(audit_path.resolve()),
        "expected_semantic_audit_summary_sha256": audit_sha,
        "authorization_receipt_path": str(authorization_path.resolve()),
        "expected_authorization_receipt_sha256": sha256_file(authorization_path),
        "training_stage": "chk1",
    }


def test_chk1_semantic_override_verifies_all_hash_bound_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _write_override_bundle(tmp_path, monkeypatch)

    result = verify_chk1_semantic_override(**kwargs)

    assert result["mode"] == "chk1_semantic_override"
    assert result["quality_status"] == "pending_semantic_audit"
    assert set(result["split_files"]) == {"train", "eval", "test"}
    assert result["scope"] == {
        "stage": "chk1",
        "operation": "sft_training",
        "downstream_stages_allowed": [],
    }


def test_chk1_semantic_override_rejects_passed_or_drifted_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _write_override_bundle(tmp_path, monkeypatch)
    audit_path = Path(kwargs["semantic_audit_summary_path"])
    audit = json.loads(audit_path.read_text())
    audit["status"] = "passed"
    _write_json(audit_path, audit)
    audit_sha = sha256_file(audit_path)
    kwargs["expected_semantic_audit_summary_sha256"] = audit_sha
    authorization_path = Path(kwargs["authorization_receipt_path"])
    _write_authorization(
        authorization_path,
        dataset=Path(kwargs["dataset_dir"]),
        candidate_id="candidate",
        candidate_sha=kwargs["expected_candidate_manifest_sha256"],
        validation_sha=kwargs["expected_deterministic_validation_sha256"],
        audit_sha=audit_sha,
        split_counts={"train": 2, "eval": 1, "test": 1},
    )
    kwargs["expected_authorization_receipt_sha256"] = sha256_file(authorization_path)

    with pytest.raises(DatasetReleaseValidationError, match="requires a failed"):
        verify_chk1_semantic_override(**kwargs)


def test_chk1_semantic_override_rejects_downstream_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _write_override_bundle(tmp_path, monkeypatch)
    kwargs["training_stage"] = "chk3"

    with pytest.raises(DatasetReleaseValidationError, match="prohibits chk2/chk3/chk4"):
        verify_chk1_semantic_override(**kwargs)


class _DatasetTrainer(Trainer):
    def load_trainer(self):
        raise NotImplementedError

    def plot_customized_curve(self):
        raise NotImplementedError


class _Dataset(dict):
    def __init__(self) -> None:
        split = SimpleNamespace(column_names=[])
        super().__init__({"train": split, "validation": split})

    def map(self, _function):
        return self


class _Logger:
    def info(self, *_args, **_kwargs) -> None:
        pass

    def warning(self, *_args, **_kwargs) -> None:
        pass


def _trainer_args_from_bundle(bundle: dict[str, str]) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_name=bundle["dataset_dir"],
        dataset_prompt_column="prompt",
        user_prompt_suffix=None,
        dataset_release_manifest=None,
        dataset_release_manifest_sha256=None,
        dataset_semantic_override_stage="chk1",
        dataset_semantic_override_candidate_manifest=bundle[
            "candidate_manifest_path"
        ],
        dataset_semantic_override_candidate_manifest_sha256=bundle[
            "expected_candidate_manifest_sha256"
        ],
        dataset_semantic_override_validation_receipt=bundle[
            "deterministic_validation_path"
        ],
        dataset_semantic_override_validation_receipt_sha256=bundle[
            "expected_deterministic_validation_sha256"
        ],
        dataset_semantic_override_audit_summary=bundle[
            "semantic_audit_summary_path"
        ],
        dataset_semantic_override_audit_summary_sha256=bundle[
            "expected_semantic_audit_summary_sha256"
        ],
        dataset_semantic_override_authorization_receipt=bundle[
            "authorization_receipt_path"
        ],
        dataset_semantic_override_authorization_receipt_sha256=bundle[
            "expected_authorization_receipt_sha256"
        ],
    )


def test_trainer_loads_only_exact_override_train_and_eval_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _write_override_bundle(tmp_path, monkeypatch)
    captured: dict[str, object] = {}

    def fake_verify(**kwargs):
        captured["verify"] = kwargs
        return {
            "release_id": "candidate",
            "split_files": {
                "train": {"path": "analysis_sft/train.jsonl"},
                "eval": {"path": "analysis_sft/eval.jsonl"},
                "test": {"path": "analysis_sft/test.jsonl"},
            },
        }

    def fake_load(data_path, *, split_files):
        captured["load"] = {"data_path": data_path, "split_files": split_files}
        return _Dataset()

    monkeypatch.setattr(trainer_module, "verify_chk1_semantic_override", fake_verify)
    monkeypatch.setattr(trainer_module, "load_train_eval_datasets", fake_load)
    trainer = _DatasetTrainer(
        _trainer_args_from_bundle(bundle),
        SimpleNamespace(system_prompt=None),
        SimpleNamespace(),
        SimpleNamespace(),
    )
    trainer._logger = _Logger()

    trainer.load_dataset()

    assert captured["verify"]["training_stage"] == "chk1"
    candidate_root = Path(bundle["candidate_manifest_path"]).parent
    assert captured["load"] == {
        "data_path": bundle["dataset_dir"],
        "split_files": {
            "train": candidate_root / "analysis_sft/train.jsonl",
            "validation": candidate_root / "analysis_sft/eval.jsonl",
        },
    }


def test_trainer_rejects_partial_or_mixed_override_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _write_override_bundle(tmp_path, monkeypatch)
    args = _trainer_args_from_bundle(bundle)
    args.dataset_semantic_override_audit_summary_sha256 = None
    trainer = _DatasetTrainer(
        args, SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
    )
    with pytest.raises(ValueError, match="must all be configured"):
        trainer._dataset_binding()

    args = _trainer_args_from_bundle(bundle)
    args.dataset_release_manifest = "/tmp/release.json"
    args.dataset_release_manifest_sha256 = "a" * 64
    trainer = _DatasetTrainer(
        args, SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        trainer._dataset_binding()


def test_runtime_receipt_records_override_scope_and_four_hash_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _write_override_bundle(tmp_path, monkeypatch)
    trainer = _DatasetTrainer(
        _trainer_args_from_bundle(bundle),
        SimpleNamespace(
            output_dir=str(tmp_path / "out"),
            system_prompt=None,
            report_to=[],
        ),
        SimpleNamespace(
            model_name_or_path="base",
            model_revision=None,
            torch_dtype="bfloat16",
            dtype="bfloat16",
            attn_implementation=None,
            load_in_4bit=False,
            load_in_8bit=False,
        ),
        SimpleNamespace(
            peft_merged_model_path=None,
            peft_r=8,
            peft_lora_alpha=16,
            peft_lora_dropout=0.0,
            peft_target_modules=[],
        ),
    )
    monkeypatch.setattr(trainer_module, "get_quantization_config", lambda _args: None)

    receipt = trainer._build_runtime_config()["dataset"]

    assert receipt["binding_mode"] == "chk1_semantic_override"
    assert receipt["semantic_override"]["scope"]["downstream_stages_allowed"] == []
    for key in (
        "candidate_manifest",
        "deterministic_validation_receipt",
        "failed_semantic_audit_summary",
        "authorization_receipt",
    ):
        assert set(receipt["semantic_override"][key]) == {"path", "sha256"}
