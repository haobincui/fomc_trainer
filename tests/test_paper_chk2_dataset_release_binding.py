from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_r1.configs import SFTScriptArguments
from open_r1.trainer import dataset_release
from open_r1.trainer.dataset_release import (
    PAPER_CHK2_BINDING_SCHEMA,
    PAPER_CHK2_DATASET_ROLE,
    PAPER_CHK2_PARENT_AUTHORIZATION_FILE_SHA256,
    PAPER_CHK2_PARENT_AUTHORIZATION_RELATIVE,
    PAPER_CHK2_PARENT_AUTHORIZATION_SCHEMA,
    PAPER_CHK2_PARENT_AUTHORIZATION_SHA256,
    PAPER_CHK2_PARENT_MANIFEST_RELATIVE,
    PAPER_CHK2_PARENT_MODEL_RELATIVE,
    PAPER_CHK2_PARENT_MODEL_SHA256,
    PAPER_CHK2_PROMPT_CONTRACT_SCHEMA,
    PAPER_CHK2_RELEASE_SCHEMA,
    PAPER_CHK2_STUDENT_SYSTEM_PROMPT,
    PAPER_CHK2_STUDENT_SYSTEM_PROMPT_SHA256,
    PAPER_CHK2_TRAINING_SCOPE,
    PAPER_CHK2_USER_PROMPT_TEMPLATE,
    PAPER_CHK2_USER_PROMPT_TEMPLATE_SHA256,
    DatasetReleaseValidationError,
    sha256_file,
    verify_paper_chk2_sft_release,
)
from open_r1.trainer.trainer import Trainer


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / PAPER_CHK2_PARENT_MODEL_RELATIVE
AUTHORIZATION = REPO_ROOT / PAPER_CHK2_PARENT_AUTHORIZATION_RELATIVE
CHECKPOINT_MANIFEST = REPO_ROOT / PAPER_CHK2_PARENT_MANIFEST_RELATIVE
SPLIT_COUNTS = {"train": 305, "validation": 42, "test": 44}


class _BindingTrainer(Trainer):
    def load_trainer(self):
        raise NotImplementedError

    def plot_customized_curve(self):
        raise NotImplementedError


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_canonical(row) + "\n" for row in rows), encoding="utf-8")


def _descriptor(path: Path, *, root: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".jsonl":
        result["rows"] = len(path.read_text(encoding="utf-8").splitlines())
    return result


def _lineage() -> dict[str, bool]:
    return {
        "source_analysis_is_exact_chk1_final_answer": True,
        "source_analysis_was_repaired": False,
        "c8_used_for_training": False,
        "rewrite_teacher_saw_official_minutes": False,
        "target_is_teacher_synthetic_rewrite": True,
        "validator_a_is_factual_gate": True,
        "validator_b_is_style_gate": True,
        "validator_b_used_for_training_selection": True,
        "official_minutes_used_as_student_target": False,
        "training_only": True,
        "evaluation_eligible": False,
        "suitable_for_leakage_safe_evaluation": False,
    }


def _build_release(root: Path) -> tuple[Path, Path]:
    pass_ids: list[tuple[str, str]] = []
    minutes_artifacts: dict[str, object] = {}
    token_rows: list[dict] = []
    for split_number, (split, count) in enumerate(SPLIT_COUNTS.items(), 1):
        data_rows: list[dict] = []
        sidecars: list[dict] = []
        for index in range(count):
            sample_id = f"pass-{split}-{index:04d}"
            analysis = f"The supplied {split} analysis number {index} remained stable."
            prompt = (
                "Rewrite the following analysis as formal FOMC Minutes prose:\n\n"
                + _canonical({"analysis": analysis})
            )
            minutes = (
                "Participants noted that the supplied economic analysis remained "
                "stable, while maintaining a neutral assessment of the reported "
                "conditions and the uncertainty surrounding those developments "
                "over the period under review."
            )
            response = "Preserve every supplied claim faithfully.\n</think>\n" + minutes
            data_rows.append({"prompt": prompt, "response": response})
            sidecars.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "source_index": split_number * 10_000 + index,
                    "meeting_date": f"20{split_number:02d}-01-01",
                    "atomic_topic": "economic_conditions",
                    "section_style_id": "staff_review",
                    "terminal_status": "PASS",
                    "source_analysis_sha256": _sha_text(analysis),
                    "prompt_sha256": _sha_text(prompt),
                    "response_sha256": _sha_text(response),
                    "lineage": _lineage(),
                    "release_index": index,
                }
            )
            prompt_tokens = 100
            completion_tokens = 100
            total_tokens = prompt_tokens + completion_tokens
            if not pass_ids:
                total_tokens = 3316
                completion_tokens = total_tokens - prompt_tokens
            token_rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "single_bos": True,
                    "single_eos": True,
                    "no_truncation": True,
                    "completion_only_prompt_masked": True,
                    "completion_mask_covers_reasoning_boundary_answer_eos": True,
                }
            )
            pass_ids.append((sample_id, split))
        data_path = root / "minutes_alignment" / f"{split}.jsonl"
        sidecar_path = root / "minutes_alignment/manifests" / f"{split}.jsonl"
        _write_jsonl(data_path, data_rows)
        _write_jsonl(sidecar_path, sidecars)
        minutes_artifacts[split] = {
            "data": _descriptor(data_path, root=root),
            "manifest": _descriptor(sidecar_path, root=root),
        }

    reject_ids = [f"reject-{index:04d}" for index in range(1352)]
    all_ids = [sample_id for sample_id, _ in pass_ids] + reject_ids
    rejections = [
        {
            "sample_id": sample_id,
            "split": "train",
            "source_index": 50_000 + index,
            "terminal_status": "STYLE_QUALITY_REJECT",
            "rejection_stage": "validator_b",
            "rejection_reasons": ["style"],
        }
        for index, sample_id in enumerate(reject_ids)
    ]
    source_admission = [{"sample_id": sample_id} for sample_id in all_ids]
    evidence_ledger = [{"sample_id": sample_id} for sample_id in all_ids]
    validator_a = [{"sample_id": sample_id} for sample_id, _ in pass_ids]
    validator_b = [{"sample_id": sample_id} for sample_id, _ in pass_ids]
    repair_history: list[dict] = []
    compatibility_ids = reject_ids[:22]
    compatibility = [
        {
            "sample_id": sample_id,
            "split": "train",
            "terminal_status": "STYLE_QUALITY_REJECT",
            "rejection_stage": "validator_b",
            "terminal_record_sha256": _sha_text(sample_id),
        }
        for sample_id in compatibility_ids
    ]
    audit_rows = {
        "rejections": rejections,
        "source_admission": source_admission,
        "validator_a": validator_a,
        "validator_b": validator_b,
        "repair_history": repair_history,
        "compatibility_replay": compatibility,
        "evidence_ledger": evidence_ledger,
        "tokenizer_replay": token_rows,
    }
    audit_artifacts: dict[str, object] = {}
    for name, rows in audit_rows.items():
        path = root / "audits" / f"{name}.jsonl"
        _write_jsonl(path, rows)
        audit_artifacts[name] = _descriptor(path, root=root)
    quality_path = root / "audits/data_quality.json"
    _write_json(
        quality_path,
        {
            "schema_version": PAPER_CHK2_RELEASE_SCHEMA,
            "source_rows": 1743,
            "split_pass_counts": SPLIT_COUNTS,
            "terminal_status_counts": {},
            "rejected_rows": 1352,
            "pass_rows": 391,
            "three_pass_splits_nonempty": True,
            "validator_a_rows": 391,
            "validator_b_rows": 391,
            "repair_event_distribution": {},
            "evidence_ledger_rows": 1743,
            "compatibility_count": 22,
            "compatibility_all_rejected": True,
            "external_cache_counts": {},
        },
    )
    audit_artifacts["data_quality"] = _descriptor(quality_path, root=root)

    prompt_path = root / "prompt_contract.json"
    _write_json(
        prompt_path,
        {
            "schema_version": PAPER_CHK2_PROMPT_CONTRACT_SCHEMA,
            "system_prompt": PAPER_CHK2_STUDENT_SYSTEM_PROMPT,
            "system_prompt_sha256": PAPER_CHK2_STUDENT_SYSTEM_PROMPT_SHA256,
            "user_prompt_template": PAPER_CHK2_USER_PROMPT_TEMPLATE,
            "user_prompt_template_sha256": PAPER_CHK2_USER_PROMPT_TEMPLATE_SHA256,
            "response_boundary": "</think>",
            "opening_think_supplied_by_chat_template": True,
        },
    )
    checkpoint_manifest_path = root / "provenance/parent_checkpoint_manifest.json"
    checkpoint_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CHECKPOINT_MANIFEST, checkpoint_manifest_path)
    authorization_path = root / "provenance/parent_authorization.json"
    authorization_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(AUTHORIZATION, authorization_path)
    simple_json_paths = {
        "source_handoff_manifest": "provenance/source_handoff_manifest.json",
        "source_admission_receipt": "provenance/source_admission_receipt.json",
        "source_prompt_contract": "provenance/source_prompt_contract.json",
        "recovery_partial_handoff_manifest": (
            "provenance/recovery_partial_handoff_manifest.json"
        ),
        "recovery_receipt": "provenance/recovery_receipt.json",
        "recovery_prompt_contract": "provenance/recovery_prompt_contract.json",
    }
    provenance_artifacts: dict[str, object] = {
        "student_prompt_contract": _descriptor(prompt_path, root=root),
        "parent_checkpoint_manifest": _descriptor(checkpoint_manifest_path, root=root),
        "parent_authorization": _descriptor(authorization_path, root=root),
    }
    for key, relative in simple_json_paths.items():
        path = root / relative
        _write_json(path, {"fixture": key})
        provenance_artifacts[key] = _descriptor(path, root=root)
    reference_path = root / "provenance/official_pre_action_reference_bank.jsonl"
    _write_jsonl(reference_path, [{"fixture": True}])
    provenance_artifacts["official_pre_action_reference_bank"] = _descriptor(
        reference_path, root=root
    )
    attempt_path = root / "provenance/recovery_attempts/attempt-0001.json"
    _write_json(attempt_path, {"status": "complete"})
    provenance_artifacts["recovery_attempts"] = {
        "attempt_0001": _descriptor(attempt_path, root=root)
    }

    artifacts = {
        "minutes_alignment": minutes_artifacts,
        "audits": audit_artifacts,
        "provenance": provenance_artifacts,
    }
    manifest = {
        "schema_version": PAPER_CHK2_RELEASE_SCHEMA,
        "status": "complete",
        "quality_status": "passed",
        "dataset_role": PAPER_CHK2_DATASET_ROLE,
        "training_scope": PAPER_CHK2_TRAINING_SCOPE,
        "immutable": True,
        "training_ready": True,
        "training_only": True,
        "evaluation_eligible": False,
        "dag_bindable": False,
        "promotable_as_canonical_chk2": False,
        "source_rows": 1743,
        "split_pass_counts": SPLIT_COUNTS,
        "terminal_status_counts": {},
        "student_prompt_contract": {
            "system_prompt_sha256": PAPER_CHK2_STUDENT_SYSTEM_PROMPT_SHA256,
            "user_prompt_template_sha256": PAPER_CHK2_USER_PROMPT_TEMPLATE_SHA256,
            "response_boundary": "</think>",
        },
        "parent_checkpoint": {
            "schema_version": "chk1-cp200-merged-checkpoint-manifest-v1",
            "model_path": PAPER_CHK2_PARENT_MODEL_RELATIVE,
            "model_sha256": PAPER_CHK2_PARENT_MODEL_SHA256,
            "checkpoint_manifest_file_sha256": sha256_file(checkpoint_manifest_path),
            "authorization_schema_version": PAPER_CHK2_PARENT_AUTHORIZATION_SCHEMA,
            "authorization_sha256": PAPER_CHK2_PARENT_AUTHORIZATION_SHA256,
            "authorization_file_sha256": PAPER_CHK2_PARENT_AUTHORIZATION_FILE_SHA256,
            "allowed_stage": "chk2",
            "further_downstream_stages_allowed": [],
        },
        "source_handoff": {},
        "recovery": {
            "compatibility_ids": compatibility_ids,
            "compatibility_id_digest": (
                "e62415eecfdb37df240ab7259aba9e244d8fcdfbc8b3fecf907aaf70c5fd5ca7"
            ),
        },
        "provider_identities": {},
        "publisher_implementation": {},
        "tokenizer_runtime_contract": {},
        "artifacts": artifacts,
    }
    manifest["manifest_sha256"] = _sha_text(_canonical(manifest))
    manifest_path = root / "release_manifest.json"
    _write_json(manifest_path, manifest)
    handoff = {
        "schema_version": "paper-chk2-downstream-recovery-release-handoff-v1",
        "status": "complete",
        "created_at": "2026-09-01T00:00:00Z",
        "release_manifest_sha256": manifest["manifest_sha256"],
        "release_manifest_file_sha256": sha256_file(manifest_path),
        "source_handoff_manifest_sha256": _sha_text("source"),
        "source_handoff_manifest_file_sha256": _sha_text("source-file"),
        "recovery_receipt_sha256": _sha_text("recovery"),
        "recovery_receipt_file_sha256": _sha_text("recovery-file"),
        "parent_authorization_sha256": PAPER_CHK2_PARENT_AUTHORIZATION_SHA256,
        "parent_authorization_file_sha256": (
            PAPER_CHK2_PARENT_AUTHORIZATION_FILE_SHA256
        ),
        "parent_checkpoint_model_sha256": PAPER_CHK2_PARENT_MODEL_SHA256,
        "publisher_implementation": {},
        "split_data_sha256": {
            split: minutes_artifacts[split]["data"]["sha256"] for split in SPLIT_COUNTS
        },
    }
    handoff["handoff_sha256"] = _sha_text(_canonical(handoff))
    _write_json(root / "handoff.json", handoff)
    return root / "minutes_alignment", manifest_path


@pytest.fixture
def release(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    checkpoint = json.loads(CHECKPOINT_MANIFEST.read_text(encoding="utf-8"))
    fingerprint = checkpoint["model_fingerprint"]
    monkeypatch.setattr(
        dataset_release,
        "fingerprint_artifact_path",
        lambda path: {
            "path": str(Path(path).resolve()),
            "kind": "directory",
            "sha256": PAPER_CHK2_PARENT_MODEL_SHA256,
            "file_count": fingerprint["file_count"],
            "total_bytes": fingerprint["total_bytes"],
            "algorithm": "fixture",
        },
    )
    return _build_release(tmp_path / "paper_chk2_release")


def _verify(dataset: Path, manifest: Path):
    return verify_paper_chk2_sft_release(
        dataset_dir=dataset,
        manifest_path=manifest,
        expected_manifest_sha256=sha256_file(manifest),
        training_scope=PAPER_CHK2_TRAINING_SCOPE,
        system_prompt=PAPER_CHK2_STUDENT_SYSTEM_PROMPT,
        model_path=MODEL_ROOT,
    )


def test_release_verifier_loads_only_train_and_validation(release) -> None:
    dataset, manifest = release
    result = _verify(dataset, manifest)

    assert result["schema_version"] == PAPER_CHK2_BINDING_SCHEMA
    assert result["split_counts"] == SPLIT_COUNTS
    assert set(result["split_files"]) == {"train", "validation"}
    assert result["test_verified_but_not_loaded"] is True
    assert result["sealed_test"]["rows"] == 44
    assert result["token_audit"]["max_total_tokens"] == 3316
    assert result["quality_audit"]["compatibility_count"] == 22


def test_release_verifier_rejects_member_and_prompt_drift(release) -> None:
    dataset, manifest = release
    train = dataset / "train.jsonl"
    train.write_bytes(train.read_bytes() + b"{}\n")
    with pytest.raises(DatasetReleaseValidationError, match="byte-size drift"):
        _verify(dataset, manifest)

    dataset, manifest = _build_release(manifest.parent.parent / "other_release")
    with pytest.raises(DatasetReleaseValidationError, match="system_prompt"):
        verify_paper_chk2_sft_release(
            dataset_dir=dataset,
            manifest_path=manifest,
            expected_manifest_sha256=sha256_file(manifest),
            training_scope=PAPER_CHK2_TRAINING_SCOPE,
            system_prompt=PAPER_CHK2_STUDENT_SYSTEM_PROMPT + " drift",
            model_path=MODEL_ROOT,
        )


def test_sft_arguments_expose_three_paper_chk2_fields() -> None:
    assert {
        "dataset_paper_chk2_scope",
        "dataset_paper_chk2_release_manifest",
        "dataset_paper_chk2_release_manifest_sha256",
    } <= set(SFTScriptArguments.__dataclass_fields__)


def test_binding_is_all_or_none_exclusive_and_requires_sft_contract() -> None:
    trainer = object.__new__(_BindingTrainer)
    trainer.training_args = SimpleNamespace(
        completion_only_loss=True, packing=False, max_length=4096
    )
    trainer.script_args = SimpleNamespace(
        dataset_paper_chk2_scope=PAPER_CHK2_TRAINING_SCOPE
    )
    with pytest.raises(ValueError, match="fields must all be configured"):
        trainer._dataset_binding()

    trainer.script_args = SimpleNamespace(
        dataset_release_manifest="/clean/release_manifest.json",
        dataset_release_manifest_sha256="b" * 64,
        dataset_paper_chk2_scope=PAPER_CHK2_TRAINING_SCOPE,
        dataset_paper_chk2_release_manifest="/paper/release_manifest.json",
        dataset_paper_chk2_release_manifest_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        trainer._dataset_binding()

    trainer.script_args = SimpleNamespace(
        dataset_paper_chk2_scope=PAPER_CHK2_TRAINING_SCOPE,
        dataset_paper_chk2_release_manifest="/paper/release_manifest.json",
        dataset_paper_chk2_release_manifest_sha256="a" * 64,
        dataset_prompt_column="prompt",
        dataset_train_split="train",
        dataset_test_split="validation",
        user_prompt_suffix=None,
    )
    mode, _ = trainer._dataset_binding()
    assert mode == "paper_chk2_minutes_sft"
    trainer.training_args.completion_only_loss = False
    with pytest.raises(ValueError, match="completion_only_loss=true"):
        trainer._dataset_binding()


def test_runtime_receipt_binds_release_parent_prompt_and_config(
    monkeypatch, tmp_path: Path
) -> None:
    config = tmp_path / "paper.yaml"
    config.write_text("max_length: 4096\n", encoding="utf-8")
    monkeypatch.setenv("FOMC_PAPER_CHK2_CONFIG_PATH", str(config.resolve()))
    monkeypatch.setenv("FOMC_PAPER_CHK2_CONFIG_SHA256", sha256_file(config))
    monkeypatch.setenv(
        "FOMC_PAPER_CHK2_BRANCH_ID",
        "paper_chk2_chk1_cp200_minutes_v6_recovery_full3ep_lr1e6_v1_20260901",
    )
    trainer = object.__new__(_BindingTrainer)
    trainer.script_args = SimpleNamespace(
        dataset_name="/release/minutes_alignment",
        dataset_prompt_column="prompt",
        dataset_train_split="train",
        dataset_test_split="validation",
        dataset_paper_chk2_scope=PAPER_CHK2_TRAINING_SCOPE,
        dataset_paper_chk2_release_manifest="/release/release_manifest.json",
        dataset_paper_chk2_release_manifest_sha256="a" * 64,
        user_prompt_suffix=None,
        reward_funcs=[],
    )
    trainer.training_args = SimpleNamespace(
        output_dir=str(tmp_path),
        system_prompt=PAPER_CHK2_STUDENT_SYSTEM_PROMPT,
        completion_only_loss=True,
        packing=False,
        max_length=4096,
        report_to=[],
        device="cpu",
        n_gpu=0,
        world_size=1,
        process_index=0,
        local_process_index=0,
    )
    trainer.model_args = SimpleNamespace(
        model_name_or_path=str(MODEL_ROOT),
        load_in_4bit=False,
        load_in_8bit=False,
    )
    trainer.peft_args = SimpleNamespace()
    trainer._paper_chk2_release_binding = {
        "schema_version": PAPER_CHK2_BINDING_SCHEMA,
        "dataset_role": PAPER_CHK2_DATASET_ROLE,
        "training_scope": PAPER_CHK2_TRAINING_SCOPE,
        "release_manifest": {"manifest_sha256": "b" * 64},
        "split_counts": SPLIT_COUNTS,
        "test_verified_but_not_loaded": True,
        "student_prompt_contract": {
            "system_prompt_sha256": PAPER_CHK2_STUDENT_SYSTEM_PROMPT_SHA256
        },
        "parent_binding": {"model": {"sha256": PAPER_CHK2_PARENT_MODEL_SHA256}},
        "token_audit": {"rows": 391},
        "quality_audit": {"pass_rows": 391},
        "scope": {"canonical_dag_bindable": False},
    }
    monkeypatch.setattr(
        "open_r1.trainer.trainer.get_quantization_config", lambda _: None
    )

    runtime = trainer._build_runtime_config()
    receipt = runtime["dataset"]["paper_chk2_minutes_sft"]
    assert receipt["schema_version"] == PAPER_CHK2_BINDING_SCHEMA
    assert receipt["split_counts"] == SPLIT_COUNTS
    assert receipt["test_verified_but_not_loaded"] is True
    assert receipt["training_config"] == {
        "path": str(config.resolve()),
        "sha256": sha256_file(config),
    }
    inventory = receipt["environment_inventory"]
    assert inventory["schema_version"] == "paper-chk2-runtime-environment-v1"
    assert inventory["core_packages"]["torch"]
    assert len(inventory["installed_distributions_sha256"]) == 64
    serialized_inventory = json.dumps(inventory, sort_keys=True).lower()
    assert "api_key" not in serialized_inventory
    assert "secret" not in serialized_inventory
