from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_r1.configs import GRPOScriptArguments, SFTScriptArguments
from open_r1.trainer.dataset_release import (
    CHK4_STUDENT_SYSTEM_PROMPT,
    CHK4_STUDENT_SYSTEM_PROMPT_SHA256,
    DatasetReleaseValidationError,
    verify_chk4_decision_release,
)
from open_r1.trainer.trainer import Trainer


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = (
    REPO_ROOT / "dataset/processed/retrain_v2/"
    "chk4_decision_warmstart_grpo_core_v3_20260810"
)
MANIFEST_SHA256 = "8b05dce09bcb0a0b27ee240da1e5d730be93921ce19e650a3b3e8b2689adb893"
MODEL_ROOT = (
    REPO_ROOT / "output/training/retrain_v2/"
    "chk1_clean_v2_lr1e6_selected_cp200_for_chk2_20260810/merged/chk1"
)


class _BindingTrainer(Trainer):
    def load_trainer(self):
        raise NotImplementedError

    def plot_customized_curve(self):
        raise NotImplementedError


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_writable(root: Path) -> None:
    root.chmod(0o755)
    for path in root.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)


def _seal(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _copy_release(tmp_path: Path) -> Path:
    destination = tmp_path / RELEASE_ROOT.name
    shutil.copytree(RELEASE_ROOT, destination)
    return destination


def _rewrite_manifest_binding(root: Path) -> str:
    manifest_path = root / "release_manifest.json"
    manifest_sha = _sha256_file(manifest_path)
    handoff_path = root / "handoff.json"
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    handoff["release_manifest_sha256"] = manifest_sha
    handoff_path.write_text(
        json.dumps(handoff, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_sha


def _verify(root: Path, role: str, manifest_sha: str = MANIFEST_SHA256):
    return verify_chk4_decision_release(
        dataset_dir=root / role,
        manifest_path=root / "release_manifest.json",
        expected_manifest_sha256=manifest_sha,
        dataset_role=role,
        system_prompt=CHK4_STUDENT_SYSTEM_PROMPT,
        model_path=MODEL_ROOT,
    )


@pytest.mark.parametrize("role", ["decision_sft", "decision_grpo"])
def test_real_v3_release_binds_only_train_and_validation(role: str) -> None:
    result = _verify(RELEASE_ROOT, role)
    assert result["release_id"] == RELEASE_ROOT.name
    assert result["dataset_role"] == role
    assert set(result["split_files"]) == {"train", "validation"}
    assert result["test_verified_but_not_loaded"] is True
    assert all("test.jsonl" not in str(path) for path in result["split_files"].values())
    assert result["system_prompt_sha256"] == CHK4_STUDENT_SYSTEM_PROMPT_SHA256


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_superseded_release_cannot_be_runtime_bound(version: str) -> None:
    root = (
        REPO_ROOT / "dataset/processed/retrain_v2/"
        f"chk4_decision_warmstart_grpo_core_{version}_20260810"
    )
    with pytest.raises(DatasetReleaseValidationError, match="structurally replayed v3"):
        verify_chk4_decision_release(
            dataset_dir=root / "decision_sft",
            manifest_path=root / "release_manifest.json",
            expected_manifest_sha256=_sha256_file(root / "release_manifest.json"),
            dataset_role="decision_sft",
            system_prompt=CHK4_STUDENT_SYSTEM_PROMPT,
            model_path=MODEL_ROOT,
        )


def test_script_argument_types_expose_chk4_binding_fields() -> None:
    for argument_type in (SFTScriptArguments, GRPOScriptArguments):
        fields = argument_type.__dataclass_fields__
        assert {
            "dataset_chk4_role",
            "dataset_chk4_release_manifest",
            "dataset_chk4_release_manifest_sha256",
        } <= set(fields)


def test_binding_requires_all_fields_and_exact_train_eval_routes() -> None:
    trainer = object.__new__(_BindingTrainer)
    trainer.script_args = SimpleNamespace(
        dataset_chk4_role="decision_sft",
        dataset_chk4_release_manifest=str(RELEASE_ROOT / "release_manifest.json"),
        dataset_chk4_release_manifest_sha256=None,
        dataset_prompt_column="prompt",
        dataset_train_split="train",
        dataset_test_split="validation",
        user_prompt_suffix=None,
    )
    with pytest.raises(ValueError, match="must all be configured"):
        trainer._dataset_binding()

    trainer.script_args.dataset_chk4_release_manifest_sha256 = MANIFEST_SHA256
    trainer.script_args.dataset_test_split = "test"
    with pytest.raises(ValueError, match="dataset_test_split=validation"):
        trainer._dataset_binding()

    trainer.script_args.dataset_test_split = "validation"
    mode, binding = trainer._dataset_binding()
    assert mode == "chk4_decision_release"
    assert binding["dataset_chk4_role"] == "decision_sft"


def test_runtime_receipt_records_pinned_chk4_binding(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "chk4-grpo.yaml"
    config_path.write_text("reward_funcs: [decision_dense_v2]\n", encoding="utf-8")
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    monkeypatch.setenv(
        "FOMC_CHK4_BRANCH_ID",
        "chk4_from_chk1_cp200_sft_grpo_core_v3_20260810",
    )
    monkeypatch.setenv("FOMC_CHK4_BRANCH_STAGE", "decision_grpo")
    monkeypatch.setenv("FOMC_CHK4_BRANCH_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("FOMC_CHK4_BRANCH_CONFIG_SHA256", config_sha256)
    trainer = object.__new__(_BindingTrainer)
    trainer.script_args = SimpleNamespace(
        dataset_name=str(RELEASE_ROOT / "decision_grpo"),
        dataset_prompt_column="prompt",
        dataset_train_split="train",
        dataset_test_split="validation",
        dataset_chk4_role="decision_grpo",
        dataset_chk4_release_manifest=str(RELEASE_ROOT / "release_manifest.json"),
        dataset_chk4_release_manifest_sha256=MANIFEST_SHA256,
        user_prompt_suffix=None,
        reward_funcs=["decision_dense_v2"],
    )
    trainer.training_args = SimpleNamespace(
        output_dir=str(tmp_path),
        system_prompt=CHK4_STUDENT_SYSTEM_PROMPT,
        reward_weights=[1.0],
        report_to=[],
        device="cpu",
        n_gpu=0,
        world_size=1,
        process_index=0,
        local_process_index=0,
    )
    trainer.model_args = SimpleNamespace(
        model_name_or_path=str(MODEL_ROOT),
        model_revision=None,
        dtype="bfloat16",
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
        load_in_4bit=False,
        load_in_8bit=False,
        use_bnb_nested_quant=False,
    )
    trainer.peft_args = SimpleNamespace(
        peft_merged_model_path="merged",
        peft_r=32,
        peft_lora_alpha=64,
        peft_lora_dropout=0.05,
        peft_target_modules=["q_proj"],
    )
    monkeypatch.setattr(
        "open_r1.trainer.trainer.get_quantization_config", lambda _: None
    )

    runtime = trainer._build_runtime_config()
    receipt = runtime["dataset"]["chk4_decision"]
    assert receipt["schema_version"] == "chk4-decision-runtime-binding-v1"
    assert receipt["scope"] == {
        "role": "decision_grpo",
        "canonical_dag_bindable": False,
        "test_is_sealed_evaluation_only": True,
    }
    assert receipt["release_manifest"] == {
        "path": str(RELEASE_ROOT / "release_manifest.json"),
        "sha256": MANIFEST_SHA256,
    }
    assert receipt["system_prompt_sha256"] == CHK4_STUDENT_SYSTEM_PROMPT_SHA256
    tokenizer = receipt["tokenizer_bundle"]
    assert tokenizer["model_path"] == str(MODEL_ROOT.resolve())
    assert set(tokenizer["files"]) == {
        "chat_template.jinja",
        "config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert len(tokenizer["bundle_sha256"]) == 64
    assert runtime["chk4_standalone_branch"] == {
        "branch_id": "chk4_from_chk1_cp200_sft_grpo_core_v3_20260810",
        "stage": "decision_grpo",
        "config": {"path": str(config_path.resolve()), "sha256": config_sha256},
    }


def test_wrong_manifest_role_or_system_prompt_fails_closed() -> None:
    with pytest.raises(DatasetReleaseValidationError, match="SHA-256 disagrees"):
        _verify(RELEASE_ROOT, "decision_sft", "0" * 64)
    with pytest.raises(DatasetReleaseValidationError, match="role directory"):
        verify_chk4_decision_release(
            dataset_dir=RELEASE_ROOT / "decision_sft",
            manifest_path=RELEASE_ROOT / "release_manifest.json",
            expected_manifest_sha256=MANIFEST_SHA256,
            dataset_role="decision_grpo",
            system_prompt=CHK4_STUDENT_SYSTEM_PROMPT,
            model_path=MODEL_ROOT,
        )
    with pytest.raises(DatasetReleaseValidationError, match="system_prompt"):
        verify_chk4_decision_release(
            dataset_dir=RELEASE_ROOT / "decision_sft",
            manifest_path=RELEASE_ROOT / "release_manifest.json",
            expected_manifest_sha256=MANIFEST_SHA256,
            dataset_role="decision_sft",
            system_prompt=CHK4_STUDENT_SYSTEM_PROMPT + "drift",
            model_path=MODEL_ROOT,
        )


def test_current_model_must_inherit_the_release_tokenizer(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    tokenizer_files = json.loads(
        (RELEASE_ROOT / "release_manifest.json").read_text(encoding="utf-8")
    )["sources"]["tokenizer"]["files"]
    for filename in tokenizer_files:
        shutil.copy2(MODEL_ROOT / filename, model / filename)

    result = verify_chk4_decision_release(
        dataset_dir=RELEASE_ROOT / "decision_sft",
        manifest_path=RELEASE_ROOT / "release_manifest.json",
        expected_manifest_sha256=MANIFEST_SHA256,
        dataset_role="decision_sft",
        system_prompt=CHK4_STUDENT_SYSTEM_PROMPT,
        model_path=model,
    )
    assert result["tokenizer_binding"]["model_path"] == str(model.resolve())

    tokenizer_config = model / "tokenizer_config.json"
    tokenizer_config.write_bytes(tokenizer_config.read_bytes() + b"\n")
    with pytest.raises(
        DatasetReleaseValidationError, match="tokenizer byte-size drift"
    ):
        verify_chk4_decision_release(
            dataset_dir=RELEASE_ROOT / "decision_sft",
            manifest_path=RELEASE_ROOT / "release_manifest.json",
            expected_manifest_sha256=MANIFEST_SHA256,
            dataset_role="decision_sft",
            system_prompt=CHK4_STUDENT_SYSTEM_PROMPT,
            model_path=model,
        )


def test_tampered_release_member_fails_its_manifest_hash(tmp_path: Path) -> None:
    root = _copy_release(tmp_path)
    _make_writable(root)
    train_path = root / "decision_grpo/train.jsonl"
    train_path.write_text(
        train_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8"
    )
    _seal(root)
    with pytest.raises(DatasetReleaseValidationError, match="byte-size drift"):
        _verify(root, "decision_grpo")


def test_self_consistent_bad_role_schema_still_fails(tmp_path: Path) -> None:
    root = _copy_release(tmp_path)
    _make_writable(root)
    train_path = root / "decision_sft/train.jsonl"
    rows = [
        json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["unexpected"] = True
    train_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    manifest_path = root / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    descriptor = manifest["files"]["decision_sft/train.jsonl"]
    descriptor["bytes"] = train_path.stat().st_size
    descriptor["sha256"] = _sha256_file(train_path)
    descriptor["rows"] = len(rows)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_sha = _rewrite_manifest_binding(root)
    _seal(root)
    with pytest.raises(DatasetReleaseValidationError, match="Decision-SFT schema"):
        _verify(root, "decision_sft", manifest_sha)


def test_handoff_unsigned_payload_is_manifest_bound(tmp_path: Path) -> None:
    root = _copy_release(tmp_path)
    _make_writable(root)
    handoff_path = root / "handoff.json"
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    handoff["known_limitations"].append("tampered")
    handoff_path.write_text(
        json.dumps(handoff, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _seal(root)
    with pytest.raises(DatasetReleaseValidationError, match="unsigned payload drift"):
        _verify(root, "decision_sft")
