from __future__ import annotations

import hashlib
import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from safetensors.numpy import save_file

from jobs.retrain_v2 import execution_receipt
from jobs.retrain_v2.execution_contract import canonical_stage_topology
from jobs.retrain_v2.execution_receipt import (
    ExecutionReceiptError,
    merge_and_record_receipt,
    record_merge_receipt,
    record_training_receipt,
    stage_recovery_state,
    verify_stage_receipts,
)
from jobs.retrain_v2.merge_adapter import merge_from_config
from jobs.retrain_v2.merge_attestation import (
    SEMANTIC_BOUNDARY,
    SEMANTIC_METHOD,
    create_merge_attestation,
)
from jobs.retrain_v2.stage_lock import canonical_stage_lock_path
from open_r1.provenance import fingerprint_artifact_path
from open_r1.trainer.trainer import Trainer


class _RuntimeBuilderTrainer(Trainer):
    def load_trainer(self):
        raise NotImplementedError

    def plot_customized_curve(self):
        raise NotImplementedError


def _canonical_sha(payload: object) -> str:
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(rendered).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _runtime_config(config: dict, *, stage_id: str) -> dict:
    topology = canonical_stage_topology()[stage_id]
    reward_funcs = config.get("reward_funcs", [])
    payload = {
        "schema_version": 1,
        "model": {
            "model_name_or_path": config["model_name_or_path"],
            "model_revision": config["model_revision"],
            "torch_dtype": config["dtype"],
            "dtype": config["dtype"],
            "attn_implementation": config["attn_implementation"],
            "quantization": {
                "enabled": True,
                "load_in_4bit": True,
                "load_in_8bit": False,
                "bnb_4bit_quant_type": config["bnb_4bit_quant_type"],
                "bnb_4bit_compute_dtype": config["dtype"],
                "bnb_4bit_quant_storage": config["bnb_4bit_quant_storage"],
                "bnb_4bit_use_double_quant": config["use_bnb_nested_quant"],
                "prepared_for_kbit_training": True,
            },
        },
        "dataset": {
            "name": config["dataset_name"],
            "prompt_column": config["dataset_prompt_column"],
            "train_split": config["dataset_train_split"],
            "eval_split": config["dataset_test_split"],
        },
        "training": {
            "output_dir": config["output_dir"],
            "learning_rate": config["learning_rate"],
            "num_train_epochs": config["num_train_epochs"],
            "max_steps": -1,
            "optimizer": config["optim"],
            "lr_scheduler_type": config["lr_scheduler_type"],
            "warmup_ratio": config["warmup_ratio"],
            "gradient_accumulation_steps": config[
                "gradient_accumulation_steps"
            ],
            "gradient_checkpointing": config["gradient_checkpointing"],
            "per_device_train_batch_size": config[
                "per_device_train_batch_size"
            ],
            "per_device_eval_batch_size": config["per_device_eval_batch_size"],
            "seed": config["seed"],
            "bf16": config["bf16"],
        },
        "generation": {
            key: config.get(key)
            for key in (
                "max_prompt_length",
                "max_completion_length",
                "num_generations",
                "temperature",
                "top_p",
            )
        },
        "peft": {
            "merged_model_path": config["peft_merged_model_path"],
            "r": config["peft_r"],
            "lora_alpha": config["peft_lora_alpha"],
            "lora_dropout": config["peft_lora_dropout"],
            "target_modules": config["peft_target_modules"],
        },
        "rewards": {
            "reward_funcs": reward_funcs,
            "reward_weights": config.get("reward_weights"),
        },
        "environment": {
            "device": "cuda:0",
            "n_gpu": 1,
            "world_size": topology["world_size"],
            "process_index": 0,
            "local_process_index": 0,
            "cuda_visible_devices": ",".join(
                str(item) for item in topology["policy_gpus"]
            ),
            "cuda_available": True,
            "cuda_device_count": len(topology["policy_gpus"]),
            "cuda_device_names": [
                "NVIDIA A30" for _item in topology["policy_gpus"]
            ],
        },
    }
    if reward_funcs == ["grounded_analysis_v2"]:
        payload["judge"] = {
            "url": config["judge_url"],
            "model": config["judge_model"],
            "timeout": config["judge_timeout"],
            "verbose": config["judge_verbose"],
            "sleep_seconds": 0.0,
            "api_key_env": config["judge_api_key_env"],
            "max_retries": config["judge_max_retries"],
            "backoff_seconds": config["judge_backoff_seconds"],
            "tokenizer_path": config["judge_tokenizer_path"],
            "max_model_len": config["judge_max_model_len"],
            "max_completion_tokens": config["judge_max_completion_tokens"],
            "candidate_reserve_tokens": config[
                "judge_candidate_reserve_tokens"
            ],
            "boundary_margin_tokens": config["judge_boundary_margin_tokens"],
        }
    return payload


def _resolved_config(*, stage_id: str) -> dict:
    run_root = "output/training/retrain_v2/run_test"
    config = {
        "model_name_or_path": "models/parent",
        "model_revision": None,
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "use_bnb_nested_quant": True,
        "bnb_4bit_quant_storage": "bfloat16",
        "dataset_name": f"fixtures/{stage_id}",
        "dataset_prompt_column": "prompt",
        "dataset_train_split": "train",
        "dataset_test_split": "validation",
        "output_dir": f"{run_root}/adapters/{stage_id}",
        "peft_merged_model_path": f"{run_root}/merged/{stage_id}",
        "learning_rate": 5.0e-5,
        "num_train_epochs": 2,
        "optim": "paged_adamw_8bit",
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "seed": 42,
        "bf16": True,
        "peft_r": 32,
        "peft_lora_alpha": 64,
        "peft_lora_dropout": 0.05,
        "peft_target_modules": TARGET_MODULES,
    }
    if stage_id == "chk2":
        config.update(
            {
                "reward_funcs": ["grounded_analysis_v2"],
                "reward_weights": [1.0],
                "judge_url": "http://127.0.0.1:8000/v1/chat/completions",
                "judge_model": "Qwen3.5-9B",
                "judge_timeout": 180,
                "judge_verbose": False,
                "judge_api_key_env": "OPEN_R1_JUDGE_API_KEY",
                "judge_max_retries": 3,
                "judge_backoff_seconds": 1.0,
                "judge_tokenizer_path": "models/Qwen3.5-9B",
                "judge_max_model_len": 8192,
                "judge_max_completion_tokens": 2048,
                "judge_candidate_reserve_tokens": 1536,
                "judge_boundary_margin_tokens": 32,
                "max_completion_length": 1024,
                "num_generations": 4,
                "temperature": 0.7,
                "top_p": 0.9,
            }
        )
    return config


def _write_adapter(root: Path, *, config: dict, stage_id: str) -> None:
    root.mkdir(parents=True)
    _write_json(
        root / "adapter_config.json",
        {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": config["model_name_or_path"],
            "r": config["peft_r"],
            "lora_alpha": config["peft_lora_alpha"],
            "lora_dropout": config["peft_lora_dropout"],
            "target_modules": config["peft_target_modules"],
            "bias": "none",
            "fan_in_fan_out": False,
            "use_dora": False,
            "use_rslora": False,
            "inference_mode": True,
            "corda_config": None,
            "eva_config": None,
            "exclude_modules": None,
            "lora_bias": False,
            "trainable_token_indices": None,
            "alpha_pattern": {},
            "rank_pattern": {},
            "modules_to_save": None,
            "layers_to_transform": None,
            "layer_replication": None,
        },
    )
    _write_json(root / "trainer_state.json", {"global_step": 3})
    _write_json(
        root / "train_results.json", {"train_samples": 8, "train_runtime": 1.5}
    )
    _write_json(
        root / "resolved_runtime_config.json",
        _runtime_config(config, stage_id=stage_id),
    )
    (root / "training_args.bin").write_bytes(b"opaque-training-args")
    save_file(
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": np.ones(
                (2, 2), dtype=np.float32
            ),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": np.ones(
                (2, 2), dtype=np.float32
            ),
        },
        root / "adapter_model.safetensors",
    )


def _write_merged(root: Path) -> None:
    root.mkdir(parents=True)
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        _write_json(root / filename, {"artifact": filename})
    save_file(
        {"model.embed_tokens.weight": np.ones((2, 3), dtype=np.float32)},
        root / "model.safetensors",
    )


def _semantic_evidence() -> dict:
    return {
        "method": SEMANTIC_METHOD,
        "merge_completed": True,
        "lora_modules_before": 2,
        "residual_lora_modules_after": 0,
        "residual_lora_state_keys_after": 0,
        "merged_parameter_tensors": 1,
        "merged_parameter_count": 6,
        "functional_equivalence_boundary": SEMANTIC_BOUNDARY,
    }


def _create_expected_merge_attestation(
    root: Path, manifest: Path, *, stage_id: str
) -> dict:
    context = execution_receipt._manifest_context(
        manifest, stage_id, root, require_pending=True
    )
    training = execution_receipt._load_receipt(
        execution_receipt._receipt_path(context, "training"),
        expected_type="training",
    )
    return create_merge_attestation(
        context["merged_path"],
        binding=execution_receipt._merge_attestation_binding(context, training),
        semantic_evidence=_semantic_evidence(),
    )


def _record_merge_receipt(root: Path, manifest: Path, stage_id: str) -> dict:
    attestation = manifest.parent / f"merged/{stage_id}/merge_attestation.json"
    if not attestation.exists() and not attestation.is_symlink():
        _create_expected_merge_attestation(root, manifest, stage_id=stage_id)
    return record_merge_receipt(manifest, stage_id, root)


def test_runtime_config_producer_emits_strict_nonsecret_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = object.__new__(_RuntimeBuilderTrainer)
    trainer.model_args = SimpleNamespace(
        model_name_or_path="models/parent",
        model_revision=None,
        torch_dtype="bfloat16",
        dtype="bfloat16",
        attn_implementation="sdpa",
        load_in_4bit=True,
        load_in_8bit=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=None,
        bnb_4bit_quant_storage="bfloat16",
        use_bnb_nested_quant=True,
    )
    trainer.script_args = SimpleNamespace(
        dataset_name="fixtures/chk2",
        dataset_prompt_column="prompt",
        dataset_train_split="train",
        dataset_test_split="validation",
        reward_funcs=["grounded_analysis_v2"],
        judge_url="http://127.0.0.1:8000/v1/chat/completions",
        judge_model="Qwen3.5-9B",
        judge_timeout=180,
        judge_verbose=False,
        judge_sleep_seconds=0.0,
        judge_api_key_env="TEST_JUDGE_KEY",
        judge_max_retries=3,
        judge_backoff_seconds=1.0,
        judge_tokenizer_path="models/Qwen3.5-9B",
        judge_max_model_len=8192,
        judge_max_completion_tokens=2048,
        judge_candidate_reserve_tokens=1536,
        judge_boundary_margin_tokens=32,
    )
    trainer.training_args = SimpleNamespace(
        output_dir="output/training/retrain_v2/run_test/adapters/chk2",
        learning_rate=5.0e-7,
        num_train_epochs=1,
        max_steps=-1,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=4,
        seed=42,
        bf16=True,
        max_prompt_length=None,
        max_completion_length=1024,
        num_generations=4,
        temperature=0.7,
        top_p=0.9,
        reward_weights=[1.0],
        device="cuda:0",
        n_gpu=1,
        world_size=1,
        process_index=0,
        local_process_index=0,
    )
    trainer.peft_args = SimpleNamespace(
        peft_merged_model_path="output/training/retrain_v2/run_test/merged/chk2",
        peft_r=32,
        peft_lora_alpha=64,
        peft_lora_dropout=0.05,
        peft_target_modules=TARGET_MODULES,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("TEST_JUDGE_KEY", "must-not-be-persisted")
    monkeypatch.setattr(
        "open_r1.trainer.trainer.get_quantization_config", lambda _args: object()
    )
    monkeypatch.setattr("open_r1.trainer.trainer.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("open_r1.trainer.trainer.torch.cuda.device_count", lambda: 1)
    monkeypatch.setattr(
        "open_r1.trainer.trainer.torch.cuda.get_device_name",
        lambda _index: "NVIDIA A30",
    )

    runtime = trainer._build_runtime_config()

    assert set(runtime) == execution_receipt.RUNTIME_TOP_LEVEL_KEYS | {"judge"}
    assert set(runtime["model"]["quantization"]) == (
        execution_receipt.RUNTIME_QUANTIZATION_KEYS
    )
    assert set(runtime["environment"]) == execution_receipt.RUNTIME_ENVIRONMENT_KEYS
    assert runtime["environment"]["cuda_visible_devices"] == "1"
    assert "api_key" not in runtime["judge"]
    assert "must-not-be-persisted" not in json.dumps(runtime)


def _make_receipt_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, stage_id: str
) -> tuple[Path, Path]:
    parent = tmp_path / "models/parent"
    parent.mkdir(parents=True)
    (parent / "config.json").write_text("{}\n", encoding="utf-8")
    run_root = tmp_path / "output/training/retrain_v2/run_test"
    config_path = run_root / f"resolved_configs/{stage_id}.yaml"
    config_path.parent.mkdir(parents=True)
    config = _resolved_config(stage_id=stage_id)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    dataset = tmp_path / config["dataset_name"]
    dataset.mkdir(parents=True)
    (dataset / "train.jsonl").write_text("{}\n", encoding="utf-8")
    adapter = run_root / f"adapters/{stage_id}"
    merged = run_root / f"merged/{stage_id}"
    _write_adapter(adapter, config=config, stage_id=stage_id)
    _write_merged(merged)

    execution_contract = {
        "contract_sha256": "e" * 64,
        "stage_topology": canonical_stage_topology(),
    }
    manifest = {
        "schema_version": 2,
        "run_id": "run_test",
        "execution_contract": execution_contract,
        "stages": {
            stage_id: {
                "status": "pending",
                "artifact_path": str(merged),
                "resolved_config": {
                    "path": str(config_path),
                    "sha256": _file_sha(config_path),
                },
                "data_binding": {
                    "release_role": "base",
                    "release_sha256": "b" * 64,
                    "dataset_path": str(dataset),
                    "dataset_sha256": "d" * 64,
                },
            }
        },
    }
    manifest_path = run_root / "run_manifest.json"
    _write_json(manifest_path, manifest)

    def fake_verify_parent(*_args, **_kwargs):
        return {
            "parent": {"path": str(parent), "sha256": "a" * 64},
            "dataset": {"artifact": {"sha256": "d" * 64}},
        }

    def fake_verify_execution_contract(record, _repo_root):
        return {"status": "verified", "contract_sha256": record["contract_sha256"]}

    monkeypatch.setattr(execution_receipt, "verify_parent", fake_verify_parent)
    monkeypatch.setattr(
        execution_receipt,
        "verify_execution_contract",
        fake_verify_execution_contract,
    )
    if stage_id == "chk2":
        from jobs.retrain_v2 import judge_attestation

        attestation_root = run_root / "attestations"
        for phase in ("pre", "post"):
            _write_json(
                attestation_root / f"judge.{phase}.json",
                {"phase": phase, "content": f"test-{phase}"},
            )

        def fake_verify_judge_attestations(*_args, **_kwargs):
            result = {
                "status": "verified",
                "run_id": "run_test",
                "service_identity_sha256": "f" * 64,
            }
            for phase in ("pre", "post"):
                path = attestation_root / f"judge.{phase}.json"
                if not path.is_file():
                    raise judge_attestation.JudgeAttestationError(
                        f"Judge {phase} attestation is missing"
                    )
                result[f"{phase}_attestation"] = {
                    "path": path.relative_to(tmp_path).as_posix(),
                    "file_sha256": _file_sha(path),
                    "canonical_payload_sha256": (
                        "c" * 64 if phase == "pre" else "d" * 64
                    ),
                }
            return result

        monkeypatch.setattr(
            judge_attestation,
            "verify_judge_attestations",
            fake_verify_judge_attestations,
        )
    return tmp_path, manifest_path


def _acquire_stage_lock(
    root: Path,
    manifest: Path,
    *,
    stage_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> int:
    path = canonical_stage_lock_path(manifest, stage_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    monkeypatch.setenv("FOMC_RETRAIN_STAGE_LOCK_PATH", str(path))
    monkeypatch.setenv("FOMC_RETRAIN_STAGE_LOCK_STAGE", stage_id)
    monkeypatch.setenv("FOMC_RETRAIN_STAGE_LOCK_FD", str(descriptor))
    return descriptor


@pytest.fixture
def receipt_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    root, manifest = _make_receipt_repo(tmp_path, monkeypatch, stage_id="chk1")
    descriptor = _acquire_stage_lock(
        root, manifest, stage_id="chk1", monkeypatch=monkeypatch
    )
    try:
        yield root, manifest
    finally:
        os.close(descriptor)


@pytest.fixture
def grpo_receipt_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    root, manifest = _make_receipt_repo(tmp_path, monkeypatch, stage_id="chk2")
    descriptor = _acquire_stage_lock(
        root, manifest, stage_id="chk2", monkeypatch=monkeypatch
    )
    try:
        yield root, manifest
    finally:
        os.close(descriptor)


def test_record_and_verify_training_and_merge_receipts(
    receipt_repo: tuple[Path, Path],
) -> None:
    root, manifest = receipt_repo
    training = record_training_receipt(manifest, "chk1", root)
    merge = _record_merge_receipt(root, manifest, "chk1")
    verified = verify_stage_receipts(manifest, "chk1", root)

    assert training["status"] == merge["status"] == "recorded"
    assert training["lineage"]["topology"]["world_size"] == 2
    assert training["lineage"]["stable_files"].keys() >= {
        "adapter_config.json",
        "trainer_state.json",
        "train_results.json",
        "resolved_runtime_config.json",
        "training_args.bin",
        "adapter_model.safetensors",
    }
    assert merge["lineage"]["training_receipt_sha256"] == training[
        "lineage_sha256"
    ]
    assert verified["status"] == "verified"
    assert verified["adapter_sha256"] == merge["lineage"]["adapter_sha256"]
    assert verified["adapter_sha256"] == fingerprint_artifact_path(
        manifest.parent / "adapters/chk1"
    )["sha256"]
    assert verified["merged_sha256"] == fingerprint_artifact_path(
        manifest.parent / "merged/chk1"
    )["sha256"]

    run_root = manifest.parent
    assert (run_root / "receipts/chk1.training.json").is_file()
    assert (run_root / "receipts/chk1.merge.json").is_file()
    assert not list((run_root / "receipts").glob("*.tmp"))


def test_receipts_are_exclusive_and_never_overwritten(
    receipt_repo: tuple[Path, Path],
) -> None:
    root, manifest = receipt_repo
    first = record_training_receipt(manifest, "chk1", root)
    receipt_path = Path(first["path"])
    before = receipt_path.read_bytes()
    with pytest.raises(ExecutionReceiptError, match="already exists"):
        record_training_receipt(manifest, "chk1", root)
    assert receipt_path.read_bytes() == before

    _record_merge_receipt(root, manifest, "chk1")
    merge_path = manifest.parent / "receipts/chk1.merge.json"
    merge_before = merge_path.read_bytes()
    with pytest.raises(ExecutionReceiptError, match="already exists"):
        _record_merge_receipt(root, manifest, "chk1")
    assert merge_path.read_bytes() == merge_before


@pytest.mark.parametrize(
    "filename",
    [
        "adapter_config.json",
        "trainer_state.json",
        "train_results.json",
        "resolved_runtime_config.json",
        "training_args.bin",
        "adapter_model.safetensors",
    ],
)
def test_training_receipt_requires_all_stable_root_files(
    receipt_repo: tuple[Path, Path], filename: str
) -> None:
    root, manifest = receipt_repo
    (manifest.parent / "adapters/chk1" / filename).unlink()
    with pytest.raises(ExecutionReceiptError):
        record_training_receipt(manifest, "chk1", root)


def test_pickle_adapter_and_incomplete_lora_safetensors_are_rejected(
    receipt_repo: tuple[Path, Path],
) -> None:
    root, manifest = receipt_repo
    adapter = manifest.parent / "adapters/chk1"
    (adapter / "adapter_model.bin").write_bytes(b"pickle")
    with pytest.raises(ExecutionReceiptError, match="Only the root"):
        record_training_receipt(manifest, "chk1", root)

    (adapter / "adapter_model.bin").unlink()
    save_file(
        {"layer.lora_A.weight": np.ones((1, 1), dtype=np.float32)},
        adapter / "adapter_model.safetensors",
    )
    with pytest.raises(ExecutionReceiptError, match="no LoRA B"):
        record_training_receipt(manifest, "chk1", root)


@pytest.mark.parametrize(
    ("filename", "field", "value", "error"),
    [
        ("adapter_config.json", "peft_type", "IA3", "peft_type"),
        ("adapter_config.json", "task_type", "SEQ_CLS", "task_type"),
        (
            "adapter_config.json",
            "base_model_name_or_path",
            "models/other",
            "does not exist",
        ),
        ("trainer_state.json", "global_step", 0, "global_step"),
        ("train_results.json", "train_samples", 0, "train_samples"),
        ("train_results.json", "train_runtime", 0, "train_runtime"),
    ],
)
def test_adapter_semantic_evidence_is_fail_closed(
    receipt_repo: tuple[Path, Path],
    filename: str,
    field: str,
    value: object,
    error: str,
) -> None:
    root, manifest = receipt_repo
    path = manifest.parent / "adapters/chk1" / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    _write_json(path, payload)
    with pytest.raises(ExecutionReceiptError, match=error):
        record_training_receipt(manifest, "chk1", root)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("r", 8, "adapter r mismatch"),
        ("lora_alpha", 32, "adapter lora_alpha mismatch"),
        ("lora_dropout", 0.1, "adapter lora_dropout mismatch"),
        ("target_modules", ["q_proj"], "target_modules mismatch"),
        ("bias", "all", "adapter bias mismatch"),
        ("use_dora", True, "adapter use_dora mismatch"),
        ("use_rslora", True, "adapter use_rslora mismatch"),
    ],
)
def test_adapter_lora_semantics_must_exactly_match_resolved_yaml(
    receipt_repo: tuple[Path, Path], field: str, value: object, error: str
) -> None:
    root, manifest = receipt_repo
    adapter_config = manifest.parent / "adapters/chk1/adapter_config.json"
    payload = json.loads(adapter_config.read_text(encoding="utf-8"))
    payload[field] = value
    _write_json(adapter_config, payload)

    with pytest.raises(ExecutionReceiptError, match=error):
        record_training_receipt(manifest, "chk1", root)


@pytest.mark.parametrize(
    ("section", "field", "value", "error"),
    [
        ("model", "dtype", "float16", "runtime model.dtype mismatch"),
        ("dataset", "prompt_column", "forged", "Runtime dataset mismatch"),
        (
            "training",
            "output_dir",
            "output/training/retrain_v2/run_test/adapters/forged",
            "runtime training.output_dir mismatch",
        ),
        ("generation", "max_completion_length", 999, "runtime generation"),
        ("peft", "r", 8, "Runtime PEFT config mismatch"),
        ("environment", "world_size", 1, "Runtime world_size mismatch"),
        ("environment", "device", "cuda", "rank-0 device"),
        (
            "environment",
            "cuda_visible_devices",
            "1",
            "CUDA_VISIBLE_DEVICES topology mismatch",
        ),
    ],
)
def test_runtime_config_values_are_bound_to_yaml_and_topology(
    receipt_repo: tuple[Path, Path],
    section: str,
    field: str,
    value: object,
    error: str,
) -> None:
    root, manifest = receipt_repo
    runtime_path = manifest.parent / "adapters/chk1/resolved_runtime_config.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime[section][field] = value
    _write_json(runtime_path, runtime)

    with pytest.raises(ExecutionReceiptError, match=error):
        record_training_receipt(manifest, "chk1", root)


def test_runtime_config_schema_is_exact_and_placeholder_is_rejected(
    receipt_repo: tuple[Path, Path],
) -> None:
    root, manifest = receipt_repo
    runtime_path = manifest.parent / "adapters/chk1/resolved_runtime_config.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["forged"] = True
    _write_json(runtime_path, runtime)
    with pytest.raises(ExecutionReceiptError, match="runtime config schema mismatch"):
        record_training_receipt(manifest, "chk1", root)

    _write_json(runtime_path, {"schema_version": 1, "world_size": 2})
    with pytest.raises(ExecutionReceiptError, match="runtime config schema mismatch"):
        record_training_receipt(manifest, "chk1", root)


def test_runtime_dataset_is_bound_to_manifest_release(
    receipt_repo: tuple[Path, Path],
) -> None:
    root, manifest = receipt_repo
    other = root / "fixtures/other"
    other.mkdir(parents=True)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["stages"]["chk1"]["data_binding"]["dataset_path"] = str(other)
    _write_json(manifest, payload)

    with pytest.raises(ExecutionReceiptError, match="Runtime dataset binding mismatch"):
        record_training_receipt(manifest, "chk1", root)


@pytest.mark.parametrize(
    ("section", "field", "value", "error"),
    [
        (
            "rewards",
            "reward_funcs",
            ["decision_dense_v2"],
            "Runtime reward functions mismatch",
        ),
        ("rewards", "reward_weights", [0.5], "Runtime reward weights mismatch"),
        ("judge", "model", "forged", "Runtime judge config mismatch"),
    ],
)
def test_grpo_runtime_rewards_and_judge_are_exactly_bound(
    grpo_receipt_repo: tuple[Path, Path],
    section: str,
    field: str,
    value: object,
    error: str,
) -> None:
    root, manifest = grpo_receipt_repo
    runtime_path = manifest.parent / "adapters/chk2/resolved_runtime_config.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime[section][field] = value
    _write_json(runtime_path, runtime)

    with pytest.raises(ExecutionReceiptError, match=error):
        record_training_receipt(manifest, "chk2", root)


def test_stage_reward_contract_rejects_self_consistent_forgery(
    receipt_repo: tuple[Path, Path],
) -> None:
    root, manifest = receipt_repo
    config_path = manifest.parent / "resolved_configs/chk1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["reward_funcs"] = ["decision_dense_v2"]
    config["reward_weights"] = [1.0]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["stages"]["chk1"]["resolved_config"]["sha256"] = _file_sha(
        config_path
    )
    _write_json(manifest, manifest_payload)
    runtime_path = manifest.parent / "adapters/chk1/resolved_runtime_config.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["rewards"] = {
        "reward_funcs": ["decision_dense_v2"],
        "reward_weights": [1.0],
    }
    _write_json(runtime_path, runtime)

    with pytest.raises(ExecutionReceiptError, match="Resolved reward functions mismatch"):
        record_training_receipt(manifest, "chk1", root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("corda_config", {"method": "ipm"}),
        ("eva_config", {}),
        ("exclude_modules", ["q_proj"]),
        ("lora_bias", True),
        ("trainable_token_indices", [1]),
    ],
)
def test_receipt_and_merge_reject_nondefault_peft_compatibility_semantics(
    receipt_repo: tuple[Path, Path],
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    root, manifest = receipt_repo
    source_adapter = manifest.parent / "adapters/chk1"
    adapter_config = source_adapter / "adapter_config.json"
    payload = json.loads(adapter_config.read_text(encoding="utf-8"))
    payload[field] = value
    _write_json(adapter_config, payload)
    with pytest.raises(ExecutionReceiptError, match=field):
        record_training_receipt(manifest, "chk1", root)

    merge_config = tmp_path / "merge.yaml"
    destination = tmp_path / "standalone-merged"
    merge_config.write_text(
        yaml.safe_dump(
            {
                "model_name_or_path": str(root / "models/parent"),
                "output_dir": str(source_adapter),
                "peft_merged_model_path": str(destination),
            }
        ),
        encoding="utf-8",
    )
    called = False

    def should_not_merge(*_args):
        nonlocal called
        called = True

    with pytest.raises(ValueError, match=f"Adapter {field} uses unsupported"):
        merge_from_config(merge_config, merge_executor=should_not_merge)
    assert called is False
    assert not destination.exists()


def test_adapter_drift_after_training_blocks_merge_and_verification(
    receipt_repo: tuple[Path, Path],
) -> None:
    root, manifest = receipt_repo
    record_training_receipt(manifest, "chk1", root)
    (manifest.parent / "adapters/chk1/untracked.txt").write_text(
        "drift\n", encoding="utf-8"
    )
    with pytest.raises(ExecutionReceiptError, match="Training receipt lineage drift"):
        _record_merge_receipt(root, manifest, "chk1")


def test_self_consistent_receipt_path_claim_is_not_trusted(
    receipt_repo: tuple[Path, Path],
) -> None:
    root, manifest = receipt_repo
    result = record_training_receipt(manifest, "chk1", root)
    path = Path(result["path"])
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["lineage"]["adapter_path"] = "../../outside"
    receipt["lineage_sha256"] = _canonical_sha(receipt["lineage"])
    _write_json(path, receipt)

    with pytest.raises(ExecutionReceiptError, match="Training receipt lineage drift"):
        _record_merge_receipt(root, manifest, "chk1")


def test_sharded_merged_model_is_bound_through_canonical_index(
    receipt_repo: tuple[Path, Path],
) -> None:
    root, manifest = receipt_repo
    merged = manifest.parent / "merged/chk1"
    (merged / "model.safetensors").unlink()
    shard_one = "model-00001-of-00002.safetensors"
    shard_two = "model-00002-of-00002.safetensors"
    save_file(
        {"a.weight": np.ones((2, 2), dtype=np.float32)}, merged / shard_one
    )
    save_file(
        {"b.weight": np.ones((2, 2), dtype=np.float32)}, merged / shard_two
    )
    _write_json(
        merged / "model.safetensors.index.json",
        {"weight_map": {"a.weight": shard_one, "b.weight": shard_two}},
    )

    record_training_receipt(manifest, "chk1", root)
    merge = _record_merge_receipt(root, manifest, "chk1")
    assert {Path(item["path"]).name for item in merge["lineage"]["weights"]} == {
        shard_one,
        shard_two,
    }
    assert verify_stage_receipts(manifest, "chk1", root)["status"] == "verified"


def test_corrupt_or_empty_merged_safetensors_are_rejected(
    receipt_repo: tuple[Path, Path],
) -> None:
    root, manifest = receipt_repo
    merged_weight = manifest.parent / "merged/chk1/model.safetensors"
    merged_weight.write_bytes(b"not-safetensors")
    record_training_receipt(manifest, "chk1", root)

    with pytest.raises(
        ExecutionReceiptError, match="Unable to validate Merged model.safetensors"
    ):
        _record_merge_receipt(root, manifest, "chk1")


def test_empty_merged_tensor_shape_is_rejected(
    receipt_repo: tuple[Path, Path],
) -> None:
    root, manifest = receipt_repo
    save_file(
        {"empty.weight": np.empty((0, 2), dtype=np.float32)},
        manifest.parent / "merged/chk1/model.safetensors",
    )
    record_training_receipt(manifest, "chk1", root)

    with pytest.raises(ExecutionReceiptError, match="tensor 'empty.weight' is empty"):
        _record_merge_receipt(root, manifest, "chk1")


@pytest.mark.parametrize("mutation", ["key_set", "placement", "duplicate"])
def test_sharded_tensor_inventory_must_exactly_match_index(
    receipt_repo: tuple[Path, Path], mutation: str
) -> None:
    root, manifest = receipt_repo
    merged = manifest.parent / "merged/chk1"
    (merged / "model.safetensors").unlink()
    shard_one = "model-00001-of-00002.safetensors"
    shard_two = "model-00002-of-00002.safetensors"
    first_key = "shared.weight" if mutation == "duplicate" else "a.weight"
    second_key = "shared.weight" if mutation == "duplicate" else "b.weight"
    save_file(
        {first_key: np.ones((2, 2), dtype=np.float32)}, merged / shard_one
    )
    save_file(
        {second_key: np.ones((2, 2), dtype=np.float32)}, merged / shard_two
    )
    if mutation == "key_set":
        weight_map = {"a.weight": shard_one, "missing.weight": shard_two}
        message = "tensor key set"
    elif mutation == "placement":
        weight_map = {"a.weight": shard_two, "b.weight": shard_one}
        message = "wrong shard"
    else:
        weight_map = {"shared.weight": shard_one, "placeholder.weight": shard_two}
        message = "duplicated across shards"
    _write_json(
        merged / "model.safetensors.index.json", {"weight_map": weight_map}
    )
    record_training_receipt(manifest, "chk1", root)

    with pytest.raises(ExecutionReceiptError, match=message):
        _record_merge_receipt(root, manifest, "chk1")


def test_merged_index_escape_missing_shard_and_nested_symlink_are_rejected(
    receipt_repo: tuple[Path, Path], tmp_path: Path
) -> None:
    root, manifest = receipt_repo
    merged = manifest.parent / "merged/chk1"
    (merged / "model.safetensors").unlink()
    _write_json(
        merged / "model.safetensors.index.json",
        {"weight_map": {"a.weight": "../outside.safetensors"}},
    )
    record_training_receipt(manifest, "chk1", root)
    with pytest.raises(ExecutionReceiptError, match="canonical root filename"):
        _record_merge_receipt(root, manifest, "chk1")

    (merged / "model.safetensors.index.json").unlink()
    save_file(
        {"model.weight": np.ones((2, 2), dtype=np.float32)},
        merged / "model.safetensors",
    )
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    (merged / "nested-link").symlink_to(outside)
    with pytest.raises(ExecutionReceiptError, match="must not contain symlinks"):
        _record_merge_receipt(root, manifest, "chk1")


def test_configured_output_escape_is_rejected(
    receipt_repo: tuple[Path, Path], tmp_path: Path
) -> None:
    root, manifest = receipt_repo
    config_path = manifest.parent / "resolved_configs/chk1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    outside = tmp_path / "outside-adapter"
    outside.mkdir()
    config["output_dir"] = str(outside)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["stages"]["chk1"]["resolved_config"]["sha256"] = _file_sha(config_path)
    _write_json(manifest, payload)

    with pytest.raises(ExecutionReceiptError, match="Unexpected adapter root"):
        record_training_receipt(manifest, "chk1", root)


def test_verify_requires_merge_receipt_and_allows_sealed_stage(
    receipt_repo: tuple[Path, Path],
) -> None:
    root, manifest = receipt_repo
    record_training_receipt(manifest, "chk1", root)
    with pytest.raises(ExecutionReceiptError, match="merge receipt is missing"):
        verify_stage_receipts(manifest, "chk1", root)
    _record_merge_receipt(root, manifest, "chk1")

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["stages"]["chk1"]["status"] = "sealed"
    _write_json(manifest, payload)
    assert verify_stage_receipts(manifest, "chk1", root)["status"] == "verified"


def test_cli_failure_is_strict_blocked_json(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jobs.retrain_v2.execution_receipt",
            "--repo-root",
            str(tmp_path),
            "verify",
            "--run-manifest",
            str(tmp_path / "missing.json"),
            "--stage",
            "chk1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "blocked"
    assert result.stdout.strip().startswith('{"error":')

    invalid = subprocess.run(
        [
            sys.executable,
            "-m",
            "jobs.retrain_v2.execution_receipt",
            "verify",
            "--run-manifest",
            "missing.json",
            "--stage",
            "not-a-stage",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid.returncode == 2
    assert json.loads(invalid.stdout)["status"] == "blocked"


def test_receipt_writes_and_recovery_require_inherited_stage_lock(
    receipt_repo: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest = receipt_repo
    for name in (
        "FOMC_RETRAIN_STAGE_LOCK_PATH",
        "FOMC_RETRAIN_STAGE_LOCK_STAGE",
        "FOMC_RETRAIN_STAGE_LOCK_FD",
    ):
        monkeypatch.delenv(name)

    with pytest.raises(ExecutionReceiptError, match="lock"):
        record_training_receipt(manifest, "chk1", root)
    with pytest.raises(ExecutionReceiptError, match="lock"):
        record_merge_receipt(manifest, "chk1", root)
    with pytest.raises(ExecutionReceiptError, match="lock"):
        stage_recovery_state(manifest, "chk1", root)


def test_chk2_training_lineage_binds_and_reverifies_both_judge_attestations(
    grpo_receipt_repo: tuple[Path, Path],
) -> None:
    root, manifest = grpo_receipt_repo
    training = record_training_receipt(manifest, "chk2", root)
    binding = training["lineage"]["judge_attestations"]

    assert set(binding) == {
        "schema_version",
        "pre",
        "post",
        "service_identity_sha256",
    }
    for phase in ("pre", "post"):
        assert set(binding[phase]) == {
            "path",
            "file_sha256",
            "canonical_payload_sha256",
        }
    _record_merge_receipt(root, manifest, "chk2")
    post = manifest.parent / "attestations/judge.post.json"
    post.write_text('{"phase":"post","content":"drift"}\n', encoding="utf-8")
    with pytest.raises(ExecutionReceiptError, match="Training receipt lineage drift"):
        verify_stage_receipts(manifest, "chk2", root)


def test_chk2_completed_adapter_is_recoverable_before_post_attestation(
    grpo_receipt_repo: tuple[Path, Path],
) -> None:
    root, manifest = grpo_receipt_repo
    shutil.rmtree(manifest.parent / "merged/chk2")
    (manifest.parent / "attestations/judge.post.json").unlink()

    state = stage_recovery_state(manifest, "chk2", root)
    assert state["state"] == "training_complete_ready_to_record"
    with pytest.raises(ExecutionReceiptError, match="post attestation"):
        record_training_receipt(manifest, "chk2", root)


def test_merge_transaction_attests_publishes_and_records_atomically(
    receipt_repo: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest = receipt_repo
    record_training_receipt(manifest, "chk1", root)
    shutil.rmtree(manifest.parent / "merged/chk1")
    monkeypatch.chdir(root)

    def fake_merge(_base: Path, _adapter: Path, output: Path) -> dict:
        _write_merged(output)
        return _semantic_evidence()

    result = merge_and_record_receipt(
        manifest, "chk1", root, merge_executor=fake_merge
    )

    assert result["status"] == "merged_and_recorded"
    attestation_path = manifest.parent / "merged/chk1/merge_attestation.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    assert attestation["semantic_evidence"] == _semantic_evidence()
    assert "no end-to-end logits equivalence" in attestation[
        "semantic_evidence"
    ]["functional_equivalence_boundary"]
    merge_receipt = json.loads(
        (manifest.parent / "receipts/chk1.merge.json").read_text(encoding="utf-8")
    )
    assert merge_receipt["lineage"]["merge_attestation"][
        "canonical_payload_sha256"
    ] == attestation["canonical_payload_sha256"]
    assert attestation["binding"]["training_receipt"]["file_sha256"] == (
        merge_receipt["lineage"]["training_receipt_file_sha256"]
    )
    assert verify_stage_receipts(manifest, "chk1", root)["status"] == "verified"


def test_merge_transaction_recovers_only_a_fully_attested_publish(
    receipt_repo: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest = receipt_repo
    record_training_receipt(manifest, "chk1", root)
    shutil.rmtree(manifest.parent / "merged/chk1")
    monkeypatch.chdir(root)
    context = execution_receipt._manifest_context(
        manifest, "chk1", root, require_pending=True
    )
    training = execution_receipt._verify_training_receipt(context)

    def fake_merge(_base: Path, _adapter: Path, output: Path) -> dict:
        _write_merged(output)
        return _semantic_evidence()

    merge_from_config(
        context["config_path"],
        merge_executor=fake_merge,
        merge_attestation_binding=execution_receipt._merge_attestation_binding(
            context, training
        ),
    )
    assert not (manifest.parent / "receipts/chk1.merge.json").exists()

    recovered = merge_and_record_receipt(manifest, "chk1", root)
    assert recovered["status"] == "recovered_after_attested_publish"
    assert verify_stage_receipts(manifest, "chk1", root)["status"] == "verified"


def test_merge_transaction_rejects_unattested_or_drifted_published_output(
    receipt_repo: tuple[Path, Path],
) -> None:
    root, manifest = receipt_repo
    record_training_receipt(manifest, "chk1", root)
    with pytest.raises(ExecutionReceiptError, match="attestation is missing"):
        merge_and_record_receipt(manifest, "chk1", root)
    assert not (manifest.parent / "receipts/chk1.merge.json").exists()

    _create_expected_merge_attestation(root, manifest, stage_id="chk1")
    (manifest.parent / "merged/chk1/config.json").write_text(
        '{"drift":true}\n', encoding="utf-8"
    )
    with pytest.raises(ExecutionReceiptError, match="output payload drift"):
        merge_and_record_receipt(manifest, "chk1", root)
    assert not (manifest.parent / "receipts/chk1.merge.json").exists()


def test_legacy_config_only_merge_cli_cannot_publish(
    receipt_repo: tuple[Path, Path],
) -> None:
    _root, manifest = receipt_repo
    config = manifest.parent / "resolved_configs/chk1.yaml"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jobs.retrain_v2.merge_adapter",
            "--config",
            str(config),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--run-manifest" in result.stderr
    assert "--stage" in result.stderr


def _remove_merged(manifest: Path) -> None:
    shutil.rmtree(manifest.parent / "merged/chk1")


def test_recovery_states_fresh_and_completed_training(
    receipt_repo: tuple[Path, Path],
) -> None:
    root, manifest = receipt_repo
    _remove_merged(manifest)
    assert stage_recovery_state(manifest, "chk1", root)["state"] == (
        "training_complete_ready_to_record"
    )

    shutil.rmtree(manifest.parent / "adapters/chk1")
    assert stage_recovery_state(manifest, "chk1", root)["state"] == "fresh_train"


def test_recovery_state_accepts_only_a_complete_latest_checkpoint(
    receipt_repo: tuple[Path, Path],
) -> None:
    root, manifest = receipt_repo
    adapter = manifest.parent / "adapters/chk1"
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    runtime = json.loads(
        (adapter / "resolved_runtime_config.json").read_text(encoding="utf-8")
    )
    weights = (adapter / "adapter_model.safetensors").read_bytes()
    _remove_merged(manifest)
    shutil.rmtree(adapter)
    adapter.mkdir()
    _write_json(adapter / "resolved_runtime_config.json", runtime)
    checkpoint = adapter / "checkpoint-3"
    checkpoint.mkdir()
    _write_json(checkpoint / "adapter_config.json", config)
    _write_json(checkpoint / "trainer_state.json", {"global_step": 3})
    (checkpoint / "adapter_model.safetensors").write_bytes(weights)
    for filename in (
        "training_args.bin",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state_0.pth",
        "rng_state_1.pth",
    ):
        (checkpoint / filename).write_bytes(b"opaque-state")

    state = stage_recovery_state(manifest, "chk1", root)
    assert state["state"] == "checkpoint_resume_ready"
    assert state["checkpoint"]["global_step"] == 3

    (adapter / "checkpoint-4").mkdir()
    with pytest.raises(ExecutionReceiptError, match="resume checkpoint contains no files"):
        stage_recovery_state(manifest, "chk1", root)


def test_grpo_recovery_accepts_only_canonical_completion_logs(
    tmp_path: Path,
) -> None:
    # The full chk2 recovery fixture is exercised elsewhere; this test targets
    # the strict side-directory validator used before checkpoint selection.
    from jobs.retrain_v2.execution_receipt import _validate_partial_completion_logs

    completion_root = tmp_path / "completions"
    completion_root.mkdir()
    (completion_root / "completions_00001.parquet").write_bytes(b"parquet")
    context = {
        "stage_id": "chk2",
        "config": {"log_completions": True},
    }
    _validate_partial_completion_logs(context, completion_root)

    (completion_root / "scratch.tmp").write_bytes(b"partial")
    with pytest.raises(ExecutionReceiptError, match="Unsafe partial completion"):
        _validate_partial_completion_logs(context, completion_root)


def test_recovery_states_receipts_publish_and_seal(
    receipt_repo: tuple[Path, Path],
) -> None:
    root, manifest = receipt_repo
    _remove_merged(manifest)
    record_training_receipt(manifest, "chk1", root)
    assert stage_recovery_state(manifest, "chk1", root)["state"] == (
        "training_receipt_ready_to_merge"
    )

    _write_merged(manifest.parent / "merged/chk1")
    _create_expected_merge_attestation(root, manifest, stage_id="chk1")
    assert stage_recovery_state(manifest, "chk1", root)["state"] == (
        "attested_merge_ready_to_record"
    )

    record_merge_receipt(manifest, "chk1", root)
    assert stage_recovery_state(manifest, "chk1", root)["state"] == (
        "merge_receipt_ready_to_seal"
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["stages"]["chk1"]["status"] = "sealed"
    _write_json(manifest, payload)
    assert stage_recovery_state(manifest, "chk1", root)["state"] == (
        "sealed_complete"
    )


def test_recovery_state_blocks_arbitrary_partial_directories(
    receipt_repo: tuple[Path, Path],
) -> None:
    root, manifest = receipt_repo
    _remove_merged(manifest)
    shutil.rmtree(manifest.parent / "adapters/chk1")
    partial = manifest.parent / "adapters/chk1/interrupted-write"
    partial.mkdir(parents=True)

    with pytest.raises(ExecutionReceiptError, match="Unexpected partial directory"):
        stage_recovery_state(manifest, "chk1", root)
