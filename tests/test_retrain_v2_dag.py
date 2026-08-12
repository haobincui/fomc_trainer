from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from trl import TrlParser

from jobs.retrain_v2 import dag as dag_module
from jobs.retrain_v2 import execution_receipt as receipt_module
from jobs.retrain_v2.dag import (
    REQUIRED_BASE_AUDITS,
    REQUIRED_DERIVED_AUDITS,
    DagValidationError,
    bind_derived_data,
    build_plan,
    init_run,
    load_dag,
    main as dag_main,
    seal_stage,
    stage_config_path,
    validate_dag,
    validate_base_release,
    validate_derived_release,
    verify_judge_artifact,
    verify_parent,
)
from jobs.retrain_v2.execution_contract import (
    ExecutionContractError,
    canonical_stage_topology,
)
from jobs.retrain_v2.execution_receipt import ExecutionReceiptError
from jobs.retrain_v2.chk1.prompt_projection import (
    GRPO_PROMPT_TOKEN_LIMIT,
    projection_contract_sha256,
)
from jobs.retrain_v2.tokenizer_provenance import (
    TOKENIZER_LOADER_CONTRACT,
    snapshot_tokenizer_bundle,
)
from jobs.retrain_v2.merge_adapter import merge_from_config
from open_r1.configs import (
    GRPOConfig,
    GRPOScriptArguments,
    LoraArguments,
    ModelConfig,
    SFTConfig,
    SFTScriptArguments,
)
from open_r1.provenance import (
    fingerprint_artifact_path,
    sha256_file,
    sha256_text,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_CONTRACT_SHA = "1" * 64
FAKE_TRAINING_RECEIPT_SHA = "2" * 64
FAKE_MERGE_RECEIPT_SHA = "3" * 64


def _fake_execution_contract(created_at_utc: str) -> dict:
    return {
        "schema_version": 1,
        "created_at_utc": created_at_utc,
        "source_bundle": {
            "algorithm": "sha256",
            "payload": {"files": []},
            "sha256": "4" * 64,
        },
        "environment": {
            "algorithm": "sha256",
            "payload": {"distributions": []},
            "sha256": "5" * 64,
        },
        "stage_topology": canonical_stage_topology(),
        "contract_sha256": FAKE_CONTRACT_SHA,
    }


@pytest.fixture(autouse=True)
def _stub_external_execution_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep DAG fixtures tiny; the two evidence modules have independent tests."""

    monkeypatch.setattr(
        dag_module,
        "build_execution_contract",
        lambda _root, *, created_at_utc=None: _fake_execution_contract(
            created_at_utc or "2026-08-03T00:00:00Z"
        ),
    )
    monkeypatch.setattr(
        dag_module,
        "verify_execution_contract",
        lambda record, _root: {
            "status": "verified",
            "contract_sha256": record["contract_sha256"],
        },
    )

    def fake_verify_receipts(run_manifest, stage_id, repo_root):
        manifest = _read_manifest(Path(run_manifest))
        config = yaml.safe_load(
            Path(manifest["stages"][stage_id]["resolved_config"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        root = Path(repo_root)
        adapter = Path(config["output_dir"])
        merged = Path(config["peft_merged_model_path"])
        if not adapter.is_absolute():
            adapter = root / adapter
        if not merged.is_absolute():
            merged = root / merged
        return {
            "status": "verified",
            "training_receipt_sha256": FAKE_TRAINING_RECEIPT_SHA,
            "merge_receipt_sha256": FAKE_MERGE_RECEIPT_SHA,
            "adapter_sha256": fingerprint_artifact_path(adapter)["sha256"],
            "merged_sha256": fingerprint_artifact_path(merged)["sha256"],
        }

    monkeypatch.setattr(receipt_module, "verify_stage_receipts", fake_verify_receipts)


@pytest.mark.parametrize(
    ("filename", "argument_types"),
    [
        (
            "chk1_analysis_sft.yaml",
            (SFTScriptArguments, SFTConfig, ModelConfig, LoraArguments),
        ),
        (
            "chk2_analysis_grpo.yaml",
            (GRPOScriptArguments, GRPOConfig, ModelConfig, LoraArguments),
        ),
        (
            "chk2_analysis_grpo_v3.yaml",
            (GRPOScriptArguments, GRPOConfig, ModelConfig, LoraArguments),
        ),
        (
            "chk3_minutes_sft.yaml",
            (SFTScriptArguments, SFTConfig, ModelConfig, LoraArguments),
        ),
        (
            "chk4_decision_grpo.yaml",
            (GRPOScriptArguments, GRPOConfig, ModelConfig, LoraArguments),
        ),
    ],
)
def test_training_config_parses_with_current_trl(
    filename: str, argument_types: tuple[type, ...]
) -> None:
    config = REPO_ROOT / "configs/retrain_v2" / filename
    parsed = TrlParser(argument_types).parse_args_and_config(
        args=["--config", str(config)]
    )
    assert parsed[2].load_in_4bit is True
    expected_attention = (
        "flex_attention" if filename == "chk1_analysis_sft.yaml" else "sdpa"
    )
    assert parsed[2].attn_implementation == expected_attention


def test_static_dag_has_checkpoint_branch_and_derived_data_barrier() -> None:
    dag = load_dag(REPO_ROOT / "configs/retrain_v2/dag.yaml")
    result = validate_dag(dag, repo_root=REPO_ROOT, check_rewards=False)

    assert dag["schema_version"] == 2
    assert result["edges"] == [
        ["chk0", "chk1"],
        ["chk1", "chk2"],
        ["chk2", "chk3"],
        ["chk2", "chk4"],
    ]
    assert result["data_barrier"] == ["chk2", "bind_derived_data", ["chk3", "chk4"]]
    assert dag["stages"]["chk4"]["parent"] == "chk2"
    assert dag["stages"]["chk1"]["launcher"] == "ddp_2xa30"
    assert dag["stages"]["chk3"]["dataset_release_role"] == "derived_chk2"
    assert dag["judge"] == {
        "artifact": "models/Qwen3.5-9B",
        "tokenizer_path": "models/Qwen3.5-9B",
        "served_model_name": "Qwen3.5-9B",
        "url": "http://127.0.0.1:8000/v1/chat/completions",
        "timeout": 180,
        "max_retries": 3,
        "backoff_seconds": 1.0,
        "max_model_len": 8192,
        "max_completion_tokens": 2048,
        "candidate_reserve_tokens": 1536,
        "boundary_margin_tokens": 32,
        "used_by": "chk2",
    }
    assert "decision_sft" not in json.dumps(dag)
    assert dag["hardware"]["gpu_count"] == 2
    assert dag["hardware"]["gpu_memory_gib"] == 24
    assert dag["token_budgets"]["chk1"] == {
        "prompt": 3072,
        "completion": None,
        "total": 7168,
        "overflow_policy": "error",
    }
    assert dag["token_budgets"]["chk3"] == {
        "prompt": 3072,
        "completion": 1024,
        "total": 4096,
        "overflow_policy": "error",
    }


def test_reward_v3_dag_pins_dense_reward_and_6656_policy_budget() -> None:
    dag = load_dag(REPO_ROOT / "configs/retrain_v2/dag_reward_v3.yaml")
    result = validate_dag(dag, repo_root=REPO_ROOT, check_rewards=True)

    assert result["missing_rewards"] == []
    assert dag["stages"]["chk2"]["required_rewards"] == [
        "grounded_analysis_v3"
    ]
    assert dag["stages"]["chk2"]["config"].endswith(
        "chk2_analysis_grpo_v3.yaml"
    )
    assert dag["token_budgets"]["chk2"] == {
        "prompt": 2560,
        "completion": 4096,
        "overflow_policy": "error",
    }
    assert dag["judge"]["max_model_len"] == 12288
    assert dag["judge"]["candidate_reserve_tokens"] == 4352


def test_chk3_full_completion_dag_allows_total_only_sft_admission() -> None:
    dag = load_dag(REPO_ROOT / "configs/retrain_v2/dag_chk3_full_completion.yaml")
    validate_dag(dag, repo_root=REPO_ROOT, check_rewards=False)

    assert dag["token_budgets"]["chk3"] == {
        "prompt": 3072,
        "completion": None,
        "total": 4096,
        "overflow_policy": "error",
    }


def test_liger_is_enabled_only_for_sft_stages() -> None:
    expected = {
        "chk1_analysis_sft.yaml": (True, 7168),
        "chk2_analysis_grpo.yaml": (False, None),
        "chk2_analysis_grpo_v3.yaml": (False, None),
        "chk3_minutes_sft.yaml": (True, 4096),
        "chk4_decision_grpo.yaml": (False, None),
    }
    for filename, (enabled, max_length) in expected.items():
        config = yaml.safe_load(
            (REPO_ROOT / "configs/retrain_v2" / filename).read_text(encoding="utf-8")
        )
        assert config["use_liger_kernel"] is enabled
        if enabled:
            assert config["max_length"] == max_length


def test_plan_resolves_upstream_but_leaves_downstream_unbound() -> None:
    plan = build_plan(
        load_dag(REPO_ROOT / "configs/retrain_v2/dag.yaml"),
        repo_root=REPO_ROOT,
        run_id="run_20260803",
        dataset_release_id="release_20260803",
    )

    assert plan["base_release_id"] == "release_20260803"
    assert plan["dataset_release_id"] == "release_20260803"
    assert plan["stages"]["chk2"]["dataset"].endswith("release_20260803/analysis_grpo")
    for stage_id in ("chk3", "chk4"):
        assert plan["stages"][stage_id]["status"] == "awaiting_derived_data"
        assert plan["stages"][stage_id]["training"] is None
        assert "__DERIVED_RELEASE_ID__" in plan["stages"][stage_id]["dataset_template"]


def test_wrong_chk4_parent_is_rejected() -> None:
    dag = load_dag(REPO_ROOT / "configs/retrain_v2/dag.yaml")
    dag["stages"]["chk4"]["parent"] = "chk3"
    with pytest.raises(DagValidationError, match="Wrong parent for chk4"):
        validate_dag(dag, repo_root=REPO_ROOT, check_rewards=False)


@pytest.mark.parametrize("stage_id", ["chk2", "chk4"])
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_prompt_length", 2559, "prompt length"),
        ("chat_template_max_length", 2559, "chat-template length"),
    ],
)
def test_grpo_prompt_budget_is_explicit_and_non_truncating(
    tmp_path: Path, stage_id: str, field: str, value: int, message: str
) -> None:
    _build_tiny_repo(tmp_path)
    dag = load_dag(tmp_path / "configs/retrain_v2/dag.yaml")
    config_path = tmp_path / str(dag["stages"][stage_id]["config"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if field == "chat_template_max_length":
        config["chat_template_kwargs"]["max_length"] = value
    else:
        config[field] = value
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(DagValidationError, match=message):
        validate_dag(dag, repo_root=tmp_path, check_rewards=False)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tokenizer_path", "models/other"),
        ("max_model_len", 8193),
        ("max_completion_tokens", 2047),
        ("candidate_reserve_tokens", 1535),
        ("boundary_margin_tokens", 31),
    ],
)
def test_static_judge_context_contract_rejects_drift(field: str, value: object) -> None:
    dag = load_dag(REPO_ROOT / "configs/retrain_v2/dag.yaml")
    dag["judge"][field] = value
    with pytest.raises(DagValidationError, match="must pin the local Qwen"):
        validate_dag(dag, repo_root=REPO_ROOT, check_rewards=False)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("url", "http://127.0.0.1:18080/v1/chat/completions", "fixed local judge URL"),
        ("model", "wrong", "served-model alias"),
        ("timeout", 60, "fixed judge timeout"),
        ("max_retries", 4, "fixed judge retry count"),
        ("backoff_seconds", 2.0, "fixed judge retry backoff"),
    ],
)
def test_chk2_judge_runtime_contract_is_canonical(
    tmp_path: Path, key, value, message
) -> None:
    _build_tiny_repo(tmp_path)
    dag = load_dag(tmp_path / "configs/retrain_v2/dag.yaml")
    config_path = tmp_path / "configs/retrain_v2/chk2_analysis_grpo.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_key = "judge_model" if key == "model" else f"judge_{key}"
    config[config_key] = value
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(DagValidationError, match=message):
        validate_dag(dag, repo_root=tmp_path, check_rewards=False)


def _write_dataset(path: Path) -> None:
    path.mkdir(parents=True)
    for split in ("train", "eval", "test"):
        (path / f"{split}.jsonl").write_text(
            json.dumps({"prompt": f"prompt-{split}", "response": "response"}) + "\n",
            encoding="utf-8",
        )


def _write_base_release_manifest(release: Path, *, release_id: str) -> Path:
    provenance_dir = release / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    provenance_files = {
        "code": provenance_dir / "builder.py",
        "prompt": provenance_dir / "prompt_contract.json",
        "generation_config": provenance_dir / "generation_config.json",
    }
    tokenizer_source = (
        release.parents[3] / "models/DeepSeek-R1-Distill-Llama-8B"
    )
    tokenizer_bundle = provenance_dir / "tokenizer_bundle"
    tokenizer_bundle.mkdir(exist_ok=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(tokenizer_source / name, tokenizer_bundle / name)
    tokenizer_snapshot = snapshot_tokenizer_bundle(tokenizer_bundle)
    tokenizer_manifest = provenance_dir / "tokenizer_bundle_manifest.json"
    tokenizer_manifest.write_text(
        json.dumps(tokenizer_snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance_files["code"].write_text("# immutable test builder\n")
    config_paths = {
        "analysis_sft": "configs/retrain_v2/chk1_analysis_sft.yaml",
        "analysis_grpo": "configs/retrain_v2/chk2_analysis_grpo.yaml",
    }
    training_system_prompts = {
        dataset_kind: yaml.safe_load(
            (release.parents[3] / config_path).read_text(encoding="utf-8")
        )["system_prompt"]
        for dataset_kind, config_path in config_paths.items()
    }
    prompt_contract = {
        "schema_version": 1,
        "source_prompt_template_sha256": sha256_text("test teacher template"),
        "training_system_prompts": training_system_prompts,
        "training_system_prompt_sha256": {
            dataset_kind: sha256_text(prompt)
            for dataset_kind, prompt in training_system_prompts.items()
        },
        "stage_config_paths": config_paths,
        "chat_rendering_contract": {
            "messages": ["system", "user"],
            "add_generation_prompt": True,
            "grpo_truncation": False,
        },
        "grpo_projection_contract_sha256": projection_contract_sha256(),
        "grpo_projection_max_prompt_tokens": GRPO_PROMPT_TOKEN_LIMIT,
    }
    provenance_files["prompt"].write_text(
        json.dumps(prompt_contract, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    provenance_files["generation_config"].write_text(
        '{"temperature":0,"do_sample":false}\n'
    )

    audits: dict[str, dict] = {}
    for audit_name in REQUIRED_BASE_AUDITS:
        path = release / "audits" / f"{audit_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": "passed"}) + "\n", encoding="utf-8")
        audits[audit_name] = {
            "status": "passed",
            "path": str(path.relative_to(release)),
            "sha256": sha256_file(path),
        }

    datasets: dict[str, dict] = {}
    for dataset_name in ("analysis_sft", "analysis_grpo"):
        dataset = release / dataset_name
        datasets[dataset_name] = {
            "path": dataset_name,
            "artifact_sha256": fingerprint_artifact_path(dataset)["sha256"],
            "split_files": {
                "train": {
                    "path": f"{dataset_name}/train.jsonl",
                    "sha256": sha256_file(dataset / "train.jsonl"),
                },
                "validation": {
                    "path": f"{dataset_name}/eval.jsonl",
                    "sha256": sha256_file(dataset / "eval.jsonl"),
                },
                "test": {
                    "path": f"{dataset_name}/test.jsonl",
                    "sha256": sha256_file(dataset / "test.jsonl"),
                },
            },
        }

    artifact_records = {
        name: {
            "path": str(path.relative_to(release)),
            "sha256": sha256_file(path),
        }
        for name, path in provenance_files.items()
    }
    artifact_records["tokenizer"] = {
        "path": str(tokenizer_bundle.relative_to(release)),
        "sha256": fingerprint_artifact_path(tokenizer_bundle)["sha256"],
    }
    payload = {
        "schema_version": 1,
        "release_type": "analysis_base",
        "release_id": release_id,
        "provenance": {
            "teacher_model": "deepseek-reasoner",
            "teacher_model_version": "fixed-test-version",
            "tokenizer_sha256": artifact_records["tokenizer"]["sha256"],
            "code_sha256": artifact_records["code"]["sha256"],
            "prompt_sha256": artifact_records["prompt"]["sha256"],
            "generation_config_sha256": artifact_records["generation_config"]["sha256"],
            "tokenizer_runtime": {
                "mode": "auto_tokenizer_local_bundle",
                "loader_contract": dict(TOKENIZER_LOADER_CONTRACT),
                "bundle_artifact_sha256": artifact_records["tokenizer"]["sha256"],
                "bundle_payload_sha256": tokenizer_snapshot["payload_sha256"],
            },
            "tokenizer_bundle_manifest": {
                "path": str(tokenizer_manifest.relative_to(release)),
                "sha256": sha256_file(tokenizer_manifest),
            },
            "artifacts": artifact_records,
        },
        "audits": audits,
        "datasets": datasets,
    }
    manifest = release / "base_release_manifest.json"
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _build_tiny_repo(root: Path) -> None:
    shutil.copytree(REPO_ROOT / "configs/retrain_v2", root / "configs/retrain_v2")
    registry = root / "src/open_r1/trainer/rewards/reward_register.py"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        "def get_reward_funcs():\n"
        "    REWARD_FUNCS_REGISTRY = {\n"
        "        'grounded_analysis_v2': object(),\n"
        "        'grounded_analysis_v3': object(),\n"
        "        'decision_dense_v2': object(),\n"
        "    }\n",
        encoding="utf-8",
    )
    model = root / "models/DeepSeek-R1-Distill-Llama-8B"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"model_type":"llama"}\n', encoding="utf-8")
    (model / "weights.bin").write_bytes(b"base-weights")
    (model / "tokenizer.json").write_text(
        '{"name":"test-tokenizer"}\n', encoding="utf-8"
    )
    (model / "tokenizer_config.json").write_text(
        '{"tokenizer_class":"PreTrainedTokenizerFast",'
        '"chat_template":"{{ messages }}"}\n',
        encoding="utf-8",
    )
    judge = root / "models/Qwen3.5-9B"
    judge.mkdir(parents=True)
    (judge / "config.json").write_text('{"model_type":"qwen3_5"}\n', encoding="utf-8")
    (judge / "weights.bin").write_bytes(b"judge-weights")

    # Phase A intentionally contains only datasets required by chk1/chk2.
    base = root / "dataset/processed/retrain_v2/base_test"
    _write_dataset(base / "analysis_sft")
    _write_dataset(base / "analysis_grpo")
    _write_base_release_manifest(base, release_id="base_test")


def _read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rebind_base_prompt_artifact(release: Path) -> None:
    manifest_path = release / "base_release_manifest.json"
    manifest = _read_manifest(manifest_path)
    prompt_record = manifest["provenance"]["artifacts"]["prompt"]
    prompt_sha = sha256_file(release / prompt_record["path"])
    prompt_record["sha256"] = prompt_sha
    manifest["provenance"]["prompt_sha256"] = prompt_sha
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _tree_snapshot(root: Path) -> dict[str, tuple[str, str]]:
    snapshot: dict[str, tuple[str, str]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            snapshot[relative] = ("symlink", os.readlink(path))
        elif path.is_dir():
            snapshot[relative] = ("directory", "")
        else:
            snapshot[relative] = ("file", sha256_file(path))
    return snapshot


def _create_stage_outputs(root: Path, run_id: str, stage_id: str) -> None:
    adapter = root / f"output/training/retrain_v2/{run_id}/adapters/{stage_id}"
    merged = root / f"output/training/retrain_v2/{run_id}/merged/{stage_id}"
    adapter.mkdir(parents=True)
    merged.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(f"adapter-{stage_id}".encode())
    (merged / "model.safetensors").write_bytes(f"merged-{stage_id}".encode())
    (merged / "tokenizer.json").write_text(
        json.dumps({"stage": stage_id}) + "\n", encoding="utf-8"
    )


def _init_tiny_run(root: Path, *, run_id: str = "run_test") -> Path:
    return init_run(
        "configs/retrain_v2/dag.yaml",
        repo_root=root,
        run_id=run_id,
        dataset_release_id="base_test",  # compatibility alias is intentional
        generated_at_utc="2026-08-03T00:00:00Z",
    )


def _seal_through_chk2(root: Path, *, run_id: str = "run_test") -> Path:
    manifest_path = _init_tiny_run(root, run_id=run_id)
    _create_stage_outputs(root, run_id, "chk1")
    seal_stage(
        manifest_path,
        stage_id="chk1",
        repo_root=root,
        generated_at_utc="2026-08-03T01:00:00Z",
    )
    _create_stage_outputs(root, run_id, "chk2")
    seal_stage(
        manifest_path,
        stage_id="chk2",
        repo_root=root,
        generated_at_utc="2026-08-03T02:00:00Z",
    )
    return manifest_path


def _write_derived_release(
    root: Path,
    manifest_path: Path,
    *,
    release_id: str = "derived_test",
    temperature: float = 0,
    do_sample: bool = False,
    source_run_id: str | None = None,
    source_base_sha256: str | None = None,
    producer_chk2_sha256: str | None = None,
    audit_status: str = "passed",
) -> Path:
    run = _read_manifest(manifest_path)
    release = root / f"dataset/processed/retrain_v2/{release_id}"
    _write_dataset(release / "minutes_alignment")
    _write_dataset(release / "decision_grpo")
    analysis = release / "frozen_chk2_analysis.jsonl"
    analysis.write_text('{"meeting_id":"m1","analysis":"a"}\n', encoding="utf-8")
    provenance_dir = release / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    provenance_files = {
        "tokenizer": provenance_dir / "producer_tokenizer.json",
        "prompt": provenance_dir / "analysis_prompt.txt",
        "generation_config": provenance_dir / "generation_config.json",
        "code": provenance_dir / "generate_analysis.py",
    }
    sealed_tokenizer = (
        Path(run["stages"]["chk2"]["artifact"]["path"]) / "tokenizer.json"
    )
    shutil.copy2(sealed_tokenizer, provenance_files["tokenizer"])
    provenance_files["prompt"].write_text(
        "Generate analysis from point-in-time evidence.\n", encoding="utf-8"
    )
    provenance_files["generation_config"].write_text(
        '{"temperature":0,"do_sample":false}\n', encoding="utf-8"
    )
    provenance_files["code"].write_text(
        "# immutable test generation entrypoint\n", encoding="utf-8"
    )
    provenance_records = {
        name: {
            "path": str(path.relative_to(release)),
            "sha256": sha256_file(path),
        }
        for name, path in provenance_files.items()
    }
    audits: dict[str, dict] = {}
    for audit_name in REQUIRED_DERIVED_AUDITS:
        path = release / "audits" / f"{audit_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": audit_status}) + "\n", encoding="utf-8")
        audits[audit_name] = {
            "status": audit_status,
            "path": str(path.relative_to(release)),
            "sha256": sha256_file(path),
        }
    datasets: dict[str, dict] = {}
    for dataset_name in ("minutes_alignment", "decision_grpo"):
        dataset = release / dataset_name
        datasets[dataset_name] = {
            "path": dataset_name,
            "artifact_sha256": fingerprint_artifact_path(dataset)["sha256"],
            "split_files": {
                "train": {
                    "path": f"{dataset_name}/train.jsonl",
                    "sha256": sha256_file(dataset / "train.jsonl"),
                },
                "validation": {
                    "path": f"{dataset_name}/eval.jsonl",
                    "sha256": sha256_file(dataset / "eval.jsonl"),
                },
                "test": {
                    "path": f"{dataset_name}/test.jsonl",
                    "sha256": sha256_file(dataset / "test.jsonl"),
                },
            },
        }
    payload = {
        "schema_version": 1,
        "release_type": "chk2_derived",
        "release_id": release_id,
        "source_run_id": source_run_id or run["run_id"],
        "source_base_release_sha256": source_base_sha256
        or run["data_releases"]["base"]["sha256"],
        "producer": {
            "stage_id": "chk2",
            "chk2_artifact_sha256": producer_chk2_sha256
            or run["stages"]["chk2"]["artifact"]["sha256"],
            "tokenizer_sha256": provenance_records["tokenizer"]["sha256"],
            "tokenizer_artifact": provenance_records["tokenizer"],
        },
        "generation": {
            "temperature": temperature,
            "do_sample": do_sample,
            "prompt_sha256": provenance_records["prompt"]["sha256"],
            "generation_config_sha256": provenance_records["generation_config"][
                "sha256"
            ],
            "code_sha256": provenance_records["code"]["sha256"],
            "artifacts": {
                name: provenance_records[name]
                for name in ("prompt", "generation_config", "code")
            },
        },
        "analysis_generation": {
            "path": str(analysis.relative_to(release)),
            "sha256": sha256_file(analysis),
            "row_count": 1,
        },
        "audits": audits,
        "datasets": datasets,
    }
    release_manifest = release / "derived_release_manifest.json"
    release_manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return release_manifest


def test_init_pins_only_base_analysis_release_and_blocks_downstream(
    tmp_path: Path,
) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _init_tiny_run(tmp_path)
    manifest = _read_manifest(manifest_path)

    assert manifest["schema_version"] == 2
    assert manifest["phase"] == "upstream_ready"
    assert set(manifest["data_releases"]["base"]["datasets"]) == {"chk1", "chk2"}
    assert manifest["data_releases"]["derived_chk2"] is None
    for stage_id in ("chk3", "chk4"):
        assert manifest["stages"][stage_id]["status"] == "awaiting_derived_data"
        assert manifest["stages"][stage_id]["resolved_config"] is None
        assert manifest["stages"][stage_id]["data_binding"] is None
        assert not (
            manifest_path.parent / "resolved_configs" / f"{stage_id}.yaml"
        ).exists()
    assert (
        verify_parent(manifest_path, stage_id="chk1", repo_root=tmp_path)[
            "data_release_role"
        ]
        == "base"
    )


def test_reward_v3_reuses_base_dataset_with_versioned_chk2_prompt(
    tmp_path: Path,
) -> None:
    _build_tiny_repo(tmp_path)

    result = validate_base_release(
        repo_root=tmp_path,
        base_release_id="base_test",
        dag_path="configs/retrain_v2/dag_reward_v3.yaml",
    )

    assert result["status"] == "valid"
    assert result["mode"] == "read_only"


def test_init_manifest_embeds_complete_execution_contract(tmp_path: Path) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _init_tiny_run(tmp_path)

    assert _read_manifest(manifest_path)["execution_contract"] == (
        _fake_execution_contract("2026-08-03T00:00:00Z")
    )


@pytest.mark.parametrize(
    "message",
    [
        "Source bundle is unreadable",
        "Interpreter inventory is unreadable",
        "Interpreter environment drift detected",
    ],
)
def test_init_contract_failure_does_not_publish_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
) -> None:
    _build_tiny_repo(tmp_path)
    monkeypatch.setattr(
        dag_module,
        "build_execution_contract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ExecutionContractError(message)
        ),
    )

    with pytest.raises(DagValidationError, match=message):
        _init_tiny_run(tmp_path, run_id="contract_build_blocked")

    assert not (tmp_path / "output/training/retrain_v2/contract_build_blocked").exists()


@pytest.mark.parametrize(
    "message",
    ["Source bundle drift detected", "Interpreter environment drift detected"],
)
def test_execution_drift_blocks_parent_and_judge_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _init_tiny_run(tmp_path)
    monkeypatch.setattr(
        dag_module,
        "verify_execution_contract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ExecutionContractError(message)
        ),
    )

    with pytest.raises(DagValidationError, match=message):
        verify_parent(manifest_path, stage_id="chk1", repo_root=tmp_path)
    with pytest.raises(DagValidationError, match=message):
        verify_judge_artifact(manifest_path, repo_root=tmp_path)


def test_execution_drift_blocks_derived_validation_and_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _seal_through_chk2(tmp_path)
    _write_derived_release(tmp_path, manifest_path)
    monkeypatch.setattr(
        dag_module,
        "verify_execution_contract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ExecutionContractError("Source bundle drift detected")
        ),
    )

    with pytest.raises(DagValidationError, match="Source bundle drift detected"):
        validate_derived_release(
            manifest_path,
            derived_release_id="derived_test",
            repo_root=tmp_path,
        )
    with pytest.raises(DagValidationError, match="Source bundle drift detected"):
        bind_derived_data(
            manifest_path,
            derived_release_id="derived_test",
            repo_root=tmp_path,
        )


def test_init_pins_complete_base_release_manifest_audits_and_provenance(
    tmp_path: Path,
) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _init_tiny_run(tmp_path)
    base = _read_manifest(manifest_path)["data_releases"]["base"]
    release = tmp_path / "dataset/processed/retrain_v2/base_test"

    assert base["sha256"] == fingerprint_artifact_path(release)["sha256"]
    assert base["artifact"]["sha256"] == base["sha256"]
    assert base["release_manifest"]["sha256"] == sha256_file(
        release / "base_release_manifest.json"
    )
    assert set(base["audits"]) == set(REQUIRED_BASE_AUDITS)
    assert base["provenance"]["teacher_model"] == "deepseek-reasoner"
    assert set(base["datasets"]) == {"chk1", "chk2"}

    judge = _read_manifest(manifest_path)["judge"]
    assert judge["served_model_name"] == "Qwen3.5-9B"
    assert judge["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert judge["timeout"] == 180
    assert judge["max_retries"] == 3
    assert judge["backoff_seconds"] == 1.0
    assert (
        judge["artifact"]["sha256"]
        == fingerprint_artifact_path(tmp_path / "models/Qwen3.5-9B")["sha256"]
    )


def test_validate_base_release_is_complete_and_strictly_read_only(
    tmp_path: Path,
) -> None:
    _build_tiny_repo(tmp_path)
    before = _tree_snapshot(tmp_path)

    result = validate_base_release(
        "configs/retrain_v2/dag.yaml",
        repo_root=tmp_path,
        base_release_id="base_test",
    )

    assert result["status"] == "valid"
    assert result["mode"] == "read_only"
    assert result["base_release"]["release_id"] == "base_test"
    assert result["chk0"]["tokenizer"]["sha256"] == sha256_file(
        tmp_path / "models/DeepSeek-R1-Distill-Llama-8B/tokenizer.json"
    )
    assert result["judge"]["served_model_name"] == "Qwen3.5-9B"
    assert result["judge"]["artifact"] == fingerprint_artifact_path(
        tmp_path / "models/Qwen3.5-9B"
    )
    assert _tree_snapshot(tmp_path) == before
    assert not (tmp_path / "output").exists()


def test_verify_judge_artifact_is_hash_bound_and_data_independent(
    tmp_path: Path,
) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _init_tiny_run(tmp_path)
    result = verify_judge_artifact(manifest_path, repo_root=tmp_path)
    assert result["served_model_name"] == "Qwen3.5-9B"
    assert result["artifact"] == fingerprint_artifact_path(
        tmp_path / "models/Qwen3.5-9B"
    )
    (tmp_path / "models/Qwen3.5-9B/weights.bin").write_bytes(b"tampered")
    with pytest.raises(DagValidationError, match="artifact hash mismatch"):
        verify_judge_artifact(manifest_path, repo_root=tmp_path)


def test_verify_judge_artifact_rejects_symlinked_manifest_path(tmp_path: Path) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _init_tiny_run(tmp_path)
    judge = tmp_path / "models/Qwen3.5-9B"
    backing = tmp_path / "models/Qwen3.5-9B.backing"
    judge.rename(backing)
    judge.symlink_to(backing, target_is_directory=True)
    with pytest.raises(DagValidationError, match="canonical and symlink-free"):
        verify_judge_artifact(manifest_path, repo_root=tmp_path)


def test_verify_judge_artifact_rejects_context_contract_manifest_drift(
    tmp_path: Path,
) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _init_tiny_run(tmp_path)
    manifest = _read_manifest(manifest_path)
    manifest["judge"]["max_model_len"] = 8193
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(DagValidationError, match="max_model_len mismatch"):
        verify_judge_artifact(manifest_path, repo_root=tmp_path)


def test_validate_base_release_fails_on_chk0_tokenizer_or_judge(
    tmp_path: Path,
) -> None:
    _build_tiny_repo(tmp_path)
    tokenizer = tmp_path / "models/DeepSeek-R1-Distill-Llama-8B/tokenizer.json"
    tokenizer.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(DagValidationError, match="does not match the pinned chk0"):
        validate_base_release(
            repo_root=tmp_path,
            base_release_id="base_test",
            dag_path="configs/retrain_v2/dag.yaml",
        )

    tokenizer.write_text('{"name":"test-tokenizer"}\n', encoding="utf-8")
    shutil.rmtree(tmp_path / "models/Qwen3.5-9B")
    with pytest.raises(DagValidationError, match="Judge model does not exist"):
        validate_base_release(
            repo_root=tmp_path,
            base_release_id="base_test",
            dag_path="configs/retrain_v2/dag.yaml",
        )


def test_validate_base_release_cli_returns_json_without_run_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _build_tiny_repo(tmp_path)
    before = _tree_snapshot(tmp_path)

    return_code = dag_main(
        [
            "--repo-root",
            str(tmp_path),
            "--dag",
            "configs/retrain_v2/dag.yaml",
            "validate-base-release",
            "--base-release-id",
            "base_test",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert return_code == 0
    assert payload["status"] == "valid"
    assert payload["mode"] == "read_only"
    assert _tree_snapshot(tmp_path) == before
    assert not (tmp_path / "output").exists()


def test_pipeline_dry_run_calls_plan_then_read_only_base_gate(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "conda.log"
    conda = fake_bin / "conda"
    conda.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$*" >> "${FAKE_CONDA_LOG}"\n',
        encoding="utf-8",
    )
    conda.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["FAKE_CONDA_LOG"] = str(log)

    result = subprocess.run(
        [
            str(REPO_ROOT / "run/retrain_v2/pipeline.sh"),
            "--run-id",
            "dry_run_test",
            "--base-release-id",
            "base_test",
            "--dag",
            "configs/retrain_v2/dag_reward_v3.yaml",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert "--dag configs/retrain_v2/dag_reward_v3.yaml" in calls[0]
    assert "--dag configs/retrain_v2/dag_reward_v3.yaml" in calls[1]
    assert " plan --run-id dry_run_test --base-release-id base_test" in calls[0]
    assert " validate-base-release --base-release-id base_test" in calls[1]


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("schema_version", 2, "Unsupported base release schema"),
        ("release_type", "wrong", "Wrong base release type"),
        ("release_id", "wrong", "Base release ID mismatch"),
    ],
)
def test_base_release_identity_contract_is_fail_closed(
    tmp_path: Path, field: str, value: object, expected_error: str
) -> None:
    _build_tiny_repo(tmp_path)
    release_manifest = (
        tmp_path / "dataset/processed/retrain_v2/base_test/base_release_manifest.json"
    )
    payload = _read_manifest(release_manifest)
    payload[field] = value
    release_manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(DagValidationError, match=expected_error):
        _init_tiny_run(tmp_path)


def test_base_release_requires_manifest_and_all_audits(tmp_path: Path) -> None:
    _build_tiny_repo(tmp_path)
    release = tmp_path / "dataset/processed/retrain_v2/base_test"
    release_manifest = release / "base_release_manifest.json"
    release_manifest.unlink()
    with pytest.raises(
        DagValidationError, match="base release manifest does not exist"
    ):
        _init_tiny_run(tmp_path, run_id="missing_manifest")

    _write_base_release_manifest(release, release_id="base_test")
    payload = _read_manifest(release_manifest)
    payload["audits"].pop("teacher_grounding")
    release_manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(DagValidationError, match="missing required audit"):
        _init_tiny_run(tmp_path, run_id="missing_audit")


def test_base_release_reads_and_requires_passed_audit_artifact(tmp_path: Path) -> None:
    _build_tiny_repo(tmp_path)
    release = tmp_path / "dataset/processed/retrain_v2/base_test"
    release_manifest = release / "base_release_manifest.json"
    payload = _read_manifest(release_manifest)
    audit = release / payload["audits"]["point_in_time"]["path"]
    audit.write_text('{"status":"failed"}\n', encoding="utf-8")
    payload["audits"]["point_in_time"]["sha256"] = sha256_file(audit)
    release_manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(DagValidationError, match="audit artifact did not pass"):
        _init_tiny_run(tmp_path)


def test_base_release_rejects_invalid_provenance_sha_and_path_escape(
    tmp_path: Path,
) -> None:
    _build_tiny_repo(tmp_path)
    release_manifest = (
        tmp_path / "dataset/processed/retrain_v2/base_test/base_release_manifest.json"
    )
    payload = _read_manifest(release_manifest)
    payload["provenance"]["tokenizer_sha256"] = "not-a-sha"
    release_manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(DagValidationError, match="lowercase 64-character"):
        _init_tiny_run(tmp_path, run_id="bad_sha")

    _write_base_release_manifest(release_manifest.parent, release_id="base_test")
    payload = _read_manifest(release_manifest)
    payload["datasets"]["analysis_sft"]["split_files"]["train"]["path"] = (
        "../../outside.jsonl"
    )
    release_manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(DagValidationError, match="escapes its release"):
        _init_tiny_run(tmp_path, run_id="escaped_split")


def test_base_release_requires_all_provenance_artifacts(tmp_path: Path) -> None:
    _build_tiny_repo(tmp_path)
    release_manifest = (
        tmp_path / "dataset/processed/retrain_v2/base_test/base_release_manifest.json"
    )
    payload = _read_manifest(release_manifest)
    payload["provenance"]["artifacts"].pop("prompt")
    release_manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(DagValidationError, match="must bind tokenizer, code"):
        _init_tiny_run(tmp_path, run_id="missing_provenance_artifact")


def test_base_release_rejects_self_consistent_prompt_artifact_config_mismatch(
    tmp_path: Path,
) -> None:
    _build_tiny_repo(tmp_path)
    release = tmp_path / "dataset/processed/retrain_v2/base_test"
    manifest = _read_manifest(release / "base_release_manifest.json")
    prompt_path = release / manifest["provenance"]["artifacts"]["prompt"]["path"]
    contract = _read_manifest(prompt_path)
    forged_prompt = "Forged but internally self-consistent system prompt.\n"
    contract["training_system_prompts"]["analysis_sft"] = forged_prompt
    contract["training_system_prompt_sha256"]["analysis_sft"] = sha256_text(
        forged_prompt
    )
    prompt_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rebind_base_prompt_artifact(release)

    with pytest.raises(
        DagValidationError,
        match="system_prompt/config mismatch for chk1",
    ):
        validate_base_release(
            repo_root=tmp_path,
            base_release_id="base_test",
            dag_path="configs/retrain_v2/dag.yaml",
        )


@pytest.mark.parametrize(
    ("stage_id", "config_filename"),
    [
        ("chk1", "chk1_analysis_sft.yaml"),
        ("chk2", "chk2_analysis_grpo.yaml"),
    ],
)
def test_base_release_rejects_yaml_system_prompt_mismatch(
    tmp_path: Path,
    stage_id: str,
    config_filename: str,
) -> None:
    _build_tiny_repo(tmp_path)
    config_path = tmp_path / "configs/retrain_v2" / config_filename
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["system_prompt"] = f"Tampered system prompt for {stage_id}.\n"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(
        DagValidationError,
        match=rf"system_prompt/config mismatch for {stage_id}",
    ):
        validate_base_release(
            repo_root=tmp_path,
            base_release_id="base_test",
            dag_path="configs/retrain_v2/dag.yaml",
        )


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("config_path", "stage config path mismatch for chk1"),
        ("chat_contract", "chat rendering contract mismatch"),
        ("projection_sha", "GRPO projection contract SHA mismatch"),
        (
            "projection_limit",
            "GRPO projection max prompt tokens mismatch",
        ),
    ],
)
def test_base_release_rejects_rebound_prompt_contract_semantic_drift(
    tmp_path: Path,
    mutation: str,
    expected_error: str,
) -> None:
    _build_tiny_repo(tmp_path)
    release = tmp_path / "dataset/processed/retrain_v2/base_test"
    manifest = _read_manifest(release / "base_release_manifest.json")
    prompt_path = release / manifest["provenance"]["artifacts"]["prompt"]["path"]
    contract = _read_manifest(prompt_path)
    if mutation == "config_path":
        contract["stage_config_paths"]["analysis_sft"] = (
            "configs/retrain_v2/chk2_analysis_grpo.yaml"
        )
    elif mutation == "chat_contract":
        contract["chat_rendering_contract"]["add_generation_prompt"] = False
    elif mutation == "projection_sha":
        contract["grpo_projection_contract_sha256"] = "b" * 64
    else:
        contract["grpo_projection_max_prompt_tokens"] = (
            GRPO_PROMPT_TOKEN_LIMIT - 1
        )
    prompt_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rebind_base_prompt_artifact(release)

    with pytest.raises(DagValidationError, match=expected_error):
        validate_base_release(
            repo_root=tmp_path,
            base_release_id="base_test",
            dag_path="configs/retrain_v2/dag.yaml",
        )


def test_base_release_rejects_symlink_even_when_target_is_a_file(
    tmp_path: Path,
) -> None:
    _build_tiny_repo(tmp_path)
    release = tmp_path / "dataset/processed/retrain_v2/base_test"
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    (release / "unexpected_link").symlink_to(outside)

    with pytest.raises(DagValidationError, match="must not contain symlinks"):
        _init_tiny_run(tmp_path)


def test_verify_rejects_any_post_init_base_release_mutation(tmp_path: Path) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _init_tiny_run(tmp_path)
    release = tmp_path / "dataset/processed/retrain_v2/base_test"
    (release / "unselected-extra-file.txt").write_text("tamper\n", encoding="utf-8")

    with pytest.raises(DagValidationError, match="Base release hash mismatch"):
        verify_parent(manifest_path, stage_id="chk1", repo_root=tmp_path)


def test_verify_rejects_base_dataset_binding_path_escape(tmp_path: Path) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _init_tiny_run(tmp_path)
    manifest = _read_manifest(manifest_path)
    manifest["data_releases"]["base"]["datasets"]["chk1"]["path"] = str(
        tmp_path / "dataset/processed/retrain_v2/base_test/analysis_grpo"
    )
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(DagValidationError, match="base chk1 path mismatch"):
        verify_parent(manifest_path, stage_id="chk1", repo_root=tmp_path)


def test_chk2_verify_rejects_local_judge_artifact_mutation(tmp_path: Path) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _init_tiny_run(tmp_path)
    _create_stage_outputs(tmp_path, "run_test", "chk1")
    seal_stage(manifest_path, stage_id="chk1", repo_root=tmp_path)
    (tmp_path / "models/Qwen3.5-9B/weights.bin").write_bytes(b"tampered")

    with pytest.raises(DagValidationError, match="judge artifact hash mismatch"):
        verify_parent(manifest_path, stage_id="chk2", repo_root=tmp_path)


def test_downstream_verify_and_bind_are_blocked_before_chk2_is_sealed(
    tmp_path: Path,
) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _init_tiny_run(tmp_path)

    with pytest.raises(DagValidationError, match="Parent chk2 is not sealed"):
        verify_parent(manifest_path, stage_id="chk3", repo_root=tmp_path)
    with pytest.raises(DagValidationError, match="not awaiting a derived release"):
        bind_derived_data(
            manifest_path,
            derived_release_id="derived_missing",
            repo_root=tmp_path,
        )


def test_bind_requires_sealed_chk2_and_atomically_unblocks_both_branches(
    tmp_path: Path,
) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _seal_through_chk2(tmp_path)
    manifest = _read_manifest(manifest_path)
    assert manifest["phase"] == "awaiting_derived_release"
    with pytest.raises(DagValidationError, match="awaiting derived data binding"):
        verify_parent(manifest_path, stage_id="chk3", repo_root=tmp_path)

    _write_derived_release(tmp_path, manifest_path)
    validated = validate_derived_release(
        manifest_path, derived_release_id="derived_test", repo_root=tmp_path
    )
    result = bind_derived_data(
        manifest_path,
        derived_release_id="derived_test",
        repo_root=tmp_path,
        generated_at_utc="2026-08-03T03:00:00Z",
    )
    assert result["status"] == "bound"
    assert (
        result["chk2_artifact_sha256"] == validated["producer"]["chk2_artifact_sha256"]
    )
    assert set(validated["audits"]) >= set(REQUIRED_DERIVED_AUDITS)
    assert (
        validated["producer"]["tokenizer_artifact"]["sha256"]
        == validated["producer"]["tokenizer_sha256"]
    )
    assert set(validated["generation"]["artifacts"]) == {
        "prompt",
        "generation_config",
        "code",
    }

    manifest = _read_manifest(manifest_path)
    assert manifest["phase"] == "downstream_ready"
    assert len(manifest["binding_events"]) == 1
    for stage_id in ("chk3", "chk4"):
        record = manifest["stages"][stage_id]
        assert record["status"] == "pending"
        assert record["data_binding"]["release_role"] == "derived_chk2"
        assert "derived_bindings/derived_chk2" in record["resolved_config"]["path"]
        ready = verify_parent(manifest_path, stage_id=stage_id, repo_root=tmp_path)
        assert ready["parent"]["stage"] == "chk2"
        assert ready["data_release_role"] == "derived_chk2"
        assert stage_config_path(manifest_path, stage_id=stage_id).is_file()


@pytest.mark.parametrize(
    ("override", "expected_error"),
    [
        ({"temperature": 0.1}, "temperature must be 0"),
        ({"do_sample": True}, "do_sample=false"),
        ({"source_run_id": "another_run"}, "source_run_id mismatch"),
        ({"source_base_sha256": "a" * 64}, "base hash mismatch"),
        ({"producer_chk2_sha256": "b" * 64}, "chk2 hash mismatch"),
        ({"audit_status": "failed"}, "audit did not pass"),
    ],
)
def test_derived_release_contract_is_fail_closed(
    tmp_path: Path, override: dict, expected_error: str
) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _seal_through_chk2(tmp_path)
    _write_derived_release(tmp_path, manifest_path, **override)

    with pytest.raises(DagValidationError, match=expected_error):
        bind_derived_data(
            manifest_path, derived_release_id="derived_test", repo_root=tmp_path
        )
    manifest = _read_manifest(manifest_path)
    assert manifest["phase"] == "awaiting_derived_release"
    assert manifest["data_releases"]["derived_chk2"] is None
    assert not (manifest_path.parent / "derived_bindings/derived_chk2").exists()


@pytest.mark.parametrize(
    ("section", "sha_key", "expected_error"),
    [
        ("producer", "tokenizer_sha256", "tokenizer artifact does not match"),
        ("generation", "prompt_sha256", "prompt artifact does not match"),
        (
            "generation",
            "generation_config_sha256",
            "generation_config artifact does not match",
        ),
        ("generation", "code_sha256", "code artifact does not match"),
    ],
)
def test_derived_provenance_sha_must_match_release_local_artifact(
    tmp_path: Path, section: str, sha_key: str, expected_error: str
) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _seal_through_chk2(tmp_path)
    release_manifest = _write_derived_release(tmp_path, manifest_path)
    payload = _read_manifest(release_manifest)
    payload[section][sha_key] = "a" * 64
    release_manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(DagValidationError, match=expected_error):
        validate_derived_release(
            manifest_path, derived_release_id="derived_test", repo_root=tmp_path
        )


def test_derived_tokenizer_must_match_sealed_chk2(tmp_path: Path) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _seal_through_chk2(tmp_path)
    release_manifest = _write_derived_release(tmp_path, manifest_path)
    payload = _read_manifest(release_manifest)
    tokenizer_path = (
        release_manifest.parent / payload["producer"]["tokenizer_artifact"]["path"]
    )
    tokenizer_path.write_text('{"wrong":"tokenizer"}\n', encoding="utf-8")
    wrong_sha = sha256_file(tokenizer_path)
    payload["producer"]["tokenizer_sha256"] = wrong_sha
    payload["producer"]["tokenizer_artifact"]["sha256"] = wrong_sha
    release_manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(DagValidationError, match="does not match the sealed chk2"):
        validate_derived_release(
            manifest_path, derived_release_id="derived_test", repo_root=tmp_path
        )


def test_derived_generation_artifacts_are_complete_and_cannot_escape(
    tmp_path: Path,
) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _seal_through_chk2(tmp_path)
    release_manifest = _write_derived_release(tmp_path, manifest_path)
    payload = _read_manifest(release_manifest)
    payload["generation"]["artifacts"].pop("code")
    release_manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(DagValidationError, match="must bind prompt"):
        validate_derived_release(
            manifest_path, derived_release_id="derived_test", repo_root=tmp_path
        )

    shutil.rmtree(release_manifest.parent)
    release_manifest = _write_derived_release(tmp_path, manifest_path)
    payload = _read_manifest(release_manifest)
    payload["generation"]["artifacts"]["prompt"]["path"] = "../../outside.txt"
    release_manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(DagValidationError, match="escapes its release"):
        validate_derived_release(
            manifest_path, derived_release_id="derived_test", repo_root=tmp_path
        )


def test_derived_requires_all_passed_json_audit_artifacts(tmp_path: Path) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _seal_through_chk2(tmp_path)
    release_manifest = _write_derived_release(tmp_path, manifest_path)
    payload = _read_manifest(release_manifest)
    payload["audits"].pop("analysis_lineage")
    release_manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(DagValidationError, match="missing required audit"):
        validate_derived_release(
            manifest_path, derived_release_id="derived_test", repo_root=tmp_path
        )

    shutil.rmtree(release_manifest.parent)
    release_manifest = _write_derived_release(tmp_path, manifest_path)
    payload = _read_manifest(release_manifest)
    audit_record = payload["audits"]["target_integrity"]
    audit_path = release_manifest.parent / audit_record["path"]
    audit_path.write_text('{"status":"failed"}\n', encoding="utf-8")
    audit_record["sha256"] = sha256_file(audit_path)
    release_manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(DagValidationError, match="audit artifact did not pass"):
        validate_derived_release(
            manifest_path, derived_release_id="derived_test", repo_root=tmp_path
        )


def test_derived_analysis_row_count_is_recomputed(tmp_path: Path) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _seal_through_chk2(tmp_path)
    release_manifest = _write_derived_release(tmp_path, manifest_path)
    payload = _read_manifest(release_manifest)
    payload["analysis_generation"]["row_count"] = 2
    release_manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(DagValidationError, match="row_count does not match"):
        validate_derived_release(
            manifest_path, derived_release_id="derived_test", repo_root=tmp_path
        )


def test_derived_release_rejects_root_and_nested_symlinks(tmp_path: Path) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _seal_through_chk2(tmp_path)
    release_manifest = _write_derived_release(tmp_path, manifest_path)
    release = release_manifest.parent
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    (release / "nested_link").symlink_to(outside)
    with pytest.raises(DagValidationError, match="must not contain symlinks"):
        validate_derived_release(
            manifest_path, derived_release_id="derived_test", repo_root=tmp_path
        )

    (release / "nested_link").unlink()
    moved_release = tmp_path / "derived_release_target"
    release.rename(moved_release)
    release.symlink_to(moved_release, target_is_directory=True)
    with pytest.raises(DagValidationError, match="root must not be a symlink"):
        validate_derived_release(
            manifest_path, derived_release_id="derived_test", repo_root=tmp_path
        )


def test_rebind_and_derived_mutation_are_rejected(tmp_path: Path) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _seal_through_chk2(tmp_path)
    _write_derived_release(tmp_path, manifest_path)
    bind_derived_data(
        manifest_path, derived_release_id="derived_test", repo_root=tmp_path
    )

    with pytest.raises(DagValidationError, match="not awaiting a derived release"):
        bind_derived_data(
            manifest_path, derived_release_id="derived_test", repo_root=tmp_path
        )
    target = (
        tmp_path / "dataset/processed/retrain_v2/derived_test/decision_grpo/test.jsonl"
    )
    target.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(DagValidationError, match="hash mismatch"):
        verify_parent(manifest_path, stage_id="chk4", repo_root=tmp_path)


def test_bound_config_bundle_mutation_is_rejected(tmp_path: Path) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _seal_through_chk2(tmp_path)
    _write_derived_release(tmp_path, manifest_path)
    bind_derived_data(
        manifest_path, derived_release_id="derived_test", repo_root=tmp_path
    )
    binding_record = (
        manifest_path.parent / "derived_bindings/derived_chk2/binding_record.json"
    )
    binding_record.write_text('{"tampered":true}\n', encoding="utf-8")

    with pytest.raises(
        DagValidationError, match="Derived config binding artifact hash mismatch"
    ):
        verify_parent(manifest_path, stage_id="chk3", repo_root=tmp_path)


def test_sealing_chk3_does_not_change_chk4_binding_or_parent(tmp_path: Path) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _seal_through_chk2(tmp_path)
    _write_derived_release(tmp_path, manifest_path)
    bind_derived_data(
        manifest_path, derived_release_id="derived_test", repo_root=tmp_path
    )
    before = _read_manifest(manifest_path)["stages"]["chk4"]["data_binding"]

    _create_stage_outputs(tmp_path, "run_test", "chk3")
    seal_stage(manifest_path, stage_id="chk3", repo_root=tmp_path)

    after_manifest = _read_manifest(manifest_path)
    assert after_manifest["phase"] == "downstream_ready"
    assert after_manifest["stages"]["chk4"]["data_binding"] == before
    ready = verify_parent(manifest_path, stage_id="chk4", repo_root=tmp_path)
    assert ready["parent"]["stage"] == "chk2"


def test_missing_receipts_block_sealing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _init_tiny_run(tmp_path)
    _create_stage_outputs(tmp_path, "run_test", "chk1")
    monkeypatch.setattr(
        receipt_module,
        "verify_stage_receipts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ExecutionReceiptError("Receipt is missing")
        ),
    )

    with pytest.raises(DagValidationError, match="Receipt is missing"):
        seal_stage(manifest_path, stage_id="chk1", repo_root=tmp_path)

    assert _read_manifest(manifest_path)["stages"]["chk1"]["status"] == "pending"


def test_seal_records_and_revalidates_contract_and_receipt_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _init_tiny_run(tmp_path)
    _create_stage_outputs(tmp_path, "run_test", "chk1")
    seal_stage(manifest_path, stage_id="chk1", repo_root=tmp_path)
    stage = _read_manifest(manifest_path)["stages"]["chk1"]

    assert stage["execution_contract_sha256"] == FAKE_CONTRACT_SHA
    assert stage["training_receipt_sha256"] == FAKE_TRAINING_RECEIPT_SHA
    assert stage["merge_receipt_sha256"] == FAKE_MERGE_RECEIPT_SHA

    original = receipt_module.verify_stage_receipts

    def changed_training_receipt(*args, **kwargs):
        verified = original(*args, **kwargs)
        verified["training_receipt_sha256"] = "9" * 64
        return verified

    monkeypatch.setattr(
        receipt_module, "verify_stage_receipts", changed_training_receipt
    )
    with pytest.raises(DagValidationError, match="training_receipt_sha256 mismatch"):
        seal_stage(manifest_path, stage_id="chk1", repo_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "execution_contract_sha256",
            "9" * 64,
            "execution contract SHA mismatch",
        ),
        (
            "training_receipt_sha256",
            "9" * 64,
            "training_receipt_sha256 mismatch",
        ),
        (
            "merge_receipt_sha256",
            "9" * 64,
            "merge_receipt_sha256 mismatch",
        ),
    ],
)
def test_verify_parent_revalidates_stored_sealed_ancestor_hashes(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _init_tiny_run(tmp_path)
    _create_stage_outputs(tmp_path, "run_test", "chk1")
    seal_stage(manifest_path, stage_id="chk1", repo_root=tmp_path)
    manifest = _read_manifest(manifest_path)
    manifest["stages"]["chk1"][field] = value
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DagValidationError, match=message):
        verify_parent(manifest_path, stage_id="chk2", repo_root=tmp_path)


def test_verify_parent_revalidates_current_sealed_ancestor_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _init_tiny_run(tmp_path)
    _create_stage_outputs(tmp_path, "run_test", "chk1")
    seal_stage(manifest_path, stage_id="chk1", repo_root=tmp_path)
    monkeypatch.setattr(
        receipt_module,
        "verify_stage_receipts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ExecutionReceiptError("receipt was removed")
        ),
    )

    with pytest.raises(
        DagValidationError,
        match="Sealed chk1 receipt verification failed: receipt was removed",
    ):
        verify_parent(manifest_path, stage_id="chk2", repo_root=tmp_path)


def test_downstream_parent_verifies_each_sealed_ancestor_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _seal_through_chk2(tmp_path)
    _write_derived_release(tmp_path, manifest_path)
    bind_derived_data(
        manifest_path, derived_release_id="derived_test", repo_root=tmp_path
    )
    original = receipt_module.verify_stage_receipts
    verified_stages: list[str] = []

    def recording_verify(*args, **kwargs):
        stage_id = args[1]
        verified_stages.append(stage_id)
        return original(*args, **kwargs)

    monkeypatch.setattr(receipt_module, "verify_stage_receipts", recording_verify)

    verify_parent(manifest_path, stage_id="chk3", repo_root=tmp_path)

    assert verified_stages == ["chk1", "chk2"]


@pytest.mark.parametrize("operation", ["validate", "bind"])
def test_derived_validation_and_binding_require_complete_chk2_receipt_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _seal_through_chk2(tmp_path)
    _write_derived_release(tmp_path, manifest_path)
    original = receipt_module.verify_stage_receipts

    def fail_chk2_receipt(*args, **kwargs):
        if args[1] == "chk2":
            raise ExecutionReceiptError("chk2 receipt was removed")
        return original(*args, **kwargs)

    monkeypatch.setattr(receipt_module, "verify_stage_receipts", fail_chk2_receipt)

    with pytest.raises(
        DagValidationError,
        match="Sealed chk2 receipt verification failed: chk2 receipt was removed",
    ):
        if operation == "validate":
            validate_derived_release(
                manifest_path,
                derived_release_id="derived_test",
                repo_root=tmp_path,
            )
        else:
            bind_derived_data(
                manifest_path,
                derived_release_id="derived_test",
                repo_root=tmp_path,
            )


def test_stage_shell_uses_locked_recovery_attestations_and_atomic_merge() -> None:
    script = (REPO_ROOT / "run/retrain_v2/stage.sh").read_text(encoding="utf-8")
    training = script.index("jobs.train.train_sft")
    record_training = script.index("record-training")
    merge = script.index("jobs.retrain_v2.merge_adapter")
    seal = script.index("seal-stage")

    assert script.index("jobs.retrain_v2.stage_recovery") < training
    assert training < record_training < merge < seal
    assert "record-merge" not in script
    assert "--phase pre" in script
    assert "--phase post" in script
    assert "--verify-pair" in script
    assert 'resource_gate.sh\" --merge' in script


def test_base_or_parent_mutation_is_rejected(tmp_path: Path) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _init_tiny_run(tmp_path)
    base_weight = tmp_path / "models/DeepSeek-R1-Distill-Llama-8B/weights.bin"
    base_weight.write_bytes(b"changed")
    with pytest.raises(DagValidationError, match="Parent chk0 artifact hash mismatch"):
        verify_parent(manifest_path, stage_id="chk1", repo_root=tmp_path)


def test_sealed_stage_cannot_be_resealed_after_artifact_mutation(
    tmp_path: Path,
) -> None:
    _build_tiny_repo(tmp_path)
    manifest_path = _init_tiny_run(tmp_path)
    _create_stage_outputs(tmp_path, "run_test", "chk1")
    seal_stage(manifest_path, stage_id="chk1", repo_root=tmp_path)
    merged = (
        tmp_path / "output/training/retrain_v2/run_test/merged/chk1/model.safetensors"
    )
    merged.write_bytes(b"mutated-after-seal")

    with pytest.raises(DagValidationError, match="Sealed chk1 artifact hash mismatch"):
        seal_stage(manifest_path, stage_id="chk1", repo_root=tmp_path)


def test_init_run_fails_closed_when_v2_rewards_are_not_registered(
    tmp_path: Path,
) -> None:
    _build_tiny_repo(tmp_path)
    registry = tmp_path / "src/open_r1/trainer/rewards/reward_register.py"
    registry.write_text(
        "REWARD_FUNCS_REGISTRY = {'format': object()}\n", encoding="utf-8"
    )
    with pytest.raises(DagValidationError, match="grounded_analysis_v2"):
        _init_tiny_run(tmp_path, run_id="run_blocked")
    assert not (tmp_path / "output/training/retrain_v2/run_blocked").exists()


def test_init_rejects_chk2_judge_context_config_drift(tmp_path: Path) -> None:
    _build_tiny_repo(tmp_path)
    config_path = tmp_path / "configs/retrain_v2/chk2_analysis_grpo.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["judge_max_model_len"] = 8193
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(DagValidationError, match="judge_max_model_len"):
        _init_tiny_run(tmp_path)


def test_merge_uses_sanitized_copy_without_rewriting_source(tmp_path: Path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    merged = tmp_path / "merged"
    base.mkdir()
    adapter.mkdir()
    (base / "config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": str(base),
                "peft_type": "LORA",
                "corda_config": None,
                "trainable_token_indices": None,
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "resolved.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model_name_or_path": str(base),
                "output_dir": str(adapter),
                "peft_merged_model_path": str(merged),
            }
        ),
        encoding="utf-8",
    )
    before = fingerprint_artifact_path(adapter)["sha256"]

    def fake_merge(_base: Path, adapter_copy: Path, output: Path) -> None:
        copied = json.loads((adapter_copy / "adapter_config.json").read_text())
        assert "corda_config" not in copied
        assert "trainable_token_indices" not in copied
        output.mkdir()
        (output / "model.safetensors").write_bytes(b"merged")

    result = merge_from_config(config, merge_executor=fake_merge)
    assert result["status"] == "merged"
    assert result["source_adapter_unchanged"] is True
    assert fingerprint_artifact_path(adapter)["sha256"] == before
    assert (merged / "model.safetensors").read_bytes() == b"merged"


def test_failed_merge_never_deletes_a_concurrent_destination(tmp_path: Path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    destination = tmp_path / "merged"
    base.mkdir()
    adapter.mkdir()
    (base / "config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    config = tmp_path / "resolved.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model_name_or_path": str(base),
                "output_dir": str(adapter),
                "peft_merged_model_path": str(destination),
            }
        ),
        encoding="utf-8",
    )

    def racing_merge(_base: Path, _adapter_copy: Path, _output: Path) -> None:
        destination.mkdir()
        (destination / "owned-by-another-process").write_text(
            "preserve me\n", encoding="utf-8"
        )
        raise RuntimeError("simulated merge failure")

    with pytest.raises(RuntimeError, match="simulated merge failure"):
        merge_from_config(config, merge_executor=racing_merge)
    assert (destination / "owned-by-another-process").read_text(
        encoding="utf-8"
    ) == "preserve me\n"
