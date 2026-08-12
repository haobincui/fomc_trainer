from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from trl import TrlParser

from open_r1.configs import (
    GRPOConfig,
    GRPOScriptArguments,
    LoraArguments,
    ModelConfig,
    SFTConfig,
    SFTScriptArguments,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SFT_CONFIG = REPO_ROOT / (
    "configs/retrain_v2/chk4_decision_sft_from_chk1_cp200_core_v3_20260810.yaml"
)
GRPO_CONFIG = REPO_ROOT / (
    "configs/retrain_v2/chk4_decision_grpo_from_sft_core_v3_20260810.yaml"
)
PARENT_MERGE_CONFIG = REPO_ROOT / (
    "configs/retrain_v2/chk1_cp200_merge_for_chk4_core_v3_20260810.yaml"
)
WARM_FIX_SFT_CONFIG = REPO_ROOT / (
    "configs/retrain_v2/chk4_decision_sft_from_chk1_cp200_warm_fix_v1_20260810.yaml"
)
WARM_FIX_GRPO_CONFIG = REPO_ROOT / (
    "configs/retrain_v2/chk4_decision_grpo_from_sft_warm_fix_v1_20260810.yaml"
)
WARM_FIX_PARENT_CONFIG = REPO_ROOT / (
    "configs/retrain_v2/chk1_cp200_parent_reuse_for_chk4_warm_fix_v1_20260810.yaml"
)
WARM_FIX_LR1E5_SFT_CONFIG = REPO_ROOT / (
    "configs/retrain_v2/"
    "chk4_decision_sft_from_chk1_cp200_warm_fix_lr1e5_v2_20260810.yaml"
)
WARM_FIX_LR1E5_GRPO_CONFIG = REPO_ROOT / (
    "configs/retrain_v2/chk4_decision_grpo_from_sft_warm_fix_lr1e5_v2_20260810.yaml"
)
WARM_FIX_LR1E5_PARENT_CONFIG = REPO_ROOT / (
    "configs/retrain_v2/"
    "chk1_cp200_parent_reuse_for_chk4_warm_fix_lr1e5_v2_20260810.yaml"
)
WARM_FIX_STEPS24_SFT_CONFIG = REPO_ROOT / (
    "configs/retrain_v2/"
    "chk4_decision_sft_from_chk1_cp200_warm_fix_lr1e5_steps24_v3_20260811.yaml"
)
WARM_FIX_STEPS24_GRPO_CONFIG = REPO_ROOT / (
    "configs/retrain_v2/"
    "chk4_decision_grpo_from_sft_warm_fix_lr1e5_steps24_v3_20260811.yaml"
)
WARM_FIX_STEPS24_PARENT_CONFIG = REPO_ROOT / (
    "configs/retrain_v2/"
    "chk1_cp200_parent_reuse_for_chk4_warm_fix_lr1e5_steps24_v3_20260811.yaml"
)
RELEASE_MANIFEST = REPO_ROOT / (
    "dataset/processed/retrain_v2/"
    "chk4_decision_warmstart_grpo_core_v3_20260810/release_manifest.json"
)
EXPECTED_MANIFEST_SHA256 = (
    "8b05dce09bcb0a0b27ee240da1e5d730be93921ce19e650a3b3e8b2689adb893"
)
EXPECTED_SYSTEM_PROMPT_SHA256 = (
    "426b64532dfb955a3941db2cc2bcfc9a785c0540ffba9ee94afabc189fe4ce18"
)


def _yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_sft_and_grpo_configs_parse_with_current_trl() -> None:
    sft_script, sft_training, sft_model, sft_peft = TrlParser(
        (SFTScriptArguments, SFTConfig, ModelConfig, LoraArguments)
    ).parse_args_and_config(args=["--config", str(SFT_CONFIG)])
    grpo_script, grpo_training, grpo_model, grpo_peft = TrlParser(
        (GRPOScriptArguments, GRPOConfig, ModelConfig, LoraArguments)
    ).parse_args_and_config(args=["--config", str(GRPO_CONFIG)])

    assert sft_script.dataset_chk4_role == "decision_sft"
    assert sft_training.max_length == 3072
    assert sft_training.completion_only_loss is True
    assert sft_model.load_in_4bit is True
    assert sft_peft.peft_r == 32
    assert sft_peft.peft_bias == "none"

    assert grpo_script.dataset_chk4_role == "decision_grpo"
    assert grpo_script.reward_funcs == ["decision_dense_v2"]
    assert grpo_training.max_prompt_length == 2560
    assert grpo_training.max_completion_length == 512
    assert grpo_training.num_generations == 4
    assert grpo_training.generation_batch_size == 8
    assert grpo_model.load_in_4bit is True
    assert grpo_peft.peft_r == 32
    assert grpo_peft.peft_bias == "none"


def test_configs_bind_release_system_prompt_and_parent_chain() -> None:
    sft = _yaml(SFT_CONFIG)
    grpo = _yaml(GRPO_CONFIG)
    parent = _yaml(PARENT_MERGE_CONFIG)
    release = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
    contract_path = REPO_ROOT / (
        "dataset/processed/retrain_v2/"
        "chk4_decision_warmstart_grpo_core_v3_20260810/"
        "contracts/decision_input_contract.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert hashlib.sha256(RELEASE_MANIFEST.read_bytes()).hexdigest() == (
        EXPECTED_MANIFEST_SHA256
    )
    assert sft["dataset_chk4_release_manifest_sha256"] == EXPECTED_MANIFEST_SHA256
    assert grpo["dataset_chk4_release_manifest_sha256"] == EXPECTED_MANIFEST_SHA256
    assert sft["system_prompt"] == grpo["system_prompt"]
    assert sft["system_prompt"] == contract["student_system_prompt"]
    assert hashlib.sha256(str(sft["system_prompt"]).encode()).hexdigest() == (
        EXPECTED_SYSTEM_PROMPT_SHA256
    )
    assert release["training_roles"]["decision_sft"]["parent_role"] == (
        "selected_chk1_merged"
    )
    assert release["training_roles"]["decision_grpo"]["parent_role"] == (
        "merged_decision_sft_warm_start"
    )

    assert sft["model_name_or_path"] == parent["peft_merged_model_path"]
    assert grpo["model_name_or_path"] == sft["peft_merged_model_path"]
    assert sft["output_dir"] != grpo["output_dir"]
    assert sft["peft_merged_model_path"] != grpo["peft_merged_model_path"]
    assert "selected_cp200_for_chk2" not in str(sft["model_name_or_path"])


def test_gpu1_training_and_checkpoint_contract_is_consistent() -> None:
    for payload in (_yaml(SFT_CONFIG), _yaml(GRPO_CONFIG)):
        assert payload["per_device_train_batch_size"] == 1
        assert payload["gradient_accumulation_steps"] == 8
        assert payload["num_train_epochs"] == 3
        assert payload["learning_rate"] == 1.0e-6
        assert payload["save_strategy"] == "steps"
        assert payload["save_steps"] == 1
        assert payload["save_total_limit"] is None
        assert payload["checkpoint_keep_last"] == 3
        assert payload["checkpoint_keep_every_n_steps"] == 10
        assert payload["peft_bias"] == "none"
        assert payload["overwrite_output_dir"] is False
        assert payload["resume_from_checkpoint"] is None
        assert payload["load_in_4bit"] is True
        assert payload["bnb_4bit_quant_type"] == "nf4"
        assert payload["use_bnb_nested_quant"] is True
        assert payload["dtype"] == "bfloat16"

    sft = _yaml(SFT_CONFIG)
    grpo = _yaml(GRPO_CONFIG)
    assert sft["do_eval"] is True
    assert sft["eval_steps"] == 10
    assert grpo["do_eval"] is False
    assert grpo["eval_strategy"] == "no"
    assert grpo["reward_funcs"] == ["decision_dense_v2"]
    assert grpo["use_vllm"] is False


def test_warm_fix_profile_parses_and_is_isolated_from_core_v3_outputs() -> None:
    sft_script, sft_training, sft_model, sft_peft = TrlParser(
        (SFTScriptArguments, SFTConfig, ModelConfig, LoraArguments)
    ).parse_args_and_config(args=["--config", str(WARM_FIX_SFT_CONFIG)])
    grpo_script, grpo_training, grpo_model, grpo_peft = TrlParser(
        (GRPOScriptArguments, GRPOConfig, ModelConfig, LoraArguments)
    ).parse_args_and_config(args=["--config", str(WARM_FIX_GRPO_CONFIG)])

    assert sft_script.dataset_chk4_role == "decision_sft"
    assert sft_training.learning_rate == 5.0e-6
    assert sft_training.max_steps == 12
    assert sft_training.num_train_epochs == 3
    assert sft_training.warmup_steps == 2
    assert sft_training.eval_steps == 6
    assert str(sft_training.lr_scheduler_type) == "SchedulerType.COSINE"
    assert sft_training.gradient_accumulation_steps == 8
    assert sft_model.load_in_4bit is True
    assert sft_peft.peft_r == 32

    assert grpo_script.dataset_chk4_role == "decision_grpo"
    assert grpo_script.reward_funcs == ["decision_dense_v2"]
    assert grpo_training.max_completion_length == 1024
    assert grpo_training.gradient_accumulation_steps == 8
    assert grpo_model.load_in_4bit is True
    assert grpo_peft.peft_r == 32

    parent = _yaml(WARM_FIX_PARENT_CONFIG)
    sft = _yaml(WARM_FIX_SFT_CONFIG)
    grpo = _yaml(WARM_FIX_GRPO_CONFIG)
    old_sft = _yaml(SFT_CONFIG)
    old_grpo = _yaml(GRPO_CONFIG)
    assert parent["peft_merged_model_path"] == old_sft["model_name_or_path"]
    assert sft["model_name_or_path"] == parent["peft_merged_model_path"]
    assert grpo["model_name_or_path"] == sft["peft_merged_model_path"]
    assert sft["output_dir"] != old_sft["output_dir"]
    assert sft["peft_merged_model_path"] != old_sft["peft_merged_model_path"]
    assert grpo["output_dir"] != old_grpo["output_dir"]
    assert "warm_fix_v1" in str(sft["output_dir"])
    assert "warm_fix_v1" in str(grpo["output_dir"])


def test_lr1e5_warm_fix_profile_parses_and_has_fresh_outputs() -> None:
    sft_script, sft_training, sft_model, sft_peft = TrlParser(
        (SFTScriptArguments, SFTConfig, ModelConfig, LoraArguments)
    ).parse_args_and_config(args=["--config", str(WARM_FIX_LR1E5_SFT_CONFIG)])
    grpo_script, grpo_training, grpo_model, grpo_peft = TrlParser(
        (GRPOScriptArguments, GRPOConfig, ModelConfig, LoraArguments)
    ).parse_args_and_config(args=["--config", str(WARM_FIX_LR1E5_GRPO_CONFIG)])

    assert sft_script.dataset_chk4_role == "decision_sft"
    assert sft_training.learning_rate == 1.0e-5
    assert sft_training.max_steps == 8
    assert sft_training.num_train_epochs == 3
    assert sft_training.warmup_steps == 2
    assert sft_training.eval_steps == 4
    assert str(sft_training.lr_scheduler_type) == "SchedulerType.COSINE"
    assert sft_training.gradient_accumulation_steps == 8
    assert sft_model.load_in_4bit is True
    assert sft_peft.peft_r == 32

    assert grpo_script.dataset_chk4_role == "decision_grpo"
    assert grpo_script.reward_funcs == ["decision_dense_v2"]
    assert grpo_training.max_completion_length == 1024
    assert grpo_training.gradient_accumulation_steps == 8
    assert grpo_model.load_in_4bit is True
    assert grpo_peft.peft_r == 32

    parent = _yaml(WARM_FIX_LR1E5_PARENT_CONFIG)
    sft = _yaml(WARM_FIX_LR1E5_SFT_CONFIG)
    grpo = _yaml(WARM_FIX_LR1E5_GRPO_CONFIG)
    old_sft = _yaml(SFT_CONFIG)
    prior_warm_sft = _yaml(WARM_FIX_SFT_CONFIG)
    prior_warm_grpo = _yaml(WARM_FIX_GRPO_CONFIG)
    assert parent["peft_merged_model_path"] == old_sft["model_name_or_path"]
    assert sft["model_name_or_path"] == parent["peft_merged_model_path"]
    assert grpo["model_name_or_path"] == sft["peft_merged_model_path"]
    assert sft["output_dir"] != prior_warm_sft["output_dir"]
    assert sft["peft_merged_model_path"] != prior_warm_sft["peft_merged_model_path"]
    assert grpo["output_dir"] != prior_warm_grpo["output_dir"]
    assert "warm_fix_lr1e5_v2" in str(sft["output_dir"])
    assert "warm_fix_lr1e5_v2" in str(grpo["output_dir"])


def test_steps24_v3_profile_parses_and_is_fully_isolated() -> None:
    sft_script, sft_training, sft_model, sft_peft = TrlParser(
        (SFTScriptArguments, SFTConfig, ModelConfig, LoraArguments)
    ).parse_args_and_config(args=["--config", str(WARM_FIX_STEPS24_SFT_CONFIG)])
    grpo_script, grpo_training, grpo_model, grpo_peft = TrlParser(
        (GRPOScriptArguments, GRPOConfig, ModelConfig, LoraArguments)
    ).parse_args_and_config(args=["--config", str(WARM_FIX_STEPS24_GRPO_CONFIG)])

    assert sft_script.dataset_chk4_role == "decision_sft"
    assert sft_training.learning_rate == 1.0e-5
    assert sft_training.max_steps == 24
    assert sft_training.num_train_epochs == 3
    assert sft_training.warmup_steps == 2
    assert sft_training.eval_steps == 4
    assert str(sft_training.lr_scheduler_type) == "SchedulerType.COSINE"
    assert sft_training.gradient_accumulation_steps == 8
    assert sft_training.save_steps == 1
    assert sft_training.checkpoint_keep_last == 3
    assert sft_training.checkpoint_keep_every_n_steps == 10
    assert sft_model.load_in_4bit is True
    assert sft_peft.peft_r == 32

    assert grpo_script.dataset_chk4_role == "decision_grpo"
    assert grpo_script.reward_funcs == ["decision_dense_v3"]
    assert grpo_training.max_completion_length == 1024
    assert grpo_training.gradient_accumulation_steps == 8
    assert grpo_model.load_in_4bit is True
    assert grpo_peft.peft_r == 32

    parent = _yaml(WARM_FIX_STEPS24_PARENT_CONFIG)
    sft = _yaml(WARM_FIX_STEPS24_SFT_CONFIG)
    grpo = _yaml(WARM_FIX_STEPS24_GRPO_CONFIG)
    core_sft = _yaml(SFT_CONFIG)
    prior_sft = _yaml(WARM_FIX_LR1E5_SFT_CONFIG)
    prior_grpo = _yaml(WARM_FIX_LR1E5_GRPO_CONFIG)
    assert parent["peft_merged_model_path"] == core_sft["model_name_or_path"]
    assert sft["model_name_or_path"] == parent["peft_merged_model_path"]
    assert grpo["model_name_or_path"] == sft["peft_merged_model_path"]
    assert sft["output_dir"] != prior_sft["output_dir"]
    assert sft["peft_merged_model_path"] != prior_sft["peft_merged_model_path"]
    assert grpo["output_dir"] != prior_grpo["output_dir"]
    assert "warm_fix_lr1e5_steps24_v3" in str(sft["output_dir"])
    assert "warm_fix_lr1e5_steps24_v3" in str(grpo["output_dir"])
