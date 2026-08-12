from __future__ import annotations

from pathlib import Path

import yaml
from trl import TrlParser

from open_r1.configs import GRPOConfig, GRPOScriptArguments, LoraArguments, ModelConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO_ROOT
    / "configs/retrain_v2/chk2_analysis_grpo_cp200_deepseek_high_totalsl_v1_20260810.yaml"
)
LAUNCHER = (
    REPO_ROOT / "run/retrain_v2/start_chk2_cp200_deepseek_candidate.sh"
)


def test_candidate_config_parses_with_current_trl() -> None:
    script, training, model, peft = TrlParser(
        (GRPOScriptArguments, GRPOConfig, ModelConfig, LoraArguments)
    ).parse_args_and_config(args=["--config", str(CONFIG)])

    assert script.reward_funcs == ["grounded_analysis_v3_deepseek_high"]
    assert script.judge_model == "deepseek-v4-flash"
    assert script.judge_api_key_env == "DEEPSEEK_API_KEY"
    assert training.max_completion_length == 4096
    assert training.generation_batch_size == 4
    assert model.load_in_4bit is True
    assert peft.peft_r == 32


def test_candidate_contract_is_single_gpu_fresh_and_checkpoint_retained() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert payload["model_name_or_path"].endswith(
        "chk1_clean_v2_lr1e6_selected_cp200_for_chk2_20260810/merged/chk1"
    )
    assert payload["dataset_name"].endswith(
        "analysis_grpo_full_v7_totalsl_billions_v1_20260810"
    )
    assert payload["num_generations"] == 4
    assert payload["generation_batch_size"] == 4
    assert payload["gradient_accumulation_steps"] == 8
    assert payload["num_train_epochs"] == 1
    assert payload["judge_url"] == "https://api.deepseek.com"
    assert payload["judge_max_completion_tokens"] == 16384
    assert payload["judge_timeout"] == 420
    assert payload["judge_max_retries"] == 2
    assert payload["judge_backoff_seconds"] == 2.0
    assert payload["overwrite_output_dir"] is False
    assert payload["save_steps"] == 1
    assert payload["save_total_limit"] is None
    assert payload["checkpoint_keep_last"] == 3
    assert payload["checkpoint_keep_every_n_steps"] == 10


def test_launcher_uses_isolated_tmux_path_without_local_judge_or_canonical_stage() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "start_chk2_deepseek_candidate" in text
    assert "conda run --no-capture-output -n fomc_trainer" in text
    assert "start_chk2_background.sh" not in text
    assert "stage.sh" not in text
    assert "judge.sh" not in text
