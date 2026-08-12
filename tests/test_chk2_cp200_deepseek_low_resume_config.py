from __future__ import annotations

import json
from pathlib import Path

import yaml
from trl import TrlParser

from jobs.retrain_v2.start_chk2_deepseek_low_resume import (
    EXPECTED_OUTPUT_DIR,
    EXPECTED_RESUME_CHECKPOINT,
    EXPECTED_RUN_ROOT,
    EXPECTED_SESSION,
    _validate_benchmark,
    _validate_config,
    _validate_resume_checkpoint,
)
from open_r1.configs import GRPOConfig, GRPOScriptArguments, LoraArguments, ModelConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
HIGH_CONFIG = (
    REPO_ROOT
    / "configs/retrain_v2/"
    "chk2_analysis_grpo_cp200_deepseek_high_totalsl_v1_20260810.yaml"
)
LOW_CONFIG = (
    REPO_ROOT
    / "configs/retrain_v2/"
    "chk2_analysis_grpo_cp200_deepseek_low_totalsl_v1_resume_cp1_20260810.yaml"
)
LAUNCHER = REPO_ROOT / "run/retrain_v2/start_chk2_cp200_deepseek_low_resume_cp1.sh"
LAUNCH_MODULE = REPO_ROOT / "jobs/retrain_v2/start_chk2_deepseek_low_resume.py"
BENCHMARK = (
    REPO_ROOT
    / "docs/summary/20260810T123000Z/chk1_cp200_deepseek_chk2/"
    "chk2_low_judge_benchmark_v1/benchmark_summary.json"
)
BINDING = (
    REPO_ROOT
    / "docs/summary/20260810T123000Z/chk1_cp200_deepseek_chk2/"
    "chk2_low_resume_launch_binding.json"
)


def _yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_low_resume_config_parses_with_current_trl() -> None:
    script, training, model, peft = TrlParser(
        (GRPOScriptArguments, GRPOConfig, ModelConfig, LoraArguments)
    ).parse_args_and_config(args=["--config", str(LOW_CONFIG)])

    assert script.reward_funcs == ["grounded_analysis_v3_deepseek_low"]
    assert script.judge_model == "deepseek-v4-flash"
    assert script.judge_api_key_env == "DEEPSEEK_API_KEY"
    assert training.resume_from_checkpoint == EXPECTED_RESUME_CHECKPOINT.as_posix()
    assert training.output_dir == EXPECTED_OUTPUT_DIR.as_posix()
    assert training.max_completion_length == 4096
    assert training.generation_batch_size == 4
    assert model.load_in_4bit is True
    assert peft.peft_r == 32


def test_low_resume_changes_only_reward_resume_and_destination() -> None:
    high = _yaml(HIGH_CONFIG)
    low = _yaml(LOW_CONFIG)
    allowed_changes = {
        "reward_funcs",
        "peft_merged_model_path",
        "output_dir",
        "resume_from_checkpoint",
    }
    common_keys = set(high) & set(low)
    changed_common = {key for key in common_keys if high[key] != low[key]}

    assert changed_common == allowed_changes - {"resume_from_checkpoint"}
    assert set(low) - set(high) == {"resume_from_checkpoint"}
    assert set(high) - set(low) == set()
    assert low["reward_funcs"] == ["grounded_analysis_v3_deepseek_low"]
    assert low["resume_from_checkpoint"] == EXPECTED_RESUME_CHECKPOINT.as_posix()
    assert low["output_dir"] == EXPECTED_OUTPUT_DIR.as_posix()
    assert low["judge_max_completion_tokens"] == 16384
    assert low["judge_timeout"] == 420
    assert low["judge_max_retries"] == 2


def test_low_resume_config_and_checkpoint_contracts_are_valid() -> None:
    payload = _validate_config(LOW_CONFIG, REPO_ROOT)
    state = _validate_resume_checkpoint(REPO_ROOT / EXPECTED_RESUME_CHECKPOINT)

    assert payload["output_dir"] == EXPECTED_OUTPUT_DIR.as_posix()
    assert state["global_step"] == 1
    assert not (REPO_ROOT / EXPECTED_RUN_ROOT).exists()


def test_low_benchmark_is_passing_and_bound_to_low_effort() -> None:
    payload = _validate_benchmark(BENCHMARK)

    assert payload["status"] == "passed"
    assert payload["suitable_for_chk2_trial"] is True
    assert payload["provider"]["reasoning_effort"] == "low"
    assert payload["provider"]["max_output_tokens"] == 16384
    assert payload["provider"]["logical_requests"] == 10
    assert all(payload["gates"].values())


def test_launcher_isolated_to_gpu1_and_excludes_other_stages_and_local_judge() -> None:
    wrapper = LAUNCHER.read_text(encoding="utf-8")
    module = LAUNCH_MODULE.read_text(encoding="utf-8")

    assert "conda run --no-capture-output -n fomc_trainer" in wrapper
    assert "--execute" in wrapper
    assert "CUDA_VISIBLE_DEVICES=1" in module
    assert '"jobs.train.train_grpo"' in module
    assert EXPECTED_SESSION in module
    assert "start_chk2_background.sh" not in wrapper + module
    assert "stage.sh" not in wrapper + module
    assert "judge.sh" not in wrapper + module
    assert "Qwen3.5-9B" not in wrapper + module
    assert "chk3" not in wrapper + module
    assert "chk4" not in wrapper + module


def test_benchmark_summary_contains_no_persisted_sensitive_payloads() -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    privacy = payload["privacy"]

    assert privacy["api_key_persisted"] is False
    assert privacy["hidden_reasoning_persisted"] is False
    assert privacy["provider_output_persisted"] is False
    assert privacy["raw_candidates_persisted"] is False
    assert privacy["raw_evidence_persisted"] is False


def test_launch_binding_pins_runtime_provenance_and_resume_checkpoint() -> None:
    payload = json.loads(BINDING.read_text(encoding="utf-8"))
    artifacts = payload["artifacts"]

    assert payload["run_id"] == EXPECTED_RUN_ROOT.name
    assert payload["session_name"] == EXPECTED_SESSION
    assert artifacts["config"]["path"] == str(LOW_CONFIG.relative_to(REPO_ROOT))
    assert artifacts["resume_checkpoint"]["path"] == (
        EXPECTED_RESUME_CHECKPOINT.as_posix()
    )
    assert artifacts["resume_checkpoint"]["sha256"] == (
        "c59fc57f1b8920201633de074f26497929c3a7a9ed18c39d69880c15c975ab61"
    )
    for required in (
        "reward_module",
        "reward_registry",
        "trainer",
        "config",
        "benchmark_summary",
        "resume_checkpoint",
    ):
        assert required in artifacts
        assert len(artifacts[required]["sha256"]) == 64
