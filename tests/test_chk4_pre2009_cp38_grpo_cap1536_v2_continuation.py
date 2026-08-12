from __future__ import annotations

from itertools import islice
from pathlib import Path

import pytest
from trl import TrlParser
from trl.trainer.utils import RepeatSampler

from jobs.retrain_v2 import chk4_pre2009_cp38_grpo_cap1536_v2_continuation as retry
from jobs.retrain_v2 import chk4_pre2009_cp38_grpo_continuation as v1
from jobs.retrain_v2 import chk4_v5_cp24_grpo_continuation as engine
from open_r1.configs import GRPOConfig, GRPOScriptArguments, LoraArguments, ModelConfig
from open_r1.provenance import sha256_file
from open_r1.utils.callbacks import Chk4Pre2009Cp38SmokeGateCallback


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def test_retry_configs_are_hash_bound_and_only_change_cap_plus_paths() -> None:
    smoke = retry._validate_config(SOURCE_ROOT, retry.SMOKE)
    full = retry._validate_config(SOURCE_ROOT, retry.FULL)
    assert sha256_file(SOURCE_ROOT / retry.SMOKE.config) == retry.SMOKE.config_sha256
    assert sha256_file(SOURCE_ROOT / retry.FULL.config) == retry.FULL.config_sha256
    assert smoke["max_completion_length"] == full["max_completion_length"] == 1536
    smoke_delta = retry._config_delta(SOURCE_ROOT, retry.SMOKE)
    full_delta = retry._config_delta(SOURCE_ROOT, retry.FULL)
    for delta in (smoke_delta, full_delta):
        assert (
            delta["claim"] == "max_completion_length_is_the_only_hyperparameter_change"
        )
        assert delta["hyperparameter_change"] == {"from": 1024, "to": 1536}
        assert set(delta["required_versioned_path_changes"]) == {
            "output_dir",
            "peft_merged_model_path",
        }
    assert smoke_delta["smoke_seed_contract"] == {
        "seed": 31416,
        "data_seed": 31416,
        "sampler_prefix_sha256": (
            "ee14d73632b79cd7e0610ca0464b19523cd076f6ceb98d96408c9b3d48ebabf2"
        ),
        "target_groups": {"1": ["hold", "hike"], "2": ["hold", "cut"]},
    }
    assert full_delta["smoke_seed_contract"] is None


def test_retry_predecessor_failure_is_exactly_hash_bound() -> None:
    evidence = retry._validate_predecessor_failure(SOURCE_ROOT)
    assert evidence["outcome"] == "failed_closed_at_step_1"
    assert evidence["max_completion_length"] == 1024
    assert evidence["completion_clipped_ratio"] == 0.625
    assert evidence["failed_checks"] == [
        "clipped_ratio_le_0_25",
        "each_target_direction_correct_nonzero",
        "each_target_reward_std_gt_zero",
    ]
    assert evidence["passed_checks"] == ["loss_finite", "grad_norm_gt_1e_12"]
    assert evidence["artifacts"]["reward"]["rows"] == 8
    assert evidence["artifacts"]["runtime_safety"]["rows"] == 1
    assert evidence["artifacts"]["step_gate"]["rows"] == 1
    for name, (_, expected_sha) in retry.PREDECESSOR_ARTIFACTS.items():
        assert evidence["artifacts"][name]["sha256"] == expected_sha


def test_retry_smoke_keeps_seed_sampler_and_callback_gate() -> None:
    config = retry._validate_config(SOURCE_ROOT, retry.SMOKE)
    contract = retry._smoke_sampling_contract(SOURCE_ROOT, config)
    sampler = RepeatSampler(
        range(312),
        mini_repeat_count=4,
        batch_size=2,
        repeat_count=8,
        shuffle=True,
        seed=31416,
    )
    observed = list(islice(sampler, 72))
    assert observed[:64] == ([238] * 4 + [133] * 4) * 8
    assert observed[64:] == [104] * 4 + [87] * 4
    assert contract["verified_sampler_prefix"]["sha256"] == (
        "ee14d73632b79cd7e0610ca0464b19523cd076f6ceb98d96408c9b3d48ebabf2"
    )
    assert contract["optimizer_step_target_groups"] == {
        "1": ["hold", "hike"],
        "2": ["hold", "cut"],
    }
    _, training, model, _ = TrlParser(
        (GRPOScriptArguments, GRPOConfig, ModelConfig, LoraArguments)
    ).parse_args_and_config(args=["--config", str(SOURCE_ROOT / retry.SMOKE.config)])
    callback = Chk4Pre2009Cp38SmokeGateCallback(training, model)
    assert callback._EXPECTED_SEED == callback._EXPECTED_DATA_SEED == 31416
    assert callback._AUDIT_SCHEMA == retry.STEP_GATE_SCHEMA


def test_retry_context_restores_v1_engine_defaults() -> None:
    original = {
        "smoke": engine.SMOKE,
        "full": engine.FULL,
        "max_completion_length": engine.MAX_COMPLETION_LENGTH,
        "provider": engine.AUTHORIZATION_CONTEXT_PROVIDER,
    }
    retry._validate_config(SOURCE_ROOT, retry.SMOKE)
    assert engine.SMOKE is original["smoke"]
    assert engine.FULL is original["full"]
    assert engine.MAX_COMPLETION_LENGTH == original["max_completion_length"] == 1024
    assert engine.AUTHORIZATION_CONTEXT_PROVIDER is original["provider"] is None


def test_v1_failure_artifacts_and_configs_remain_unchanged() -> None:
    assert sha256_file(SOURCE_ROOT / v1.SMOKE.config) == v1.SMOKE.config_sha256
    assert sha256_file(SOURCE_ROOT / v1.FULL.config) == v1.FULL.config_sha256
    for relative, expected_sha in retry.PREDECESSOR_ARTIFACTS.values():
        assert sha256_file(SOURCE_ROOT / relative) == expected_sha


def test_full_retry_authorization_still_requires_retry_smoke_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked(_: Path):
        raise retry.ContinuationError("retry smoke gate receipt is missing")

    monkeypatch.setattr(engine, "_verify_smoke_gate", blocked)
    with pytest.raises(retry.ContinuationError, match="retry smoke gate"):
        retry._authorization_payload(SOURCE_ROOT, retry.FULL)


def test_retry_launcher_is_gpu1_only_and_fresh_versioned() -> None:
    path = (
        SOURCE_ROOT / "run/retrain_v2/chk4_pre2009_cp38_grpo_cap1536_v2_continuation.sh"
    )
    text = path.read_text(encoding="utf-8")
    assert '"${ROOT_DIR}/run/retrain_v2/gpu_gate.sh" 1' in text
    assert "cap1536_v2" in text
    assert "resume" not in text
    assert "--num_processes" in engine.training_command.__code__.co_consts
