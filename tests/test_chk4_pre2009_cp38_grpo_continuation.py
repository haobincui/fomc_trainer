from __future__ import annotations

from itertools import islice
from pathlib import Path

import pytest
from trl import TrlParser
from trl.trainer.utils import RepeatSampler

from jobs.retrain_v2 import chk4_pre2009_cp38_grpo_continuation as continuation
from jobs.retrain_v2 import chk4_v5_cp24_grpo_continuation as v5
from jobs.retrain_v2 import merge_chk4_selected_sft_checkpoint as selected
from open_r1.configs import GRPOConfig, GRPOScriptArguments, LoraArguments, ModelConfig
from open_r1.provenance import sha256_file
from open_r1.utils.callbacks import Chk4Pre2009Cp38SmokeGateCallback


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def test_cp38_selection_profile_is_strict_and_hash_pinned() -> None:
    profile = selected.SELECTION_PROFILES[selected.PRE2009_CP38_PROFILE_NAME]
    assert profile.completed_step == 39
    assert profile.eval_steps == 13
    assert profile.allowed_checkpoint_steps == frozenset({38})
    assert profile.checkpoint_keep_steps == (13, 26, 39)
    assert profile.sft_dataset_role == "decision_sft_pre2009_balanced"
    assert profile.grpo_dataset_role == "decision_grpo_pre2009_balanced"
    assert profile.fixed_schedule_schema == "manifest-fixed-schedule-runtime-v2"
    assert profile.fixed_schedule_type == "manifest_fixed_schedule_v2"
    assert profile.fixed_schedule_rows == profile.fixed_train_rows == 312
    assert profile.required_checkpoint_sha256 == continuation.CHECKPOINT_SHA256
    assert (
        profile.required_adapter_weights_sha256 == continuation.ADAPTER_WEIGHTS_SHA256
    )
    assert profile.required_pilot_manifest_sha256 == continuation.PILOT_MANIFEST_SHA256
    assert profile.required_pilot_summary_sha256 == continuation.PILOT_SUMMARY_SHA256
    assert profile.required_pilot_results_sha256 == continuation.PILOT_RESULTS_SHA256
    assert profile.required_pilot_launch_sha256 == continuation.PILOT_LAUNCH_SHA256


def test_versioned_configs_are_hash_bound_and_isolated() -> None:
    smoke = continuation._validate_config(SOURCE_ROOT, continuation.SMOKE)
    full = continuation._validate_config(SOURCE_ROOT, continuation.FULL)
    assert (
        sha256_file(SOURCE_ROOT / continuation.SMOKE.config)
        == continuation.SMOKE.config_sha256
    )
    assert (
        sha256_file(SOURCE_ROOT / continuation.FULL.config)
        == continuation.FULL.config_sha256
    )
    assert smoke["model_name_or_path"] == full["model_name_or_path"]
    assert (
        "selected_sft_checkpoints/checkpoint-38/merged/chk4_sft"
        in smoke["model_name_or_path"]
    )
    assert smoke["output_dir"] != full["output_dir"]
    assert smoke["max_steps"] == 2
    assert "max_steps" not in full
    assert (
        smoke["dataset_chk4_role"]
        == full["dataset_chk4_role"]
        == "decision_grpo_pre2009_balanced"
    )
    assert smoke["callbacks"].count("chk4_pre2009_cp38_smoke_gate") == 1
    assert "chk4_pre2009_cp38_smoke_gate" not in full["callbacks"]


def test_seed31416_sampler_prefix_binds_passed_pilot_directions() -> None:
    config = continuation._validate_config(SOURCE_ROOT, continuation.SMOKE)
    contract = continuation._smoke_sampling_contract(SOURCE_ROOT, config)
    sampler = RepeatSampler(
        range(312),
        mini_repeat_count=4,
        batch_size=2,
        repeat_count=8,
        shuffle=True,
        seed=continuation.SMOKE_SEED,
    )
    observed = list(islice(sampler, 72))
    assert observed[:64] == ([238] * 4 + [133] * 4) * 8
    assert observed[64:] == [104] * 4 + [87] * 4
    assert contract["source_indices"] == [238, 133, 104, 87]
    assert [row["direction"] for row in contract["source_rows"]] == [
        "hold",
        "hike",
        "hold",
        "cut",
    ]
    assert contract["verified_sampler_prefix"]["sha256"] == (
        "ee14d73632b79cd7e0610ca0464b19523cd076f6ceb98d96408c9b3d48ebabf2"
    )
    assert contract["source_rows"][1]["sample_id"] == "dec-4dfab939b0a910c949931475"
    assert contract["source_rows"][3]["sample_id"] == "dec-8b8d55ea065b662a19cefe88"


def test_pre2009_smoke_callback_accepts_only_seed31416_contract() -> None:
    _, training, model, _ = TrlParser(
        (GRPOScriptArguments, GRPOConfig, ModelConfig, LoraArguments)
    ).parse_args_and_config(
        args=["--config", str(SOURCE_ROOT / continuation.SMOKE.config)]
    )
    callback = Chk4Pre2009Cp38SmokeGateCallback(training, model)
    assert callback._EXPECTED_SEED == callback._EXPECTED_DATA_SEED == 31416
    assert callback._AUDIT_SCHEMA == continuation.STEP_GATE_SCHEMA
    training.seed = 5
    with pytest.raises(ValueError, match="seed=31416"):
        Chk4Pre2009Cp38SmokeGateCallback(training, model)


def test_engine_contract_restores_historical_v5_defaults() -> None:
    original = {
        "source_profile": v5.SOURCE_PROFILE,
        "checkpoint_step": v5.CHECKPOINT_STEP,
        "smoke_seed": v5.SMOKE_SEED,
        "callback": v5.SMOKE_CALLBACK_NAME,
    }
    continuation._validate_config(SOURCE_ROOT, continuation.SMOKE)
    assert v5.SOURCE_PROFILE == original["source_profile"]
    assert v5.CHECKPOINT_STEP == original["checkpoint_step"]
    assert v5.SMOKE_SEED == original["smoke_seed"]
    assert v5.SMOKE_CALLBACK_NAME == original["callback"]


def test_full_authorization_is_blocked_before_smoke_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked(_: Path):
        raise continuation.ContinuationError("smoke gate receipt is missing")

    monkeypatch.setattr(v5, "_verify_smoke_gate", blocked)
    with pytest.raises(continuation.ContinuationError, match="smoke gate"):
        continuation._authorization_payload(SOURCE_ROOT, continuation.FULL)


def test_launcher_is_gpu1_only_and_merge_is_cpu_only() -> None:
    path = SOURCE_ROOT / "run/retrain_v2/chk4_pre2009_cp38_grpo_continuation.sh"
    text = path.read_text(encoding="utf-8")
    assert '"${ROOT_DIR}/run/retrain_v2/gpu_gate.sh" 1' in text
    assert 'export CUDA_VISIBLE_DEVICES=""' in text
    assert "--checkpoint-step 38" in text
    assert "merge-selected requires --execute" in text
    assert "--num_processes" in v5.training_command.__code__.co_consts
