from pathlib import Path

import yaml

from jobs.retrain_v2 import chk4_pre2009_cp38_grpo_direct_full as direct
from open_r1.provenance import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_direct_full_config_only_versions_outputs() -> None:
    contract = direct._config_contract(ROOT)
    assert set(contract["differences"]) == {"output_dir", "peft_merged_model_path"}
    assert sha256_file(ROOT / direct.CONFIG) == direct.CONFIG_SHA256
    config = contract["config"]
    assert "max_steps" not in config
    assert config["num_train_epochs"] == 3
    assert "chk4_pre2009_cp38_smoke_gate" not in config["callbacks"]


def test_direct_full_is_fresh_and_explicitly_no_smoke() -> None:
    result = direct.preflight(ROOT)
    assert result["status"] == "ready"
    assert result["source"]["checkpoint_step"] == 38
    assert result["smoke"] == {
        "required": False,
        "removed_by": "explicit_user_instruction_2026-08-12",
        "prior_artifacts_preserved": True,
    }
    assert result["data"]["train"]["rows"] == 312


def test_direct_config_matches_reference_training_payload() -> None:
    direct_cfg = yaml.safe_load((ROOT / direct.CONFIG).read_text(encoding="utf-8"))
    reference_cfg = yaml.safe_load(
        (ROOT / direct.REFERENCE_CONFIG).read_text(encoding="utf-8")
    )
    for key in ("output_dir", "peft_merged_model_path"):
        direct_cfg.pop(key)
        reference_cfg.pop(key)
    assert direct_cfg == reference_cfg


def test_launcher_is_gpu1_only_and_has_no_smoke_command() -> None:
    text = (ROOT / "run/retrain_v2/chk4_pre2009_cp38_grpo_direct_full.sh").read_text(
        encoding="utf-8"
    )
    assert 'gpu_gate.sh" 1' in text
    assert "CUDA_VISIBLE_DEVICES=1" in text
    assert "authorize-smoke" not in text
    assert "launch-smoke" not in text
    assert "gate-smoke" not in text
