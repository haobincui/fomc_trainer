from __future__ import annotations

from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    "output/training/retrain_v2/"
    "chk3_direct_chk1_cp200_full3ep_lr1e6_20260810"
)
BASE_MODEL = (
    "output/training/retrain_v2/"
    "chk1_clean_v2_lr1e6_selected_cp200_for_chk2_20260810/merged/chk1"
)
EVALUATION_ROOT = (
    "output/evaluation/main/"
    "chk3_checkpoint_selection_cp200_cp318_20260811_v1"
)
TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


@pytest.mark.parametrize("step", (200, 318))
def test_chk3_candidate_merge_config_is_exact_and_evaluation_only(step: int) -> None:
    path = (
        ROOT
        / "configs/retrain_v2"
        / f"chk3_direct_chk1_cp200_cp{step}_merge_for_evaluation_20260811.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["model_name_or_path"] == BASE_MODEL
    assert payload["output_dir"] == f"{RUN_ROOT}/adapters/chk3/checkpoint-{step}"
    assert payload["peft_merged_model_path"] == (
        f"{EVALUATION_ROOT}/models/chk3_cp{step}_merged"
    )
    assert payload["peft_r"] == 32
    assert payload["peft_lora_alpha"] == 64
    assert payload["peft_lora_dropout"] == 0.05
    assert payload["peft_bias"] == "none"
    assert set(payload["peft_target_modules"]) == TARGET_MODULES
    assert len(payload["peft_target_modules"]) == 7
    assert payload["source_native_generation_manifest"] == (
        "output/evaluation/main/chk3_native_cp_sweep_n12_20260811_v1/"
        f"cp{step}/manifest.json"
    )
    assert payload["scope"] == f"chk3_checkpoint_{step}_evaluation_merge_only"
    assert payload["canonical_dag_promotable"] is False


def test_chk3_candidate_merge_destinations_are_distinct() -> None:
    destinations = set()
    for step in (200, 318):
        path = (
            ROOT
            / "configs/retrain_v2"
            / f"chk3_direct_chk1_cp200_cp{step}_merge_for_evaluation_20260811.yaml"
        )
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        destinations.add(payload["peft_merged_model_path"])

    assert len(destinations) == 2
