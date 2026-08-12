from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import save_file

from jobs.main.verify_lora_merge_lineage import verify_exact_lora_merge
from open_r1.validator.loo_generation_spec import validate_manifest_integrity


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    base = root / "models" / "base"
    adapter = root / "output" / "adapters" / "analysis"
    merged = root / "output" / "merged" / "analysis"
    base.mkdir(parents=True)
    adapter.mkdir(parents=True)
    merged.mkdir(parents=True)

    base_tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16
        ),
        "model.layers.0.self_attn.k_proj.weight": torch.tensor(
            [[5.0, 6.0], [7.0, 8.0]], dtype=torch.bfloat16
        ),
    }
    lora_a = torch.tensor([[0.25, -0.5]], dtype=torch.bfloat16)
    lora_b = torch.tensor([[0.75], [-0.25]], dtype=torch.bfloat16)
    merged_tensors = dict(base_tensors)
    merged_tensors["model.layers.0.self_attn.q_proj.weight"] = (
        base_tensors["model.layers.0.self_attn.q_proj.weight"].float()
        + (lora_b.float() @ lora_a.float()) * 2.0
    ).to(torch.bfloat16)
    shard = "model-00001-of-00001.safetensors"
    save_file(base_tensors, base / shard)
    save_file(merged_tensors, merged / shard)
    weight_map = {name: shard for name in base_tensors}
    for model in (base, merged):
        _write_json(
            model / "model.safetensors.index.json",
            {"metadata": {}, "weight_map": weight_map},
        )
        _write_json(model / "config.json", {"model_type": "llama"})
        _write_json(model / "tokenizer.json", {"version": "1"})
        _write_json(model / "tokenizer_config.json", {"tokenizer_class": "test"})

    _write_json(
        adapter / "adapter_config.json",
        {
            "base_model_name_or_path": "models/base",
            "bias": "none",
            "fan_in_fan_out": False,
            "lora_alpha": 2,
            "peft_type": "LORA",
            "r": 1,
            "target_modules": ["q_proj"],
            "task_type": "CAUSAL_LM",
            "use_dora": False,
            "use_rslora": False,
        },
    )
    save_file(
        {
            (
                "base_model.model.model.layers.0.self_attn.q_proj."
                "lora_A.weight"
            ): lora_a,
            (
                "base_model.model.model.layers.0.self_attn.q_proj."
                "lora_B.weight"
            ): lora_b,
        },
        adapter / "adapter_model.safetensors",
    )
    training_config = root / "sft.yaml"
    training_config.write_text(
        yaml.safe_dump(
            {
                "model_name_or_path": "models/base",
                "output_dir": "output/adapters/analysis",
                "peft_merged_model_path": "output/merged/analysis",
                "peft_r": 1,
                "peft_lora_alpha": 2,
                "peft_target_modules": ["q_proj"],
                "peft_bias": "none",
            }
        ),
        encoding="utf-8",
    )
    return base, adapter, merged, training_config


def test_exact_lora_merge_is_sealed_and_counts_all_tensors(tmp_path: Path) -> None:
    base, adapter, merged, training_config = _fixture(tmp_path)
    evidence = verify_exact_lora_merge(
        base_model=base,
        adapter=adapter,
        merged_model=merged,
        training_config=training_config,
        fingerprint_sources=False,
    )

    validate_manifest_integrity(evidence)
    assert evidence["conclusion"] == "exact_base_plus_adapter_merge_verified"
    verification = evidence["tensor_verification"]
    assert verification["model_tensor_count"] == 2
    assert verification["adapted_model_tensor_count"] == 1
    assert verification["unchanged_model_tensor_count"] == 1
    assert verification["mismatch_count"] == 0
    assert evidence["metadata_evidence"]["training_config"]["path_checks"] == {
        "model_name_or_path_matches_base": True,
        "output_dir_matches_adapter": True,
        "peft_merged_model_path_matches_merged": True,
    }


def test_rejects_change_to_non_adapter_tensor(tmp_path: Path) -> None:
    base, adapter, merged, _ = _fixture(tmp_path)
    shard = merged / "model-00001-of-00001.safetensors"
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.tensor(
            [[1.375, 1.25], [2.875, 4.25]], dtype=torch.bfloat16
        ),
        "model.layers.0.self_attn.k_proj.weight": torch.tensor(
            [[99.0, 6.0], [7.0, 8.0]], dtype=torch.bfloat16
        ),
    }
    save_file(tensors, shard)

    with pytest.raises(ValueError, match="Non-adapted tensor changed"):
        verify_exact_lora_merge(
            base_model=base,
            adapter=adapter,
            merged_model=merged,
            fingerprint_sources=False,
        )


def test_rejects_inexact_adapter_merge(tmp_path: Path) -> None:
    base, adapter, merged, _ = _fixture(tmp_path)
    shard = merged / "model-00001-of-00001.safetensors"
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.tensor(
            [[1.5, 1.25], [2.875, 4.25]], dtype=torch.bfloat16
        ),
        "model.layers.0.self_attn.k_proj.weight": torch.tensor(
            [[5.0, 6.0], [7.0, 8.0]], dtype=torch.bfloat16
        ),
    }
    save_file(tensors, shard)

    with pytest.raises(ValueError, match="exact PEFT merge expression"):
        verify_exact_lora_merge(
            base_model=base,
            adapter=adapter,
            merged_model=merged,
            fingerprint_sources=False,
        )
