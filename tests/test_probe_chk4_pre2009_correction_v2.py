from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from jobs.retrain_v2 import probe_chk4_pre2009_correction_v2 as probe


def test_frozen_partitions_are_disjoint_and_ordered() -> None:
    expected = {
        probe.SELECTION_STAGE: (
            "dec-e2006fdd25f1021f2b918f01",
            "dec-661e567e81571eb9774f86fe",
            "dec-9b8503773d5ad0d118e1b8eb",
            "dec-b2b27e148171d130a09ea6d5",
            "dec-bf310b21fe82b5c0cb502f51",
            "dec-26b22d74159882e6864770f4",
        ),
        probe.BLIND_STAGE: (
            "dec-8e89d44528247ac319a2afbf",
            "dec-7a8dfb4edce9d6cc39a316eb",
            "dec-e80b4a3b73db89db41f95794",
            "dec-cfe8cd5e89cdf2d58e4f33c6",
            "dec-612552be86b5fab9002f2be4",
            "dec-b48f42df555b84e3ccf47ad7",
        ),
        probe.GRPO_SMOKE_STAGE: (
            "dec-b1914990f5551d18fa78a28a",
            "dec-75d77b434823305ef5db5378",
            "dec-a1b3aa8508c5c893ef267fe3",
            "dec-de51ed8cd5d92206c363873c",
        ),
        probe.RETENTION_STAGE: (
            "dec-aa54ffdf045b3106220b84c4",
            "dec-974aefdccb2aa2c4c14ed652",
            "dec-761b0ff6e4a9fd1de29d71f1",
            "dec-860f054c9765d7bbe54a6dec",
            "dec-5a165503b73751ffc5d8ff63",
        ),
    }
    assert probe.PARTITIONS == expected
    flattened = [item for stage in probe.ALL_STAGES for item in expected[stage]]
    assert len(flattened) == len(set(flattened)) == 21


def test_stage_seeds_are_domain_separated_sha256_values() -> None:
    for stage, expected_seed in probe.EXPECTED_STAGE_SEEDS.items():
        contract = probe.derive_stage_seed(stage)
        digest = hashlib.sha256(
            probe.STAGE_SEED_SALTS[stage].encode("utf-8")
        ).hexdigest()
        assert contract == {
            "algorithm": "int(sha256(stage_salt)[:8],16)&0x7fffffff",
            "stage_salt": probe.STAGE_SEED_SALTS[stage],
            "stage_salt_sha256": digest,
            "seed": expected_seed,
        }
    assert len(set(probe.EXPECTED_STAGE_SEEDS.values())) == 4


def test_stage_contracts_freeze_gates_and_smoke_optimizer_order() -> None:
    selection = probe._stage_contract(probe.SELECTION_STAGE)
    assert selection["quality_gate"]["selection_rule"] == (
        "earliest_all_pass_in_preregistered_order_2_4_6"
    )
    assert selection["quality_gate"]["per_prompt"] == {
        "hold_correct_nonzero_min": 2,
        "action_correct_nonzero_min": 1,
        "strict_json_exact_min": 1,
        "reward_variation": "pstdev_gt_0_or_strict_json_exact_eq_4",
        "boundary_count_min": 3,
        "cap_count_max": 1,
        "periodic_tail_count_max": 0,
        "all_rewards_finite": True,
    }
    smoke = probe._stage_contract(probe.GRPO_SMOKE_STAGE)
    assert smoke["batches"] == [
        ["dec-de51ed8cd5d92206c363873c", "dec-75d77b434823305ef5db5378"],
        ["dec-b1914990f5551d18fa78a28a", "dec-a1b3aa8508c5c893ef267fe3"],
    ]
    assert smoke["quality_gate"]["optimizer"] == {
        "loss_finite": True,
        "grad_norm_min_when_either_prompt_not_all_strict_exact": 1e-12,
    }
    assert smoke["quality_gate"]["failure_is_terminal_and_blocks_full"] is True


def test_first_bound_eos_preserves_raw_and_rejects_non_eos_suffix() -> None:
    normalized, evidence = probe.normalize_at_first_bound_eos(
        [10, 11, 128001, 128001, 128001], {128001}
    )
    assert normalized == [10, 11, 128001]
    assert evidence["raw_count"] == 5
    assert evidence["normalized_count"] == 3
    assert evidence["discarded_count"] == 2
    assert evidence["discarded_unique_ids"] == [128001]
    assert evidence["discarded_all_bound_eos"] is True

    with pytest.raises(probe.CorrectionV2ProbeError, match="non-EOS token"):
        probe.normalize_at_first_bound_eos([10, 128001, 99], {128001})


def test_prompt_metric_uses_exact_nonzero_and_strict_exact() -> None:
    rows = []
    for index, (reward, exact, strict) in enumerate(
        (
            (1.0, True, True),
            (0.2375, True, False),
            (0.05, False, True),
            (0, False, False),
        )
    ):
        rows.append(
            {
                "decision_dense_v3_reward": reward,
                "decision_exact": exact,
                "strict_json": strict,
                "think_boundary_count": 1,
                "cap_reached": index == 3,
                "strict_periodic_tail": False,
            }
        )
    block = probe._metric_block(rows)
    assert block["correct_nonzero_count"] == 2
    assert block["strict_json_exact_count"] == 1
    assert block["reward_std"] > 0
    assert block["boundary_count"] == 4
    assert block["cap_count"] == 1


def test_model_source_rejects_noncanonical_adapter_even_with_matching_fingerprints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    canonical_adapter = probe.SFT_OUTPUT / "checkpoint-2"
    forged_source = {
        "mode": probe.reward_probe.PEFT_ADAPTER_MODE,
        "model_fingerprint": {"sha256": probe.SELECTED_CP38_MODEL_SHA256},
        "effective_model_fingerprint": {"sha256": "2" * 64},
        "provenance": {
            "mode": probe.reward_probe.PEFT_ADAPTER_MODE,
            "base_model": {
                "directory": {
                    "path": str(probe.SELECTED_CP38_MODEL),
                    "sha256": probe.SELECTED_CP38_MODEL_SHA256,
                }
            },
            "adapter": {"directory": {"path": str(canonical_adapter)}},
        },
    }
    monkeypatch.setattr(
        probe.reward_probe,
        "_prepare_model_source",
        lambda **_kwargs: forged_source,
    )

    with pytest.raises(probe.CorrectionV2ProbeError, match="canonical correction-v2"):
        probe._model_source(
            role="candidate",
            model_path=None,
            base_model_path=probe.SELECTED_CP38_MODEL,
            adapter_path=tmp_path / "checkpoint-2",
        )

    # The path gate runs before fingerprinting, so a self-consistent forged
    # source cannot make the arbitrary adapter eligible.
    assert (
        probe._model_source(
            role="candidate",
            model_path=None,
            base_model_path=probe.SELECTED_CP38_MODEL,
            adapter_path=canonical_adapter,
        )
        == forged_source
    )

    with pytest.raises(probe.CorrectionV2ProbeError, match="canonical exact cp38"):
        probe._model_source(
            role="baseline",
            model_path=tmp_path / "copied-cp38",
            base_model_path=None,
            adapter_path=None,
        )


def test_forged_self_sealed_authorization_cannot_change_model_fingerprint(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "suite.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    output = probe.SCREEN_ROOT / "selection/checkpoint-2"
    authorization_path = (
        probe.SCREEN_ROOT / "authorizations/selection-checkpoint-2.json"
    )
    source = {
        "provenance": {"mode": "base_model_plus_adapter"},
        "model_fingerprint": {"sha256": "1" * 64},
        "effective_model_fingerprint": {"sha256": "2" * 64},
    }
    authorization = {
        "scope": {
            "stage": probe.SELECTION_STAGE,
            "model_role": "candidate",
            "checkpoint_step": 2,
            "gpu": 1,
            "fresh_output": True,
        },
        "output": str(output.resolve()),
        "manifest": {"path": str(manifest_path.resolve()), "sha256": "a" * 64},
        "stage_contract": {"seed": 1},
        "model_source": source["provenance"],
        "model_sha256": "1" * 64,
        # A forged receipt claims another effective model while remaining
        # internally self-hashable.
        "effective_model_sha256": "3" * 64,
        "implementation": {},
        "predecessors": {"selection": None, "blind": None},
    }
    with pytest.raises(probe.CorrectionV2ProbeError, match="model fingerprint"):
        probe._verify_authorization_payload(
            authorization=authorization,
            authorization_path=authorization_path,
            manifest={"stages": {probe.SELECTION_STAGE: {"seed": 1}}},
            manifest_path=manifest_path,
            manifest_sha256="a" * 64,
            stage=probe.SELECTION_STAGE,
            role="candidate",
            checkpoint_step=2,
            output_dir=output,
            source=source,
        )

    authorization["effective_model_sha256"] = "2" * 64
    with pytest.raises(probe.CorrectionV2ProbeError, match="implementation inventory"):
        probe._verify_authorization_payload(
            authorization=authorization,
            authorization_path=authorization_path,
            manifest={"stages": {probe.SELECTION_STAGE: {"seed": 1}}},
            manifest_path=manifest_path,
            manifest_sha256="a" * 64,
            stage=probe.SELECTION_STAGE,
            role="candidate",
            checkpoint_step=2,
            output_dir=output,
            source=source,
        )
