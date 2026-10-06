from jobs.retrain_v2 import probe_chk4_cp38_direct_grpo_checkpoints as probe


def _samples(counts):
    result = {}
    for direction, count in counts.items():
        for index in range(count):
            sample_id = f"{direction}-{index}"
            result[sample_id] = {"direction": direction}
    return result


def test_validation_partition_is_disjoint_complete_and_frozen_shape():
    samples = _samples({"hold": 3, "hike": 10, "cut": 0})
    selection, blind = probe.partition_validation(samples)
    assert len(selection) == 9
    assert len(blind) == 4
    assert set(selection).isdisjoint(blind)
    assert set(selection) | set(blind) == set(samples)
    assert sum(samples[item]["direction"] == "hold" for item in blind) == 1
    assert sum(samples[item]["direction"] == "hike" for item in blind) == 3


def test_retention_panel_has_two_per_direction():
    samples = _samples({"hold": 4, "hike": 5, "cut": 6})
    selected = probe.partition_retention(samples)
    assert len(selected) == 6
    for direction in ("hold", "hike", "cut"):
        assert sum(samples[item]["direction"] == direction for item in selected) == 2


def test_panel_contract_never_exceeds_eight_generated_sequences():
    ids = [f"sample-{index}" for index in range(9)]
    selection = probe._panel_contract(
        population_role="selection", sample_ids=ids, sampled_per_prompt=1, seed=1
    )
    blind = probe._panel_contract(
        population_role="blind", sample_ids=ids[:4], sampled_per_prompt=4, seed=2
    )
    assert max(map(len, selection["batches"]["sampled"])) == 8
    assert max(map(len, blind["batches"]["sampled"])) == 2
    for contract in (selection, blind):
        for mode, batches in contract["batches"].items():
            returns = 1 if mode == "greedy" else contract["sampled_per_prompt"]
            assert all(len(batch) * returns <= 8 for batch in batches)


def _summary(step, reward, delivery=1.0, cap=0.0, exact=0.5, direction=0.5):
    return {
        "checkpoint_step": step,
        "metrics": {
            "reward_mean": reward,
            "delivery_valid_rate": delivery,
            "cap_rate": cap,
            "periodic_tail_count": 0,
            "exact_rate": exact,
            "direction_correct_rate": direction,
        },
    }


def test_selection_rank_prefers_gate_then_reward_then_earlier_step():
    failed_high_reward = _summary(468, 0.9, delivery=0.5)
    passed_lower_reward = _summary(450, 0.4)
    assert probe.selection_rank_key(passed_lower_reward) > probe.selection_rank_key(
        failed_high_reward
    )
    early = _summary(410, 0.4)
    late = _summary(450, 0.4)
    assert probe.selection_rank_key(early) > probe.selection_rank_key(late)


def test_final_rank_requires_blind_and_all_direction_retention():
    blind = {
        "reward_mean": 0.5,
        "delivery_valid_rate": 1.0,
        "cap_rate": 0.0,
        "periodic_tail_count": 0,
        "exact_rate": 0.5,
    }
    combined = {"reward_mean": 0.45, "exact_rate": 0.5}
    retention = {
        "reward_mean": 0.5,
        "directions": {
            direction: {"direction_correct_rate": 0.5}
            for direction in ("hold", "hike", "cut")
        },
    }
    assert probe.final_rank_key(
        step=410, blind=blind, combined=combined, retention=retention
    )[0] == 1
    retention["directions"]["cut"]["direction_correct_rate"] = 0.0
    assert probe.final_rank_key(
        step=410, blind=blind, combined=combined, retention=retention
    )[0] == 0
