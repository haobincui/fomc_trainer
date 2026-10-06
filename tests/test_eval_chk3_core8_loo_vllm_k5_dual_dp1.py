from __future__ import annotations

from pathlib import Path

import pytest

from jobs.eval import eval_chk3_core8_loo_vllm_k5_dual_dp1 as runner
from jobs.eval import eval_chk3_core8_loo_stochastic_k10 as k10
from jobs.eval import orchestrate_chk3_core8_loo_vllm_k5_dual_dp1 as orchestrator

COHORT_SHA256 = "26c1a06231fd8b10a0ccf2a9de22a59b7d657696da88a18981bad6fb82762e1c"
POLICY_SHA256 = "611b25af74ac21409071ab10ce08574d2afe9aaf38b920d9f40e02b3febbbde3"


def _source_samples() -> list[dict]:
    manifest, _rows, _tokenizer, _observed = k10._load_inputs(
        k10.DEFAULT_SAMPLE_MANIFEST,
        k10.DEFAULT_SAMPLE_MANIFEST_SHA256,
        load_tokenizer=False,
    )
    return manifest["samples"]


def _ledger(samples: list[dict]) -> list[dict]:
    return [
        {
            "absolute_prompt_index": index,
            "sample_id": sample["sample_id"],
            "meeting_id": sample["meeting_id"],
            "prompt_token_count": sample["prompt_token_count"],
            "prompt_token_ids": [1, 2, 3],
            "prompt_token_ids_sha256": "synthetic",
        }
        for index, sample in enumerate(samples)
    ]


def test_frozen_k5_sampling_contract_excludes_greedy_k1() -> None:
    contract = runner.sampling_contract(max_num_seqs=16)
    assert runner.REPLICATE_SEEDS == (
        20260811,
        21260811,
        22260811,
        23260811,
        24260811,
    )
    assert contract["max_new_tokens"] == 2560
    assert contract["temperature"] == 0.6
    assert contract["top_p"] == 0.95
    assert contract["top_k"] == 50
    assert contract["max_num_seqs_per_worker"] == 16
    assert contract["gpu_memory_utilization_per_worker"] == 0.95
    assert contract["historical_greedy_k1_reused_as_replicate"] is False
    assert contract["five_fresh_same_distribution_stochastic_replicates"] is True
    assert contract["token_identity_replay_gate"] is False
    with pytest.raises(runner.Core8LooVllmK5Error, match="requires max_num_seqs=16"):
        runner.sampling_contract(max_num_seqs=12)


def test_sealed_v2_policy_and_cohort_round_trip() -> None:
    policy = runner.load_execution_policy()
    assert policy["receipt_binding"]["sha256"] == POLICY_SHA256
    cohort, ledger, manifest, bound = runner.load_cohort(
        runner.DEFAULT_COHORT,
        COHORT_SHA256,
    )
    assert cohort["execution_policy"] == policy["receipt_binding"]
    assert len(ledger) == len(manifest["samples"]) == len(bound) == 2176
    cases = runner.canonical_cases(
        samples=manifest["samples"],
        ledger=ledger,
        smoke=False,
    )
    assert len(cases) == 10880
    for case in cases:
        prompt_index = int(case["absolute_case_index"]) // 5
        replicate_id = int(case["absolute_case_index"]) % 5
        meeting_index = prompt_index // 17
        assert int(case["variant_rank"]) == prompt_index % 17
        assert int(case["replicate_id"]) == replicate_id
        assert runner.assigned_shard(int(case["absolute_case_index"])) == (
            meeting_index * 5 + replicate_id
        ) % 2


def test_full_case_matrix_and_paired_block_sharding() -> None:
    samples = _source_samples()
    cases = runner.canonical_cases(
        samples=samples,
        ledger=_ledger(samples),
        smoke=False,
    )
    assert len(cases) == runner.CASES_PER_MODEL == 10880
    shards = [runner.shard_cases(cases, shard) for shard in (0, 1)]
    assert [len(values) for values in shards] == [5440, 5440]
    assert {
        int(case["absolute_case_index"])
        for values in shards
        for case in values
    } == set(range(10880))

    for meeting_rank in range(128):
        meeting = [
            case for case in cases if int(case["meeting_rank"]) == meeting_rank
        ]
        assert len(meeting) == 85
        for replicate_id in range(5):
            block = [
                case
                for case in meeting
                if int(case["replicate_id"]) == replicate_id
            ]
            assert len(block) == 17
            assert len({case["row_seed"] for case in block}) == 1
            assert len({case["paired_seed_key"] for case in block}) == 1
            assert len(
                {
                    runner.assigned_shard(int(case["absolute_case_index"]))
                    for case in block
                }
            ) == 1
            assert runner.assigned_shard(
                int(block[0]["absolute_case_index"])
            ) == (meeting_rank * 5 + replicate_id) % 2


def test_two_meeting_smoke_is_170_and_85_per_card() -> None:
    samples = _source_samples()
    cases = runner.canonical_cases(
        samples=samples,
        ledger=_ledger(samples),
        smoke=True,
    )
    assert len(cases) == 170
    assert {case["meeting_rank"] for case in cases} == {0, 1}
    assert [len(runner.shard_cases(cases, shard)) for shard in (0, 1)] == [85, 85]
    assert [
        len(runner._chunk_indexes(0, shard, total_cases=170))
        for shard in (0, 1)
    ] == [85, 85]


def test_resume_plan_never_redispatches_durable_tuple_keys() -> None:
    samples = _source_samples()
    cases = runner.canonical_cases(
        samples=samples,
        ledger=_ledger(samples),
        smoke=True,
    )
    shard_id = 0
    expected = {
        int(case["absolute_case_index"])
        for case in runner.shard_cases(cases, shard_id)
    }
    durable = set(sorted(expected)[:11])
    receipt_only, pending = runner._resume_dispatch_plan(
        cases=cases,
        durable_indexes=durable,
        committed_receipt_count=0,
        shard_id=shard_id,
    )
    dispatched = {
        int(case["absolute_case_index"])
        for _chunk_id, chunk in pending
        for case in chunk
    }
    assert receipt_only == []
    assert durable.isdisjoint(dispatched)
    assert durable | dispatched == expected


def test_profile_replaces_shared_beta_data_and_persistence_contract() -> None:
    original_id = runner.shared.EVALUATION_ID
    with runner.configured_shared_implementation():
        assert runner.shared.EVALUATION_ID == runner.EVALUATION_ID
        assert runner.shared.MODEL_ORDER == ("chk3",)
        assert runner.shared.ABSOLUTE_CHUNK_SIZE == 170
        assert runner.shared.CASES_PER_SHARD_CHUNK == 85
        persistence = runner.shared._persistence_contract()
        assert persistence["append_flush_fsync_per_completed_request"] is True
        assert persistence["chunk_receipt_after_exact_85_key_union_only"] is True
        assert persistence["token_identity_replay_required"] is False
    assert runner.shared.EVALUATION_ID == original_id


def test_orchestrator_worker_is_single_gpu_and_uses_new_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("VLLM_DP_RANK", "7")
    monkeypatch.setenv("NCCL_COMM_ID", "unsafe")
    env = orchestrator._worker_environment(
        physical_gpu_index=1,
        lock_fd=17,
        run_token="a" * 64,
    )
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert not any(key.startswith("VLLM_DP_") for key in env)
    assert not any(key.startswith("NCCL_") for key in env)
    command = orchestrator._runner_command(
        python_bin=Path("/python"),
        model_id="chk3",
        shard_id=1,
        cohort=Path("/cohort.json"),
        cohort_sha256="b" * 64,
        output_dir=Path("/output/shard1"),
        scope="infrastructure_smoke",
        max_num_seqs=16,
        control_dir=Path("/control"),
        run_token="a" * 64,
        lock_fd=17,
        resume=False,
        gpu_wait_timeout_seconds=10,
        gpu_poll_seconds=1,
    )
    assert orchestrator.RUNNER_MODULE in command
    assert "--smoke" in command


def test_cohort_rejects_runtime_model_or_tokenizer_inventory_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort_sha = runner.shared.core._sha256_file(runner.DEFAULT_COHORT)
    monkeypatch.setattr(
        runner.preparation,
        "load_model_runtime_inventory",
        lambda manifest: {"path": "/drifted-runtime"},
    )
    with pytest.raises(
        runner.Core8LooVllmK5Error,
        match="runtime model/tokenizer inventory drift",
    ):
        runner.load_cohort(runner.DEFAULT_COHORT, cohort_sha)
