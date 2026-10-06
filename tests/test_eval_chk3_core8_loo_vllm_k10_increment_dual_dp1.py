from __future__ import annotations

import asyncio
import hashlib

from jobs.eval import eval_chk3_core8_loo_stochastic_k10 as source
from jobs.eval import eval_chk3_core8_loo_vllm_k10_increment_dual_dp1 as runner


def _source_samples() -> list[dict]:
    manifest, _rows, _tokenizer, _observed = source._load_inputs(
        source.DEFAULT_SAMPLE_MANIFEST,
        source.DEFAULT_SAMPLE_MANIFEST_SHA256,
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
            "prompt_token_ids_sha256": "fixture",
        }
        for index, sample in enumerate(samples)
    ]


def test_increment_sampling_contract_uses_only_global_ids_five_through_nine() -> None:
    contract = runner.sampling_contract(max_num_seqs=16)
    assert runner.CASES_PER_MODEL == 10_880
    assert runner.REPLICATE_SEEDS == (
        25260811,
        26260811,
        27260811,
        28260811,
        29260811,
    )
    assert contract["replicate_indexing"]["local_replicate_ids"] == [0, 1, 2, 3, 4]
    assert contract["replicate_indexing"]["global_replicate_ids"] == [5, 6, 7, 8, 9]
    assert "2026-08-24" in contract["increment_authorization"]


def test_full_case_matrix_preserves_local_sharding_and_adds_global_identity() -> None:
    samples = _source_samples()
    cases = runner.canonical_cases(
        samples=samples,
        ledger=_ledger(samples),
        smoke=False,
    )
    assert len(cases) == 10_880
    assert {int(case["local_replicate_id"]) for case in cases} == set(range(5))
    assert {int(case["global_replicate_id"]) for case in cases} == set(range(5, 10))
    for case in cases:
        local_id = int(case["replicate_id"])
        assert int(case["local_replicate_id"]) == local_id
        assert int(case["global_replicate_id"]) == local_id + 5
        assert runner.assigned_shard(int(case["absolute_case_index"])) == (
            int(case["meeting_rank"]) * 5 + local_id
        ) % 2
    assert [len(runner.shard_cases(cases, shard)) for shard in (0, 1)] == [5440, 5440]


def test_two_meeting_smoke_remains_170_rows_and_85_per_worker() -> None:
    samples = _source_samples()
    cases = runner.canonical_cases(
        samples=samples,
        ledger=_ledger(samples),
        smoke=True,
    )
    assert len(cases) == 170
    assert [len(runner.shard_cases(cases, shard)) for shard in (0, 1)] == [85, 85]


def test_shared_runtime_configuration_is_scoped_and_uses_new_schemas() -> None:
    original_id = runner.frozen.shared.EVALUATION_ID
    with runner.configured_shared_implementation():
        assert runner.frozen.shared.EVALUATION_ID == runner.EVALUATION_ID
        assert runner.frozen.shared.ROW_SCHEMA == runner.ROW_SCHEMA
        assert runner.frozen.shared.CASES_PER_MODEL == 10_880
        assert runner.frozen.shared.build_result is runner.build_result
        assert runner.frozen.shared.validate_result is runner.validate_result
    assert runner.frozen.shared.EVALUATION_ID == original_id


def test_source_hash_profile_replaces_k5_launcher_and_orchestrator_bindings() -> None:
    sources = runner._binding
    assert sources(runner.ROOT / "jobs/eval/orchestrate_chk3_core8_loo_vllm_k10_increment_dual_dp1.py")["sha256"]
    assert sources(runner.ROOT / "run/eval_chk3_core8_loo_vllm_k10_increment_dual_dp1.sh")["sha256"]
    replacements = runner._shared_replacements()
    assert replacements["_source_hashes"] is runner._source_hashes


def test_already_sealed_increment_preparation_round_trips_without_k5_default_leakage() -> None:
    cohort_sha256 = hashlib.sha256(runner.DEFAULT_COHORT.read_bytes()).hexdigest()
    policy = runner.load_execution_policy()
    cohort, ledger, manifest, bound = runner.load_cohort(
        runner.DEFAULT_COHORT,
        cohort_sha256,
    )
    assert policy["manifest"]["evaluation_id"] == runner.EVALUATION_ID
    assert cohort["evaluation_id"] == runner.EVALUATION_ID
    assert len(ledger) == len(manifest["samples"]) == len(bound) == 2_176
    assert cohort["replicate_indexing"]["global_replicate_ids"] == [5, 6, 7, 8, 9]


def test_overlapping_async_requests_never_mutate_or_leak_frozen_globals(
    monkeypatch,
) -> None:
    original_evaluation_id = runner.frozen.EVALUATION_ID
    both_entered = asyncio.Event()
    entered = 0

    async def overlapping_consume(engine, sampling_params_cls, case):
        nonlocal entered
        assert runner.frozen.EVALUATION_ID == original_evaluation_id
        entered += 1
        if entered == 2:
            both_entered.set()
        await both_entered.wait()
        # Force task 0 to leave before task 1.  The old nested global-patching
        # implementation then restored in non-LIFO order and leaked K10 state.
        if case["case_id"] == 1:
            await asyncio.sleep(0.01)
        assert runner.frozen.EVALUATION_ID == original_evaluation_id
        return case, case["case_id"]

    monkeypatch.setattr(runner, "_FROZEN_CONSUME_REQUEST", overlapping_consume)

    async def exercise_overlap():
        return await asyncio.gather(
            runner._consume_request(None, None, {"case_id": 0}),
            runner._consume_request(None, None, {"case_id": 1}),
        )

    assert asyncio.run(exercise_overlap()) == [
        ({"case_id": 0}, 0),
        ({"case_id": 1}, 1),
    ]
    assert runner.frozen.EVALUATION_ID == original_evaluation_id
