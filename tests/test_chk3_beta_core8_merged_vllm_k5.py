from __future__ import annotations

import asyncio
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jobs.eval import eval_chk3_beta_core8_merged_vllm_k5 as runner
from jobs.eval import prepare_chk3_beta_core8_merged_vllm_k5 as preparer
from open_r1.validator.loo_generation_spec import derive_row_seed


TEST_MAX_NUM_SEQS = 12


def _sample(index: int, *, prompt: str = "prompt") -> dict:
    result = {
        "sample_id": f"sample-{index}",
        "meeting_id": "2000-01-02",
        "length_bucket": "short",
        "prompt_token_count": 3,
        "prompt_sha256": runner._sha256_text(prompt),
        "analysis_reference_exact_identity": True,
        "normalized_identity": True,
        "punctuation_insensitive_identity": True,
    }
    result.update({key: None for key in runner.k10_profile.PROFILE_METADATA_KEYS})
    result.update(
        {
            "era": "pre2009_external",
            "source_split": "external_holdout",
            "meeting_type": "regular",
            "sensitivity_flag": False,
            "scheduled": True,
            "meeting_start_date": "2000-01-01",
            "meeting_end_date": "2000-01-02",
            "evidence_cutoff": "1999-12-31",
            "topic": runner.data_contract.CORE_TOPICS[index % 8],
            "topic_order": index % 8,
            "cp318_selection_exposed": False,
            "source_sample_id": f"source-{index}",
            "source_id": f"id-{index}",
            "source_core8_line_number": index + 1,
            "prompt_sha256": runner._sha256_text(prompt),
            "source_analysis_sha256": "a" * 64,
            "reference_minutes_sha256": "b" * 64,
            "topic_evidence_sha256": "c" * 64,
            "source_release": {},
            "research_scope": runner.k10_profile.RESEARCH_SCOPE,
            "transport_split_role": runner.k10_profile.TRANSPORT_SPLIT_ROLE,
            "not_all_held_out": True,
        }
    )
    return result


def _ledger(index: int) -> dict:
    token_ids = [128000, 10 + index, 11 + index]
    return {
        "absolute_prompt_index": index,
        "prompt_token_count": len(token_ids),
        "prompt_token_ids": token_ids,
        "prompt_token_ids_sha256": runner._sha256_text(runner._canonical(token_ids)),
    }


def _build_fixture() -> tuple[dict, dict, dict]:
    analysis = "Inflation was 2 percent in January 2020."
    prompt = runner.core.native_eval.native_probe.USER_PROMPT_PREFIX + json.dumps(
        {"analysis": analysis}
    )
    source = {
        "prompt": prompt,
        "response": "Reference reasoning.\n</think>\n" + analysis,
    }
    sample = _sample(0, prompt=prompt)
    sample["source_analysis_sha256"] = runner._sha256_text(analysis)
    token_entry = _ledger(0)
    token_entry["prompt_token_count"] = sample["prompt_token_count"]
    case = {
        "sample": sample,
        "token_entry": token_entry,
        "replicate_id": 0,
        "replicate_seed": runner.REPLICATE_SEEDS[0],
        "row_seed": derive_row_seed(runner.REPLICATE_SEEDS[0], sample["sample_id"]),
        "absolute_case_index": 0,
        "absolute_chunk_id": 0,
    }
    return source, sample, case


def _result() -> tuple[dict, dict, dict]:
    source, sample, case = _build_fixture()
    row = runner.build_result(
        model_id="chk1",
        model_label=runner.MODEL_LABELS["chk1"],
        case=case,
        source_row=source,
        sample_manifest_sha256="d" * 64,
        source_artifact_sha256s={"cohort": "e" * 64},
        generated_text=(
            "Reasoning.\n</think>\nInflation was 2 percent in January 2020."
        ),
        generated_token_ids=[99, 128001],
        eos_token_ids=[128001],
        pad_token_id=128001,
        raw_finish_reason="stop",
        raw_stop_reason=None,
        max_num_seqs=TEST_MAX_NUM_SEQS,
    )
    return row, source, case


def test_sampling_contract_is_unquantized_bf16_async_v1() -> None:
    contract = runner.sampling_contract(max_num_seqs=TEST_MAX_NUM_SEQS)
    assert runner.ABSOLUTE_CHUNK_SIZE == 40
    assert runner.CASES_PER_MODEL == 10_240
    assert runner.CHUNKS_PER_MODEL == 256
    assert contract["backend"] == "vllm-async-engine-v1-continuous-batching"
    assert contract["dtype"] == "bfloat16"
    assert contract["quantization"] is None
    assert contract["max_num_seqs"] == TEST_MAX_NUM_SEQS
    assert contract["max_num_batched_tokens"] == 4096
    assert contract["enable_prefix_caching"] is False
    assert contract["data_parallel_size"] == 2
    assert contract["tensor_parallel_size"] == 1
    assert contract["pipeline_parallel_size"] == 1
    assert contract["gpu_memory_utilization"] == 0.95
    assert contract["enforce_eager"] is True
    assert contract["cuda_graphs"] is False
    assert contract["async_output_processing"] is False
    assert contract["required_environment"] == runner.REQUIRED_VLLM_ENV
    with pytest.raises(runner.VllmK5GenerationError, match="8, 12, 16"):
        runner.sampling_contract(max_num_seqs=10)


def test_v1_engine_args_omit_unsupported_disable_async_output_proc() -> None:
    source = inspect.getsource(runner._run_async_generation)
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AsyncEngineArgs"
    ]
    assert len(calls) == 1
    keywords = {keyword.arg for keyword in calls[0].keywords}
    assert "disable_async_output_proc" not in keywords
    assert "enforce_eager" in keywords
    # V1/DP2/eager must still prove that the resolved model config disabled
    # async output processing after engine construction.
    assert "model_config.use_async_output_proc is not False" in source


def test_smoke_is_full_core8_meeting_and_formal_pairs_all_five_seeds() -> None:
    samples = [_sample(index) for index in range(8)]
    ledger = [_ledger(index) for index in range(8)]
    smoke = runner.canonical_cases(samples=samples, ledger=ledger, smoke=True)
    assert len(smoke) == 40
    assert [case["sample"]["topic_order"] for case in smoke[::5]] == list(range(8))
    assert {case["replicate_seed"] for case in smoke} == set(runner.REPLICATE_SEEDS)

    formal = runner.canonical_cases(samples=samples, ledger=ledger, smoke=False)
    assert len(formal) == 40
    for sample_index in range(8):
        block = formal[sample_index * 5 : sample_index * 5 + 5]
        assert [case["replicate_seed"] for case in block] == list(
            runner.REPLICATE_SEEDS
        )
        assert [case["absolute_case_index"] for case in block] == list(
            range(sample_index * 5, sample_index * 5 + 5)
        )
        assert all(
            case["row_seed"]
            == derive_row_seed(case["replicate_seed"], f"sample-{sample_index}")
            for case in block
        )


def test_finish_reason_requires_retained_final_eos_and_no_explicit_stop() -> None:
    assert (
        runner._normalize_finish_reason(
            raw_finish_reason="stop",
            raw_stop_reason=None,
            generated_token_ids=[7, 128001],
            eos_token_ids=[128001],
        )
        == "eos"
    )
    assert (
        runner._normalize_finish_reason(
            raw_finish_reason="length",
            raw_stop_reason=None,
            generated_token_ids=[7] * runner.MAX_NEW_TOKENS,
            eos_token_ids=[128001],
        )
        == "length"
    )
    with pytest.raises(runner.VllmK5GenerationError, match="final EOS"):
        runner._normalize_finish_reason(
            raw_finish_reason="stop",
            raw_stop_reason=None,
            generated_token_ids=[7],
            eos_token_ids=[128001],
        )
    with pytest.raises(runner.VllmK5GenerationError, match="explicit stop"):
        runner._normalize_finish_reason(
            raw_finish_reason="stop",
            raw_stop_reason=128001,
            generated_token_ids=[7, 128001],
            eos_token_ids=[128001],
        )


def test_completion_order_can_vary_only_inside_active_absolute_chunk() -> None:
    full_first = set(range(40))
    partial_second = {41, 47, 55}
    assert runner._completion_status(
        full_first | partial_second, expected_cases=80
    ) == (1, partial_second)
    with pytest.raises(runner.VllmK5GenerationError, match="skips"):
        runner._completion_status(full_first | {81}, expected_cases=120)


def test_canonical_wal_scan_supports_one_row_ahead_prefix(tmp_path: Path) -> None:
    path = tmp_path / "wal.jsonl"
    first = {"absolute_case_index": 1, "value": "first"}
    second = {"absolute_case_index": 0, "value": "second"}
    encoded_first = runner._encoded_row(first)
    path.write_bytes(encoded_first + runner._encoded_row(second))
    rows, tracker, prefix = runner._read_canonical_jsonl(path, recorded_prefix_rows=1)
    assert rows == [first, second]
    assert tracker.rows == 2
    assert prefix == {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(encoded_first).hexdigest(),
        "bytes": len(encoded_first),
        "rows": 1,
    }
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(runner.VllmK5GenerationError, match="incomplete tail"):
        runner._read_canonical_jsonl(path)


def test_build_and_deep_validate_full_result_and_detect_tampering() -> None:
    row, source, case = _result()
    assert row["finish_reason"] == "eos"
    assert row["answer"] == "Inflation was 2 percent in January 2020."
    assert row["generation_metrics"] == {
        "structure_delivery": True,
        "numeric_fidelity": True,
        "date_fidelity": True,
        "degeneration_free": True,
    }
    kwargs = {
        "model_id": "chk1",
        "model_label": runner.MODEL_LABELS["chk1"],
        "case": case,
        "source_row": source,
        "sample_manifest_sha256": "d" * 64,
        "source_artifact_sha256s": {"cohort": "e" * 64},
        "max_num_seqs": TEST_MAX_NUM_SEQS,
    }
    runner.validate_result(row, **kwargs)
    tampered = copy.deepcopy(row)
    tampered["answer"] += " altered"
    with pytest.raises(runner.VllmK5GenerationError, match="drift"):
        runner.validate_result(tampered, **kwargs)
    wrong_scheduler = {**kwargs, "max_num_seqs": 8}
    with pytest.raises(runner.VllmK5GenerationError, match="drift"):
        runner.validate_result(row, **wrong_scheduler)


def test_completion_order_rows_canonicalize_and_receipt_bindings_close() -> None:
    row, source, case = _result()
    cases = []
    bound_rows = {}
    unordered = []
    for index in range(3):
        next_case = copy.deepcopy(case)
        next_case["absolute_case_index"] = index
        next_case["sample"] = copy.deepcopy(case["sample"])
        next_case["sample"]["sample_id"] = f"sample-{index}"
        next_case["row_seed"] = derive_row_seed(
            next_case["replicate_seed"], f"sample-{index}"
        )
        cases.append(next_case)
        bound_rows[f"sample-{index}"] = source
        next_row = runner.build_result(
            model_id="chk1",
            model_label=runner.MODEL_LABELS["chk1"],
            case=next_case,
            source_row=source,
            sample_manifest_sha256="d" * 64,
            source_artifact_sha256s={"cohort": "e" * 64},
            generated_text=row["generated_text"],
            generated_token_ids=row["generated_token_ids"],
            eos_token_ids=row["eos_token_ids"],
            pad_token_id=row["pad_token_id"],
            raw_finish_reason="stop",
            raw_stop_reason=None,
            max_num_seqs=TEST_MAX_NUM_SEQS,
        )
        unordered.append(next_row)
    unordered = [unordered[2], unordered[0], unordered[1]]
    by_index = runner.validate_wal_rows(
        unordered,
        cases=cases,
        model_id="chk1",
        model_label=runner.MODEL_LABELS["chk1"],
        bound_rows=bound_rows,
        sample_manifest_sha256="d" * 64,
        source_artifact_sha256s={"cohort": "e" * 64},
        max_num_seqs=TEST_MAX_NUM_SEQS,
    )
    assert list(sorted(by_index)) == [0, 1, 2]
    receipt = runner._chunk_receipt(
        model_id="chk1", chunk_id=0, rows_by_index=by_index, expected_cases=3
    )
    runner._validate_receipts(
        [receipt], model_id="chk1", rows_by_index=by_index, expected_cases=3
    )
    receipt["canonical_row_bindings_sha256"] = "0" * 64
    with pytest.raises(runner.VllmK5GenerationError, match="receipt drift"):
        runner._validate_receipts(
            [receipt], model_id="chk1", rows_by_index=by_index, expected_cases=3
        )


def test_token_ledger_materializes_ids_once(monkeypatch: pytest.MonkeyPatch) -> None:
    class Tokenizer:
        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt, **kwargs
        ):
            assert tokenize is True
            assert add_generation_prompt is True
            assert kwargs == {"truncation": False, "return_dict": False}
            assert messages[0]["role"] == "system"
            return [128000, 42, 43]

    monkeypatch.setattr(preparer, "EXPECTED_PROMPTS", 1)
    prompt = "plain prompt"
    sample = {
        "sample_id": "sample-0",
        "meeting_id": "2000-01-02",
        "prompt_sha256": runner._sha256_text(prompt),
        "prompt_token_count": 3,
    }
    rows = preparer.build_token_ledger(
        sample_manifest={"samples": [sample]},
        bound_rows={"sample-0": {"prompt": prompt}},
        tokenizer=Tokenizer(),
        system_prompt="system",
        user_prompt_suffix=None,
    )
    assert rows[0]["prompt_token_ids"] == [128000, 42, 43]
    assert rows[0]["prompt_token_ids_sha256"] == runner._sha256_text(
        runner._canonical([128000, 42, 43])
    )


def test_required_vllm_environment_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in runner.REQUIRED_VLLM_ENV.items():
        monkeypatch.setenv(key, value)
    runner._assert_required_environment(require_cuda=False)
    monkeypatch.setenv("VLLM_USE_V1", "0")
    with pytest.raises(runner.VllmK5GenerationError, match="VLLM_USE_V1"):
        runner._assert_required_environment(require_cuda=False)


def test_cuda_environment_requires_exact_dual_physical_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in runner.REQUIRED_VLLM_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    runner._assert_required_environment(require_cuda=True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    with pytest.raises(runner.VllmK5GenerationError, match="exact order"):
        runner._assert_required_environment(require_cuda=True)


def _fake_engine(*pids: int) -> SimpleNamespace:
    core_engines = []
    for rank, pid in enumerate(pids):
        process = SimpleNamespace(pid=pid, is_alive=lambda: True)
        core_engines.append(
            SimpleNamespace(
                index=rank,
                proc_handle=SimpleNamespace(proc=process),
            )
        )
    return SimpleNamespace(engine_core=SimpleNamespace(core_engines=core_engines))


def test_post_load_gate_proves_distinct_engine_core_pid_per_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _fake_engine(101, 202)
    monkeypatch.setattr(runner, "_descendant_pids", lambda: {1, 101, 202})
    monkeypatch.setattr(
        runner,
        "_external_gpu_compute_processes",
        lambda index: [
            {
                "physical_gpu_index": index,
                "pid": (101, 202)[index],
                "process_name": f"EngineCore_{index}",
            }
        ],
    )
    evidence = runner._verify_dual_engine_gpu_processes(
        engine, timeout_seconds=0, poll_seconds=0
    )
    assert [row["engine_core_pid"] for row in evidence] == [101, 202]

    monkeypatch.setattr(
        runner,
        "_external_gpu_compute_processes",
        lambda index: [
            {
                "physical_gpu_index": index,
                "pid": 101,
                "process_name": "wrong-rank",
            }
        ],
    )
    with pytest.raises(runner.VllmK5GenerationError, match="could not prove"):
        runner._verify_dual_engine_gpu_processes(
            engine, timeout_seconds=0, poll_seconds=0
        )


def test_dual_lock_releases_gpu0_when_gpu1_is_initially_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner,
        "gpu_lock_path",
        lambda index: tmp_path / f"gpu{index}.lock",
    )
    monkeypatch.setattr(runner, "_external_gpu_compute_processes", lambda index: [])
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    operations: list[int] = []
    exclusive_attempts = 0

    def fake_flock(descriptor: int, operation: int) -> None:
        nonlocal exclusive_attempts
        operations.append(operation)
        if operation == runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB:
            exclusive_attempts += 1
            if exclusive_attempts == 2:
                raise BlockingIOError

    monkeypatch.setattr(runner.fcntl, "flock", fake_flock)
    with runner.exclusive_dual_gpu_lease(timeout_seconds=1, poll_seconds=1) as lease:
        assert lease["physical_gpu_indexes"] == [0, 1]
    assert operations[:3] == [
        runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB,
        runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB,
        runner.fcntl.LOCK_UN,
    ]


def test_sealed_runtime_binds_dual_gpu_and_positive_benchmark_timing() -> None:
    identities = [
        {
            "physical_gpu_index": index,
            "uuid": f"GPU-{index}",
            "pci_bus_id": f"00000000:0{index}:00.0",
        }
        for index in range(2)
    ]
    launch_runtime = {
        "vllm": runner.EXPECTED_VLLM_VERSION,
        "cuda_visible_devices": "0,1",
        "cuda_device_order": "PCI_BUS_ID",
        "physical_gpu_indexes": [0, 1],
        "physical_gpu_identities": identities,
        "gpu_lock_paths": [
            str(runner.gpu_lock_path(index).resolve()) for index in range(2)
        ],
        "max_num_seqs": TEST_MAX_NUM_SEQS,
    }
    output_tokens = 400
    seconds = 2.0
    engine_runtime = {
        "vllm": runner.EXPECTED_VLLM_VERSION,
        "visible_cuda_devices": [
            {"logical_cuda_index": index, **identity}
            for index, identity in enumerate(identities)
        ],
        "data_parallel_master_ip": runner.DATA_PARALLEL_MASTER_IP,
        "data_parallel_master_port": 32123,
        "data_parallel_master_port_policy": (runner.DATA_PARALLEL_MASTER_PORT_POLICY),
        "distributed_world_backend": "nccl",
        "data_parallel_control_backend": "gloo_cpu",
        "dense_model_per_layer_tensor_collectives": False,
        "data_parallel_request_routing": "least_inflight_requests",
        "enforce_eager": True,
        "cuda_graphs": False,
        "async_output_processing": False,
        "engine": runner.sampling_contract(max_num_seqs=TEST_MAX_NUM_SEQS),
        "engine_core_gpu_bindings": [
            {
                "physical_gpu_index": index,
                "data_parallel_rank": index,
                "engine_core_pid": 100 + index,
                "descendant_compute_processes": [
                    {
                        "physical_gpu_index": index,
                        "pid": 100 + index,
                        "process_name": f"EngineCore_{index}",
                    }
                ],
            }
            for index in range(2)
        ],
        "gpu_lease": {
            "physical_gpu_indexes": [0, 1],
            "lock_paths": launch_runtime["gpu_lock_paths"],
            "external_compute_pids_at_acquire": [],
            "acquired_at_utc": "2026-08-16T00:00:00Z",
            "wait_seconds": 0.1,
        },
        "model_load_wall_seconds": 3.0,
        "generation_wall_seconds": seconds,
        "generated_output_tokens": output_tokens,
        "timed_session_generated_output_tokens": output_tokens,
        "completed_requests_in_timed_session": 40,
        "output_tokens_per_second": output_tokens / seconds,
        "timing_scope": "current_engine_session_after_dp_pid_validation",
        "preemption_count": None,
        "preemption_count_status": "unavailable_vllm_async_public_api",
        "speed_measurement_valid_for_candidate_selection": True,
    }
    runner._validate_sealed_runtime(
        launch_runtime=launch_runtime,
        engine_runtime=engine_runtime,
        max_num_seqs=TEST_MAX_NUM_SEQS,
        expected_cases=40,
        expected_generated_output_tokens=output_tokens,
        resume_count=0,
    )
    engine_runtime["output_tokens_per_second"] += 1
    with pytest.raises(runner.VllmK5GenerationError, match="timing"):
        runner._validate_sealed_runtime(
            launch_runtime=launch_runtime,
            engine_runtime=engine_runtime,
            max_num_seqs=TEST_MAX_NUM_SEQS,
            expected_cases=40,
            expected_generated_output_tokens=output_tokens,
            resume_count=0,
        )


def test_async_request_consumes_exact_ids_and_per_row_seed() -> None:
    captured = {}

    class Params:
        def __init__(self, **kwargs):
            captured["params"] = kwargs

    class Output:
        finished = True

    class Engine:
        async def generate(self, prompt, params, request_id):
            captured.update(
                {"prompt": prompt, "params_object": params, "request_id": request_id}
            )
            yield Output()

    case = {
        "absolute_case_index": 7,
        "row_seed": 123456,
        "token_entry": {"prompt_token_ids": [128000, 42, 128014]},
    }
    observed_case, output = asyncio.run(runner._consume_request(Engine(), Params, case))
    assert observed_case is case
    assert output.finished is True
    assert captured["prompt"] == {"prompt_token_ids": [128000, 42, 128014]}
    assert captured["request_id"] == "case-00007-seed-123456"
    assert captured["params"]["seed"] == 123456
    assert captured["params"]["n"] == 1
    assert captured["params"]["skip_special_tokens"] is False
    assert "best_of" not in captured["params"]


@pytest.mark.parametrize(
    "command,arguments",
    [
        (
            "run-model",
            [
                "--model-id",
                "chk1",
                "--cohort",
                "cohort.json",
                "--cohort-sha256",
                "a" * 64,
                "--output-dir",
                "output",
            ],
        ),
        (
            "validate-run",
            [
                "--manifest",
                "manifest.json",
                "--cohort",
                "cohort.json",
                "--cohort-sha256",
                "a" * 64,
                "--model-id",
                "chk1",
                "--scope",
                "infrastructure_smoke",
            ],
        ),
        (
            "seal-suite",
            [
                "--cohort",
                "cohort.json",
                "--cohort-sha256",
                "a" * 64,
                "--output-dir",
                "output",
                "--scope",
                "infrastructure_smoke",
            ],
        ),
        (
            "validate-suite",
            [
                "--manifest",
                "manifest.json",
                "--cohort",
                "cohort.json",
                "--cohort-sha256",
                "a" * 64,
                "--scope",
                "infrastructure_smoke",
            ],
        ),
    ],
)
def test_every_cli_stage_requires_and_freezes_max_num_seqs(
    command: str, arguments: list[str]
) -> None:
    parser = runner._parser()
    with pytest.raises(SystemExit):
        parser.parse_args([command, *arguments])
    parsed = parser.parse_args(
        [command, *arguments, "--max-num-seqs", str(TEST_MAX_NUM_SEQS)]
    )
    assert parsed.max_num_seqs == TEST_MAX_NUM_SEQS
