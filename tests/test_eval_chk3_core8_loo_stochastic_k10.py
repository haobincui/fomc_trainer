from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from jobs.eval import eval_chk3_core8_loo_stochastic_k10 as subject
from open_r1.validator.loo_generation_spec import derive_row_seed


def _inputs() -> tuple[dict, list[dict], str]:
    manifest, rows, tokenizer, observed = subject._load_inputs(
        subject.DEFAULT_SAMPLE_MANIFEST,
        subject.DEFAULT_SAMPLE_MANIFEST_SHA256,
        load_tokenizer=False,
    )
    assert tokenizer is None
    return manifest, rows, observed


def test_frozen_input_matrix_and_k10_common_random_number_cases() -> None:
    manifest, rows, _ = _inputs()
    samples = manifest["samples"]
    cases = subject._canonical_cases(samples, smoke=False)
    assert len(samples) == len(rows) == 2176
    assert len(cases) == subject.EXPECTED_CASES == 21760
    assert subject.REPLICATE_SEEDS == tuple(
        range(20260811, 29260812, 1000000)
    )

    first_meeting_cases = cases[: 17 * 10]
    paired_keys = {case[3] for case in first_meeting_cases}
    assert paired_keys == {
        "core8-loo::fomc-19930202-19930203::full::none"
    }
    for replicate_id, replicate_seed in enumerate(subject.REPLICATE_SEEDS):
        arm_cases = [case for case in first_meeting_cases if case[1] == replicate_id]
        assert len(arm_cases) == 17
        seeds = {
            derive_row_seed(replicate_seed, paired_seed_key)
            for _, _, _, paired_seed_key in arm_cases
        }
        assert len(seeds) == 1


def test_sampling_and_gpu_contract_are_frozen() -> None:
    contract = subject._sampling_contract()
    assert contract == {
        "generation_mode": "stochastic_sampling",
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.0,
        "max_new_tokens": 2560,
        "tail_tokens": 1024,
        "batch_size": 1,
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_storage": "bfloat16",
        "bnb_4bit_use_double_quant": True,
        "model_dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "replicate_seeds": list(subject.REPLICATE_SEEDS),
        "row_seed": "derive_row_seed(replicate_seed,paired_seed_key)",
        "paired_seed_key": "meeting_full_sample_id",
        "common_random_numbers": "same-meeting-replicate-across-all-17-arms-v1",
        "rng_reset": (
            "transformers.set_seed+torch.manual_seed+"
            "torch.cuda.manual_seed_all-per-tuple-v1"
        ),
        "canonical_tuple_order": "sealed_sample_order_then_replicate_id",
    }
    assert subject.PHYSICAL_GPU_INDEX == 1
    assert "gpu1" in str(subject.GPU_LOCK)


def test_gpu_environment_rejects_any_exposure_other_than_physical_gpu1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert subject._validate_gpu_environment() == "1"
    for value in ("0", "0,1", "1,0", " 1", ""):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
        with pytest.raises(subject.Core8StochasticK10Error, match="exactly 1"):
            subject._validate_gpu_environment()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    with pytest.raises(subject.Core8StochasticK10Error, match="PCI_BUS_ID"):
        subject._validate_gpu_environment()


def test_logical_cuda0_must_match_physical_gpu1_uuid() -> None:
    expected = "GPU-0d2b981c-7a91-ad4c-7015-89445a0283a1"
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            device_count=lambda: 1,
            get_device_properties=lambda _index: SimpleNamespace(uuid=expected),
        )
    )
    assert subject._verify_logical_cuda0_is_physical_gpu1(
        torch, {"uuid": expected}
    ) == expected
    torch.cuda.get_device_properties = lambda _index: SimpleNamespace(
        uuid="GPU-ba3a7f4c-74c8-4922-364d-031c5187624c"
    )
    with pytest.raises(subject.Core8StochasticK10Error, match="physical GPU1"):
        subject._verify_logical_cuda0_is_physical_gpu1(torch, {"uuid": expected})


def test_gpu1_lease_waits_for_busy_then_two_idle_polls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observations = [
        [{"pid": 123, "process_name": "external"}],
        [],
        [],
    ]
    monkeypatch.setattr(
        subject,
        "_external_gpu1_compute_processes",
        lambda: observations.pop(0),
    )
    monkeypatch.setattr(subject.time, "sleep", lambda _seconds: None)
    with subject.exclusive_gpu1_lease(
        lock_path=tmp_path / "gpu1.lock",
        timeout_seconds=10,
        poll_seconds=1,
    ) as lease:
        assert lease["physical_gpu_index"] == 1
    assert observations == []


def test_post_load_external_gpu1_process_check_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subject,
        "_external_gpu1_compute_processes",
        lambda: [{"pid": 456, "process_name": "intruder"}],
    )
    with pytest.raises(subject.ExternalGpu1ProcessError, match="post_generate"):
        subject._require_no_external_gpu1_processes(phase="post_generate")


def test_final_runtime_evidence_requires_verified_gpu1_uuid_and_lease() -> None:
    physical = {
        "uuid": "GPU-0d2b981c-7a91-ad4c-7015-89445a0283a1",
        "pci_bus_id": "00000000:e1:00.0",
    }
    launch_runtime = {"physical_gpu_identity": physical}
    evidence = {
        "cuda_device_order": "PCI_BUS_ID",
        "cuda_visible_devices": "1",
        "physical_gpu_index": 1,
        "physical_gpu_identity": physical,
        "logical_device": "cuda:0",
        "logical_cuda0_uuid": physical["uuid"],
        "gpu_lease": {
            "physical_gpu_index": 1,
            "lock_path": str(subject.GPU_LOCK.resolve()),
        },
        "torch": "2.10.0+cu128",
        "transformers": "4.57.6",
    }
    subject._validate_runtime_evidence(evidence, launch_runtime)
    tampered = copy.deepcopy(evidence)
    tampered["logical_cuda0_uuid"] = (
        "GPU-ba3a7f4c-74c8-4922-364d-031c5187624c"
    )
    with pytest.raises(subject.Core8StochasticK10Error, match="evidence drift"):
        subject._validate_runtime_evidence(tampered, launch_runtime)


def test_tuple_prefix_rejects_duplicate_gap_and_reordering() -> None:
    manifest, _rows, _ = _inputs()
    cases = subject._canonical_cases(manifest["samples"], smoke=False)
    rows = [
        {
            "model_id": case[0] and subject.MODEL_ID,
            "sample_id": case[0]["sample_id"],
            "replicate_id": case[1],
        }
        for case in cases[:3]
    ]
    subject._validate_tuple_prefix(rows, cases)
    with pytest.raises(subject.Core8StochasticK10Error, match="out-of-order"):
        subject._validate_tuple_prefix([rows[1], rows[0]], cases)
    with pytest.raises(subject.Core8StochasticK10Error, match="duplicate"):
        subject._validate_tuple_prefix([rows[0], copy.deepcopy(rows[0])], cases)
    with pytest.raises(subject.Core8StochasticK10Error, match="count drift"):
        subject._validate_tuple_prefix(rows, cases, complete=True)


def test_resume_state_accepts_only_exact_or_one_fsynced_row_ahead(
    tmp_path: Path,
) -> None:
    manifest, _rows, _ = _inputs()
    cases = subject._canonical_cases(manifest["samples"], smoke=True)
    partial = tmp_path / ".partial/generations.progress.v1.jsonl"
    partial.parent.mkdir()
    lines = [b'{"a":1}\n']
    partial.write_bytes(lines[0])
    empty_sha = hashlib.sha256().hexdigest()
    state = subject._state_payload(
        status="initialized",
        completed=0,
        expected=1,
        resume_count=0,
        results_path=partial,
        results_sha256=empty_sha,
        results_bytes=0,
        cases=cases,
    )
    assert subject._validate_resume_state(
        state,
        lines=lines,
        active_path=partial,
        partial_path=partial,
        cases=cases,
    ) == (1, 0)
    tampered = copy.deepcopy(state)
    tampered["results"]["sha256"] = "f" * 64
    # Resealing is intentionally omitted: integrity must fail first.
    with pytest.raises(subject.Core8StochasticK10Error, match="integrity"):
        subject._validate_resume_state(
            tampered,
            lines=lines,
            active_path=partial,
            partial_path=partial,
            cases=cases,
        )


def test_canonical_jsonl_reader_rejects_noncanonical_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"a":1,"b":2}\n')
    rows, lines = subject._read_canonical_jsonl(path)
    assert rows == [{"a": 1, "b": 2}]
    assert lines == [b'{"a":1,"b":2}\n']
    path.write_bytes(b'{"b":2, "a":1}\n')
    with pytest.raises(subject.Core8StochasticK10Error, match="not canonical"):
        subject._read_canonical_jsonl(path)


def test_result_persists_variant_and_paired_seed_and_detects_drift() -> None:
    manifest, input_rows, observed = _inputs()
    sample = manifest["samples"][0]
    input_row = input_rows[0]
    paired_key = str(sample["sample_id"])
    source_hashes = subject._source_hashes(manifest, observed)
    generated = "reasoning\n</think>\n" + str(input_row["source_analysis"])
    row = subject._build_result(
        text=generated,
        generated_ids=[10, 11, 999],
        eos_ids=[999],
        pad_token_id=999,
        sample=sample,
        input_row=input_row,
        replicate_id=0,
        replicate_seed=subject.REPLICATE_SEEDS[0],
        paired_seed_key=paired_key,
        sample_manifest_sha256=observed,
        source_hashes=source_hashes,
    )
    subject._validate_result(
        row,
        sample=sample,
        input_row=input_row,
        replicate_id=0,
        replicate_seed=subject.REPLICATE_SEEDS[0],
        paired_seed_key=paired_key,
        sample_manifest_sha256=observed,
        source_hashes=source_hashes,
        tokenizer=None,
    )
    assert row["variant_sample_id"] == sample["sample_id"]
    assert row["paired_seed_key"] == paired_key
    assert row["seed"] == row["row_seed"] == derive_row_seed(
        subject.REPLICATE_SEEDS[0], paired_key
    )
    assert row["generated_text"] == generated
    assert row["answer"]
    assert row["generated_token_ids"] == [10, 11, 999]
    tampered = copy.deepcopy(row)
    tampered["paired_seed_key"] += "-tampered"
    with pytest.raises(subject.Core8StochasticK10Error, match="paired_seed_key"):
        subject._validate_result(
            tampered,
            sample=sample,
            input_row=input_row,
            replicate_id=0,
            replicate_seed=subject.REPLICATE_SEEDS[0],
            paired_seed_key=paired_key,
            sample_manifest_sha256=observed,
            source_hashes=source_hashes,
            tokenizer=None,
        )
