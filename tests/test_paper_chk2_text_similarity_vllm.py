from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from jobs.eval import paper_chk2_text_similarity_vllm as subject


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(subject.canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _file_record(path: Path, rows: int) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": subject.sha256_file(path),
        "rows": rows,
    }


def _model(path: Path) -> dict[str, object]:
    path.mkdir(parents=True)
    for name, value in (
        ("config.json", {"model_type": "llama"}),
        ("model.safetensors.index.json", {"weight_map": {}}),
        ("tokenizer.json", {"version": "1"}),
    ):
        _write_json(path / name, value)
    return {"path": str(path)}


def _prepared(tmp_path: Path) -> tuple[Path, subject.EvaluationInputs]:
    samples: list[dict[str, object]] = []
    ledger: list[dict[str, object]] = []
    for index in range(subject.SAMPLE_COUNT):
        sample_id = f"sample-{index:03d}"
        prompt = f"prompt {index}"
        reference = f"reference {index}"
        token_ids = [128000, 1000 + index, 128006]
        samples.append(
            {
                "sample_id": sample_id,
                "split": ("train", "validation", "test")[index % 3],
                "meeting_id": f"meeting-{index % 17:02d}",
                "prompt": prompt,
                "reference": reference,
                "prompt_sha256": subject.sha256_text(prompt),
                "reference_sha256": subject.sha256_text(reference),
                "prompt_token_ids": token_ids,
            }
        )
        ledger.append(
            {
                "sample_id": sample_id,
                "prompt_token_ids": token_ids,
                "prompt_token_count": len(token_ids),
                "prompt_token_ids_sha256": subject._token_ids_hash(token_ids),
            }
        )
    samples_path = tmp_path / "samples.jsonl"
    ledger_path = tmp_path / "prompt_token_ledger.jsonl"
    _write_jsonl(samples_path, samples)
    _write_jsonl(ledger_path, ledger)
    chk0 = _model(tmp_path / "chk0")
    chk1 = _model(tmp_path / "chk1")
    adapter = tmp_path / "cp50"
    adapter.mkdir()
    _write_json(
        adapter / "adapter_config.json",
        {
            "base_model_name_or_path": str(tmp_path / "chk1"),
            "bias": "none",
            "lora_alpha": 64,
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": 32,
            "target_modules": ["q_proj", "v_proj"],
            "use_dora": False,
        },
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    manifest = {
        "inputs": {
            "samples": _file_record(samples_path, subject.SAMPLE_COUNT),
            "prompt_token_ledger": _file_record(ledger_path, subject.SAMPLE_COUNT),
        },
        "models": {
            "chk0": chk0,
            "chk1": chk1,
            "chk2": {
                "adapter_path": str(adapter),
                "base_model_path": str(tmp_path / "chk1"),
            },
        },
        "generation": {
            "k": 10,
            "temperature": 0.6,
            "top_p": 0.9,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "max_tokens": 2048,
            "max_model_len": 4096,
            "max_num_seqs": 16,
            "tensor_parallel_size_per_engine": 1,
            "data_parallel_size_per_engine": 1,
            "replicate_seeds": list(subject.DEFAULT_REPLICATE_SEEDS),
        },
    }
    manifest_path = tmp_path / "evaluation_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, subject.load_evaluation_inputs(manifest_path)


def _result(case: dict[str, object], manifest_sha: str) -> dict[str, object]:
    completion = "reasoning</think>Minutes paragraph."
    token_ids = [42, {"chk0": 43, "chk1": 44, "chk2": 45}[str(case["model_id"])]]
    return {
        "schema_version": subject.ROW_SCHEMA,
        "evaluation_manifest_sha256": manifest_sha,
        **{key: value for key, value in case.items() if key != "prompt_token_ids"},
        "request_id": f"request-{case['generation_key']}",
        "completion": completion,
        "completion_sha256": subject.sha256_text(completion),
        "output_token_ids": token_ids,
        "output_token_ids_sha256": subject._token_ids_hash(token_ids),
        "raw_generated_tokens": len(token_ids),
        "finish_reason": "stop",
        "stop_reason": None,
        "generation_contract": subject._generation_contract(),
        "completed_at_utc": "2026-09-01T00:00:00Z",
    }


def test_full_matrix_has_paired_seed_and_gpu_closure(tmp_path: Path) -> None:
    _, inputs = _prepared(tmp_path)
    cases = subject.build_case_matrix(inputs)
    assert len(cases) == 391 * 10 * 3
    assert Counter(case["model_id"] for case in cases) == Counter(
        {"chk0": 3910, "chk1": 3910, "chk2": 3910}
    )
    paired: dict[str, list[dict[str, object]]] = defaultdict(list)
    for case in cases:
        paired[str(case["tuple_id"])].append(case)
    assert len(paired) == 3910
    for rows in paired.values():
        assert {row["model_id"] for row in rows} == set(subject.MODEL_IDS)
        assert len({row["row_seed"] for row in rows}) == 1
        assert len({row["shard_id"] for row in rows}) == 1
        assert len({row["prompt_token_ids_sha256"] for row in rows}) == 1


def test_shard_function_does_not_depend_on_model() -> None:
    sample_id = "same-sample"
    for replicate in range(10):
        expected = subject.tuple_shard(sample_id, replicate)
        assert expected in {0, 1}
        assert {
            subject.tuple_shard(sample_id, replicate)
            for _model_id in subject.MODEL_IDS
        } == {expected}


def test_dynamic_gpu_fraction_is_floored_and_fails_below_minimum() -> None:
    assert subject.compute_gpu_memory_utilization(21_226, 24_576) == 0.78
    assert subject.compute_gpu_memory_utilization(22_633, 24_576) == 0.83
    with pytest.raises(subject.PaperChk2VllmError, match="insufficient GPU memory"):
        subject.compute_gpu_memory_utilization(20_000, 24_576)


def test_smoke_wals_validate_two_shard_closure(tmp_path: Path) -> None:
    manifest_path, inputs = _prepared(tmp_path)
    cases = subject.build_case_matrix(
        inputs, smoke=True, smoke_samples=6, smoke_replicates=2
    )
    output_root = tmp_path / "outputs"
    for shard_id in subject.SHARD_IDS:
        shard_rows = [
            _result(case, inputs.manifest_sha256)
            for case in cases
            if case["shard_id"] == shard_id
        ]
        _write_jsonl(
            output_root / f"shard-{shard_id}" / "generations.wal.jsonl",
            shard_rows,
        )
    result = subject.validate_outputs(
        manifest_path=manifest_path,
        output_root=output_root,
        smoke=True,
        smoke_samples=6,
        smoke_replicates=2,
    )
    assert result["status"] == "complete"
    assert result["generation_rows"] == 6 * 2 * 3
    assert result["paired_tuples"] == 6 * 2
    assert result["model_counts"] == {"chk0": 12, "chk1": 12, "chk2": 12}
    assert result["gates"]["base_lora_outputs_not_all_identical"] is True


def test_smoke_selection_is_two_rows_per_split(tmp_path: Path) -> None:
    _, inputs = _prepared(tmp_path)
    cases = subject.build_case_matrix(
        inputs, smoke=True, smoke_samples=6, smoke_replicates=2
    )
    sample_ids_by_split: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        sample_ids_by_split[str(case["split"])].add(str(case["sample_id"]))
    assert {key: len(value) for key, value in sample_ids_by_split.items()} == {
        "train": 2,
        "validation": 2,
        "test": 2,
    }


def test_delivery_projection_recovers_and_marks_transport_failures() -> None:
    valid = subject.project_delivery(
        {
            "replicate_index": 3,
            "completion": "A fidelity plan.</think>A single Minutes paragraph.",
            "output_token_ids": [1, 2, 3, 4, 5, 6],
            "finish_reason": "stop",
        }
    )
    assert valid["replicate_id"] == 3
    assert valid["recovered_text"] == "A single Minutes paragraph."
    assert valid["completion_token_ids"] == [1, 2, 3, 4, 5, 6]
    assert valid["delivery_valid"] is True
    assert valid["delivery_failures"] == []

    malformed = subject.project_delivery(
        {
            "replicate_index": 0,
            "completion": "draft candidate\n\nLast nonempty candidate.",
            "output_token_ids": [1, 2, 3, 4] * 20,
            "finish_reason": "length",
        }
    )
    assert malformed["recovered_text"] == "Last nonempty candidate."
    assert malformed["recovery_method"] == "raw_best_effort_last_nonempty_paragraph"
    assert {
        "think_boundary_count_not_one",
        "finish_reason_not_stop",
        "full_4gram_repetition_ge_0.50",
        "tail_4gram_repetition_ge_0.60",
    } <= set(malformed["delivery_failures"])
    assert malformed["delivery_valid"] is False

    structure = subject.project_delivery(
        {
            "replicate_index": 1,
            "completion": "plan</think>First paragraph.\n<answer>Second paragraph.</answer>",
            "output_token_ids": [7, 8, 9, 10, 11],
            "finish_reason": "stop",
        }
    )
    assert "recovered_text_multi_paragraph" in structure["delivery_failures"]
    assert "recovered_text_control_marker" in structure["delivery_failures"]


def test_consolidate_full_outputs_and_resume_noop(tmp_path: Path) -> None:
    manifest_path, inputs = _prepared(tmp_path)
    cases = subject.build_case_matrix(inputs)
    output_root = tmp_path / "outputs"
    for shard_id in subject.SHARD_IDS:
        _write_jsonl(
            output_root / f"shard-{shard_id}" / "generations.wal.jsonl",
            [
                _result(case, inputs.manifest_sha256)
                for case in cases
                if case["shard_id"] == shard_id
            ],
        )

    result = subject.consolidate_outputs(
        manifest_path=manifest_path,
        generation_output_root=output_root,
        destination_root=output_root,
    )
    assert result["status"] == "complete"
    assert result["resume_noop"] is False
    assert result["generation_rows"]["rows"] == 11_730
    rows = subject._read_jsonl(
        output_root / "generation_rows.jsonl", label="generation rows"
    )
    assert len(rows) == 11_730
    assert rows[0]["replicate_id"] == rows[0]["replicate_index"]
    assert rows[0]["completion_token_ids"] == rows[0]["output_token_ids"]
    assert rows[0]["recovered_text"] == "Minutes paragraph."
    assert rows[0]["delivery_valid"] is True

    complete = json.loads(
        (output_root / "complete_evaluation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert complete["status"] == "complete"
    assert complete["model_order"] == ["chk0", "chk1", "chk2"]
    assert complete["k"] == 10
    assert complete["temperature"] == 0.6
    assert complete["top_p"] == 0.9
    assert complete["total_generations"] == 11_730
    assert complete["artifacts"]["samples"]["sha256"] == inputs.samples_binding["sha256"]
    assert complete["generation_rows"] == result["generation_rows"]

    resumed = subject.consolidate_outputs(
        manifest_path=manifest_path,
        generation_output_root=output_root,
        destination_root=output_root,
        resume=True,
    )
    assert resumed["resume_noop"] is True


def test_duplicate_wal_key_is_rejected(tmp_path: Path) -> None:
    _, inputs = _prepared(tmp_path)
    case = subject.build_case_matrix(
        inputs, smoke=True, smoke_samples=1, smoke_replicates=1
    )[0]
    row = _result(case, inputs.manifest_sha256)
    path = tmp_path / "duplicate.jsonl"
    _write_jsonl(path, [row, row])
    with pytest.raises(subject.PaperChk2VllmError, match="duplicate WAL key"):
        subject.load_wal(
            path,
            cases_by_key={str(case["generation_key"]): case},
            manifest_sha=inputs.manifest_sha256,
        )


def test_wal_torn_tail_is_discarded_with_a_recovery_ledger(tmp_path: Path) -> None:
    _, inputs = _prepared(tmp_path)
    case = subject.build_case_matrix(
        inputs, smoke=True, smoke_samples=1, smoke_replicates=1
    )[0]
    row = _result(case, inputs.manifest_sha256)
    path = tmp_path / "recoverable.wal.jsonl"
    valid = (subject.canonical_json(row) + "\n").encode("utf-8")
    tail = b'{"schema_version":"partial'
    path.write_bytes(valid + tail)

    recovered = subject.load_wal(
        path,
        cases_by_key={str(case["generation_key"]): case},
        manifest_sha=inputs.manifest_sha256,
    )

    assert set(recovered) == {case["generation_key"]}
    assert path.read_bytes() == valid
    ledger = path.with_name(f"{path.name}.torn-tail-recoveries.jsonl")
    receipt = json.loads(ledger.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == subject.TORN_TAIL_RECOVERY_SCHEMA
    assert receipt["discarded_bytes"] == len(tail)
    assert receipt["recovered_bytes"] == len(valid)
