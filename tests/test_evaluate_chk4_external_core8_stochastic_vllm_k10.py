from __future__ import annotations

import math
import json
from pathlib import Path

import pytest

from jobs.retrain_v2 import (
    evaluate_chk4_external_core8_stochastic_vllm_k10 as module,
)


def _samples() -> list[dict[str, object]]:
    result = []
    directions = ["cut", "hold", "hike"]
    for index in range(31):
        panel = "historical_n19" if index < 19 else "postcutoff_n12"
        direction = directions[index % 3] if panel == "historical_n19" else (
            "cut" if index % 4 == 0 else "hold"
        )
        result.append(
            {
                "sample_id": f"sample-{index:02d}",
                "panel": panel,
                "direction": direction,
                "meeting_id": f"meeting-{index:02d}",
                "meeting_start_date": f"2025-01-{(index % 28) + 1:02d}",
            }
        )
    return result


def _models() -> list[dict[str, object]]:
    return [
        {"label": "m0", "display_name": "M0"},
        {"label": "m1", "display_name": "M1"},
        {"label": "m2", "display_name": "M2"},
    ]


def _vectors() -> dict[str, dict[str, dict[str, object]]]:
    values: dict[str, dict[str, dict[str, object]]] = {}
    for model_index, model in enumerate(_models()):
        model_rows = {}
        for sample_index, sample in enumerate(_samples()):
            correctness = [
                (replicate + sample_index + model_index) % 4 != 0
                for replicate in range(module.REPLICATES)
            ]
            delivery = [
                (replicate + 2 * sample_index + model_index) % 5 != 0
                for replicate in range(module.REPLICATES)
            ]
            model_rows[str(sample["sample_id"])] = {
                "sample_id": sample["sample_id"],
                "panel": sample["panel"],
                "target_direction": sample["direction"],
                "correctness": correctness,
                "delivery": delivery,
                "correctness_rate": sum(correctness) / module.REPLICATES,
                "delivery_rate": sum(delivery) / module.REPLICATES,
                "modal_correct": sample_index % (model_index + 2) == 0,
            }
        values[str(model["label"])] = model_rows
    return values


def test_canonical_cases_share_seed_and_gpu_for_all_models() -> None:
    cases = module.canonical_cases(_samples(), _models())

    assert len(cases) == 930
    assert {row["absolute_case_index"] for row in cases} == set(range(930))
    for block_id in range(310):
        block = [row for row in cases if row["paired_block_id"] == block_id]
        assert len(block) == 3
        assert len({row["row_seed"] for row in block}) == 1
        assert len({row["assigned_shard"] for row in block}) == 1
    assert sum(row["assigned_shard"] == 0 for row in cases) == 465
    assert sum(row["assigned_shard"] == 1 for row in cases) == 465


def test_generation_contract_is_exactly_requested_sampling_profile() -> None:
    assert module.TEMPERATURE == 0.6
    assert module.TOP_P == 0.9
    assert module.TOP_K == -1
    assert module.REPETITION_PENALTY == 1.0
    assert module.MAX_NEW_TOKENS == 1536
    assert module.REPLICATES == 10
    assert module.GPU_MEMORY_UTILIZATION == 0.95


def test_v3_identity_supersedes_failed_attempts_without_row_reuse() -> None:
    assert module.DEFAULT_OUTPUT_ROOT.name.endswith("_v3_20260824")
    assert module.EVALUATION_ID.endswith("-v3")
    smoke = module._failed_smoke_v1_supersession_record()
    assert smoke["status"] == "failed_two_gpu_smoke_superseded"
    assert smoke["smoke_wal_rows_generated"] == 20
    assert smoke["smoke_rows_reused"] == 0
    assert smoke["formal_rows_reused"] == 0
    assert smoke["formal_generation_started"] is False
    formal = module._failed_formal_v2_supersession_record()
    assert formal["status"] == "failed_formal_worker_dependency_check_superseded"
    assert formal["smoke_rows_generated"] == 20
    assert formal["smoke_rows_reused"] == 0
    assert formal["formal_rows_generated"] == 0
    assert formal["formal_rows_reused"] == 0


def test_meeting_metric_averages_within_meeting_first() -> None:
    meetings = [
        {
            "sample_id": "a",
            "target_direction": "cut",
            "correctness": [True] * 5 + [False] * 5,
            "delivery": [True] * 10,
            "correctness_rate": 0.5,
            "delivery_rate": 1.0,
            "modal_correct": False,
            "modal_tie": True,
            "modal_share": 0.5,
            "normalized_four_label_entropy": 0.5,
            "unanimous": False,
        },
        {
            "sample_id": "b",
            "target_direction": "hold",
            "correctness": [True] * 10,
            "delivery": [False] * 10,
            "correctness_rate": 1.0,
            "delivery_rate": 0.0,
            "modal_correct": True,
            "modal_tie": False,
            "modal_share": 1.0,
            "normalized_four_label_entropy": 0.0,
            "unanimous": True,
        },
    ]

    block = module._metric_from_meetings(meetings, panel="postcutoff_n12")

    assert block["meeting_averaged_direction_accuracy"] == 0.75
    assert block["meeting_averaged_delivery_rate"] == 0.5
    assert block["balanced_accuracy"] == 0.75
    assert block["fixed_three_class_balanced_accuracy"] is None
    assert block["per_class"]["hike"]["recall"] is None


def test_hierarchical_bootstrap_is_paired_and_deterministic() -> None:
    vectors = _vectors()
    first, first_ledger = module.hierarchical_bootstrap(
        vectors,
        _samples(),
        _models(),
        panel="historical_n19",
        draws=25,
        seed=123,
    )
    second, second_ledger = module.hierarchical_bootstrap(
        vectors,
        _samples(),
        _models(),
        panel="historical_n19",
        draws=25,
        seed=123,
    )

    assert first == second
    assert first_ledger == second_ledger
    assert first["meeting_draws_within_target_class"] is True
    assert first["replicate_draws_within_meeting"] is True
    assert first["draws_shared_across_models"] is True
    assert len(first_ledger) == 25
    assert len(first_ledger[0]["draw_plan"]) == 19
    assert all(
        len(item["replicate_indexes"]) == 10
        for item in first_ledger[0]["draw_plan"]
    )


def test_exact_vector_swap_uses_meeting_vectors_not_generation_rows() -> None:
    left = [
        {
            "sample_id": "a",
            "target_direction": "cut",
            "correctness_rate": 1.0,
            "delivery_rate": 1.0,
        },
        {
            "sample_id": "b",
            "target_direction": "hold",
            "correctness_rate": 0.0,
            "delivery_rate": 0.0,
        },
    ]
    right = [
        {
            "sample_id": "a",
            "target_direction": "cut",
            "correctness_rate": 0.0,
            "delivery_rate": 0.0,
        },
        {
            "sample_id": "b",
            "target_direction": "hold",
            "correctness_rate": 0.0,
            "delivery_rate": 0.0,
        },
    ]

    result = module.exact_vector_label_swap(left, right, metric="accuracy")

    assert result["meetings"] == 2
    assert result["assignments"] == 4
    assert result["difference_a_minus_b"] == 0.5
    assert result["p_value_raw"] == 1.0


def test_holm_family_is_monotone_and_retains_declared_size() -> None:
    rows = [{"p_value_raw": value} for value in (0.01, 0.03, 0.02, 0.9, 0.4, 0.5)]
    adjusted = module.holm_adjust(rows, family="six-test")

    assert len(adjusted) == 6
    assert all(row["holm_family_size"] == 6 for row in adjusted)
    assert adjusted[0]["p_value_holm"] == pytest.approx(0.06)
    assert all(row["p_value_holm"] >= row["p_value_raw"] for row in adjusted)


def test_normalized_entropy_has_expected_endpoints() -> None:
    assert -sum(p * math.log(p) for p in (1.0,) if p > 0) / math.log(4) == 0.0
    uniform = -sum(0.25 * math.log(0.25) for _ in range(4)) / math.log(4)
    assert uniform == pytest.approx(1.0)


class _FakeTokenizer:
    pieces = {10: "alpha", 11: " beta", 99: "<eos>"}

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        spaces_between_special_tokens: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert spaces_between_special_tokens is True
        return "".join(self.pieces[value] for value in token_ids)


def _raw_replay_inputs() -> tuple[dict, dict, dict]:
    case = {
        "absolute_case_index": 0,
        "paired_block_id": 0,
        "meeting_rank": 0,
        "model_rank": 0,
        "model_label": "m0",
        "sample_id": "sample-00",
        "panel": "historical_n19",
        "replicate_id": 0,
        "replicate_seed": module.REPLICATE_SEEDS[0],
        "row_seed": module._row_seed("sample-00", 0),
    }
    token = {
        "prompt_token_count": 2,
        "prompt_token_ids_sha256": "a" * 64,
    }
    model = {"label": "m0", "sha256": "b" * 64}
    return case, token, model


def test_vllm_v1_terminal_eos_is_retained_in_ids_but_omitted_from_text() -> None:
    tokenizer = _FakeTokenizer()
    text, evidence = module._accepted_vllm_text_from_raw_ids(
        tokenizer=tokenizer,
        generated_token_ids=[10, 11, 99],
        finish_reason="stop",
        stop_reason=99,
        eos_ids={99},
    )

    assert text == "alpha beta"
    assert evidence["omitted_terminal_stop_id"] == 99
    assert evidence["raw_token_count"] == 3
    assert evidence["accepted_text_token_count"] == 2

    case, token, model = _raw_replay_inputs()
    row = module._raw_worker_row(
        case=case,
        token=token,
        manifest_sha256="c" * 64,
        model=model,
        shard_id=0,
        physical_gpu_index=0,
        generated_token_ids=[10, 11, 99],
        generated_text="alpha beta",
        finish_reason="stop",
        stop_reason=99,
        request_id="request-0",
    )
    assert module._validate_raw_row(
        row,
        case=case,
        token=token,
        model=model,
        manifest_sha256="c" * 64,
        shard_id=0,
        physical_gpu_index=0,
        tokenizer=tokenizer,
        eos_ids={99},
    ) == row


def test_vllm_length_completion_decodes_all_raw_tokens_and_rejects_tamper() -> None:
    tokenizer = _FakeTokenizer()
    text, evidence = module._accepted_vllm_text_from_raw_ids(
        tokenizer=tokenizer,
        generated_token_ids=[10, 11],
        finish_reason="length",
        stop_reason=None,
        eos_ids={99},
    )
    assert text == "alpha beta"
    assert evidence["omitted_terminal_stop_id"] is None
    assert evidence["accepted_text_token_count"] == 2

    case, token, model = _raw_replay_inputs()
    row = module._raw_worker_row(
        case=case,
        token=token,
        manifest_sha256="c" * 64,
        model=model,
        shard_id=0,
        physical_gpu_index=0,
        generated_token_ids=[10, 11],
        generated_text="alpha beta",
        finish_reason="length",
        stop_reason=None,
        request_id="request-0",
    )
    tampered = dict(row)
    tampered["vllm_generated_text"] = "alpha"
    tampered["vllm_generated_text_sha256"] = module._sha256_text("alpha")
    with pytest.raises(
        module.StochasticDecisionEvaluationError,
        match="accepted-text/raw-token drift",
    ):
        module._validate_raw_row(
            tampered,
            case=case,
            token=token,
            model=model,
            manifest_sha256="c" * 64,
            shard_id=0,
            physical_gpu_index=0,
            tokenizer=tokenizer,
            eos_ids={99},
        )


def test_terminal_eos_text_inclusion_is_rejected_as_vllm_v1_tamper() -> None:
    tokenizer = _FakeTokenizer()
    case, token, model = _raw_replay_inputs()
    row = module._raw_worker_row(
        case=case,
        token=token,
        manifest_sha256="c" * 64,
        model=model,
        shard_id=0,
        physical_gpu_index=0,
        generated_token_ids=[10, 99],
        generated_text="alpha<eos>",
        finish_reason="stop",
        stop_reason=99,
        request_id="request-0",
    )
    with pytest.raises(
        module.StochasticDecisionEvaluationError,
        match="accepted-text/raw-token drift",
    ):
        module._validate_raw_row(
            row,
            case=case,
            token=token,
            model=model,
            manifest_sha256="c" * 64,
            shard_id=0,
            physical_gpu_index=0,
            tokenizer=tokenizer,
            eos_ids={99},
        )


def test_wal_recovers_only_an_unterminated_final_fragment(tmp_path: Path) -> None:
    path = tmp_path / "wal.jsonl"
    first = {"case": 1}
    second = {"case": 2}
    path.write_bytes(
        (module.canonical_json(first) + "\n" + module.canonical_json(second)[:5]).encode()
    )

    assert module._read_wal_jsonl_recover_torn_final(path, label="test WAL") == [first]
    assert path.read_bytes() == (module.canonical_json(first) + "\n").encode()

    path.write_bytes((module.canonical_json(first) + "\n{bad}\n").encode())
    with pytest.raises(
        module.StochasticDecisionEvaluationError,
        match="invalid committed row",
    ):
        module._read_wal_jsonl_recover_torn_final(path, label="test WAL")


def test_compare_or_create_terminal_outputs_are_idempotent_and_detect_drift(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "runtime.json"
    jsonl_path = tmp_path / "results.jsonl"
    value = {"status": "complete", "rows": 2}
    rows = [{"case": 0}, {"case": 1}]

    module._write_or_validate_json(json_path, value, label="runtime")
    module._write_or_validate_json(json_path, value, label="runtime")
    module._write_or_validate_jsonl(jsonl_path, rows, label="results")
    module._write_or_validate_jsonl(jsonl_path, rows, label="results")
    with pytest.raises(module.StochasticDecisionEvaluationError, match="drift"):
        module._write_or_validate_json(json_path, {**value, "rows": 3}, label="runtime")
    with pytest.raises(module.StochasticDecisionEvaluationError, match="drift"):
        module._write_or_validate_jsonl(
            jsonl_path, [{"case": 9}], label="results"
        )


def test_worker_runtime_gate_checks_full_packages_and_gpu_uuid() -> None:
    packages = {
        "vllm": "0.8.5.post1",
        "torch": "2.6.0",
        "transformers": "4.51.3",
        "tokenizers": "0.21.4",
        "safetensors": "0.5.3",
        "numpy": "2.2.6",
    }
    gpu = {
        "index": 0,
        "uuid": "GPU-expected",
        "name": "NVIDIA A30",
        "driver_version": "550.163.01",
        "memory_mib": 24576,
        "compute_capability": "8.0",
    }
    manifest = {
        "runtime_prebinding": {
            "vllm_python": {
                "python": {"executable": "/env/python", "version": "3.10.9"},
                "packages": packages,
            },
            "physical_gpu_inventory": [gpu, {**gpu, "index": 1, "uuid": "GPU-other"}],
        }
    }
    environment = {
        **module.REQUIRED_VLLM_ENV,
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    observation = {
        "python": manifest["runtime_prebinding"]["vllm_python"]["python"],
        "packages": packages,
        "physical_gpu_inventory": manifest["runtime_prebinding"]["physical_gpu_inventory"],
        "selected_physical_gpu": {
            "uuid": "GPU-expected",
            "memory_total_mib": 24576,
        },
        "environment": environment,
    }
    module._validate_worker_runtime_observation(
        observation, manifest=manifest, physical_gpu_index=0
    )

    changed = {**observation, "packages": {**packages, "vllm": "0.0.0"}}
    with pytest.raises(module.StochasticDecisionEvaluationError, match="package-version"):
        module._validate_worker_runtime_observation(
            changed, manifest=manifest, physical_gpu_index=0
        )
    changed = {
        **observation,
        "selected_physical_gpu": {"uuid": "GPU-wrong", "memory_total_mib": 24576},
    }
    with pytest.raises(module.StochasticDecisionEvaluationError, match="UUID/memory"):
        module._validate_worker_runtime_observation(
            changed, manifest=manifest, physical_gpu_index=0
        )


def test_run_receipt_deep_validation_detects_runtime_and_receipt_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    (root / "run").mkdir()
    manifest_path = root / "evaluation_manifest.json"
    authorization_path = root / "formal_generation_authorization.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    authorization_path.write_text("{}\n", encoding="utf-8")
    results = [
        {"absolute_case_index": 0, "panel": "historical_n19"},
        {"absolute_case_index": 1, "panel": "postcutoff_n12"},
    ]
    module._write_exclusive_jsonl(root / "run/results.jsonl", results)
    runtime = {"workers": [], "contract": "bound"}
    module._write_exclusive_json(root / "run/runtime_summary.json", runtime)
    workers = [{"path": "worker", "sha256": "a" * 64}]
    manifest = {"models": []}
    monkeypatch.setattr(module, "EXPECTED_RAW_ROWS", 2)
    monkeypatch.setattr(
        module,
        "EXPECTED_PANEL_ROWS",
        {"historical_n19": 1, "postcutoff_n12": 1},
    )
    monkeypatch.setattr(module, "_validate_authorization", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        module,
        "_load_worker_rows",
        lambda **kwargs: (results, workers),
    )
    monkeypatch.setattr(module, "_expected_runtime_summary", lambda *args: runtime)
    receipt_payload = {
        "schema_version": module.RUN_RECEIPT_SCHEMA,
        "status": "complete",
        "created_at_utc": "2026-08-24T00:00:00Z",
        "evaluation_id": module.EVALUATION_ID,
        "manifest": module._file_record(manifest_path),
        "authorization": module._file_record(authorization_path),
        "workers": workers,
        "results": module._file_record(root / "run/results.jsonl", relative_to=root),
        "runtime": module._file_record(
            root / "run/runtime_summary.json", relative_to=root
        ),
        "rows": 2,
        "panel_rows": {"historical_n19": 1, "postcutoff_n12": 1},
        "source_v4_generation_rows_reused": 0,
    }
    module._write_exclusive_json(
        root / "run/run_receipt.json", module._sealed(receipt_payload)
    )
    module._validate_run_receipt(
        manifest_path=manifest_path,
        manifest_sha256="b" * 64,
        authorization_path=authorization_path,
        authorization_sha256="c" * 64,
        manifest=manifest,
    )

    (root / "run/runtime_summary.json").write_text(
        json.dumps({"workers": [], "contract": "tampered"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(module.StochasticDecisionEvaluationError, match="runtime replay"):
        module._validate_run_receipt(
            manifest_path=manifest_path,
            manifest_sha256="b" * 64,
            authorization_path=authorization_path,
            authorization_sha256="c" * 64,
            manifest=manifest,
        )


def test_worker_resume_launch_ignores_only_timestamp_and_validates_integrity() -> None:
    payload = {
        "schema_version": module.WORKER_RECEIPT_SCHEMA,
        "status": "launched",
        "created_at_utc": "first",
        "model_label": "m0",
    }
    existing = module._sealed(payload)
    module._validate_integrity(existing, label="worker launch")
    restarted = module._sealed({**payload, "created_at_utc": "second"})
    stable_keys = set(restarted) - {"created_at_utc", "integrity"}
    assert all(existing[key] == restarted[key] for key in stable_keys)
    corrupted = dict(existing)
    corrupted["model_label"] = "m1"
    with pytest.raises(module.StochasticDecisionEvaluationError, match="payload drift"):
        module._validate_integrity(corrupted, label="worker launch")


def test_official_source_archive_hashes_every_copied_document(
    tmp_path: Path,
) -> None:
    samples = []
    official_rows = []
    for index in range(module.MEETING_COUNT):
        panel = "historical_n19" if index < 19 else "postcutoff_n12"
        meeting_id = f"meeting-{index:02d}"
        url = f"https://www.federalreserve.gov/example/{meeting_id}.htm"
        source_path = (
            tmp_path / f"sources/official/{panel}/{meeting_id}.html"
        )
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(f"official document {index}".encode())
        record = {
            **module._file_record(source_path, relative_to=tmp_path),
            "meeting_id": meeting_id,
            "panel": panel,
            "requested_url": url,
            "final_url": url,
            "role": "gold_label_source_only_not_model_input",
            "validation": {"domain_check": "passed"},
        }
        official_rows.append(record)
        sample = {
            "meeting_id": meeting_id,
            "panel": panel,
            "official_source_url": url,
        }
        if panel == "historical_n19":
            sample["official_source_sha256"] = record["sha256"]
        samples.append(sample)
    manifest_path = tmp_path / "sources/official_source_manifest.jsonl"
    module._write_exclusive_jsonl(manifest_path, official_rows)

    assert len(
        module._validate_official_source_archive(
            root=tmp_path,
            manifest_path=manifest_path,
            samples=samples,
        )
    ) == module.MEETING_COUNT

    tampered = tmp_path / str(official_rows[7]["path"])
    tampered.write_bytes(b"tampered official document")
    with pytest.raises(
        module.StochasticDecisionEvaluationError,
        match="archived official source drift",
    ):
        module._validate_official_source_archive(
            root=tmp_path,
            manifest_path=manifest_path,
            samples=samples,
        )


def test_vllm_v1_shutdown_uses_public_shutdown_api() -> None:
    class FakeAsyncLLM:
        def __init__(self) -> None:
            self.closed = False

        def shutdown(self) -> None:
            self.closed = True

    engine = FakeAsyncLLM()
    module._shutdown_vllm_v1_engine(engine)
    assert engine.closed is True


def test_vllm_v1_shutdown_fails_closed_on_api_drift() -> None:
    with pytest.raises(
        module.StochasticDecisionEvaluationError,
        match="shutdown API drift",
    ):
        module._shutdown_vllm_v1_engine(object())


def test_formal_worker_authorization_is_dependency_light(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "evaluation_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    manifest_sha = "a" * 64
    gates = {"full_raw_token_replay": True}
    smoke = module._sealed(
        {
            "schema_version": module.SMOKE_RECEIPT_SCHEMA,
            "status": "passed",
            "evaluation_id": module.EVALUATION_ID,
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": manifest_sha,
            },
            "rows": module.SMOKE_ROWS,
            "shards": {"shard0": 10, "shard1": 10},
            "gates": gates,
        }
    )
    smoke_path = tmp_path / "smoke_receipt.json"
    module._write_exclusive_json(smoke_path, smoke)
    authorization = module._sealed(
        {
            "schema_version": module.AUTHORIZATION_SCHEMA,
            "status": "authorized_after_independent_audit",
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": manifest_sha,
            },
            "validated_infrastructure_smoke": {
                "receipt": module._file_record(smoke_path),
                "rows": module.SMOKE_ROWS,
                "shards": {"shard0": 10, "shard1": 10},
                "gates": gates,
            },
        }
    )
    authorization_path = tmp_path / "authorization.json"
    module._write_exclusive_json(authorization_path, authorization)
    monkeypatch.setattr(
        module,
        "_validate_smoke_receipt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("deep smoke replay must remain in the parent")
        ),
    )

    observed = module._validate_authorization(
        authorization_path,
        module._sha256_file(authorization_path),
        manifest_path,
        manifest_sha,
        deep_smoke=False,
    )
    assert observed == authorization
