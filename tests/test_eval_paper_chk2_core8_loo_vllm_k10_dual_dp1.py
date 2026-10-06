from __future__ import annotations

import asyncio
import json
import sys
import types
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from jobs.eval import eval_paper_chk2_core8_loo_vllm_k10_dual_dp1 as runner


def _frozen_source_and_ledger() -> tuple[SimpleNamespace, tuple[dict, ...]]:
    rows: list[dict] = []
    ledger: list[dict] = []
    full_ids: dict[str, str] = {}
    for meeting_rank in range(runner.EXPECTED_MEETINGS):
        meeting_id = f"meeting-{meeting_rank:03d}"
        full_id = f"core8-loo::{meeting_id}::full::none"
        full_ids[meeting_id] = full_id
        for variant_rank in range(runner.VARIANTS_PER_MEETING):
            arm = "full" if variant_rank == 0 else "delete"
            topic = None if variant_rank == 0 else f"topic-{variant_rank:02d}"
            sample_id = (
                full_id
                if variant_rank == 0
                else f"core8-loo::{meeting_id}::{arm}::{topic}"
            )
            prompt = f"prompt meeting={meeting_rank} variant={variant_rank}"
            source = f"Analysis meeting {meeting_rank}, variant {variant_rank}."
            reference = f"Minutes meeting {meeting_rank}, variant {variant_rank}."
            token_ids = [101, meeting_rank + 1000, variant_rank + 2000]
            prompt_sha = runner._sha256_text(prompt)
            source_sha = runner._sha256_text(source)
            reference_sha = runner._sha256_text(reference)
            rows.append(
                {
                    "sample_id": sample_id,
                    "meeting_id": meeting_id,
                    "meeting_rank": meeting_rank,
                    "variant_rank": variant_rank,
                    "arm": arm,
                    "intervention_topic": topic,
                    "prompt": prompt,
                    "prompt_sha256": prompt_sha,
                    "prompt_token_count": len(token_ids),
                    "source_analysis": source,
                    "source_analysis_sha256": source_sha,
                    "reference_minutes": reference,
                    "reference_minutes_sha256": reference_sha,
                    "response": f"Reference transport wrapper.\n</think>\n{reference}",
                    "full_source_analysis_sha256": source_sha,
                    "full_reference_sha256": reference_sha,
                    "full_prompt_token_count": len(token_ids),
                }
            )
            ledger.append(
                {
                    "sample_id": sample_id,
                    "meeting_rank": meeting_rank,
                    "variant_rank": variant_rank,
                    "prompt_sha256": prompt_sha,
                    "prompt_token_ids": token_ids,
                    "prompt_token_ids_sha256": runner._token_ids_sha256(token_ids),
                    "prompt_token_count": len(token_ids),
                    "paired_seed_key": full_id,
                }
            )
    return (
        SimpleNamespace(
            ordered_rows=tuple(rows), full_sample_id_by_meeting=full_ids
        ),
        tuple(ledger),
    )


@pytest.fixture(scope="module")
def frozen() -> tuple[SimpleNamespace, tuple[dict, ...]]:
    return _frozen_source_and_ledger()


def test_formal_case_matrix_cardinality_sharding_and_pairing(frozen) -> None:
    source, ledger = frozen
    cases = runner.build_case_matrix(source, ledger)
    assert len(cases) == 21_760
    assert Counter(case["shard_id"] for case in cases) == {0: 10_880, 1: 10_880}
    assert cases[0]["absolute_case_index"] == 0
    assert cases[-1]["absolute_case_index"] == 21_759

    block = [
        case
        for case in cases
        if case["meeting_rank"] == 17 and case["replicate_id"] == 8
    ]
    assert len(block) == 17
    assert {case["variant_rank"] for case in block} == set(range(17))
    assert len({case["row_seed"] for case in block}) == 1
    assert len({case["shard_id"] for case in block}) == 1
    assert block[0]["shard_id"] == runner.assigned_shard(17, 8)


def test_smoke_uses_two_complete_meetings_and_two_replicates(frozen) -> None:
    source, ledger = frozen
    cases = runner.build_case_matrix(source, ledger, smoke=True)
    assert len(cases) == 68
    assert Counter(case["shard_id"] for case in cases) == {0: 34, 1: 34}
    assert {case["meeting_rank"] for case in cases} == {0, 1}
    assert {case["replicate_id"] for case in cases} == {0, 1}


def test_generation_contract_binds_cp50_dynamic_lora_and_sampling() -> None:
    contract = runner.generation_contract()
    assert contract["temperature"] == 0.6
    assert contract["top_p"] == 0.95
    assert contract["top_k"] == 50
    assert contract["max_new_tokens"] == 2560
    assert contract["lora_request"] == {
        "name": "paper-chk2-cp50",
        "integer_id": 50,
        "max_lora_rank": 32,
        "max_loras": 1,
        "max_cpu_loras": 1,
        "lora_dtype": "bfloat16",
    }
    assert contract["paired_block_never_crosses_workers"] is True


def _minimal_case() -> dict:
    prompt = "Rewrite this analysis."
    source = "Prices rose 2 percent."
    reference = "Prices rose 2 percent."
    return {
        "absolute_case_index": 0,
        "generation_key": runner.generation_key("sample-1", 0),
        "sample_id": "sample-1",
        "meeting_id": "meeting-000",
        "meeting_rank": 0,
        "variant_rank": 0,
        "arm": "full",
        "intervention_topic": None,
        "replicate_id": 0,
        "replicate_seed": runner.common.REPLICATE_SEEDS[0],
        "paired_seed_key": "sample-1",
        "paired_block_id": 0,
        "row_seed": runner.derive_row_seed(
            runner.common.REPLICATE_SEEDS[0], "sample-1"
        ),
        "shard_id": 0,
        "prompt": prompt,
        "prompt_sha256": runner._sha256_text(prompt),
        "prompt_token_ids": [10, 11],
        "prompt_token_ids_sha256": runner._token_ids_sha256([10, 11]),
        "prompt_token_count": 2,
        "source_analysis": source,
        "source_analysis_sha256": runner._sha256_text(source),
        "reference_minutes": reference,
        "reference_minutes_sha256": runner._sha256_text(reference),
        "reference_response": f"Reference.\n</think>\n{reference}",
        "full_source_analysis_sha256": runner._sha256_text(source),
        "full_reference_sha256": runner._sha256_text(reference),
        "full_prompt_token_count": 2,
    }


def test_result_persists_recomputable_gates_and_cp50_identity() -> None:
    case = _minimal_case()
    text = "Preserve the claim.\n</think>\nPrices rose 2 percent."
    row = runner._build_result(
        case=case,
        generated_text=text,
        generated_token_ids=[101, 102, 128001],
        raw_finish_reason="stop",
        raw_stop_reason=None,
        eos_token_ids=[128001],
        pad_token_id=128001,
        manifest_sha="a" * 64,
    )
    assert row["model_id"] == "paper_chk2"
    assert row["answer"] == "Prices rose 2 percent."
    assert row["finish_reason"] == "eos"
    assert row["generation_metrics"] == {
        "structure_delivery": True,
        "numeric_fidelity": True,
        "date_fidelity": True,
        "degeneration_free": True,
    }
    assert row["preregistered_core_valid"] is True
    assert row["lora_request"]["integer_id"] == 50


class _FixedDecodeTokenizer:
    eos_token = "<eos>"

    def __init__(self, text: str):
        self.text = text

    def decode(self, *_args, **_kwargs) -> str:
        return self.text


def test_wal_recovers_only_final_unterminated_fragment(tmp_path: Path) -> None:
    case = _minimal_case()
    text = "Preserve the claim.\n</think>\nPrices rose 2 percent."
    row = runner._build_result(
        case=case,
        generated_text=text,
        generated_token_ids=[101, 102, 128001],
        raw_finish_reason="stop",
        raw_stop_reason=None,
        eos_token_ids=[128001],
        pad_token_id=128001,
        manifest_sha="a" * 64,
    )
    wal = tmp_path / "generations.wal.jsonl"
    wal.write_bytes((runner._canonical(row) + "\n{\"torn\":").encode())
    observed = runner.load_wal(
        wal,
        cases_by_key={case["generation_key"]: case},
        manifest_sha="a" * 64,
        tokenizer=_FixedDecodeTokenizer(text),
        eos_token_ids=[128001],
        pad_token_id=128001,
    )
    assert list(observed) == [case["generation_key"]]
    assert wal.read_bytes().endswith(b"\n")
    recovery = wal.with_name(f"{wal.name}.torn-tail-recoveries.jsonl")
    assert recovery.is_file()
    assert json.loads(recovery.read_text().splitlines()[0])["discarded_bytes"] > 0


def test_prepare_resume_is_existing_directory_verify_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "preparation/evaluation_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(
        runner.common,
        "validate_preparation_artifacts",
        lambda **_kwargs: calls.append("validate")
        or SimpleNamespace(
            ledger_rows=(1,), neutral_proofs=(1,), preparation_dir=manifest.parent
        ),
    )
    monkeypatch.setattr(
        runner.common,
        "build_preparation_artifacts",
        lambda **_kwargs: calls.append("build"),
    )
    result = runner.prepare_evaluation(tmp_path, resume=True)
    assert result["resume_verify_only"] is True
    assert calls == ["validate"]


def test_default_formal_authorization_uses_smoke_validation(
    tmp_path: Path,
) -> None:
    smoke = tmp_path / "smoke/validation.json"
    smoke.parent.mkdir(parents=True)
    value = {
        "schema_version": runner.VALIDATION_SCHEMA,
        "status": "complete",
        "mode": "smoke",
        "evaluation_manifest_sha256": "b" * 64,
        "generation_rows": 68,
        "gates": {"route": True, "closure": True},
    }
    smoke.write_text(json.dumps(value) + "\n", encoding="utf-8")
    observed = runner._smoke_authorization(
        output_root=tmp_path, manifest_sha="b" * 64, explicit_path=None
    )
    assert observed["artifact"]["path"] == str(smoke.resolve())
    assert observed["gates"] == value["gates"]


def test_async_engine_always_routes_through_cp50_lora(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    text = "Plan.\n</think>\nPrices rose 2 percent."

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

    torch = types.ModuleType("torch")
    torch.cuda = FakeCuda()

    class SamplingParams:
        def __init__(self, **kwargs):
            captured["sampling"] = kwargs

    class AsyncEngineArgs:
        def __init__(self, **kwargs):
            captured["engine_args"] = kwargs

    class LoRARequest:
        def __init__(self, name, integer_id, *, lora_path):
            captured["lora"] = (name, integer_id, lora_path)

    completion = SimpleNamespace(
        text=text,
        token_ids=[101, 102, 128001],
        finish_reason="stop",
        stop_reason=None,
    )

    class FakeEngine:
        @classmethod
        def from_engine_args(cls, _args):
            return cls()

        async def generate(self, prompt, _params, _request_id, *, lora_request):
            captured["request_lora"] = lora_request
            yield SimpleNamespace(
                prompt_token_ids=prompt["prompt_token_ids"], outputs=[completion]
            )

        def shutdown(self):
            captured["shutdown"] = True

    vllm = types.ModuleType("vllm")
    vllm.__version__ = runner.EXPECTED_VLLM_VERSION
    vllm.SamplingParams = SamplingParams
    engine_pkg = types.ModuleType("vllm.engine")
    arg_utils = types.ModuleType("vllm.engine.arg_utils")
    arg_utils.AsyncEngineArgs = AsyncEngineArgs
    async_engine = types.ModuleType("vllm.engine.async_llm_engine")
    async_engine.AsyncLLMEngine = FakeEngine
    lora_pkg = types.ModuleType("vllm.lora")
    lora_request = types.ModuleType("vllm.lora.request")
    lora_request.LoRARequest = LoRARequest
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.engine", engine_pkg)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arg_utils)
    monkeypatch.setitem(
        sys.modules, "vllm.engine.async_llm_engine", async_engine
    )
    monkeypatch.setitem(sys.modules, "vllm.lora", lora_pkg)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", lora_request)
    monkeypatch.setattr(
        runner,
        "_build_result",
        lambda **kwargs: {
            "generation_key": kwargs["case"]["generation_key"],
            "lora_request": runner.generation_contract()["lora_request"],
        },
    )
    rows: list[dict] = []

    async def on_result(row):
        rows.append(row)

    case = _minimal_case()
    report = asyncio.run(
        runner._run_cp50_engine(
            parent_model_path=Path("/base"),
            adapter_path=Path("/adapter"),
            tokenizer_path=Path("/adapter"),
            tokenizer=_FixedDecodeTokenizer(text),
            eos_token_ids=[128001],
            pad_token_id=128001,
            cases=[case],
            manifest_sha="c" * 64,
            gpu_memory_utilization=0.8,
            on_result=on_result,
        )
    )
    assert captured["lora"] == ("paper-chk2-cp50", 50, "/adapter")
    assert captured["request_lora"] is not None
    assert captured["engine_args"]["enable_lora"] is True
    assert captured["engine_args"]["tokenizer"] == "/adapter"
    assert captured["engine_args"]["max_lora_rank"] == 32
    assert captured["sampling"]["top_p"] == 0.95
    assert captured["sampling"]["max_tokens"] == 2560
    assert report["enable_lora"] is True
    assert rows[0]["generation_key"] == case["generation_key"]


def test_cli_supports_launcher_root_semantics_and_resume() -> None:
    parser = runner.build_parser()
    root = str(runner.DEFAULT_OUTPUT_ROOT)
    prepared = parser.parse_args(["prepare", "--output-root", root, "--resume"])
    assert prepared.output_root == runner.DEFAULT_OUTPUT_ROOT
    assert prepared.resume is True
    worker = parser.parse_args(
        [
            "worker",
            "--manifest",
            str(Path(root) / "preparation/evaluation_manifest.json"),
            "--output-root",
            root,
            "--shard-id",
            "1",
            "--smoke",
            "--resume",
        ]
    )
    assert worker.output_root == runner.DEFAULT_OUTPUT_ROOT
    assert worker.smoke is True
    assert worker.shard_id == 1
