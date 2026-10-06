from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from jobs.eval import eval_paper_chk2_tadle_lm_vllm_k10_dual_dp1 as generation
from jobs.eval import paper_chk2_tadle_lm_k10_common as common


@pytest.mark.parametrize(
    "writer_module", (common, generation), ids=("common", "worker")
)
def test_create_only_writer_survives_interruption_before_atomic_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer_module: types.ModuleType,
) -> None:
    """A killed publisher cannot expose a partial canonical artifact."""

    target = tmp_path / "nested/canonical.jsonl"
    payload = b'{"complete":true}\n'
    real_link = os.link
    interrupted = {"done": False}

    def interrupt_once(source: Path, destination: Path) -> None:
        if Path(destination) == target and not interrupted["done"]:
            interrupted["done"] = True
            raise RuntimeError("injected interruption before publish")
        real_link(source, destination)

    monkeypatch.setattr(os, "link", interrupt_once)
    with pytest.raises(RuntimeError, match="injected interruption"):
        writer_module._write_new_bytes(target, payload)
    assert not target.exists()
    assert not list(target.parent.glob(f".{target.name}.tmp-*"))

    monkeypatch.setattr(os, "link", real_link)
    writer_module._write_new_bytes(target, payload)
    assert target.read_bytes() == payload
    with pytest.raises(FileExistsError):
        writer_module._write_new_bytes(target, b"different\n")
    assert target.read_bytes() == payload


def test_vllm_runtime_import_does_not_require_pandas() -> None:
    """The GPU worker environment deliberately has no pandas installation."""

    repository_root = Path(__file__).resolve().parents[1]
    vllm_python = Path("/home/haobin_cui/.conda/envs/vllm_env/bin/python")
    if not vllm_python.is_file():
        pytest.skip("pinned vllm runtime is unavailable")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(repository_root / "src"), str(repository_root))
    )
    process = subprocess.run(
        [
            str(vllm_python),
            "-c",
            (
                "import importlib.util; "
                "assert importlib.util.find_spec('pandas') is None; "
                "import jobs.eval.paper_chk2_tadle_lm_k10_common; "
                "import jobs.eval.eval_paper_chk2_tadle_lm_vllm_k10_dual_dp1"
            ),
        ],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr


def test_worker_cli_exposes_shared_chk1_cp50_engine_group() -> None:
    parser = generation.build_parser()
    common_args = [
        "--manifest",
        "/tmp/manifest.json",
        "--output-root",
        "/tmp/output",
        "--shard-id",
        "0",
    ]
    assert (
        parser.parse_args(["worker", *common_args, "--model-id", "chk0"]).model_id
        == "chk0"
    )
    assert (
        parser.parse_args(
            ["worker", *common_args, "--model-id", generation.CHK1_CP50_GROUP_ID]
        ).model_id
        == generation.CHK1_CP50_GROUP_ID
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["worker", *common_args, "--model-id", "chk1"])


def test_fixed_a30_memory_budget_is_about_20_gib() -> None:
    assert generation.compute_gpu_memory_utilization(24_000, 24_576) == 0.83
    with pytest.raises(
        generation.PaperChk2TadleGenerationError,
        match="fixed 20 GiB budget",
    ):
        generation.compute_gpu_memory_utilization(22_000, 24_576)
    assert (
        generation.compute_gpu_memory_utilization(
            21_000,
            24_576,
            allow_busy_gpu=True,
        )
        == 0.83
    )


def test_resume_accepts_only_exact_legacy_idle_gate_implementation() -> None:
    expected = {
        "schema_version": "launch-v1",
        "mode": "smoke",
        "implementation": generation._implementation_bindings(),
    }
    legacy = {
        **expected,
        "implementation": generation._legacy_idle_gate_implementation_bindings(),
    }
    assert generation._resume_launch_contract_matches(expected, expected)
    assert generation._resume_launch_contract_matches(legacy, expected)
    assert not generation._resume_launch_contract_matches(
        {**legacy, "mode": "formal"}, expected
    )
    tampered = dict(legacy)
    tampered["implementation"] = {
        **generation._legacy_idle_gate_implementation_bindings(),
        "generation_module": {
            **generation._legacy_idle_gate_implementation_bindings()[
                "generation_module"
            ],
            "sha256": "0" * 64,
        },
    }
    assert not generation._resume_launch_contract_matches(tampered, expected)


def test_read_and_binding_helpers_reject_terminal_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}\n")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(common.PaperChk2TadlePreparationError, match="symlink"):
        common._read_json(link)
    with pytest.raises(generation.PaperChk2TadleGenerationError, match="symlink"):
        generation._read_json(link)
    with pytest.raises(generation.PaperChk2TadleGenerationError, match="symlink"):
        generation._record(link)


def _fake_inputs(tmp_path: Path) -> generation.GenerationInputs:
    ledger = []
    for meeting_rank in range(common.EXPECTED_MEETINGS):
        meeting_id = f"meeting-{meeting_rank:03d}"
        for topic_rank, topic in enumerate(common.TOPICS):
            sample_id = f"{meeting_id}-topic-{topic_rank}"
            ids = [128000, 128001 + topic_rank]
            ledger.append(
                {
                    "absolute_prompt_index": len(ledger),
                    "sample_id": sample_id,
                    "meeting_id": meeting_id,
                    "meeting_start_date": meeting_id,
                    "meeting_end_date": meeting_id,
                    "meeting_rank": meeting_rank,
                    "topic": topic,
                    "topic_rank": topic_rank,
                    "source_analysis": "Output was 2 percent.",
                    "source_analysis_sha256": common.sha256_text(
                        "Output was 2 percent."
                    ),
                    "user_prompt_sha256": "a" * 64,
                    "messages_sha256": "b" * 64,
                    "prompt_token_count": len(ids),
                    "prompt_token_ids": ids,
                    "prompt_token_ids_sha256": common.sha256_text(
                        common.canonical_json(ids)
                    ),
                }
            )
    return generation.GenerationInputs(
        manifest_path=tmp_path / "manifest.json",
        manifest_sha256="c" * 64,
        manifest={
            "population": {
                "meeting_ids": [
                    f"meeting-{index:03d}" for index in range(common.EXPECTED_MEETINGS)
                ]
            }
        },
        ledger_rows=tuple(ledger),
        model_paths={
            "chk0": Path("/chk0"),
            "chk1": Path("/chk1"),
            "paper_chk2_cp50": Path("/chk1"),
            "paper_chk2_cp50_adapter": Path("/cp50"),
        },
        tokenizer_path=Path("/chk1"),
    )


def test_real_source_projection_and_overlap_contract() -> None:
    rows, metadata = common._load_source_population()
    audit = common._build_overlap_audit(metadata)

    assert len(rows) == 672
    assert len({row["meeting_end_date"] for row in rows}) == 84
    assert {
        topic: sum(row["topic"] == topic for row in rows) for topic in common.TOPICS
    } == {topic: 84 for topic in common.TOPICS}
    assert metadata["lag_only_meeting_ids"] == ["2004-09-21", "2009-12-16"]
    assert audit["event_pair_overlap"] == {
        "estimable_events": 82,
        "post_2011_estimable_events": 30,
        "post_2011_current_or_lag_in_paper_train": 30,
        "current_and_lag_both_absent_from_paper_train": 34,
        "all_no_overlap_events_are_pre_2011": True,
    }


def test_case_matrix_is_three_model_paired_and_core8_colocated(tmp_path: Path) -> None:
    inputs = _fake_inputs(tmp_path)
    matrices = {
        model: generation.build_case_matrix(inputs, model_id=model)
        for model in common.MODEL_IDS
    }
    assert all(len(rows) == 6_720 for rows in matrices.values())

    keyed = {
        model: {row["tuple_id"]: row for row in rows}
        for model, rows in matrices.items()
    }
    assert set(keyed["chk0"]) == set(keyed["chk1"]) == set(keyed["paper_chk2_cp50"])
    for tuple_key in keyed["chk0"]:
        group = [keyed[model][tuple_key] for model in common.MODEL_IDS]
        assert len({row["row_seed"] for row in group}) == 1
        assert len({row["shard_id"] for row in group}) == 1
        assert len({row["prompt_token_ids_sha256"] for row in group}) == 1

    blocks: dict[str, list[dict[str, object]]] = {}
    for row in matrices["chk0"]:
        blocks.setdefault(str(row["paired_block_id"]), []).append(row)
    assert all(len(rows) == 8 for rows in blocks.values())
    assert all(len({row["shard_id"] for row in rows}) == 1 for rows in blocks.values())
    assert matrices["chk1"][0]["route"]["lora_request"] is None
    assert matrices["paper_chk2_cp50"][0]["route"]["lora_request"]["integer_id"] == 50


def test_group_engine_loads_base_once_then_routes_no_lora_before_cp50(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    engine_args: list[object] = []

    class FakeSamplingParams:
        def __init__(self, **_kwargs: object) -> None:
            pass

    class FakeEngineArgs:
        def __init__(self, **kwargs: object) -> None:
            engine_args.append(kwargs)

    class FakeLoRARequest:
        def __init__(self, name: str, integer_id: int, *, lora_path: str) -> None:
            self.name = name
            self.integer_id = integer_id
            self.lora_path = lora_path

    class FakeEngine:
        created = 0

        @classmethod
        def from_engine_args(cls, _args: object) -> "FakeEngine":
            cls.created += 1
            return cls()

        async def generate(
            self,
            prompt: dict[str, object],
            _params: object,
            request_id: str,
            *,
            lora_request: object,
        ):
            calls.append(lora_request)
            yield SimpleNamespace(
                prompt_token_ids=prompt["prompt_token_ids"],
                outputs=[
                    SimpleNamespace(
                        text="completion",
                        token_ids=[9],
                        finish_reason="stop",
                        stop_reason=None,
                    )
                ],
            )

        def shutdown(self) -> None:
            pass

    torch_module = types.ModuleType("torch")
    torch_module.cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
    )
    vllm_module = types.ModuleType("vllm")
    vllm_module.__version__ = generation.EXPECTED_VLLM_VERSION
    vllm_module.SamplingParams = FakeSamplingParams
    arg_module = types.ModuleType("vllm.engine.arg_utils")
    arg_module.AsyncEngineArgs = FakeEngineArgs
    engine_module = types.ModuleType("vllm.engine.async_llm_engine")
    engine_module.AsyncLLMEngine = FakeEngine
    lora_module = types.ModuleType("vllm.lora.request")
    lora_module.LoRARequest = FakeLoRARequest
    for name, module in {
        "torch": torch_module,
        "vllm": vllm_module,
        "vllm.engine.arg_utils": arg_module,
        "vllm.engine.async_llm_engine": engine_module,
        "vllm.lora.request": lora_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr(
        generation,
        "_build_result",
        lambda **kwargs: {
            "generation_key": kwargs["case"]["generation_key"],
            "model_id": kwargs["case"]["model_id"],
        },
    )
    inputs = _fake_inputs(tmp_path)
    cases_by_model = {
        model_id: [
            {
                "generation_key": f"{model_id}-key",
                "model_id": model_id,
                "row_seed": 7,
                "shard_id": 0,
                "prompt_token_ids": [1, 2],
            }
        ]
        for model_id in ("chk1", "paper_chk2_cp50")
    }
    results: list[dict[str, object]] = []
    load_events: list[dict[str, object]] = []

    async def on_result(row: dict[str, object]) -> None:
        results.append(row)

    async def on_failure(_row: dict[str, object]) -> None:
        raise AssertionError("unexpected retry")

    async def on_engine_loaded(row: dict[str, object]) -> None:
        load_events.append(row)

    report = asyncio.run(
        generation._run_chk1_cp50_engine_group(
            inputs=inputs,
            cases_by_model=cases_by_model,
            manifest_sha="c" * 64,
            tokenizer=object(),
            eos_token_ids=(2,),
            gpu_memory_utilization=0.83,
            initial_failure_counts={"chk1": {}, "paper_chk2_cp50": {}},
            shard_id=0,
            invocation_id="invocation-000",
            invocation_index=0,
            on_engine_loaded=on_engine_loaded,
            on_failure=on_failure,
            on_result=on_result,
        )
    )
    assert FakeEngine.created == 1
    assert len(engine_args) == 1
    assert engine_args[0]["enable_lora"] is True
    assert engine_args[0]["max_loras"] == 1
    assert calls[0] is None
    assert isinstance(calls[1], FakeLoRARequest)
    assert calls[1].integer_id == 50
    assert [row["model_id"] for row in results] == ["chk1", "paper_chk2_cp50"]
    assert report["model_load_count"] == 1
    assert len(load_events) == 1


def test_cross_model_tuple_drift_fails_closed() -> None:
    rows = [
        {
            "model_id": model,
            "row_seed": 11,
            "shard_id": 0,
            "prompt_token_ids_sha256": "a" * 64,
            "output_token_ids": [index],
        }
        for index, model in enumerate(common.MODEL_IDS)
    ]
    assert generation._validate_paired_tuple_group("tuple", rows) is True
    rows[1]["row_seed"] = 12
    with pytest.raises(generation.PaperChk2TadleGenerationError, match="seed drift"):
        generation._validate_paired_tuple_group("tuple", rows)


def test_smoke_matrix_is_32_rows_per_model(tmp_path: Path) -> None:
    inputs = _fake_inputs(tmp_path)
    rows = generation.build_case_matrix(inputs, model_id="chk0", smoke=True)
    assert len(rows) == 32
    assert len({row["meeting_id"] for row in rows}) == 2
    assert {row["replicate_id"] for row in rows} == {0, 1}


def test_strict_boundary_and_diagnostics_do_not_filter() -> None:
    valid = generation._diagnostics(
        completion="Preserve the quantity.\n</think>\nOutput was 2 percent.",
        output_token_ids=[1, 2, 3, 4, 5],
        finish_reason="stop",
        source_analysis="Output was 2 percent.",
    )
    assert valid["answer"] == "Output was 2 percent."
    assert valid["think_boundary_count"] == 1
    assert valid["diagnostics_used_for_filtering_or_resampling"] is False

    invalid = generation._diagnostics(
        completion="Output was 2 percent.",
        output_token_ids=[1, 2, 3],
        finish_reason="length",
        source_analysis="Output was 2 percent.",
    )
    assert invalid["answer"] == ""
    assert invalid["diagnostic_pass"] is False
    assert "think_boundary_count_not_one" in invalid["diagnostic_failures"]


def test_attempt_ledger_repairs_only_final_torn_fragment(tmp_path: Path) -> None:
    inputs = _fake_inputs(tmp_path)
    case = generation.build_case_matrix(inputs, model_id="chk0", smoke=True)[0]
    path = tmp_path / "attempts.wal.jsonl"
    event = {
        "schema_version": generation.ATTEMPT_SCHEMA,
        "evaluation_manifest_sha256": inputs.manifest_sha256,
        "generation_key": case["generation_key"],
        "model_id": "chk0",
        "shard_id": case["shard_id"],
        "row_seed": case["row_seed"],
        "attempt_index": 0,
        "request_id": "request-0",
        "status": "transport_failure",
        "error_type": "RuntimeError",
        "error_message_sha256": "d" * 64,
        "route": case["route"],
        "failed_at_utc": "2026-09-02T00:00:00Z",
    }
    path.write_bytes((common.canonical_json(event) + '\n{"broken":').encode())
    loaded = generation.load_attempt_ledger(
        path,
        cases_by_key={str(case["generation_key"]): case},
        manifest_sha=inputs.manifest_sha256,
    )
    assert len(loaded[str(case["generation_key"])]) == 1
    assert path.read_bytes().endswith(b"\n")
    assert (tmp_path / "attempts.wal.jsonl.torn-tail-recoveries.jsonl").is_file()


def test_retry_budget_exhaustion_fails_closed() -> None:
    assert generation._retry_start_index("key", 0) == 0
    assert generation._retry_start_index("key", 2) == 2
    with pytest.raises(
        generation.PaperChk2TadleGenerationError, match="retry budget exhausted"
    ):
        generation._retry_start_index("key", 3)


def test_completed_worker_resume_is_byte_stable_and_remains_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "evaluation"
    manifest_path = output_root / "preparation/evaluation_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}\n")
    inputs = generation.GenerationInputs(
        manifest_path=manifest_path.resolve(),
        manifest_sha256="c" * 64,
        manifest={"population": {"meeting_ids": ["m1"]}},
        ledger_rows=(),
        model_paths={
            "chk0": Path("/chk0"),
            "chk1": Path("/chk1"),
            "paper_chk2_cp50": Path("/chk1"),
            "paper_chk2_cp50_adapter": Path("/cp50"),
        },
        tokenizer_path=Path("/chk1"),
    )
    route = generation._route_for_model("chk0")
    case = {
        "generation_key": "chk0::m1::topic::0",
        "shard_id": 0,
        "route": route,
    }
    row = {
        "generation_key": case["generation_key"],
        "attempt_index": 0,
        "route": route,
        "diagnostic_pass": True,
    }
    shard_root = output_root / "smoke/chk0/shard-0"
    shard_root.mkdir(parents=True)
    expected_launch = {
        "schema_version": "paper-chk2-tadle-lm-k10-worker-launch-v1",
        "evaluation_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": inputs.manifest_sha256,
        },
        "mode": "smoke",
        "model_id": "chk0",
        "shard_id": 0,
        "physical_gpu_index": 0,
        "route": route,
        "implementation": generation._implementation_bindings(),
        "generation_contract": common.generation_contract(),
        "replicate_seeds": list(common.REPLICATE_SEEDS[:2]),
        "formal_smoke_authorization": None,
    }
    (shard_root / "launch.json").write_text(
        json.dumps({"contract": expected_launch}) + "\n"
    )
    (shard_root / "generations.wal.jsonl").write_text("{}\n")
    (shard_root / "attempts.wal.jsonl").write_text("")
    (shard_root / "generations.canonical.jsonl").write_text(
        common.canonical_json(row) + "\n"
    )
    state = {
        "schema_version": generation.SHARD_SCHEMA,
        "status": "complete",
        "model_id": "chk0",
        "shard_id": 0,
        "expected_rows": 1,
        "completed_rows": 1,
    }
    (shard_root / "state.json").write_text(json.dumps(state) + "\n")
    shard_manifest = {
        "schema_version": generation.SHARD_SCHEMA,
        "status": "complete",
        "mode": "smoke",
        "evaluation_manifest_sha256": inputs.manifest_sha256,
        "model_id": "chk0",
        "shard_id": 0,
        "expected_rows": 1,
        "completed_rows": 1,
        "new_rows": 1,
        "generation_wall_seconds": 2.0,
        "evaluation_manifest": generation._record(manifest_path),
        "worker_launch": generation._record(shard_root / "launch.json"),
        "worker_state": generation._record(shard_root / "state.json"),
        "generation_wal": generation._record(
            shard_root / "generations.wal.jsonl", rows=1
        ),
        "attempt_wal": generation._record(shard_root / "attempts.wal.jsonl", rows=0),
        "canonical": generation._record(
            shard_root / "generations.canonical.jsonl", rows=1
        ),
    }
    (shard_root / "manifest.json").write_text(json.dumps(shard_manifest) + "\n")

    monkeypatch.setattr(generation, "SHARD_IDS", (0,))
    monkeypatch.setattr(generation, "_validate_visible_gpu", lambda _shard: None)
    monkeypatch.setattr(
        generation, "load_generation_inputs", lambda *_args, **_kwargs: inputs
    )
    monkeypatch.setattr(
        generation, "_load_tokenizer", lambda _path: (object(), (2,), 2)
    )
    monkeypatch.setattr(
        generation,
        "build_case_matrix",
        lambda *_args, **_kwargs: [case],
    )
    monkeypatch.setattr(
        generation,
        "load_generation_wal",
        lambda *_args, **_kwargs: {case["generation_key"]: row},
    )
    monkeypatch.setattr(generation, "load_attempt_ledger", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        generation,
        "_delivery_funnel",
        lambda rows: {"rows": len(rows)},
    )

    before = {path: path.read_bytes() for path in shard_root.iterdir()}
    resumed = generation.run_worker(
        manifest_path=manifest_path,
        output_root=output_root,
        shard_id=0,
        model_id="chk0",
        smoke=True,
        resume=True,
    )
    after = {path: path.read_bytes() for path in shard_root.iterdir()}
    assert resumed == shard_manifest
    assert before == after
    validated = generation.validate_model(
        manifest_path=manifest_path,
        output_root=output_root,
        model_id="chk0",
        smoke=True,
    )
    assert validated["generation_rows"] == 1
    assert validated["shards"]["shard-0"]["worker_manifest"][
        "sha256"
    ] == generation.sha256_file(shard_root / "manifest.json")


def test_group_worker_seals_two_models_from_one_ordered_engine_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "evaluation"
    manifest_path = output_root / "preparation/evaluation_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}\n")
    inputs = generation.GenerationInputs(
        manifest_path=manifest_path.resolve(),
        manifest_sha256="d" * 64,
        manifest={"population": {"meeting_ids": ["m1"]}},
        ledger_rows=(),
        model_paths={
            "chk0": Path("/chk0"),
            "chk1": Path("/chk1"),
            "paper_chk2_cp50": Path("/chk1"),
            "paper_chk2_cp50_adapter": Path("/cp50"),
        },
        tokenizer_path=Path("/chk1"),
    )
    cases = {
        model_id: [
            {
                "generation_key": f"{model_id}::m1::topic::0",
                "tuple_id": "m1::topic::0",
                "model_id": model_id,
                "shard_id": 0,
                "row_seed": 11,
                "route": generation._route_for_model(model_id),
            }
        ]
        for model_id in common.MODEL_IDS
    }
    engine_calls: list[tuple[str, ...]] = []

    def fake_load_generation_wal(
        path: Path, **_kwargs: object
    ) -> dict[str, dict[str, object]]:
        if not path.exists():
            return {}
        rows = generation._read_jsonl(path)
        if any("generation_key" not in row for row in rows):
            raise generation.PaperChk2TadleGenerationError(
                "underlying smoke WAL binding drift"
            )
        return {str(row["generation_key"]): row for row in rows}

    async def fake_group_engine(**kwargs: object) -> dict[str, object]:
        cases_by_model = kwargs["cases_by_model"]
        assert isinstance(cases_by_model, dict)
        engine_calls.append(tuple(cases_by_model))
        await kwargs["on_engine_loaded"](
            {
                "schema_version": generation.ENGINE_INVOCATION_SCHEMA,
                "evaluation_manifest_sha256": inputs.manifest_sha256,
                "engine_group_id": generation.CHK1_CP50_GROUP_ID,
                "shard_id": 0,
                "invocation_id": kwargs["invocation_id"],
                "invocation_index": kwargs["invocation_index"],
                "model_order": ["chk1", "paper_chk2_cp50"],
                "pending_rows_by_model": {
                    model_id: len(cases_by_model[model_id])
                    for model_id in ("chk1", "paper_chk2_cp50")
                },
                "model_load_count": 1,
            }
        )
        per_model: dict[str, dict[str, object]] = {}
        for model_id in ("chk1", "paper_chk2_cp50"):
            selected = cases_by_model[model_id]
            for case in selected:
                await kwargs["on_result"](
                    {
                        "generation_key": case["generation_key"],
                        "tuple_id": case["tuple_id"],
                        "model_id": model_id,
                        "shard_id": 0,
                        "row_seed": 11,
                        "route": case["route"],
                        "attempt_index": 0,
                        "engine_invocation_id": case["engine_invocation_id"],
                        "prompt_token_ids_sha256": "a" * 64,
                        "output_token_ids": [1 if model_id == "chk1" else 2],
                        "diagnostic_pass": True,
                    }
                )
            per_model[model_id] = {
                "new_rows": len(selected),
                "generation_wall_seconds": 1.0,
                "lora_request": generation._route_for_model(model_id)["lora_request"],
            }
        return {
            "engine_group_id": generation.CHK1_CP50_GROUP_ID,
            "model_order": ["chk1", "paper_chk2_cp50"],
            "model_load_count": 1,
            "model_load_wall_seconds": 2.0,
            "generation_wall_seconds": 2.0,
            "per_model": per_model,
        }

    async def fake_single_engine(**kwargs: object) -> dict[str, object]:
        for case in kwargs["cases"]:
            await kwargs["on_result"](
                {
                    "generation_key": case["generation_key"],
                    "tuple_id": case["tuple_id"],
                    "model_id": "chk0",
                    "shard_id": 0,
                    "row_seed": 11,
                    "route": case["route"],
                    "attempt_index": 0,
                    "prompt_token_ids_sha256": "a" * 64,
                    "output_token_ids": [0],
                    "diagnostic_pass": True,
                }
            )
        return {
            "model_load_wall_seconds": 1.0,
            "generation_wall_seconds": 1.0,
            "new_rows": len(kwargs["cases"]),
        }

    monkeypatch.setattr(generation, "SHARD_IDS", (0,))
    monkeypatch.setattr(generation, "SMOKE_MEETINGS", 1)
    monkeypatch.setattr(generation, "SMOKE_REPLICATES", 1)
    monkeypatch.setattr(common, "TOPICS", ("topic",))
    monkeypatch.setattr(generation, "_validate_visible_gpu", lambda _shard: None)
    monkeypatch.setattr(
        generation, "load_generation_inputs", lambda *_args, **_kwargs: inputs
    )
    monkeypatch.setattr(
        generation, "_load_tokenizer", lambda _path: (object(), (2,), 2)
    )
    monkeypatch.setattr(
        generation,
        "build_case_matrix",
        lambda _inputs, *, model_id, smoke: cases[model_id],
    )
    monkeypatch.setattr(generation, "load_generation_wal", fake_load_generation_wal)
    monkeypatch.setattr(generation, "load_attempt_ledger", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        generation,
        "gpu_snapshot",
        lambda _shard: {
            "vllm_gpu_memory_utilization": 0.83,
            "vllm_target_memory_mib": 20_480,
        },
    )
    monkeypatch.setattr(
        generation,
        "_run_chk1_cp50_engine_group",
        fake_group_engine,
    )
    monkeypatch.setattr(generation, "_run_engine", fake_single_engine)
    monkeypatch.setattr(
        generation,
        "_delivery_funnel",
        lambda rows: {"rows": len(rows)},
    )

    generation.run_worker(
        manifest_path=manifest_path,
        output_root=output_root,
        shard_id=0,
        model_id="chk0",
        smoke=True,
    )
    result = generation.run_chk1_cp50_group_worker(
        manifest_path=manifest_path,
        output_root=output_root,
        shard_id=0,
        smoke=True,
    )
    assert result["cumulative_model_load_count"] == 1
    assert result["uninterrupted_single_shared_load"] is True
    assert engine_calls == [("chk1", "paper_chk2_cp50")]
    for model_id in ("chk1", "paper_chk2_cp50"):
        validated = generation.validate_model(
            manifest_path=manifest_path,
            output_root=output_root,
            model_id=model_id,
            smoke=True,
        )
        assert validated["generation_rows"] == 1
        worker_manifest = generation._read_json(
            output_root / f"smoke/{model_id}/shard-0/manifest.json"
        )
        assert worker_manifest["shared_base_model_load"] is True
        assert worker_manifest["engine_group_id"] == generation.CHK1_CP50_GROUP_ID

    smoke_receipt = generation.validate_all_models(
        manifest_path=manifest_path,
        output_root=output_root,
        smoke=True,
    )
    authorized = generation._smoke_authorization(output_root, inputs.manifest_sha256)
    assert authorized["path"] == str((output_root / "smoke/validation.json").resolve())
    assert smoke_receipt["generation_rows"] == 3

    tampered_wal = output_root / "smoke/chk1/shard-0/generations.wal.jsonl"
    pristine_wal = tampered_wal.read_bytes()
    with tampered_wal.open("ab") as handle:
        handle.write(b"{}\n")
    with pytest.raises(
        generation.PaperChk2TadleGenerationError,
        match="incomplete|binding drift",
    ):
        generation._smoke_authorization(output_root, inputs.manifest_sha256)
    tampered_wal.write_bytes(pristine_wal)

    sealed_paths = [
        output_root / "smoke/engine-groups/chk1_paper_chk2_cp50/shard-0/manifest.json",
        output_root / "smoke/chk1/shard-0/manifest.json",
        output_root / "smoke/paper_chk2_cp50/shard-0/manifest.json",
    ]
    before = {path: path.read_bytes() for path in sealed_paths}
    resumed = generation.run_chk1_cp50_group_worker(
        manifest_path=manifest_path,
        output_root=output_root,
        shard_id=0,
        smoke=True,
        resume=True,
    )
    assert resumed == result
    assert {path: path.read_bytes() for path in sealed_paths} == before
    assert len(engine_calls) == 1

    # Simulate a crash after both WALs closed but before any receipt publish.
    for path in sealed_paths:
        path.unlink()
    for model_id in ("chk1", "paper_chk2_cp50"):
        validation_path = output_root / f"smoke/{model_id}/validation.json"
        if validation_path.exists():
            validation_path.unlink()
    original_write_new_json = generation._write_new_json
    injected = {"raised": False}

    def fail_after_first_model_receipt(path: Path, value: dict[str, object]) -> None:
        if (
            path.name == "manifest.json"
            and path.parent.parent.name == "paper_chk2_cp50"
            and not injected["raised"]
        ):
            injected["raised"] = True
            raise RuntimeError("injected crash after chk1 receipt")
        original_write_new_json(path, value)

    monkeypatch.setattr(generation, "_write_new_json", fail_after_first_model_receipt)
    with pytest.raises(RuntimeError, match="injected crash"):
        generation.run_chk1_cp50_group_worker(
            manifest_path=manifest_path,
            output_root=output_root,
            shard_id=0,
            smoke=True,
            resume=True,
        )
    first_receipt_path = output_root / "smoke/chk1/shard-0/manifest.json"
    first_receipt_bytes = first_receipt_path.read_bytes()
    monkeypatch.setattr(generation, "_write_new_json", original_write_new_json)
    recovered = generation.run_chk1_cp50_group_worker(
        manifest_path=manifest_path,
        output_root=output_root,
        shard_id=0,
        smoke=True,
        resume=True,
    )
    assert recovered["receipt_recovered_from_closed_wal"] is True
    assert first_receipt_path.read_bytes() == first_receipt_bytes
    assert len(engine_calls) == 1
    generation.validate_chk1_cp50_group_shard(
        manifest_path=manifest_path,
        output_root=output_root,
        shard_id=0,
        smoke=True,
    )

    # A process death between arms requires a second base-model load on resume.
    # The completed chk1 row must not be regenerated, and the final provenance
    # must not claim that both arms shared one uninterrupted invocation.
    for path in sealed_paths:
        path.unlink()
    group_root = output_root / "smoke/engine-groups/chk1_paper_chk2_cp50/shard-0"
    for path in (group_root / "state.json", group_root / "invocations.wal.jsonl"):
        path.unlink()
    for model_id in ("chk1", "paper_chk2_cp50"):
        model_root = output_root / f"smoke/{model_id}/shard-0"
        for name in (
            "generations.wal.jsonl",
            "generations.canonical.jsonl",
            "attempts.wal.jsonl",
            "state.json",
        ):
            path = model_root / name
            if path.exists():
                path.unlink()
        validation_path = model_root.parent / "validation.json"
        if validation_path.exists():
            validation_path.unlink()

    between_arm_calls: list[dict[str, int]] = []

    async def crash_between_arms(**kwargs: object) -> dict[str, object]:
        selected = kwargs["cases_by_model"]
        assert isinstance(selected, dict)
        between_arm_calls.append(
            {model_id: len(selected[model_id]) for model_id in selected}
        )
        await kwargs["on_engine_loaded"](
            {
                "schema_version": generation.ENGINE_INVOCATION_SCHEMA,
                "evaluation_manifest_sha256": inputs.manifest_sha256,
                "engine_group_id": generation.CHK1_CP50_GROUP_ID,
                "shard_id": 0,
                "invocation_id": kwargs["invocation_id"],
                "invocation_index": kwargs["invocation_index"],
                "model_order": ["chk1", "paper_chk2_cp50"],
                "pending_rows_by_model": {
                    model_id: len(selected[model_id]) for model_id in selected
                },
                "model_load_count": 1,
            }
        )
        for model_id in ("chk1", "paper_chk2_cp50"):
            for case in selected[model_id]:
                await kwargs["on_result"](
                    {
                        "generation_key": case["generation_key"],
                        "tuple_id": case["tuple_id"],
                        "model_id": model_id,
                        "shard_id": 0,
                        "row_seed": 11,
                        "route": case["route"],
                        "attempt_index": 0,
                        "engine_invocation_id": case["engine_invocation_id"],
                        "prompt_token_ids_sha256": "a" * 64,
                        "output_token_ids": [1 if model_id == "chk1" else 2],
                        "diagnostic_pass": True,
                    }
                )
            if len(between_arm_calls) == 1 and model_id == "chk1":
                raise RuntimeError("injected death between grouped arms")
        return {
            "engine_group_id": generation.CHK1_CP50_GROUP_ID,
            "model_order": ["chk1", "paper_chk2_cp50"],
            "model_load_count": 1,
            "model_load_wall_seconds": 2.0,
            "generation_wall_seconds": 2.0,
            "per_model": {
                model_id: {
                    "new_rows": len(selected[model_id]),
                    "generation_wall_seconds": 1.0,
                    "lora_request": generation._route_for_model(model_id)[
                        "lora_request"
                    ],
                }
                for model_id in ("chk1", "paper_chk2_cp50")
            },
        }

    monkeypatch.setattr(generation, "_run_chk1_cp50_engine_group", crash_between_arms)
    with pytest.raises(RuntimeError, match="death between grouped arms"):
        generation.run_chk1_cp50_group_worker(
            manifest_path=manifest_path,
            output_root=output_root,
            shard_id=0,
            smoke=True,
            resume=True,
        )
    closed = generation.run_chk1_cp50_group_worker(
        manifest_path=manifest_path,
        output_root=output_root,
        shard_id=0,
        smoke=True,
        resume=True,
    )
    assert between_arm_calls == [
        {"chk1": 1, "paper_chk2_cp50": 1},
        {"chk1": 0, "paper_chk2_cp50": 1},
    ]
    invocation_rows = generation._read_jsonl(group_root / "invocations.wal.jsonl")
    assert len(invocation_rows) == 2
    assert closed["cumulative_model_load_count"] == 2
    assert closed["uninterrupted_single_shared_load"] is False
    assert (
        closed["arm_invocation_ids"]["chk1"]
        != closed["arm_invocation_ids"]["paper_chk2_cp50"]
    )
    for model_id in ("chk1", "paper_chk2_cp50"):
        assert (
            len(
                generation._read_jsonl(
                    output_root / f"smoke/{model_id}/shard-0/generations.wal.jsonl"
                )
            )
            == 1
        )
        generation.validate_model(
            manifest_path=manifest_path,
            output_root=output_root,
            model_id=model_id,
            smoke=True,
        )
    generation.validate_chk1_cp50_group_shard(
        manifest_path=manifest_path,
        output_root=output_root,
        shard_id=0,
        smoke=True,
    )


def test_smoke_authorization_requires_complete_all_model_seal(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    path = smoke / "validation.json"
    value = {
        "schema_version": generation.ALL_VALIDATION_SCHEMA,
        "status": "complete",
        "mode": "smoke",
        "evaluation_manifest_sha256": "a" * 64,
        "gates": {gate: True for gate in generation.SMOKE_AUTHORIZATION_GATES},
    }
    path.write_text(json.dumps(value) + "\n")
    with pytest.raises(
        common.PaperChk2TadlePreparationError,
        match="preparation",
    ):
        generation._smoke_authorization(tmp_path, "a" * 64)


def test_preparation_lock_serializes_processes(tmp_path: Path) -> None:
    """Linux flock prevents two publishers from entering the rename window."""

    root = tmp_path / "locked"
    root.mkdir()
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - assertion is performed by the parent
        os.close(read_fd)
        with common._preparation_lock(root):
            os.write(write_fd, b"1")
            time.sleep(0.35)
        os.close(write_fd)
        os._exit(0)
    os.close(write_fd)
    try:
        assert os.read(read_fd, 1) == b"1"
        started = time.monotonic()
        with common._preparation_lock(root):
            elapsed = time.monotonic() - started
        assert elapsed >= 0.25
    finally:
        os.close(read_fd)
        _, status = os.waitpid(pid, 0)
        assert status == 0


def test_prepare_is_resume_idempotent_and_rejects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    meetings = [f"m{index:02d}" for index in range(84)]
    ledger = []
    for meeting_rank, meeting in enumerate(meetings):
        for topic_rank, topic in enumerate(common.TOPICS):
            token_count = 282
            if meeting_rank == 83 and topic_rank == 7:
                token_count = 378
            row = {
                "schema_version": common.LEDGER_ROW_SCHEMA,
                "absolute_prompt_index": len(ledger),
                "sample_id": f"{meeting}-{topic_rank}",
                "meeting_id": meeting,
                "topic": topic,
                "topic_rank": topic_rank,
                "prompt_token_count": token_count,
                "prompt_token_ids": list(range(token_count)),
                "prompt_token_ids_sha256": common.sha256_text(
                    common.canonical_json(list(range(token_count)))
                ),
                "official_minutes_in_model_input": False,
                "statement_in_model_input": False,
            }
            row["row_sha256"] = common.sha256_text(common.canonical_json(row))
            ledger.append(row)
    source_meta = {
        "needed_meeting_ids": meetings,
        "estimable_current_meeting_ids": meetings[1:83],
        "lag_only_meeting_ids": [meetings[0], meetings[20]],
        "standardization_meeting_ids": meetings[1:],
        "release_meeting_ids": meetings[1:],
    }
    overlap = {
        "schema_version": common.OVERLAP_SCHEMA,
        "counts": {"train": {"standardization_83": 47}},
    }
    paper_bindings = SimpleNamespace(
        file_bindings={"prompt_contract": {"path": "/prompt", "sha256": "a" * 64}},
        system_prompt_sha256="b" * 64,
        user_template_sha256="c" * 64,
    )
    model_bindings = {
        "chk0": {},
        "chk1": {},
        "paper_chk2_cp50": {},
    }
    monkeypatch.setattr(common, "_load_source_population", lambda: ([], source_meta))
    monkeypatch.setattr(common, "_build_ledger", lambda *_args: ledger)
    monkeypatch.setattr(common, "_build_overlap_audit", lambda _meta: overlap)
    monkeypatch.setattr(common, "_model_manifest_bindings", lambda: model_bindings)
    monkeypatch.setattr(
        common.paper_contract,
        "verify_paper_chk2_bindings",
        lambda **_kwargs: paper_bindings,
    )

    root = tmp_path / "prepared"
    first = common.build_preparation_artifacts(root)
    second = common.build_preparation_artifacts(root)
    assert first.manifest_path == second.manifest_path
    assert len(second.ledger_rows) == 672

    with (root / "preparation/atomic_prompt_ledger.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(common.PaperChk2TadlePreparationError, match="ledger SHA drift"):
        common.build_preparation_artifacts(root)


def test_document_assembly_exact_topic_order_and_empty_answer_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "evaluation"
    (output_root / "preparation").mkdir(parents=True)
    (output_root / "preparation/evaluation_manifest.json").write_text("{}\n")
    (output_root / "generation").mkdir()
    (output_root / "generation/validation.json").write_text("{}\n")

    monkeypatch.setattr(common, "MODEL_IDS", ("chk0", "chk1", "paper_chk2_cp50"))
    monkeypatch.setattr(common, "TOPICS", ("topic-a", "topic-b"))
    monkeypatch.setattr(common, "REPLICATE_SEEDS", (101,))
    monkeypatch.setattr(common, "EXPECTED_MEETINGS", 2)
    monkeypatch.setattr(common, "EXPECTED_GENERATIONS", 12)
    monkeypatch.setattr(common, "EXPECTED_DOCUMENTS", 6)
    monkeypatch.setattr(
        common,
        "load_prepared",
        lambda _root: SimpleNamespace(
            manifest={"population": {"meeting_ids": ["m1", "m2"]}}
        ),
    )
    monkeypatch.setattr(
        generation,
        "validate_all_models",
        lambda **_kwargs: {"status": "complete"},
    )
    rows = []
    for model in common.MODEL_IDS:
        for meeting in ("m1", "m2"):
            for topic_rank, topic in enumerate(common.TOPICS):
                answer = (
                    ""
                    if (model == "chk0" and meeting == "m1" and topic_rank == 0)
                    else f"{model}-{meeting}-{topic}"
                )
                rows.append(
                    {
                        "model_id": model,
                        "meeting_id": meeting,
                        "meeting_start_date": meeting,
                        "replicate_id": 0,
                        "replicate_seed": 101,
                        "topic": topic,
                        "topic_rank": topic_rank,
                        "answer": answer,
                        "generation_key": f"{model}|{meeting}|{topic}",
                        "diagnostic_pass": bool(answer),
                    }
                )
    (output_root / "generation/generation_rows.jsonl").write_text(
        "".join(common.canonical_json(row) + "\n" for row in rows)
    )
    result = generation.assemble_generated_documents(output_root=output_root)
    assert result["documents"] == 6
    resumed = generation.assemble_generated_documents(
        output_root=output_root, resume=True
    )
    assert resumed["generated_documents"] == result["generated_documents"]
    documents = [
        json.loads(line)
        for line in (output_root / "documents/generated_documents.jsonl")
        .read_text()
        .splitlines()
    ]
    first = next(
        row for row in documents if row["arm"] == "chk0" and row["meeting_id"] == "m1"
    )
    assert first["section_count"] == 2
    assert first["nonempty_section_count"] == 1
    assert first["document_text"].startswith("\n\nchk0-m1-topic-b")
    assert first["assembly_separator"] == "two_newlines_exact_section_answers"
