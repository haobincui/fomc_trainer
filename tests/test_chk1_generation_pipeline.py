from __future__ import annotations

import hashlib
import json
import threading
from datetime import date, timedelta
from pathlib import Path

import pytest

import jobs.retrain_v2.chk1.generation_pipeline as generation_pipeline_module
from jobs.main.checkpoint_provenance import fingerprint_tokenizer_payload
from jobs.retrain_v2.chk1.contracts import (
    canonical_json,
    render_generator_payload,
    render_student_prompt,
    sha256_text,
)
from jobs.retrain_v2.chk1.generation_pipeline import (
    PILOT_SAMPLE_COUNT,
    SMOKE_SAMPLE_COUNT,
    PreparedDataError,
    ResumeIntegrityError,
    SelectionError,
    run_generation_pipeline,
    select_pilot_rows,
    select_smoke_rows,
)
from jobs.retrain_v2.chk1.deepseek_teacher import TeacherResponseRejectedError
from jobs.retrain_v2.chk1.pipeline import (
    MINUTES_REFERENCE_HASH_POLICY,
    PREPARE_HANDOFF_SCHEMA_VERSION,
    PREPARED_SAMPLE_SCHEMA_VERSION,
    compute_minutes_reference_sha256,
)


DIGEST = "a" * 64
TOPICS = ("GDP Growth", "Unemployment Rate", "Treasury Yields")
DEFAULT_MINUTES = "same-sample Minutes reference used only at runtime"


def _model_repo(root: Path) -> tuple[str, str]:
    for name in ("DeepSeek-R1-Distill-Llama-8B",):
        model = root / "models" / name
        model.mkdir(parents=True)
        (model / "config.json").write_text(
            json.dumps({"architectures": ["TestModel"]}), encoding="utf-8"
        )
        (model / "model.safetensors").write_bytes(f"weights:{name}".encode())
        (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (model / "tokenizer.json").write_text(
            json.dumps({"model": name}), encoding="utf-8"
        )
    student = fingerprint_tokenizer_payload(
        root / "models" / "DeepSeek-R1-Distill-Llama-8B"
    )
    return student["sha256"], student["sha256"]


def _prepared_rows(
    count: int,
    *,
    minutes: str | tuple[str, ...] = DEFAULT_MINUTES,
    generator_tokenizer_sha256: str = DIGEST,
    student_tokenizer_sha256: str = DIGEST,
) -> list[dict]:
    start = date(2010, 1, 1)
    rows: list[dict] = []
    for index in range(count):
        meeting = start + timedelta(days=index)
        topic = TOPICS[index % len(TOPICS)]
        meeting_text = meeting.isoformat()
        sample_id = f"sample-{index:04d}"
        cutoff = f"{meeting_text}T23:59:59Z"
        evidence = {
            "evidence_id": "E1",
            "source_id": "TEST_SERIES",
            "value": "1.0",
            "units": "Percent",
            "metric": "test metric",
            "fact_kind": "latest",
            "observation_date": (meeting - timedelta(days=30)).isoformat(),
            "formula": None,
            "operand_evidence_ids": [],
            "release_ts": f"{(meeting - timedelta(days=1)).isoformat()}T12:00:00Z",
            "availability_upper_bound_ts": None,
            "availability_basis": "actual_release_ts",
            "requested_vintage_date": (meeting - timedelta(days=1)).isoformat(),
            "cutoff_ts": cutoff,
            "source_sha256": DIGEST,
        }
        fact_card = {
            "schema_version": "chk1-fact-card-v1",
            "sample_id": sample_id,
            "canonical_key": {
                "meeting_date": meeting_text,
                "atomic_topic": topic,
            },
            "meeting_date": meeting_text,
            "atomic_topic": topic,
            "cutoff_ts": cutoff,
            "evidence": [evidence],
        }
        lineage = {
            "evidence_id": "E1",
            "evidence_sha256": DIGEST,
            "source_id": "TEST_SERIES",
            "source_kind": "macro",
            "source_interface": "alfred-series-observations-csv-v1",
            "availability_evidence_type": "release_calendar",
            "source_sha256": DIGEST,
            "raw_sha256": DIGEST,
            "request_id": DIGEST,
            "snapshot_manifest_payload_sha256": DIGEST,
            "registry_sha256": DIGEST,
            "release_ts": evidence["release_ts"],
            "availability_upper_bound_ts": None,
            "availability_basis": "actual_release_ts",
            "requested_vintage_date": evidence["requested_vintage_date"],
            "information_as_of_date": evidence["requested_vintage_date"],
            "availability_as_of_date": evidence["requested_vintage_date"],
            "cutoff_ts": cutoff,
        }
        style = {
            "section_style_id": f"style-{index % 2}",
            "guide_text": "Use concise, neutral evidence comparisons.",
        }
        generator_prompt = render_generator_payload(
            fact_card=fact_card,
            atomic_topic=topic,
            style_guide=style,
        )
        student_prompt = render_student_prompt(
            fact_card=fact_card,
            atomic_topic=topic,
            style_guide=style,
        )
        provided_data = canonical_json(fact_card)
        rows.append(
            {
                "schema_version": PREPARED_SAMPLE_SCHEMA_VERSION,
                "sample_id": sample_id,
                "split": ("train", "eval", "test")[index % 3],
                "meeting_date": meeting_text,
                "atomic_topic": topic,
                "section_style_id": f"style-{index % 2}",
                "cutoff_ts": cutoff,
                "fact_card": fact_card,
                "evidence_lineage": [lineage],
                "section_style_guide": style,
                "generator_prompt": generator_prompt,
                "student_prompt": student_prompt,
                "provided_data": provided_data,
                "minutes_reference_sha256": compute_minutes_reference_sha256(minutes),
                "prompt_budget": {
                    "generator": {
                        "schema_version": "prompt-token-budget-v1",
                        "prompt_sha256": sha256_text(generator_prompt),
                        "token_count": 100,
                        "max_tokens": 4096,
                        "overflow_policy": "error",
                        "truncated": False,
                        "tokenizer_sha256": generator_tokenizer_sha256,
                    },
                    "student": {
                        "schema_version": "prompt-token-budget-v1",
                        "prompt_sha256": sha256_text(student_prompt),
                        "token_count": 100,
                        "max_tokens": 4096,
                        "overflow_policy": "error",
                        "truncated": False,
                        "tokenizer_sha256": student_tokenizer_sha256,
                    },
                },
                "style_guide_sha256": DIGEST,
                "input_truncated": False,
                "generator_fact_card_sha256": sha256_text(provided_data),
            }
        )
    return rows


def _runtime_rows(
    root: Path,
    count: int,
    *,
    minutes: str | tuple[str, ...] = DEFAULT_MINUTES,
) -> list[dict]:
    generator_tokenizer, student_tokenizer = _model_repo(root)
    return _prepared_rows(
        count,
        minutes=minutes,
        generator_tokenizer_sha256=generator_tokenizer,
        student_tokenizer_sha256=student_tokenizer,
    )


def _prepare_bundle(
    root: Path,
    rows: list[dict],
    *,
    generator_tokenizer_sha256: str,
    student_tokenizer_sha256: str,
) -> Path:
    bundle = root / "prepared-bundle"

    def file_sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def record(path: Path, *, row_count: int | None = None) -> dict:
        value = {
            "path": path.relative_to(bundle).as_posix(),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        if row_count is not None:
            value["row_count"] = row_count
        return value

    prepared_records: dict[str, dict] = {}
    for split in ("train", "eval", "test"):
        path = bundle / "prepared" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        split_rows = []
        for source in rows:
            if source["split"] != split:
                continue
            body = dict(source)
            split_rows.append(
                {
                    **body,
                    "row_sha256": sha256_text(canonical_json(body)),
                }
            )
        path.write_text(
            "".join(canonical_json(row) + "\n" for row in split_rows),
            encoding="utf-8",
        )
        prepared_records[split] = record(path, row_count=len(split_rows))

    static_records: dict[str, dict] = {}
    for field, relative in (
        ("inventory", "inventory/legacy_inventory.json"),
        ("style_guide", "style/minutes_style_guide.json"),
        ("canonical_population", "canonical/canonical_population.json"),
    ):
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        static_records[field] = record(path)

    audit_records: dict[str, dict] = {}
    for field in (
        "evidence_exclusions",
        "sample_exclusions",
        "precanonical_exclusions",
    ):
        path = bundle / "audit" / f"{field}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        audit_records[field] = record(path, row_count=0)

    payload = {
        "schema_version": PREPARE_HANDOFF_SCHEMA_VERSION,
        "source_handoff": {"sha256": DIGEST, "payload_sha256": DIGEST},
        "inventory_payload_sha256": DIGEST,
        "canonical_population_payload_sha256": DIGEST,
        "style_guide_sha256": DIGEST,
        "minutes_reference_hash_policy": MINUTES_REFERENCE_HASH_POLICY,
        "generator_tokenizer_sha256": generator_tokenizer_sha256,
        "student_tokenizer_sha256": student_tokenizer_sha256,
        "max_prompt_tokens": 4096,
        "counts": {
            "canonical_samples": len(rows),
            "prepared": {
                split: sum(row["split"] == split for row in rows)
                for split in ("train", "eval", "test")
            },
            "evidence_exclusions": 0,
            "sample_exclusions": 0,
            "precanonical_exclusions": 0,
        },
        "artifacts": {
            **static_records,
            "prepared": prepared_records,
            "audit": audit_records,
        },
    }
    handoff = {**payload, "payload_sha256": sha256_text(canonical_json(payload))}
    path = bundle / "prepare_handoff.json"
    path.write_text(json.dumps(handoff, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _accepted_result(**kwargs) -> dict:
    candidate = {
        "reasoning": "The available reading was 1.0 percent.",
        "final_analysis": "The available observation was reviewed.",
        "evidence_ids": ["E1"],
    }
    response = f"{candidate['reasoning']}\n</think>\n{candidate['final_analysis']}"
    return {
        "status": "accepted",
        "cache_key": sha256_text(kwargs["prompt"]),
        "prompt_sha256": sha256_text(kwargs["prompt"]),
        "teacher_output": {
            "analysis": candidate["reasoning"],
            "answer": candidate["final_analysis"],
            "evidence_ids": candidate["evidence_ids"],
        },
        "teacher_provenance": {
            "response_id": "mock-deepseek-response",
            "returned_model": "deepseek-v4-pro",
            "system_fingerprint": "mock-fingerprint",
        },
        "selected_from": "initial",
        "selected_candidate": candidate,
        "selected_response": response,
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_pilot_is_deterministic_topic_covering_and_time_spread() -> None:
    population = _prepared_rows(240)
    first = select_pilot_rows(population)
    reordered = select_pilot_rows(list(reversed(population)))

    assert len(first) == PILOT_SAMPLE_COUNT
    assert [row["sample_id"] for row in first] == [
        row["sample_id"] for row in reordered
    ]
    assert {row["atomic_topic"] for row in first} == set(TOPICS)
    for topic in TOPICS:
        population_dates = sorted(
            row["meeting_date"] for row in population if row["atomic_topic"] == topic
        )
        selected_dates = sorted(
            row["meeting_date"] for row in first if row["atomic_topic"] == topic
        )
        assert selected_dates[0] == population_dates[0]
        assert selected_dates[-1] == population_dates[-1]

    with pytest.raises(SelectionError, match="at least 200"):
        select_pilot_rows(population[: PILOT_SAMPLE_COUNT - 1])


def test_fixed_smoke_selection_is_stable_and_exactly_twenty() -> None:
    population = _prepared_rows(60)
    first = select_smoke_rows(population)
    second = select_smoke_rows(population[10:] + population[:10])
    assert len(first) == SMOKE_SAMPLE_COUNT
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    with pytest.raises(SelectionError, match="at least 20"):
        select_smoke_rows(population[:19])


def test_generation_binds_candidate_response_and_never_serializes_minutes(
    tmp_path: Path,
) -> None:
    secret_minutes = "SECRET SAME SAMPLE MINUTES TEXT MUST NEVER BE WRITTEN"
    rows = _runtime_rows(tmp_path, 3, minutes=secret_minutes)
    calls: list[dict] = []

    def generator(**kwargs):
        calls.append(kwargs)
        assert kwargs["same_sample_minutes"] == secret_minutes
        assert secret_minutes not in kwargs["prompt"]
        for forbidden in ("sample_id", "meeting_date", "canonical_key", "cutoff_ts"):
            assert forbidden not in kwargs["prompt"]
        return _accepted_result(**kwargs)

    resolver_requests: list[dict] = []

    def resolver(sample):
        resolver_requests.append(dict(sample))
        assert set(sample) == {
            "sample_id",
            "split",
            "meeting_date",
            "atomic_topic",
        }
        return secret_minutes

    output = tmp_path / "generated"
    handoff_path = run_generation_pipeline(
        prepared_rows=rows,
        output_dir=output,
        repo_root=tmp_path,
        generator=generator,
        allow_test_generator=True,
        minutes_resolver=resolver,
        environment={},
    )

    assert handoff_path == output / "generation_handoff.json"
    assert len(calls) == len(rows)
    assert len(resolver_requests) == len(rows)
    sft_rows = [
        row
        for split in ("train", "eval", "test")
        for row in _read_jsonl(output / "sft" / f"{split}.jsonl")
    ]
    manifests = [
        row
        for split in ("train", "eval", "test")
        for row in _read_jsonl(output / "manifests" / f"{split}.jsonl")
    ]
    assert len(sft_rows) == len(manifests) == len(rows)
    manifest_by_prompt = {row["prompt_sha256"]: row for row in manifests}
    for sft in sft_rows:
        manifest = manifest_by_prompt[sha256_text(sft["prompt"])]
        safe_fact = json.loads(sft["provided_data"])
        assert set(safe_fact) == {"schema_version", "atomic_topic", "evidence"}
        assert all("cutoff_ts" not in item for item in safe_fact["evidence"])
        assert sft["prompt"].count(sft["provided_data"]) == 1
        reasoning, final = sft["response"].split("\n</think>\n")
        assert manifest["reasoning_sha256"] == sha256_text(reasoning)
        assert manifest["final_analysis_sha256"] == sha256_text(final)
        assert manifest["response_sha256"] == sha256_text(sft["response"])
        assert "verifier" not in manifest
        assert manifest["selected_evidence_ids"] == ["E1"]
        assert (
            manifest["generation"]["model_input_projection"]["input_truncated"] is False
        )
    for path in output.rglob("*"):
        if path.is_file():
            content = path.read_text(encoding="utf-8")
            assert secret_minutes not in content
            assert "same_sample_minutes" not in content


def test_formal_prepare_handoff_is_replayed_and_bound(tmp_path: Path) -> None:
    generator_tokenizer, student_tokenizer = _model_repo(tmp_path)
    rows = _prepared_rows(
        3,
        generator_tokenizer_sha256=generator_tokenizer,
        student_tokenizer_sha256=student_tokenizer,
    )
    prepare_handoff = _prepare_bundle(
        tmp_path,
        rows,
        generator_tokenizer_sha256=generator_tokenizer,
        student_tokenizer_sha256=student_tokenizer,
    )
    output = tmp_path / "generated"
    generation_handoff = run_generation_pipeline(
        prepare_handoff_path=prepare_handoff,
        output_dir=output,
        repo_root=tmp_path,
        generator=_accepted_result,
        allow_test_generator=True,
        minutes_resolver=lambda _sample: DEFAULT_MINUTES,
        environment={},
    )
    handoff = json.loads(generation_handoff.read_text(encoding="utf-8"))
    assert (
        handoff["preparation_binding"]["handoff_sha256"]
        == hashlib.sha256(prepare_handoff.read_bytes()).hexdigest()
    )
    assert handoff["generation_provenance"]["generator_tokenizer"]["sha256"] == (
        generator_tokenizer
    )
    assert handoff["generation_provenance"]["student_tokenizer"]["sha256"] == (
        student_tokenizer
    )
    generation_code = handoff["generation_provenance"]["generation_code"]
    assert generation_code["schema_version"] == "chk1-generation-code-bundle-v1"
    assert len(generation_code["files"]) == 7
    assert all(
        row["path"] not in {
            "jobs/retrain_v2/chk1/critic.py",
            "jobs/retrain_v2/chk1/verifier.py",
        }
        for row in generation_code["files"]
    )
    assert len(generation_code["payload_sha256"]) == 64
    assert handoff["model_input_projection"]["accepted_row_count"] == 3
    preparation_binding_sha256 = handoff["preparation_binding"]["binding_sha256"]
    for split in ("train", "eval", "test"):
        for manifest in _read_jsonl(output / "manifests" / f"{split}.jsonl"):
            assert (
                manifest["generation"]["preparation_binding_sha256"]
                == preparation_binding_sha256
            )
            assert len(manifest["generation"]["prepared_row_sha256"]) == 64
            assert (
                manifest["generation"]["model_input_projection"][
                    "projection_contract_sha256"
                ]
                == handoff["model_input_projection"]["projection_contract_sha256"]
            )

    prepared_train = prepare_handoff.parent / "prepared" / "train.jsonl"
    prepared_train.write_text(
        prepared_train.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(PreparedDataError, match="byte-count mismatch"):
        run_generation_pipeline(
            prepare_handoff_path=prepare_handoff,
            output_dir=output,
            repo_root=tmp_path,
            generator=_accepted_result,
            allow_test_generator=True,
            minutes_resolver=lambda _sample: DEFAULT_MINUTES,
            environment={},
            resume=True,
        )


def test_prepared_rows_cannot_smuggle_same_sample_minutes(tmp_path: Path) -> None:
    rows = _prepared_rows(1)
    rows[0]["same_sample_minutes"] = "forbidden privileged text"
    calls = 0

    def generator(**_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("unsafe prepared input reached the generator")

    output = tmp_path / "not-created"
    with pytest.raises(PreparedDataError, match="may not contain Minutes"):
        run_generation_pipeline(
            prepared_rows=rows,
            output_dir=output,
            repo_root=tmp_path,
            generator=generator,
            allow_test_generator=True,
            minutes_resolver=lambda _sample: DEFAULT_MINUTES,
            environment={},
        )
    assert calls == 0
    assert not output.exists()


def test_formal_execution_requires_resolver_and_forbids_unattested_generator(
    tmp_path: Path,
) -> None:
    with pytest.raises(PreparedDataError, match="minutes_resolver is required"):
        run_generation_pipeline(
            prepared_rows=[],
            output_dir=tmp_path / "missing-resolver",
            repo_root=tmp_path,
        )

    with pytest.raises(PreparedDataError, match="custom generators are test-only"):
        run_generation_pipeline(
            prepared_rows=[],
            output_dir=tmp_path / "custom-generator",
            repo_root=tmp_path,
            generator=_accepted_result,
            minutes_resolver=lambda _sample: DEFAULT_MINUTES,
        )


def test_rejections_and_exceptions_become_sanitized_sample_exclusions(
    tmp_path: Path,
) -> None:
    rows = _runtime_rows(tmp_path, 3)
    call_count = 0

    def generator(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _accepted_result(**kwargs)
        if call_count == 2:
            raise TeacherResponseRejectedError(
                "private rejected candidate detail",
                payload={
                    "rejection_error_codes": ["critic_not_grounded"],
                    "raw_candidates": {"42": "private raw output"},
                },
            )
        return {
            "status": "rejected",
            "rejection_error_codes": ["candidate_unknown_evidence_id"],
        }

    output = tmp_path / "generated"
    run_generation_pipeline(
        prepared_rows=rows,
        output_dir=output,
        repo_root=tmp_path,
        generator=generator,
        allow_test_generator=True,
        minutes_resolver=lambda _sample: DEFAULT_MINUTES,
        environment={},
    )

    exclusions = _read_jsonl(output / "audit" / "exclusions.jsonl")
    assert len(exclusions) == 2
    assert {row["reason_code"] for row in exclusions} == {"candidate_rejected"}
    rejected_codes = {
        tuple(row["error_codes"])
        for row in exclusions
        if row["reason_code"] == "candidate_rejected"
    }
    assert rejected_codes == {
        ("critic_not_grounded",),
        ("candidate_unknown_evidence_id",),
    }
    serialized = canonical_json(exclusions)
    assert "private raw output" not in serialized
    handoff = json.loads(
        (output / "generation_handoff.json").read_text(encoding="utf-8")
    )
    assert handoff["accepted_count"] == 1
    assert handoff["excluded_count"] == 2


def test_systemic_runtime_error_aborts_without_handoff(tmp_path: Path) -> None:
    rows = _runtime_rows(tmp_path, 1)

    def failing_generator(**_kwargs):
        raise RuntimeError("backend safety failure")

    output = tmp_path / "generated"
    with pytest.raises(RuntimeError, match="backend safety failure"):
        run_generation_pipeline(
            prepared_rows=rows,
            output_dir=output,
            repo_root=tmp_path,
            generator=failing_generator,
            allow_test_generator=True,
            minutes_resolver=lambda _sample: DEFAULT_MINUTES,
            environment={},
        )
    assert not (output / "generation_handoff.json").exists()


def test_default_execution_reuses_one_teacher_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _runtime_rows(tmp_path, 2)
    teacher_backend = object()
    constructed_teachers: list[object] = []
    calls: list[dict] = []

    def teacher_backend_factory():
        constructed_teachers.append(teacher_backend)
        return teacher_backend

    def fake_local_generation(**kwargs):
        calls.append(kwargs)
        return _accepted_result(**kwargs)

    monkeypatch.setattr(
        generation_pipeline_module, "OpenAIDeepSeekBackend", teacher_backend_factory
    )
    monkeypatch.setattr(
        generation_pipeline_module,
        "run_deepseek_chk1_generation",
        fake_local_generation,
    )
    run_generation_pipeline(
        prepared_rows=rows,
        output_dir=tmp_path / "generated",
        repo_root=tmp_path,
        minutes_resolver=lambda _sample: DEFAULT_MINUTES,
        environment={},
        model_input_projector=(
            generation_pipeline_module._build_test_model_input_projector()
        ),
        sft_token_auditor=generation_pipeline_module._test_sft_token_auditor,
    )
    assert constructed_teachers == [teacher_backend]
    assert len(calls) == 2
    assert {id(call["teacher_backend"]) for call in calls} == {id(teacher_backend)}
    assert all("critic_backend" not in call for call in calls)
    assert all("allowed_pids" not in call for call in calls)
    assert len({str(call["cache_dir"]) for call in calls}) == 1


def test_default_deepseek_generation_uses_configured_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _runtime_rows(tmp_path, 4)
    barrier = threading.Barrier(4)
    thread_names: list[str] = []

    monkeypatch.setattr(
        generation_pipeline_module, "OpenAIDeepSeekBackend", lambda: object()
    )

    def concurrent_generation(**kwargs):
        thread_names.append(threading.current_thread().name)
        barrier.wait(timeout=2)
        return _accepted_result(**kwargs)

    monkeypatch.setattr(
        generation_pipeline_module,
        "run_deepseek_chk1_generation",
        concurrent_generation,
    )
    handoff = run_generation_pipeline(
        prepared_rows=rows,
        output_dir=tmp_path / "generated-concurrent",
        repo_root=tmp_path,
        minutes_resolver=lambda _sample: DEFAULT_MINUTES,
        environment={"DEEPSEEK_CONCURRENCY": "4"},
        model_input_projector=(
            generation_pipeline_module._build_test_model_input_projector()
        ),
        sft_token_auditor=generation_pipeline_module._test_sft_token_auditor,
    )
    assert handoff.is_file()
    assert len(thread_names) == 4
    assert all(name.startswith("chk1-deepseek") for name in thread_names)


def test_target_token_overflow_is_recorded_without_dropping_response(tmp_path: Path) -> None:
    rows = _runtime_rows(tmp_path, 1)

    def overflow_auditor(_prompt: str, _response: str) -> dict:
        return {
            "schema_version": "chk1-sft-token-budget-v1",
            "prompt_tokens": 100,
            "completion_tokens": 1025,
            "total_tokens": 1125,
            "max_prompt_tokens": 3072,
            "max_completion_tokens": 1024,
            "max_total_tokens": 4096,
            "overflow_policy": "error",
            "truncated": False,
            "passed": False,
        }

    output = tmp_path / "generated"
    run_generation_pipeline(
        prepared_rows=rows,
        output_dir=output,
        repo_root=tmp_path,
        generator=_accepted_result,
        allow_test_generator=True,
        minutes_resolver=lambda _sample: DEFAULT_MINUTES,
        environment={},
        sft_token_auditor=overflow_auditor,
    )

    assert len(_read_jsonl(output / "sft" / "train.jsonl")) == 1
    manifests = _read_jsonl(output / "manifests" / "train.jsonl")
    assert manifests[0]["generation"]["sft_token_budget"]["passed"] is False
    assert _read_jsonl(output / "audit" / "exclusions.jsonl") == []


def test_token_auditor_failure_does_not_drop_response(tmp_path: Path) -> None:
    rows = _runtime_rows(tmp_path, 1)

    def broken_auditor(_prompt: str, _response: str) -> dict:
        raise RuntimeError("tokenizer unavailable")

    output = tmp_path / "generated"
    run_generation_pipeline(
        prepared_rows=rows,
        output_dir=output,
        repo_root=tmp_path,
        generator=_accepted_result,
        allow_test_generator=True,
        minutes_resolver=lambda _sample: DEFAULT_MINUTES,
        environment={},
        sft_token_auditor=broken_auditor,
    )

    assert len(_read_jsonl(output / "sft" / "train.jsonl")) == 1
    budget = _read_jsonl(output / "manifests" / "train.jsonl")[0]["generation"][
        "sft_token_budget"
    ]
    assert budget == {
        "schema_version": "chk1-sft-token-budget-observation-v1",
        "status": "unavailable",
        "error_type": "RuntimeError",
    }
    assert _read_jsonl(output / "audit" / "exclusions.jsonl") == []


def test_minutes_sequence_hash_is_verified_before_model_call(tmp_path: Path) -> None:
    members = ("second excerpt", "first excerpt", "second excerpt")
    rows = _runtime_rows(tmp_path, 1, minutes=members)
    calls = 0

    def generator(**_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("mismatched Minutes reached model")

    output = tmp_path / "generated"
    run_generation_pipeline(
        prepared_rows=rows,
        output_dir=output,
        repo_root=tmp_path,
        generator=generator,
        allow_test_generator=True,
        minutes_resolver=lambda _sample: ("wrong excerpt",),
        environment={},
    )
    assert calls == 0
    exclusions = _read_jsonl(output / "audit" / "exclusions.jsonl")
    assert exclusions[0]["reason_code"] == "split_leakage"
    assert exclusions[0]["error_codes"] == ["minutes_reference_sha256_mismatch"]
    serialized = "".join(
        path.read_text(encoding="utf-8") for path in output.rglob("*") if path.is_file()
    )
    assert "wrong excerpt" not in serialized


def test_json_escaped_short_minutes_cannot_escape_into_response(
    tmp_path: Path,
) -> None:
    secret = 'alpha\n"beta"\\gamma'
    rows = _runtime_rows(tmp_path, 1, minutes=secret)

    def leaking_generator(**kwargs):
        candidate = {
            "reasoning": secret,
            "final_analysis": "The available observation was reviewed.",
            "evidence_ids": ["E1"],
        }
        return {
            **_accepted_result(**kwargs),
            "selected_candidate": candidate,
            "selected_response": (
                f"{candidate['reasoning']}\n</think>\n{candidate['final_analysis']}"
            ),
        }

    output = tmp_path / "generated"
    run_generation_pipeline(
        prepared_rows=rows,
        output_dir=output,
        repo_root=tmp_path,
        generator=leaking_generator,
        allow_test_generator=True,
        minutes_resolver=lambda _sample: secret,
        environment={},
    )
    exclusions = _read_jsonl(output / "audit" / "exclusions.jsonl")
    assert exclusions[0]["reason_code"] == "invalid_generator_result"
    assert _read_jsonl(output / "sft" / "train.jsonl") == []
    for path in output.rglob("*"):
        if path.is_file():
            assert secret not in path.read_text(encoding="utf-8")


def test_resume_uses_sample_cache_and_detects_corruption(tmp_path: Path) -> None:
    rows = _runtime_rows(tmp_path, 3)
    first_calls = 0

    def first_generator(**kwargs):
        nonlocal first_calls
        first_calls += 1
        return _accepted_result(**kwargs)

    output = tmp_path / "generated"
    handoff = run_generation_pipeline(
        prepared_rows=rows,
        output_dir=output,
        repo_root=tmp_path,
        generator=first_generator,
        allow_test_generator=True,
        minutes_resolver=lambda _sample: DEFAULT_MINUTES,
        environment={},
    )
    assert first_calls == len(rows)

    forbidden_calls = 0

    def forbidden_generator(**_kwargs):
        nonlocal forbidden_calls
        forbidden_calls += 1
        raise AssertionError("resume must not invoke the generator")

    assert (
        run_generation_pipeline(
            prepared_rows=rows,
            output_dir=output,
            repo_root=tmp_path,
            generator=forbidden_generator,
            allow_test_generator=True,
            minutes_resolver=lambda _sample: DEFAULT_MINUTES,
            environment={},
            resume=True,
        )
        == handoff
    )
    assert forbidden_calls == 0

    handoff.unlink()
    run_generation_pipeline(
        prepared_rows=rows,
        output_dir=output,
        repo_root=tmp_path,
        generator=forbidden_generator,
        allow_test_generator=True,
        minutes_resolver=lambda _sample: DEFAULT_MINUTES,
        environment={},
        resume=True,
    )
    assert forbidden_calls == 0

    sample_cache = next((output / "cache" / "samples").rglob("*.json"))
    original_cache = sample_cache.read_text(encoding="utf-8")
    sample_cache.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ResumeIntegrityError, match="hash mismatch"):
        run_generation_pipeline(
            prepared_rows=rows,
            output_dir=output,
            repo_root=tmp_path,
            generator=forbidden_generator,
            allow_test_generator=True,
            minutes_resolver=lambda _sample: DEFAULT_MINUTES,
            environment={},
            resume=True,
        )
    sample_cache.write_text(original_cache, encoding="utf-8")

    sft_path = output / "sft" / "train.jsonl"
    sft_path.write_text("corrupted\n", encoding="utf-8")
    with pytest.raises(ResumeIntegrityError, match="content mismatch|hash mismatch"):
        run_generation_pipeline(
            prepared_rows=rows,
            output_dir=output,
            repo_root=tmp_path,
            generator=forbidden_generator,
            allow_test_generator=True,
            minutes_resolver=lambda _sample: DEFAULT_MINUTES,
            environment={},
            resume=True,
        )
    assert sft_path.read_text(encoding="utf-8") == "corrupted\n"
    assert forbidden_calls == 0


def test_dry_run_reads_jsonl_but_calls_neither_resolver_nor_generator(
    tmp_path: Path,
) -> None:
    rows = _runtime_rows(tmp_path, 5)
    prepared_path = tmp_path / "prepared.jsonl"
    prepared_path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    calls = {"generator": 0, "resolver": 0}

    def generator(**_kwargs):
        calls["generator"] += 1
        raise AssertionError("dry-run called generator")

    def resolver(_sample):
        calls["resolver"] += 1
        raise AssertionError("dry-run called resolver")

    output = tmp_path / "not-created"
    plan = run_generation_pipeline(
        prepared_path=prepared_path,
        output_dir=output,
        repo_root=tmp_path,
        generator=generator,
        allow_test_generator=True,
        minutes_resolver=resolver,
        dry_run=True,
        environment={},
    )

    assert plan["status"] == "dry_run"
    assert plan["selection"]["selected_count"] == 5
    assert plan["planned_model_calls"] == 5
    assert calls == {"generator": 0, "resolver": 0}
    assert not output.exists()
