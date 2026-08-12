from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from jobs.eval import eval_chk3_stochastic_bootstrap_generation as subject
from open_r1.validator.loo_generation_spec import derive_row_seed, seal_manifest


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MANIFEST = (
    ROOT
    / "docs/summary/20260811T162400Z/chk3_native_stochastic_bootstrap_cp318/samples_n190.json"
)
CANONICAL_MANIFEST_SHA = (
    "314623a5e361e241da0f4602d993fc84919bb76606404b42b90b322e7e3a2ec9"
)
GREEDY_N12 = (
    ROOT
    / "docs/summary/20260811T003000Z/chk3_native_analysis_to_minutes_cp250_eval/samples_n12.json"
)


class ExactDecodeTokenizer:
    eos_token = "<eos>"

    def __init__(self, expected: str):
        self.expected = expected

    def decode(self, token_ids, **_kwargs):
        assert token_ids == [10, 11]
        return self.expected


def _fixture() -> tuple[dict, dict, str, str, str, dict]:
    source = "In April 2024, the rate was -22.8 percent."
    prompt = subject.native_eval.native_probe.USER_PROMPT_PREFIX + json.dumps(
        {"analysis": source}, separators=(",", ":")
    )
    reference = f"reference reasoning\n</think>\n{source}"
    generated = f"generation reasoning\n</think>\n{source}"
    sample = {
        "sample_id": "chk1-analysis-2023-07-26-deadbeef",
        "meeting_id": "2023-07-26",
        "length_bucket": "short",
        "prompt_token_count": 42,
        "prompt_sha256": subject._sha256_text(prompt),
        "analysis_reference_exact_identity": True,
        "normalized_identity": True,
        "punctuation_insensitive_identity": True,
    }
    source_hashes = {
        "sample_manifest_sha256": "a" * 64,
        "implementation_sources": {"runner": {"sha256": "b" * 64}},
    }
    bound = {sample["sample_id"]: {"prompt": prompt, "response": reference}}
    return sample, bound, source, reference, generated, source_hashes


def _row(*, model_id: str = "chk1", replicate_id: int = 0) -> dict:
    sample, _bound, source, reference, generated, source_hashes = _fixture()
    return subject.build_stochastic_result(
        text=generated,
        generated_token_ids=[10, 11, 999],
        eos_token_ids=[999],
        source_prompt=_bound[sample["sample_id"]]["prompt"],
        source_analysis=source,
        reference_response=reference,
        pad_token_id=999,
        model_id=model_id,
        model_label=subject.MODEL_LABELS[model_id],
        sample_manifest_sha256="a" * 64,
        sample=sample,
        replicate_id=replicate_id,
        replicate_seed=subject.REPLICATE_SEEDS[replicate_id],
        source_artifact_sha256s=source_hashes,
    )


def test_sampling_contract_and_paired_row_seed_are_frozen() -> None:
    contract = subject._sampling_contract()
    assert contract["do_sample"] is True
    assert contract["temperature"] == 0.6
    assert contract["top_p"] == 0.95
    assert contract["top_k"] == 50
    assert contract["repetition_penalty"] == 1.0
    assert contract["max_new_tokens"] == 3072
    assert contract["tail_tokens"] == 1024
    assert contract["replicate_seeds"] == list(subject.REPLICATE_SEEDS)

    rows = [_row(model_id=model_id) for model_id in subject.MODEL_ORDER]
    expected = derive_row_seed(subject.REPLICATE_SEEDS[0], rows[0]["sample_id"])
    assert {row["seed"] for row in rows} == {expected}
    assert {row["row_seed"] for row in rows} == {expected}


def test_canonical_n190_manifest_matches_training_tokenizer_and_greedy_anchor() -> None:
    manifest, observed = subject.load_full_test_sample_manifest(
        CANONICAL_MANIFEST, CANONICAL_MANIFEST_SHA
    )
    assert observed == CANONICAL_MANIFEST_SHA
    assert len(manifest["samples"]) == 190
    assert len({row["meeting_id"] for row in manifest["samples"]}) == 13
    assert sum(row["normalized_identity"] for row in manifest["samples"]) == 19

    greedy = json.loads(GREEDY_N12.read_text(encoding="utf-8"))
    full_counts = {
        row["sample_id"]: row["prompt_token_count"] for row in manifest["samples"]
    }
    assert {
        row["sample_id"]: row["prompt_token_count"] for row in greedy["samples"]
    } == {row["sample_id"]: full_counts[row["sample_id"]] for row in greedy["samples"]}


def test_row_deep_validation_recomputes_text_tokens_hashes_and_gates() -> None:
    sample, _bound, source, reference, generated, source_hashes = _fixture()
    row = _row()
    subject.validate_stochastic_result(
        row,
        model_id="chk1",
        model_label=subject.MODEL_LABELS["chk1"],
        sample_manifest_sha256="a" * 64,
        sample=sample,
        source_analysis=source,
        reference_response=reference,
        replicate_id=0,
        replicate_seed=subject.REPLICATE_SEEDS[0],
        source_artifact_sha256s=source_hashes,
        tokenizer=ExactDecodeTokenizer(generated),
    )
    assert row["input_truncated"] is False
    assert row["finish_reason"] == "eos"
    assert all(row["generation_metrics"].values())
    assert row["six_metrics"]["bertscore_f1"] is None
    assert row["six_metrics"]["mpnet_cosine"] is None

    tampered = copy.deepcopy(row)
    tampered["generated_token_ids"][0] += 1
    with pytest.raises(subject.StochasticBootstrapGenerationError, match="SHA256"):
        subject.validate_stochastic_result(
            tampered,
            model_id="chk1",
            model_label=subject.MODEL_LABELS["chk1"],
            sample_manifest_sha256="a" * 64,
            sample=sample,
            source_analysis=source,
            reference_response=reference,
            replicate_id=0,
            replicate_seed=subject.REPLICATE_SEEDS[0],
            source_artifact_sha256s=source_hashes,
        )

    tampered = copy.deepcopy(row)
    tampered["generated_text"] += " tampered"
    with pytest.raises(subject.StochasticBootstrapGenerationError, match="SHA256"):
        subject.validate_stochastic_result(
            tampered,
            model_id="chk1",
            model_label=subject.MODEL_LABELS["chk1"],
            sample_manifest_sha256="a" * 64,
            sample=sample,
            source_analysis=source,
            reference_response=reference,
            replicate_id=0,
            replicate_seed=subject.REPLICATE_SEEDS[0],
            source_artifact_sha256s=source_hashes,
        )

    tampered = copy.deepcopy(row)
    tampered["temperature"] = 0.7
    with pytest.raises(subject.StochasticBootstrapGenerationError, match="temperature"):
        subject.validate_stochastic_result(
            tampered,
            model_id="chk1",
            model_label=subject.MODEL_LABELS["chk1"],
            sample_manifest_sha256="a" * 64,
            sample=sample,
            source_analysis=source,
            reference_response=reference,
            replicate_id=0,
            replicate_seed=subject.REPLICATE_SEEDS[0],
            source_artifact_sha256s=source_hashes,
        )


def test_progress_requires_exact_canonical_prefix() -> None:
    sample, bound, _source, _reference, _generated, source_hashes = _fixture()
    rows = [_row(replicate_id=0), _row(replicate_id=1)]
    samples = [sample]
    subject.validate_progress_prefix(
        rows,
        model_id="chk1",
        model_label=subject.MODEL_LABELS["chk1"],
        sample_manifest_sha256="a" * 64,
        samples=samples,
        bound_rows=bound,
        source_artifact_sha256s=source_hashes,
        smoke=False,
    )
    with pytest.raises(
        subject.StochasticBootstrapGenerationError, match="out-of-order"
    ):
        subject.validate_progress_prefix(
            list(reversed(rows)),
            model_id="chk1",
            model_label=subject.MODEL_LABELS["chk1"],
            sample_manifest_sha256="a" * 64,
            samples=samples,
            bound_rows=bound,
            source_artifact_sha256s=source_hashes,
            smoke=False,
        )
    with pytest.raises(subject.StochasticBootstrapGenerationError, match="duplicate"):
        subject.validate_progress_prefix(
            [rows[0], copy.deepcopy(rows[0])],
            model_id="chk1",
            model_label=subject.MODEL_LABELS["chk1"],
            sample_manifest_sha256="a" * 64,
            samples=samples,
            bound_rows=bound,
            source_artifact_sha256s=source_hashes,
            smoke=False,
        )

    missing_middle = [rows[0], _row(replicate_id=2)]
    with pytest.raises(
        subject.StochasticBootstrapGenerationError, match="out-of-order"
    ):
        subject.validate_progress_prefix(
            missing_middle,
            model_id="chk1",
            model_label=subject.MODEL_LABELS["chk1"],
            sample_manifest_sha256="a" * 64,
            samples=samples,
            bound_rows=bound,
            source_artifact_sha256s=source_hashes,
            smoke=False,
        )


def test_resume_state_binds_path_bytes_sha_and_allows_one_row_state_lag(
    tmp_path: Path,
) -> None:
    rows = [_row(replicate_id=0), _row(replicate_id=1)]
    progress = tmp_path / "progress.jsonl"
    progress.write_text(subject._canonical_json(rows[0]) + "\n", encoding="utf-8")
    state = seal_manifest(
        subject._state_payload(
            status="generating",
            model_id="chk1",
            results=rows[:1],
            expected_cases=5,
            resume_count=2,
            progress_path=progress,
        )
    )
    with progress.open("a", encoding="utf-8") as handle:
        handle.write(subject._canonical_json(rows[1]) + "\n")
    assert (
        subject._validate_resume_state(
            state=state,
            model_id="chk1",
            rows=rows,
            progress_path=progress,
            expected_cases=5,
        )
        == 3
    )

    tampered = copy.deepcopy(state)
    tampered.pop("integrity")
    tampered["partial_results"]["path"] = str(tmp_path / "other.jsonl")
    with pytest.raises(subject.StochasticBootstrapGenerationError, match="path"):
        subject._validate_resume_state(
            state=seal_manifest(tampered),
            model_id="chk1",
            rows=rows,
            progress_path=progress,
            expected_cases=5,
        )

    tampered = copy.deepcopy(state)
    tampered.pop("integrity")
    tampered["partial_results"]["bytes"] += 1
    with pytest.raises(subject.StochasticBootstrapGenerationError, match="one-row-ahead"):
        subject._validate_resume_state(
            state=seal_manifest(tampered),
            model_id="chk1",
            rows=rows,
            progress_path=progress,
            expected_cases=5,
        )


def test_resume_rejects_resealed_state_contract_drift(tmp_path: Path) -> None:
    row = _row()
    progress = tmp_path / "progress.jsonl"
    progress.write_text(subject._canonical_json(row) + "\n", encoding="utf-8")
    state = subject._state_payload(
        status="generating",
        model_id="chk1",
        results=[row],
        expected_cases=5,
        resume_count=0,
        progress_path=progress,
    )
    state["expected_cases"] = 4
    with pytest.raises(subject.StochasticBootstrapGenerationError, match="contract"):
        subject._validate_resume_state(
            state=seal_manifest(state),
            model_id="chk1",
            rows=[row],
            progress_path=progress,
            expected_cases=5,
        )


def test_resume_allows_only_canonical_rename_before_frozen_state_window(
    tmp_path: Path,
) -> None:
    row = _row()
    partial_dir = tmp_path / ".partial"
    partial_dir.mkdir()
    partial = partial_dir / "generations.progress.v1.jsonl"
    final = tmp_path / "generations.jsonl"
    partial.write_text(subject._canonical_json(row) + "\n", encoding="utf-8")
    state = seal_manifest(
        subject._state_payload(
            status="generating",
            model_id="chk1",
            results=[row],
            expected_cases=1,
            resume_count=0,
            progress_path=partial,
        )
    )
    partial.replace(final)
    assert (
        subject._validate_resume_state(
            state=state,
            model_id="chk1",
            rows=[row],
            progress_path=final,
            expected_cases=1,
            canonical_partial_path=partial,
        )
        == 1
    )
    with pytest.raises(subject.StochasticBootstrapGenerationError, match="path"):
        subject._validate_resume_state(
            state=state,
            model_id="chk1",
            rows=[row],
            progress_path=final,
            expected_cases=1,
            canonical_partial_path=tmp_path / "wrong.progress.jsonl",
        )


def test_rng_reset_calls_transformers_torch_and_cuda() -> None:
    calls: list[tuple[str, int]] = []

    class Transformers:
        @staticmethod
        def set_seed(seed):
            calls.append(("transformers", seed))

    class Cuda:
        @staticmethod
        def manual_seed_all(seed):
            calls.append(("cuda", seed))

    class Torch:
        cuda = Cuda()

        @staticmethod
        def manual_seed(seed):
            calls.append(("torch", seed))

    subject.reset_row_rng(Transformers(), Torch(), 123)
    assert calls == [("transformers", 123), ("torch", 123), ("cuda", 123)]


def test_cli_exposes_only_smoke_single_model_and_formal_suite() -> None:
    parser = subject._build_parser()
    suite = parser.parse_args(
        [
            "run-suite",
            "--sample-manifest",
            "samples.json",
            "--sample-manifest-sha256",
            "a" * 64,
            "--output-dir",
            "out",
        ]
    )
    assert suite.command == "run-suite"
    assert subject.MODEL_ORDER == ("chk1", "chk3", "chk0")


def test_suite_child_object_accepts_canonical_json_key_sorting() -> None:
    # JSON artifacts are serialized with sort_keys=True, so object order is
    # chk0/chk1/chk3 even though the authoritative execution array is
    # chk1/chk3/chk0.
    sorted_bindings = {
        model_id: {"path": f"/{model_id}/manifest.json"}
        for model_id in sorted(subject.MODEL_ORDER)
    }
    assert list(sorted_bindings) == ["chk0", "chk1", "chk3"]
    subject._validate_suite_child_binding_inventory(sorted_bindings)

    missing = dict(sorted_bindings)
    missing.pop("chk3")
    with pytest.raises(subject.StochasticBootstrapGenerationError, match="inventory"):
        subject._validate_suite_child_binding_inventory(missing)

    extra = {**sorted_bindings, "chk2": {"path": "/chk2/manifest.json"}}
    with pytest.raises(subject.StochasticBootstrapGenerationError, match="inventory"):
        subject._validate_suite_child_binding_inventory(extra)


def test_gpu_process_gate_ignores_self_and_reports_external(monkeypatch) -> None:
    stdout = f"{os.getpid()}, self\n424242, external_worker\n"

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(subject.subprocess, "run", fake_run)
    assert subject._external_gpu0_compute_processes() == [
        {"pid": 424242, "process_name": "external_worker"}
    ]


def test_suite_retries_only_dedicated_external_gpu_failure(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[bool] = []
    output_dir = tmp_path / "chk1"

    def fake_run_model(**kwargs):
        calls.append(kwargs["resume"])
        if len(calls) == 1:
            output_dir.mkdir()
            raise subject.ExternalGpuProcessDetectedError("external")
        return {"status": "complete"}

    monkeypatch.setattr(subject, "run_model", fake_run_model)
    result = subject._run_model_with_external_retry(
        model_id="chk1",
        sample_manifest_path=tmp_path / "samples.json",
        sample_manifest_sha256="a" * 64,
        output_dir=output_dir,
        smoke=True,
        gpu_wait_timeout_seconds=60,
        gpu_poll_seconds=1,
        gpu_lock_path=tmp_path / "gpu.lock",
    )
    assert result == {"status": "complete"}
    assert calls == [False, True]
